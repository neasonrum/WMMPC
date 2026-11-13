"""Train the V5 future adapter on 2Hz chains of the 4Hz recording; V3 stays frozen."""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv4.batched_rolling import BatchedRollingPlanStore
from method.full_diffusionv4.bootstrap import ROOT
from method.full_diffusionv4.data import RouteStreamBatchSampler
from method.full_diffusionv4.joint import detach_tree
from method.full_diffusionv4.joint_data import DistributedRouteStreams
from method.full_diffusionv4.joint_train import (barrier, close_loader, move, rank_world, restore_rng,
                                                 rng_state, seed)
from method.full_diffusionv4.world import sha256

from .data import V5Dataset
from .model import FutureAdaptedPlanner, align_future, build_world, generate_future, load_v3_head

STOP = False
AGES = [0.0, 0.25, 0.5]
PER_FRAME = ('raw_bev', 'memory', 'ego', 'history_transforms', 'history_mask', 'speed', 'command',
             'target_point', 'path', 'spatial_mask', 'temporal_delta', 'temporal_mask', 'future_positions',
             'behavior_targets', 'behavior_mask', 'current_pose')


def stop(*_):
    global STOP
    STOP = True


def frame(batch, j):
    return {k: batch[k][:, j] for k in PER_FRAME}


class CachedJointPlanner(nn.Module):
    """One world rollout per item at its refresh frame; cache reused at 0.25s and 0.5s."""
    def __init__(self, planner, world, diffusion, config):
        super().__init__()
        self.planner, self.world, self.config = planner, world, config
        self.diffusion = diffusion

    def train(self, mode=True):
        super().train(mode)
        self.planner.base.eval()
        self.world.eval()
        return self

    def future(self, x, prior, prior_valid, ages):
        return generate_future(self.world, self.planner, x['raw_bev'], x['history_transforms'],
                               x['history_mask'], prior, prior_valid, ages, x['speed'], x['ego'])

    def condition(self, x, future):
        return self.planner.condition(x['raw_bev'][:, -1], x['memory'], x['ego'], x['target_point'],
                                      x['raw_bev'][:, :2], x['history_transforms'][:, :2],
                                      x['history_mask'][:, :2], x['speed'], x['command'], future)

    def aligned(self, future, batch, j, device):
        if future is None:
            return None
        transform = torch.linalg.solve(batch['current_pose'][:, j].double(), batch['current_pose'][:, 0].double())
        return align_future(future, transform, torch.full((len(transform),), AGES[j], device=device))

    def sample(self, condition, batch_size, device, generator, steps):
        result = self.diffusion.sample(lambda x, t: self.planner(x, t, condition)['predictions'][-1],
                                       batch_size, device, generator, steps)
        return self.planner.codec.decode(result)

    def forward(self, batch, prior, prior_valid, ages, generator):
        settings, weights = self.config['training'], self.config['loss']
        device, b = prior.device, len(prior)
        logs = {}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            future = self.future(frame(batch, 0), prior, prior_valid, ages)
            ready = (future['valid'].any(-1) if future is not None
                     else torch.zeros(b, dtype=torch.bool, device=device))
            total, first = 0., None
            for j in range(3):
                x = frame(batch, j)
                condition = self.condition(x, self.aligned(future, batch, j, device))
                first = condition if j == 0 else first
                clean = self.planner.codec.encode(x['path'], x['temporal_delta'])
                steps = torch.randint(self.diffusion.train_steps, (b,), device=device)
                steps = torch.where(torch.rand(b, device=device) < settings['max_noise_fraction'],
                                    self.diffusion.train_steps - 1, steps)
                noisy, t = self.diffusion.corrupt(clean, steps)
                output = self.planner(noisy, t, condition)
                # Cold rows equal frozen V3 and carry no gradient; average over WM-ready rows only.
                parts = self.planner.base.loss_components(
                    output, clean, x['spatial_mask'] & ready[:, None], x['temporal_mask'] & ready[:, None],
                    x['behavior_targets'], x['behavior_mask'] & ready[:, None], weights)
                loss = weights['spatial_weight'] * parts['spatial_l1'] + weights['temporal_weight'] * parts['temporal_delta_l1']
                total = total + weights['age_weights'][j] * loss
                logs[f'spatial_l1_age{j}'] = parts['spatial_l1'].detach()
                logs[f'temporal_l1_age{j}'] = parts['temporal_delta_l1'].detach()
            anchor = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            _, delta = self.sample(detach_tree(first), b, device, generator, settings['rolling_inference_steps'])
        return {'loss': total + anchor, **logs, 'sampled_delta': delta.detach().float(),
                'future_ready': ready.float().sum()}


def build(config, manifest, device):
    seed(config['seed'])
    if config['schedule']['training_cache_ages_seconds'] != AGES:
        raise ValueError('Items are (t, t+0.25, t+0.5); cache ages must be [0, 0.25, 0.5]')
    path = ROOT / config['v3']['checkpoint']
    if sha256(path) != config['v3']['sha256']:
        raise ValueError('V3 checkpoint checksum differs')
    base, v3config = load_v3_head(path)
    if v3config['labels'] != config['labels'] or manifest['label_config'] != config['labels']:
        raise ValueError('V5 labels/index must match the frozen V3 labels')
    world = build_world(config, ROOT, training=True)
    if (world.provenance['teacher_sha256'] != manifest['teacher_sha256']
            or world.provenance['teacher_config_sha256'] != manifest['teacher_config_sha256']):
        raise ValueError('Index and world use different frozen perception identities')
    planner = FutureAdaptedPlanner(base, config['future_sampler'])
    diffusion = SpatialTemporalVPFullDiffusion(base.codec, **{
        k: v for k, v in v3config['diffusion'].items() if k != 'inference_steps'})
    return CachedJointPlanner(planner, world, diffusion, config).to(device)


def optimizer_groups(policy, settings):
    rates, modules, groups = settings['learning_rates'], dict(policy.named_modules()), {}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('world.'):
            group = 'world'
        elif name.endswith('.gate'):
            group = 'gate'
        elif name.endswith(('.distance_logit', '.time_logit')):
            group = 'distance'
        else:
            group = 'new'
        owner = modules[name.rpartition('.')[0]]
        plain = p.ndim > 1 and not isinstance(owner, (nn.LayerNorm, nn.Embedding))
        decay = (settings['weight_decay'] if group == 'new' else 1e-4 if group == 'world' else 0.) if plain else 0.
        key = (group, decay)
        if key not in groups:
            groups[key] = {'params': [], 'lr': rates[group], 'weight_decay': decay, 'name': f'{group}/wd{decay}'}
        groups[key]['params'].append(p)
    return list(groups.values())


def save(output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling, generator,
         device, best=False):
    rank, size = rank_world()
    trained = [p.detach().double() for p in policy.parameters() if p.requires_grad]
    checksum = torch.stack((sum(p.sum() for p in trained), sum(p.square().sum() for p in trained))).cpu().tolist()
    local = {'sampler': sampler.state_dict(), 'rolling': rolling.state_dict(), 'rng': rng_state(generator),
             'parameter_checksum': checksum}
    states = [None] * size if rank == 0 else None
    if size > 1:
        dist.gather_object(local, states, dst=0)
    else:
        states = [local]
    if rank == 0:
        if any(s['parameter_checksum'] != states[0]['parameter_checksum'] for s in states):
            raise RuntimeError('DDP parameters differ between ranks at checkpoint')
        state = {'format': 'full_diffusionv5_v1', 'config': config, 'adapter': policy.planner.adapter.state_dict(),
                 'world_tuned_state': policy.world.export_tuned() if config['world'].get('tuning') else None,
                 'world': policy.world.provenance, 'v3': config['v3'], 'dataset': metadata,
                 'progress': deepcopy(progress), 'gates': policy.planner.gates(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rank_states': states}
        torch.save(state, output / 'latest.partial')
        if (output / 'latest.pt').exists():
            (output / 'latest.pt').replace(output / 'latest.previous.pt')
        (output / 'latest.partial').replace(output / 'latest.pt')
        if best:
            shutil.copyfile(output / 'latest.pt', output / 'best.partial')
            (output / 'best.partial').replace(output / 'best.pt')
    barrier(device)


@torch.no_grad()
def validate(policy, config, device):
    """Chronological 2Hz chains; V5 and frozen V3 share each initial noise sample."""
    rank, size = rank_world()
    dataset = V5Dataset(ROOT / config['index_directory'], 'val')
    routes = sorted({int(r) for r in dataset.samples[:, 0]})
    keep = np.isin(dataset.samples[:, 0], routes[rank::size])
    dataset.samples = dataset.samples[keep]
    chains = dataset.chains()
    batch_size = min(config['training']['batch_per_gpu'], max(1, len(np.unique(chains[:, 0]))))
    loader = DataLoader(dataset, batch_sampler=RouteStreamBatchSampler(chains, batch_size, config['seed']),
                        num_workers=0)
    rolling = BatchedRollingPlanStore(config['schedule']['max_prior_age_seconds'])
    generator = torch.Generator(device=device).manual_seed(config['seed'] + 500009 + rank)
    steps = config['diffusion']['inference_steps']
    totals = torch.zeros(len(dataset.routes), 3, 5, dtype=torch.float64, device=device)
    policy.eval()
    for batch in loader:
        batch = move(batch, device)
        prior, valid, ages = rolling.prior(batch['chain_id'], batch['current_pose'][:, 0], batch['timestamp'])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            future = policy.future(frame(batch, 0), prior, valid, ages)
            ready = future['valid'].any(-1) if future is not None else torch.zeros(len(prior), dtype=torch.bool, device=device)
            for j in range(3):
                x = frame(batch, j)
                state = generator.get_state()
                path, delta = policy.sample(policy.condition(x, policy.aligned(future, batch, j, device)),
                                            len(prior), device, generator, steps)
                generator.set_state(state)
                base_path, base_delta = policy.sample(policy.condition(x, None), len(prior), device, generator, steps)
                stored = delta if j == 0 else stored
                measured = torch.stack((
                    (path - x['path']).float().norm(dim=-1).mean(1),
                    (delta.cumsum(1) - x['future_positions']).float().norm(dim=-1).mean(1),
                    (base_path - x['path']).float().norm(dim=-1).mean(1),
                    (base_delta.cumsum(1) - x['future_positions']).float().norm(dim=-1).mean(1),
                    torch.ones(len(prior), device=device)), -1).double() * ready[:, None]
                totals[:, j].index_add_(0, batch['route_id'], measured)
        rolling.update(batch['chain_id'], batch['current_pose'][:, 0], batch['timestamp'], stored.float())
    if size > 1:
        dist.all_reduce(totals)
    metrics = {}
    for j in range(3):
        present = totals[:, j, -1] > 0
        macro = (totals[present, j, :4] / totals[present, j, -1:]).mean(0).tolist() if bool(present.any()) else [math.nan] * 4
        for name, value in zip(('v5_spatial_ade_m', 'v5_future_ade_m', 'v3_spatial_ade_m', 'v3_future_ade_m'), macro):
            metrics[f'{name}_age{j}'] = value
        metrics[f'ready_frames_age{j}'] = int(totals[:, j, -1].sum())
    metrics['score_v5_future_ade_m'] = float(np.mean([metrics[f'v5_future_ade_m_age{j}'] for j in range(3)]))
    metrics['gain_vs_v3_future_ade_m'] = float(np.mean([metrics[f'v3_future_ade_m_age{j}'] - metrics[f'v5_future_ade_m_age{j}']
                                                       for j in range(3)]))
    policy.train()
    return {**metrics, 'aggregation': 'route_macro over world-ready frames', 'inference_steps': steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID')):
        raise RuntimeError('Run only inside an existing allocation')
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    rank, size = rank_world()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads(args.config.read_text())
    settings = config['training']
    if size != settings['world_size']:
        raise ValueError('DDP world size differs from the config')
    directory = ROOT / config['index_directory']
    manifest = json.loads((directory / 'manifest.json').read_text())
    policy = build(config, manifest, device)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise FileExistsError('Use a separate output directory or --resume')
        args.output.mkdir(parents=True, exist_ok=True)
    barrier(device)
    dataset = V5Dataset(directory)
    sampler = DistributedRouteStreams(dataset.chains(), settings['batch_per_gpu'], rank, size, config['seed'])
    optimizer = torch.optim.AdamW(optimizer_groups(policy, settings), fused=True)
    steps_per_epoch = math.ceil(len(dataset) / (size * settings['batch_per_gpu']))
    total_updates = settings['epochs'] * steps_per_epoch
    warmup = min(settings['warmup_updates'], max(1, total_updates - 1))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, total_updates - warmup))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metadata = {'index_identity': manifest['identity'], 'train_items': len(dataset), 'world_size': size,
                'global_items_per_update': size * settings['batch_per_gpu'], 'steps_per_epoch': steps_per_epoch}
    rolling = BatchedRollingPlanStore(config['schedule']['max_prior_age_seconds'])
    generator = torch.Generator(device=device).manual_seed(config['seed'] + rank + 100003)
    seed(config['seed'] + rank + 200003)
    progress = {'epoch': 0, 'step_in_epoch': 0, 'updates': 0, 'best_score': None, 'validation_pending': False}
    if args.resume:
        state = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=False)
        if state['config'] != config or state['dataset'] != metadata:
            raise ValueError('Resume config, index or world size differs')
        policy.planner.adapter.load_state_dict(state['adapter'], strict=True)
        if config['world'].get('tuning'):
            from method.full_diffusionv4.world_tuning import load_tuned_state
            load_tuned_state(policy.world, state['world_tuned_state'], config['world']['tuning'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        local_state = state['rank_states'][rank]
        sampler.load_state_dict(local_state['sampler'])
        rolling.load_state_dict(local_state['rolling'])
        restore_rng(local_state['rng'], generator)
        progress = state['progress']
        del state
    if rank == 0:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (args.output / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
    ddp = DDP(policy, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False) if size > 1 else policy
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings['workers_per_rank'],
                        pin_memory=True, persistent_workers=bool(settings['workers_per_rank']),
                        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if settings['workers_per_rank'] else {}))
    iterator = iter(loader)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    policy.train()

    def finish_epoch():
        metrics = validate(policy, config, device)
        score = metrics['score_v5_future_ade_m']
        best = progress['best_score'] is None or score < progress['best_score']
        progress.update(validation_pending=False, validation=metrics,
                        best_score=score if best else progress['best_score'])
        if rank == 0:
            with (args.output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps({**progress, 'gates': policy.planner.gates()}) + '\n')
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, device, best=best)

    if progress['validation_pending']:
        finish_epoch()
    for epoch in range(progress['epoch'], settings['epochs']):
        first = progress['step_in_epoch'] if epoch == progress['epoch'] else 0
        for step in range(first, steps_per_epoch):
            begin = time.perf_counter()
            batch = move(next(iterator), device)
            data_wait = time.perf_counter() - begin
            prior, valid, ages = rolling.prior(batch['chain_id'], batch['current_pose'][:, 0], batch['timestamp'])
            optimizer.zero_grad(set_to_none=True)
            output = ddp(batch, prior, valid, ages, generator)
            if not bool(torch.isfinite(output['loss'])):
                raise FloatingPointError('Nonfinite V5 adapter loss')
            output['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            rolling.update(batch['chain_id'], batch['current_pose'][:, 0], batch['timestamp'], output['sampled_delta'])
            sampler.commit()
            progress.update(epoch=epoch, step_in_epoch=step + 1, updates=progress['updates'] + 1)
            if progress['updates'] < 5 or progress['updates'] % settings['log_every'] == 0:
                names = ['loss'] + [f'{k}_l1_age{j}' for j in range(3) for k in ('spatial', 'temporal')] + ['future_ready']
                values = torch.stack([output[k].detach().float() for k in names])
                if size > 1:
                    dist.all_reduce(values)
                if rank == 0:
                    record = {**progress, **{k: float(v) / (1 if k == 'future_ready' else size) for k, v in zip(names, values)},
                              'global_items': size * len(prior), 'seconds_per_update': time.perf_counter() - begin,
                              'data_wait_seconds': data_wait, 'gradient_norm': float(norm),
                              'gates': policy.planner.gates(),
                              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                              'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups}}
                    with (args.output / 'train.jsonl').open('a') as handle:
                        handle.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
            stop_now = torch.tensor(int(STOP or (args.output / 'STOP').exists()
                                        or (args.max_updates is not None and progress['updates'] >= args.max_updates)),
                                    device=device)
            if size > 1:
                dist.all_reduce(stop_now, op=dist.ReduceOp.MAX)
            if progress['updates'] == 1 or progress['updates'] % settings['checkpoint_every'] == 0 or bool(stop_now):
                save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
                     generator, device)
            if bool(stop_now):
                close_loader(iterator)
                if size > 1:
                    dist.destroy_process_group()
                return
        progress.update(epoch=epoch + 1, step_in_epoch=0, validation_pending=True)
        save(args.output, policy, optimizer, scheduler, config, metadata, progress, sampler, rolling,
             generator, device)
        finish_epoch()
    close_loader(iterator)
    if size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
