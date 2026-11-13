"""Prepare the 4Hz V5 index.

Every recorded 0.25s frame with two contiguous history frames and a complete 3s
future is an anchor. A training item is three consecutive anchors (t, t+0.25,
t+0.5): the world refreshes at t and its cache is read at ages 0, 0.25 and 0.5s.
Items of one route split into two 2Hz chains by frame parity, so both phases of
the 4Hz recording are used without ever creating a 0.25s-old previous plan.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from method.full_diffusionv4.bootstrap import ROOT
from method.full_diffusionv4.world import sha256

HISTORY, FUTURE = 2, 12
PROCESSOR = None


def require_allocation():
    if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID')):
        raise RuntimeError('Prepare inside an existing compute allocation')


def worker_init(backbone):
    global PROCESSOR
    torch.set_num_threads(1)
    from method.full_diffusion.garage import load_config, make_preprocessor
    PROCESSOR = make_preprocessor(load_config(backbone), training=False)


def anchor_indices(frames, skip_first):
    good = np.zeros(len(frames), dtype=bool)
    width = HISTORY + FUTURE + 1
    if len(frames) >= width:
        windows = np.lib.stride_tricks.sliding_window_view(frames, width)
        good[HISTORY:len(frames) - FUTURE] = (np.diff(windows, axis=1) == 1).all(1)
    return np.flatnonzero(good & (frames >= skip_first))


def prepare_route(task):
    from method.full_diffusion.data import geometric_path, read_measurement
    from method.full_diffusion2.data import frame_pose
    from method.full_diffusionv3.data import future_motion_targets
    import transfuser_utils
    rid, name, source, directory, data_root, teacher_root, skip_first, labels = task
    with np.load(Path(source) / 'metadata' / f'{rid:05}.npz') as f:
        frames, poses, speeds = f['frames'].astype(np.int64), f['poses'], f['speed']
    cache = Path(teacher_root) / name
    existing = ({int(p.name[:-4]) for p in os.scandir(cache) if p.name.endswith('.npy')}
                if cache.is_dir() else set())
    anchors = anchor_indices(frames, skip_first)
    records = defaultdict(list)
    for anchor in anchors:
        frame = int(frames[anchor])
        current = read_measurement(Path(data_root) / name / 'measurements' / f'{frame:04}.json.gz')
        future = {2 * k: {'pos_global': poses[anchor + 2 * k, :2], 'speed': speeds[anchor + 2 * k]}
                  for k in range(1, 7)}
        targets = future_motion_targets(current, future, labels)
        pose = frame_pose(current)
        observed = []
        for a in (anchor - 2, anchor - 1, anchor):
            x, y, yaw = poses[a]
            c, s = np.cos(yaw), np.sin(yaw)
            observed.append(np.linalg.solve(pose, np.array([[c, -s, x], [s, c, y], [0, 0, 1]])))
        values = {**targets, 'frame': frame, 'history_frames': frames[anchor - 2:anchor + 1],
                  'path': geometric_path(PROCESSOR, current), 'spatial_mask': np.ones(10, bool),
                  'speed': np.float32(current['speed']),
                  'command': np.asarray(transfuser_utils.command_to_one_hot(current['command']), dtype=np.float32),
                  'target_point': np.asarray(PROCESSOR.augment_target_point(
                      np.asarray(current['target_point']), y_augmentation=0., yaw_augmentation=0.), dtype=np.float32),
                  'current_pose': pose, 'history_transforms': np.asarray(observed, dtype=np.float32),
                  'history_mask': np.ones(3, bool)}
        for key, value in values.items():
            records[key].append(value)
    slot = {int(frames[a]): s for s, a in enumerate(anchors)}
    items, missing = [], 0
    for s, a in enumerate(anchors):
        f = int(frames[a])
        if f + 1 in slot and f + 2 in slot:
            if all(x in existing for x in range(f - 2, f + 3)):
                items.append((rid, s, f))
            else:
                missing += 1
    output = Path(directory) / 'labels' / f'{rid:05}.npz'
    if records:
        arrays = {k: np.asarray(v) for k, v in records.items()}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f'Nonfinite labels in {name}')
        with output.with_suffix('.partial').open('wb') as handle:
            np.savez(handle, **arrays)
        output.with_suffix('.partial').replace(output)
    digest = sha256(output) if records else None
    return rid, len(anchors), np.asarray(items, dtype=np.int64).reshape(-1, 3), missing, digest


def labels(args):
    require_allocation()
    config = json.loads(args.config.read_text())
    v3 = torch.load(ROOT / config['v3']['checkpoint'], map_location='cpu', weights_only=False)
    if v3['config']['labels'] != config['labels']:
        raise ValueError('V5 labels must be exactly the frozen V3 labels')
    del v3
    source = (ROOT / config['source_index']).resolve()
    output = (ROOT / config['index_directory']).resolve()
    original = json.loads((source / 'manifest.json').read_text())
    if output.exists():
        raise FileExistsError(f'Use a new V5 index directory: {output}')
    (output / 'labels').mkdir(parents=True)
    routes = json.loads((source / 'routes.json').read_text())
    parts = {}
    for split in ('train', 'val'):
        for rid in np.unique(np.load(source / f'{split}.npy')[:, 0]):
            if parts.setdefault(int(rid), split) != split:
                raise ValueError('Train and validation route overlap')
    teacher_root = (ROOT / 'data/irwm_bev/teacher_cache' / (
        original['teacher_sha256'][:16] + '_' + original['teacher_config_sha256'][:12] + '_bf16_v1'))
    tasks = [(rid, routes[rid], str(source), str(output), config['data_root'], str(teacher_root),
              int(original['skip_first']), config['labels']) for rid in sorted(parts)]
    items = {'train': [], 'val': []}
    anchors = {'train': 0, 'val': 0}
    missing = {'train': 0, 'val': 0}
    identities = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(str(ROOT / config['backbone_directory']),)) as pool:
        for i, (rid, count, route_items, absent, digest) in enumerate(pool.map(prepare_route, tasks), 1):
            split = parts[rid]
            anchors[split] += count
            missing[split] += absent
            items[split].append(route_items)
            if digest:
                identities[f'{rid:05}.npz'] = digest
            if i % 100 == 0:
                print(json.dumps({'event': 'labels', 'routes': i, 'total': len(tasks),
                                  'seconds': time.perf_counter() - started}), flush=True)
    for split, values in items.items():
        np.save(output / f'{split}.npy', np.concatenate(values).astype(np.int32))
    (output / 'routes.json').write_text(json.dumps(routes))
    counts = {split: len(np.load(output / f'{split}.npy')) for split in items}
    manifest = {'method': 'full_diffusionv5_index_v1', 'data_root': config['data_root'],
                'backbone_directory': str(ROOT / config['backbone_directory']), 'label_config': config['labels'],
                'teacher_cache': str(teacher_root), 'teacher_sha256': original['teacher_sha256'],
                'teacher_config_sha256': original['teacher_config_sha256'],
                'source_index': str(source), 'source_sha256': {n: sha256(source / n) for n in
                                                            ('manifest.json', 'routes.json', 'train.npy', 'val.npy')},
                'cadence': {'recorded_hz': 4, 'anchor_stride_frames': 1, 'chain_stride_frames': 2,
                            'item_frames': [0, 1, 2], 'cache_ages_seconds': [0.0, 0.25, 0.5]},
                'anchors': anchors, 'items': counts, 'items_missing_teacher_features': missing,
                'routes': {split: sum(s == split for s in parts.values()) for split in ('train', 'val')},
                'labels_sha256': identities,
                'prepared_index_sha256': {n: sha256(output / n) for n in ('routes.json', 'train.npy', 'val.npy')},
                'future_input_policy': 'future poses/speeds create labels only; observed frames only are inputs'}
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'labels_complete', 'anchors': anchors, 'items': counts,
                      'items_missing_teacher_features': missing, 'identity': manifest['identity'],
                      'seconds': time.perf_counter() - started}), flush=True)


class AnchorSensors(Dataset):
    def __init__(self, directory, manifest, tasks):
        from method.full_diffusion.data import GarageGeoDataset
        self.loader = GarageGeoDataset(directory, data_root=manifest['data_root'],
                                       backbone_directory=manifest['backbone_directory'], training=False)
        self.root = Path(manifest['data_root'])
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        rid, slot, frame = self.tasks[index]
        current = self.loader.load_frame(self.root / self.loader.routes[rid], frame, random.Random(901))
        return {k: current[k] for k in ('rgb', 'lidar', 'speed', 'command', 'target_point')} | {
            'route_id': rid, 'slot': slot}


def contexts(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from method.full_diffusion2.perception import FrozenPlannerPerception
    require_allocation()
    config = json.loads(args.config.read_text())
    directory = ROOT / config['index_directory']
    manifest = json.loads((directory / 'manifest.json').read_text())
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    target = directory / 'context'
    target.mkdir(exist_ok=True)
    routes = sorted({int(r) for split in ('train', 'val') for r in np.load(directory / f'{split}.npy')[:, 0]})
    selected = [r for r in routes if r % world == rank and not (target / f'{r:05}.pt').exists()]
    perception = FrozenPlannerPerception(manifest['backbone_directory']).to(device).eval()
    frames = 0
    for start in range(0, len(selected), 64):
        tasks = []
        for rid in selected[start:start + 64]:
            with np.load(directory / 'labels' / f'{rid:05}.npz') as f:
                tasks.extend((rid, s, int(frame)) for s, frame in enumerate(f['frame']))
        loader = DataLoader(AnchorSensors(directory, manifest, tasks), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True,
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if args.workers else {}))
        collected = defaultdict(list)
        with torch.inference_mode():
            for batch in loader:
                sensor = {k: batch[k].to(device, non_blocking=True)
                          for k in ('rgb', 'lidar', 'speed', 'command', 'target_point')}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, memory, ego = perception(sensor)
                for i, (rid, slot) in enumerate(zip(batch['route_id'].tolist(), batch['slot'].tolist())):
                    collected[rid].append((slot, memory[i].bfloat16().cpu(), ego[i].bfloat16().cpu()))
                frames += len(batch['rgb'])
        for rid, records in collected.items():
            ordered = sorted(records, key=lambda x: x[0])
            if [x[0] for x in ordered] != list(range(len(ordered))):
                raise ValueError('Incomplete or duplicated context slots')
            state = {'memory': torch.stack([x[1] for x in ordered]), 'ego': torch.stack([x[2] for x in ordered]),
                     'dataset_sha256': manifest['identity']}
            torch.save(state, target / f'{rid:05}.partial')
            (target / f'{rid:05}.partial').replace(target / f'{rid:05}.pt')
        print(json.dumps({'event': 'contexts', 'rank': rank, 'frames': frames,
                          'routes_done': min(start + 64, len(selected)), 'routes': len(selected)}), flush=True)
    if world > 1:
        dist.barrier(device_ids=[local])
    if rank == 0:
        if any(not (target / f'{r:05}.pt').is_file() for r in routes):
            raise ValueError('Context cache is incomplete')
        (target / 'complete.json').write_text(json.dumps({
            'dataset_sha256': manifest['identity'], 'routes': len(routes), 'precision': 'bf16',
            'source': 'frozen TF++ current-frame memory/ego of every 4Hz anchor'}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['labels', 'contexts'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    (labels if args.stage == 'labels' else contexts)(args)


if __name__ == '__main__':
    main()
