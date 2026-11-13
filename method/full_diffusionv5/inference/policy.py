"""V5 closed loop: 20Hz frozen-V3 denoising corrected by a 2Hz cached world rollout.

Every tick plans from fresh observations. Every 0.5s of simulation time the
plan stored at the previous refresh (age 0.5s, as in training) drives one world
rollout; its future tokens are cached and re-expressed in the current ego frame
and current relative time on every later tick.
"""
from contextlib import nullcontext
import os
from pathlib import Path

import torch
from torch import nn

from method.full_diffusion2.inference.policy import TemporalBEVHistory
from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3.inference.policy import FullDiffusionV3Network
from method.full_diffusionv4.rolling import RollingPlanStore
from method.full_diffusionv4.world import sha256
from method.full_diffusionv5.model import (FutureAdaptedPlanner, align_future, build_world,
                                           generate_future, load_v3_head)

ROOT = Path(__file__).resolve().parents[3]


class FullDiffusionV5Network(FullDiffusionV3Network):
    def __init__(self, garage_net, checkpoint_path, device):
        nn.Module.__init__(self)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state.get('format') != 'full_diffusionv5_v1':
            raise ValueError('V5 inference requires a V5 adapter checkpoint')
        config = self.run_config = state['config']
        v3_path = Path(os.environ.get('WMMPC_V3_CHECKPOINT', ROOT / config['v3']['checkpoint']))
        if sha256(v3_path) != config['v3']['sha256']:
            raise ValueError('The adapter was trained on a different frozen V3 checkpoint')
        base, v3config = load_v3_head(v3_path)
        self.planner = FutureAdaptedPlanner(base, config['future_sampler'])
        self.planner.adapter.load_state_dict(state['adapter'], strict=True)
        self.world = build_world(config, ROOT, training=False)
        if state['world']['sha256'] != self.world.provenance['sha256']:
            raise ValueError('Adapter was trained with a different world checkpoint')
        if config['world'].get('tuning'):
            from method.full_diffusionv4.world_tuning import load_tuned_state
            load_tuned_state(self.world, state['world_tuned_state'], config['world']['tuning'])
        self.world.requires_grad_(False)
        self.diffusion = SpatialTemporalVPFullDiffusion(base.codec, **{
            k: v for k, v in v3config['diffusion'].items() if k != 'inference_steps'})
        steps = int(os.environ.get('WMMPC_DIFFUSION_INFERENCE_STEPS', config['diffusion']['inference_steps']))
        if not 1 <= steps <= self.diffusion.train_steps:
            raise ValueError('Invalid V5 inference step override')
        self.inference_steps = steps
        self.precision = v3config['precision']
        schedule = config['schedule']
        self.tick_seconds = schedule['control_interval_seconds']
        self.refresh_seconds = schedule['refresh_interval_seconds']
        self.max_cache_age = schedule['max_cache_age_seconds']
        self.world_enabled = os.environ.get('WMMPC_V5_DISABLE_WORLD', '0') != '1'
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        self.history = TemporalBEVHistory(v3config['model']['history_frames'], v3config['observation_interval_seconds'])
        # Accept one tick of float jitter around the trained 0.5s prior age.
        self.rolling = RollingPlanStore(schedule['max_prior_age_seconds'] + self.tick_seconds / 2)
        self.generator = torch.Generator(device=device).manual_seed(config['seed'])
        self.cache = None
        self.last_refresh = None
        self.last_target_speed = None
        self.last_temporal_delta = None
        self.last_behavior_probabilities = None
        self.last_world_active = False
        self.last_cache_age = None
        self._scene = 'route'
        self._timestamp = None
        self._pose = None
        if (garage_net.config.speed_kp, garage_net.config.speed_ki,
                garage_net.config.speed_kd, garage_net.config.speed_n) != (1.75, 1.0, 2.0, 20):
            raise ValueError('V5 keeps the V3 AutoMoT temporal PID settings')
        self.to(device)
        self.eval()

    def reset(self):
        self.history.entries.clear()
        self.rolling.reset()
        self.cache = None
        self.last_refresh = None

    def set_observation(self, pose, timestamp, scene_id=None):
        # Snap to the simulator tick so 0.5s ages are exact multiples of 0.05s.
        timestamp = round(float(timestamp) / self.tick_seconds) * self.tick_seconds
        scene = self._scene if scene_id is None else str(scene_id)
        if scene != self._scene or (self._timestamp is not None and timestamp <= self._timestamp):
            self.reset()
        self._scene = scene
        self.history.set_observation(pose, timestamp)
        self._pose = torch.as_tensor(self.history.pose, dtype=torch.float64)
        self._timestamp = timestamp

    @torch.no_grad()
    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
        if self._pose is None or self._timestamp is None or len(rgb) != 1:
            raise RuntimeError('V5 online inference needs a batch1 observation pose and timestamp')
        batch = {'rgb': rgb, 'lidar': lidar_bev, 'target_point': target_point,
                 'target_point_next': target_point_next, 'speed': ego_vel, 'command': command}
        context = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if rgb.is_cuda and self.precision == 'bf16' else nullcontext())
        pose, now = self._pose.to(rgb.device)[None], self._timestamp
        refresh = self.last_refresh is None or now - self.last_refresh >= self.refresh_seconds - self.tick_seconds / 2
        with context:
            bev, memory, ego = self.perception(batch)
            history, transforms, history_valid = self.history.get_inputs(bev)
            if refresh:
                self.last_refresh, self.cache = now, None
                if self.world_enabled:
                    prior, prior_valid, ages = self.rolling.prior([self._scene], pose, [now])
                    identity = torch.eye(3, device=rgb.device).expand(1, 1, 3, 3)
                    future = generate_future(
                        self.world, self.planner, torch.cat((history, bev[:, None]), 1),
                        torch.cat((transforms, identity), 1),
                        torch.cat((history_valid, history_valid.new_ones(1, 1)), 1),
                        prior, prior_valid, ages, ego_vel, ego)
                    if future is not None and bool(future['valid'].any()):
                        self.cache = (future, pose.clone(), now)
            aligned, self.last_cache_age = None, None
            if self.cache is not None:
                future, source_pose, source_time = self.cache
                age = now - source_time
                if age <= self.max_cache_age + self.tick_seconds / 2:
                    transform = torch.linalg.solve(pose.double(), source_pose.double())
                    aligned = align_future(future, transform, torch.tensor([age], device=rgb.device))
                    self.last_cache_age = age
            condition = self.planner.condition(bev, memory, ego, target_point, history, transforms,
                                               history_valid, ego_vel, command, aligned)
            sample = self.diffusion.sample(
                lambda noisy, time: self.planner(noisy, time, condition)['predictions'][-1],
                1, rgb.device, self.generator, self.inference_steps)
            path, delta = self.planner.codec.decode(sample)
            speed = self.planner.codec.target_speed(delta)
            if not bool(torch.isfinite(path).all() & torch.isfinite(delta).all() & torch.isfinite(speed).all()):
                raise FloatingPointError('V5 produced nonfinite control targets')
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        if refresh:
            # Only refresh-tick plans drive the next rollout, keeping the trained 0.5s prior age.
            self.rolling.update([self._scene], pose, [now], delta.float())
        self.last_target_speed = float(speed[0])
        self.last_temporal_delta = delta.float().detach()
        self.last_behavior_probabilities = condition['behavior_logits'].float().softmax(-1).detach()
        self.last_world_active = aligned is not None
        bins = torch.tensor(self.config.target_speeds, device=speed.device, dtype=speed.dtype)
        compatibility_speed = speed.clamp(0, max(self.config.target_speeds))
        upper = torch.searchsorted(bins, compatibility_speed).clamp(1, len(bins) - 1)
        weight = (compatibility_speed - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device, dtype=speed.dtype)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        return None, probabilities.clamp_min(1e-12).log(), path.float(), None, None, None, boxes, None, None, None
