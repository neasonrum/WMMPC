"""Frozen V3 head plus zero-gated readouts of a cached action-conditioned future.

The world model runs once per 0.5s refresh. Its sampled future tokens stay in the
refresh ego frame; every later planning tick re-expresses them in the current
ego frame and current relative time before the denoiser reads them.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from method.full_diffusion.model import blocks
from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead as V3Head
from method.full_diffusionv4.future import DEFAULT_FUTURE_CONFIG, FutureNeighborhoodSampler
from method.full_diffusionv4.geometry import motion_features

MAP_TIMES = (0.5, 1.0, 1.5, 2.0, 2.5)


def load_v3_head(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if (config.get('method') != 'full_diffusionv3'
            or config.get('architecture') != 'independent_dual_query64'):
        raise ValueError('V5 corrects the independently trained V3 dual-query checkpoint only')
    artifact = checkpoint['dataset']['v3_normalization']
    if artifact['label_config'] != config['labels']:
        raise ValueError('V3 checkpoint normalization and label settings differ')
    head = V3Head(**config['model'], normalization=artifact['normalization'])
    head.load_state_dict(checkpoint['model'], strict=True)
    return head, config


def align_future(future, transform, age):
    """Generation-frame future -> current frame; `transform` maps generation to current."""
    with torch.autocast(device_type=transform.device.type, enabled=False):
        rotation = transform[:, :2, :2].float()
        translation = transform[:, :2, 2].float()
        age = age.float().to(rotation.device)
        times = torch.tensor(MAP_TIMES, device=rotation.device)
        map_time = times[None] - age[:, None]
        token_time = times[future['map_ids']][None] - age[:, None]
        map_valid = (future['window_fraction'] > 0) & (map_time > 1e-3)
        positions = torch.einsum('bij,bnj->bni', rotation, future['positions'].float()) + translation[:, None]
        axes = torch.einsum('bij,bnaj->bnai', rotation, future['axes'].float())
        centres = torch.einsum('bij,bkj->bki', rotation, future['centres'].float()) + translation[:, None]
        window_axes = torch.einsum('bij,bkaj->bkai', rotation, future['window_axes'].float())
    return {'tokens': future['tokens'], 'log_prior': future['log_prior'],
            'valid': future['valid'] & (token_time > 1e-3),
            'positions': positions, 'axes': axes, 'radii': future['radii'], 'centres': centres,
            'window_axes': window_axes, 'window_radii': future['window_radii'],
            'window_fraction': future['window_fraction'] * map_valid,
            'map_valid': map_valid, 'map_time': map_time, 'token_time': token_time, 'age': age}


class CachedFutureReadout(nn.Module):
    """Geometry/time-aware attention to cached future tokens; exact no-op at gate 0."""
    def __init__(self, d_model=256, heads=8, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.heads = heads
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.position = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.timing = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.reliability = nn.Sequential(nn.Linear(d_model + 8, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.distance_logit = nn.Parameter(torch.tensor(math.log(.01 / .99)))
        if temporal:
            self.time_logit = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.zeros(()))

    def prepare(self, future):
        if future is None:
            return None
        b, n, d = future['tokens'].shape
        timing = torch.stack((future['token_time'] / 3., future['age'][:, None].expand(-1, n) / .5), -1)
        timed = future['tokens'] + self.timing(timing.to(future['tokens'].dtype))
        return {**future, 'key': self.key(timed).reshape(b, n, self.heads, d // self.heads),
                'value': self.value(timed).reshape(b, n, self.heads, d // self.heads)}

    def forward(self, query, positions, future):
        if future is None:
            return query
        b, count, _ = positions.shape
        hidden = self.norm(query).expand(-1, count, -1)
        queries = self.query(hidden + self.position(positions / 32)).reshape(b, count, self.heads, -1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            offset = positions.float()[:, :, None] - future['positions'].float()[:, None]
            local = torch.einsum('bpni,bnai->bpna', offset, future['axes'].float())
            distance = (local / future['radii'].float()[:, None].clamp_min(.1)).square().sum(-1).clamp(max=9)
            score = torch.einsum('bphd,bnhd->bhpn', queries.float(), future['key'].float()) / math.sqrt(queries.shape[-1])
            score = score + future['log_prior'].float()[:, None, None] - self.distance_logit.sigmoid() * distance[:, None]
            map_valid = future['map_valid']
            if self.temporal:
                query_time = torch.arange(1, count + 1, device=query.device).float() * .5
                gap = (future['token_time'][:, None] - query_time[None, :, None]) / .5
                score = score - F.softplus(self.time_logit) * gap.square().clamp(max=9)[:, None]
                gaps = (future['map_time'][:, None] - query_time[None, :, None]).abs()
                match = gaps.masked_fill(~map_valid[:, None], float('inf')).argmin(-1)
                time_gap = gaps.gather(-1, match[..., None]).squeeze(-1)
            else:
                spread = (positions.float()[:, :, None] - future['centres'][:, None]).square().sum(-1)
                match = spread.masked_fill(~map_valid[:, None], float('inf')).argmin(-1)
                time_gap = torch.zeros(b, count, device=query.device)
            valid = future['valid']
            any_valid = valid.any(-1)
            safe_valid = valid.clone()
            safe_valid[:, 0] |= ~any_valid
            weights = score.masked_fill(~safe_valid[:, None, None], float('-inf')).softmax(-1)
            context = torch.einsum('bhpn,bnhd->bphd', weights, future['value'].float()).reshape(b, count, -1)
            has_map = map_valid.gather(1, match)
            centre = future['centres'].gather(1, match[..., None].expand(-1, -1, 2))
            axes = future['window_axes'].gather(1, match[..., None, None].expand(-1, -1, 2, 2))
            radii = future['window_radii'].float().gather(1, match[..., None].expand(-1, -1, 2))
            window_local = torch.einsum('bpi,bpai->bpa', positions.float() - centre, axes) / radii.clamp_min(.1)
            window_distance = window_local.abs().amax(-1)
            fraction = future['window_fraction'].float().gather(1, match)
            supported = (window_distance <= 1) & (fraction > 0) & has_map & any_valid[:, None]
            time_gap = torch.where(has_map, time_gap, 0.)
            extra = torch.cat(((positions.float() - centre) / 32, window_distance[..., None].clamp(max=4),
                               fraction[..., None], supported.float()[..., None], has_map.float()[..., None],
                               (future['age'][:, None, None] / .5).expand(-1, count, -1),
                               (time_gap[..., None] / .5).clamp(max=4)), -1)
        reliability = self.reliability(torch.cat((hidden, extra.to(hidden)), -1)).sigmoid()
        update = self.output(context.to(query.dtype)) * reliability * supported[..., None]
        update = update.sum(1, keepdim=True) / supported.sum(1, keepdim=True)[..., None].clamp_min(1)
        return query + self.gate.tanh() * update


def decode_layer(layer, query, metric_points, bev_values, memory, ego, time_embedding, readout, future):
    """V3 `SpatialTemporalDecoderLayer.forward`, with the readout after ego attention."""
    attention = layer.cross_bev_attention
    references = metric_points.cumsum(dim=1) if layer.temporal else metric_points
    grid = references[:, None] / 32.0
    sampled = F.grid_sample(bev_values, grid, align_corners=False, padding_mode="zeros")
    weights = attention.attention_weights(query).softmax(-1)
    sampled = (sampled * weights[:, None]).sum(-1).transpose(1, 2)
    query = query + attention.dropout(attention.output_proj(sampled))
    query = layer.norm1(query + layer.dropout(
        layer.cross_agent_attention(query, memory, memory, need_weights=False)[0]
    ))
    query = layer.norm2(query + layer.dropout(
        layer.cross_ego_attention(query, ego, ego, need_weights=False)[0]
    ))
    if future is not None:
        query = readout(query, references, future)
    query = layer.norm3(layer.ffn(query))
    scale, shift = layer.time_modulation(time_embedding).chunk(2, dim=-1)
    query = query * (1 + scale) + shift
    return metric_points + layer.task_decoder(query).reshape(metric_points.shape)


class FutureAdaptedPlanner(nn.Module):
    """Frozen V3 (`base`) plus trainable adapter; without a future it is exactly V3."""
    def __init__(self, base, future_sampler=None):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        d_model = base.time_mlp[-1].out_features
        self.adapter = nn.ModuleDict({
            'sampler': FutureNeighborhoodSampler(64, d_model, future_sampler or DEFAULT_FUTURE_CONFIG),
            'spatial': nn.ModuleList([CachedFutureReadout(d_model) for _ in base.layers]),
            'temporal': nn.ModuleList([CachedFutureReadout(d_model, temporal=True) for _ in base.temporal_layers]),
        })

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @property
    def codec(self):
        return self.base.codec

    def tokenize(self, maps, prior_delta, prior_valid, ego, motion):
        return self.adapter['sampler'](maps, prior_delta, prior_valid, ego.mean(1), motion)

    def condition(self, bev, memory, ego, target_point, history_bev, history_transforms,
                  history_mask, speed, command, future=None):
        with torch.no_grad():
            condition = self.base.prepare_condition(bev, memory, ego, target_point, history_bev,
                                                    history_transforms, history_mask, speed, command)
        condition['future'] = [r.prepare(future) for r in self.adapter['spatial']]
        condition['temporal_future'] = [r.prepare(future) for r in self.adapter['temporal']]
        return condition

    def forward(self, noisy, time, condition):
        base = self.base
        base.codec.check_shape(noisy)
        points = base.codec.path_points
        features = blocks.gen_sineembed_for_position(noisy, hidden_dim=base.slot_width)
        features = features + base.type_embedding(base.type_ids)[None]
        features = features + base.slot_embedding(base.slot_ids)[None]
        path_query = base.path_query_encoder(features[:, :points].flatten(start_dim=1))[:, None]
        temporal_query = base.temporal_query_encoder(features[:, points:].flatten(start_dim=1))[:, None]
        path_query = path_query + condition["behavior_offset"]
        temporal_query = temporal_query + condition["temporal_behavior_offset"]
        time_embedding = base.time_mlp(time * 1000)[:, None]
        temporal_time_embedding = base.temporal_time_mlp(time * 1000)[:, None]
        metric_points = base.codec.to_metric(noisy)
        predictions = []
        for i, (layer, temporal_layer) in enumerate(zip(base.layers, base.temporal_layers)):
            spatial = decode_layer(layer, path_query, metric_points[:, :points], condition["bev"][i],
                                   condition["memory"], condition["ego"], time_embedding,
                                   self.adapter['spatial'][i], condition['future'][i])
            temporal = decode_layer(temporal_layer, temporal_query, metric_points[:, points:],
                                    condition["temporal_bev"][i], condition["memory"], condition["ego"],
                                    temporal_time_embedding, self.adapter['temporal'][i],
                                    condition['temporal_future'][i])
            normalized_prediction = base.codec.encode_metric(torch.cat((spatial, temporal), dim=1))
            predictions.append(normalized_prediction)
            metric_points = base.codec.to_metric(normalized_prediction.detach())
        return {"predictions": torch.stack(predictions), "behavior_logits": condition["behavior_logits"]}

    def gates(self):
        return {f'{branch}{i}': float(r.gate.detach().tanh())
                for branch in ('spatial', 'temporal') for i, r in enumerate(self.adapter[branch])}


def build_world(config, root, training=False):
    """Frozen BCDE world by default; optional V4-style partial tuning when configured."""
    from method.full_diffusionv4.world import TimedWorldModel
    world_config = config['world']
    paths = (root / world_config['checkpoint'], root / config['backbone_directory'] / 'model_0030_1.pth',
             world_config['checkpoint_sha256'])
    if world_config.get('tuning'):
        from method.full_diffusionv4.world_tuning import PartialTimedWorld
        world = PartialTimedWorld(*paths, tuning=world_config['tuning'], activation_checkpoint=training)
        if not training:
            world.requires_grad_(False)
        return world
    return TimedWorldModel(*paths)


def generate_future(world, planner, raw_bev, transforms, history_mask, prior, prior_valid, ages, speed, ego):
    """One world rollout driven by the aligned previous plan; tokens stay in this frame."""
    maps, valid, aligned = world(raw_bev, transforms, history_mask, prior, prior_valid, speed)
    if maps is None:
        return None
    motion = motion_features(aligned.detach(), history_mask, speed, prior, ages)
    future = planner.tokenize(maps, prior, valid, ego, motion)
    future['valid'] = future['valid'] & valid[:, :5].any(-1)[:, None]
    return future
