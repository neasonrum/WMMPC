"""Three-frame V5 items (t, t+0.25, t+0.5) and their 2Hz chain identities."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_KEYS = ('path', 'spatial_mask', 'temporal_delta', 'temporal_mask', 'future_positions',
              'behavior_targets', 'behavior_mask', 'speed', 'command', 'target_point',
              'current_pose', 'history_transforms', 'history_mask')


class V5Dataset(Dataset):
    def __init__(self, directory, split='train'):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.routes = json.loads((self.directory / 'routes.json').read_text())
        self.samples = np.load(self.directory / f'{split}.npy')
        context = json.loads((self.directory / 'context/complete.json').read_text())
        if context['dataset_sha256'] != self.manifest['identity']:
            raise ValueError('Context cache belongs to a different V5 index')
        self.teacher_root = Path(self.manifest['teacher_cache'])

    def __len__(self):
        return len(self.samples)

    def chains(self):
        """(chain key, frame) per item; one chain = one route at one frame parity (2Hz)."""
        keys = self.samples[:, 0].astype(np.int64) * 2 + self.samples[:, 2] % 2
        return np.stack((keys, self.samples[:, 2].astype(np.int64)), 1)

    @lru_cache(maxsize=256)
    def metadata(self, route):
        with np.load(self.directory / 'labels' / f'{route:05}.npz') as f:
            return {k: f[k].copy() for k in f.files}

    @lru_cache(maxsize=256)
    def context(self, route):
        state = torch.load(self.directory / 'context' / f'{route:05}.pt', map_location='cpu', weights_only=True)
        if state['dataset_sha256'] != self.manifest['identity']:
            raise ValueError('Route context belongs to a different V5 index')
        return state

    @lru_cache(maxsize=512)
    def feature(self, route, frame):
        path = self.teacher_root / self.routes[route] / f'{frame:04}.npy'
        bits = np.load(path, allow_pickle=False)
        if bits.dtype != np.uint16 or bits.shape != (64, 64, 64):
            raise ValueError(f'Invalid observed teacher feature: {path}')
        return torch.from_numpy(bits).view(torch.bfloat16)

    def __getitem__(self, index):
        route, slot, frame = map(int, self.samples[index])
        m = self.metadata(route)
        slots = [slot, slot + 1, slot + 2]
        if [int(m['frame'][s]) for s in slots] != [frame, frame + 1, frame + 2]:
            raise ValueError(f'Item {index} is not three consecutive 4Hz frames')
        result = {k: torch.from_numpy(np.stack([m[k][s] for s in slots])) for k in FRAME_KEYS}
        features = {f: self.feature(route, f) for f in range(frame - 2, frame + 3)}
        result['raw_bev'] = torch.stack([torch.stack([features[f] for f in range(frame + j - 2, frame + j + 1)])
                                         for j in range(3)])
        context = self.context(route)
        result['memory'], result['ego'] = context['memory'][slots], context['ego'][slots]
        result.update(chain_id=f'{self.routes[route]}#{frame % 2}', timestamp=frame * .25,
                      route_id=route, sample_index=index)
        return result
