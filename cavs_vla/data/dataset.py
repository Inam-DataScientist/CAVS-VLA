"""Torch dataset over a merged split: returns raw scenes; featurization happens on the GPU."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..config import Config
from .build import RawSceneStore


class RawSceneDataset(Dataset):
    def __init__(self, cfg: Config, split: str, indices: Optional[List[int]] = None) -> None:
        self.cfg = cfg
        self.split_dir = os.path.join(cfg.data.out_dir, split)
        self.store = RawSceneStore(cfg, self.split_dir)
        self.indices = list(range(len(self.store))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = self.store.get(self.indices[i])
        out = {}
        for k, v in s.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if t.dtype == torch.float64 and k != "t0":
                t = t.float()
            out[k] = t
        out["index"] = torch.tensor(self.indices[i], dtype=torch.long)
        return out


class WeightedIndexSampler(Sampler):
    """Samples dataset positions with replacement according to per-sample weights (hard-case mining)."""

    def __init__(self, weights: np.ndarray, num_samples: int, seed: int = 0) -> None:
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.n = int(num_samples)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(self.w, self.n, replacement=True, generator=g).tolist())

    def __len__(self) -> int:
        return self.n
