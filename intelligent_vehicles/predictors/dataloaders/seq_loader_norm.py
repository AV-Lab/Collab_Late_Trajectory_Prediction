#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeqDataset
==========

Each sample in the pickle file must be a dict:

    {
        "obs_cat":  <str>,        # category string
        "obs":     [[x,y,yaw], ...]   # length L
        "target":  [[x,y,yaw], ...]   # length H
    }

Returned tensors
----------------
    cat_id : int64 scalar                       (0-based class index)
    obs    : [L, 5]  → (x, y, vx, vy, vyaw)
    tgt    : [H, 5]  → (x, y, vx, vy, vyaw)

If `normalize=True` the 5 features are z-scored with mean/std computed
from the *training* set and stored in `<data>.norm.pkl`.
"""
from __future__ import annotations
import os, pickle, math
import numpy as np
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    # ---------- shared cat map ----------
    _CAT2IDX = {
        "car": 0, "motorcycle": 1, "pedestrian": 2,
        "van": 3, "truck": 4, "cyclist": 5
    }

    # ---------- constructor ----------
    def __init__(self, data_file: str,
                 normalize: bool = False,
                 stats_path: str | None = None):
        if not os.path.isfile(data_file):
            raise FileNotFoundError(data_file)
            
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        self.L = len(self.data[0]["obs"])
        self.H = len(self.data[0]["target"])

        self.normalize = normalize

        if self.normalize:
            self.compute_stats()

    # ---------- yaw helper ----------
    @staticmethod
    def _yaw_wrap(delta):
        """Wrap angle to (-π, π]."""
        return (delta + math.pi) % (2 * math.pi) - math.pi

    # ---------- build 5-feature trajectory ----------
    def _build_traj(self, arr: np.ndarray) -> torch.Tensor:
        """
        arr : [T,3] (x,y,yaw)
        ret : [T,5] (x,y,vx,vy,vyaw)
        """
        pos_xy = arr[:, :2]                          # [T,2]
        yaw    = arr[:, 2]                           # [T]

        vel_xy = np.diff(pos_xy, axis=0, prepend=pos_xy[[0]])  # [T,2]
        vyaw   = self._yaw_wrap(
                    np.diff(yaw, prepend=yaw[0]))[:, None]     # [T,1]

        feats = np.hstack([pos_xy, vel_xy, vyaw])    # [T,5]
        return torch.tensor(feats, dtype=torch.float32)

    # ---------- stats helpers ----------
    def compute_stats(self):
        acc_sum = np.zeros(5)
        acc_sq  = np.zeros(5)
        count   = 0
        for s in self.data:
            for split in ("obs", "target"):
                feats = self._build_traj(np.asarray(s[split])).numpy()
                acc_sum += feats.sum(0)
                acc_sq  += (feats ** 2).sum(0)
                count   += feats.shape[0]

        self.mean = acc_sum / count       
        var = acc_sq / count - self.mean**2
        self.std = np.sqrt(np.maximum(var, 1e-8))
        
        self.mean = torch.tensor(self.mean, dtype=torch.float32)
        self.std = torch.tensor(self.std, dtype=torch.float32)

    # ---------- Dataset API ----------
    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        cat_id = torch.tensor(self._CAT2IDX[s["obs_cat"]], dtype=torch.long)
    
        obs = self._build_traj(np.asarray(s["obs"]))      # [L,5]  raw
        tgt = self._build_traj(np.asarray(s["target"]))   # [H,5]  raw
        return cat_id, obs, tgt
