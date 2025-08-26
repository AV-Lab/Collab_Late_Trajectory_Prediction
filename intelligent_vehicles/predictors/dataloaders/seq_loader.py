#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 09:31:28 2025

@author: nadya
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeqDataset that

• reads samples in the form
      {"cat": <str>, "obs": [L, 3], "target": [H, 3]}
  where pose = [x, y, yaw]

• builds past / future tensors whose *rows* are
      [x, y, yaw, vx, vy, vxy, vyaw]

• returns **three** tensors per __getitem__:
      cat_id  – one-hot vector  (shape [6])
      obs_arr – past trajectory (shape [L,7])
      tgt_arr – future trajectory(shape [H,7])

Nothing else is touched.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    # ------------------------------------------------------------------ #
    _CAT2IDX = {
        "car": 0, "motorcycle": 1, "pedestrian": 2,
        "van": 3, "truck": 4, "cyclist": 5
    }
    _NUM_CAT = len(_CAT2IDX)

    # ------------------------------------------------------------------ #
    def __init__(self, data_file: str, include_velocity: bool = True):
        self.include_velocity = include_velocity

        if not os.path.isfile(data_file):
            raise FileNotFoundError(data_file)

        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        # ---------- infer basic dims from first sample ---------------- #
        first = self.data[0]
        self.L = len(first["obs"])
        self.H = len(first["target"])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _yaw_wrap(delta):
        """Wrap to (-π, π]."""
        return (delta + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------ #
    def _build_traj(self, arr):
        """
        arr : np.ndarray [T,3]  (x,y,yaw)
        →     torch.Tensor [T,5] (x,y,vx,vy,vyaw)
        """
        pos_xy = arr[:, :2]                          # [T,2]
        yaw    = arr[:, 2]                           # [T]

        # ----- linear velocities on x-y ------------------------------- #
        vel_xy = pos_xy[1:] - pos_xy[:-1]            # [T-1,2]
        vel_xy = np.vstack([vel_xy[[0]], vel_xy])    # keep length T

        # ----- angular velocity (Δyaw) ------------------------------- #
        vyaw = self._yaw_wrap(yaw[1:] - yaw[:-1])
        vyaw = np.concatenate([[vyaw[0]], vyaw])     # [T]
        vyaw = vyaw[:, None]                         # [T,1]

        # ----- stack all ------------------------------------------------
        feats = np.hstack([pos_xy, vel_xy, vyaw])
        return torch.tensor(feats, dtype=torch.float32)   # [T,7]

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        cat_vec = torch.tensor(self._CAT2IDX[sample["obs_cat"]], dtype=torch.long)

        # ---- build trajectories ------------------------------------- #
        obs_arr = self._build_traj(np.asarray(sample["obs"]))      # [L,7]
        tgt_arr = self._build_traj(np.asarray(sample["target"]))   # [H,7]

        return cat_vec, obs_arr, tgt_arr
