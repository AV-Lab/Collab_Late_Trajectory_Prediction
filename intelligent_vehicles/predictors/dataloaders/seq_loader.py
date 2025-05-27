#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeqDataset that (1) infers dimensionalities, (2) adds velocities only for x-,y-,z-
coordinates, and (3) guarantees obs / target tensors have equal feature length.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, data_file, include_velocity: bool = True):
        """
        Args
        ----
        data_file : str
            Path to *.pkl containing a list of samples.
            Each sample can be either
                ([obs_seq], [tgt_seq])     # tuple / list
            or
                {"obs": [obs_seq], "target": [tgt_seq]}  # dict
        include_velocity : bool
            Whether to append (vx, vy, [vz]) to position channels.
        """
        self.include_velocity = include_velocity

        if not os.path.isfile(data_file):
            raise FileNotFoundError(data_file)

        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        # ---- infer dimensionalities from first sample -----------------
        first_obs, first_tgt = self._unpack_sample(self.data[0])
        self.input_dim  = len(first_obs[0])   # 2, 3, or 4
        self.output_dim = len(first_tgt[0])   # 2 or 3
        self.pos_dim    = min(self.output_dim, 3)  # dims that get velocity

        assert self.output_dim <= self.input_dim, (
            "Output cannot have more positional dims than input")

        # ---- compute stats -------------------------------------------
        self.mean, self.std = self._calculate_statistics()

    # ------------------------------------------------------------------
    def _unpack_sample(self, sample):
        """Return (obs, target) regardless of tuple / dict layout."""
        if isinstance(sample, dict):
            return sample["obs"], sample["target"]
        return sample  # assume (obs, target)

    # ------------------------------------------------------------------
    def _add_velocity(self, seq_tensor):
        """Append vx,vy(,vz) computed on the first `pos_dim` columns."""
        vel = seq_tensor[1:, :self.pos_dim] - seq_tensor[:-1, :self.pos_dim]
        vel = torch.cat([vel[[0]], vel], dim=0)                   # keep length
        return torch.cat([seq_tensor, vel], dim=1)                # [T, D+pos]

    # ------------------------------------------------------------------
    def _pad_target(self, tgt_tensor):
        """Pad zeros if input has yaw but output does not."""
        pad_cols = self.input_dim - self.output_dim               # 0 or 1
        if pad_cols:                                              # yaw present
            zeros = torch.zeros(tgt_tensor.shape[0], pad_cols,
                                dtype=tgt_tensor.dtype)
            tgt_tensor = torch.cat([tgt_tensor, zeros], dim=1)    # +yaw (0)
        return tgt_tensor

    # ------------------------------------------------------------------
    def _calculate_statistics(self):
        all_tensors = []

        for sample in self.data:
            obs, tgt = self._unpack_sample(sample)
            obs_t  = torch.tensor(obs,  dtype=torch.float32)
            tgt_t  = torch.tensor(tgt,  dtype=torch.float32)

            if self.include_velocity:
                obs_t = self._add_velocity(obs_t)
                tgt_t = self._add_velocity(tgt_t)

            tgt_t = self._pad_target(tgt_t)                       # align width
            all_tensors.extend([obs_t, tgt_t])

        cat = torch.cat(all_tensors, dim=0)                       # safe concat
        mean = cat.mean(dim=0)
        std  = cat.std(dim=0)
        std[std < 1e-6] = 1.0                                     # avoid /0
        return mean, std

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        obs, tgt = self._unpack_sample(self.data[idx])
        obs_t = torch.tensor(obs, dtype=torch.float32)
        tgt_t = torch.tensor(tgt, dtype=torch.float32)

        if self.include_velocity:
            obs_t = self._add_velocity(obs_t)
            tgt_t = self._add_velocity(tgt_t)

        tgt_t = self._pad_target(tgt_t)                           # align width
        return obs_t, tgt_t
