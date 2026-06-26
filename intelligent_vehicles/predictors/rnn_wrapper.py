#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:56:44 2025

@author: nadya
"""


import logging
import numpy as np 

logger = logging.getLogger(__name__)

from intelligent_vehicles.predictors.sequential.rnn import RNNPredictor

class RNNWrapper:
    def __init__(self, prediction_config):           
        self.predictor = RNNPredictor(prediction_config)
        self.fps = self.predictor.trained_fps
        self.observation_length = self.predictor.observation_length
        self.prediction_horizon = self.fps * prediction_config["prediction_horizon"]
        self.input_size = self.predictor.input_size
            
    def format_input(self, tracklets):
        """Collect past trajectories as arrays (no resampling)."""
        past_trajs = []
        ids = []
        for t in tracklets:
            ids.append(t['id'])
            traj = [np.array([record.x, record.y, record.yaw]) for record in t['tracklet']]
            past_trajs.append(np.array(traj))
            
        return ids, past_trajs
      
    def predict(self, past_trajs, trajectories):
        """
        No resampling. Returns step-indexed dicts.

        Args:
            past_trajs: tuple/list where:
                past_trajs[0] = ids
                past_trajs[1] = list[np.ndarray] with shape [T_obs, input_dim_raw]
            trajectories: dict containing GT future for each id

        Returns:
            pred_means : list[dict]  # [{time: [x,y]}, ...]
            pred_covs  : list[dict]  # [{time: [[var_x,0],[0,var_y]]}, ...]
        """
        ids = past_trajs[0]
        past_trajs = past_trajs[1]

        pred_positions = self.predictor.predict(
            past_trajs,
            self.prediction_horizon
        )

        dt = 1.0 / self.fps

        mean_trajs, cov_trajs = [], []
        cov_floor = 1e-4
        fallback_cov = 1.0

        for id_, pred_traj in zip(ids, pred_positions):
            pred_traj = np.asarray(pred_traj, dtype=np.float32)
            H_pred = len(pred_traj)

            t_orig = np.arange(1, H_pred + 1, dtype=np.float64) * dt

            Sigma = np.zeros((H_pred, 2, 2), dtype=np.float32)

            if id_ not in trajectories:
                Sigma[:, 0, 0] = fallback_cov
                Sigma[:, 1, 1] = fallback_cov
            else:
                gt_future = trajectories[id_].get("future", [])

                gt_xy = []
                for p in gt_future:
                    if hasattr(p, "x"):
                        gt_xy.append([p.x, p.y])
                    else:
                        gt_xy.append(p[:2])

                gt_xy = np.asarray(gt_xy, dtype=np.float32)
                H_gt = len(gt_xy)

                L = min(H_pred, H_gt)

                if L > 0:
                    err = gt_xy[:L, :2] - pred_traj[:L, :2]
                    var = np.maximum(err ** 2, cov_floor)

                    Sigma[:L, 0, 0] = var[:, 0]
                    Sigma[:L, 1, 1] = var[:, 1]

                    if H_pred > L:
                        Sigma[L:] = Sigma[L - 1]
                else:
                    Sigma[:, 0, 0] = fallback_cov
                    Sigma[:, 1, 1] = fallback_cov

            mean_dict = {
                float(f"{t:.3f}"): pred_traj[i, :2].tolist()
                for i, t in enumerate(t_orig)
            }

            cov_dict = {
                float(f"{t:.3f}"): Sigma[i].tolist()
                for i, t in enumerate(t_orig)
            }

            mean_trajs.append(mean_dict)
            cov_trajs.append(cov_dict)

        return mean_trajs, cov_trajs
