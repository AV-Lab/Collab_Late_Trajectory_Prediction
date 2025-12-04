#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:56:44 2025

@author: nadya
"""


import logging
import numpy as np 

logger = logging.getLogger(__name__)

from intelligent_vehicles.predictors.sequential.rnn_nll import RNNPredictorNLL

class RNNWrapperNLL:
    def __init__(self, prediction_config):           
        self.predictor = RNNPredictorNLL(prediction_config)
        self.fps = self.predictor.trained_fps
        self.observation_length = self.predictor.observation_length
        self.prediction_horizon = self.fps * prediction_config["prediction_horizon"]
        self.input_size = self.predictor.input_size
            
    def format_input(self, tracklets):
        """Collect past trajectories as arrays (no resampling)."""
        past_trajs = []
        for t in tracklets:
            traj = [np.array([record.x, record.y, record.yaw]) for record in t['tracklet']]
            past_trajs.append(np.array(traj))
        return past_trajs
      
    def predict(self, past_trajs):
        """
        No resampling. Returns step-indexed dicts.

        Args:
            past_trajs: list[np.ndarray] with shape [T_obs, input_dim_raw]
            prediction_horizon: seconds (the internal predictor handles FPS/steps)
            prediction_sampling: IGNORED (kept for API compatibility)

        Returns:
            pred_means : list[dict]  # [{step: [x,y,...]}, ...] with step ∈ {0..H-1}
            pred_covs  : list[dict]  # [{step: [[..],[..],...]} per-step Σ_pos], same keys
        """

        pred_means, pred_covs = self.predictor.predict(past_trajs, self.prediction_horizon)

        dt = 1.0 / self.fps 
        H  = self.prediction_horizon 
        t_orig = np.arange(1, H + 1, dtype=np.float64) * dt
        
        mean_trajs, cov_trajs = [], []
        for mu, Sigma in zip(pred_means, pred_covs):
            mean_dict = {float(f"{t:.3f}"): mu[i].tolist()     for i, t in enumerate(t_orig)}
            cov_dict  = {float(f"{t:.3f}"): Sigma[i].tolist()  for i, t in enumerate(t_orig)}  # [pos_dim, pos_dim]
        
            mean_trajs.append(mean_dict)
            cov_trajs.append(cov_dict)
        
        return mean_trajs, cov_trajs
