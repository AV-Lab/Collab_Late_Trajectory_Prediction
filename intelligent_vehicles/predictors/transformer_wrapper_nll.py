#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:56:44 2025

@author: nadya
"""


import logging
from queue import Queue
import numpy as np 
from collections import namedtuple

logger = logging.getLogger(__name__)

from intelligent_vehicles.predictors.sequential.transformer_nll import TransformerPredictorNLL

class TransformerWrapperNLL:
    def __init__(self, prediction_config):
        self.predictor = TransformerPredictorNLL(prediction_config)
        self.fps = self.predictor.trained_fps
        self.observation_length = self.predictor.past_trajectory
        self.prediction_horizon = self.fps * prediction_config["prediction_horizon"]
        self.input_size = self.predictor.in_features
            
    def format_input(self, tracklets):
        past_trajs = []
        for t in tracklets:
            traj = [np.array([record.x, record.y, record.yaw]) for record in t['tracklet']]
            past_trajs.append(np.array(traj))
            
        return past_trajs
    
            
    def predict(self, past_trajs):
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
        
        