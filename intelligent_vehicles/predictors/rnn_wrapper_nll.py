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

from intelligent_vehicles.predictors.sequential.rnn_nll import RNNPredictorNLL
from intelligent_vehicles.predictors.dataloaders.seq_loader import SeqDataset
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import os

class RNNWrapperNLL:
    def __init__(self, prediction_config):
        ## Add here all parameters
        # Configuration 
        if prediction_config["mode"] == "train":
            prediction_config["checkpoint"] = None
            prediction_config["hidden_size"] = 128
            prediction_config["num_layers"] = 2
            prediction_config["input_size"] = 2
            prediction_config["output_size"] = 2
    
            # Training 
            prediction_config["num_epochs"] = 30
            prediction_config["learning_rate"] = 0.001
            prediction_config["patience"] = 5
            prediction_config["normalize"] = False
            
            prediction_config["observation_length"] = 10
            prediction_config["prediction_horizon"] = 20
            
        self.batch_size = 128
        self.predictor = RNNPredictorNLL(prediction_config)

        if prediction_config["mode"] == "train":
            self.train_predictor(prediction_config["data_path"], prediction_config["save_path"])
            
    def format_input(self, tracklets):
        """Collect past trajectories as arrays (no resampling)."""
        past_trajs = []
        for t in tracklets:
            traj = [np.array([record.x, record.y, record.yaw]) for record in t['tracklet']]
            past_trajs.append(np.array(traj))
        return past_trajs
      
    def train_predictor(self, data_path, save_path):
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Provided path does not exist or is not a directory: {data_path}")
    
        train_path = os.path.join(data_path, "train.pkl")
        valid_path = os.path.join(data_path, "valid.pkl")
        test_path  = os.path.join(data_path, "test.pkl")
    
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"train.pkl is missing in: {data_path}")
    
        train_loader = DataLoader(SeqDataset(train_path), batch_size=self.batch_size, shuffle=True)
        test_loader = None
        valid_loader = None
        
        if os.path.isfile(test_path):
            test_loader = DataLoader(SeqDataset(test_path), batch_size=self.batch_size, shuffle=False)
            
        if os.path.isfile(valid_path):
            if test_loader is None:
                test_loader = DataLoader(SeqDataset(valid_path), batch_size=self.batch_size, shuffle=False)
            else:
                valid_loader = DataLoader(SeqDataset(valid_path), batch_size=self.batch_size, shuffle=True)
                
        save_path = os.path.join(save_path, "lstm_predictor_nll.pth")
        
        self.predictor.train(train_loader, valid_loader, save_path)
        self.predictor.evaluate(test_loader)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  predict  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def predict(self, past_trajs, prediction_horizon, prediction_sampling):
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

        pred_means, pred_covs = self.predictor.predict(past_trajs, prediction_horizon)

        N_orig = pred_means[0].shape[0]
        t_orig = np.linspace(0.0, prediction_horizon, N_orig)
        
        mean_trajs, cov_trajs = [], []
        for mu, Sigma in zip(pred_means, pred_covs):
            if mu.shape[0] != N_orig or Sigma.shape[0] != N_orig:
                raise ValueError(f"Expected {N_orig} steps, got {mu.shape[0]} (mean) and {Sigma.shape[0]} (cov)")
        
            mean_dict = {float(f"{t:.3f}"): mu[i].tolist()     for i, t in enumerate(t_orig)}
            cov_dict  = {float(f"{t:.3f}"): Sigma[i].tolist()  for i, t in enumerate(t_orig)}  # [pos_dim, pos_dim]
        
            mean_trajs.append(mean_dict)
            cov_trajs.append(cov_dict)
        
        return mean_trajs, cov_trajs
