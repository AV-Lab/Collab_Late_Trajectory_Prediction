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

from intelligent_vehicles.predictors.sequential.transformer_category import TransformerPredictorWithCategory
from intelligent_vehicles.predictors.sequential.transformer import TransformerPredictor
from intelligent_vehicles.predictors.dataloaders.seq_loader import SeqDataset
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import os

class TransformerWrapper:
    def __init__(self, prediction_config):
        ## Add here all parameters
        # Configuration 
        if prediction_config["mode"] == "train":
            prediction_config["past_trajectory"] = 10
            prediction_config["future_trajectory"] = 20
            prediction_config["checkpoint"] = None
            prediction_config["in_features"] = 2
            prediction_config["out_features"] = 2
            prediction_config["num_heads"] = 4
            prediction_config["num_encoder_layers"] = 3
            prediction_config["num_decoder_layers"] = 3
            prediction_config["embedding_size"] = 256
            prediction_config["dropout_encoder"] = 0.25
            prediction_config["dropout_decoder"] = 0.25
            prediction_config["batch_first"] = True
            prediction_config["actn"] = "relu"
            prediction_config["num_epochs"] = 50
            prediction_config["cat_embed_dim"] = 8
            prediction_config["num_categories"] = 6
            prediction_config["normalize"] = False 
        
            # Optimizer parameters
            prediction_config["lr_mul"] = 0.2
            prediction_config["n_warmup_steps"] = 3500
            prediction_config["optimizer_betas"] = (0.9, 0.98)
            prediction_config["optimizer_eps"] = 1e-9
        
            # Early stopping parameters
            prediction_config["early_stopping_patience"] = 15
            prediction_config["early_stopping_delta"] = 0.01
            
        self.batch_size = 32
                
        if prediction_config["category"]:
            self.predictor = TransformerPredictorWithCategory(prediction_config) 
        else:
            self.predictor = TransformerPredictor(prediction_config)   

        if prediction_config["mode"] == "train":
            self.train_predictor(prediction_config["data_path"], prediction_config["save_path"])
            
    def format_input(self, tracklets):
        past_trajs = []
        for t in tracklets:
            traj = [np.array([record.x, record.y, record.yaw]) for record in t['tracklet']]
            cur_pos = np.concatenate([t['current_pos'][:2], t['current_pos'][-1:]])
            traj.append(cur_pos)
            past_trajs.append(np.array(traj))
        return past_trajs
    
    def resample_input(self, past_trajs, tracker_fps):
        resampled_trajs = []
        yaw_index = 2
        
        for traj in past_trajs:
            N, D = traj.shape
            duration = (N - 1) / tracker_fps
            new_len = int(round(duration * self.predictor.trained_fps)) + 1
    
            t_original = np.linspace(0, duration, N)
            t_new = np.linspace(0, duration, new_len)
    
            interpolated = np.zeros((new_len, D))
    
            for d in range(D):
                if d == yaw_index:
                    yaw = traj[:, d]
                    x = np.cos(yaw)
                    y = np.sin(yaw)
    
                    x_interp = interp1d(t_original, x, kind='linear', fill_value='extrapolate')(t_new)
                    y_interp = interp1d(t_original, y, kind='linear', fill_value='extrapolate')(t_new)
                    interpolated[:, d] = np.arctan2(y_interp, x_interp)
                else:
                    interpolated[:, d] = interp1d(t_original, traj[:, d], kind='linear', fill_value='extrapolate')(t_new)
    
            if interpolated.shape[0] < self.predictor.observation_length:
                pad_len = self.predictor.observation_length - interpolated.shape[0]
                padded = np.pad(interpolated, ((pad_len, 0), (0, 0)), mode='constant')
                resampled_trajs.append(padded)
            else:
                resampled_trajs.append(interpolated[-self.predictor.observation_length:])
    
        return resampled_trajs
      
    def train_predictor(self, data_path, save_path):
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Provided path does not exist or is not a directory: {data_path}")
    
        train_path = os.path.join(data_path, "train.pkl")
        valid_path = os.path.join(data_path, "valid.pkl")
        test_path  = os.path.join(data_path, "test.pkl")
    
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"train.pkl is missing in: {data_path}")
    
        train_loader = DataLoader(SeqDataset(train_path, normalize=True), batch_size=self.batch_size, shuffle=True)
        test_loader = None
        valid_loader = None
        
        if os.path.isfile(test_path):
            test_loader = DataLoader(SeqDataset(test_path), batch_size=self.batch_size, shuffle=False)
            
        if os.path.isfile(valid_path):
            if test_loader == None:
                test_loader = DataLoader(SeqDataset(valid_path), batch_size=self.batch_size, shuffle=False)
            else:
                valid_loader = DataLoader(SeqDataset(valid_path), batch_size=self.batch_size, shuffle=True)
                
        save_path = os.path.join(save_path, "transformer_predictor_norm.pth")
        
        self.predictor.train(train_loader, valid_loader, save_path)
        self.predictor.evaluate(test_loader)
            
    def predict(self, past_trajs, prediction_horizon, prediction_sampling):
        predictions = self.predictor.predict(past_trajs, prediction_horizon)
        
        N_orig = len(predictions[0])
        N_target = prediction_horizon * prediction_sampling
    
        t_orig = np.linspace(0, prediction_horizon, N_orig)
        t_new = np.linspace(0, prediction_horizon, N_target)
    
        resampled_trajs = []
        for pred in predictions:
            if pred.shape[0] != N_orig:
                raise ValueError(f"Expected {N_orig} steps, got {pred.shape[0]}")
    
            D = pred.shape[1]
            interp_pred = np.zeros((N_target, D))
            for d in range(D):
                interp_pred[:, d] = interp1d(t_orig, pred[:, d], kind='linear', fill_value='extrapolate')(t_new)
    
            resampled_dict = {float(f"{t:.3f}"): interp_pred[i].tolist() for i, t in enumerate(t_new)}
            resampled_trajs.append(resampled_dict)
    
        return resampled_trajs
        
        