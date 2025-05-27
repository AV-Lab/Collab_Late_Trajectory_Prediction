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

from intelligent_vehicles.predictors.sequential.rnn import RNNPredictor
from intelligent_vehicles.predictors.dataloaders.seq_loader import SeqDataset
from torch.utils.data import DataLoader
import os

class RNNWrapper:
    def __init__(self, prediction_config):
        ## Add here all parameters
        # Configuration 
        prediction_config["hidden_size"] = 256
        prediction_config["num_layers"] = 2
        prediction_config["input_size"] = 4
        prediction_config["output_size"] = 3
    
        # Training 
        prediction_config["num_epochs"] = 2
        prediction_config["learning_rate"] = 0.001
        prediction_config["patience"] = 5
        prediction_config["normalize"] = True

        if prediction_config["mode"] == "train":
            prediction_config["checkpoint"] = None
            
        self.batch_size = 32
                        
        self.predictor = RNNPredictor(prediction_config)   

        if prediction_config["mode"] == "train":
            self.train_predictor(prediction_config["data_path"], prediction_config["save_path"])
            
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
            if test_loader == None:
                test_loader = DataLoader(SeqDataset(valid_path), batch_size=self.batch_size, shuffle=False)
            else:
                valid_loader = DataLoader(SeqDataset(valid_path), batch_size=self.batch_size, shuffle=True)
                
        save_path = os.path.join(save_path, "lstm_predictor.pth")
        
        self.predictor.train(train_loader, valid_loader, save_path)
        self.predictor.evaluate(test_loader)
            
    def predict(self, tracklets):
        pass
        
        