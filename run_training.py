#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:18:44 2024

@author: nadya
"""


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataloaders.data_loader_deepaccident import DeepAccidentDataset
from visualization.visualize import Visualizer
from agents import IntelligentAgent
from utils import preprocess_dataset
from evaluation import calculate_ade, calculate_fde
import torch
import torch.nn as nn




if __name__ == '__main__':
    
    data_folder = '/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/DeepAccident'
    output_folder = '/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/output'
    preprocess = False
    
    if preprocess:
        preprocess_dataset(data_folder, prefix='train')
        preprocess_dataset(data_folder, prefix='valid')
        #self.preprocess_dataset(data_folder, prefix='test')
            

    # Initialize the agents 
    ego_vehicle = IntelligentAgent(name='ego_vehicle', data_folder=data_folder) #, predictor_checkpoint_file='checkpoints/ego_vehicle_trained_rnn2_model.pth')
    ego_vehicle_behind = IntelligentAgent(name='ego_vehicle_behind', data_folder=data_folder) #, predictor_checkpoint_file='checkpoints/ego_vehicle_behind_trained_rnn2_model.pth')
    other_vehicle = IntelligentAgent(name='other_vehicle', data_folder=data_folder) #, predictor_checkpoint_file='checkpoints/other_vehicle_trained_rnn2_model.pth')
    other_vehicle_behind = IntelligentAgent(name='other_vehicle_behind', data_folder=data_folder) #, predictor_checkpoint_file='checkpoints/other_vehicle_behind_trained_rnn2_model.pth')
    
    # Train individual predictiors for each agent
    ego_vehicle.predictor.train()
    ego_vehicle.predictor.evaluate()
    
    ego_vehicle_behind.predictor.train()
    ego_vehicle_behind.predictor.evaluate()
         
    other_vehicle.predictor.train()
    other_vehicle.predictor.evaluate()
         
    other_vehicle_behind.predictor.train()
    other_vehicle_behind.predictor.evaluate()
         
         
