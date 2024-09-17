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
    
    #data_folder = '/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/DeepAccident'
    data_folder = 'data/DeepAccident'
    output_folder = '/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/output'
    preprocess = False
    
    if preprocess:
        preprocess_dataset(data_folder, prefix='train')
        preprocess_dataset(data_folder, prefix='valid')
        #self.preprocess_dataset(data_folder, prefix='test')
            

    # Initialize the agents 
    vehicle = IntelligentAgent(name='ego_vehicle', data_folder=data_folder, predictor_checkpoint_file='checkpoints/ego_vehicle_trained_rnn_model.pth')
    #vehicle = IntelligentAgent(name='ego_vehicle_behind', data_folder=data_folder, predictor_checkpoint_file='checkpoints/ego_vehicle_behind_trained_rnn_model.pth')
    #vehicle = IntelligentAgent(name='other_vehicle', data_folder=data_folder, predictor_checkpoint_file='checkpoints/other_vehicle_trained_rnn_model.pth')
    #vehicle = IntelligentAgent(name='other_vehicle_behind', data_folder=data_folder, predictor_checkpoint_file='checkpoints/other_vehicle_behind_trained_rnn_model.pth')
    
    
    ade = 0 
    fde = 0
    total_number_of_scenes = 0
    
    
    # Replay Scenarios
    for idx, (scenario_id, scenario) in enumerate(vehicle.dataloader):
        print(scenario_id)
        for scene_id, scene_data in scenario.items():
            
            # Extract observations and update collaboration graph 
            vehicle_observations = vehicle.get_observations(scene_data['labels'])
            vehicle.push_observations(int(scene_id), vehicle_observations, add_noise=True)
            
            # Run predictions on tracklets and update collaboration graph
            tracks = vehicle.extract_current_tracklets()
            predictions = vehicle.indiv_predict(tracks)
            vehicle.push_predictions(tracks.keys(), predictions)
            
            
            # Query ground truth trajectories 
            trajectories = vehicle.query_ground_truth(scenario_id, scene_id, tracks.keys())
                        
            # Compute Errors
            ade += calculate_ade(predictions, trajectories)
            fde += calculate_fde(predictions, trajectories)
            
        total_number_of_scenes += len(scenario.keys())
  
    print(ade/total_number_of_scenes)
    print(fde/total_number_of_scenes)
