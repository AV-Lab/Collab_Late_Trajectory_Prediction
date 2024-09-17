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
from models.rnn_prediction import IndvidualRNNPredictor, MultiRNNPredictor
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
    agent_name = 'ego_vehicle'
    ego_vehicle_predictor = IndvidualRNNPredictor(data_folder, agent=agent_name, checkpoint_file='checkpoints/ego_vehicle_trained_rnn_model.pth')
    ego_vehicle = IntelligentAgent(name=agent_name, data_folder=data_folder,  predictor = ego_vehicle_predictor)
    
    agent_name = 'ego_vehicle_behind'
    ego_vehicle_behind_predictor = MultiRNNPredictor(data_folder, agent=agent_name, checkpoint_file='checkpoints/ego_vehicle_behind_trained_rnn2_model.pth')
    ego_vehicle_behind = IntelligentAgent(name=agent_name, data_folder=data_folder,  predictor = ego_vehicle_behind_predictor)
    
    agent_name = 'other_vehicle'
    other_vehicle_predictor = IndvidualRNNPredictor(data_folder, agent=agent_name, checkpoint_file='checkpoints/other_vehicle_trained_rnn_model.pth')
    other_vehicle = IntelligentAgent(name=agent_name, data_folder=data_folder,  predictor = other_vehicle_predictor)
    
    
    agent_name = 'other_vehicle_behind'
    other_vehicle_behind_predictor = MultiRNNPredictor(data_folder, agent=agent_name, checkpoint_file='checkpoints/other_vehicle_behind_trained_rnn2_model.pth')
    other_vehicle_behind = IntelligentAgent(name=agent_name, data_folder=data_folder,  predictor = other_vehicle_behind_predictor)

    
    
    #other_vehicle_behind = IntelligentAgent(name='other_vehicle_behind', 
    #                                        data_folder=data_folder, 
    #                                        predictor_checkpoint_file='checkpoints/other_vehicle_behind_trained_rnn2_model.pth')
    
    collab_vehicle = ego_vehicle
    vehicles = [ego_vehicle_behind, other_vehicle_behind, other_vehicle]
    ade = 0 
    fde = 0
    total_number_of_scenes = 0
    
    
    # Replay Scenarios
    for idx, (scenario_id, scenario_data) in enumerate(collab_vehicle.dataloader):
        print(scenario_id)
        
        for scene_id, scene_data in scenario_data.items():
            sid = int(scene_id) 
            for vehicle in vehicles:    
                vehcile_scenario_id, vehicle_scenario_data  = vehicle.dataloader[idx]
                vehicle_scene_data = vehicle_scenario_data[scene_id]
                vehicle_observations = vehicle.get_observations(vehicle_scene_data['labels'])
                vehicle.push_observations(sid, vehicle_observations, add_noise=True)
                
                # Extract past trajectories and predict 
                tracks = vehicle.extract_current_tracklets()
                predictions = vehicle.indiv_predict(tracks)
                vehicle.push_predictions(tracks.keys(), predictions)
                package = vehicle.transmit(sid, vehicle_scene_data['calib'])
                collab_vehicle.associate(package, scene_data['calib'])
                
            ##########################################################################################
            
            #For ego-vehicle 
            collab_vehicle_observations = collab_vehicle.get_observations(scene_data['labels'])
            collab_vehicle.push_observations(sid, collab_vehicle_observations, add_noise=True)
            tracks = collab_vehicle.extract_current_tracklets()
            trajectories = collab_vehicle.query_ground_truth(scenario_id, scene_id, tracks.keys())
            predictions = collab_vehicle.collab_predict(sid, tracks, trajectories)
            collab_vehicle.push_predictions(tracks.keys(), predictions)

            ade += calculate_ade(predictions, trajectories)
            fde += calculate_fde(predictions, trajectories)

        total_number_of_scenes += len(scenario_data.keys())
        
    print(ade/total_number_of_scenes)
    print(fde/total_number_of_scenes)

    
    # Initialize visualizer
    #visualizer = Visualizer(output_folder)
    
    # Visualize validation dataset with ground truth trajectories    
    #ego_vehicle.dataloader.load_valid()   
    #for idx, (scenario, objects_trajectories, objects_per_scene) in enumerate(ego_vehicle.dataloader):
    #    visualizer.visualize_scenario(idx, scenario, objects_trajectories, objects_per_scene)
    #    print('scenario {} processed'.format(idx))
        
        
         
