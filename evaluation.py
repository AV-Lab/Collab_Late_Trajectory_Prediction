#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:56:13 2024

@author: nadya
"""


import numpy as np

def calculate_ade(predictions, ground_truth):
    """
    Calculate Average Displacement Error (ADE) for motion forecasting.
    
    Args:
    - predictions: List of predicted trajectories, each trajectory is a numpy array of shape (T, 2)
    - ground_truth: List of ground truth trajectories, each trajectory is a numpy array of shape (T, 2)
    
    Returns:
    - ade: Average Displacement Error
    """
    num_trajectories = len(ground_truth)
    total_pred = 0
    total_error = 0.0
    
    for i in range(num_trajectories):
        pred_traj = predictions[i]
        true_traj = ground_truth[i]
        
        if len(ground_truth[i]) > 0: 

            # Calculate Euclidean distance for each time step
            errors = np.linalg.norm(pred_traj - true_traj, axis=1)
            # Average displacement error for this trajectory
            total_error += np.sum(errors) / len(pred_traj)
            #print(pred_traj, true_traj, total_error)
            #print("###########################################################")
            total_pred += 1
    
    ade = total_error / total_pred

    return ade

def calculate_fde(predictions, ground_truth):
    """
    Calculate Final Displacement Error (FDE) for motion forecasting.
    
    Args:
    - predictions: List of predicted trajectories, each trajectory is a numpy array of shape (T, 2)
    - ground_truth: List of ground truth trajectories, each trajectory is a numpy array of shape (T, 2)
    
    Returns:
    - fde: Final Displacement Error
    """
    num_trajectories = len(predictions)
    total_error = 0.0
    total_pred = 0
    
    for i in range(num_trajectories):
        pred_traj = predictions[i][-1]  # Last position of predicted trajectory
        
        if len(ground_truth[i]) > 0: 
            
            true_traj = ground_truth[i][-1]  # Last position of ground truth trajectory
    
            # Calculate Euclidean distance between final positions
            error = np.linalg.norm(pred_traj - true_traj)
            total_error += error
            
            total_pred += 1
    
    fde = total_error / total_pred
    return fde