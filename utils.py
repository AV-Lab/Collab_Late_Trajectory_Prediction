#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:06:36 2024

@author: nadya
"""

import os
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
from constants import Constants


def preprocess_dataset(dataset_dir, prefix):
    """
        Transofrms dataset into the format:
            scenario 1 (name of the scenario): {scene_id: {images: [Camera_Front_image_path, 
                                                                               Camera_Front_Left_image_path, ...],  
            scenario 2 (name of the scenario): {scene_id: {images: [Camera_Front_image_path, 
                                                                               Camera_Front_Left_image_path, ...], 
                                                                      label: '', calib: ''}}},
                                                                     
            .
            .
            .
    """
    
    # First fetch all the scenarios and scenes
    dataset_dir = dataset_dir + '/' + prefix
    dirs = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    sub_dirs = []
    for d in dirs:
        sub_dirs.extend(['{}/{}'.format(d,f) for f in os.listdir('{}/{}'.format(dataset_dir, d))])
    
    for agent in Constants.AGENTS:
        scenarios = {}
        #This Loop generates scenarios paths 
        for s in sub_dirs: # for each type (total 4)
            for sensor in Constants.SENSORS: # iterate through each sensor
                sensor_data_path = '{}/{}/{}/{}/'.format(dataset_dir, s, agent, sensor)
                files = [f for f in os.listdir(sensor_data_path)]
                for f in files:
                    scenario = s+'/'+f
                    if f not in scenarios:
                        scenarios[scenario] = {}
                    
                    scenes = sorted([sc for sc in os.listdir(sensor_data_path+f+'/')])
                    
                    for sc in scenes:
                        scene = sc[:-4].split('_')[-1]
                        
                        if scene not in scenarios[scenario]:
                            scenarios[scenario][scene] = {'images':[], 'labels': '', 'calib':''} 
                            
                                
        # For each scenario and scene we fetch sensor, labels and calib data 
        train_data = []
                         
        for k,v in scenarios.items():
            k_path = k.split('/')
            s = '{}/{}'.format(k_path[0], k_path[1])
            scenario = k_path[2]
            objects_trajectories = {}
            for scene in v.keys():
                for sensor in Constants.SENSORS: # iterate through each sensor
                    scenarios[k][scene]['images'].append('{}/{}/{}/{}/{}/{}_{}.jpg'.format(dataset_dir, s, agent, sensor, scenario, scenario, str(scene)))
                scenarios[k][scene]['labels'] = '{}/{}/{}/{}/{}/{}_{}.txt'.format(dataset_dir, s, agent, Constants.ANN_DIR, scenario, scenario, str(scene))
                scenarios[k][scene]['calib'] = '{}/{}/{}/{}/{}/{}_{}.pkl'.format(dataset_dir, s, agent, Constants.CALLIB_DIR, scenario, scenario, str(scene))
                exctract_trajectories_for_trainning(scenarios[k][scene]['labels'], objects_trajectories)
                
            ### for each scenario generate train data
            generate_train_samples(objects_trajectories, train_data)

        with open('{}/{}_{}.pkl'.format(dataset_dir, agent, Constants.TRAJ_FILE_SUF), 'wb') as f:
            pickle.dump(train_data, f)
            
        with open('{}/{}_{}.pkl'.format(dataset_dir, agent, Constants.SENSORS_DATA_SUF), 'wb') as f:
            pickle.dump(scenarios, f)
            
def parse_state(data):
    label = data[0]
    x = float(data[1])
    y = float(data[2])
    z = float(data[3])
    width = float(data[4])
    height = float(data[5])
    length = float(data[6])
    yaw = float(data[7])
    vel_x = float(data[8])
    vel_y = float(data[9])
    vector = [x, y, z, yaw, vel_x, vel_y]
    
    return label, width, height, length, vector
        
def exctract_trajectories_for_trainning(labels_file, objects_trajectories):
    ''' format:
        cls_label, x, y, z, bbox_x, bbox_y, bbox_z, yaw_angle (in radians, range -pi ~ pi), vel_x (unit m/s), vel_y (unit m/s), obj_id, number_lidar_pts_of_this_obj, flag_visible_in_cameras 
        obj_id > 20000 static objects
        first line ego vehicle vel_x and vel_y
        obj_id = -100 ego vehicle
    '''   
    with open(labels_file, 'r') as file:
        next(file)
        for line in file:
            data = line.strip().split(' ')
            id_ = int(data[10])
            
            if id_ == -100 or id_ >= 20000:
                continue 
            
            label, width, height, length, vector = parse_state(data)
            
            if id_ in objects_trajectories:
                objects_trajectories[id_][2].append(vector)
            else:
                objects_trajectories[id_] = (label, (width, height, length), [vector])      

        
def generate_train_samples(objects_trajectories, train_data):
    for k,v in objects_trajectories.items():
        trajectories = v[2]
        count = (len(trajectories) - Constants.MIN_TRACKING_FRAMES - Constants.PREDICTION_HORIZON) / Constants.SLIDING_WINDOW + 1
        for i in range(int(count)):
            step = Constants.SLIDING_WINDOW*i
            x = trajectories[step : Constants.MIN_TRACKING_FRAMES+step]
            y = trajectories[Constants.MIN_TRACKING_FRAMES+step : Constants.MIN_TRACKING_FRAMES+Constants.PREDICTION_HORIZON+step]
            train_data.append((x,y))
            
            
# Add extracting for min_len
def extract_trajectories_per_scene(scene_id, objects_trajectories, scene_objects, min_len=-1):
    trajectories = []
    for obj in scene_objects:
        bgs = int(objects_trajectories[obj][0]) # first scene_id where object appeared
        lb = int(scene_id) - bgs + 1 #  correctly caldulate the offset 
        rb = lb + Constants.PREDICTION_HORIZON # calculate right border
        traj = objects_trajectories[obj][1]
        if rb >= len(traj): rb = len(traj)
        if lb <= rb: 
            traj = traj[lb:rb]
        else: traj = []
        traj.extend([[0,0,0] for _ in range(Constants.PREDICTION_HORIZON - len(traj))])
        trajectories.append(traj)                
    return trajectories

def weighted_average(predictions, weights):
    if not predictions or not weights or len(predictions) != len(weights):
        raise ValueError("Predictions and weights must be non-empty and of the same length.")
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0
    
    weighted_sum = sum(p * w for p, w in zip(predictions, weights))
    return weighted_sum / total_weight

def normalize_confidences(confidences):
    total_confidence = sum(confidences)
    if total_confidence == 0:
        return [0] * len(confidences)  # Avoid division by zero, return zeros
    return [conf / total_confidence for conf in confidences]


def simulate_detection_error(position, distance, base_noise_std, distance_scaling_factor):
    """
    Simulate detection error for a single position based on its distance from the sensor.

    :param true_position: The true position of the object (x, y).
    :param distance: The distance to the object.
    :param base_noise_std: Base standard deviation of Gaussian noise at reference distance.
    :param distance_scaling_factor: Factor to scale the noise with distance.
    :return: Detected position with noise.
    """
    # Calculate the noise standard deviation based on distance
    noise_std = base_noise_std * (1 + distance_scaling_factor * distance)
    noise = np.random.normal(0, noise_std, size=(6,))
    noise[3:] = 0
    position = position + noise
    
    return position


