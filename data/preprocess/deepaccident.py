#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: preprocess_dataset.py
Created on Sun Mar 17 15:06:36 2024
Author: nadya

Description:
This file contains a function to preprocess the DeepAccident dataset and store it in a unified format as a pickle file.
The dataset is organized in a hierarchical folder structure where data for each vehicle (e.g., "ego_vehicle", "other_vehicle", etc.)
is stored under separate subdirectories for different sensors (e.g., Camera and LiDAR). For each scenario, the data is further divided
by frame timestamps (computed from a fixed FPS). For each frame, the following are stored:
  - 'images': a dictionary with keys as camera sensor names and values as the file path to the corresponding image.
  - 'lidar': the file path to the LiDAR data (.npz file).
  - 'labels': a list of object states parsed from the label file.
  - 'ego_state': the parsed state of the ego vehicle (with id -100) from the label file.
  - 'calibration': the processed calibration data loaded from the corresponding file.

The output is a pickle file containing a dictionary with keys:
  - 'meta': metadata (dataset name, agents, sensors, and fps)
  - 'scenarios': a nested dictionary organized as:
      { scenario_name: { vehicle_name: { timestamp: { 'images': ..., 'lidar': ..., 'labels': ...,
                                                      'ego_state': ..., 'calibration': ... } } } }
The resulting pickle file is stored in the dataset folder under the respective prefix (e.g., 'train_data.pkl', 'valid_data.pkl').
"""

import os
import pickle
import copy
from typing import NamedTuple
import numpy as np


class Constants:
    CAMERA_SENSORS = ['Camera_FrontLeft',
                      'Camera_Front',
                      'Camera_FrontRight',
                      'Camera_BackLeft',
                      'Camera_Back',
                      'Camera_BackRight']
    LIDAR_SENSOR = 'lidar01'
    AGENTS = ['ego_vehicle',
              'ego_vehicle_behind',
              'other_vehicle',
              'other_vehicle_behind']
    FPS = 20

# Parse a label file to extract object states and ego vehicle state.
def parse_label_file(label_path):
    objects = []
    ego_state = None
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split()            
            obj = {
                'label': data[0],
                'x': float(data[1]),
                'y': float(data[2]),
                'z': float(data[3]),
                'length': float(data[4]),
                'width': float(data[5]),
                'height': float(data[6]),
                'yaw': float(data[7]),
                'vel_x': float(data[8]),
                'vel_y': float(data[9]),
                'obj_id': int(data[10])
            }
            
            # Skip static obstacles: if id is not in range 0 to 19999.
            # Also, for id -100 (ego vehicle), store separately.
            if obj['obj_id'] == -100:
                    ego_state = obj    
            if obj['obj_id'] < 0 or obj['obj_id'] >= 20000: continue
            objects.append(obj)
    return objects, ego_state

# Parse a calibration file and return its data. # to-do add processing if needed
def parse_calibration_file(calib_path):
    with open(calib_path, 'rb') as f:
        return pickle.load(f)

def preprocess_dataset(dataset_dir, prefix="train"):
    # Update dataset_dir to include the prefix folder.
    dataset_dir = os.path.join(dataset_dir, prefix)
    step = 1.0 / Constants.FPS

    # Gather all top-level subdirectories within the dataset directory.
    dirs = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    sub_dirs = []
    for d in dirs:
        d_path = os.path.join(dataset_dir, d)
        sub_dirs.extend([f"{d}/{subf}" for subf in os.listdir(d_path)])
        
    # Create a sensors list by copying camera sensors and appending the lidar sensor.
    sensors = copy.deepcopy(Constants.CAMERA_SENSORS)
    sensors.append(Constants.LIDAR_SENSOR)
    # Initialize the data dictionary with metadata.
    data = {
        'meta': {
            'name': 'DeepAccident',
            'agents': Constants.AGENTS,
            'sensors': sensors,
            'fps': Constants.FPS
        },
        'scenarios': {}
    }
    scenarios = {}
   
    # Iterate over each agent and each subdirectory to build the structure.
    for agent in Constants.AGENTS:
        for s in sub_dirs:
            sensor_base_path = os.path.join(dataset_dir, s, agent)
            if not os.path.isdir(sensor_base_path):
                continue
            for sensor in sensors:
                sensor_data_path = os.path.join(sensor_base_path, sensor)
                if not os.path.isdir(sensor_data_path):
                    continue
                scenario_folders = [sf for sf in os.listdir(sensor_data_path)]
                for scenario_folder in scenario_folders:
                    scenario_name = f"{s}/{scenario_folder}"
                    if scenario_name not in scenarios:
                        scenarios[scenario_name] = {}
                    if agent not in scenarios[scenario_name]:
                        scenarios[scenario_name][agent] = {}
                    folder_path = os.path.join(sensor_data_path, scenario_folder)
                    scene_files = sorted(os.listdir(folder_path))
                    for i, sc_file in enumerate(scene_files):
                        timestamp = i * step
                        if timestamp not in scenarios[scenario_name][agent]:
                            # Create a dict for each timestamp containing links and placeholders
                            scenarios[scenario_name][agent][timestamp] = {
                                'images': {},
                                'lidar': "",
                                'labels': [],
                                'ego_state': None,
                                'calibration': None
                            }

    # Fill in the data for images, lidar, annotations, ego_vehicle state, and calibration.
    for scenario_name, agents_dict in scenarios.items():
        s_parts = scenario_name.split('/')
        s_part = f"{s_parts[0]}/{s_parts[1]}"  # e.g., "Town01/weather_1"
        scenario_folder = s_parts[2]           # e.g., "scenarioA"
        for agent, timestamps_dict in agents_dict.items():
            for timestamp, data_dict in timestamps_dict.items():
                frame_index = int(timestamp / 0.05) + 1
                frame_str = str(frame_index).zfill(3)
                # Build image paths for each camera sensor.
                for sensor in Constants.CAMERA_SENSORS:
                    img_path = (
                        f"{dataset_dir}/{s_part}/{agent}/{sensor}/{scenario_folder}/"
                        f"{scenario_folder}_{frame_str}.jpg"
                    )
                    data_dict["images"][sensor] = img_path
                # Build lidar path.
                lidar_path = (
                    f"{dataset_dir}/{s_part}/{agent}/{Constants.LIDAR_SENSOR}/{scenario_folder}/"
                    f"{scenario_folder}_{frame_str}.npz"
                )
                data_dict["lidar"] = lidar_path
                # Process label file: parse annotations and ego vehicle state.
                lbl_path = (
                    f"{dataset_dir}/{s_part}/{agent}/label/{scenario_folder}/"
                    f"{scenario_folder}_{frame_str}.txt"
                )
                objs, ego_st = parse_label_file(lbl_path)
                data_dict["labels"] = objs
                data_dict["ego_state"] = ego_st
                # Process calibration file.
                calib_path = (
                    f"{dataset_dir}/{s_part}/{agent}/calib/{scenario_folder}/"
                    f"{scenario_folder}_{frame_str}.pkl"
                )
                calib_data = parse_calibration_file(calib_path)
                data_dict["calibration"] = calib_data
        print(f"{scenario_name} is processed")

    data['scenarios'] = scenarios
    output_pickle_path = os.path.join(dataset_dir, f"{prefix}_data.pkl")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved dataset to {output_pickle_path}")
    
if __name__ == '__main__':
    data_folder = '/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/DeepAccident'
    print(f"Preprocessing train dataset...{data_folder}")
    # preprocess_dataset(data_folder, prefix="train")
    preprocess_dataset(data_folder, prefix="valid")
