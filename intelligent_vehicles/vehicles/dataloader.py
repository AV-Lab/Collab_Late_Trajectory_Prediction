#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:30:27 2025

@author: nadya
"""


import os
import pickle
import cv2
import numpy as np
import logging


class TrajDataloader:
    """
    This class holds basic parameters (pickle path, scenario name, sensors filter) and
    defers preloading of sensor data into memory to a separate method 'preload_data()'.
    Once preloaded, it stores loaded sensor data (images, LiDAR, calibration, labels, ego_state)
    keyed by timestamp, and computes object trajectories across frames.
    
    After preloading, get_frame_data(t) returns all loaded sensor data plus trajectories for timestamp t.
    """
    def __init__(self, pickle_path, sensors, fps):
        self.data_file = pickle_path
        self.sensors = sensors
        self.vehicle_fps = fps
        
        print(sensors)
        
    def extract_all_scenarios(self):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            print(f"data loaded")
            return data.keys()
        
    def preload_data(self, scenario_name):
        """
        Preloads sensor data from the pickle file for the specified scenario.
        Skips frames based on vehicle FPS.
        For each timestamp, reads sensor files (images, LiDAR, calibration) into memory.
        Also computes trajectories for each object (past, current, future) across frames.
        """
        
        self.loaded_frames = {}
        self.current_frame = 0
        
        with open(self.data_file, 'rb') as f:
            dataset = pickle.load(f)

        # Extract scenario data: expected format is { scenario_name: { timestamp: frame_data, ... } }
        scenario_data = dataset.get(scenario_name, {})
        if not scenario_data:
            raise ValueError(f"Scenario '{scenario_name}' not found in dataset.")
    
        original_timestamps = sorted(scenario_data.keys())
    
        if len(original_timestamps) < 2:
            raise ValueError("Scenario data must contain at least two timestamps to compute dataset FPS.")
    
        time_diffs = [t2 - t1 for t1, t2 in zip(original_timestamps[:-1], original_timestamps[1:])]
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        dataset_fps = round(1 / avg_time_diff)
    
        # Determine frame skipping step
        frame_step = int(dataset_fps / self.vehicle_fps)
        if frame_step < 1:
            frame_step = 1
    
        for t in original_timestamps[::frame_step]:
            frame_info = scenario_data[t]
            loaded = {}
    
            # Load images for each sensor
            loaded['images'] = {}
            for sensor, path in frame_info.get("images", {}).items():
                if sensor in self.sensors:
                    loaded['images'][sensor] = cv2.imread(path) if os.path.isfile(path) else None
    
            # Load LiDAR data
            lidar_path = frame_info.get("lidar", "")
            if lidar_path and os.path.isfile(lidar_path):
                loaded['lidar'] = np.load(lidar_path)
            else:
                loaded['lidar'] = None
    
            loaded['labels'] = frame_info.get("labels", [])
            loaded['ego_state'] = frame_info.get("ego_state", None)
            loaded['calibration'] = frame_info.get("calibration", {})
    
            self.loaded_frames[t] = loaded
    
        self.timestamps = sorted(self.loaded_frames.keys())
        self.trajectories = self._compute_trajectories()

    def _compute_trajectories(self):
        """
        Computes trajectories for each object across all frames and stores them directly into loaded_frames.
        For each timestamp and each object, computes:
            - 'past': states from timestamps < t
            - 'current': state and bbox at timestamp t
            - 'future': states from timestamps > t
        """
        trajectories = {}
    
        # Accumulate states for each object across timestamps
        for t in self.timestamps:
            labels = self.loaded_frames[t]["labels"]
            for label in labels:
                obj_id = label['obj_id'] if isinstance(label, dict) else label.obj_id
                if obj_id not in trajectories:
                    trajectories[obj_id] = []
                trajectories[obj_id].append((t, label))
    
        # For each timestamp, store past, current (with bbox), future
        for t in self.timestamps:
            frame_traj = {}
            for obj_id, entries in trajectories.items():
                past = [s for (ts, s) in entries if ts < t]
                future = [s for (ts, s) in entries if ts > t]
                current = None
                bbox = None
    
                for (ts, s) in entries:
                    if abs(ts - t) < 1e-3:
                        current = s
                        bbox = {
                            'width': s['width'],
                            'height': s['height'],
                            'length': s['length'],
                            'position': (s['x'], s['y'], s['z']),
                            'yaw': s['yaw']
                        }
                        break
    
                if current is not None:
                    frame_traj[obj_id] = {
                        'past': past,
                        'current': current,
                        'bbox': bbox,
                        'future': future
                    }
    
            # Store trajectories within the loaded_frames at the current timestamp
            self.loaded_frames[t]['trajectories'] = frame_traj

        

    def get_frame_data(self, t):
        """
        Returns the loaded sensor data and trajectories for timestamp t.
        """
        if self.current_frame >= len(self.timestamps):
            return None
        else:
            t_frame = self.timestamps[self.current_frame]
            self.current_frame += 1
            return self.loaded_frames[t_frame],self.current_frame 

    def __iter__(self):
        pass
