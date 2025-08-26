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

from collections import namedtuple
from typing import List, Dict
position = namedtuple('Position', ['x', 'y', 'z','yaw'])

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

    def ego_motion_compensation(self, detections, calibration) -> List[Dict[str, float]]:
        """
        LiDAR-frame → world-frame for each detection.
    
        Parameters
        ----------
        detections   : list of dicts with keys at least
                       {'x','y','z','yaw', 'length','width','height', 'obj_id', ...}
        calibration  : {
            'lidar_to_ego':   4×4 np.ndarray,
            'ego_to_world':   4×4 np.ndarray
          }
    
        Returns
        -------
        list of dicts in world frame (same objects, in-place edited & returned)
        """
        T_lw = calibration["ego_to_world"] @ calibration["lidar_to_ego"]  # 4×4
        R_lw = T_lw[:3, :3]
    
        # ego heading = yaw of LiDAR X-axis in world frame
        ego_heading = np.arctan2(R_lw[1, 0], R_lw[0, 0])   # atan2(y,x)   
        compensated = []
        
        for det in detections:
            # ----- position
            pos_lidar = np.array([det["x"], det["y"], det["z"], 1.0])
            pos_world = T_lw @ pos_lidar
    
            # ----- yaw  (add headings, then wrap)
            yaw_world = det["yaw"] + ego_heading
            yaw_world = (yaw_world + np.pi) % (2 * np.pi) - np.pi   # wrap to (-π,π]
    
            new_det = det.copy()
            new_det["x"], new_det["y"], new_det["z"] = pos_world[:3]
            new_det["yaw"] = yaw_world
            compensated.append(new_det)
    
        return compensated

    def _compute_trajectories(self):
        """
        Build per-object trajectories in **world frame** and store them into
        self.loaded_frames[t]['trajectories'].
    
            past   : list[position]  (ts < t)
            current_state : np.ndarray[7]  (x,y,z,l,w,h,yaw)
            future : list[position]  (ts > t)
        """
        # 1) gather every object’s time-ordered states in WORLD frame
        trajectories = {}             # obj_id → [(t, box_dict_world), ...]
    
        for t in self.timestamps:
            calib   = self.loaded_frames[t]['calibration']
            labels  = self.loaded_frames[t]['labels']
            labels_w = self.ego_motion_compensation(labels, calib)  # NEW

            # keep for later (debug / visualisation if you want)
            self.loaded_frames[t]['labels_world'] = labels_w
    
            for det in labels_w:
                oid = det['obj_id'] if isinstance(det, dict) else det.obj_id
                trajectories.setdefault(oid, []).append((t, det))
    
        # 2) for every frame build {obj_id: {past,current_state,future}}
        for t in self.timestamps:
            frame_traj = {}
    
            for oid, seq in trajectories.items():
                past   = [position(s['x'], s['y'], s['z'], s['yaw'])
                          for (ts, s) in seq if ts <= t]
                future = [position(s['x'], s['y'], s['z'], s['yaw'])
                          for (ts, s) in seq if ts > t]
                
                if len(past) == 0 or len(future) == 0: continue
    
                state = next(
                    ((np.array([s['x'], s['y'], s['z'],
                               s.get('length', 0), s.get('width', 0),
                               s.get('height', 0), s['yaw']], dtype=np.float32), s['label'])
                     for (ts, s) in seq if abs(ts - t) < 1e-3),
                    None)
    
                if state is not None:
                    frame_traj[oid] = {
                        'category': state[1],    
                        'past': past,
                        'current_state': state[0],
                        'future': future
                    }
                    
    
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
            # print(f"timestamp : {t} timestamp_frame: {t_frame}")
            print(f"--------- timestamp : {t_frame}")
            return self.loaded_frames[t_frame]

    def __iter__(self):
        pass
