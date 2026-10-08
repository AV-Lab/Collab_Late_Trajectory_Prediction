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

from collections import namedtuple
position = namedtuple('Position', ['x', 'y', 'z','yaw'])

class TrajDataloader:
    """
    This class holds basic parameters (pickle path, scenario name, sensors filter) and
    defers preloading of sensor data into memory to a separate method 'preload_data()'.
    Once preloaded, it stores loaded sensor data (images, LiDAR, calibration, labels, ego_state)
    keyed by timestamp, and computes object trajectories across frames.
    
    After preloading, get_frame_data(t) returns all loaded sensor data plus trajectories for timestamp t.
    """
    def __init__(self, pickle_path, sensors, fps, load_lidar):
        self.data_file = pickle_path
        self.sensors = sensors
        self.vehicle_fps = fps
        self.load_lidar = load_lidar

    @staticmethod
    def _load_pcd_xyz(path):
        fields = None
        sizes = None
        types = None
        counts = None
        points = None

        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"PCD file has no DATA header: {path}")

                parts = line.decode("ascii").strip().split()
                if not parts or parts[0].startswith("#"):
                    continue

                key = parts[0].upper()
                if key == "FIELDS":
                    fields = parts[1:]
                elif key == "SIZE":
                    sizes = [int(value) for value in parts[1:]]
                elif key == "TYPE":
                    types = parts[1:]
                elif key == "COUNT":
                    counts = [int(value) for value in parts[1:]]
                elif key == "POINTS":
                    points = int(parts[1])
                elif key == "DATA":
                    data_format = parts[1].lower()
                    break

            if fields is None:
                raise ValueError(f"PCD file has no FIELDS header: {path}")

            xyz_indices = [fields.index(axis) for axis in ("x", "y", "z")]
            if data_format == "ascii":
                return np.loadtxt(
                    f,
                    usecols=xyz_indices,
                    dtype=np.float32,
                    ndmin=2,
                )

            if data_format != "binary":
                raise ValueError(f"Unsupported PCD DATA format '{data_format}': {path}")

            if sizes is None or types is None:
                raise ValueError(f"Binary PCD file has incomplete type metadata: {path}")
            if counts is None:
                counts = [1] * len(fields)

            type_map = {
                ("F", 4): "<f4", ("F", 8): "<f8",
                ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
                ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
            }
            dtype_fields = []
            for field, size, type_, count in zip(fields, sizes, types, counts):
                dtype = type_map[(type_.upper(), size)]
                dtype_fields.append((field, dtype) if count == 1 else (field, dtype, (count,)))

            data = np.fromfile(f, dtype=np.dtype(dtype_fields), count=points or -1)
            return np.column_stack([data[axis] for axis in ("x", "y", "z")]).astype(
                np.float32,
                copy=False,
            )
        
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
            return False
         
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
            lidar_path = frame_info["lidar"]
            if not self.load_lidar:
                loaded['lidar'] = None
            elif lidar_path and os.path.isfile(lidar_path):
            
                if lidar_path.endswith(".npy") or lidar_path.endswith(".npz"):
                    loaded['lidar'] = np.load(lidar_path)
                    if 'data' in loaded['lidar'].files:
                        loaded['lidar'] = loaded['lidar']['data'][:, :3]
            
                elif lidar_path.endswith(".pcd"):
                    loaded['lidar'] = self._load_pcd_xyz(lidar_path)
                    
                elif lidar_path.endswith(".bin"):
                    arr = np.fromfile(str(lidar_path), dtype=np.float32)
                    if arr.size % 4 != 0:
                        arr = arr[: arr.size - (arr.size % 4)]
                    pts = arr.reshape(-1, 4)
                    loaded['lidar'] = pts[:, :3]
                else:
                    print(f"[WARN] Unsupported lidar format: {lidar_path}")
                    loaded['lidar'] = None
            else:
                loaded['lidar'] = None
    
            loaded['labels'] = frame_info["labels"]
            loaded['ego_state'] = frame_info["ego_state"]
            loaded['calibration'] = frame_info["calibration"]
    
            self.loaded_frames[t] = loaded
    
        self.timestamps = sorted(self.loaded_frames.keys())
        self.trajectories = self._compute_trajectories()
        
        return True

    def _compute_trajectories(self):
        """
        Build per-object trajectories in **world frame** and store them into
        self.loaded_frames[t]['trajectories'].
    
            past   : consecutive list[position]  (ts <= t)
            current_state : np.ndarray[8]  (x,y,z,l,w,h,yaw,occlusion)
            future : consecutive list[position]  (ts > t)
        """
        objects_by_frame = []
        for t in self.timestamps:
            labels = self.loaded_frames[t]['labels']
            self.loaded_frames[t]['labels_world'] = labels
            objects_by_frame.append({det['obj_id']: det for det in labels})

        for frame_index, t in enumerate(self.timestamps):
            frame_traj = {}
            current_objects = objects_by_frame[frame_index]

            for oid, current in current_objects.items():
                past = []
                for index in range(frame_index, -1, -1):
                    obj = objects_by_frame[index].get(oid)
                    if obj is None:
                        break
                    past.append(position(obj['x'], obj['y'], obj['z'], obj['yaw']))
                past.reverse()

                future = []
                for index in range(frame_index + 1, len(self.timestamps)):
                    obj = objects_by_frame[index].get(oid)
                    if obj is None:
                        break
                    future.append(position(obj['x'], obj['y'], obj['z'], obj['yaw']))

                if not future:
                    continue

                frame_traj[oid] = {
                    'category': current['label'],
                    'past': past,
                    'current_state': np.array([
                        current['x'], current['y'], current['z'],
                        current['length'], current['width'], current['height'],
                        current['yaw'], current['occ_l1'],
                    ], dtype=np.float32),
                    'future': future,
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
