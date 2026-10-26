#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:30:27 2025

@author: nadya
"""


import os
import operator
import pickle
import cv2
import numpy as np

from collections import namedtuple
from collections.abc import Sequence
position = namedtuple('Position', ['x', 'y', 'z','yaw'])


class _TrajectoryView(Sequence):
    """Read-only window into one continuous, immutable GT trajectory."""

    __slots__ = ("_points", "_start", "_stop")

    def __init__(self, points, start, stop):
        self._points, self._start, self._stop = points, start, stop

    def __len__(self):
        return self._stop - self._start

    def __iter__(self):
        return (self._points[index] for index in range(self._start, self._stop))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._points[self._start + index]
                    for index in range(*key.indices(len(self)))]
        index = operator.index(key)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("trajectory index out of range")
        return self._points[self._start + index]


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
    def load_lidar_points(path):
        """Read one LiDAR frame as XYZ, without retaining an open archive."""
        if not path or not os.path.isfile(path):
            return None
        suffix = os.path.splitext(os.fspath(path))[1].lower()
        if suffix == ".npy":
            points = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "data" not in archive.files:
                    raise ValueError(f"LiDAR archive has no 'data' array: {path}")
                points = archive["data"]
        elif suffix == ".pcd":
            points = TrajDataloader._load_pcd_xyz(path)
        elif suffix == ".bin":
            data = np.fromfile(path, dtype=np.float32)
            points = data[:data.size - data.size % 4].reshape(-1, 4)
        else:
            print(f"[WARN] Unsupported lidar format: {path}")
            return None
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"LiDAR points must have shape (N, >=3): {path}")
        return points[:, :3]

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
            loaded['lidar_path'] = lidar_path
            loaded['lidar'] = self.load_lidar_points(lidar_path) if self.load_lidar else None
    
            loaded['labels'] = frame_info["labels"]
            loaded['ego_state'] = frame_info["ego_state"]
            loaded['calibration'] = frame_info["calibration"]
    
            self.loaded_frames[t] = loaded
    
        self.timestamps = sorted(self.loaded_frames.keys())
        self.trajectories = self._compute_trajectories()
        
        return True

    def _compute_trajectories(self):
        """Share full world-frame segments across per-frame past/future views.

        An absence in a selected frame ends a segment. Past includes current;
        future starts at the next frame. Appearances without future are omitted.
        """
        frames, segments, latest = [], [], {}
        for frame_index, timestamp in enumerate(self.timestamps):
            frame = self.loaded_frames[timestamp]
            labels = frame['labels']
            frame['labels_world'] = labels
            objects = {obj['obj_id']: obj for obj in labels}
            current = []
            for oid, obj in objects.items():
                previous_index, segment_index = latest.get(oid, (-2, None))
                if previous_index != frame_index - 1:
                    segment_index = len(segments)
                    segments.append([])
                points = segments[segment_index]
                current.append((oid, obj, segment_index, len(points)))
                points.append(position(obj['x'], obj['y'], obj['z'], obj['yaw']))
                latest[oid] = (frame_index, segment_index)
            frames.append(current)
        segments = [tuple(points) for points in segments]

        for timestamp, objects in zip(self.timestamps, frames):
            trajectories = {}
            for oid, current, segment_index, index in objects:
                points = segments[segment_index]
                if index + 1 == len(points):
                    continue
                trajectories[oid] = {
                    'category': current['label'],
                    'past': _TrajectoryView(points, 0, index + 1),
                    'current_state': np.array([
                        current['x'], current['y'], current['z'],
                        current['length'], current['width'], current['height'],
                        current['yaw'], current['occ_l1'],
                    ], dtype=np.float32),
                    'future': _TrajectoryView(points, index + 1, len(points)),
                }
            self.loaded_frames[timestamp]['trajectories'] = trajectories

        

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
