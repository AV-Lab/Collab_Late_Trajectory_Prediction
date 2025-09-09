#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: preprocess_dataset_with_L1_occlusion.py
Created on Sun Mar 17 15:06:36 2024
Author: nadya (updated with L1 BEV angular occlusion)

Description:
Preprocess the DeepAccident dataset and store it in a unified format as a pickle file.
Additionally, compute an L1 (Fast BEV Angular Occlusion, 2D) score per object for each frame
from the perspective of the current agent (ego sensor origin).

The dataset is organized in a hierarchical folder structure where data for each vehicle
(e.g., "ego_vehicle", "other_vehicle", etc.) is stored under separate subdirectories
for different sensors (e.g., Camera and LiDAR). For each scenario, the data is further divided
by frame timestamps (computed from a fixed FPS). For each frame, we store:
  - 'images': dict camera_name -> image path
  - 'lidar': LiDAR path (.npz)
  - 'labels': list of object dicts (with added 'occ_l1' occlusion field)
  - 'ego_state': ego vehicle state (id -100) from the label file
  - 'calibration': processed calibration data

Output: a pickle with keys:
  - 'meta': dataset metadata
  - 'scenarios': nested dict
      { scenario_name: { vehicle_name: { timestamp: { 'images': ..., 'lidar': ..., 'labels': ...,
                                                      'ego_state': ..., 'calibration': ... } } } }
"""

import os
import pickle
import copy
import math
from typing import List, Tuple, Dict, Optional
import numpy as np


class Constants:
    # Sensors
    CAMERA_SENSORS = [
        'Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight',
        'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight'
    ]
    LIDAR_SENSOR = 'lidar01'

    # Agents (per DeepAccident folder naming)
    AGENTS = ['ego_vehicle', 'ego_vehicle_behind', 'other_vehicle', 'other_vehicle_behind']

    # Framerate
    FPS = 10

    # --- L1 BEV Angular Occlusion parameters ---
    OCCLUSION_RAYS_K = 31        # number of rays sampled across the target's angular span
    OCCLUSION_EPS = 1e-2         # numerical margin for comparing occluder vs target distances
    OCCLUSION_VERTICAL_CHECK = True  # enable height-based vertical clearance check
    YAW_IN_DEGREES = False       # set True if label yaw is in degrees


# --------------------------
# Label / Calibration Parsers
# --------------------------

def parse_label_file(label_path: str):
    """Parse a label file into dynamic objects and ego state.

    Expected columns per non-header line:
      label, x, y, z, length, width, height, yaw, vel_x, vel_y, obj_id

    Returns
    -------
    objects : List[dict]
        Dynamic objects with 3D box + meta. Static obstacles (id<0 or id>=20000) removed.
    ego_state : Optional[dict]
        Dict for ego (-100) if present.
    """
    objects = []
    ego_state = None

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[1:]):  # skip header
            parts = line.strip().split()
            if not parts:
                continue
            try:
                obj = {
                    'label': parts[0],
                    'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3]),
                    'length': float(parts[4]), 'width': float(parts[5]), 'height': float(parts[6]),
                    'yaw': float(parts[7]),
                    'vel_x': float(parts[8]), 'vel_y': float(parts[9]),
                    'obj_id': int(parts[10])
                }
            except Exception as e:
                raise ValueError(f"Malformed label line {i+2} in {label_path}: '{line}'. Error: {e}")

            # Ego vehicle
            if obj['obj_id'] == -100:
                ego_state = obj
                continue  # do not include ego in 'objects'

            # Keep only dynamic actor IDs in [0, 19999]
            if 0 <= obj['obj_id'] < 20000:
                objects.append(obj)
            # else: static/invalid -> skip

    return objects, ego_state


def parse_calibration_file(calib_path: str):
    with open(calib_path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------
# Geometry helpers for BEV occlusion (2D)
# ---------------------------------------

def _rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _box_corners_xy(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
    """Return 4 corners (x,y) of an oriented rectangle in BEV.
    Corners are ordered (front-left, front-right, back-right, back-left) in the local box frame before rotation.
    """
    # local half-dims
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw]
    ], dtype=np.float64)

    R = _rot2d(yaw)
    world = (local @ R.T) + np.array([cx, cy], dtype=np.float64)
    return world  # shape (4,2)


def _angles_from(origin: np.ndarray, pts: np.ndarray) -> np.ndarray:
    v = pts - origin[None, :]
    return np.arctan2(v[:, 1], v[:, 0])


def _min_covering_arc(angles: np.ndarray) -> Tuple[float, float]:
    """Smallest arc [L,R] on circle (unwrapped R>=L) that covers all input angles.
    Uses circular gap method. Returns L and R in a continuous angle domain where R>=L.
    """
    a = np.sort((angles + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi)
    # duplicate for circular wrap
    a2 = np.concatenate([a, a + 2 * np.pi])
    n = len(a)
    # window of size n covering all points -> minimal arc is 2π - max gap between consecutive sorted angles
    # equivalently, find i maximizing (a[i+1] - a[i]); arc start is a[i+1]
    gaps = a2[1:n] - a2[0:n - 1]
    max_gap_idx = int(np.argmax(gaps))
    L = a2[max_gap_idx + 1]
    R = L + (2 * np.pi - gaps[max_gap_idx])
    return L, R


def _ray_segment_intersection(s: np.ndarray, d: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Optional[float]:
    """Ray s + t d (t>=0) intersect segment p0->p1.
    Returns t (distance along ray direction) if intersects in front of the origin and within segment; else None.
    """
    v = p1 - p0
    M = np.array([[d[0], -v[0]], [d[1], -v[1]]], dtype=np.float64)
    b = p0 - s
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        return None
    inv = np.array([[ M[1, 1], -M[0, 1]], [-M[1, 0],  M[0, 0]]], dtype=np.float64) / det
    t, u = (inv @ b)
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return float(t)
    return None


def _ray_rect_distance(s: np.ndarray, theta: float, corners: np.ndarray) -> Optional[float]:
    """Distance along ray to the first intersection with the rectangle defined by corners (4x2)."""
    d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    dists = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        t = _ray_segment_intersection(s, d, p0, p1)
        if t is not None:
            dists.append(t)
    if not dists:
        return None
    return float(min(dists))


def _rect_angular_span(s: np.ndarray, corners: np.ndarray) -> Tuple[float, float]:
    angs = _angles_from(s, corners)
    L, R = _min_covering_arc(angs)
    return L, R


# -----------------------------
# L1 BEV Angular Occlusion Core
# -----------------------------

def compute_l1_occlusion_for_frame(objects: List[dict], ego_state: Optional[dict],
                                   K: int = Constants.OCCLUSION_RAYS_K,
                                   eps: float = Constants.OCCLUSION_EPS,
                                   vertical_check: bool = Constants.OCCLUSION_VERTICAL_CHECK,
                                   yaw_in_degrees: bool = Constants.YAW_IN_DEGREES) -> Dict[int, float]:
    """Compute O_L1 occlusion per object as seen from the current agent's ego origin.

    Parameters
    ----------
    objects : list of dict
        Each must contain x,y,length,width,height,yaw,obj_id.
    ego_state : dict or None
        If provided, its (x,y) is used as sensor origin; else (0,0) is used.
    K : int
        Number of rays per target across its BEV angular span.
    eps : float
        Numerical slack for occluder-vs-target distance comparison.
    vertical_check : bool
        If True, apply optional height check: skip occlusion if
        h_o < h_t * (d_o / d_t).
    yaw_in_degrees : bool
        If True, convert yaw to radians.

    Returns
    -------
    occ_map : dict obj_id -> O_L1 in [0,1]
    """
    if not objects:
        return {}

    s = np.array([
        (ego_state['x'] if ego_state is not None else 0.0),
        (ego_state['y'] if ego_state is not None else 0.0)
    ], dtype=np.float64)

    # Precompute rectangles (corners) and spans
    rects = {}
    spans = {}
    heights = {}

    for o in objects:
        yaw = math.radians(o['yaw']) if yaw_in_degrees else o['yaw']
        corners = _box_corners_xy(o['x'], o['y'], o['length'], o['width'], yaw)
        rects[o['obj_id']] = corners
        heights[o['obj_id']] = float(o['height'])
        L, R = _rect_angular_span(s, corners)
        spans[o['obj_id']] = (L, R)

    occ_map: Dict[int, float] = {}

    for tgt in objects:
        tid = tgt['obj_id']
        tL, tR = spans[tid]
        corners_t = rects[tid]
        h_t = heights[tid]

        if K <= 0:
            occ_map[tid] = 0.0
            continue

        # Sample K thetas uniformly on [tL, tR]
        thetas = np.linspace(tL, tR, K)
        occluded = 0
        valid_rays = 0

        # Precompute d_t per ray (distance to target)
        d_t_list: List[Optional[float]] = []
        for theta in thetas:
            dt = _ray_rect_distance(s, float(theta), corners_t)
            d_t_list.append(dt)

        for k, theta in enumerate(thetas):
            d_t = d_t_list[k]
            if d_t is None:
                # Ray didn't hit target (degenerate numeric case) -> ignore in stats
                continue
            valid_rays += 1

            # Test occluders
            ray_blocked = False
            for o in objects:
                oid = o['obj_id']
                if oid == tid:
                    continue

                oL, oR = spans[oid]

                # Quick angular rejection: if theta not within occluder's span, skip
                # Handle wrap by unwrapping theta into [oL,oR] domain
                # Since spans are unwrapped (R>=L), shift theta by multiples of 2π to land near span
                theta_unwrapped = float(theta)
                while theta_unwrapped < oL:
                    theta_unwrapped += 2 * math.pi
                while theta_unwrapped > oR:
                    theta_unwrapped -= 2 * math.pi
                if not (oL - 1e-9 <= theta_unwrapped <= oR + 1e-9):
                    continue

                d_o = _ray_rect_distance(s, float(theta), rects[oid])
                if d_o is None:
                    continue

                if d_o < d_t - eps:
                    if vertical_check:
                        h_o = heights[oid]
                        # If occluder too short, skip blocking
                        if h_o < h_t * (d_o / d_t):
                            continue
                    ray_blocked = True
                    break

            if ray_blocked:
                occluded += 1

        # If no valid rays, define 0 occlusion
        occ = (occluded / valid_rays) if valid_rays > 0 else 0.0
        occ_map[tid] = float(max(0.0, min(1.0, occ)))

    return occ_map


# --------------------------
# Main Preprocessing Pipeline
# --------------------------

def preprocess_dataset(dataset_dir: str, prefix: str = "train"):
    # Update dataset_dir to include the prefix folder.
    dataset_dir = os.path.join(dataset_dir, prefix)
    step = 1.0 / Constants.FPS

    # Gather all top-level subdirectories within the dataset directory.
    dirs = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    sub_dirs = []
    for d in dirs:
        d_path = os.path.join(dataset_dir, d)
        for subf in sorted(os.listdir(d_path)):
            sub_dirs.append(f"{d}/{subf}")

    # Create sensors list
    sensors = copy.deepcopy(Constants.CAMERA_SENSORS)
    sensors.append(Constants.LIDAR_SENSOR)

    # Initialize the data dictionary with metadata.
    data = {
        'meta': {
            'name': 'DeepAccident',
            'agents': Constants.AGENTS,
            'sensors': sensors,
            'fps': Constants.FPS,
            'occlusion': {
                'method': 'L1_BEV_Angular',
                'K': Constants.OCCLUSION_RAYS_K,
                'eps': Constants.OCCLUSION_EPS,
                'vertical_check': Constants.OCCLUSION_VERTICAL_CHECK,
                'yaw_in_degrees': Constants.YAW_IN_DEGREES,
            }
        },
        'scenarios': {}
    }
    scenarios = {}

    # Build index structure across agents/sensors
    for agent in Constants.AGENTS:
        for s in sub_dirs:
            sensor_base_path = os.path.join(dataset_dir, s, agent)
            if not os.path.isdir(sensor_base_path):
                continue
            for sensor in sensors:
                sensor_data_path = os.path.join(sensor_base_path, sensor)
                if not os.path.isdir(sensor_data_path):
                    continue
                scenario_folders = sorted(os.listdir(sensor_data_path))
                for scenario_folder in scenario_folders:
                    scenario_name = f"{s}/{scenario_folder}"
                    if scenario_name not in scenarios:
                        scenarios[scenario_name] = {}
                    if agent not in scenarios[scenario_name]:
                        scenarios[scenario_name][agent] = {}
                    folder_path = os.path.join(sensor_data_path, scenario_folder)
                    scene_files = sorted(os.listdir(folder_path))
                    for i, _ in enumerate(scene_files):
                        timestamp = i * step
                        if timestamp not in scenarios[scenario_name][agent]:
                            scenarios[scenario_name][agent][timestamp] = {
                                'images': {},
                                'lidar': "",
                                'labels': [],
                                'ego_state': None,
                                'calibration': None
                            }

    # Fill in the data and compute occlusion per frame
    for scenario_name, agents_dict in scenarios.items():
        s_parts = scenario_name.split('/')
        s_part = f"{s_parts[0]}/{s_parts[1]}"  # e.g., "Town01/weather_1"
        scenario_folder = s_parts[2]           # e.g., "scenarioA"
        for agent, timestamps_dict in agents_dict.items():
            for timestamp, data_dict in timestamps_dict.items():
                frame_index = int(timestamp / step) + 1
                frame_str = str(frame_index).zfill(3)  # NOTE: adjust if >999 frames

                # Build image paths for each camera sensor.
                for sensor in Constants.CAMERA_SENSORS:
                    img_path = (
                        f"{dataset_dir}/{s_part}/{agent}/{sensor}/{scenario_folder}/"
                        f"{scenario_folder}_{frame_str}.jpg"
                    )
                    data_dict['images'][sensor] = img_path

                # LiDAR path
                lidar_path = (
                    f"{dataset_dir}/{s_part}/{agent}/{Constants.LIDAR_SENSOR}/{scenario_folder}/"
                    f"{scenario_folder}_{frame_str}.npz"
                )
                data_dict['lidar'] = lidar_path

                # Labels / ego state
                lbl_path = (
                    f"{dataset_dir}/{s_part}/{agent}/label/{scenario_folder}/"
                    f"{scenario_folder}_{frame_str}.txt"
                )
                objs, ego_st = parse_label_file(lbl_path)

                # Compute occlusion (L1) from this agent's origin
                occ_map = compute_l1_occlusion_for_frame(
                    objs, ego_st,
                    K=Constants.OCCLUSION_RAYS_K,
                    eps=Constants.OCCLUSION_EPS,
                    vertical_check=Constants.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Constants.YAW_IN_DEGREES,
                )
                # attach to objects
                for o in objs:
                    o['occ_l1'] = float(occ_map.get(o['obj_id'], 0.0))

                data_dict['labels'] = objs
                data_dict['ego_state'] = ego_st

                # Calibration
                calib_path = (
                    f"{dataset_dir}/{s_part}/{agent}/calib/{scenario_folder}/"
                    f"{scenario_folder}_{frame_str}.pkl"
                )
                data_dict['calibration'] = parse_calibration_file(calib_path)

            print(f"{scenario_name} :: {agent} processed")

    data['scenarios'] = scenarios
    output_pickle_path = os.path.join(dataset_dir, f"{prefix}_data.pkl")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset to {output_pickle_path}")


if __name__ == '__main__':
    data_folder = '/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/DeepAccident'
    print(f"Preprocessing datasets at {data_folder}")
    preprocess_dataset(data_folder, prefix='train')
    preprocess_dataset(data_folder, prefix='valid')
