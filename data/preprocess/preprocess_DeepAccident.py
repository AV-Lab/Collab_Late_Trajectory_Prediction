#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: preprocess_dataset_with_L1_occlusion.py
Created on Sun Mar 17 15:06:36 2024
Author: nadya (updated with L1 BEV angular occlusion)

Description:
Preprocess DeepAccident into the common waymo_v1 world convention.
Positions and headings use the reflected world frame; velocities recover the
released label convention. LiDAR files and sensor-local axes stay native.
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
  - 'scenarios': nested dict
      { scenario_name: { vehicle_name: { timestamp: { 'images': ..., 'lidar': ..., 'labels': ...,
                                                      'ego_state': ..., 'calibration': ... } } } }
"""

import os
import pickle
import math
from pathlib import Path
from typing import List, Optional
import numpy as np
import argparse

from .constants import Constants
from .math_helper import wrap_yaw
from .occlusions import compute_l1_occlusion_for_frame


WORLD_Y_REFLECTION = np.diag([1.0, -1.0, 1.0, 1.0])


DEEPACCIDENT_CATEGORY_MAPPING = {
    "vehicle": "vehicle",
    "car": "vehicle",
    "Car": "vehicle",
    "van": "vehicle",
    "Van": "vehicle",
    "truck": "vehicle",
    "Truck": "vehicle",
    "bus": "vehicle",
    "Bus": "vehicle",
    "pedestrian": "pedestrian",
    "Pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "Cyclist": "cyclist",
    "motorcycle": "motorcyclist",
    "motorcyclist": "motorcyclist",
    "Motorcyclist": "motorcyclist",
}


def _canonicalize_label(raw_label: str) -> str:
    try:
        return DEEPACCIDENT_CATEGORY_MAPPING[raw_label]
    except KeyError as error:
        raise ValueError(
            f"Unsupported DeepAccident category: {raw_label!r}."
        ) from error


def _load_lidar_xyz(npz_path: str) -> Optional[np.ndarray]:
    """Return Nx3 XYZ from a LiDAR .npz; tries common keys, falls back to arr_0."""
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        for k in ('points', 'xyz', 'lidar', 'data', 'arr_0'):
            if k in data:
                arr = np.asarray(data[k])
                break
        else:
            key0 = list(data.keys())[0]
            arr = np.asarray(data[key0])
        arr = np.reshape(arr, (-1, arr.shape[-1]))
        if arr.shape[-1] >= 3:
            return arr[:, :3].astype(np.float64)
    except Exception:
        pass
    return None


def _filter_objects_by_radius(
    objs: List[dict], ego_state: dict, keep_radius: float
) -> List[dict]:
    """Keep objects within keep_radius of the ego."""
    if not objs:
        return []
    ex = float(ego_state['x'])
    ey = float(ego_state['y'])

    kept = []
    for o in objs:
        dx = o['x'] - ex
        dy = o['y'] - ey
        if math.hypot(dx, dy) <= keep_radius:
            kept.append(o)
    return kept


def parse_label_file(label_path: str):
    """Read raw object states, ego state, and the header's world ego velocity."""
    objects = []
    ego_state = None

    with open(label_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].split() if lines else []
        if len(header) < 2:
            raise ValueError(f"Missing ego world velocity in {label_path}")
        ego_velocity = (float(header[0]), float(header[1]))
        if not np.isfinite(ego_velocity).all():
            raise ValueError(f"Nonfinite ego world velocity in {label_path}")
        for i, line in enumerate(lines[1:]):  # skip header
            parts = line.strip().split()
            if not parts:
                continue
            try:
                original_label = parts[0]
                obj = {
                    'label': _canonicalize_label(original_label),
                    'original_label': original_label,
                    'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3]),
                    'length': float(parts[4]), 'width': float(parts[5]), 'height': float(parts[6]),
                    'yaw': float(parts[7]),
                    'vel_x': float(parts[8]), 'vel_y': float(parts[9]),
                    'obj_id': int(parts[10])
                }
            except Exception as e:
                raise ValueError(f"Malformed label line {i+2} in {label_path}: '{line}'. Error: {e}")

            if obj['obj_id'] == -100:
                ego_state = obj
                continue

            if 0 <= obj['obj_id'] < 20000:
                objects.append(obj)

    return objects, ego_state, ego_velocity


def parse_calibration_file(calib_path: str):
    with open(calib_path, 'rb') as f:
        source = pickle.load(f)

    T_ego_world = np.asarray(source["ego_to_world"], dtype=np.float64)
    T_lidar_ego = np.asarray(source["lidar_to_ego"], dtype=np.float64)
    if any(matrix.shape != (4, 4) or not np.isfinite(matrix).all()
           for matrix in (T_ego_world, T_lidar_ego)):
        raise ValueError(f"Invalid ego/LiDAR calibration in {calib_path}")
    if abs(np.linalg.det((T_ego_world @ T_lidar_ego)[:2, :2])) < 1e-8:
        raise ValueError(f"Singular LiDAR/world XY projection in {calib_path}")
    T_ego_world = WORLD_Y_REFLECTION @ T_ego_world

    cameras = {}
    for camera in Constants.DEEPACCIDENT_CAMERA_SENSORS:
        K = np.asarray(source[f"intrinsic_{camera}"], dtype=np.float64)
        T_lidar_camera = np.asarray(source[f"lidar_to_{camera}"], dtype=np.float64)
        cameras[camera] = {
            "K": K.tolist(),
            "lidar_to_camera": T_lidar_camera.tolist(),
            "camera_to_lidar": np.linalg.inv(T_lidar_camera).tolist(),
        }

    return {
        "ego_to_world": T_ego_world.tolist(),
        "world_to_ego": np.linalg.inv(T_ego_world).tolist(),
        "lidar_to_ego": T_lidar_ego.tolist(),
        "ego_to_lidar": np.linalg.inv(T_lidar_ego).tolist(),
        "cameras": cameras,
    }


def _state_to_world(state: dict, calibration: dict, *, ego_velocity=None) -> dict:
    """Convert raw labels directly to TrajZoo's common-coordinate states."""
    T_lidar_world = (
        np.asarray(calibration["ego_to_world"], dtype=np.float64)
        @ np.asarray(calibration["lidar_to_ego"], dtype=np.float64)
    )
    R_lidar_world = T_lidar_world[:3, :3]

    position_lidar = np.array(
        [state["x"], state["y"], state["z"], 1.0],
        dtype=np.float64,
    )
    position_world = T_lidar_world @ position_lidar

    if ego_velocity is None:
        c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
        forward = R_lidar_world @ np.array([c, s, 0.0])
        yaw_world = wrap_yaw(math.atan2(forward[1], forward[0]))
        # Released object velocities are world velocities rotated by raw local
        # box yaw. Undo that rotation, then reflect world Y (F @ R(-yaw) @ v).
        vx = c * state["vel_x"] + s * state["vel_y"]
        vy = s * state["vel_x"] - c * state["vel_y"]
    else:
        # Preserve TrajZoo's ego pose heading; header velocity is already world.
        yaw_world = wrap_yaw(state["yaw"] + math.atan2(R_lidar_world[1, 0], R_lidar_world[0, 0]))
        vx, vy = ego_velocity[0], -ego_velocity[1]

    world_state = state.copy()
    world_state["x"] = float(position_world[0])
    world_state["y"] = float(position_world[1])
    world_state["z"] = float(position_world[2])
    world_state["yaw"] = float(yaw_world)
    world_state["vel_x"] = float(vx)
    world_state["vel_y"] = float(vy)
    return world_state


def preprocess_dataset(dataset_dir: str, prefix: str = "train") -> Path:
    dataset_dir = os.path.join(dataset_dir, prefix)
    output_dir = (
        Path(__file__).resolve().parent
        / "preprocessed"
        / "DeepAccident"
        / prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    step = Constants.DEEPACCIDENT_STEP

    dirs = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    sub_dirs = []
    for d in dirs:
        d_path = os.path.join(dataset_dir, d)
        for subf in sorted(os.listdir(d_path)):
            sub_dirs.append(f"{d}/{subf}")

    scenarios = {}

    for agent in Constants.DEEPACCIDENT_AGENTS:
        for s in sub_dirs:
            label_data_path = os.path.join(dataset_dir, s, agent, "label")
            if not os.path.isdir(label_data_path):
                continue

            scenario_folders = sorted(os.listdir(label_data_path))
            for scenario_folder in scenario_folders:
                folder_path = os.path.join(label_data_path, scenario_folder)
                if not os.path.isdir(folder_path):
                    continue

                label_files = sorted(
                    Path(folder_path).glob("*.txt"),
                    key=lambda path: int(path.stem.rsplit("_", 1)[1]),
                )
                if not label_files:
                    continue
                first_frame_number = int(label_files[0].stem.rsplit("_", 1)[1])
                scenario_name = f"{s}/{scenario_folder}"
                scenarios.setdefault(scenario_name, {})
                scenarios[scenario_name].setdefault(agent, {})

                for label_file in label_files:
                    frame_str = label_file.stem.rsplit("_", 1)[1]
                    timestamp = (int(frame_str) - first_frame_number) / Constants.DEEPACCIDENT_FPS
                    scenarios[scenario_name][agent][timestamp] = {
                        "_frame_str": frame_str,
                    }

    passed_scenarios = 0
    failed_scenarios = 0

    for scenario_name, agents_dict in list(scenarios.items()):
        s_parts = scenario_name.split('/')
        s_part = f"{s_parts[0]}/{s_parts[1]}"
        scenario_folder = s_parts[2]
        frame_cache = {}
        validation_failure = None

        missing_agents = [
            agent
            for agent in Constants.DEEPACCIDENT_AGENTS
            if agent not in agents_dict
        ]
        if missing_agents:
            validation_failure = f"missing required agent: {missing_agents[0]}"
        elif len({tuple(int(frame["_frame_str"]) for frame in frames.values())
                  for frames in agents_dict.values()}) != 1:
            validation_failure = "agent frame sets are unsynchronized"

        if validation_failure is None:
            for agent, timestamps_dict in agents_dict.items():
                for timestamp, data_dict in timestamps_dict.items():
                    frame_str = data_dict["_frame_str"]

                    images = {
                        sensor: (
                            f"{dataset_dir}/{s_part}/{agent}/{sensor}/{scenario_folder}/"
                            f"{scenario_folder}_{frame_str}.jpg"
                        )
                        for sensor in Constants.DEEPACCIDENT_CAMERA_SENSORS
                    }
                    lidar_path = (
                        f"{dataset_dir}/{s_part}/{agent}/{Constants.DEEPACCIDENT_LIDAR_SENSOR}/"
                        f"{scenario_folder}/{scenario_folder}_{frame_str}.npz"
                    )
                    label_path = (
                        f"{dataset_dir}/{s_part}/{agent}/label/{scenario_folder}/"
                        f"{scenario_folder}_{frame_str}.txt"
                    )
                    calibration_path = (
                        f"{dataset_dir}/{s_part}/{agent}/calib/{scenario_folder}/"
                        f"{scenario_folder}_{frame_str}.pkl"
                    )

                    required_paths = [
                        *images.values(),
                        lidar_path,
                        label_path,
                        calibration_path,
                    ]
                    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
                    if missing_paths:
                        validation_failure = f"required file missing: {missing_paths[0]}"
                        break

                    if _load_lidar_xyz(lidar_path) is None:
                        validation_failure = f"invalid LiDAR file: {lidar_path}"
                        break

                    objs, ego_state, ego_velocity = parse_label_file(label_path)
                    if ego_state is None:
                        validation_failure = f"ego state missing: {label_path}"
                        break

                    frame_cache[(agent, timestamp)] = {
                        "images": images,
                        "lidar": lidar_path,
                        "labels": objs,
                        "ego_state": ego_state,
                        "ego_velocity": ego_velocity,
                        "calibration": calibration_path,
                    }

                if validation_failure is not None:
                    break

        if validation_failure is not None:
            del scenarios[scenario_name]
            failed_scenarios += 1
            print(
                f"[SKIP] Scenario {prefix}/{scenario_name}: "
                f"{validation_failure}"
            )
            continue

        for agent, timestamps_dict in agents_dict.items():
            dropped_total = 0
            kept_total = 0

            for timestamp, data_dict in timestamps_dict.items():
                frame = frame_cache[(agent, timestamp)]
                data_dict.pop("_frame_str")
                data_dict['images'] = frame['images']
                data_dict['lidar'] = frame['lidar']
                calibration = parse_calibration_file(frame['calibration'])
                objs = [
                    _state_to_world(obj, calibration)
                    for obj in frame['labels']
                ]
                ego_st = _state_to_world(frame['ego_state'], calibration,
                                         ego_velocity=frame['ego_velocity'])

                before = len(objs)
                objs = _filter_objects_by_radius(
                    objs,
                    ego_st,
                    Constants.KEEP_RADIUS_M,
                )
                after = len(objs)
                kept_total += after
                dropped_total += (before - after)

                occ_map, _ = compute_l1_occlusion_for_frame(
                    objs, ego_st,
                    K=Constants.OCCLUSION_RAYS_K,
                    eps=Constants.OCCLUSION_EPS,
                    vertical_check=Constants.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Constants.YAW_IN_DEGREES,
                    return_debug=False,
                )
                for o in objs:
                    o['occ_l1'] = float(occ_map[o['obj_id']])

                data_dict['labels'] = objs
                data_dict['ego_state'] = ego_st
                data_dict['calibration'] = calibration

            print(f"{scenario_name} :: {agent} processed "
                  f"(kept {kept_total}, dropped {dropped_total})")

        passed_scenarios += 1

    print(
        f"[SUMMARY] DeepAccident {prefix}: "
        f"{passed_scenarios} scenarios passed, {failed_scenarios} scenarios failed"
    )

    scene_vehicle_counts = {scene_name: len(agents_dict) for scene_name, agents_dict in scenarios.items()}
    max_vehicles = max(scene_vehicle_counts.values()) if scene_vehicle_counts else 0
    
    scene_durations = {
        scene_name: max(max(frames) + step for frames in agents.values())
        for scene_name, agents in scenarios.items()
    }

    meta_path = output_dir / "meta.txt"
    with open(meta_path, "w") as mf:
        mf.write("dataset=DeepAccident\n")
        mf.write(f"split={prefix}\n")
        mf.write(f"fps={Constants.DEEPACCIDENT_FPS}\n")
        mf.write(f"global=True\n")
        mf.write(f"max_vehicles={max_vehicles}\n")
        mf.write("\nscenario_name,num_vehicles,duration\n")
        
        for scene_name, nveh in sorted(scene_vehicle_counts.items()):
            duration = scene_durations[scene_name]
            mf.write(f"{scene_name},{nveh},{duration:.2f}\n")

    data = {
        "metadata": {
            "coordinate_convention": "waymo_v1",
            "coordinate_frame": "world",
            "coordinate_units": "meters",
            "sample_fps": float(Constants.DEEPACCIDENT_FPS),
            "common_coordinates": True,
            "coordinate_source_profile": "deepaccident",
            "coordinate_transform_version": 1,
            "preprocessing_version": 2,
        },
        "scenarios": scenarios,
    }
    output_pickle_path = output_dir / f"{prefix}_data.pkl"
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    for path in (output_dir / "tmp").glob("vehicle_*.pkl"):
        path.unlink()
    print(f"Saved dataset to {output_pickle_path}")
    print(f"[META] {meta_path}")
    return output_pickle_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepAccident → unified pickle with optional occlusion visualization"
    )
    parser.add_argument("dataset_root", type=str, help="Path containing train/valid/test")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], help="Splits to process")
    args = parser.parse_args()

    root = args.dataset_root
    for split in args.splits:
        preprocess_dataset(root, prefix=split)
