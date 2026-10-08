#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPV2V → unified pickle (DeepAccident-compatible).

Structure mirrors DeepAccident's preprocessor, but yaw is converted from
degrees to radians so that downstream code (BBoxVisualizer, occlusions)
sees exactly the same convention as in DeepAccident.
"""

from __future__ import annotations
import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .math_helper import pose_to_T, inv_T
from .occlusions import compute_l1_occlusion_for_frame
from .constants import Constants


# ───────────────────────────────────────────────────────── helpers ───────────────────────────────────────────────────────── #

def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def _read_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            pass
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.UnsafeLoader)
    return _to_builtin(data)


def _as_matrix(x, shape: Tuple[int, int]) -> np.ndarray:
    A = np.array(x, dtype=np.float64)
    if A.shape == shape:
        return A
    if A.size == shape[0] * shape[1]:
        return A.reshape(shape)
    raise ValueError(f"Cannot reshape array of shape {A.shape} to {shape}")


def _build_calibration(frame_yaml: dict) -> dict:
    """
    Build a calibration dict that mirrors the DeepAccident style:
      - 'ego_to_world', 'world_to_ego'
      - 'lidar_to_ego', 'ego_to_lidar'
      - per-camera intrinsics + extrinsics (lidar<->camera)
    """
    vx, vy, vz, vroll, vyaw_deg, vpitch = frame_yaml["true_ego_pos"]
    T_vw = pose_to_T(vx, vy, vz, vroll, vyaw_deg, vpitch, degrees=True)  # ego->world
    T_wv = inv_T(T_vw)                                                  # world->ego

    cams = {}
    for cam in Constants.OPV2V_CAMERA_SENSORS:
        K = _as_matrix(frame_yaml[cam]["intrinsic"], (3, 3))
        T_lidar_camera = _as_matrix(frame_yaml[cam]["extrinsic"], (4, 4))
        cams[cam] = {
            "K": K.tolist(),
            "lidar_to_camera": T_lidar_camera.tolist(),
            "camera_to_lidar": inv_T(T_lidar_camera).tolist(),
        }

    # LiDAR pose: either explicit pose in world, or extrinsic in YAML
    if "lidar_pose" in frame_yaml and frame_yaml["lidar_pose"] is not None:
        lx, ly, lz, lroll, lyaw_deg, lpitch = frame_yaml["lidar_pose"]
        T_lw = pose_to_T(lx, ly, lz, lroll, lyaw_deg, lpitch, degrees=True)  # lidar->world
        T_lv = T_wv @ T_lw                                                  # lidar->ego
    elif (
        "lidar" in frame_yaml
        and isinstance(frame_yaml["lidar"], dict)
        and "extrinsic" in frame_yaml["lidar"]
    ):
        T_lv = _as_matrix(frame_yaml["lidar"]["extrinsic"], (4, 4))         # lidar->ego
    else:
        print("[Warn] No LiDAR pose/extrinsic in YAML; using identity.")
        T_lv = np.eye(4, dtype=np.float64)

    calib = {
        "cameras": cams,
        "lidar_to_ego": T_lv.tolist(),
        "ego_to_lidar": inv_T(T_lv).tolist(),
        "ego_to_world": T_vw.tolist(),
        "world_to_ego": T_wv.tolist(),
    }
    return calib


def _build_ego_state(frame_yaml: dict) -> dict:
    """
    Ego state in world coordinates, yaw in **radians** (DeepAccident convention).
    """
    vx, vy, vz, vroll, vyaw_deg, vpitch = frame_yaml["true_ego_pos"]
    spd = float(frame_yaml.get("ego_speed", 0.0))

    yaw_rad = math.radians(float(vyaw_deg))
    return {
        "x": float(vx),
        "y": float(vy),
        "z": float(vz),
        "yaw": yaw_rad,                        # radians
        "vel_x": spd * math.cos(yaw_rad),
        "vel_y": spd * math.sin(yaw_rad),
        "obj_id": -100,
    }


def _build_labels(frame_yaml: dict) -> List[dict]:
    """
    Per-vehicle labels in **world frame**, yaw in radians and boxes using full
    dimensions (length/width/height), matching DeepAccident fields.
    """
    labels: List[dict] = []
    vehicles = frame_yaml.get("vehicles", {}) or {}
    for sid, v in vehicles.items():
        # CARLA stores half-dimensions in 'extent'
        L = 2.0 * float(v["extent"][0])
        W = 2.0 * float(v["extent"][1])
        H = 2.0 * float(v["extent"][2])

        x, y, z = map(float, v["location"])
        roll_deg, yaw_deg, pitch_deg = map(float, v["angle"])
        spd = float(v.get("speed", 0.0))

        yaw_rad = math.radians(yaw_deg)

        labels.append(
            {
                "label": "vehicle",
                "original_label": "vehicle",
                "x": x,
                "y": y,
                "z": z,
                "length": L,
                "width": W,
                "height": H,
                "yaw": yaw_rad,                     # radians
                "vel_x": spd * math.cos(yaw_rad),
                "vel_y": spd * math.sin(yaw_rad),
                "obj_id": int(sid),
            }
        )
    return labels


def _filter_by_radius(
    objs: List[dict], ego_state: dict, keep_radius_m: float
) -> List[dict]:
    """Keep objects whose (x,y) are within KEEP_RADIUS_M of the ego."""
    if not objs:
        return []
    ex, ey = float(ego_state["x"]), float(ego_state["y"])
    out = []
    for o in objs:
        if math.hypot(o["x"] - ex, o["y"] - ey) <= keep_radius_m + 1e-6:
            out.append(o)
    return out


# ───────────────────────────────────────────────────────── main preprocessing ───────────────────────────────────────────────────────── #

def preprocess_dataset(dataset_root: str, prefix: str) -> Path:
    """
    Process a single split (train/valid/test) into <split>_data.pkl.

    - Yaw is converted from degrees → radians (DeepAccident convention).
    - Objects are filtered by radius 50 m around ego.
    - Occlusion scores are computed using the same code/flags as DeepAccident.
    """
    split_dir = Path(dataset_root).expanduser().resolve() / prefix
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Requested split '{prefix}' was not found at {split_dir}"
        )

    output_dir = Path(__file__).resolve().parent / "preprocessed" / "OPV2V" / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios: Dict[str, Dict[str, Dict[float, dict]]] = {}
    passed_scenarios = 0
    failed_scenarios = 0

    for scene_path in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scenario_name = scene_path.name
        veh_dirs = sorted([p for p in scene_path.iterdir() if p.is_dir()])

        all_yaml = [y for vdir in veh_dirs for y in vdir.glob("*.yaml")]
        frame_yamls = {ypath: _read_yaml(ypath) for ypath in all_yaml}
        missing_ego_pose = [
            ypath
            for ypath, frame_yaml in frame_yamls.items()
            if "true_ego_pos" not in frame_yaml or frame_yaml["true_ego_pos"] is None
        ]
        missing_sensor_files = []
        for ypath in all_yaml:
            for cam in Constants.OPV2V_CAMERA_SENSORS:
                image_path = ypath.parent / f"{ypath.stem}_{cam}.png"
                if not image_path.is_file():
                    missing_sensor_files.append(image_path)
            lidar_path = ypath.parent / f"{ypath.stem}.pcd"
            if not lidar_path.is_file():
                missing_sensor_files.append(lidar_path)

        if not all_yaml or missing_ego_pose or missing_sensor_files:
            failed_scenarios += 1
            if missing_ego_pose:
                print(
                    f"[SKIP] Scenario {prefix}/{scenario_name}: "
                    f"missing true_ego_pos in {len(missing_ego_pose)} frame(s); "
                    f"first missing frame: {missing_ego_pose[0]}"
                )
            elif missing_sensor_files:
                print(
                    f"[SKIP] Scenario {prefix}/{scenario_name}: "
                    f"{len(missing_sensor_files)} required sensor file(s) missing; "
                    f"first missing file: {missing_sensor_files[0]}"
                )
            else:
                print(f"[SKIP] Scenario {prefix}/{scenario_name}: no YAML frames found")
            continue

        scenarios.setdefault(scenario_name, {})

        for v_idx, vdir in enumerate(veh_dirs):
            agent = f"vehicle_{v_idx}"
            scenarios[scenario_name].setdefault(agent, {})

            yaml_files = sorted(
                vdir.glob("*.yaml"),
                key=lambda p: int(p.stem),
            )
            if not yaml_files:
                print(f"[Warn] No YAML frames in {vdir}")
                continue

            pad_len = len(yaml_files[0].stem)

            for f_idx, ypath in enumerate(yaml_files):
                frame_idx = int(ypath.stem)

                timestamp = f_idx * Constants.OPV2V_STEP
                frame_yaml = frame_yamls[ypath]

                # image + lidar paths
                images = {}
                for cam in Constants.OPV2V_CAMERA_SENSORS:
                    img = vdir / f"{str(frame_idx).zfill(pad_len)}_{cam}.png"
                    images[cam] = str(img)
                lidar_path = vdir / f"{str(frame_idx).zfill(pad_len)}.pcd"
            

                # labels + ego
                labels_full = _build_labels(frame_yaml)   # world frame, yaw in rad
                ego_state   = _build_ego_state(frame_yaml)

                # filter by radius (50 m)
                labels = _filter_by_radius(
                    labels_full, ego_state, Constants.KEEP_RADIUS_M
                )

                # occlusion (DeepAccident-identical settings)
                occ_map, _ = compute_l1_occlusion_for_frame(
                    labels,
                    ego_state,
                    K=Constants.OCCLUSION_RAYS_K,
                    eps=Constants.OCCLUSION_EPS,
                    vertical_check=Constants.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Constants.YAW_IN_DEGREES,   # False – yaw already in rad
                    return_debug=False,
                )
                for o in labels:
                    o["occ_l1"] = float(occ_map[o["obj_id"]])

                # calibration
                calibration = _build_calibration(frame_yaml)

                scenarios[scenario_name][agent][timestamp] = {
                    "images": images,
                    "lidar": str(lidar_path),
                    "labels": labels,
                    "ego_state": ego_state,
                    "calibration": calibration,
                }


        print(f"[OK] Scenario processed: {prefix}/{scenario_name}")
        passed_scenarios += 1

    print(
        f"[SUMMARY] OPV2V {prefix}: "
        f"{passed_scenarios} scenarios passed, {failed_scenarios} scenarios failed"
    )

    scene_vehicle_counts = {scene_name: len(agent_dict) for scene_name, agent_dict in scenarios.items()}
    max_vehicles = max(scene_vehicle_counts.values()) if scene_vehicle_counts else 0

    # duration per scenario based on sensor observation length, duration = number of frames * STEP
    scene_durations = {}
    for scene_name, agent_dict in scenarios.items():
        max_num_frames = 0

        for _, frame_dict in agent_dict.items():
            num_frames = len(frame_dict)
            max_num_frames = max(max_num_frames, num_frames)

        scene_durations[scene_name] = max_num_frames * Constants.OPV2V_STEP

    meta_path = output_dir / "meta.txt"
    with open(meta_path, "w") as mf:
        mf.write("dataset=OPV2V\n")
        mf.write(f"split={prefix}\n")
        mf.write(f"fps={Constants.OPV2V_FPS}\n")
        mf.write(f"global=True\n")
        mf.write(f"max_vehicles={max_vehicles}\n")
        mf.write("\nscenario_name,num_vehicles,duration\n")

        for scene_name, nveh in sorted(scene_vehicle_counts.items()):
            duration = scene_durations[scene_name]
            mf.write(f"{scene_name},{nveh},{duration:.2f}\n")
            
    data = {"scenarios": scenarios}
    output_pickle_path = output_dir / f"{prefix}_data.pkl"
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset to {output_pickle_path}")
    print(f"[META] {meta_path}")
    return output_pickle_path


# ───────────────────────────────────────────────────────── CLI ───────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPV2V → unified pickle"
    )
    parser.add_argument("dataset_root", type=str, help="Path containing train/valid/test")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Splits to process")
    args = parser.parse_args()

    root = args.dataset_root
    for split in args.splits:
        preprocess_dataset(root, prefix=split)
