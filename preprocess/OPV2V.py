#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPV2V → unified pickle (DeepAccident-compatible) + optional occlusion visualization.

Structure mirrors DeepAccident's preprocessor, but yaw is converted from
degrees to radians so that downstream code (BBoxVisualizer, occlusions)
sees exactly the same convention as in DeepAccident.
"""

from __future__ import annotations
import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
import open3d as o3d

from .math_helper import pose_to_T, inv_T, cosd, sind   
from .occlusions import compute_l1_occlusion_for_frame


class Constants:
    CAMERA_SENSORS = ["camera0", "camera1", "camera2", "camera3"]
    LIDAR_SENSOR   = "lidar01"

    FPS  = 10
    STEP = 1.0 / FPS

    # Occlusion settings – now fully matching DeepAccident:
    OCCLUSION_RAYS_K        = 31
    OCCLUSION_EPS           = 1e-2
    OCCLUSION_VERTICAL_CHECK = True
    YAW_IN_DEGREES          = False  # we convert to radians in preprocessing

    KEEP_RADIUS_M = 50.0


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


def _is_rotation(R: np.ndarray) -> bool:
    if R.shape != (3, 3):
        return False
    return np.allclose(R.T @ R, np.eye(3), atol=1e-3) and 0.9 < np.linalg.det(R) < 1.1


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
      - per-camera intrinsics + extrinsics (cam<->ego)
    """
    vx, vy, vz, vroll, vyaw_deg, vpitch = frame_yaml["true_ego_pos"]
    T_vw = pose_to_T(vx, vy, vz, vroll, vyaw_deg, vpitch, degrees=True)  # ego->world
    T_wv = inv_T(T_vw)                                                  # world->ego

    cams = {}
    for cam in Constants.CAMERA_SENSORS:
        if cam not in frame_yaml:
            continue
        K = _as_matrix(frame_yaml[cam]["intrinsic"], (3, 3))
        T_c2v = _as_matrix(frame_yaml[cam]["extrinsic"], (4, 4))
        # Dataset sometimes stores T_veh->cam; ensure it's cam->veh
        if not _is_rotation(T_c2v[:3, :3]):
            T_c2v = inv_T(T_c2v)
        cams[cam] = {
            "K": K.tolist(),
            "camera_to_ego": T_c2v.tolist(),
            "ego_to_camera": inv_T(T_c2v).tolist(),
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


def _lidar_xyz_from_pcd(pcd_path: Path) -> Optional[np.ndarray]:
    """Load LiDAR XYZ from a .pcd file (for debugging visualisation)."""
    if not pcd_path.is_file(): return None
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0: return None
    return pts[:, :3]


# ───────────────────────────────────────────────────────── main preprocessing ───────────────────────────────────────────────────────── #

def preprocess_dataset(dataset_root: str, prefix: str) -> Path:
    """
    Process a single split (train/valid/test) into <split>_data.pkl.

    - Yaw is converted from degrees → radians (DeepAccident convention).
    - Objects are filtered by radius 50 m around ego.
    - Occlusion scores are computed using the same code/flags as DeepAccident.
    - If `visualize=True`, we:
         * render 2D BEV occlusion plots (VisOcclusionScores),
    """
    split_dir = Path(dataset_root).expanduser().resolve() / prefix
    if not split_dir.exists():
        print(f"[Info] Split '{prefix}' not found at {split_dir}, skipping.")

    scenarios: Dict[str, Dict[str, Dict[float, dict]]] = {}
    agent_name_set = set()

    for scene_path in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scenario_name = scene_path.name
        scenarios.setdefault(scenario_name, {})
        veh_dirs = sorted([p for p in scene_path.iterdir() if p.is_dir()])

        # Align timestamps across vehicles: find global min numeric frame index
        all_yaml = [y for vdir in veh_dirs for y in vdir.glob("*.yaml")]
        numeric_stems: List[int] = []
        for pth in all_yaml:
            try:
                numeric_stems.append(int(pth.stem))
            except Exception:
                pass
        min_idx_scenario = min(numeric_stems) if numeric_stems else 0

        for v_idx, vdir in enumerate(veh_dirs):
            agent = f"vehicle_{v_idx}"
            agent_name_set.add(agent)
            scenarios[scenario_name].setdefault(agent, {})

            yaml_files = sorted(
                vdir.glob("*.yaml"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
            )
            if not yaml_files:
                print(f"[Warn] No YAML frames in {vdir}")
                continue

            pad_len = len(yaml_files[0].stem)

            for f_idx, ypath in enumerate(yaml_files):
                try:
                    frame_idx = int(ypath.stem)
                except Exception:
                    frame_idx = min_idx_scenario + f_idx

                timestamp = f_idx * Constants.STEP
                frame_yaml = _read_yaml(ypath)

                # image + lidar paths
                images = {}
                for cam in Constants.CAMERA_SENSORS:
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
                occ_map, debug = compute_l1_occlusion_for_frame(
                    labels,
                    ego_state,
                    K=Constants.OCCLUSION_RAYS_K,
                    eps=Constants.OCCLUSION_EPS,
                    vertical_check=Constants.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Constants.YAW_IN_DEGREES,   # False – yaw already in rad
                    return_debug=True,
                )
                for o in labels:
                    o["occ_l1"] = float(occ_map.get(o["obj_id"], 0.0))

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

    scene_vehicle_counts = {scene_name: len(agent_dict) for scene_name, agent_dict in scenarios.items()}
    max_vehicles = max(scene_vehicle_counts.values()) if scene_vehicle_counts else 0

    # duration per scenario based on sensor observation length, duration = number of frames * STEP
    scene_durations = {}
    for scene_name, agent_dict in scenarios.items():
        max_num_frames = 0

        for _, frame_dict in agent_dict.items():
            num_frames = len(frame_dict)
            max_num_frames = max(max_num_frames, num_frames)

        scene_durations[scene_name] = max_num_frames * Constants.STEP

    meta_path = split_dir / "meta.txt"
    with open(meta_path, "w") as mf:
        mf.write("dataset=OPV2V\n")
        mf.write(f"split={prefix}\n")
        mf.write(f"fps={Constants.FPS}\n")
        mf.write(f"global=True\n")
        mf.write(f"max_vehicles={max_vehicles}\n")
        mf.write("\nscenario_name,num_vehicles,duration\n")

        for scene_name, nveh in sorted(scene_vehicle_counts.items()):
            duration = scene_durations.get(scene_name, 0.0)
            mf.write(f"{scene_name},{nveh},{duration:.2f}\n")
            
    data = {"scenarios": scenarios}
    output_pickle_path = split_dir / f"{prefix}_data.pkl"
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset to {output_pickle_path}")
    print(f"[META] {meta_path}")


# ───────────────────────────────────────────────────────── CLI ───────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPV2V → unified pickle with optional occlusion visualization"
    )
    parser.add_argument("dataset_root", type=str, help="Path containing train/valid/test")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Splits to process")
    args = parser.parse_args()

    root = args.dataset_root
    for split in args.splits:
        preprocess_dataset(root, prefix=split)
