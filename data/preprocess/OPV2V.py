#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: preprocess_opv2v.py
Author: Nadya (with assistant)

Purpose
-------
Preprocess OPV2V-style scenarios into the SAME unified pickle format we use for
DeepAccident, so the exact same downstream loader can be reused.

⚠ FPS NOTE (fixed, not a parameter)
----------------------------------
OPV2V data is RECORDED at **10 Hz**. We therefore fix the step to 0.1 s and do
NOT expose FPS as a command-line flag. This keeps timestamps consistent across
all splits and avoids configuration drift.

Expected directory layout (per your screenshot)
-----------------------------------------------
<dataset_root>/
  train/                 # (and/or valid/, test/)
    <scenario_A>/        # one folder per scenario
      <vehicle_folder_0>/    # one folder per vehicle in the scenario
        000069.yaml
        000069.pcd
        000069_camera0.png
        000069_camera1.png
        000069_camera2.png
        000069_camera3.png
        000070.yaml
        ...
      <vehicle_folder_1>/
        ... (same pattern)
    <scenario_B>/
      ...

Output (one pickle per split)
-----------------------------
<split>_data.pkl with structure:
{
  'meta': {
      'name': 'OPV2V',
      'agents': [ 'vehicle_0', 'vehicle_1', ... ],
      'sensors': ['camera0','camera1','camera2','camera3','lidar01'],
      'fps': 10,
      'occlusion': {...}
  },
  'scenarios': {
     '<scenario_name>': {
        'vehicle_0': {
           <timestamp: float> : {                       # timestamp = frame_idx * 0.1 s
              'images': { 'camera0': path, ... },
              'lidar': path,
              'labels': [ {x,y,z,length,width,height,yaw,vel_x,vel_y,obj_id, occ_l1}, ... ],
              'ego_state': {x,y,z,yaw,vel_x,vel_y,obj_id=-100},
              'calibration': {
                  'cameras': {
                      'camera0': {'K': [[..]], 'T_cam_to_veh': [[..]], 'T_veh_to_cam': [[..]]},
                      ...
                  },
                  'lidar': {'T_lidar_to_veh': [[..]], 'T_veh_to_lidar': [[..]]},
                  'world': {'T_veh_to_world': [[..]], 'T_world_to_veh': [[..]]},
              }
           },
           ...
        },
        'vehicle_1': {...},
        ...
     },
     ...
  }
}

Implementation highlights
------------------------
- Enumerate **scenarios** by folders in split dir; enumerate **vehicles** by folders
  inside a scenario; enumerate **frames** by YAML files (sorted by numeric stem).
- Build paths for images and LiDAR using the same stem and the `_camera{0..3}` suffix.
- Parse per-frame YAML to get:
  • Camera intrinsics/extrinsics;  • LiDAR/world poses;  • Ego state;  • Other vehicles.
- Compute **L1 Fast BEV Angular Occlusion (2D)** per object and store as `occ_l1`.
- Store both directions of every transform to avoid ambiguity later.

"""

from __future__ import annotations
import os
import math
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

import yaml


# ============================
# Config / Constants (fixed)
# ============================
class Cfg:
    # Camera names in OPV2V folders (as seen in your directory preview)
    CAMERAS = ["camera0", "camera1", "camera2", "camera3"]

    # Name used for the LiDAR entry in the unified meta
    LIDAR_NAME = "lidar01"

    # ***** FIXED FPS FOR OPV2V *****
    FPS = 10                         # recorded at 10 Hz → step = 0.1 s
    STEP = 1.0 / FPS

    # L1 BEV Angular Occlusion settings (lightweight, deterministic)
    OCCLUSION_RAYS_K = 31
    OCCLUSION_EPS = 1e-2
    OCCLUSION_VERTICAL_CHECK = True

    # OPV2V yaw angles are in DEGREES (camera/ego/vehicles); keep True
    YAW_IN_DEGREES = True


# ============================
# Small math helpers
# ============================

def cosd(a_deg: float) -> float:
    """cosine with degree input."""
    return math.cos(math.radians(a_deg))


def sind(a_deg: float) -> float:
    """sine with degree input."""
    return math.sin(math.radians(a_deg))


def rpy_to_R(roll: float, yaw: float, pitch: float, degrees: bool = True) -> np.ndarray:
    """Rotation from roll (x), yaw (z), pitch (y).
    OPV2V angle order from YAML examples: [roll, yaw, pitch].
    We apply Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    if degrees:
        roll, yaw, pitch = map(math.radians, (roll, yaw, pitch))
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return (Rz @ Ry) @ Rx


def pose_to_T(x: float, y: float, z: float, roll: float, yaw: float, pitch: float, degrees: bool = True) -> np.ndarray:
    """Compose a 4×4 pose matrix from position + RPY angles."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_R(roll, yaw, pitch, degrees=degrees)
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Efficient rigid transform inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ============================
# Geometry helpers for L1 BEV
# ============================

def _rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _box_corners_xy(cx: float, cy: float, length: float, width: float, yaw_rad: float) -> np.ndarray:
    """Oriented rectangle corners (BEV) for a 3D box footprint."""
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]], dtype=np.float64)
    R = _rot2d(yaw_rad)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float64)


def _angles_from(origin: np.ndarray, pts: np.ndarray) -> np.ndarray:
    v = pts - origin[None, :]
    return np.arctan2(v[:, 1], v[:, 0])


def _min_covering_arc(angles: np.ndarray) -> Tuple[float, float]:
    """Smallest unwrapped arc [L,R] covering all angles (handles ±π wrap)."""
    a = np.sort((angles + np.pi) % (2 * np.pi)) - np.pi
    a2 = np.concatenate([a, a + 2 * np.pi])
    n = len(a)
    gaps = a2[1:n] - a2[0:n - 1]
    i = int(np.argmax(gaps))
    L = a2[i + 1]
    R = L + (2 * np.pi - gaps[i])
    return L, R


def _ray_segment_intersection(s: np.ndarray, d: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Optional[float]:
    """Ray s + t d vs segment p0→p1. Return t≥0 if intersects; else None."""
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
    """Closest intersection distance from origin s to rectangle edges along angle theta."""
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
    return _min_covering_arc(angs)


def compute_l1_occlusion_for_frame(objects: List[dict], ego_state: Optional[dict],
                                   K: int = Cfg.OCCLUSION_RAYS_K,
                                   eps: float = Cfg.OCCLUSION_EPS,
                                   vertical_check: bool = Cfg.OCCLUSION_VERTICAL_CHECK,
                                   yaw_in_degrees: bool = Cfg.YAW_IN_DEGREES) -> Dict[int, float]:
    """Compute O_L1 (fraction of occluded rays) for each object as seen from ego.

    Steps (matches your LaTeX): find target span → sample K rays → distance to target
    → check nearer occluders (with optional vertical height test) → O = occluded/valid.
    """
    if not objects:
        return {}

    s = np.array([
        (ego_state['x'] if ego_state is not None else 0.0),
        (ego_state['y'] if ego_state is not None else 0.0)
    ], dtype=np.float64)

    rects, spans, heights = {}, {}, {}

    # Precompute BEV rectangles & angular spans for all objects
    for o in objects:
        yaw = math.radians(o['yaw']) if yaw_in_degrees else o['yaw']
        corners = _box_corners_xy(o['x'], o['y'], o['length'], o['width'], yaw)
        rects[o['obj_id']] = corners
        heights[o['obj_id']] = float(o['height'])
        spans[o['obj_id']] = _rect_angular_span(s, corners)

    occ_map: Dict[int, float] = {}

    for tgt in objects:
        tid = tgt['obj_id']
        corners_t = rects[tid]
        h_t = heights[tid]
        tL, tR = spans[tid]

        if K <= 0:
            occ_map[tid] = 0.0
            continue

        thetas = np.linspace(tL, tR, K)
        occluded = 0
        valid = 0

        # Distance from ego to *target* along each ray (precompute)
        d_t_list: List[Optional[float]] = [
            _ray_rect_distance(s, float(th), corners_t) for th in thetas
        ]

        for th, d_t in zip(thetas, d_t_list):
            if d_t is None:   # numerical edge cases: skip
                continue
            valid += 1
            blocked = False

            # Test candidate occluders (angular pre-filter + distance check)
            for o in objects:
                oid = o['obj_id']
                if oid == tid:
                    continue

                oL, oR = spans[oid]
                # unwrap th into [oL, oR] domain (accounts for 2π periodicity)
                th_u = float(th)
                while th_u < oL:
                    th_u += 2 * math.pi
                while th_u > oR:
                    th_u -= 2 * math.pi
                if not (oL - 1e-9 <= th_u <= oR + 1e-9):
                    continue

                d_o = _ray_rect_distance(s, float(th), rects[oid])
                if d_o is None:
                    continue

                # In-front test (+ slack)
                if d_o < d_t - eps:
                    if vertical_check:
                        h_o = heights[oid]
                        # Short occluder check: if too short relative to distances, un-occlude
                        if h_o < h_t * (d_o / d_t):
                            continue
                    blocked = True
                    break

            if blocked:
                occluded += 1

        occ_map[tid] = float(occluded / valid) if valid > 0 else 0.0
    return occ_map


# ============================
# YAML parsing helpers
# ============================

def _read_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def _build_calibration(frame_yaml: dict) -> dict:
    """Build minimal, explicit calibration structure from the per-frame YAML.

    Cameras: intrinsic K and T_cam→veh (plus inverse).  LiDAR: T_lidar→veh derived
    from poses.  World: T_veh→world from true_ego_pos (plus inverse).
    """
    # World pose of the *vehicle folder's ego* for this frame
    vx, vy, vz, vroll, vyaw, vpitch = frame_yaml['true_ego_pos']
    T_vw = pose_to_T(vx, vy, vz, vroll, vyaw, vpitch, degrees=True)
    T_wv = inv_T(T_vw)

    # Cameras
    cams = {}
    for cam in Cfg.CAMERAS:
        if cam not in frame_yaml:
            continue
        K = frame_yaml[cam]['intrinsic']
        # In OPV2V exports, camera 'extrinsic' is typically T_cam→veh
        T_c2v = np.array(frame_yaml[cam]['extrinsic'], dtype=np.float64)
        T_v2c = inv_T(T_c2v)
        cams[cam] = {
            'K': K,
            'T_cam_to_veh': T_c2v.tolist(),
            'T_veh_to_cam': T_v2c.tolist(),
        }

    # LiDAR pose (world) → convert to veh frame via T_wv
    lx, ly, lz, lroll, lyaw, lpitch = frame_yaml.get('lidar_pose', [0, 0, 0, 0, 0, 0])
    T_lw = pose_to_T(lx, ly, lz, lroll, lyaw, lpitch, degrees=True)
    T_lv = T_wv @ T_lw
    T_vl = inv_T(T_lv)

    return {
        'cameras': cams,
        'lidar': {
            'T_lidar_to_veh': T_lv.tolist(),
            'T_veh_to_lidar': T_vl.tolist(),
        },
        'world': {
            'T_veh_to_world': T_vw.tolist(),
            'T_world_to_veh': T_wv.tolist(),
        }
    }


def _build_ego_state(frame_yaml: dict) -> dict:
    """Create the ego_state entry (-100 id) from true_ego_pos and ego_speed."""
    vx, vy, vz, vroll, vyaw, vpitch = frame_yaml['true_ego_pos']
    spd = float(frame_yaml.get('ego_speed', 0.0))
    return {
        'x': float(vx), 'y': float(vy), 'z': float(vz),
        'yaw': float(vyaw),  # degrees
        'vel_x': spd * cosd(vyaw),
        'vel_y': spd * sind(vyaw),
        'obj_id': -100,
    }


def _build_labels(frame_yaml: dict) -> List[dict]:
    """Convert the 'vehicles' dict into a list of box dicts for our unified format.

    CARLA extents are half-dimensions → multiply by 2 to get (length, width, height).
    Planar velocities are approximated from scalar speed and yaw (degrees).
    """
    labels: List[dict] = []
    vehicles = frame_yaml.get('vehicles', {}) or {}
    for sid, v in vehicles.items():
        # extents are half sizes → full box sizes
        l = 2.0 * float(v['extent'][0])
        w = 2.0 * float(v['extent'][1])
        h = 2.0 * float(v['extent'][2])
        x, y, z = map(float, v['location'])
        roll, yaw, pitch = map(float, v['angle'])  # yaw is the middle value (deg)
        spd = float(v.get('speed', 0.0))
        labels.append({
            'label': 'vehicle',
            'x': x, 'y': y, 'z': z,
            'length': l, 'width': w, 'height': h,
            'yaw': yaw,  # degrees
            'vel_x': spd * cosd(yaw),
            'vel_y': spd * sind(yaw),
            'obj_id': int(sid),
        })
    return labels


# ============================
# Core preprocessing for a split
# ============================

def preprocess_split(root: Path, split: str) -> Optional[Path]:
    """Process one split directory (train/valid/test)."""
    split_dir = root / split
    if not split_dir.exists():
        print(f"[Info] Split '{split}' not found at {split_dir}, skipping.")
        return None

    scenarios: Dict[str, Dict[str, Dict[float, dict]]] = {}
    agent_name_set = set()

    # Each folder in split is a SCENARIO
    for scen_path in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scenario_name = scen_path.name
        scenarios.setdefault(scenario_name, {})

        # Each subfolder inside a scenario is a VEHICLE
        veh_dirs = sorted([p for p in scen_path.iterdir() if p.is_dir()])
        for v_idx, vdir in enumerate(veh_dirs):
            agent = f"vehicle_{v_idx}"
            agent_name_set.add(agent)
            scenarios[scenario_name].setdefault(agent, {})

            # YAMLs define frames. Sort by numeric stem (e.g., 000069 → 69)
            yaml_files = sorted(
                vdir.glob("*.yaml"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
            )
            if not yaml_files:
                print(f"[Warn] No YAML frames in {vdir}")
                continue

            # Determine zero-padding (e.g., '000069' → 6)
            pad_len = len(yaml_files[0].stem)

            for f_idx, ypath in enumerate(yaml_files):
                # Prefer numeric stem; fall back to enumerated index
                try:
                    frame_idx = int(ypath.stem)
                except Exception:
                    frame_idx = f_idx + 1
                timestamp = frame_idx * Cfg.STEP  # fixed 0.1 s stepping

                # --- Parse YAML ---
                frame_yaml = _read_yaml(ypath)

                # --- Build file paths (record even if missing to keep alignment) ---
                images = {}
                for cam in Cfg.CAMERAS:
                    img = vdir / f"{str(frame_idx).zfill(pad_len)}_{cam}.png"
                    images[cam] = str(img)
                    if not img.exists():
                        print(f"[Miss] {img}")
                lidar_path = vdir / f"{str(frame_idx).zfill(pad_len)}.pcd"
                if not lidar_path.exists():
                    print(f"[Miss] {lidar_path}")

                # --- Annotations ---
                labels = _build_labels(frame_yaml)
                ego_state = _build_ego_state(frame_yaml)

                # --- Occlusion (L1) from this agent's viewpoint ---
                occ_map = compute_l1_occlusion_for_frame(
                    labels, ego_state,
                    K=Cfg.OCCLUSION_RAYS_K,
                    eps=Cfg.OCCLUSION_EPS,
                    vertical_check=Cfg.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Cfg.YAW_IN_DEGREES,
                )
                for o in labels:
                    o['occ_l1'] = float(occ_map.get(o['obj_id'], 0.0))

                # --- Calibration (cameras, lidar, world) ---
                calibration = _build_calibration(frame_yaml)

                # --- Store unified record ---
                scenarios[scenario_name][agent][timestamp] = {
                    'images': images,
                    'lidar': str(lidar_path),
                    'labels': labels,
                    'ego_state': ego_state,
                    'calibration': calibration,
                }

        print(f"[OK] Scenario processed: {split}/{scenario_name}")

    # Global meta (shared sensors list; fps fixed to 10)
    sensors = [*Cfg.CAMERAS, Cfg.LIDAR_NAME]
    meta = {
        'name': 'OPV2V',
        'agents': sorted(agent_name_set),
        'sensors': sensors,
        'fps': Cfg.FPS,
        'occlusion': {
            'method': 'L1_BEV_Angular',
            'K': Cfg.OCCLUSION_RAYS_K,
            'eps': Cfg.OCCLUSION_EPS,
            'vertical_check': Cfg.OCCLUSION_VERTICAL_CHECK,
            'yaw_in_degrees': Cfg.YAW_IN_DEGREES,
        }
    }

    data = {'meta': meta, 'scenarios': scenarios}

    out_path = split_dir / f"{split}_data.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] {out_path}")
    return out_path


# ============================
# Entry point (no FPS flag)
# ============================

def main():
    parser = argparse.ArgumentParser(description='Preprocess OPV2V into DeepAccident-style pickle (fixed 10 FPS)')
    parser.add_argument('dataset_root', type=str, help='Path containing train/valid/test splits')
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    for split in ['train', 'valid', 'test']:
        preprocess_split(root, split)


if __name__ == '__main__':
    main()
