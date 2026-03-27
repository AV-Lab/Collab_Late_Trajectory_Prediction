#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2V4Real → unified pickle (DeepAccident-compatible) + optional occlusion visualization.

Assumptions based on the YAML format:

- `true_ego_pos` is a 4x4 homogeneous matrix (ego → world).
- `lidar_pose` in the YAML is the same 4x4 matrix; we assume LiDAR and ego
  share the same coordinate frame, so lidar_to_ego = I and ego_to_world
  is given by true_ego_pos.

- Vehicle entries are already in the global/world frame:
    vehicles:
      <id>:
        location: [x, y, z]
        extent:   [half_len_x, half_len_y, half_len_z]
        angle:    [roll, yaw, pitch] (we take yaw = angle[1])

The output structure matches the OPV2V preprocessor and DeepAccident style:
    data = {
        "scenarios": {
            scenario_name: {
                agent_name: {
                    timestamp: {
                        "images": {},          # empty (no cameras in V2V4Real)
                        "lidar": <path>,       # .pcd if exists else .bin
                        "lidar_pcd": <path or "">,
                        "lidar_bin": <path or "">,
                        "labels": [...],
                        "ego_state": {...},
                        "calibration": {...},
                    }
                }
            }
        }
    }

Occlusions are computed in world frame with yaw in radians.
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

from preprocess.occlusions import compute_l1_occlusion_for_frame
from preprocess.visualize import VisOcclusionScores
from visualization.bbox_visualize import BBoxVisualizer


class Constants:
    FPS  = 10
    STEP = 1.0 / FPS

    # Occlusion settings (same as DeepAccident / OPV2V script)
    OCCLUSION_RAYS_K         = 31
    OCCLUSION_EPS            = 1e-2
    OCCLUSION_VERTICAL_CHECK = True
    YAW_IN_DEGREES           = False  # we use radians everywhere

    KEEP_RADIUS_M = 50.0


# ───────────────────────────────────────── helpers ───────────────────────────────────────── #

def _to_builtin(obj):
    """Recursively convert numpy scalars/arrays to Python builtins."""
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
    """Load YAML that may contain numpy objects, converting them to builtins."""
    with open(yaml_path, "rb") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            f.seek(0)
            data = yaml.load(f, Loader=yaml.UnsafeLoader)
    return _to_builtin(data)


def _get_ego_T(frame_yaml: dict) -> np.ndarray:
    """
    Extract ego->world transform from frame_yaml["true_ego_pos"] which is a 4x4
    homogeneous matrix (as reconstructed from numpy in the YAML).
    """
    M = np.asarray(frame_yaml["true_ego_pos"], dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"true_ego_pos is not 4x4, got {M.shape}")
    return M


def _build_calibration(frame_yaml: dict) -> dict:
    """
    Build a minimal calibration dict:

      - 'ego_to_world', 'world_to_ego'
      - 'lidar_to_ego', 'ego_to_lidar'

    For V2V4Real we assume LiDAR and ego share the same frame, so:
      lidar_to_ego = I
    """
    T_ev = _get_ego_T(frame_yaml)             # ego -> world
    T_ve = np.linalg.inv(T_ev)               # world -> ego

    T_lidar_to_ego = np.eye(4, dtype=np.float64)
    T_ego_to_lidar = np.eye(4, dtype=np.float64)

    calib = {
        "lidar_to_ego": T_lidar_to_ego.tolist(),
        "ego_to_lidar": T_ego_to_lidar.tolist(),
        "ego_to_world": T_ev.tolist(),
        "world_to_ego": T_ve.tolist(),
    }
    return calib


def _build_ego_state(frame_yaml: dict) -> dict:
    """
    Ego state in world coordinates, yaw in radians.

    Position and yaw are derived from the 4x4 ego->world matrix.
    Velocity uses ego_speed * [cos(yaw), sin(yaw)].
    """
    T_ev = _get_ego_T(frame_yaml)
    R = T_ev[:3, :3]
    t = T_ev[:3, 3]

    yaw = float(math.atan2(R[1, 0], R[0, 0]))  # radians

    spd = float(frame_yaml.get("ego_speed", 0.0))

    return {
        "x": float(t[0]),
        "y": float(t[1]),
        "z": float(t[2]),
        "yaw": yaw,                        # radians
        "vel_x": spd * math.cos(yaw),
        "vel_y": spd * math.sin(yaw),
        "obj_id": -100,
    }


def _build_labels(frame_yaml: dict) -> List[dict]:
    """
    Per-vehicle labels in *world* frame, yaw in radians (world).

    V2V4Real vehicle format (from YAML):
        vehicles:
          id:
            location: [x, y, z]   # in ego/LiDAR frame
            extent:   [hx, hy, hz]
            angle:    [roll, yaw_deg, pitch]  # yaw in ego frame
            obj_type: "Car" / "Truck" / ...
    """
    labels: List[dict] = []
    vehicles = frame_yaml.get("vehicles", {}) or {}

    # ego -> world transform and ego yaw in world frame
    T_ev = _get_ego_T(frame_yaml)
    R_ev = T_ev[:3, :3]
    ego_yaw = math.atan2(R_ev[1, 0], R_ev[0, 0])  # radians

    for sid, v in vehicles.items():
        # box size (full dims)
        ex, ey, ez = map(float, v["extent"])
        L = 2.0 * ex
        W = 2.0 * ey
        H = 2.0 * ez

        # position: ego/LiDAR -> world
        x_e, y_e, z_e = map(float, v["location"])
        p_e = np.array([x_e, y_e, z_e, 1.0], dtype=np.float64)
        p_w = T_ev @ p_e
        x_w, y_w, z_w = map(float, p_w[:3])

        # yaw: ego frame -> world frame
        ang = v.get("angle", [0.0, 0.0, 0.0])
        yaw_deg_local = float(ang[1]) if len(ang) >= 2 else 0.0
        yaw_local = math.radians(yaw_deg_local)      # yaw in ego frame
        yaw_world = ego_yaw + yaw_local              # rotate into world frame

        # optional: wrap to [-pi, pi] (not strictly required but nice)
        yaw_world = (yaw_world + math.pi) % (2 * math.pi) - math.pi

        # speed: given along local yaw -> rotate into world
        spd = float(v.get("speed", 0.0))
        vel_local = np.array(
            [spd * math.cos(yaw_local), spd * math.sin(yaw_local), 0.0],
            dtype=np.float64,
        )
        vel_world = R_ev @ vel_local

        labels.append(
            {
                "label": v.get("obj_type", "vehicle"),
                "x": x_w,
                "y": y_w,
                "z": z_w,
                "length": L,
                "width": W,
                "height": H,
                "yaw": yaw_world,                   # world yaw (radians)
                "vel_x": float(vel_world[0]),
                "vel_y": float(vel_world[1]),
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


def _lidar_xyz_from_files(pcd_path: Path, bin_path: Path) -> Optional[np.ndarray]:
    """
    Load LiDAR XYZ from .pcd or .bin file (for debug visualization).

    Preference:
        - use .pcd if it exists
        - else use .bin (KITTI-like float32 [x,y,z,intensity])
    """
    if pcd_path.is_file():
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            return None
        return pts[:, :3]

    if bin_path.is_file():
        raw = np.fromfile(str(bin_path), dtype=np.float32)
        if raw.size == 0:
            return None
        pts = raw.reshape(-1, 4)
        return pts[:, :3]

    return None


# ───────────────────────────────────────── main preprocessing ───────────────────────────────────────── #

def _find_agent_dirs(scen_path: Path) -> List[Path]:
    """
    V2V4Real structure can be, e.g.:

        <split>/<scenario>/astuff/0/000000.yaml
        <split>/<scenario>/tesla/1/000000.yaml

    or directly:

        <split>/<scenario>/0/000000.yaml

    This helper returns all *leaf* dirs that actually contain YAML files.
    """
    agent_dirs: List[Path] = []

    # first level under scenario
    for d in sorted(p for p in scen_path.iterdir() if p.is_dir()):
        yaml_here = list(d.glob("*.yaml"))
        if yaml_here:
            agent_dirs.append(d)
            continue

        # second level (e.g. astuff/0, tesla/1)
        for dd in sorted(p2 for p2 in d.iterdir() if p2.is_dir()):
            if list(dd.glob("*.yaml")):
                agent_dirs.append(dd)

    return agent_dirs


def preprocess_dataset(dataset_root: str, prefix: str, visualize: bool = False) -> Path:
    """
    Process a single split (train/valid/test) into <split>_data.pkl.
    """
    split_dir = Path(dataset_root).expanduser().resolve() / prefix
    if not split_dir.exists():
        print(f"[Info] Split '{prefix}' not found at {split_dir}, skipping.")
        return split_dir / f"{prefix}_data.pkl"

    scenarios: Dict[str, Dict[str, Dict[float, dict]]] = {}
    agent_name_set = set()

    viz_dir = split_dir / "viz"
    if visualize and not viz_dir.exists():
        viz_dir.mkdir(parents=True, exist_ok=True)

    bbox_vis: Optional[BBoxVisualizer] = BBoxVisualizer() if not visualize else None

    for scen_path in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scenario_name = f"{prefix}/{scen_path.name}"
        scenarios.setdefault(scenario_name, {})

        agent_dirs = _find_agent_dirs(scen_path)
        if not agent_dirs:
            print(f"[Warn] No agent dirs (with YAML) found in {scen_path}")
            continue

        # global minimum numeric frame index across all agents (for fallback)
        all_yaml = [y for a in agent_dirs for y in a.glob("*.yaml")]
        numeric_stems: List[int] = []
        for pth in all_yaml:
            try:
                numeric_stems.append(int(pth.stem))
            except Exception:
                pass
        min_idx_scenario = min(numeric_stems) if numeric_stems else 0

        for a_idx, agent_dir in enumerate(agent_dirs):
            # agent name encodes both parent folder and leaf folder
            agent_name = f"{agent_dir.parent.name}_{agent_dir.name}"
            agent_name_set.add(agent_name)
            scenarios[scenario_name].setdefault(agent_name, {})

            yaml_files = sorted(
                agent_dir.glob("*.yaml"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
            )
            if not yaml_files:
                print(f"[Warn] No YAML frames in {agent_dir}")
                continue

            pad_len = len(yaml_files[0].stem)

            for f_idx, ypath in enumerate(yaml_files):
                try:
                    frame_idx = int(ypath.stem)
                except Exception:
                    frame_idx = min_idx_scenario + f_idx

                timestamp = f_idx * Constants.STEP
                frame_yaml = _read_yaml(ypath)

                # lidar file names: 000000.pcd and/or 000000.bin
                stem = str(frame_idx).zfill(pad_len)
                pcd_path = agent_dir / f"{stem}.pcd"
                bin_path = agent_dir / f"{stem}.bin"

                lidar_main: Optional[Path] = None
                if pcd_path.is_file():
                    lidar_main = pcd_path
                elif bin_path.is_file():
                    lidar_main = bin_path

                if lidar_main is None:
                    print(f"[Warn] No .pcd or .bin for frame {stem} in {agent_dir}")
                    lidar_str = ""
                else:
                    lidar_str = str(lidar_main)

                # labels + ego
                labels_full = _build_labels(frame_yaml)   # world frame, yaw in rad
                ego_state   = _build_ego_state(frame_yaml)

                # filter by radius (50 m)
                labels = _filter_by_radius(
                    labels_full, ego_state, Constants.KEEP_RADIUS_M
                )

                # occlusion
                occ_map, debug = compute_l1_occlusion_for_frame(
                    labels,
                    ego_state,
                    K=Constants.OCCLUSION_RAYS_K,
                    eps=Constants.OCCLUSION_EPS,
                    vertical_check=Constants.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Constants.YAW_IN_DEGREES,   # False – yaw in rad
                    return_debug=True,
                )
                for o in labels:
                    o["occ_l1"] = float(occ_map.get(o["obj_id"], 0.0))

                # calibration
                calibration = _build_calibration(frame_yaml)

                scenarios[scenario_name][agent_name][timestamp] = {
                    "images": {},                     # no cameras in V2V4Real
                    "lidar": lidar_str,              # main lidar file
                    "lidar_pcd": str(pcd_path) if pcd_path.is_file() else "",
                    "lidar_bin": str(bin_path) if bin_path.is_file() else "",
                    "labels": labels,
                    "ego_state": ego_state,
                    "calibration": calibration,
                }

                # 2D BEV occlusion debug PNGs (optional)
                if visualize:
                    out_png = viz_dir / f"{scen_path.name}_{agent_name}_{stem}.png"
                    title = f"{prefix}/{scen_path.name} | {agent_name} | frame {frame_idx}"
                    VisOcclusionScores().render(
                        debug, occ_map, title=title, out_path=str(out_png)
                    )

                # 3D BBoxVisualizer check (once per scenario & first agent)
                if visualize and bbox_vis is not None: # and a_idx == 0 and f_idx == 0:
                    xyz_lidar = _lidar_xyz_from_files(pcd_path, bin_path)
                    if xyz_lidar is not None:
                        calib = calibration
                        T_ego_world  = np.array(calib["ego_to_world"])  # 4x4
                        T_lidar_ego  = np.array(calib["lidar_to_ego"])  # 4x4 (I)
                        T_lidar_world = T_ego_world @ T_lidar_ego       # lidar->world

                        xyz_h = np.hstack([xyz_lidar,
                                           np.ones((xyz_lidar.shape[0], 1),
                                                   dtype=np.float32)])
                        xyz_world = (T_lidar_world @ xyz_h.T).T[:, :3]

                        pc_dict = {"data": xyz_world}

                        dets = [
                            {
                                "x": o["x"],
                                "y": o["y"],
                                "z": o["z"],
                                "dx": o["length"],
                                "dy": o["width"],
                                "dz": o["height"],
                                "yaw": o["yaw"],            # radians
                                "occ_score": o.get("occ_l1"),
                            }
                            for o in labels
                        ]

                        ego_for_vis = {
                            "x": ego_state["x"],
                            "y": ego_state["y"],
                            "z": ego_state["z"],
                            "yaw": ego_state["yaw"],
                        }

                        bbox_vis.visualize(pc_dict, dets, ego_for_vis)

        print(f"[OK] Scenario processed: {scenario_name}")

    # meta summary per split
    scene_vehicle_counts = {
        scene_name: len(agent_dict) for scene_name, agent_dict in scenarios.items()
    }
    max_vehicles = max(scene_vehicle_counts.values()) if scene_vehicle_counts else 0

    meta_path = split_dir / "meta.txt"
    with open(meta_path, "w") as mf:
        mf.write("dataset=V2V4Real\n")
        mf.write(f"split={prefix}\n")
        mf.write(f"fps={Constants.FPS}\n")
        mf.write(f"max_vehicles={max_vehicles}\n")
        mf.write("\nscenario_name,num_agents\n")
        for scen_name, nveh in sorted(scene_vehicle_counts.items()):
            mf.write(f"{scen_name},{nveh}\n")

    data = {"scenarios": scenarios}
    out_path = split_dir / f"{prefix}_data.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] {out_path}")
    print(f"[META] {meta_path}")
    return out_path


# ───────────────────────────────────────── CLI ───────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V2V4Real → unified pickle with optional occlusion visualization"
    )
    parser.add_argument("dataset_root", type=str, help="Path containing train/valid/test")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        help="Splits to process (directory names under dataset_root)."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize occlusions + one 3D frame per scenario."
    )
    args = parser.parse_args()

    root = args.dataset_root
    for split in args.splits:
        preprocess_dataset(root, prefix=split, visualize=args.visualize)
