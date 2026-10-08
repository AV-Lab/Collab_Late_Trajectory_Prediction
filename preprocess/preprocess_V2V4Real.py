#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2V4Real → unified pickle (DeepAccident-compatible).

Assumptions based on the YAML format:

- `true_ego_pos` is a 4x4 homogeneous matrix (ego → world).
- `lidar_pose` in the YAML is the same 4x4 matrix; we assume LiDAR and ego
  share the same coordinate frame, so lidar_to_ego = I and ego_to_world
  is given by true_ego_pos.

- Vehicle entries are expressed in the local ego/LiDAR frame:
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
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from preprocess.occlusions import compute_l1_occlusion_for_frame
from preprocess.constants import Constants


V2V4REAL_CATEGORY_MAPPING = {
    "vehicle": "vehicle",
    "car": "vehicle",
    "Car": "vehicle",
    "Truck": "vehicle",
    "ConcreteTruck": "vehicle",
    "Bus": "vehicle",
    "Van": "vehicle",
    "Pedestrian": "pedestrian",
    "BicycleRider": "cyclist",
    "Scooter": "motorcyclist",
    "ScooterRider": "motorcyclist",
}


def _canonicalize_label(raw_label: str) -> str:
    try:
        return V2V4REAL_CATEGORY_MAPPING[raw_label]
    except KeyError as error:
        raise ValueError(
            f"Unsupported V2V4Real category: {raw_label!r}."
        ) from error


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
        "cameras": {},
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

    spd = float(frame_yaml["ego_speed"])

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
    vehicles = frame_yaml["vehicles"] or {}

    # ego -> world transform and ego yaw in world frame
    T_ev = _get_ego_T(frame_yaml)
    R_ev = T_ev[:3, :3]
    ego_yaw = math.atan2(R_ev[1, 0], R_ev[0, 0])  # radians

    for sid, v in vehicles.items():
        original_label = str(v["obj_type"])
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
        ang = v["angle"]
        yaw_deg_local = float(ang[1])
        yaw_local = math.radians(yaw_deg_local)      # yaw in ego frame
        yaw_world = ego_yaw + yaw_local              # rotate into world frame

        # optional: wrap to [-pi, pi] (not strictly required but nice)
        yaw_world = (yaw_world + math.pi) % (2 * math.pi) - math.pi

        labels.append(
            {
                "label": _canonicalize_label(original_label),
                "original_label": original_label,
                "x": x_w,
                "y": y_w,
                "z": z_w,
                "length": L,
                "width": W,
                "height": H,
                "yaw": yaw_world,                   # world yaw (radians)
                "vel_x": 0.0,
                "vel_y": 0.0,
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


def _assign_temporal_velocities(frames: Dict[float, dict]) -> None:
    """Estimate world-frame object velocities from consecutive positions."""
    observations = {}
    for timestamp, frame in frames.items():
        for obj in frame["labels"]:
            observations.setdefault(obj["obj_id"], []).append((timestamp, obj))

    for sequence in observations.values():
        sequence.sort(key=lambda item: item[0])
        for index, (timestamp, obj) in enumerate(sequence):
            if len(sequence) == 1:
                continue
            if index == 0:
                other_timestamp, other_obj = sequence[1]
            elif index == len(sequence) - 1:
                other_timestamp, other_obj = sequence[-2]
            else:
                previous_timestamp, previous_obj = sequence[index - 1]
                next_timestamp, next_obj = sequence[index + 1]
                delta_t = next_timestamp - previous_timestamp
                obj["vel_x"] = (next_obj["x"] - previous_obj["x"]) / delta_t
                obj["vel_y"] = (next_obj["y"] - previous_obj["y"]) / delta_t
                continue

            delta_t = other_timestamp - timestamp
            obj["vel_x"] = (other_obj["x"] - obj["x"]) / delta_t
            obj["vel_y"] = (other_obj["y"] - obj["y"]) / delta_t


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


def preprocess_dataset(dataset_root: str, prefix: str) -> Path:
    """
    Process a single split (train/valid/test) into <split>_data.pkl.
    """
    split_dir = Path(dataset_root).expanduser().resolve() / prefix
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Requested split '{prefix}' was not found at {split_dir}"
        )

    output_dir = (
        Path(__file__).resolve().parent
        / "preprocessed"
        / "V2V4Real"
        / prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios: Dict[str, Dict[str, Dict[float, dict]]] = {}
    passed_scenarios = 0
    failed_scenarios = 0

    for scen_path in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scenario_name = f"{prefix}/{scen_path.name}"
        agent_dirs = _find_agent_dirs(scen_path)
        if not agent_dirs:
            failed_scenarios += 1
            print(f"[SKIP] Scenario {scenario_name}: no agent directories found")
            continue

        yaml_files_by_agent = {
            agent_dir: sorted(agent_dir.glob("*.yaml"), key=lambda path: int(path.stem))
            for agent_dir in agent_dirs
        }
        frame_sets = {
            tuple(path.stem for path in yaml_files)
            for yaml_files in yaml_files_by_agent.values()
        }
        if len(frame_sets) != 1 or not next(iter(frame_sets), ()):
            failed_scenarios += 1
            print(f"[SKIP] Scenario {scenario_name}: agent frame sets are incomplete or unsynchronized")
            continue

        scenario_data = {}
        validation_failure = None

        for agent_dir in agent_dirs:
            # agent name encodes both parent folder and leaf folder
            agent_name = f"{agent_dir.parent.name}_{agent_dir.name}"
            agent_frames = {}

            for f_idx, ypath in enumerate(yaml_files_by_agent[agent_dir]):
                timestamp = f_idx * Constants.V2V4REAL_STEP
                frame_yaml = _read_yaml(ypath)

                # lidar file names: 000000.pcd and/or 000000.bin
                stem = ypath.stem
                pcd_path = agent_dir / f"{stem}.pcd"
                bin_path = agent_dir / f"{stem}.bin"

                lidar_main: Optional[Path] = None
                if pcd_path.is_file():
                    lidar_main = pcd_path
                elif bin_path.is_file():
                    lidar_main = bin_path

                if lidar_main is None:
                    validation_failure = f"required LiDAR file missing for {ypath}"
                    break
                if lidar_main.stat().st_size == 0:
                    validation_failure = f"required LiDAR file is empty: {lidar_main}"
                    break
                if lidar_main.suffix == ".bin" and lidar_main.stat().st_size % 16 != 0:
                    validation_failure = f"invalid LiDAR BIN file: {lidar_main}"
                    break

                required_yaml_fields = ("true_ego_pos", "lidar_pose", "ego_speed", "vehicles")
                missing_yaml_fields = [
                    field
                    for field in required_yaml_fields
                    if field not in frame_yaml or frame_yaml[field] is None
                ]
                if missing_yaml_fields:
                    validation_failure = (
                        f"required field '{missing_yaml_fields[0]}' missing in {ypath}"
                    )
                    break

                T_ego_world = np.asarray(frame_yaml["true_ego_pos"], dtype=np.float64)
                T_lidar_world = np.asarray(frame_yaml["lidar_pose"], dtype=np.float64)
                if T_ego_world.shape != (4, 4) or T_lidar_world.shape != (4, 4):
                    validation_failure = f"invalid ego/LiDAR pose shape in {ypath}"
                    break
                if not np.allclose(T_ego_world, T_lidar_world):
                    validation_failure = f"LiDAR and ego frames differ in {ypath}"
                    break

                # labels + ego
                labels_full = _build_labels(frame_yaml)   # world frame, yaw in rad
                ego_state   = _build_ego_state(frame_yaml)

                # filter by radius (50 m)
                labels = _filter_by_radius(
                    labels_full, ego_state, Constants.KEEP_RADIUS_M
                )

                # occlusion
                occ_map, _ = compute_l1_occlusion_for_frame(
                    labels,
                    ego_state,
                    K=Constants.OCCLUSION_RAYS_K,
                    eps=Constants.OCCLUSION_EPS,
                    vertical_check=Constants.OCCLUSION_VERTICAL_CHECK,
                    yaw_in_degrees=Constants.YAW_IN_DEGREES,   # False – yaw in rad
                    return_debug=False,
                )
                for o in labels:
                    o["occ_l1"] = float(occ_map[o["obj_id"]])

                # calibration
                calibration = _build_calibration(frame_yaml)

                agent_frames[timestamp] = {
                    "images": {},                     # no cameras in V2V4Real
                    "lidar": str(lidar_main),
                    "labels": labels,
                    "ego_state": ego_state,
                    "calibration": calibration,
                }

            if validation_failure is not None:
                break

            _assign_temporal_velocities(agent_frames)
            scenario_data[agent_name] = agent_frames

        if validation_failure is not None:
            failed_scenarios += 1
            print(f"[SKIP] Scenario {scenario_name}: {validation_failure}")
            continue

        scenarios[scenario_name] = scenario_data
        passed_scenarios += 1
        print(f"[OK] Scenario processed: {scenario_name}")

    print(
        f"[SUMMARY] V2V4Real {prefix}: "
        f"{passed_scenarios} scenarios passed, {failed_scenarios} scenarios failed"
    )

    # meta summary per split (for quick sanity checking)
    scene_vehicle_counts = {
        scene_name: len(agent_dict) for scene_name, agent_dict in scenarios.items()
    }
    max_vehicles = max(scene_vehicle_counts.values()) if scene_vehicle_counts else 0

    # duration per scenario based on sensor observation length
    # duration = number of frames * STEP
    scene_durations = {}
    for scene_name, agent_dict in scenarios.items():
        max_num_frames = 0

        for _, frame_dict in agent_dict.items():
            num_frames = len(frame_dict)
            max_num_frames = max(max_num_frames, num_frames)

        scene_durations[scene_name] = max_num_frames * Constants.V2V4REAL_STEP

    meta_path = output_dir / "meta.txt"
    with open(meta_path, "w") as mf:
        mf.write("dataset=V2V4Real\n")
        mf.write(f"split={prefix}\n")
        mf.write(f"fps={Constants.V2V4REAL_FPS}\n")
        mf.write(f"global=True\n")
        mf.write(f"max_vehicles={max_vehicles}\n")
        mf.write("\nscenario_name,num_vehicles,duration\n")

        for scen_name, nveh in sorted(scene_vehicle_counts.items()):
            duration = scene_durations[scen_name]
            mf.write(f"{scen_name},{nveh},{duration:.2f}\n")

    data = {"scenarios": scenarios}
    output_pickle_path = output_dir / f"{prefix}_data.pkl"
    with open(output_pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset to {output_pickle_path}")
    print(f"[META] {meta_path}")
    return output_pickle_path


# ───────────────────────────────────────── CLI ───────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V2V4Real → unified pickle"
    )
    parser.add_argument("dataset_root", type=str, help="Path containing train/valid/test")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        help="Splits to process (directory names under dataset_root)."
    )
    args = parser.parse_args()

    root = args.dataset_root
    for split in args.splits:
        preprocess_dataset(root, prefix=split)
