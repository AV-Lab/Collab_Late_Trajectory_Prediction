#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-process V2V-4Real into the *same* pickle layout as DeepAccident
(images • lidar • labels • ego_state • calibration).

Directory layout expected
-------------------------
<root>/
    train/  |  valid/
        <scenario_X>/
            astuff/   |  tesla/
                gps/
                lidar/              000000.bin …
                tf/                 000000.txt (4×4 matrix)
                label_opv2v/        000000.json …

Assumptions
-----------
* 10 Hz lidar (dt = 0.1 s → FPS = 10).
* No cameras in V2V-4Real → `images` dict is left **empty**.
* `label_opv2v/*.json` is used (same format as the public repo).
* The 4×4 **tf** matrix already maps LiDAR → global frame.
"""

from __future__ import annotations
import json, os, pickle, copy
from pathlib import Path
from typing import Dict, List
import numpy as np


# ───────────────────────── constants ──────────────────────────
FPS            = 10             # 10 Hz as in the paper
STEP           = 1.0 / FPS
LIDAR_SENSOR   = "lidar"
AGENTS         = ["astuff", "tesla"]          # two CAVs per scenario
CAMERA_SENSORS: List[str] = []                # none in this dataset


# ──────────────────── small parsing helpers ───────────────────
def parse_tf(tf_path: Path) -> Dict | None:
    """Return {R,t,yaw,pitch,roll} from 4×4 txt.  None if file missing."""
    if not tf_path.exists():
        return None
    M = np.loadtxt(tf_path, dtype=float).reshape(4, 4)
    R, t = M[:3, :3], M[:3, 3]
    # ZYX Euler
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    return dict(R=R, t=t, yaw=float(yaw),
                pitch=float(pitch), roll=float(roll))


def parse_json_label(j_path: Path) -> List[dict]:
    """
    OPV2V / V2V-4Real json → list of dicts with the *same* keys used in
    DeepAccident (‘label’, x,y,z, length,width,height, yaw, vel_x, vel_y, obj_id)
    """
    if not j_path.exists():
        return []

    with open(j_path) as f:
        raw = json.load(f)

    out = []
    for obj in raw["objects"]:
        bbox = obj["box"]           # [x, y, z, l, w, h, yaw]
        vel  = obj.get("velocity", [0, 0, 0])
        out.append(dict(
            label   = obj["type"],
            x       = bbox[0],
            y       = bbox[1],
            z       = bbox[2],
            length  = bbox[3],
            width   = bbox[4],
            height  = bbox[5],
            yaw     = bbox[6],
            vel_x   = vel[0],
            vel_y   = vel[1],
            obj_id  = int(obj["id"])
        ))
    return out


# ────────────────────── main routine ──────────────────────────
def preprocess_dataset(root: str, split: str = "train"):
    """
    Parameters
    ----------
    root  : path to V2V-4Real directory (containing train/ valid/ …)
    split : "train" or "valid"
    """
    root   = Path(root)
    split_dir = root / split
    assert split_dir.is_dir(), f"{split_dir} not found"

    # meta -----------------------------------------------------------------
    sensors = copy.deepcopy(CAMERA_SENSORS) + [LIDAR_SENSOR]
    data = dict(
        meta=dict(
            name="V2V4Real",
            agents=AGENTS,
            sensors=sensors,
            fps=FPS
        ),
        scenarios={}
    )

    # discover scenarios ----------------------------------------------------
    for scen_dir in sorted(split_dir.iterdir()):
        if not scen_dir.is_dir():
            continue
        scen_name = scen_dir.name                           # e.g. testoutput_…
        data["scenarios"][scen_name] = {}

        # each vehicle ------------------------------------------------------
        for agent in AGENTS:
            veh_dir = scen_dir / agent
            if not veh_dir.exists():
                continue
            data["scenarios"][scen_name][agent] = {}

            lidar_dir = veh_dir / "lidar"
            label_dir = veh_dir / "label_opv2v"
            tf_dir    = veh_dir / "tf"

            frame_ids = sorted([p.stem for p in lidar_dir.glob("*.bin")])

            for i, fid in enumerate(frame_ids):
                t = round(i * STEP, 1)        # 0.0, 0.1, …
                entry = dict(
                    images={},                # no cameras
                    lidar=str(lidar_dir / f"{fid}.bin"),
                    labels=parse_json_label(label_dir / f"{fid}.json"),
                    ego_state=None,           # filled below
                    calibration=None          # ditto
                )

                # ego pose / calibration from tf ---------------------------
                tf_path = tf_dir / f"{fid}.txt"
                tf_data = parse_tf(tf_path)
                entry["ego_state"]  = tf_data
                entry["calibration"] = tf_data   # store the same struct

                data["scenarios"][scen_name][agent][t] = entry

        print(f"✓ {scen_name} processed")

    # write pickle ----------------------------------------------------------
    out_path = split_dir / f"{split}_data.pkl"
    with out_path.open("wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_path}\n")


# ───────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    ROOT = "/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/V2V4Real"
    preprocess_dataset(ROOT, "train")
    preprocess_dataset(ROOT, "valid")
