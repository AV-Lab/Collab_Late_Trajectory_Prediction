#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_obs_target_pickle.py
-------------------------
Build a list of
    {'obs': [L,4],  'target':[H,4],
     'obs_cat': <str|int>, 'target_cat': <str|int>}
samples in WORLD coordinates and save to a .pkl file.
"""

import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np


def to_world(det: dict, calib: dict):
    """
    Return (state, category) where `state` = [x, y, z, yaw] in world frame.
    """
    T = calib["ego_to_world"] @ calib["lidar_to_ego"]      
    R = T[:3, :3]
    ego_yaw = np.arctan2(R[1, 0], R[0, 0])
    xyz1 = np.array([det["x"], det["y"], det["z"], 1.0])
    xyz_w = T @ xyz1 
    yaw_w = (det["yaw"] + ego_yaw + np.pi) % (2 * np.pi) - np.pi
    state   = [xyz_w[0], xyz_w[1], xyz_w[2], yaw_w]
    cat_id  = det["label"]   
       
    return state, cat_id


def split_contiguous(det_list, fps, tol=1.5):
    """
    Yield contiguous sub-lists from `det_list` (sorted by time).
    A new segment starts whenever the gap > tol × frame_period.
    """

    
    frame_period = 1.0 / fps
    seg = [det_list[0]]
    for prev, cur in zip(det_list, det_list[1:]):
        if cur[0] - prev[0] > tol * frame_period:
            yield seg
            seg = []
        seg.append(cur)
    yield seg


def build_pairs(pkl_path, L, H, shift, fps=10):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    pairs = []

    for frames in dataset.values():                          
        ts_sorted = sorted(frames.keys())

        # gather detections keyed by obj_id
        tracks = defaultdict(list)                           
        for t in ts_sorted:
            calib = frames[t]["calibration"]
            for lab in frames[t]["labels"]:
                state, cat = to_world(lab, calib) # to global coordinates
                tracks[lab["obj_id"]].append((t, state, cat))

        # 2) split into contiguous segments and create samples
        for det_list in tracks.values():
            if len(det_list) == 0:
                
            for segment in split_contiguous(det_list, fps=fps):
                if len(segment) < L + H:
                    continue                                 # segment too short

                # split time - state - cat
                states = np.array([s for _, s, _ in segment], dtype=float)   # [M,4]
                cat_id = segment[0][2]   # constant for this track/segment

                M = len(states)
                idx = 0
                while idx + L + H <= M:
                    obs   = states[idx: idx + L , :]   
                    futur = states[idx + L : idx + L + H, :]   
                    pairs.append({"obs": obs, "target": futur, "obs_cat": cat_id, "target_cat": cat_id})
                    idx += shift
    return pairs


def save_pickle(data, out_file):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    pickle_in  = "/media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/DeepAccident/valid/ego_vehicle_valid_data.pkl"
    pickle_out = "../prediction/DeepAccident/10_20/ego_vehicle/valid.pkl"
    
    L, H, step = 10, 20, 3

    dataset = build_pairs(pickle_in, L, H, step, fps=10)
    print("built obs/target pairs:", len(dataset))

    print("saving →", pickle_out)
    save_pickle(dataset, pickle_out)
    print("done.")
