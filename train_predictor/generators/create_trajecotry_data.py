from __future__ import annotations
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

def to_world(det: dict, calib: dict) -> Tuple[List[float], Any]:
    """
    Return (state, category) where state=[x,y,z,yaw] in world coordinates.
    Expects 4x4 transforms at calib['ego_to_world'] and calib['lidar_to_ego'].
    If transforms are missing, falls back to identity.
    """
    ego2world = calib["ego_to_world"]
    lidar2ego = calib["lidar_to_ego"]
    T = np.array(ego2world, dtype=np.float64) @ np.array(lidar2ego, dtype=np.float64)
    R = T[:3, :3]
    ego_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    xyz1 = np.array([det["x"], det["y"], det["z"], 1.0], dtype=np.float64)
    xyz_w = (T @ xyz1)
    yaw_w = float((det["yaw"] + ego_yaw + np.pi) % (2 * np.pi) - np.pi)
    state = [float(xyz_w[0]), float(xyz_w[1]), float(xyz_w[2]), yaw_w]
    cat_id = det["label"]    
    return state, cat_id

def split_contiguous(times: List[float], fps: float, tol: float = 1.5) -> List[List[int]]:
    """
    Split sorted timestamps into contiguous index segments.
    New segment starts if gap > tol * (1/fps).
    Return list of lists of indices into the sorted times array.
    """
    frame_period = 1.0 / float(fps)
    segs = [[0]]
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) > tol * frame_period:
            segs.append([i])
        else:
            segs[-1].append(i)
    return segs

def build_samples(data_path: Path, L: int, H: int, stride: int, fps: float, use_global: bool) -> List[dict]:
    """
    Iterate the unified dataset and build [obs,target] windows for all objects.

    Returns
    -------
    samples : list of dicts with keys:
      'obs' [L,4], 'target' [H,4], 'obs_cat', 'target_cat',
      'meta' (scenario, agent, obj_id, track_id, timestamps)
    """
    with data_path.open("rb") as f:
        data = pickle.load(f)

    scenarios = data.get("scenarios", {})
    samples: List[dict] = []

    for scenario_name, agents in scenarios.items():
        for agent_name, ts_dict in agents.items():
            times_sorted = sorted(ts_dict.keys())
            per_obj: Dict[Any, List[Tuple[float, List[float], Any]]] = defaultdict(list)
            for t in times_sorted:
                rec = ts_dict[t]
                calib = rec.get("calibration", {})
                labels = rec.get("labels", []) or []
                for lab in labels:
                    if not use_global:
                        state, cat = to_world(lab, calib)
                    else:
                        state = [
                            float(lab["x"]),
                            float(lab["y"]),
                            float(lab["z"]),
                            float(lab["yaw"]),
                        ]
                        cat = lab["label"]
                    per_obj[lab["obj_id"]].append((float(t), state, cat))
            
            # For each object, we create contiguous segments and then samples
            for obj_id, entries in per_obj.items():
                entries.sort(key=lambda e: e[0])
                times = [e[0] for e in entries]
                states = [e[1] for e in entries]
                cats = [e[2] for e in entries]  # should be constant
                cat_id = cats[0] if cats else None
                
                segs = split_contiguous(times, fps=fps, tol=1.5)

                for seg in segs:
                    if len(seg) < (L + H): continue
                    seg_states = np.array([states[i] for i in seg], dtype=np.float32)  # [M,4]
                    seg_times = [times[i] for i in seg]
                    M = len(seg)
                    idx = 0
                    
                    while (idx + L + H) <= M:
                        obs = seg_states[idx: idx + L, :]
                        fut = seg_states[idx + L: idx + L + H, :]
                        sample = {
                            "obs": obs, 
                            "target": fut,
                            "cat": cat_id,
                            "meta": {
                                "scenario": scenario_name,
                                "agent": agent_name,
                                "obj_id": obj_id,
                                "track_id": f"{scenario_name}#{agent_name}#{obj_id}",
                                "t0": seg_times[idx],
                                "t1": seg_times[idx + L - 1],
                                "t_end": seg_times[idx + L + H - 1],
                            }
                        }

                        samples.append(sample)
                        idx += stride
    return samples


def save_pickle(data, mode, out_dir):
    out_file = Path(out_dir) / f"{mode}.pkl" 
    with open(out_file, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create train/valid/test trajectory data with pattern clustering (train only).")
    ap.add_argument("prefix", type=str, help="Dataset name (prefix).")
    ap.add_argument("input", type=str, help="Path to unified dataset pickle (with 'scenarios').")
    ap.add_argument("mode", choices=["train", "valid", "test"])
    ap.add_argument("--L", type=int, default=20, help="Observation length.")
    ap.add_argument("--H", type=int, default=30, help="Prediction horizon.")
    ap.add_argument("--stride", type=int, default=3, help="Sliding window stride.")
    ap.add_argument("--fps", type=float, default=10.0, help="Frame rate (Hz).")
    ap.add_argument("--global", dest="use_global", action="store_true", help="If set, do not transform using calibration.")
    args = ap.parse_args()

    data_path = Path(args.input)
    mode = args.mode
    
    samples = build_samples(
        data_path,
        L=args.L,
        H=args.H,
        stride=args.stride,
        fps=args.fps,
        use_global=args.use_global,
    )
    if not samples:
        raise SystemExit("No samples were produced. Check input pickle and windowing parameters.")

    print(f"Total number of samples {len(samples)}")
    out_dir = Path("data") / f"{args.prefix}_L{args.L}_H{args.H}_S{args.stride}_F{int(args.fps)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples_to_save = [{"obs": s["obs"], "target": s["target"], "cat": s["cat"]} for s in samples]
    save_pickle(samples_to_save, args.mode, out_dir)
