import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# ------------------------------ geometry & metrics ------------------------------ #

def axis_aligned_bbox(box7d):
    """7-DoF box [x,y,z,dx,dy,dz,yaw] → 2D axis-aligned (xmin,ymin,xmax,ymax)."""
    x, y, z, dx, dy, dz = box7d[:6]
    dx *= 0.5
    dy *= 0.5
    return x - dx, y - dy, x + dx, y + dy


def iou(r1, r2):
    """IoU for two AABBs (xmin,ymin,xmax,ymax)."""
    xa1, ya1, xa2, ya2 = r1
    xb1, yb1, xb2, yb2 = r2
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-9)


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average displacement error over T steps (L2 per step, mean)."""
    return float(np.mean(np.linalg.norm(pred - gt, axis=1))) if len(pred) > 0 else float("nan")


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final displacement error (L2 at last step)."""
    if len(pred) == 0:
        return float("nan")
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def to_vec(obj: Any, k: int, hz: int, reverse: bool = False) -> np.ndarray:
    """
    Convert past/future into a dense array with NO zero padding.
    Returns exactly the available rows (≤ hz), each clipped to k dims.
    If nothing is available, returns shape (0, k).
    """
    def _clip(v):
        a = np.asarray(v, dtype=np.float32)
        return a[:k] if a.ndim == 1 else a.ravel()[:k]

    if obj is None:
        return np.empty((0, k), dtype=np.float32)

    rows = []

    if isinstance(obj, dict):
        ts = sorted(obj.keys(), key=float)
        ts = (ts[-hz:] if reverse else ts[:hz]) if hz > 0 else ts
        for t in ts:
            rows.append(_clip(obj[t]))
    else:
        seq = list(obj)
        seq = (seq[-hz:] if reverse else seq[:hz]) if hz > 0 else seq
        for p in seq:
            if hasattr(p, "x"):
                rows.append(_clip([p.x, p.y, getattr(p, "z", 0.0)]))
            else:
                rows.append(_clip(p))

    if not rows:
        return np.empty((0, k), dtype=np.float32)

    arr = np.vstack(rows).astype(np.float32)
    # ensure second dim is exactly k (truncate if longer)
    if arr.shape[1] != k:
        arr = arr[:, :k] if arr.shape[1] > k else np.pad(arr, ((0,0),(0,k-arr.shape[1])), mode='edge')
    return arr


def head_dict(d: Dict, n: int) -> Dict:
    """Return an ordered (by time) dict with at most the first n keys of d."""
    if not isinstance(d, dict) or n <= 0:
        return {}
    ts = sorted(d.keys(), key=float)[:n]
    return {t: d[t] for t in ts}


def compute_label_metrics(matched: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Compute per-object ADE/FDE on matched items.
    """
    ade_vals, fde_vals, stats = [], [], []

    for m in matched:
        pred_vec = m["future"]      # [hz, D]
        gt_vec   = m["gt_future"]   # [hz, D]
        ade_v = ade(pred_vec, gt_vec)
        fde_v = fde(pred_vec, gt_vec)

        ade_vals.append(ade_v)
        fde_vals.append(fde_v)

        stats.append({
            "id": m["gid"],
            "ADE": ade_v,
            "FDE": fde_v,
            "confidence": m.get("confidence", 1.0),
            "category": m.get("category", "unknown")
        })

    return np.asarray(ade_vals, dtype=np.float32), np.asarray(fde_vals, dtype=np.float32), stats


def compute_by_category_statistics(stats: List[Dict], missed: List[Dict], false_pos: List[Dict]) -> Dict:
    """Per-category aggregates."""
    by_cat: Dict[str, Dict] = {}

    def cat_entry(cat):
        return by_cat.setdefault(cat, {
            "ADE_vals": [], "FDE_vals": [],
            "num_matched": 0, "num_missed": 0, "num_false_positives": 0
        })

    for m in stats:
        ent = cat_entry(m["category"])
        ent["ADE_vals"].append(m["ADE"])
        ent["FDE_vals"].append(m["FDE"])
        ent["num_matched"] += 1

    for g in missed:
        cat_entry(g.get("category", "unknown"))["num_missed"] += 1

    for fp in false_pos:
        cat_entry(fp.get("category", "unknown"))["num_false_positives"] += 1

    for cat, ent in by_cat.items():
        ent["ADE_mean"] = float(np.mean(ent["ADE_vals"])) if ent["ADE_vals"] else float("nan")
        ent["FDE_mean"] = float(np.mean(ent["FDE_vals"])) if ent["FDE_vals"] else float("nan")
        ent.pop("ADE_vals")
        ent.pop("FDE_vals")

    return by_cat


def _match_by_id(gid, pred_id_index):
    if gid in pred_id_index:
        return pred_id_index[gid][0]
    return None

def _match_by_iou(gt_obj, pred_id_index, iou_th):
    gt_box = axis_aligned_bbox(gt_obj["current_state"])
    best_i, best_scr = None, 0.0
    for k,v in pred_id_index.items():
        trk_box = axis_aligned_bbox(v[1])
        scr = iou(gt_box, trk_box)
        if scr > best_scr:
            best_scr, best_i = scr, v[0]
    return best_i if best_scr >= iou_th else None

def compute_frame_based_performance(
    predictions: List[Dict],
    tracklets,
    gt_predictions: Dict[int, Dict],
    input_dimension: int,
    pred_len: int,
    past_len: int,
    iou_th: float = 0.7,
    use_id: bool = False
):
    
    pred_id_index = {p["id"]: (i, p["cur_location"]) for i, p in enumerate(predictions)}
    past_id_trck = {t["id"]: t["tracklet"] for t in tracklets}

    used_pred_idx = set()
    matched_raw, missed_raw, false_pos_raw = [], [], []

    for gid, gt_obj in gt_predictions.items():
        hz = min(pred_len, len(gt_obj.get("future", {})))
        gt_future_vec  = to_vec(gt_obj.get("future", {}), input_dimension, hz)

        pl = min(past_len, len(gt_obj.get("past", [])))
        gt_past_vec = to_vec(gt_obj.get("past", []), input_dimension, pl, reverse=True)

        match_idx = _match_by_id(gid, pred_id_index) if use_id else _match_by_iou(gt_obj, pred_id_index, iou_th)
        
        if match_idx is not None:
            used_pred_idx.add(match_idx)
            p = predictions[match_idx]
            id_ = p["id"]
            pred_dict = p["prediction"]
            means_dict  = pred_dict["pred"]
            cov_dict    = pred_dict["cov"]
            timestamp   = pred_dict["timestamp"]
            pred_box    = p["cur_location"]
            pred_vec = to_vec(means_dict, input_dimension, hz)
            past_vec = to_vec(past_id_trck[id_], input_dimension, pl, reverse=True)
            
            matched_raw.append({
                "gid": gid,
                "match_idx": match_idx,
                "hz": hz,
                "gt_past": gt_past_vec,
                "gt_future": gt_future_vec,              
                "gt_bbox": gt_obj["current_state"],
                "trk_past": past_vec,
                "pred_future": pred_vec,                 
                "bbox": pred_box,                   
                "means_dict": means_dict,                
                "cov_dict": cov_dict,
                "timestamp": timestamp,
                "category": gt_obj.get("category", p.get("category", "unknown"))
            })
        else:
            missed_raw.append({
                "id": gid,
                "category": gt_obj.get("category", "unknown"),
                "gt_past": gt_past_vec,
                "gt_bbox": gt_obj["current_state"],
                "gt_future": gt_future_vec
            })

    for i, p in enumerate(predictions):
        if i in used_pred_idx:
            continue
        id_ = p["id"]
        pred_dict = p["prediction"]
        fp_past_vec = to_vec(past_id_trck[id_], input_dimension, past_len, reverse=True)

        false_pos_raw.append({
            "id": id_,
            "category": p.get("category", "unknown"),
            "bbox": p["cur_location"],
            "means_dict": pred_dict.get("pred", {}),
            "cov_dict": pred_dict.get("cov", None),
            "timestamp": pred_dict.get("timestamp", None),
            "trk_past": fp_past_vec,                 
        })

    # ------------ 3) metrics (ADE/FDE/MSNE) ------------
    matched_for_metrics = [{
        "gid": m["gid"],
        "future": m["pred_future"],
        "gt_future": m["gt_future"],
        "category": m["category"]
    } for m in matched_raw]

    ade_vals, fde_vals, label_stats = compute_label_metrics(matched_for_metrics)
    
    # MSNE (assuming diagonal-only cov)
    msne_vals = []
    eps = 1e-9
    for m in matched_raw:
        cov = m["cov_dict"]
        if not cov:  continue
        ts = sorted(m["means_dict"].keys(), key=float)[:m["hz"]]
        var = np.clip(np.array([
            np.diag(cov[t]) if np.asarray(cov[t]).ndim == 2 else np.asarray(cov[t])
            for t in ts
        ], dtype=np.float32), eps, None)                        # (T,2)
        diff = (m["gt_future"][:len(ts)] - m["pred_future"][:len(ts)]).astype(np.float32)
        msne_vals.append(float(np.mean((diff * diff) / var)))

    overall = {
        "num_matched": len(matched_raw),
        "num_missed": len(missed_raw),
        "num_false_positives": len(false_pos_raw),
        "ADE_mean": float(np.mean(ade_vals)) if len(ade_vals) else float("nan"),
        "FDE_mean": float(np.mean(fde_vals)) if len(fde_vals) else float("nan"),
        "MSNE_mean": float(np.mean(msne_vals)) if len(msne_vals) else float("nan"),
        "dim": input_dimension,
    }
    by_cat = compute_by_category_statistics(label_stats, missed_raw, false_pos_raw)

    ########## For visualization 
    matched_viz = []
    for m in matched_raw:
        matched_viz.append({
            "id": m["gid"],
            "category": m["category"],
            "gt": {
                "past": m["gt_past"],                  # np.ndarray [past_cap, D]
                "bbox": m["gt_bbox"],                  # [7]
                "future": m["gt_future"],         # dict capped to pred_len
            },
            "pred": {
                "past": m["trk_past"],                # NEW: np.ndarray or None
                "bbox": m["bbox"],                # [7]
                "future": m["means_dict"],             # dict{t:[x,y]} (visualizer uses dict)
                "cov": m["cov_dict"],                  # dict{t:[[2x2]]} or None
                "timestamp": m["timestamp"],           # int or None
            }
        })

    missed_viz = []
    for g in missed_raw:
        missed_viz.append({
            "id": g["id"],
            "category": g["category"],
            "gt": {
                "past": g["gt_past"],
                "bbox": g["gt_bbox"],
                "future": g["gt_future"],         # capped to pred_len
            }
        })

    false_pos_viz = []
    for fp in false_pos_raw:
        false_pos_viz.append({
            "id": fp["id"],
            "category": fp["category"],
            "pred": {
                "past": fp["trk_past"],               # NEW: np.ndarray or None
                "bbox": fp["bbox"],
                "future": fp["means_dict"],
                "cov": fp["cov_dict"],
                "timestamp": fp["timestamp"],
            }
        })

    forecasts = {
        "matched": matched_viz,
        "missed": missed_viz,
        "false_positives": false_pos_viz,
        "label_metrics": label_stats,
    }
    metrics = {"overall": overall, "by_cat": by_cat}
    return forecasts, metrics
