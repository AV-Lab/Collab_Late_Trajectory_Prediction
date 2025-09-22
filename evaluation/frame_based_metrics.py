import numpy as np
from typing import Dict, List, Optional, Tuple

# ------------------------------ geometry & metrics ------------------------------ #

def axis_aligned_bbox(box7d):
    """
    Convert 7-DoF box [x, y, z, dx, dy, dz, yaw] → 2D axis-aligned (xmin, ymin, xmax, ymax).
    Note: ignores yaw by design (AABB).
    """
    x, y, z, dx, dy, dz = box7d[:6]
    dx *= 0.5
    dy *= 0.5
    return x - dx, y - dy, x + dx, y + dy


def iou(r1, r2):
    """
    IoU for two axis-aligned rectangles (xmin, ymin, xmax, ymax).
    """
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
    """Final displacement error at the last step (L2 on last)."""
    if len(pred) == 0:
        return float("nan")
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def to_vec(obj, k: int, hz: int, reverse: bool = False) -> np.ndarray:
    """
    Pack irregular past/future into dense (hz, k); pad missing rows with zeros.
    Accepts:
      - dict {ts → [D]}  (we will take first k dims)
      - sequence of objects/tuples with .x/.y/.z or indexable [0:3]
    """
    out = np.zeros((hz, k), dtype=np.float32)

    if isinstance(obj, dict):
        ts_sorted = sorted(obj.keys())
        sel = ts_sorted[-hz:] if reverse else ts_sorted[:hz]
        rows = [np.asarray(obj[t], dtype=np.float32)[:k] for t in sel]
    else:
        seq = obj[-hz:] if reverse else obj[:hz]
        rows = []
        for p in seq:
            if hasattr(p, "x"):
                rows.append([p.x, p.y, p.z][:k])
            else:
                rows.append(list(p)[:k])

    rows = np.asarray(rows, dtype=np.float32)
    if reverse:
        out[-len(rows):] = rows
    else:
        out[:len(rows)] = rows
    return out


def extract_parameters(tracklets, pred_means: List[Dict[float, List[float]]]) -> Tuple[int, int, int, int]:
    """
    Infer (sampling, horizon_steps, input_dimension, pred_len) from prediction MEANS.

    Supports:
      - dict form: pred_means[i] = {t: [D]}
      - array form: pred_means[i] = ndarray of shape (T, D)

    If timestamps are absent (array form or 1 timestamp), assume dt = 0.1s (10 Hz).
    """
    pred_len = len(pred_means[0])
    input_dimension = len(pred_means[0][0.0])
    t = np.array(sorted(pred_means[0].keys()), dtype=np.float32)
    dt = float(np.median(np.diff(t)))
    sampling = int(round(1.0 / dt))
    horizon = int(np.ceil(dt * (pred_len - 1)))
    return sampling, horizon, input_dimension, pred_len

# ------------------------------ matching utils ------------------------------ #

def find_matching_tracklet_by_id(gid, tracklets):
    for i, trk in enumerate(tracklets):
        if trk["id"] == gid:
            return i
    return None


def find_matching_tracklet_by_iou(gt_obj, used_trk, iou_th, tracklets):
    gt_box = axis_aligned_bbox(gt_obj["current_state"])
    best_iou = 0.0
    best_idx = None

    for i, trk in enumerate(tracklets):
        if i in used_trk:
            continue
        trk_box = axis_aligned_bbox(trk["current_pos"])
        scr = iou(gt_box, trk_box)
        if scr > best_iou:
            best_iou = scr
            best_idx = i

    if best_iou >= iou_th:
        return best_idx
    return None

# ------------------------------ evaluation helpers ------------------------------ #

def compute_label_metrics(matched: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Compute per-object ADE/FDE on matched items.
    Returns:
      ade_list: (N,) of ADE
      fde_list: (N,) of FDE
      stats   : list of dicts per matched object (metrics + metadata) for the class evaluator
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
    """
    Build per-category aggregates with counts and mean ADE/FDE across matched items.
    """
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

# ------------------------------ main evaluation ------------------------------ #

def compute_frame_based_performance(
    predictions,
    tracklets: List[Dict],
    gt_predictions: Dict,         # {gid: {"past": dict/seq, "future": dict/seq, "current_state": [7], "category": str}}
    use_id: bool = False,
    iou_th: float = 0.75,
):
    """
    predictions:
        - with uncertainty: (pred_means, pred_covs)
            pred_means[i] : dict {time: [D]}        # mean positions
            pred_covs[i]  : dict {time: [D,D]}      # per-step covariance
        - without uncertainty: pred_means
            pred_means[i] : dict {time: [D]}        # MSNE will be NaN

    Returns:
        forecasts, metrics  (metrics = {"overall": ..., "by_cat": ...})
    """

    if isinstance(predictions, tuple):
        pred_means, pred_covs = predictions
    else:
        pred_means, pred_covs = predictions, None

    sampling, horizon_steps, input_dimension, pred_len = extract_parameters(tracklets, pred_means)
    matched, missed, false_pos, used_trk = [], [], [], set()

    for gid, gt_obj in gt_predictions.items():
        hz = min(pred_len, len(gt_obj["future"]))

        # match by id or iou
        match_idx = (find_matching_tracklet_by_id(gid, tracklets) if use_id
                     else find_matching_tracklet_by_iou(gt_obj, used_trk, iou_th, tracklets))
        
        # Build GT future vector
        gt_future = to_vec(gt_obj["future"], input_dimension, hz)

        if match_idx is not None:
            used_trk.add(match_idx)

            # Past length: align GT past and tracker past to SAME length (most-recent first)
            past_len = min(len(gt_obj["past"]), len(tracklets[match_idx]["tracklet"]))
            gt_past_vec  = to_vec(gt_obj["past"], input_dimension, past_len, reverse=True)
            trk_past_vec = to_vec(tracklets[match_idx]["tracklet"], input_dimension, past_len, reverse=True)

            # Prediction means vector aligned to hz
            pred_vec = to_vec(pred_means[match_idx], input_dimension, hz)

            matched.append({
                "gid": gid,
                "match_idx": match_idx,
                "hz": hz,
                "gt_past": gt_past_vec,               # [past_len, D]
                "tracklet": trk_past_vec,             # [past_len, D]
                "gt_future": gt_future,               # [hz, D]
                "future": pred_vec,                   # [hz, D]
                "bbox": gt_obj["current_state"],      # use GT bbox as requested
                "category": gt_obj.get("category", "unknown"),
                "confidence": tracklets[match_idx].get("confidence", 1.0),
            })
        else:
            # Count as missing when GT has no matching prediction/track
            missed.append({
                "id": gid,
                "gt_past": to_vec(gt_obj["past"], input_dimension, len(gt_obj["past"]), reverse=True),
                "gt_bbox": gt_obj["current_state"],
                "gt_future": gt_future,
                "category": gt_obj.get("category", "unknown"),
            })

    # ---------------- false positives (unmatched prediction tracks) ---------------- #
    for i, trk in enumerate(tracklets):
        if i not in used_trk:
            # Use all available steps of predicted means
            pred_vec = to_vec(pred_means[i], input_dimension, pred_len)
            false_pos.append({
                "id": trk["id"],
                "tracklet": to_vec(trk["tracklet"], input_dimension, len(trk["tracklet"]), reverse=True),
                "bbox": trk["current_pos"],
                "future": pred_vec,
                "category": trk.get("category", "unknown"),
                "confidence": trk.get("confidence", 1.0),
            })

    # ---------------- per-object ADE/FDE ---------------- #
    ade_vals, fde_vals, label_stats = compute_label_metrics(matched)

    # ---------------- MSNE (only if covariances provided) ---------------- #
    msne_vals = []
    eps = 1e-9
    if pred_covs is not None and len(matched) > 0:
        for m in matched:
            idx = m["match_idx"]
            hz  = m["hz"]
            gt  = m["gt_future"]     # [hz, D]
            mu  = m["future"]        # [hz, D]

            cov_dict = pred_covs[idx]                       # {t: [D,D]}
            ts_sorted = sorted(pred_means[idx].keys())[:hz] # same ordering used by to_vec on means
            cov_seq = np.array([cov_dict[t] for t in ts_sorted], dtype=np.float32)  # [hz, D, D]

            var = np.clip(np.diagonal(cov_seq, axis1=1, axis2=2), eps, None)        # [hz, D]
            z2 = ((gt - mu) ** 2) / var                                             # [hz, D]
            msne_vals.append(float(np.mean(z2)))

    # ---------------- overall & by-category ---------------- #
    overall = {
        "num_matched": len(matched),
        "num_missed": len(missed),
        "num_false_positives": len(false_pos),
        "ADE_mean": float(np.mean(ade_vals)) if ade_vals.size > 0 else float("nan"),
        "FDE_mean": float(np.mean(fde_vals)) if fde_vals.size > 0 else float("nan"),
        "MSNE_mean": float(np.mean(msne_vals)) if len(msne_vals) > 0 else float("nan"),
        "sampling_hz": sampling,
        "horizon_steps": horizon_steps,
        "dim": input_dimension,
    }
    by_cat = compute_by_category_statistics(label_stats, missed, false_pos)

    # ---------------- outputs ---------------- #
    forecasts = {
        "matched": matched,
        "missed": missed,
        "false_positives": false_pos,
        "label_metrics": label_stats,   # for class evaluator accumulation
    }
    metrics = {
        "overall": overall,
        "by_cat": by_cat,
    }
    return forecasts, metrics
