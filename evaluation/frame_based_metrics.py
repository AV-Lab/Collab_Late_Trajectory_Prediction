import numpy as np
from typing import Dict, List, Tuple, Any

# ------------------------------ config ------------------------------ #
TRAJ_SUCCESS_THRESHOLD = 0.5  # same units as FDE (e.g., meters)

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
        arr = arr[:, :k] if arr.shape[1] > k else np.pad(arr, ((0, 0), (0, k - arr.shape[1])), mode="edge")
    return arr


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
            "category": str(m.get("category", "unknown")).lower(),
        })

    return np.asarray(ade_vals, dtype=np.float32), np.asarray(fde_vals, dtype=np.float32), stats


def compute_by_category_statistics(stats: List[Dict], missed: List[Dict], false_pos: List[Dict]) -> Dict:
    """
    Per-category aggregates.

    Adds:
      - miss_rate_cat = num_missed / (num_matched + num_missed)
      - trajectory_success_rate_cat (TSR) = num_success_tau / (num_matched + num_missed)
        where num_success_tau counts matched objects with FDE <= traj_success_threshold
    """
    by_cat: Dict[str, Dict] = {}

    def cat_entry(cat: str):
        cat = str(cat).lower()
        return by_cat.setdefault(cat, {
            "ADE_vals": [], "FDE_vals": [],
            "num_matched": 0,
            "num_missed": 0,
            "num_false_positives": 0,
            # NEW for TSR/MR
            "num_success_tau": 0,
            "N_gt": 0,  # = matched + missed
        })

    # matched (from label_stats)
    for m in stats:
        cat = m.get("category", "unknown")
        ent = cat_entry(cat)

        ade_v = m.get("ADE", float("nan"))
        fde_v = m.get("FDE", float("nan"))

        ent["ADE_vals"].append(ade_v)
        ent["FDE_vals"].append(fde_v)
        ent["num_matched"] += 1

        if np.isfinite(fde_v) and float(fde_v) <= TRAJ_SUCCESS_THRESHOLD:
            ent["num_success_tau"] += 1

    # missed GT objects
    for g in missed:
        cat_entry(g.get("category", "unknown"))["num_missed"] += 1

    # false positives (kept as count; not in GT denominators)
    for fp in false_pos:
        cat_entry(fp.get("category", "unknown"))["num_false_positives"] += 1

    # finalize per-category means + MR/TSR
    for cat, ent in by_cat.items():
        ent["ADE_mean"] = float(np.mean(ent["ADE_vals"])) if ent["ADE_vals"] else float("nan")
        ent["FDE_mean"] = float(np.mean(ent["FDE_vals"])) if ent["FDE_vals"] else float("nan")
        ent.pop("ADE_vals")
        ent.pop("FDE_vals")

        ent["N_gt"] = int(ent["num_matched"] + ent["num_missed"])
        denom = ent["N_gt"]

        ent["miss_rate"] = float(ent["num_missed"] / denom) if denom > 0 else float("nan")
        ent["trajectory_success_rate"] = float(ent["num_success_tau"] / denom) if denom > 0 else float("nan")
        ent["traj_success_threshold"] = TRAJ_SUCCESS_THRESHOLD

    return by_cat


def _match_by_id(gid, pred_id_index):
    if gid in pred_id_index:
        return pred_id_index[gid][0]
    return None


def _match_by_iou(gt_obj, pred_id_index, iou_th):
    gt_box = axis_aligned_bbox(gt_obj["current_state"])
    best_i, best_scr = None, 0.0
    for k, v in pred_id_index.items():
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
        gt_future_vec = to_vec(gt_obj.get("future", {}), input_dimension, hz)

        pl = min(past_len, len(gt_obj.get("past", [])))
        gt_past_vec = to_vec(gt_obj.get("past", []), input_dimension, pl, reverse=True)

        match_idx = _match_by_id(gid, pred_id_index) if use_id else _match_by_iou(gt_obj, pred_id_index, iou_th)

        if match_idx is not None:
            used_pred_idx.add(match_idx)
            p = predictions[match_idx]
            id_ = p["id"]

            # Ensure category is always defined (default "unknown")
            category = gt_obj.get("category", p.get("category", "unknown"))
            category = str(category).lower()
            if "category" not in gt_obj:
                gt_obj["category"] = category
            if "category" not in p:
                p["category"] = category

            # Handle ego vs fused predictions
            pred_dict = p.get("prediction")
            fused_dict = p.get("fused_prediction")

            ego_means_dict = pred_dict.get("pred") if pred_dict is not None else None
            ego_cov_dict   = pred_dict.get("cov")  if pred_dict is not None else None

            fused_means_dict = fused_dict.get("pred") if fused_dict is not None else None
            fused_cov_dict   = fused_dict.get("cov")  if fused_dict is not None else None

            has_ego = isinstance(ego_means_dict, dict) and len(ego_means_dict) > 0
            has_fused = isinstance(fused_means_dict, dict) and len(fused_means_dict) > 0

            ego_future_vec = to_vec(ego_means_dict, input_dimension, hz) if has_ego else None
            fused_future_vec = to_vec(fused_means_dict, input_dimension, hz) if has_fused else None

            # PRIMARY prediction for overall metrics: fused if available, else ego
            if has_fused:
                means_dict = fused_means_dict
                cov_dict = fused_cov_dict
            elif has_ego:
                means_dict = ego_means_dict
                cov_dict = ego_cov_dict
            else:
                means_dict = {}
                cov_dict = None

            pred_vec = to_vec(means_dict, input_dimension, hz)

            timestamp = p["timestamp"]
            pred_box = p["cur_location"]
            past_vec = to_vec(past_id_trck[id_], input_dimension, pl, reverse=True) if id_ in past_id_trck else None

            matched_raw.append({
                "gid": gid,
                "match_idx": match_idx,
                "hz": hz,
                "gt_past": gt_past_vec,
                "gt_future": gt_future_vec,
                "gt_bbox": gt_obj["current_state"],
                "trk_past": past_vec,
                "pred_future": pred_vec,     # PRIMARY
                "bbox": pred_box,
                "means_dict": means_dict,    # PRIMARY dict
                "cov_dict": cov_dict,        # PRIMARY cov
                "timestamp": timestamp,
                "category": category,
                # extra info for source-specific metrics
                "has_ego": has_ego,
                "has_fused": has_fused,
                "ego_future": ego_future_vec,
                "fused_future": fused_future_vec,
            })
        else:
            # Ensure missed has explicit category
            category = gt_obj.get("category", "unknown")
            category = str(category).lower()
            if "category" not in gt_obj:
                gt_obj["category"] = category

            missed_raw.append({
                "id": gid,
                "category": category,
                "gt_past": gt_past_vec,
                "gt_bbox": gt_obj["current_state"],
                "gt_future": gt_future_vec
            })

    for i, p in enumerate(predictions):
        if i in used_pred_idx:
            continue
        id_ = p["id"]
        pred_dict = p["prediction"]

        category = p.get("category", "unknown")
        category = str(category).lower()
        if "category" not in p:
            p["category"] = category

        fp_past_vec = to_vec(past_id_trck[id_], input_dimension, past_len, reverse=True) if id_ in past_id_trck else None

        false_pos_raw.append({
            "id": id_,
            "category": category,
            "bbox": p["cur_location"],
            "means_dict": pred_dict.get("pred", {}),
            "cov_dict": pred_dict.get("cov", None),
            "timestamp": pred_dict.get("timestamp", None),
            "trk_past": fp_past_vec,
        })

    # ------------ metrics (ADE/FDE/MSNE + TSR) ------------
    matched_for_metrics = [{
        "gid": m["gid"],
        "future": m["pred_future"],
        "gt_future": m["gt_future"],
        "category": m["category"]
    } for m in matched_raw]

    ade_vals, fde_vals, label_stats = compute_label_metrics(matched_for_metrics)

    # MSNE (assuming diagonal-only cov) using PRIMARY cov & pred
    msne_vals = []
    eps = 1e-9
    for m in matched_raw:
        cov = m["cov_dict"]
        if not cov:
            continue
        ts = sorted(m["means_dict"].keys(), key=float)[:m["hz"]]
        var = np.clip(np.array([
            np.diag(cov[t]) if np.asarray(cov[t]).ndim == 2 else np.asarray(cov[t])
            for t in ts
        ], dtype=np.float32), eps, None)  # (T,2)

        diff = (m["gt_future"][:len(ts)] - m["pred_future"][:len(ts)]).astype(np.float32)
        msne_vals.append(float(np.mean((diff * diff) / var)))

    # NEW: trajectory success rate (TSR) using FDE and consistent N over GT objects
    N_gt = len(matched_raw) + len(missed_raw)  # total GT objects this frame

    # ---- NEW (for histogram/CDF sanity checks): store raw FDE distribution + success counts ----
    fde_list = []
    num_valid_fde = 0
    num_success_tau = 0
    tsr_matched = float("nan")  # success among matched only (diagnostic)

    if len(fde_vals) > 0:
        fde_arr = np.asarray(fde_vals, dtype=np.float32)
        valid = np.isfinite(fde_arr)
        fde_valid = fde_arr[valid]
        fde_list = fde_valid.astype(np.float32).tolist()
        num_valid_fde = int(fde_valid.shape[0])
        num_success_tau = int(np.sum(fde_valid <= TRAJ_SUCCESS_THRESHOLD))
        if num_valid_fde > 0:
            tsr_matched = float(num_success_tau / num_valid_fde)

    if N_gt > 0 and num_valid_fde > 0:
        traj_success_rate = float(num_success_tau / N_gt)
    else:
        traj_success_rate = float("nan")
    # ------------------------------------------------------------------------------------------

    overall = {
        "num_matched": len(matched_raw),
        "num_missed": len(missed_raw),
        "num_false_positives": len(false_pos_raw),
        "N_gt": int(N_gt),  # NEW: explicit denominator used in TSR

        "ADE_mean": float(np.mean(ade_vals)) if len(ade_vals) else float("nan"),
        "FDE_mean": float(np.mean(fde_vals)) if len(fde_vals) else float("nan"),
        "MSNE_mean": float(np.mean(msne_vals)) if len(msne_vals) else float("nan"),

        "trajectory_success_rate": traj_success_rate,               # TSR_tau over GT (matches your current definition)
        "traj_success_threshold": float(TRAJ_SUCCESS_THRESHOLD),

        # NEW: distribution + counts for histogram/CDF checks
        "FDE_list": fde_list,                                       # per-matched-object FDEs (valid only)
        "num_valid_fde": int(num_valid_fde),                        # = len(FDE_list)
        "num_success_tau": int(num_success_tau),                    # count(FDE <= tau) among matched
        "trajectory_success_rate_matched": float(tsr_matched),      # diagnostic: successes / matched

        "dim": input_dimension,
    }

    by_cat = compute_by_category_statistics(label_stats, missed_raw, false_pos_raw)

    # ------------ separate errors by source (Cat I, Cat II, no fusion) ------------
    cat1_ego_ade, cat1_ego_fde = [], []
    cat1_fused_ade, cat1_fused_fde = [], []
    cat2_fused_ade, cat2_fused_fde = [], []
    nofusion_ade, nofusion_fde = [], []

    for m in matched_raw:
        gt = m["gt_future"]
        has_ego = m["has_ego"]
        has_fused = m["has_fused"]
        ego_future = m["ego_future"]
        fused_future = m["fused_future"]

        def _pair_errors(pred_arr, gt_arr):
            if pred_arr is None or len(pred_arr) == 0 or len(gt_arr) == 0:
                return None, None
            L = min(len(pred_arr), len(gt_arr))
            return ade(pred_arr[:L], gt_arr[:L]), fde(pred_arr[:L], gt_arr[:L])

        ego_ade, ego_fde = _pair_errors(ego_future, gt) if has_ego else (None, None)
        fused_ade, fused_fde = _pair_errors(fused_future, gt) if has_fused else (None, None)

        if has_ego and has_fused:
            if ego_ade is not None:
                cat1_ego_ade.append(ego_ade)
                cat1_ego_fde.append(ego_fde)
            if fused_ade is not None:
                cat1_fused_ade.append(fused_ade)
                cat1_fused_fde.append(fused_fde)
        elif (not has_ego) and has_fused:
            if fused_ade is not None:
                cat2_fused_ade.append(fused_ade)
                cat2_fused_fde.append(fused_fde)
        elif has_ego and (not has_fused):
            if ego_ade is not None:
                nofusion_ade.append(ego_ade)
                nofusion_fde.append(ego_fde)

    def _mean_or_nan(vals):
        return float(np.mean(vals)) if vals else float("nan")

    by_source = {
        "category_I": {
            "ego": {
                "num": len(cat1_ego_ade),
                "ADE_mean": _mean_or_nan(cat1_ego_ade),
                "FDE_mean": _mean_or_nan(cat1_ego_fde),
            },
            "fused": {
                "num": len(cat1_fused_ade),
                "ADE_mean": _mean_or_nan(cat1_fused_ade),
                "FDE_mean": _mean_or_nan(cat1_fused_fde),
            },
        },
        "category_II": {
            "fused": {
                "num": len(cat2_fused_ade),
                "ADE_mean": _mean_or_nan(cat2_fused_ade),
                "FDE_mean": _mean_or_nan(cat2_fused_fde),
            },
        },
        "no_fusion": {
            "num": len(nofusion_ade),
            "ADE_mean": _mean_or_nan(nofusion_ade),
            "FDE_mean": _mean_or_nan(nofusion_fde),
        },
    }
    # ------------------------------------------------------------------------

    # For visualization (unchanged, still uses PRIMARY prediction)
    matched_viz = []
    for m in matched_raw:
        matched_viz.append({
            "id": m["gid"],
            "category": m["category"],
            "gt": {
                "past": m["gt_past"],
                "bbox": m["gt_bbox"],
                "future": m["gt_future"],
            },
            "pred": {
                "past": m["trk_past"],
                "bbox": m["bbox"],
                "future": m["means_dict"],
                "cov": m["cov_dict"],
                "timestamp": m["timestamp"],
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
                "future": g["gt_future"],
            }
        })

    false_pos_viz = []
    for fp in false_pos_raw:
        false_pos_viz.append({
            "id": fp["id"],
            "category": fp["category"],
            "pred": {
                "past": fp["trk_past"],
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
    metrics = {"overall": overall, "by_cat": by_cat, "by_source": by_source}
    return forecasts, metrics
