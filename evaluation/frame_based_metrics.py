#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 17:01:37 2025

@author: nadya
"""


import numpy as np
from typing import Sequence  # (optional; only used in docstrings)

# ------------------------------ geometry & metrics ------------------------------ #

def axis_aligned_bbox(box7d):
    """
    Convert 7-DoF box [x, y, z, dx, dy, dz, yaw] → 2D axis-aligned (xmin, ymin, xmax, ymax).
    """
    x, y, z, dx, dy, dz, yaw = box7d
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
    if inter == 0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-9)


def ade(pred, gt):
    """Average displacement error over T steps."""
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def fde(pred, gt):
    """Final displacement error at the last step."""
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def to_vec(obj, k, hz, reverse=False):
    """
    Pack irregular past/future into dense (hz, k); pad missing rows with zeros.
    Accepts dict {ts → (x,y,z,…)} or an ordered sequence with .x/.y/.z attributes.
    """
    out = np.zeros((hz, k), dtype=np.float32)

    if isinstance(obj, dict):
        ts_sorted = sorted(obj)
        sel = ts_sorted[-hz:] if reverse else ts_sorted[:hz]
        rows = [obj[t][:k] for t in sel]
    else:
        seq = obj[-hz:] if reverse else obj[:hz]
        rows = [[getattr(p, c) for c in "xyz"[:k]] for p in seq]

    if reverse:
        out[-len(rows):] = np.asarray(rows, dtype=np.float32)
    else:
        out[:len(rows)] = np.asarray(rows, dtype=np.float32)

    return out


def extract_parameters(tracklets, predictions):
    pred_len = len(predictions[0])
    input_dimension = len(predictions[0][0.0])
    t = np.array(sorted(predictions[0].keys()), dtype=np.float32)
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


def predict_based_on_gt_past(vehicle, matched):
    predictions = []
    if matched:
        tracks = [m["gt_past"] for m in matched]
        tracks = vehicle.predictor.resample_input(tracks, vehicle.fps)
        predictions = vehicle.predictor.predict(tracks, vehicle.prediction_horizon, vehicle.prediction_sampling)
    return predictions

# ------------------------------ evaluation helpers ------------------------------ #

def compute_statistics(matched, tracklets, predictions, base_predictions, input_dimension):
    """
    Returns:
      ade_list: (N, 3) with [ade_track, ade_base, ΔADE]
      fde_list: (N, 3) with [fde_track, fde_base, ΔFDE]
      stats   : list of dicts per matched object (metrics + metadata)
    """
    ade_list, fde_list, stats = [], [], []

    for jdx, m in enumerate(matched):
        hz = m["hz"]
        category = m["category"]
        conf = m["confidence"]
        pred_vec = m["future"]
        gt_vec = m["gt_future"]
        base_vec = to_vec(base_predictions[jdx], input_dimension, hz)

        ade_track = ade(pred_vec, gt_vec)
        fde_track = fde(pred_vec, gt_vec)
        ade_base = ade(base_vec, gt_vec)
        fde_base = fde(base_vec, gt_vec)

        dADE = ade_track - ade_base
        dFDE = fde_track - fde_base

        ade_list.append([ade_track, ade_base, dADE])
        fde_list.append([fde_track, fde_base, dFDE])

        stats.append({
            "id": m["gid"],
            "ADE_t": ade_track,
            "FDE_t": fde_track,
            "ADE_p": ade_base,
            "FDE_p": fde_base,
            "ΔADE": dADE,
            "ΔFDE": dFDE,
            "confidence": conf,
            "category": category
        })

    return np.array(ade_list), np.array(fde_list), stats


def compute_by_category_statistics(stats, missed, false_pos):
    by_cat = {}

    def cat_entry(cat):
        return by_cat.setdefault(cat, {
            "ΔADE_vals": [], "ΔFDE_vals": [],
            "num_matched": 0, "num_missed": 0, "num_false_positives": 0
        })

    for m in stats:
        ent = cat_entry(m["category"])
        ent["ΔADE_vals"].append(m["ΔADE"])
        ent["ΔFDE_vals"].append(m["ΔFDE"])
        ent["num_matched"] += 1

    for g in missed:
        cat_entry(g["category"])["num_missed"] += 1

    for fp in false_pos:
        cat_entry(fp["category"])["num_false_positives"] += 1

    for cat, ent in by_cat.items():
        ent["ΔADE_mean"] = float(np.mean(ent["ΔADE_vals"])) if ent["ΔADE_vals"] else float("nan")
        ent["ΔFDE_mean"] = float(np.mean(ent["ΔFDE_vals"])) if ent["ΔFDE_vals"] else float("nan")
        ent.pop("ΔADE_vals")
        ent.pop("ΔFDE_vals")

    return by_cat

# ------------------------------ main eval ------------------------------ #

def predict_based_on_gt_past(vehicle, matched):
    """
    Returns:
        base_means : list[dict]   # [{t: [D]}, ...]
        base_covs  : list[dict]   # [{t: [D,D]}, ...]
    """
    base_means, base_covs = [], []
    if matched:
        tracks = [m["gt_past"] for m in matched]
        base_means, base_covs = vehicle.predictor.predict(tracks, vehicle.prediction_horizon)
    return base_means, base_covs


def compute_frame_based_performance(predictions, tracklets, gt_predictions, vehicle,
                                    use_id=False, iou_th=0.75):
    """
    predictions: tuple (pred_means, pred_covs)
        pred_means[i] : dict {time: [D]}        # mean positions
        pred_covs[i]  : dict {time: [D,D]}      # per-step position covariance
    """
    pred_means, pred_covs = predictions

    # derive dims/lengths from MEANS (not covs)
    sampling, horizon, input_dimension, pred_len = extract_parameters(tracklets, pred_means)
    matched, missed, false_pos, used_trk = [], [], [], set()

    # ---------------- match GT to tracker ---------------- #
    for gid, gt_obj in gt_predictions.items():
        hz = min(pred_len, len(gt_obj["future"]))
        category = gt_obj.get("category", "unknown")
        gt_past = to_vec(gt_obj["past"], input_dimension, vehicle.tracking_history, reverse=True)
        gt_pred = to_vec(gt_obj["future"], input_dimension, hz)

        match_idx = find_matching_tracklet_by_id(gid, tracklets) if use_id \
                    else find_matching_tracklet_by_iou(gt_obj, used_trk, iou_th, tracklets)

        if match_idx is not None:
            used_trk.add(match_idx)
            past = tracklets[match_idx]["tracklet"]
            pred = to_vec(pred_means[match_idx], input_dimension, hz)  # [hz, D]
            confidence = tracklets[match_idx]["confidence"]
            bbox = tracklets[match_idx]["current_pos"]
            category = tracklets[match_idx].get("category", "unknown")
            matched.append({
                "gid": gid, "match_idx": match_idx, "hz": hz,
                "gt_past": gt_past, "gt_future": gt_pred,
                "past": past, "bbox": bbox, "future": pred,
                "category": category, "confidence": confidence
            })
        else:
            missed.append({
                "id": gid, "gt_past": gt_past,
                "gt_bbox": gt_obj["current_state"],
                "gt_future": gt_pred, "category": category
            })

    # ---------------- false positives (unmatched tracker tracks) ---------------- #
    for i, trk in enumerate(tracklets):
        if i not in used_trk:
            pred = to_vec(pred_means[i], input_dimension, pred_len)
            category = trk.get("category", "unknown")
            false_pos.append({
                "id": trk["id"], "past": trk["tracklet"],
                "bbox": trk["current_pos"], "future": pred,
                "category": category
            })

    # predictor-only baseline (UNPACK to means, covs)
    base_means, base_covs = predict_based_on_gt_past(vehicle, matched)

    # ---------------- per-object ADE/FDE (unchanged) ---------------- #
    ade_list, fde_list, stats = compute_statistics(
        matched, tracklets, pred_means, base_means, input_dimension
    )

    # ---------------- MSNE (element-wise; always computed) ---------------- #
    msne_vals = []
    eps = 1e-9
    if len(matched) > 0:
        for m in matched:
            idx = m["match_idx"]
            hz  = m["hz"]
            gt  = m["gt_future"]     # [hz, D]
            mu  = m["future"]        # [hz, D] (already built from pred_means)

            # align covariance time order with mean order (dict keys sorted)
            cov_dict = pred_covs[idx]                                  # {t: [D,D]}
            ts_sorted = sorted(pred_means[idx])[:hz]                   # follow same ordering as to_vec(...)
            cov_seq = np.array([cov_dict[t] for t in ts_sorted], dtype=np.float32)  # [hz, D, D]

            var = np.clip(np.diagonal(cov_seq, axis1=1, axis2=2), eps, None)        # [hz, D]
            z2 = ((gt - mu) ** 2) / var                                             # [hz, D]
            msne_vals.append(float(np.mean(z2)))                                    # mean over time & dims

    # ---------------- overall metrics ---------------- #
    print(np.mean(msne_vals))
    overall = {
        "num_matched": len(matched),
        "num_missed": len(missed),
        "num_false_positives": len(false_pos),
        "ade": [], "fde": [],
        "msne": float(np.mean(msne_vals)) if len(msne_vals) > 0 else float("nan"),
    }
    for i in range(3):
        overall["ade"].append(float(np.mean(ade_list[:, i])) if len(ade_list) > 0 else float("nan"))
        overall["fde"].append(float(np.mean(fde_list[:, i])) if len(ade_list) > 0 else float("nan"))

    # ---------------- category metrics (unchanged) ---------------- #
    by_cat = compute_by_category_statistics(stats, missed, false_pos)

    forecasts = {"matched": matched, "missed": missed, "false_positives": false_pos}
    return forecasts, overall, by_cat



"""
def compute_frame_based_performance(predictions, tracklets, gt_predictions, vehicle, use_id=False, iou_th=0.75):
    sampling, horizon, input_dimension, pred_len = extract_parameters(tracklets, predictions)
    matched, missed, false_pos, used_trk = [], [], [], set()

    # match GT to tracker
    for gid, gt_obj in gt_predictions.items():
        hz = min(pred_len, len(gt_obj["future"]))
        category = gt_obj.get("category", "unknown")
        gt_past = to_vec(gt_obj["past"], input_dimension, vehicle.tracking_history, reverse=True)
        gt_pred = to_vec(gt_obj["future"], input_dimension, hz)

        match_idx = find_matching_tracklet_by_id(gid, tracklets) if use_id \
                    else find_matching_tracklet_by_iou(gt_obj, used_trk, iou_th, tracklets)

        if match_idx is not None:
            used_trk.add(match_idx)
            past = tracklets[match_idx]["tracklet"]
            pred = to_vec(predictions[match_idx], input_dimension, hz)
            confidence = tracklets[match_idx]["confidence"]
            bbox = tracklets[match_idx]["current_pos"]
            category = tracklets[match_idx].get("category", "unknown")
            matched.append({
                "gid": gid, "match_idx": match_idx, "hz": hz,
                "gt_past": gt_past, "gt_future": gt_pred,
                "past": past, "bbox": bbox, "future": pred,
                "category": category, "confidence": confidence
            })
        else:
            missed.append({
                "id": gid, "gt_past": gt_past,
                "gt_bbox": gt_obj["current_state"],
                "gt_future": gt_pred, "category": category
            })

    # false positives (unmatched tracker tracks)
    for i, trk in enumerate(tracklets):
        if i not in used_trk:
            pred = to_vec(predictions[i], input_dimension, pred_len)
            category = trk.get("category", "unknown")
            false_pos.append({
                "id": trk["id"], "past": trk["tracklet"],
                "bbox": trk["current_pos"], "future": pred,
                "category": category
            })

    # predictor-only baseline
    base_predictions = predict_based_on_gt_past(vehicle, matched)

    # per-object stats
    ade_list, fde_list, stats = compute_statistics(
        matched, tracklets, predictions, base_predictions, input_dimension
    )

    # overall metrics
    overall = {
        "num_matched": len(matched),
        "num_missed": len(missed),
        "num_false_positives": len(false_pos),
        "ade": [], "fde": []
    }
    for i in range(3):
        overall["ade"].append(float(np.mean(ade_list[:, i])) if len(ade_list) > 0 else float("nan"))
        overall["fde"].append(float(np.mean(fde_list[:, i])) if len(ade_list) > 0 else float("nan"))

    # category metrics
    by_cat = compute_by_category_statistics(stats, missed, false_pos)

    forecasts = {"matched": matched, "missed": missed, "false_positives": false_pos}
    return forecasts, overall, by_cat
"""