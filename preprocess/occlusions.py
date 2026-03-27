#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from math_helper import (
    box_corners_xy, rect_angular_span, ray_rect_distance
)

def compute_l1_occlusion_for_frame(
    objects: List[dict],
    ego_state: Optional[dict],
    K: int = 31,
    eps: float = 1e-2,
    vertical_check: bool = True,
    yaw_in_degrees: bool = False,
    *,
    return_debug: bool = True,
    ego_length_m: float = 4.6,   # used only to draw ego in debug (does not affect occlusion)
    ego_width_m: float = 1.9
):
    """
    DeepAccident-identical L1 BEV angular occlusion (no filtering inside).
    Also returns a debug payload exactly matching VisOcclusionScores.render(..).
    """
    occ_map: Dict[int, float] = {}

    # ---- origin (ego) ----
    sx = float(ego_state['x']) if ego_state is not None else 0.0
    sy = float(ego_state['y']) if ego_state is not None else 0.0
    s = np.array([sx, sy], dtype=np.float64)

    if not objects:
        debug = {
            "origin": s.astype(np.float32),
            "ego": {"corners": np.zeros((4,2), dtype=np.float32), "yaw_rad": 0.0},
            "corners": {},
            "rays": {},
            "bounds": (sx, sx, sy, sy)   # (minx, maxx, miny, maxy)
        } if return_debug else None
        return occ_map, debug if return_debug else occ_map

    # ---- ego shape for viz (does NOT affect occlusion math) ----
    ego_yaw = float(ego_state['yaw']) if ego_state is not None else 0.0
    ego_yaw_rad = math.radians(ego_yaw) if yaw_in_degrees else ego_yaw
    ego_corners = box_corners_xy(sx, sy, ego_length_m, ego_width_m, ego_yaw_rad)

    # ---- precompute rect corners, angular spans, heights ----
    corners_map: Dict[int, np.ndarray] = {}
    spans: Dict[int, Tuple[float, float]] = {}
    heights: Dict[int, float] = {}

    for o in objects:
        yaw = math.radians(o['yaw']) if yaw_in_degrees else float(o['yaw'])
        cs = box_corners_xy(o['x'], o['y'], o['length'], o['width'], yaw)
        corners_map[o['obj_id']] = cs
        spans[o['obj_id']] = rect_angular_span(s, cs)
        heights[o['obj_id']] = float(o['height'])

    # bounds start from ego + all rectangles
    xs = [*ego_corners[:,0].tolist()]
    ys = [*ego_corners[:,1].tolist()]
    for cs in corners_map.values():
        xs.extend(cs[:,0]); ys.extend(cs[:,1])

    rays_debug: Dict[int, dict] = {}

    # ---- per-target rays ----
    for tgt in objects:
        tid = tgt['obj_id']
        tL, tR = spans[tid]
        h_t = heights[tid]

        thetas = np.linspace(tL, tR, K) if K > 0 else np.empty((0,), dtype=np.float64)
        # scratch arrays for all K; we'll keep only valid ones in debug
        endpoints = np.zeros((K, 2), dtype=np.float64)
        blocked   = np.zeros((K,), dtype=bool)
        valid     = np.zeros((K,), dtype=bool)

        occluded = 0
        valid_cnt = 0

        cs_t = corners_map[tid]
        d_t_list: List[Optional[float]] = [ray_rect_distance(s, float(th), cs_t) for th in thetas]

        for k, th in enumerate(thetas):
            d_t = d_t_list[k]
            if d_t is None:
                continue

            valid[k] = True
            valid_cnt += 1

            end = s + d_t * np.array([math.cos(th), math.sin(th)], dtype=np.float64)
            endpoints[k] = end

            # test occluders
            hit_blocked = False
            for oc in objects:
                oid = oc['obj_id']
                if oid == tid:
                    continue
                oL, oR = spans[oid]

                # unwrap theta into occluder span domain (DeepAccident style)
                th_u = float(th)
                while th_u < oL: th_u += 2 * math.pi
                while th_u > oR: th_u -= 2 * math.pi
                if not (oL - 1e-9 <= th_u <= oR + 1e-9):
                    continue

                d_o = ray_rect_distance(s, float(th), corners_map[oid])
                if d_o is None:
                    continue

                if d_o < d_t - eps:
                    if vertical_check:
                        h_o = heights[oid]
                        if h_o < h_t * (d_o / d_t):
                            continue
                    hit_blocked = True
                    break

            blocked[k] = hit_blocked
            if hit_blocked:
                occluded += 1

        occ_map[tid] = (occluded / valid_cnt) if valid_cnt > 0 else 0.0

        # keep only VALID rays in debug (what the visualizer expects)
        if np.any(valid):
            v_idx = np.where(valid)[0]
            ends_v = endpoints[v_idx].astype(np.float32)
            blk_v  = blocked[v_idx]
            rays_debug[tid] = {"end": ends_v, "blocked": blk_v}
            xs.extend(ends_v[:,0].tolist()); ys.extend(ends_v[:,1].tolist())
        else:
            rays_debug[tid] = {"end": np.zeros((0,2), dtype=np.float32), "blocked": np.zeros((0,), dtype=bool)}

    # final tight bounds — order must be (minx, maxx, miny, maxy)
    minx, maxx = (min(xs), max(xs)) if xs else (sx, sx)
    miny, maxy = (min(ys), max(ys)) if ys else (sy, sy)

    debug = None
    if return_debug:
        debug = {
            "origin": s.astype(np.float32),
            "ego": {"corners": ego_corners.astype(np.float32), "yaw_rad": float(ego_yaw_rad)},
            "corners": {k: v.astype(np.float32) for k, v in corners_map.items()},
            "rays": rays_debug,
            "bounds": (float(minx), float(maxx), float(miny), float(maxy)),
        }

    return occ_map, debug
