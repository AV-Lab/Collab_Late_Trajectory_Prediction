#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTNoiseWrapper  –  simple ID-based tracker with a per-track Kalman filter.

Tracked state  :  x, y, vx, vy, ax, ay         (6×1)
Measured state :  x, y                         (2×1)

Input to `track((real_list, stub_list))`
    real_list : list[dict]  – KF predict ➜ update(z)
    stub_list : list[dict]  – KF predict only

`current_pos`  real  → raw detection geometry
              stub  → KF-predicted geometry
"""

from __future__ import annotations
import logging
from queue import Queue
from collections import namedtuple
from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter

logger   = logging.getLogger(__name__)
position = namedtuple("Position", ["x", "y", "z", "yaw"])


class GTNoiseWrapper:
    def __init__(self, history_len):
        logger.info("Tracklets produced by GT-/CP detections + 6-state KF.")
        self.active_tracklets: List[GTNoiseWrapper.Track] = []
        self.history_len = history_len

    # ───────────────────────── main update ──────────────────────────
    def track(self, detections):
        real_list, stub_list = detections
        seen: set[int] = set()

        # real detections ---------------------------------------------------
        for det in real_list:
            tr = self._get_or_create_track(det)
            tr.update(det)
            seen.add(tr.id)

        # stubs ------------------------------------------------------------
        for det in stub_list:
            tr = self._find_track(det["obj_id"])
            if tr is None:
                continue
            tr.predict_only()
            seen.add(tr.id)

        # keep only tracks seen this frame ---------------------------------
        self.active_tracklets = [tr for tr in self.active_tracklets if tr.id in seen]

    # ─────────────────────────── getters ────────────────────────────────
    def get_tracked_objects(self):
        return [{
            "id":          tr.id,
            "category":    tr.category,
            "confidence":  list(tr.confidence.queue),
            "current_pos": tr.current_pos.to_array(),
            "tracklet":    list(tr.history.queue)
        } for tr in self.active_tracklets]

    def reset(self):
        self.active_tracklets.clear()

    # ─────────────────── internals (create / find) ─────────────────────
    def _find_track(self, obj_id):
        return next((t for t in self.active_tracklets if t.id == obj_id), None)

    def _get_or_create_track(self, det):
        tr = self._find_track(det["obj_id"])
        if tr is None:
            tr = self.Track(self.history_len, det)
            self.active_tracklets.append(tr)
        return tr

    # ───────────────────────── inner class ─────────────────────────────
    class Track:
        """One Kalman-filter track (x, y, vx, vy, ax, ay)."""
        # ---------------------------------------------------------------
        def __init__(self, len_hist: int, det: dict, dt: float = 0.1):
            self.history  = Queue(maxsize=len_hist)
            self.category = det["label"]
            self.id       = det["obj_id"]

            # permanent size from first detection
            self.dx, self.dy, self.dz = det["dx"], det["dy"], det["dz"]

            # KF init ---------------------------------------------------
            self.kf = self._init_kf(det, dt)

            # geometry / buffers --------------------------------------
            self.yaw = det["yaw"]
            self.current_pos = GTNoiseWrapper.BBox(det)
            self.history.put(self.current_pos.to_position())

            self.miss_count = 0
            self.confidence = Queue(maxsize=len_hist)
            self.update_confidence(first=True)

        # KF model set-up (6× state, 2× measurement) -------------------
        def _init_kf(self, det: dict, dt: float) -> KalmanFilter:
            kf = KalmanFilter(dim_x=6, dim_z=2)

            # state transition (const-acc 2-D)
            F = np.eye(6)
            F[0, 2] = F[1, 3] = dt
            F[0, 4] = F[1, 5] = 0.5 * dt * dt
            F[2, 4] = F[3, 5] = dt
            kf.F = F

            # measure x,y only
            H = np.zeros((2, 6))
            H[0, 0] = H[1, 1] = 1.0
            kf.H = H

            # R: 10 cm σ on position
            kf.R = np.diag([0.01, 0.01])

            # P: uncertain velocity & accel
            kf.P = np.diag([1, 1, 10, 10, 25, 25])

            # Q: process noise
                # -------------- process noise ---------------------------------
            q_pos = 0.1 * dt**2           #  ⇡  was 0.05
            q_vel = 0.3 * dt              #  ⇡  was 0.1
            q_acc = 0.4                   #  ⇡  was 0.1
            kf.Q = np.diag([q_pos, q_pos,
                            q_vel, q_vel,
                            q_acc, q_acc])

            # initial state
            kf.x[:2] = np.array([det["x"], det["y"]]).reshape(2, 1)
            # vx,vy,ax,ay start at 0
            return kf

        # ─────────── update with a real detection ─────────────
        def update(self, det: dict):
            self.kf.predict()
            z = np.array([det["x"], det["y"]]).reshape(2, 1)
            self.kf.update(z)

            # geometry: position from detection, size fixed, yaw direct
            self.yaw = det["yaw"]
            self.current_pos = GTNoiseWrapper.BBox({
                "x": det["x"], "y": det["y"], "z": det["z"],
                "dx": self.dx, "dy": self.dy, "dz": self.dz,
                "yaw": self.yaw
            })

            self._push_history()
            self.miss_count = 0
            self.update_confidence()

        # ───────────── predict-only step ───────────────
        def predict_only(self):
            self.kf.predict()
            x_pred, y_pred = self.kf.x[0, 0], self.kf.x[1, 0]

            self.current_pos = GTNoiseWrapper.BBox({
                "x": x_pred, "y": y_pred, "z": self.current_pos.z,
                "dx": self.dx, "dy": self.dy, "dz": self.dz,
                "yaw": self.yaw          # keep last yaw
            })
    
            self._push_history()
            self.miss_count += 1
            self.update_confidence()
            

        # ───────────── confidence from P11,P22 ──────────
        def update_confidence(self, first=False):
            if first:
                conf = 1.0
            else:
                sigma = np.trace(self.kf.P[:2, :2])  
                conf = 1.0 / (1 + sigma)
            
            if self.confidence.full():
                self.confidence.get()
            self.confidence.put(conf)

        # ───────────────── helpers ────────────────────
        def _push_history(self):
            if self.history.full():
                self.history.get()
            self.history.put(self.current_pos.to_position())

    # ───────────────────── BBox helper ─────────────────────────
    class BBox:
        def __init__(self, d: dict):
            self.x, self.y, self.z = d["x"], d["y"], d["z"]
            self.dx, self.dy, self.dz = d["dx"], d["dy"], d["dz"]
            self.yaw = d["yaw"]

        @classmethod
        def from_array(cls, arr):
            # arr = [x, y] from KF; the rest must be supplied by caller
            raise RuntimeError("use explicit dict initialiser")

        def to_array(self):
            return np.array([self.x, self.y, self.z,
                             self.dx, self.dy, self.dz, self.yaw])

        def to_position(self):
            return position(self.x, self.y, self.z, self.yaw)
