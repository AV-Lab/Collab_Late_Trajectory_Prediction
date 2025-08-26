#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTNoiseWrapper + per-track Kalman Filter.

Input to `track(detections)`:
    detections[0] : real detection dicts  → KF predict() + update(z)
    detections[1] : stub  detection dicts → KF predict()           (no update)

`current_pos`:
    • real box  → keep raw detection geometry
    • stub box  → KF-predicted geometry

The public interface (`track`, `get_tracked_objects`, `reset`) is unchanged.
"""
import logging
from queue import Queue
from collections import namedtuple
import numpy as np
from filterpy.kalman import KalmanFilter   # pip install filterpy

logger   = logging.getLogger(__name__)
position = namedtuple("Position", ["x", "y", "z", "yaw"])


class GTNoiseWrapper:
    def __init__(self, history_len: int = 20):
        logger.info("Tracklets produced by GT+CP with per-track Kalman filter.")
        self.active_tracklets: list[GTNoiseWrapper.Track] = []
        self.history_len = history_len

    # ─────────────────────────── main update ──────────────────────────
    def track(self, detections):
        real_list, stub_list = detections
        matched_ids = set()

        # --- real detections → KF predict+update ----------------------
        for det in real_list:
            tr = self._get_or_create_track(det)
            tr.update(det)
            matched_ids.add(tr.id)

        # --- stubs → KF predict only ---------------------------------
        for det in stub_list:
            tr = self._find_track(det["obj_id"])
            if tr is None:
                continue          # never seen a real box yet
            tr.predict_only()
            matched_ids.add(tr.id)

        # --- drop tracks unseen this frame ---------------------------
        self.active_tracklets = [tr for tr in self.active_tracklets
                                 if tr.id in matched_ids]

    # ────────────────────────── public API ────────────────────────────
    def get_tracked_objects(self):
        return [{
            "id":          tr.id,
            "category":    tr.category,
            "confidence":  tr.confidence,            
            "current_pos": tr.current_pos.to_array(),
            "tracklet":    list(tr.history.queue)
        } for tr in self.active_tracklets]

    def reset(self):
        self.active_tracklets.clear()

    # ───────────────────── internal helpers ──────────────────────────
    def _find_track(self, obj_id):
        return next((t for t in self.active_tracklets if t.id == obj_id), None)

    def _get_or_create_track(self, det):
        tr = self._find_track(det["obj_id"])
        if tr is None:
            tr = self.Track(self.history_len, det)
            self.active_tracklets.append(tr)
        return tr

    # ───────────────────────── inner Track ───────────────────────────
    class Track:
        """One Kalman-filter track with exponential-decayed confidence."""
        # ──────────────────────────────────────────────────────────────
        def __init__(self, len_history, detection):
            self.history    = Queue(maxsize=len_history)
            self.category   = detection["label"]
            self.id         = detection["obj_id"]

            # KF ──────────────────────────────────────────────────────
            self.kf = self._init_kf(detection)

            # first geometry = raw detection
            self.current_pos = GTNoiseWrapper.BBox(detection)
            self.history.put(self.current_pos.to_position())

            # misc state
            self.miss_count = 0          # #consecutive predict_only() calls
            self.confidence = 1.0        # start fully confident
            self._compute_confidence(first=True)

        # ──────────────────────────────────────────────────────────────
        # KF set-up ----------------------------------------------------
        def _init_kf(self, det, dt: float = 0.1):
            x, y, z  = det["x"], det["y"], det["z"]
            yaw      = det["yaw"]
            dx, dy, dz = det["dx"], det["dy"], det["dz"]

            kf = KalmanFilter(dim_x=11, dim_z=7)

            # F : constant-velocity + constant yaw-rate
            F = np.eye(11)
            F[0, 7] = F[1, 8] = F[2, 9] = dt
            F[3,10] = dt
            kf.F = F

            # H : measure first 7 state variables directly
            H = np.zeros((7, 11))
            H[np.arange(7), np.arange(7)] = 1.0
            kf.H = H

            # measurement noise R
            kf.R = np.diag([0.01, 0.01, 0.01,      # 10 cm     on x,y,z
                            0.0025,                 # 0.05 rad  on yaw
                            0.04, 0.04, 0.04])      # 20 cm     on l,w,h

            # initial covariance P
            kf.P = np.diag([1, 1, 1,                # ±1 m  on position
                            0.25,                   # ±0.5 rad on yaw
                            0.3, 0.3, 0.3,          # ±0.55 m on size
                            25, 25, 25,             # high uncertainty on v
                            1.0])                   # yaw-rate

            # process noise Q
            q_pos = 0.2 * dt**2
            q_vel = 0.1 * dt
            q_yaw = 0.05 * dt
            kf.Q = np.diag([q_pos, q_pos, q_pos,
                            q_yaw,
                            1e-4, 1e-4, 1e-4,       # size ~ static
                            q_vel, q_vel, q_vel,
                            q_yaw])

            # initial state
            kf.x[:7] = np.array([x, y, z, yaw, dx, dy, dz]).reshape(7, 1)
            return kf

        # yaw wrap helper --------------------------------------------
        @staticmethod
        def _wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        # update with real detection ---------------------------------
        def update(self, det):
            self.kf.predict()

            z = np.array([det["x"], det["y"], det["z"],
                          det["yaw"], det["dx"], det["dy"], det["dz"]]).reshape(7, 1)

            # keep yaw residual small
            yaw_pred = float(self.kf.x[3, 0])
            z[3, 0] = self._wrap(z[3, 0] - yaw_pred) + yaw_pred

            self.kf.update(z)

            # normalise yaw in state
            self.kf.x[3, 0] = self._wrap(self.kf.x[3, 0])

            # store geometry from KF (you can swap to raw ‘det’ if preferred)
            self.current_pos = GTNoiseWrapper.BBox(det)
            #x7 = self.kf.x[:7].reshape(-1)
            #self.current_pos = GTNoiseWrapper.BBox.from_array(x7)

            self._push_history()
            self.miss_count = 0
            self._compute_confidence()

        # predict-only step ------------------------------------------
        def predict_only(self):
            self.kf.predict()
            self.kf.x[3, 0] = self._wrap(self.kf.x[3, 0])

            x7 = self.kf.x[:7].reshape(-1)
            self.current_pos = GTNoiseWrapper.BBox.from_array(x7)

            self._push_history()
            self.miss_count += 1
            self._compute_confidence()

        # confidence --------------------------------------------------
        def _compute_confidence(self, *, first=False,
                                sigma0: float = 0.3,
                                beta_t: float = 0.7,
                                alpha: float = 0.6):
            """
            Positional + yaw uncertainty * staleness decay → confidence ∈ (0,1].

            sigma0 : metres at which uncertainty term ≈ exp(-½)
            beta_t : multiplicative decay per predict-only frame (0–1)
            alpha  : EWMA smoothing factor
            """
            # 1) positional σ (x,y,z) and half-weighted yaw variance
            P = self.kf.P
            pos_var = np.trace(P[:3, :3])
            yaw_var = P[3, 3]
            sigma   = np.sqrt(pos_var + 0.5 * yaw_var)

            c_unc   = np.exp(-0.5 * (sigma / sigma0) ** 2)
            c_time  = beta_t ** self.miss_count
            c_inst  = c_unc * c_time

            if first:
                self.confidence = c_inst
            else:
                self.confidence = alpha * self.confidence + (1 - alpha) * c_inst

        # helpers -----------------------------------------------------
        def _push_history(self):
            if self.history.full():
                self.history.get()
            self.history.put(self.current_pos.to_position())



    # ────────────────────── BBox helper ─────────────────────────────
    class BBox:
        def __init__(self, detection):
            self.x  = detection["x"]
            self.y  = detection["y"]
            self.z  = detection["z"]
            self.dx = detection["dx"]
            self.dy = detection["dy"]
            self.dz = detection["dz"]
            self.yaw= detection["yaw"]

        @classmethod
        def from_array(cls, arr):
            x, y, z, yaw, dx, dy, dz = arr
            return cls(dict(x=x, y=y, z=z, yaw=yaw, dx=dx, dy=dy, dz=dz))

        def to_array(self):
            return np.array([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.yaw])

        def to_position(self):
            return position(self.x, self.y, self.z, self.yaw)
