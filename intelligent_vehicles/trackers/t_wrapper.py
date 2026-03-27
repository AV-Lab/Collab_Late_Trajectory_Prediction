from __future__ import annotations
import logging
from queue import Queue
from collections import namedtuple, deque
from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

logger   = logging.getLogger(__name__)
position = namedtuple("Position", ["x", "y", "z", "yaw"])

# how many hits before a track becomes "confirmed"
MIN_CONFIRM_HITS = 3


class TWrapper:
    def __init__(self, history_len, keep_track, dt: float = 0.1):
        logger.info("Tracklets produced by GT-/CP detections + 6-state KF.")
        self.active_tracklets: List[TWrapper.Track] = []
        self.keep_track = keep_track
        self.history_len = history_len
        self.dt = float(dt)

        # previous-frame detections for velocity init (x, y, label)
        self.prev_dets: List[dict] = []

        # NIS gate (Mahalanobis^2 in 2D) – ~99% chi-square(2) ≈ 9.21
        self.nis_gate = 9.21

        # maximum distance for using previous detection to init velocity (m)
        self.init_vel_max_dist = 3.0

        # Per-category, per-state association thresholds (meters)
        self.dist_thresh = {
            "vehicle": {
                "tentative": 4.0,
                "confirmed": 2.0,
            },
            "pedestrian": {
                "tentative": 2.0,
                "confirmed": 1.0,
            },
            "cyclist": {
                "tentative": 3.0,
                "confirmed": 1.5,
            },
        }
        # Fallback if category not in table
        self.default_dist = {
            "tentative": 3.0,
            "confirmed": 1.5,
        }

    # ------------------------------------------------------------------ #
    # internal: pick distance threshold for a track based on category
    #           and whether the track is tentative or confirmed
    # ------------------------------------------------------------------ #
    def _assoc_dist_for(self, track: "TWrapper.Track") -> float:
        cat = track.category  # already lowercase
        state_key = "confirmed" if track.is_confirmed else "tentative"
        table = self.dist_thresh.get(cat, self.default_dist)
        return float(table[state_key])

    def track(self, detections: List[dict]):
        """
        Update all tracks given current-frame detections using
        global distance-based association (Hungarian + gating).
        """

        # Normalize detection labels to lowercase
        for det in detections:
            if "label" in det and isinstance(det["label"], str):
                det["label"] = det["label"].lower()

        # No existing tracks: just create new ones for all detections.
        if len(self.active_tracklets) == 0:
            for det in detections:
                self._create_track(det)
            # update prev_dets for next frame
            self.prev_dets = [
                {"x": d["x"], "y": d["y"], "label": d["label"]}
                for d in detections
            ]
            return

        # No detections: predict all tracks and age them.
        if len(detections) == 0:
            for tr in self.active_tracklets:
                tr.predict()
                tr.miss_count += 1
            # Prune dead tracks
            self.active_tracklets = [
                tr for tr in self.active_tracklets
                if tr.miss_count < self.keep_track
            ]
            # clear prev_dets (no observations this frame)
            self.prev_dets = []
            return

        num_tracks = len(self.active_tracklets)
        num_dets   = len(detections)

        # ---------------- Euclidean distances (for secondary gate) ----------------
        tracks_xy = np.array(
            [[tr.current_pos.x, tr.current_pos.y] for tr in self.active_tracklets],
            dtype=float
        )
        dets_xy = np.array(
            [[det["x"], det["y"]] for det in detections],
            dtype=float
        )
        diff = tracks_xy[:, None, :] - dets_xy[None, :, :]
        dists = np.linalg.norm(diff, axis=2)  # shape (num_tracks, num_dets)

        # ---------------- NIS (Mahalanobis^2) cost matrix ----------------
        cost = np.zeros((num_tracks, num_dets), dtype=np.float64)
        nis_mat = np.zeros_like(cost)

        for i, tr in enumerate(self.active_tracklets):
            kf = tr.kf
            F, Q, H, R = kf.F, kf.Q, kf.H, kf.R

            # prior (without mutating internal state)
            x_prior = F @ kf.x
            P_prior = F @ kf.P @ F.T + Q

            S = H @ P_prior @ H.T + R
            S = S + 1e-9 * np.eye(2)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            z_pred = H @ x_prior
            tr_cat = tr.category

            for j, det in enumerate(detections):
                det_cat = det.get("label", "").lower()
                z = np.array([[det["x"]], [det["y"]]], dtype=float)
                y = z - z_pred
                nis_val = float((y.T @ S_inv @ y).item())
                nis_mat[i, j] = nis_val

                if det_cat != tr_cat:
                    # huge cost for cross-category; will be rejected in gating anyway
                    cost[i, j] = 1e6 + nis_val
                else:
                    cost[i, j] = nis_val

        # Hungarian assignment (minimize NIS-based cost)
        rows, cols = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        # Apply Mahalanobis + Euclidean distance gate and update matched tracks
        for r, c in zip(rows, cols):
            tr = self.active_tracklets[r]
            det = detections[c]

            # category check (safety)
            if det.get("label", "").lower() != tr.category:
                continue

            dist = float(dists[r, c])
            nis_val = float(nis_mat[r, c])
            max_dist = self._assoc_dist_for(tr)

            if dist <= max_dist and nis_val <= self.nis_gate:
                tr.update(det)
                matched_tracks.add(r)
                matched_dets.add(c)

        # Unmatched tracks: predict + age
        for idx, tr in enumerate(self.active_tracklets):
            if idx not in matched_tracks:
                tr.predict()
                tr.miss_count += 1

        # Unmatched detections: spawn new (tentative) tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det)

        # Prune dead tracks
        self.active_tracklets = [
            tr for tr in self.active_tracklets
            if tr.miss_count < self.keep_track
        ]

        # update prev_dets for velocity init next frame
        self.prev_dets = [
            {"x": d["x"], "y": d["y"], "label": d["label"]}
            for d in detections
        ]

    def get_tracked_objects(self):
        """
        Return ONLY confirmed tracklets.
        """
        return [{
            "id":          tr.id,
            "category":    tr.category,
            "current_pos": tr.current_pos.to_array(),
            "tracklet":    list(tr.history.queue),
            "kf_gate":     tr.get_kf_gate_features()
        } for tr in self.active_tracklets if tr.is_confirmed]

    def reset(self):
        self.active_tracklets.clear()
        self.prev_dets = []

    def _find_track(self, det):
        # kept here in case you still use obj_id somewhere else
        return next((t for t in self.active_tracklets if t.id == det["obj_id"]), None)

    def _create_track(self, det):
        tr = self.Track(
            self.history_len,
            det,
            dt=self.dt,
            prev_dets=self.prev_dets,
            init_vel_max_dist=self.init_vel_max_dist,
        )
        self.active_tracklets.append(tr)
        return tr

    class Track:
        """One Kalman-filter track (x, y, vx, vy, ax, ay)."""
        # ---------------------------------------------------------------
        def __init__(
            self,
            len_hist: int,
            det: dict,
            dt: float = 0.1,
            prev_dets: Optional[List[dict]] = None,
            init_vel_max_dist: float = 3.0,
        ):
            self.history  = Queue(maxsize=len_hist)
            # store category in lowercase
            self.category = det["label"].lower()
            self.id       = det["obj_id"]

            self.dx, self.dy, self.dz = det["dx"], det["dy"], det["dz"]

            self.kf = self._init_kf(det, dt, prev_dets, init_vel_max_dist)
            self.yaw = det["yaw"]
            self.current_pos = TWrapper.BBox(det)
            self.history.put(self.current_pos.to_position())
            self.miss_count = 0

            # tentative / confirmed logic
            self.is_confirmed = False
            self.hit_count    = 1  # we just saw it once at creation

            # KF diagnostics
            self.trS = deque(maxlen=len_hist)        # KF pos cov trace history
            self.upd_flags = deque(maxlen=len_hist)  # update (True) vs predict-only (False)
            self.nis_vals = deque(maxlen=len_hist)   # innovation (NIS)
            self.predict_streak = 0                  # consecutive predict-only frames

            self._append_kf_diag(
                u_trace=float(np.trace(self.kf.P[:2, :2])),
                updated=True,
                nis=0.0,
            )

        def _init_kf(
            self,
            det: dict,
            dt: float,
            prev_dets: Optional[List[dict]],
            init_vel_max_dist: float,
        ) -> KalmanFilter:
            kf = KalmanFilter(dim_x=6, dim_z=2)
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

            # ---------------- category-dependent process noise Q ----------------
            cat = self.category
            if cat == "vehicle":
                q_pos = 0.10 * dt**2
                q_vel = 0.40 * dt
                q_acc = 0.60
            elif cat == "pedestrian":
                q_pos = 0.15 * dt**2
                q_vel = 0.25 * dt
                q_acc = 0.40
            elif cat == "cyclist":
                q_pos = 0.12 * dt**2
                q_vel = 0.30 * dt
                q_acc = 0.50
            else:
                # fallback / unknown category
                q_pos = 0.10 * dt**2
                q_vel = 0.30 * dt
                q_acc = 0.60

            kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

            # ---------------- initial state [x, y, vx, vy, ax, ay] ----------------
            x0 = float(det["x"])
            y0 = float(det["y"])
            vx0, vy0 = 0.0, 0.0

            # try to estimate initial velocity from previous detections
            if prev_dets:
                same_cat = [p for p in prev_dets if p.get("label", "").lower() == cat]
                if same_cat:
                    pts = np.array([[p["x"], p["y"]] for p in same_cat], dtype=float)
                    diffs = pts - np.array([x0, y0], dtype=float)
                    dists = np.linalg.norm(diffs, axis=1)
                    j = int(np.argmin(dists))
                    if dists[j] < init_vel_max_dist:
                        px, py = same_cat[j]["x"], same_cat[j]["y"]
                        vx0 = (x0 - px) / dt
                        vy0 = (y0 - py) / dt

            kf.x[:] = 0.0
            kf.x[0, 0] = x0
            kf.x[1, 0] = y0
            kf.x[2, 0] = vx0
            kf.x[3, 0] = vy0
            # ax, ay left at 0

            return kf

        def update(self, det: dict, alpha: float = 0.2):
            """
            KF predict→update with detection, and EMA-smooth box size (dx,dy,dz).
            alpha in [0,1]; higher = follow detector more closely.
            """
            # 1) Kalman predict (prior at this frame)
            self.kf.predict()
            # store PRIOR values for diagnostics before the update:
            x_prior = self.kf.x.copy()
            P_prior = self.kf.P.copy()
            u_trace = float(np.trace(P_prior[:2, :2]))

            # 2) Compute innovation and NIS w.r.t. PRIOR (cheap, explicit)
            z = np.array([det["x"], det["y"]], dtype=float).reshape(2, 1)
            y = z - self.kf.H @ x_prior
            S = self.kf.H @ P_prior @ self.kf.H.T + self.kf.R
            # numerical guard
            S = S + 1e-9 * np.eye(2)
            nis = float((y.T @ np.linalg.inv(S) @ y).item())

            # 3) Apply the update
            self.kf.update(z)
       
            # 4) EMA on size using detection as measurement
            dx_m, dy_m, dz_m = float(det["dx"]), float(det["dy"]), float(det["dz"])
            if not hasattr(self, "dx"):
                self.dx, self.dy, self.dz = dx_m, dy_m, dz_m
            else:
                self.dx = (1.0 - alpha) * self.dx + alpha * dx_m
                self.dy = (1.0 - alpha) * self.dy + alpha * dy_m
                self.dz = (1.0 - alpha) * self.dz + alpha * dz_m
       
            # 5) geometry: pose from det, size from EMA
            self.yaw = det["yaw"]
            x_pos, y_pos = self.kf.x[0, 0], self.kf.x[1, 0]
            #x_pos, y_pos = det["x"], det["y"]
            self.current_pos = TWrapper.BBox({
                "x": x_pos, "y": y_pos, "z": det["z"],
                "dx": self.dx, "dy": self.dy, "dz": self.dz,
                "yaw": self.yaw
            })
       
            # 6) bookkeeping
            self._push_history()
            self.miss_count = 0

            # track confirmation logic
            self.hit_count += 1
            if not self.is_confirmed and self.hit_count >= MIN_CONFIRM_HITS:
                self.is_confirmed = True

            # KF reliabilities for this frame
            self._append_kf_diag(u_trace, updated=True, nis=nis)

        # ───────────── predict-only step ───────────────
        def predict(self):
            self.kf.predict()
            x_pred, y_pred = self.kf.x[0, 0], self.kf.x[1, 0]

            self.current_pos = TWrapper.BBox({
                "x": x_pred, "y": y_pred, "z": self.current_pos.z,
                "dx": self.dx, "dy": self.dy, "dz": self.dz,
                "yaw": self.yaw
            })
   
            self._push_history()

            # KF reliabilities (no update)
            u_trace = float(np.trace(self.kf.P[:2, :2]))
            self._append_kf_diag(u_trace, updated=False, nis=None)

        # ───────────────── helpers ────────────────────
        def _push_history(self):
            if self.history.full():
                self.history.get()
            self.history.put(self.current_pos.to_position())

        # append one frame of KF reliability data
        def _append_kf_diag(self, u_trace: float, updated: bool, nis: float | None):
            self.trS.append(u_trace)
            self.upd_flags.append(bool(updated))
            self.nis_vals.append(float(nis) if nis is not None else None)
            # maintain predict-only streak
            if updated:
                self.predict_streak = 0
            else:
                self.predict_streak += 1

        # expose a compact snapshot for gating later
        def get_kf_gate_features(self, W: int = 10, L: int = 5) -> dict:
            """
            Returns a small dict with the features needed for the KF-only gate.
            No decision is made here.
            """
            # recent traces of predicted position covariance
            trS_hist = list(self.trS)[-W:] if len(self.trS) else []
            u_now = trS_hist[-1] if trS_hist else float(np.trace(self.kf.P[:2,:2]))
            U_med = float(np.median(trS_hist)) if trS_hist else u_now

            # consecutive predict-only frames (streak)
            streak = int(self.predict_streak)

            # median NIS over last L update frames
            nis_hist = [v for v in list(self.nis_vals) if v is not None]
            nis_recent = nis_hist[-L:] if len(nis_hist) >= 1 else []
            med_nis = float(np.median(nis_recent)) if len(nis_recent) else None

            return {
                "u_now": u_now,           # trace(S_kf^- at current frame)
                "u_med": U_med,           # median over last W frames
                "streak": streak,         # consecutive predicts
                "med_nis": med_nis,       # median NIS over last L updates (None if no updates)
                "W": W,
                "L": L
            }

    # ───────────────────── BBox helper ─────────────────────────
    class BBox:
        def __init__(self, d: dict):
            self.x, self.y, self.z = d["x"], d["y"], d["z"]
            self.dx, self.dy, self.dz = d["dx"], d["dy"], d["dz"]
            self.yaw = d["yaw"]

        @classmethod
        def from_array(cls, arr):
            raise RuntimeError("use explicit dict initialiser")

        def to_array(self):
            return np.array([self.x, self.y, self.z,
                             self.dx, self.dy, self.dz, self.yaw])

        def to_position(self):
            return position(self.x, self.y, self.z, self.yaw)
