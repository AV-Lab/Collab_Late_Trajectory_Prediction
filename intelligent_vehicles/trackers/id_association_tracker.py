from __future__ import annotations
from queue import Queue
from collections import namedtuple, deque   # <-- NEW
from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter

MIN_CONFIRM_HITS = {
    "vehicle": 3,
    "pedestrian": 2,
    "cyclist": 2,
    "motorcyclist": 2,
}

TRACK_TIMEOUT_S = {
    "vehicle": 1.0,
    "pedestrian": 0.5,
    "cyclist": 0.7,
    "motorcyclist": 0.7,
}

MEASUREMENT_VARIANCE = (0.01, 0.01)
INITIAL_STATE_VARIANCE = (1.0, 1.0, 10.0, 10.0, 25.0, 25.0)

PROCESS_NOISE = {
    "vehicle": {"position": 0.10, "velocity": 0.40, "acceleration": 0.60},
    "pedestrian": {"position": 0.15, "velocity": 0.25, "acceleration": 0.40},
    "cyclist": {"position": 0.12, "velocity": 0.30, "acceleration": 0.50},
    "motorcyclist": {"position": 0.12, "velocity": 0.40, "acceleration": 0.60},
}

position = namedtuple("Position", ["x", "y", "z", "yaw"])


class IDAssociationTracker:
    def __init__(self, history_len, fps):
        print("Tracklets produced by GT-/CP detections + 6-state KF.")
        self.active_tracklets: List[IDAssociationTracker.Track] = []
        self.history_len = history_len
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.track_timeout = {
            category: max(1, round(timeout_s * self.fps))
            for category, timeout_s in TRACK_TIMEOUT_S.items()
        }

    def track(self, detections):
       
        not_asc_detections = []
        asc_tracks = []
        for det in detections:
            tracklet = self._find_track(det)
            if tracklet is None:
                not_asc_detections.append(det)
            else:
                asc_tracks.append(tracklet.id)
                tracklet.update(det)
       
        asc_tracks = set(asc_tracks)
        for tracklet in self.active_tracklets:
            if tracklet.id not in asc_tracks:
                tracklet.predict()
                tracklet.miss_count += 1
               
        for det in not_asc_detections:
            self._create_track(det)
            
        self.active_tracklets = [
            tr for tr in self.active_tracklets
            if tr.miss_count < self.track_timeout[tr.category]
        ]
           

    def get_tracked_objects(self):
        return [{
            "id":          tr.id,
            "category":    tr.category,
            "current_pos": tr.current_pos.to_array(),
            "tracklet":    list(tr.history.queue),
            "kf_gate":     tr.get_kf_gate_features()
        } for tr in self.active_tracklets if tr.is_confirmed]

    def reset(self):
        self.active_tracklets.clear()

    def _find_track(self, det):
        return next((t for t in self.active_tracklets if t.id == det["obj_id"]), None)

    def _create_track(self, det):
        tr = self.Track(self.history_len, det, dt=self.dt)
        self.active_tracklets.append(tr)
        return tr

    class Track:
        """One Kalman-filter track (x, y, vx, vy, ax, ay)."""
        # ---------------------------------------------------------------
        def __init__(self, len_hist: int, det: dict, dt: float = 0.1):
            self.history  = Queue(maxsize=len_hist)
            self.category = det["label"]
            self.id       = det["obj_id"]

            self.dx, self.dy, self.dz = det["dx"], det["dy"], det["dz"]

            self.kf = self._init_kf(det, dt)
            self.yaw = det["yaw"]
            self.current_pos = IDAssociationTracker.BBox(det)
            self.history.put(self.current_pos.to_position())
            self.miss_count = 0
            self.hit_count = 1
            self.is_confirmed = self.hit_count >= MIN_CONFIRM_HITS[self.category]
            self.trS = deque(maxlen=len_hist)        
            self.upd_flags = deque(maxlen=len_hist) 
            self.nis_vals = deque(maxlen=len_hist)   
            self.predict_streak = 0                 

            self._append_kf_diag(u_trace=float(np.trace(self.kf.P[:2, :2])), updated=True, nis=0.0)

        def _init_kf(self, det: dict, dt: float) -> KalmanFilter:
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
            kf.R = np.diag(MEASUREMENT_VARIANCE)

            # P: uncertain velocity & accel
            kf.P = np.diag(INITIAL_STATE_VARIANCE)

            params = PROCESS_NOISE[self.category]
            q_pos = params["position"] * dt**2
            q_vel = params["velocity"] * dt
            q_acc = params["acceleration"]
            kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

            # initial state
            kf.x[:2] = np.array([det["x"], det["y"]]).reshape(2, 1)
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
            nis = float(y.T @ np.linalg.inv(S) @ y)

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
            self.current_pos = IDAssociationTracker.BBox({
                "x": det["x"], "y": det["y"], "z": det["z"],
                "dx": self.dx, "dy": self.dy, "dz": self.dz,
                "yaw": self.yaw
            })
       
            # 6) bookkeeping
            self._push_history()
            self.miss_count = 0

            if not self.is_confirmed:
                self.hit_count += 1
                if self.hit_count >= MIN_CONFIRM_HITS[self.category]:
                    self.is_confirmed = True

            # ---- NEW: record KF reliabilities for this frame ----
            self._append_kf_diag(u_trace, updated=True, nis=nis)

        # ───────────── predict-only step ───────────────
        def predict(self):
            self.kf.predict()
            if not self.is_confirmed:
                self.hit_count = 0
            x_pred, y_pred = self.kf.x[0, 0], self.kf.x[1, 0]

            self.current_pos = IDAssociationTracker.BBox({
                "x": x_pred, "y": y_pred, "z": self.current_pos.z,
                "dx": self.dx, "dy": self.dy, "dz": self.dz,
                "yaw": self.yaw
            })
   
            self._push_history()

            # ---- NEW: record KF reliabilities (no update) ----
            u_trace = float(np.trace(self.kf.P[:2, :2]))
            self._append_kf_diag(u_trace, updated=False, nis=None)

        # ───────────────── helpers ────────────────────
        def _push_history(self):
            if self.history.full():
                self.history.get()
            self.history.put(self.current_pos.to_position())

        # ---- NEW: append one frame of KF reliability data ----
        def _append_kf_diag(self, u_trace: float, updated: bool, nis: float | None):
            self.trS.append(u_trace)
            self.upd_flags.append(bool(updated))
            self.nis_vals.append(float(nis) if nis is not None else None)
            # maintain predict-only streak
            if updated:
                self.predict_streak = 0
            else:
                self.predict_streak += 1

        # ---- NEW: expose a compact snapshot for gating later ----
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
