"""Consecutive ground-truth histories without filtering or imputation."""

from collections import deque, namedtuple

import numpy as np


Position = namedtuple("Position", ["x", "y", "z", "yaw"])


class GTTracker:
    def __init__(self, history_len, min_points):
        if not 2 <= min_points <= history_len:
            raise ValueError("Require 2 <= min_points <= history_len.")
        self.history_len = history_len
        self.min_points = min_points
        self._tracks = {}

    def track(self, detections):
        current = {}
        for detection in detections:
            object_id = detection["obj_id"]
            history = (self._tracks[object_id]["history"]
                       if object_id in self._tracks
                       else deque(maxlen=self.history_len))
            history.append(Position(*(detection[key] for key in ("x", "y", "z", "yaw"))))
            current[object_id] = {
                "category": detection["label"],
                "current_pos": np.asarray([
                    detection[key] for key in ("x", "y", "z", "dx", "dy", "dz", "yaw")
                ]),
                "history": history,
            }
        self._tracks = current

    def get_tracked_objects(self):
        return [{
            "id": object_id,
            "category": track["category"],
            "current_pos": track["current_pos"].copy(),
            "tracklet": list(track["history"]),
            # Exact observed GT poses have no tracking uncertainty or missed-frame streak.
            "kf_gate": {"u_now": 0.0, "u_med": 0.0, "streak": 0, "med_nis": 0.0},
        } for object_id, track in self._tracks.items()
            if len(track["history"]) >= self.min_points]

    def reset(self):
        self._tracks.clear()
