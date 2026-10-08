"""Ground-truth detector with occlusion-aware filtering."""

from typing import Dict, List, Optional

import numpy as np


class GTOccWrapper:
    """
    Ground-truth-backed detector with occlusion-aware filtering/augmentation.

    Policy:
      - High occlusion (> occ_high_drop): drop.
      - Medium occlusion [occ_mid_low, occ_mid_high]: drop with probability that
        grows with occlusion; if kept, add minor noise to (x, y) that grows with occlusion
        (and optionally with distance to ego via noise_dist_gain).
      - Low occlusion (< occ_mid_low): keep as-is.
    """

    def __init__(self, seed: int = 1337):
        print("Detections come from ground truth (occlusion-aware; xy noise only).")
        self.occ_mid_low = 0.15
        self.occ_mid_medium = 0.3
        self.occ_mid_high = 0.75
        self.occ_mid_max_pdrop = 0.5

        self.noise_pos_base_m = 0.02
        self.noise_pos_max_m = 0.12
        self.noise_dist_gain = 0.0  # no distance

        self.rng = np.random.default_rng(seed)
        self.global_coordinates = True

    def _apply_mid_occ(
        self,
        det: Dict,
        occ: float,
        ego: Optional[Dict],
    ) -> Optional[Dict]:
        """
        Mid-occlusion policy with two bands:
          A) [occ_mid_low, occ_mid_medium): noise only (no drop)
          B) [occ_mid_medium, occ_mid_high]: probabilistic drop + noise
        """
        # between 0.15 and 0.3 add noise
        if occ < self.occ_mid_medium:
            denom = self.occ_mid_medium - self.occ_mid_low
            t = (occ - self.occ_mid_low) / denom
            t = max(0.0, min(1.0, t))
        else:
            # between 0.3 and 0.5, random drop or add noise
            denom = self.occ_mid_high - self.occ_mid_medium
            t = (occ - self.occ_mid_medium) / denom
            t = max(0.0, min(1.0, t))

            p_drop = t  # * self.occ_mid_max_pdrop
            if self.rng.random() < p_drop:
                return None

        pos_sigma = self.noise_pos_base_m + (
            self.noise_pos_max_m - self.noise_pos_base_m
        ) * t

        det = det.copy()
        det["x"] += self.rng.normal(0.0, pos_sigma)
        det["y"] += self.rng.normal(0.0, pos_sigma)
        return det

    def detect(self, frame_data: Dict) -> List[Dict]:
        """
        Build detections from GT labels and apply occlusion policy.
        Expects each label to include 'occ_l1'.
        """
        labels = frame_data["labels"]

        out: List[Dict] = []

        for s in labels:
            occ = float(s["occ_l1"])

            # High occlusion: drop
            if occ > self.occ_mid_high:
                continue

            det = {
                "label": s["label"],
                "score": 1.0,
                "dx": s["length"],
                "dy": s["width"],
                "dz": s["height"],
                "x": s["x"],
                "y": s["y"],
                "z": s["z"],
                "yaw": s["yaw"],
                "occ_score": occ,
                "obj_id": s["obj_id"],
            }

            out.append(det)

        return out
