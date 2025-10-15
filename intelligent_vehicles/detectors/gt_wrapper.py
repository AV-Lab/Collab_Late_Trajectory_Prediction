#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""

"""
This class acts as a wrapper that provides perfect detections using ground-truth data.
When the `detect()` method is called with frame-level annotations (`frame_data`),
it returns a list of dictionaries, where each dictionary represents a detected object
with the following fields:
- 'label': object class label (e.g., 'car', 'pedestrian')
- 'score': detection confidence score, always set to 1.0 (since ground truth is used)
- 'dx', 'dy', 'dz': object dimensions
- 'x', 'y', 'z': coordinates in 3D space
- 'yaw': object orientation
- 'obj_id': unique identifier of the object

This module is typically used for oracle or upper-bound performance analysis.
"""

import logging
import json
import os
import math
import numpy as np
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)

class GTWrapper:
    def __init__(self):
        logger.info("The detections are taking from ground truth.") 
    
    def detect(self, frame_data):
        bboxs = [{'label': s['label'], 
          'score': 1.0,
          'dx': s['length'], 
          'dy': s['width'], 
          'dz': s['height'], 
          'x': s['x'],
          'y': s['y'],
          'z': s['z'],
          'yaw': s['yaw'],
          'occ_score': s['occ_l1'],
          'obj_id': s['obj_id']} for s in frame_data["labels"]]
        return bboxs  


    
    
    
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
        logger.info("Detections come from ground truth (occlusion-aware; xy noise only).")
        self.occ_mid_low = 0.15
        self.occ_mid_medium = 0.3
        self.occ_mid_high = 0.5
        self.occ_mid_max_pdrop = 0.5

        self.noise_pos_base_m = 0.02
        self.noise_pos_max_m  = 0.12
        self.noise_dist_gain  = 0.0 # no distance

        self.rng = np.random.default_rng(seed)

 
    def _apply_mid_occ(self, det: Dict, occ: float, ego: Optional[Dict]) -> Optional[Dict]:
        """
        Mid-occlusion policy with two bands:
          A) [occ_mid_low, occ_mid_medium): noise only (no drop)
          B) [occ_mid_medium, occ_mid_high]: probabilistic drop + noise
        """
        # between 0.15 and 0.3 add noise
        if occ < self.occ_mid_medium:
            denom = (self.occ_mid_medium - self.occ_mid_low)
            t = (occ - self.occ_mid_low) / denom
            t = max(0.0, min(1.0, t))
        else:
            # between 0.3 and 0.5, random drop or add noise
            denom = (self.occ_mid_high - self.occ_mid_medium)
            t = (occ - self.occ_mid_medium) / denom
            t = max(0.0, min(1.0, t))
            
            p_drop = t #* self.occ_mid_max_pdrop
            if self.rng.random() < p_drop:
                return None

        pos_sigma = self.noise_pos_base_m + (self.noise_pos_max_m - self.noise_pos_base_m) * t
        
        det = det.copy()
        det['x'] += self.rng.normal(0.0, pos_sigma)
        det['y'] += self.rng.normal(0.0, pos_sigma)
        return det

    def detect(self, frame_data: Dict) -> List[Dict]:
        """
        Build detections from GT labels and apply occlusion policy.
        Expects each label to include 'occ_l1'.
        """
        labels = frame_data.get("labels", [])
        ego = frame_data.get("ego_state", None)

        out: List[Dict] = []
        for s in labels:
            occ = float(s.get('occ_l1', 0.0))

            # High occlusion: drop
            if occ > self.occ_mid_high:
                continue

            det = {
                'label':  s['label'],
                'score':  1.0,
                'dx':     s['length'],
                'dy':     s['width'],
                'dz':     s['height'],
                'x':      s['x'],
                'y':      s['y'],
                'z':      s['z'],
                'yaw':    s['yaw'],        # passed through, no noise
                'occ_score': occ,          # keep occlusion in the output
                'obj_id': s['obj_id'],
            }

            # Medium occlusion: maybe drop or add minor xy noise
            if self.occ_mid_low < occ <= self.occ_mid_high:
                det2 = self._apply_mid_occ(det, occ, ego)
                if det2 is None:
                    continue
                det = det2

            # Low occlusion: keep as-is
            out.append(det)

        return out    