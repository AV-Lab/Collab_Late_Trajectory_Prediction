#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:09:53 2025

@author: nadya
"""

from intelligent_vehicles.trackers.ab3dmot.model import AB3DMOT

import numpy as np
from collections import namedtuple
position = namedtuple('Position', ['x', 'y', 'z','yaw'])

class AB3DMOTWrapper:
    
    def __init__(self):
        self.tracker = AB3DMOT()
        self.tracker.reset()
        
    def track(self, detections, ego_pose, calibration):
        # Convert a dict-based detection to array format: [h, w, l, x, y, z, ry, score]
        
        dets = []

        for d in detections:
            h = d['dz']
            w = d['dy']
            l = d['dx']
            x = d['x']
            y = d['y']
            z = d['z']
            ry = d['yaw']
            s = d['score']
            category = d['label']
    
            detection = [h, w, l, x, y, z, ry, s, category]
            dets.append(detection)
            
        self.tracker.track(dets, ego_pose, calibration)
        
    def reset(self):
        self.tracker.reset()
    
    def get_tracked_objects(self):
        # Get the tracked objects from the tracker
        tracklets = self.tracker.get_active_tracklets()
        active_tracklets = []
        
        for tr in tracklets:
            current_pos = np.concatenate((tr.current_pos[:3], tr.current_pos[4:], tr.current_pos[3:4]))
            tracklet = [position(h[0], h[1], h[2], h[-1]) for h in tr.history] 
            track = {'id': tr.id, 'category': tr.category, 'confidence': tr.confidence, 'current_pos': current_pos, 'tracklet': tracklet}
            active_tracklets.append(track)
        return active_tracklets
