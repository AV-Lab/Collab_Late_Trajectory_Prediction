#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:09:53 2025

@author: nadya
"""

from intelligent_vehicles.trackers.ab3dmot.model import AB3DMOT

class AB3DMOTWrapper:
    
    def __init__(self):
        self.tracker = AB3DMOT()
        self.tracker.reset()
        
    def track(self, detections,ego_pose,calibration):
        # Convert a dict-based detection to array format: [h, w, l, x, y, z, ry, score]
        
        dets = []
        
        for d in detections:
            h = d['height']
            w = d['width']
            l = d['length']
            x, y, z = d['position']
            ry = d['yaw']
            s = d['score']
            category = d['label']
    
            detection = [h, w, l, x, y, z, ry, s, category]
            dets.append(detection)
            
        self.tracker.track(dets, ego_pose,calibration)
    
    def get_tracked_objects(self,return_trks=True):
        # Get the tracked objects from the tracker
        return self.tracker.get_active_tracklets(return_trks)
