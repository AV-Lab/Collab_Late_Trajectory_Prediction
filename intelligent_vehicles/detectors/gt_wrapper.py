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
          'obj_id': s['obj_id']} for s in frame_data["labels"]]
        return bboxs  



class GTNoiseWrapper:
    def __init__(self, detections_path):
        logger.info("The detections are taking from noisy ground truth.") 
        self.detections = {}
        self.load(detections_path)
        
    def load(self, detections_path):
        files = os.listdir(detections_path)
        
        for file_name in files:
            scenario_json = '/'.join(file_name.split('-'))
            scenario = scenario_json.split('.')[0]
            file = os.path.join(detections_path, file_name)
            
            with open(file, 'r') as f:
                data = json.load(f)
                split_data = {}
                for t, dets in data.items():
                    detected = [d for d in dets if d['x'] != None]
                    missed = [d for d in dets if d['x'] == None]
                    split_data[t] = (detected, missed)
                self.detections[scenario] = split_data
                
        logger.info("The detections are loaded.") 
        self.load_detections = True
    
    def detect(self, scenario, t):
        return self.detections[scenario][str(round(t,1))]
    