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
