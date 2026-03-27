#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:11:37 2025

@author: nadya
"""

import os
import json
import logging
logger = logging.getLogger(__name__)


class CenterPointWrapper:
    def __init__(self, detections_path):
        logger.info("Loading CenterPOint Detetcions.")
        self.detections = {}
        self.global_coordinates = False
        self.load(detections_path)
        
    def load(self, detections_path):
        dirs = os.listdir(detections_path)
        for dir_ in dirs:
            self.detections[dir_] = []
            dir_path = os.path.join(detections_path, dir_)
            dets_files = sorted(os.listdir(dir_path))
            for file_name in dets_files:
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    reform_data = []
                    for det in data:
                        reform_data.append({"obj_id": det["obj_id"],
                                            "label": det["obj_type"], 
                                            "score": 1.0,
                                            "occ_score" : 0.0,
                                            "dx": det["psr"]["scale"]["x"], 
                                            "dy": det["psr"]["scale"]["y"], 
                                            "dz": det["psr"]["scale"]["z"], 
                                            "x": det["psr"]["position"]["x"], 
                                            "y": det["psr"]["position"]["y"], 
                                            "z": det["psr"]["position"]["z"], 
                                            "yaw": det["psr"]["rotation"]["z"]})
                self.detections[dir_].append(reform_data)                
        logger.info("The detections are loaded.") 
        self.load_detections = True
        
    def time_to_index(self, t, dt=0.1):
        return int(round(t / dt))
    
    def detect(self, scenario, t):
        if '/' in scenario:
            scenario = scenario.split('/')[-1]
        idx = self.time_to_index(t)
        return self.detections[scenario][idx]