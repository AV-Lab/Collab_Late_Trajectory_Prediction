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
        self.load(detections_path)
        
    def load(self, detections_path):
        files = os.listdir(detections_path)
        
        for file_name in files:
            scenario_json = '/'.join(file_name.split('-'))
            scenario = scenario_json.split('.')[0]
            file = os.path.join(detections_path, file_name)
            
            with open(file, 'r') as f:
                data = json.load(f)
                reformatted_data = {}
                
                for t, dets in data.items():
                    k = round(float(t),1)
                    reformatted_dets = []
                    for d in dets:
                        reformatted_dets.append({"label": d["label"], 
                                             "score": d["score"],  
                                             "dx": d["length"], 
                                             "dy": d["width"], 
                                             "dz": d["height"], 
                                             "x": d["position"][0], 
                                             "y": d["position"][1], 
                                             "z": d["position"][2], 
                                             "yaw": d["yaw"]})
                    reformatted_data[k] = reformatted_dets
                    
                self.detections[scenario] = reformatted_data
                
        logger.info("The detections are loaded.") 
        self.load_detections = True
    
    def detect(self, scenario, t):
        return self.detections[scenario][round(t,1)]