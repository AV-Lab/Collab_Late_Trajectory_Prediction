#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:11:37 2025

@author: nadya
"""


import logging
logger = logging.getLogger(__name__)


class CenterPointWrapper:
    def __init__(self, detections_path):
        logger.info("Loading CenterPOint Detetcions.")
        self.load_detections = True
        self.detections = self.load(detections_path)
        
    def load(self, detections_path):
        pass
    
    def detect(self, frame_data, scenario):
        pass