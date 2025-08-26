#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""

from intelligent_vehicles.detectors.gt_wrapper import GTWrapper, GTNoiseWrapper
from intelligent_vehicles.detectors.centerpoint_wrapper import CenterPointWrapper
import logging

logger = logging.getLogger(__name__)

def initialize_detector(detector_config):
    if detector_config["name"] == "gt":
        return GTWrapper()
    if detector_config["name"] == "gt_noise":
        return GTNoiseWrapper(detections_path=detector_config["det_path"])
    if detector_config["name"] == "centerpoint":
        return CenterPointWrapper(detections_path=detector_config["det_path"])     
    else:
        logger.error("You specified unsupported detector class in yaml.")
        exit
