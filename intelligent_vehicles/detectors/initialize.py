#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""

from intelligent_vehicles.detectors.gt_occ_wrapper import GTOccWrapper
from intelligent_vehicles.detectors.gt_wrapper import GTWrapper
from intelligent_vehicles.detectors.centerpoint_wrapper import CenterPointWrapper


def initialize_detector(detector_config):
    if detector_config["name"] == "gt":
        return GTWrapper()
    if detector_config["name"] == "gt_occ":
        return GTOccWrapper()
    if detector_config["name"] == "centerpoint":
        return CenterPointWrapper(
            detections_path=detector_config["detections"],
            reflect_lidar_y=detector_config.get("reflect_lidar_y", False),
            max_distance_m=detector_config.get("max_distance_m"),
        )
    else:
        print("You specified unsupported detector class in yaml.")
        exit
