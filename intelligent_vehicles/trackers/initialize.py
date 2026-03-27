#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:55:31 2025

@author: nadya
"""

from intelligent_vehicles.trackers.t_wrapper import TWrapper
from intelligent_vehicles.trackers.gt_wrapper import GTWrapper
import logging

logger = logging.getLogger(__name__)

def initialize_tracker(tracker_config):
    if tracker_config["name"] == "gt":
        return GTWrapper(tracker_config["tracking_history"], tracker_config["keep_track"])
    if tracker_config["name"] == "3d":
        return TWrapper(tracker_config["tracking_history"], tracker_config["keep_track"])     
    else:
        logger.error("You specified unsupported tracker class in yaml.")
        exit
