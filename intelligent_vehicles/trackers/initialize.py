#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:55:31 2025

@author: nadya
"""

from intelligent_vehicles.trackers.ab3dmot_wrapper import AB3DMOTWrapper
import logging

logger = logging.getLogger(__name__)

def initialize_tracker(tracker_config):
    if tracker_config["class"] == "AB3DMOT":
        return AB3DMOTWrapper()     
    else:
        logger.error("You specified unsupported tracker class in yaml.")
        exit
