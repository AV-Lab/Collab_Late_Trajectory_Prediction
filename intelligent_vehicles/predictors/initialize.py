#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""
from intelligent_vehicles.predictors.scene_wrapper import SceneWrapper
from intelligent_vehicles.predictors.track_wrapper import TrackWrapper


def initialize_predictor(predictor_config):
    layout = predictor_config.get("layout")
    if layout == "track":
        return TrackWrapper(predictor_config)
    elif layout == "scene":
        return SceneWrapper(predictor_config)
    else:
        raise ValueError("Unsupported predictor configuration.")
