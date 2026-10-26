#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""
from intelligent_vehicles.perception.predictors.scene_wrapper import SceneWrapper
from intelligent_vehicles.perception.predictors.track_wrapper import TrackWrapper


def initialize_predictor(predictor_config, predictor=None):
    layout = predictor_config.get("layout")
    if layout == "track":
        return TrackWrapper(predictor_config, predictor=predictor)
    elif layout == "scene":
        return SceneWrapper(predictor_config, predictor=predictor)
    else:
        raise ValueError("Unsupported predictor configuration.")
