#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""

import torch
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)

from intelligent_vehicles.vehicles.dataloader import TrajDataloader
from intelligent_vehicles.detectors.initialize import initialize_detector
from intelligent_vehicles.trackers.initialize import initialize_tracker
from intelligent_vehicles.predictors.initialize import initialize_predictor

class BasicIntelligentVehicle:
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        collaboration_graph (object, optional): Collaboration graph object.
    """
    

    def _init_dataloader(self, data_file, sensors, fps):
        return TrajDataloader(data_file, sensors, fps)

    def _init_detector(self, detector_config):
        print(f"Initializing tracker with config: {detector_config}")
        self.detector = initialize_detector(detector_config)

    def _init_tracker(self, tracker_config):
        print(f"Initializing tracker with config: {tracker_config}")
        self.tracker = initialize_tracker(tracker_config)

    def _init_predictor(self, predictor_config, prediction_horizon, forecasting_frequency):
        print(f"Initializing tracker with config: {predictor_config}")
        self.predictor = initialize_predictor(predictor_config)
    
    def run_detector(self, frame_data, scenario=None):
        if hasattr(self.detector, "load_detections") and self.detector.load_detections:
            detections = self.detector.detect(frame_data, scenario)
        else:
            detections = self.detector.detect(frame_data)
        return detections 

    def run_tracker(self, detections, ego_pose, calibration):
        self.tracker.track(detections, ego_pose, calibration)
        tracklets = self.tracker.get_tracked_objects()
        return tracklets
    
    def run_predictor(self, tracklets):
        pass
            
    def __init__(self, name, detector_config, tracker_config, predictor_config, parameters, sensors, data):
    
        self.name = name
        self.cur_location = None
        self.cur_velocity = None
        self.cur_yaw = None
        self.load_gt_detections = False
        self.delta = 0.005 # should be dt/2 from global clock
        self.starting_time = 0.0 # can include delays if needed
        self.next_observation_time = self.starting_time        
        self.fps = parameters["fps"]
        self.tracking_history = parameters["tracking_history"]
        self.prediction_horizon = parameters["prediction_horizon"]
        self.prediction_frequency = parameters["prediction_frequency"]
        self.forecasting_frequency = parameters["forecasting_frequency"]
        self.device = parameters["device"]
        
        if "train" in data:
            self.train_loader = self._init_dataloader(data["train"], sensors, self.fps)
        if "valid" in data:
            self.valid_loader = self._init_dataloader(data["valid"], sensors, self.fps)
        test = data["test"] if "test" in data else data["valid"]
        self.test_loader = self._init_dataloader(test, sensors, self.fps)
        
        detector_config["device"] = self.device
        self._init_detector(detector_config)
        
        tracker_config["tracking_history"] = self.tracking_history
        self._init_tracker(tracker_config)
        
        predictor_config["observation_length"] = self.tracking_history
        predictor_config["prediction_horizon"] = self.prediction_horizon*self.forecasting_frequency 
        predictor_config["device"] = self.device
        self._init_predictor(predictor_config)
    
    
    def run(self, t, scenario=None):
        # Check if it's time for observation
        if abs(t - self.next_observation_time) <= self.delta:
            self.next_observation_time += 1.0 / self.fps
            frame_data,frame_id = self.test_loader.get_frame_data(t)
            if frame_data == None:
                logger.info(f"Vehicle {self.name} left the scene.")
                self.next_observation_time = self.starting_time
                return None
            # if we recived observation 
            ego_state = frame_data["ego_state"]
            calibration = frame_data["calibration"]
            
            # Run detection
            detections = self.run_detector(frame_data, scenario)
        
            # Update the tracker -> tracklets is numpy str array of 3D boxes [x, y, z, theta, l, w, h,s, obj_class]
            tracklets = self.run_tracker(detections, ego_state, calibration)

            # Check if it's time for prediction ask tracker for active tracklets and run predict
            #if (t - self.last_prediction_time) >= (1.0 / self.prediction_frequency):            
            #    predictions = self.run_predictor(tracklets)
            #    self.last_prediction_time = t
            
            return (tracklets, detections, frame_data["lidar"], ego_state, calibration)