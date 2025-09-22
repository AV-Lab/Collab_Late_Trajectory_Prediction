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

class BasicIV:
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
        print(f"Initializing detector with config: {detector_config}")
        self.detector = initialize_detector(detector_config)

    def _init_tracker(self, tracker_config):
        print(f"Initializing tracker with config: {tracker_config}")
        self.tracker = initialize_tracker(tracker_config)

    def _init_predictor(self, predictor_config):
        print(f"Initializing predictor with config: {predictor_config}")
        self.predictor = initialize_predictor(predictor_config)
    
    def run_detector(self, frame_data, t, calibration, scenario=None):
        if hasattr(self.detector, "load_detections") and self.detector.load_detections:
            detections = self.detector.detect(scenario, t)
        else:
            detections = self.detector.detect(frame_data)
            
        detections = self.ego_motion_compensation(detections, calibration)
        
        return detections 

    def run_tracker(self, detections):
        self.tracker.track(detections)
        tracklets = self.tracker.get_tracked_objects()
        return tracklets
    
    def run_predictor(self, tracklets):
        past_trajs = self.predictor.format_input(tracklets)
        future_trajs = self.predictor.predict(past_trajs, self.prediction_horizon, self.prediction_sampling) 
        return future_trajs
    
    def reset(self):
        self.tracker.reset()
        
    
    def ego_motion_compensation(self, detections, calibration):
        """
        Convert LiDAR-frame boxes to world frame (position + yaw).
        """
    
        T_lw = calibration["ego_to_world"] @ calibration["lidar_to_ego"]
        R_lw = T_lw[:3, :3]
        ego_heading = np.arctan2(R_lw[1, 0], R_lw[0, 0])   

        compensated = []
        for det in detections:
            pos_lidar = np.array([det["x"], det["y"], det["z"], 1.0])
            pos_world = T_lw @ pos_lidar
    
            yaw_world = det["yaw"] + ego_heading
            yaw_world = (yaw_world + np.pi) % (2 * np.pi) - np.pi   # wrap yaw to (-π,π]
    
            new_det = det.copy()
            new_det["x"], new_det["y"], new_det["z"] = pos_world[:3]
            new_det["yaw"] = yaw_world
            compensated.append(new_det)
    
        return compensated
        
    def __init__(self, name, detector_config, tracker_config, predictor_config, parameters, sensors, data):
    
        self.name = name
        self.cur_location = None
        self.cur_velocity = None
        self.cur_yaw = None
        self.load_gt_detections = False
        self.delta = 0.01 # should be dt/2 from global clock
        self.starting_time = 0.0 # can include delays if needed
        self.next_observation_time = self.starting_time  
        self.next_prediction_time = self.starting_time + 1.0 # delay by 1 second to make sure we have track history
        self.fps = parameters["fps"]
        self.tracking_history = parameters["tracking_history"]
        self.keep_track = parameters["keep_track"]
        self.prediction_horizon = parameters["prediction_horizon"]
        self.prediction_frequency = parameters["prediction_frequency"]
        self.prediction_sampling = parameters["prediction_sampling"]
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
        tracker_config["keep_track"] = self.keep_track
        self._init_tracker(tracker_config)
        
        predictor_config["device"] = self.device
        self._init_predictor(predictor_config)
    
    
    def run(self, t, scenario=None):
        # Check if it's time for observation
        if abs(t - self.next_observation_time) <= self.delta:
            self.next_observation_time += 1.0 / self.fps
            frame_data = self.test_loader.get_frame_data(t)
            
            if frame_data == None:
                logger.info(f"Vehicle {self.name} left the scene.")
                self.next_observation_time = self.starting_time
                self.next_prediction_time = self.starting_time + 1.0
                return None
            
            # if we recived observation 
            ego_state = frame_data["ego_state"]
            calibration = frame_data["calibration"]
            point_cloud = frame_data["lidar"]
            trajectories = frame_data["trajectories"]
           
            # Run detection
            detections = self.run_detector(frame_data, t, calibration, scenario)
        
            # Update the tracker 
            tracklets = self.run_tracker(detections)

            # Check if it's time for prediction ask tracker for active tracklets and run predict
            predictions = None
            if abs(t - self.next_prediction_time) <= self.delta: 
                self.next_prediction_time += 1.0 / self.prediction_frequency  
                predictions = self.run_predictor(tracklets)
            
                return (predictions, tracklets, trajectories, point_cloud, ego_state, calibration)
            
            #return (point_cloud, detections, ego_state)
            
            return None