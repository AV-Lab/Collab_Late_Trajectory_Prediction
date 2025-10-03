#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""

import torch
import os
import time
import numpy as np
import logging
logger = logging.getLogger(__name__)

from intelligent_vehicles.vehicles.dataloader import TrajDataloader
from intelligent_vehicles.detectors.initialize import initialize_detector
from intelligent_vehicles.trackers.initialize import initialize_tracker
from intelligent_vehicles.predictors.initialize import initialize_predictor
from intelligent_vehicles.graphs.initialize import initialize_object_graph
from intelligent_vehicles.late_fusion import GPFuser

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
        
    def _init_object_graph(self):
        print(f"Initializing object graph")
        self.object_graph = initialize_object_graph()
    
    def run_detector(self, t, frame_data, calibration, scenario=None):
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
        mean_trajs, cov_trajs = self.predictor.predict(past_trajs, 
                                                       self.prediction_horizon, 
                                                       self.prediction_sampling) 
        
        # wall-clock “now” in milliseconds (integer)
        pred_ts_ms = time.time_ns() 
        
        self.object_graph.update_by_predictor(tracklets, mean_trajs, cov_trajs, pred_ts_ms)
        
        # fuse predictions
        #logger.info(f"Run fusion, current state of the graph: {self.object_graph}")
        preds_with_pools = self.object_graph.extract_pools()
        fused_predictions = GPFuser.fuse(preds_with_pools)
        
        # update the graph and reset pools 
        self.object_graph.updtae_predictions(fused_predictions)
        self.object_graph.empty_pools()
        
        #logger.info(f"Return fused predictions: {self.object_graph}")
        return fused_predictions
    
    def reset(self):
        self.tracker.reset()
        self.object_graph.reset()
    
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
        
    def __init__(self, 
                 name, 
                 detector_config, 
                 tracker_config, 
                 predictor_config, 
                 parameters, 
                 sensors, 
                 data):
    
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
        
        self._init_object_graph()
    
    
    def run(self, t, scenario=None):
        # Check if it's time for observation
        if abs(t - self.next_observation_time) <= self.delta:
            frame_data = self.test_loader.get_frame_data(t)
            self.next_observation_time += 1.0 / self.fps
            
            if frame_data == None:
                logger.info(f"Vehicle {self.name} left the scene.")
                self.next_observation_time = self.starting_time
                self.next_prediction_time = self.starting_time + 1.0
                return None
            
            # if we recived observation 
            ego_state = frame_data["ego_state"]
            calibration = frame_data["calibration"]
            self.cur_location = [{"x": ego_state["x"], 
                                  "y": ego_state["y"], 
                                  "z": ego_state["z"], 
                                  "yaw": ego_state["yaw"]}]
            self.cur_location = self.ego_motion_compensation(self.cur_location, calibration)[0] 
            
            
            point_cloud = frame_data["lidar"]
            trajectories = frame_data["trajectories"]
           
            # Run detection
            detections = self.run_detector(t, frame_data, calibration, scenario)
        
            # Update the tracker 
            tracklets = self.run_tracker(detections)

            # Check if it's time for prediction ask tracker for active tracklets and run predict
            response = None
            if abs(t - self.next_prediction_time) <= self.delta: 
                predictions = self.run_predictor(tracklets)            
                response = (predictions, tracklets, trajectories, point_cloud, ego_state, calibration)
                self.next_prediction_time += 1.0 / self.prediction_frequency  
            
            return response