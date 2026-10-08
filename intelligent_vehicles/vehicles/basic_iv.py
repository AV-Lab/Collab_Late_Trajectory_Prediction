#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""

import torch
import os
import time
import time
import numpy as np

from intelligent_vehicles.vehicles.dataloader import TrajDataloader
from intelligent_vehicles.detectors.initialize import initialize_detector
from intelligent_vehicles.trackers.initialize import initialize_tracker
from intelligent_vehicles.predictors.initialize import initialize_predictor
from intelligent_vehicles.objectstore import initialize_prediction_map


class BasicIV:
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        prediction_map (object, optional): Agent-level prediction map.
    """
    

    def _init_dataloader(self, data_file, sensors, fps, load_lidar):
        return TrajDataloader(data_file, sensors, fps, load_lidar)

    def _init_detector(self, detector_config):
        print(f"Initializing detector with config: {detector_config}")
        self.detector = initialize_detector(detector_config)

    def _init_tracker(self, tracker_config):
        print(f"Initializing tracker with config: {tracker_config}")
        self.tracker = initialize_tracker(tracker_config)

    def _init_predictor(self, predictor_config):
        print(f"Initializing predictor with config: {predictor_config}")
        self.predictor = initialize_predictor(predictor_config)
        
    def _init_prediction_map(self):
        print("Initializing prediction map")
        self.prediction_map = initialize_prediction_map()
    
    def run_detector(self, t, frame_data, calibration, scenario=None):
        if hasattr(self.detector, "load_detections") and self.detector.load_detections:
            detections = self.detector.detect(scenario, t)
        else:
            detections = self.detector.detect(frame_data)
            
        return detections 

    def run_tracker(self, detections):
        self.tracker.track(detections)
        tracklets = self.tracker.get_tracked_objects()
        return tracklets
    

    def run_predictor(self, tracklets, sim_time_s):
        # measure start time
        pred_ts_ms = int(round(sim_time_s * 1000.0))
        past_trajs = self.predictor.format_input(tracklets)       
        mean_trajs, cov_trajs = self.predictor.predict(past_trajs)
        self.prediction_map.update_by_predictor(tracklets, mean_trajs, cov_trajs, pred_ts_ms)
        predictions = self.prediction_map.extract_predictions()
        return predictions
    
    def reset_time_steps(self):
        self.starting_time = 0.0  
        self.next_observation_time = self.starting_time  
        self.next_prediction_time = self.starting_time + 1.0  
    
    def reset(self):
        self.reset_time_steps()
        self.tracker.reset()
        self.prediction_map.reset()
    
    def ego_motion_compensation(self, detections, calibration):
        """
        Convert LiDAR-frame boxes to world frame (position + yaw).
        """
    
        T_lw = np.array(calibration["ego_to_world"]) @ np.array(calibration["lidar_to_ego"])
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
                 data,
                 clock_step,
                 load_lidar):
    
        self.name = name
        self.cur_location = None
        self.cur_velocity = None
        self.cur_yaw = None
        self.load_gt_detections = False

        # timing/init
        self.reset_time_steps()

        # params
        self.fps = parameters["fps"]
        self.tracking_buffer_s = parameters["tracking_buffer_s"]
        self.observed_past_s = parameters["observed_past_s"]
        self.prediction_horizon_s = parameters["prediction_horizon_s"]
        self.device = parameters["device"]

        self.tracking_buffer = round(self.tracking_buffer_s * self.fps)
        self.observed_past = round(self.observed_past_s * self.fps)
        self.prediction_horizon = round(self.prediction_horizon_s * self.fps)
        self.prediction_frequency = self.fps
        
        detector_config["device"] = self.device
        predictor_config["device"] = self.device
        tracker_config["tracking_history"] = self.tracking_buffer
        tracker_config["fps"] = self.fps
        predictor_config["observed_past"] = self.observed_past
        predictor_config["prediction_horizon"] = self.prediction_horizon
        predictor_config["fps"] = self.fps
        
        # intialize predictor         
        self._init_predictor(predictor_config)
        self.prediction_sampling = self.fps
        self.obs_period  = 1.0 / self.fps
        self.pred_period = 1.0 / self.fps
        self.delta       = clock_step

        # intialize detector, tracker, prediction map
        self._init_detector(detector_config)
        self._init_tracker(tracker_config)
        self._init_prediction_map()
        
        self.loader = self._init_dataloader(data, sensors, self.fps, load_lidar)
    
    
    def run(self, t, scenario=None):
        response = None
        
        if (t + self.delta) >= self.next_observation_time:
            frame_data = self.loader.get_frame_data(t)
            self.next_observation_time += self.obs_period
            
            if frame_data is None:
                print(f"Vehicle {self.name} left the scene.")
                return None
            
            # if we received observation 
            ego_state = frame_data["ego_state"]
            calibration = frame_data["calibration"]
            trajectories = frame_data["trajectories"]
            point_cloud = frame_data["lidar"]
                 
            # update location
            if ego_state is not None:
                self.cur_location = ego_state

            # Run detection            
            detections = self.run_detector(t, frame_data, calibration, scenario)
            if not self.detector.global_coordinates:
                detections = self.ego_motion_compensation(detections, calibration)
                
            # Update the tracker 
            tracklets = self.run_tracker(detections)

            # Due-or-late gate for prediction            
            if (t + self.delta) >= self.next_prediction_time:
                if len(tracklets) > 0:
                    predictions = self.run_predictor(tracklets, t)
                    response = {
                        "predictions": predictions,
                        "tracklets": tracklets,
                        "trajectories": trajectories,
                        "point_cloud": point_cloud,
                        "ego_state": ego_state,
                        "calibration": calibration,
                    }
                self.next_prediction_time += self.pred_period  
            
        return response
