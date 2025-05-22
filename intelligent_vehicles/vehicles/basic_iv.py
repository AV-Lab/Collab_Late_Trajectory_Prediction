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
        if detector_config["name"] == "gt":
            self.load_gt_detections = True
        else:
            self.detector = initialize_detector(detector_config)

    def _init_tracker(self, tracker_config):
        print(f"Initializing tracker with config: {tracker_config}")
        self.tracker = initialize_tracker(tracker_config)

    def _init_predictor(self, predictor_config, data, prediction_horizon, prediction_frequency, forecasting_frequency):
        pass
    

    def laod_gt_detections(self, frame_data):
        """
        Load ground truth detections from the data.
        
        Args:
            data: Data containing ground truth detections.
        
        Returns:
            List of ground truth detections.
        """
        bboxs = [{'label': s['label'], 
                      'score': 1.0,
                      'width': s['width'], 
                      'height': s['height'], 
                      'length': s['length'], 
                      'position': (s['x'], s['y'], s['z']), 
                      'yaw': s['yaw'],
                      'obj_id' : s['obj_id']} for s in frame_data["labels"]]
        return bboxs     
       
    def run_detector(self, frame_data):
        if self.load_gt_detections:
            # return gt detections
            bboxs = self.laod_gt_detections(frame_data)
            calibration = frame_data["calibration"]
            return bboxs,  calibration     
        else:
            logger.info(f"Run detector on current frame data.")
            # to implement
            # run detect method 

    def run_tracker(self, detections, ego_pose,calibration):
        self.tracker.track(detections, ego_pose,calibration)
        tracklets = self.tracker.get_tracked_objects()


        return tracklets
    
    def run_predictor(self, tracklets):
        pass
    
    def train_predictor(self, data):
        pass
            
    def __init__(self, name, detector_config, tracker_config, predictor_config, parameters, sensors, data):
    
        self.name = name
        self.cur_location = None
        self.cur_velocity = None
        self.cur_yaw = None
        self.load_gt_detections = False
        self.delta = 0.005 # should be dt/2 from yout global clock
        self.starting_time = 0.0 # can include delays if needed
        self.next_observation_time = self.starting_time        
        self.fps = parameters["fps"]
        self.prediction_horizon = parameters["prediction_horizon"]
        self.prediction_frequency = parameters["prediction_frequency"]
        self.forecasting_frequency = parameters["forecasting_frequency"]
        self.results_folder = "results"
        
        if "train" in data:
            self.train_loader = self._init_dataloader(data["train"], sensors, self.fps)
        if "test" in data:
            self.test_loader = self._init_dataloader(data["test"], sensors, self.fps)
        elif "valid" in data:
            self.test_loader = self._init_dataloader(data["valid"], sensors, self.fps)
            
        self._init_detector(detector_config)
        self._init_tracker(tracker_config)
        self._init_predictor(predictor_config, 
                             data,
                             self.prediction_horizon,
                             self.prediction_frequency,
                             self.forecasting_frequency)
    def save_results(self, frame_id, gt, tracklets):
        """
        Save tracklets and ground truth to files for evaluation
        
        Args:
            frame_id: Current frame ID
            gt: List of ground truth objects
            tracklets: List of tracker results (tracklets)
        
        This function will overwrite files if frame_id=0, otherwise append to existing files
        """
        os.makedirs(self.results_folder, exist_ok=True)

        # Save GT
        gt_file = os.path.join(self.results_folder, f"{self.name}_gt_secs.txt")
        
        # Determine if we should write or append
        mode = 'w' if frame_id == 1 else 'a'
        
        with open(gt_file, mode) as f:
            for g in gt:
                f.write(f"{frame_id} {g['obj_id']} {g['label']} {g['position'][0]} {g['position'][1]} {g['position'][2]} "
                        f"{g['yaw']} {g['length']} {g['width']} {g['height']} {g['score']}\n")
        
        # Save tracklets
        tracklets_file = os.path.join(self.results_folder, f"{self.name}_tracklets_secs.txt")
        
        with open(tracklets_file, mode) as f:
            for t in tracklets:
                bbox3D = t.track_bbox() # [bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s, bbox.obj_class]

                f.write(f"{frame_id} {t.id} {t.category} {bbox3D.x} {bbox3D.y} {bbox3D.z} {bbox3D.ry} "
                        f"{bbox3D.l} {bbox3D.w} {bbox3D.h} {t.confidence}\n")
                # frame_id, id, category, x, y, z, ry, l, w, h, confidence
                # f.write(f"{frame_id} {t.id} {t.category} {bbox3D[3]} {bbox3D[4]} {bbox3D[5]} {bbox3D[6]} "
                #         f"{bbox3D[2]} {bbox3D[1]} {bbox3D[0]} {t.confidence}\n")
        # logger.info(f"Results saved for frame {frame_id} in {self.results_folder}.")

    def transform_to_world(self, detections,calibration):
        """
        Transform 3D bounding boxes from LiDAR frame to world coordinates.
        
        Args:
            detections: List of detection dictionaries from load_gt_detections
                
        Returns:
            List of detection dictionaries with world coordinates
        """
        # Get transformation matrices
        lidar_to_ego = calibration['lidar_to_ego']
        ego_to_world = calibration['ego_to_world']
        
        # Combine to get direct lidar-to-world transformation
        lidar_to_world = np.matmul(ego_to_world, lidar_to_ego)
        
        world_detections = []
        
        for det in detections:
            # Create a copy of the original detection
            new_det = det.copy()
            
            # Extract position
            x, y, z = det['position']
            
            # Create homogeneous coordinates for the position
            pos_lidar = np.array([x, y, z, 1.0])
            
            # Transform position to world coordinates
            pos_world = np.matmul(lidar_to_world, pos_lidar)
            
            # Update the position with world coordinates
            new_det['position'] = (pos_world[0], pos_world[1], pos_world[2])
            
            # Handle rotation transformation
            # Extract rotation part of the transformation matrix
            R_lidar_to_world = lidar_to_world[:3, :3]
            
            # Create rotation matrix for the original orientation in lidar frame
            yaw = det['yaw']
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            R_obj_lidar = np.array([
                [cos_yaw, 0, sin_yaw],
                [0, 1, 0],
                [-sin_yaw, 0, cos_yaw]
            ])
            
            # Combine rotations to get object orientation in world frame
            R_obj_world = np.matmul(R_lidar_to_world, R_obj_lidar)
            
            # Extract the new heading angle in world frame
            yaw_world = np.arctan2(R_obj_world[0, 2], R_obj_world[0, 0])
            
            # Update the yaw with world orientation
            new_det['yaw'] = yaw_world
            
            
            world_detections.append(new_det)
        
        return world_detections    
    def run(self, t,save_results=True):
       
        # Check if it's time for observation
        if abs(t - self.next_observation_time) <= self.delta:
            self.next_observation_time += 1.0 / self.fps
            frame_data,frame_id = self.test_loader.get_frame_data(t)
            if frame_data == None:
                logger.info(f"Vehicle {self.name} left the scene.")
                self.next_observation_time = self.starting_time
                return None
            # if we recived observation we run detection
            ego_state = frame_data["ego_state"]
            
            bboxs,calibration = self.run_detector(frame_data)
           
            # Update the tracker -> tracklets is numpy str array of 3D boxes [bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry,bbox.s,bbox.obj_class]
            tracklets = self.run_tracker(bboxs, ego_state,calibration)

            if save_results:
                gt_detections = self.laod_gt_detections(frame_data)
                
                # Transform detections to world coordinates
                gt_detections_wc = self.transform_to_world(gt_detections,calibration)
                self.save_results(frame_id, gt_detections_wc,tracklets)
            

            # Check if it's time for prediction
            # ask tracker for active tracklets and run predict
            #if (t - self.last_prediction_time) >= (1.0 / self.prediction_frequency):            
            #    predictions = self.run_predictor(tracklets)
            #    self.last_prediction_time = t
           
            return (bboxs, frame_data["lidar"], ego_state)
        
        
  

            
