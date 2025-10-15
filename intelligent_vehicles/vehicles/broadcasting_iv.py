#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""


from . import BasicIV 
from intelligent_vehicles.broadcaster import Broadcaster
import logging
import time
logger = logging.getLogger(__name__)

class BroadcastingIV(BasicIV):
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        collaboration_graph (object, optional): Collaboration graph object.
    """
    
    def __init__(self, name, detector_config, tracker_config, predictor_config, broadcaster_config, parameters, sensors, data, channel_root):
        
        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data)
        
        self.broadcasting_frequency = broadcaster_config["broadcasting_frequency"]
        self._broadcaster = Broadcaster(root=channel_root, topic=broadcaster_config["topic"])
        self.next_broadcasting_time = self.starting_time + 1.0
        

    def _build_packet(self, predictions, ego_state):
        # keep this schema stable; add fields as needed
        packet = {"sender": str(self.name),
                  "broadcasting_timestamp": float(time.time()),
                  "fps": float(self.fps),
                  "pred_hz": float(self.prediction_frequency),
                  "pred_sampling": float(self.prediction_sampling),
                  "predictions": predictions}
        
        if ego_state != None:
            packet["ego_position"] = {"x": float(ego_state.get("x", 0.0)),
                                      "y": float(ego_state.get("y", 0.0)),
                                      "z": float(ego_state.get("z", 0.0)),
                                      "yaw": float(ego_state.get("yaw", 0.0))}
            
        return packet
            
            
    def run(self, t, scenario=None):
        # Check if it's time for observation
        if abs(t - self.next_observation_time) <= self.delta:
            frame_data = self.test_loader.get_frame_data(t)
            self.next_observation_time += 1.0 / self.fps
            
            if frame_data == None:
                logger.info(f"Vehicle {self.name} left the scene.")
                self.next_observation_time = self.starting_time
                self.next_prediction_time = self.starting_time + 1.0
                self.next_broadcasting_time = self.starting_time + 1.0
                return None
            
            # if we recived observation 
            ego_state = frame_data["ego_state"]
            calibration = frame_data["calibration"]
            point_cloud = frame_data["lidar"]
            trajectories = frame_data["trajectories"]
            
            # update location
            if ego_state != None:
                self.cur_location = [{"x": ego_state["x"], 
                                      "y": ego_state["y"], 
                                      "z": ego_state["z"], 
                                      "yaw": ego_state["yaw"]}]
                self.cur_location = self.ego_motion_compensation(self.cur_location, calibration)[0] 
           
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
                #response = (point_cloud, detections, ego_state)
            
            # Check if it's time for broadcasting
            if abs(t - self.next_broadcasting_time) <= self.delta: 
                predictions = self.object_graph.extract_predictions()
                packet = self._build_packet(predictions, ego_state) 
                message_size_bytes = self._broadcaster.send(packet)
                self.next_broadcasting_time += 1.0 / self.broadcasting_frequency  
                logger.info(f"[{self.name}] send broadcast with message size {message_size_bytes}")
                
            return response