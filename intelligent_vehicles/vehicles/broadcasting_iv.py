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
    
    def __init__(self, name, detector_config, tracker_config, predictor_config, broadcaster_config, parameters, sensors, data, clock_step, channel_root):
        
        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data,
                         clock_step)
        
        self.broadcasting_frequency = broadcaster_config["broadcasting_frequency"]
        self._broadcaster = Broadcaster(root=channel_root, topic=broadcaster_config["topic"])
        self.next_broadcasting_time = self.starting_time + 1.0
        self.bcast_period = 1.0 / self.broadcasting_frequency
        

    def _build_packet(self, predictions, ego_state, sim_time):
        packet = {"sender": str(self.name),
                  "broadcasting_timestamp": float(sim_time),   
                  "wall_time": float(time.time()),           
                  "fps": float(self.fps),
                  "pred_hz": float(self.prediction_frequency),
                  "pred_sampling": float(self.prediction_sampling),
                  "predictions": predictions}
        
        if ego_state is not None:
            packet["ego_position"] = {"x": float(ego_state.get("x", 0.0)),
                                      "y": float(ego_state.get("y", 0.0)),
                                      "z": float(ego_state.get("z", 0.0)),
                                      "yaw": float(ego_state.get("yaw", 0.0))}
            
        return packet
            
            
    def run(self, t, scenario=None):
        response = None
        
        if (t + self.delta) >= self.next_observation_time:
            frame_data = self.loader.get_frame_data(t)
            self.next_observation_time += self.obs_period
            
            if frame_data is None:
                logger.info(f"Vehicle {self.name} left the scene.")
                self.next_observation_time = self.starting_time
                self.next_prediction_time = self.starting_time + 1.0
                self.next_broadcasting_time = self.starting_time + 1.0
                return None
            
            # if we received observation 
            ego_state = frame_data["ego_state"]
            calibration = frame_data["calibration"]
            point_cloud = frame_data["lidar"]
            trajectories = frame_data["trajectories"]
            
            # update location
            if ego_state is not None:
                self.cur_location = [{"x": ego_state["x"], 
                                      "y": ego_state["y"], 
                                      "z": ego_state["z"], 
                                      "yaw": ego_state["yaw"]}]
                self.cur_location = self.ego_motion_compensation(self.cur_location, calibration)[0] 
           
            # Run detection
            detections = self.run_detector(t, frame_data, calibration, scenario)
        
            # Update the tracker 
            tracklets = self.run_tracker(detections)

            
            # Prediction gate (due-or-late)
            if (t + self.delta) >= self.next_prediction_time: 
                predictions = self.run_predictor(tracklets, t, trajectories)  # pass sim-time 't'
                response = (predictions, tracklets, trajectories, point_cloud, ego_state, calibration)
                self.next_prediction_time += self.pred_period  
            
        # Broadcasting gate (due-or-late)
        if (t + self.delta) >= self.next_broadcasting_time: 
            predictions = self.object_graph.extract_predictions(category_II_nodes=False)
            packet = self._build_packet(predictions, ego_state, t)  # include sim-time
            message_size_bytes = self._broadcaster.send(packet)
            self.next_broadcasting_time += self.bcast_period  
            logger.info(f"[{self.name}] send broadcast with message size {message_size_bytes}")
                
        return response
