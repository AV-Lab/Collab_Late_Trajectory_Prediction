#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""


from . import BasicIV 
from intelligent_vehicles.broadcaster import Broadcaster
from evaluation.paths import MESSAGE_SIZE_TMP_DIR
import time

class BroadcastingIV(BasicIV):
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        prediction_map (object, optional): Agent-level prediction map.
    """
    
    def __init__(self, 
                 name, 
                 detector_config, 
                 tracker_config, 
                 predictor_config, 
                 broadcaster_config, 
                 parameters, 
                 sensors, 
                 data, 
                 clock_step, 
                 channel_root,
                 load_lidar):
        
        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data,
                         clock_step,
                         load_lidar)
        
        self.broadcasting_interval_s = broadcaster_config["broadcasting_interval_s"]
        self._broadcaster = Broadcaster(root=channel_root, topic=broadcaster_config["topic"])
        self.next_broadcasting_time = self.starting_time + 1.0
        self.bcast_period = self.broadcasting_interval_s
        MESSAGE_SIZE_TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.message_size_path = MESSAGE_SIZE_TMP_DIR / f"{self.name}.txt"
        

    def _build_packet(self, predictions, ego_position, sim_time):
        packet = {"sender": str(self.name),
                  "broadcasting_timestamp": float(sim_time),
                  "wall_time": float(time.time()),
                  "fps": float(self.fps),
                  "pred_hz": float(self.prediction_frequency),
                  "pred_sampling": float(self.prediction_sampling),
                  "predictions": predictions,
                  "ego_position": {"x": float(ego_position["x"]),
                                   "y": float(ego_position["y"]),
                                   "z": float(ego_position["z"]),
                                   "yaw": float(ego_position["yaw"])}}
            
        
        return packet
            
            
    def run(self, t, scenario=None):
        response = None
        
        if (t + self.delta) >= self.next_observation_time:
            frame_data = self.loader.get_frame_data(t)
            self.next_observation_time += self.obs_period
            
            if frame_data is None:
                print(f"Vehicle {self.name} left the scene.")
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
                self.cur_location = ego_state
           
            # Run detection
            detections = self.run_detector(t, frame_data, calibration, scenario)
            if not self.detector.global_coordinates:
                detections = self.ego_motion_compensation(detections, calibration)
        
            # Update the tracker 
            tracklets = self.run_tracker(detections)

            
            # Prediction gate (due-or-late)
            if (t + self.delta) >= self.next_prediction_time: 
                if len(tracklets) > 0:
                    predictions = self.run_predictor(tracklets, t)  # pass sim-time 't'
                    response = {
                        "predictions": predictions,
                        "tracklets": tracklets,
                        "trajectories": trajectories,
                        "point_cloud": point_cloud,
                        "ego_state": ego_state,
                        "calibration": calibration,
                    }
                self.next_prediction_time += self.pred_period  
            
        # Broadcasting gate (due-or-late)
        if (t + self.delta) >= self.next_broadcasting_time: 
            predictions = self.prediction_map.extract_predictions(category_II_nodes=False)
            packet = self._build_packet(predictions, self.cur_location, t)  # include sim-time
            message_size_bytes = self._broadcaster.send(packet)
            self.next_broadcasting_time += self.bcast_period  
            print(f"[{self.name}] send broadcast with message size {message_size_bytes}")
            with self.message_size_path.open("a") as file:
                file.write(f"{message_size_bytes}\n")
        return response
