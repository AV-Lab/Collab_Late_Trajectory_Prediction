#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 15:39:54 2025

Author: nadya

Description:
This module initializes intelligent vehicles by extracting sensor data from pickle files and instantiating
vehicle objects (Basic, Aggregating, Broadcasting, or Hybrid) based on configuration parameters.
It uses Redis for inter-vehicle communication in the overall system.
"""

import os
import pickle
import logging

from intelligent_vehicles.vehicles import BasicIV 
from intelligent_vehicles.vehicles import AggregatingIV
from intelligent_vehicles.vehicles import BroadcastingIV 
from intelligent_vehicles.vehicles import HybridIV 
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_meta(meta_path):
    scenarios = {}
    meta_path = Path(meta_path)
    with meta_path.open("r") as f:
        lines = [ln.strip() for ln in f]
        

    for line in lines[6:]:
        scen, n = line.rsplit(",", 1)
        scenarios[scen] = int(n)
    
    fps = int(lines[2].split("=")[1])
    max_vehicles = int(lines[3].split("=")[1])
                
    return fps, max_vehicles, scenarios


def extract_vehicles_sensors_data(data):
    
    meta_file = data["meta_file"]
    data_file = data["data_file"] 
    
    fps, max_vehicles, scenarios_arr = parse_meta(meta_file)
    loc = "/".join(data_file.split("/")[:-1])
    sensors_data_paths = [os.path.join(loc, f"vehicle_{idx}.pkl") for idx in range(max_vehicles)]
    
    sensors_data = [{} for _ in range(max_vehicles)]
    
    if not data["preprocessed"]:
        with open(data_file, 'rb') as f:
            dataset = pickle.load(f)
            scenarios = dataset["scenarios"]
        
            for scenario, vehicles_data in scenarios.items():
                for idx, (vehicle, ss_data) in enumerate(vehicles_data.items()):
                    sensors_data[idx][scenario] = ss_data 
                idx += 1
                
                for i in range(max_vehicles-idx):
                    sensors_data[idx+i][scenario] = "Nan"
              
        for i, path in enumerate(sensors_data_paths):
            with open(path, 'wb') as f:
                pickle.dump(sensors_data[i], f)
                print(f"Saved data to {path}")

    return scenarios_arr, sensors_data_paths
    

def initialize_vehicle(sensors_data, global_coordinates, veh_id, veh_params, clock_step, channel_root):
    logger.info(f"Initializing vehicle '{veh_id}' of type '{veh_params['type']}'.")
    vehicle_type = veh_params["type"]
    name = veh_id
    detector_config = veh_params["detector"]
    tracker_config = veh_params["tracker"]
    predictor_config = veh_params["predictor"]
    parameters = veh_params["parameters"]
    sensors = veh_params["sensors"]
    broadcaster_config = veh_params.get("broadcaster", None)
    listener_config = veh_params.get("listener", None)

    if vehicle_type == "aggregating":
        vehicle_obj = AggregatingIV(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            listener_config=listener_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data,
            clock_step=clock_step,
            channel_root=channel_root,
            global_coordinates=global_coordinates
        )
    elif vehicle_type == "broadcasting":
        vehicle_obj = BroadcastingIV(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            broadcaster_config=broadcaster_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data,
            clock_step=clock_step,
            channel_root=channel_root,
            global_coordinates=global_coordinates
        )
    elif vehicle_type == "hybrid":
        vehicle_obj = HybridIV(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            broadcaster_config=broadcaster_config,
            listener_config=listener_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data,
            clock_step=clock_step,
            channel_root=channel_root,
            global_coordinates=global_coordinates
        )
    else:
        vehicle_obj = BasicIV(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data,
            clock_step=clock_step,
            global_coordinates=global_coordinates
        )
    logger.info(f"Vehicle '{name}' initialized successfully.")
    return vehicle_obj


def initialize_vehicles(config, clock_step, channel_root):
    logger.info("Extracting vehicles sensors data from pickle files.")
    vehicles = config["vehicles"]
    ego_vehicle = config["ego_vehicle"]
    data = config["data"]
    global_coordinates = data["global_coordinates"] 
    
    scenarios, sensors_data_paths = extract_vehicles_sensors_data(data)
    n = min(len(sensors_data_paths), len(vehicles))
    vehicles_upd = {k: vehicles[k] for k in list(vehicles.keys())[:n]}
    ivs = []
    
    for i, (veh_id, veh_params) in enumerate(vehicles_upd.items()):
        iv = initialize_vehicle(sensors_data_paths[i], 
                                global_coordinates,
                                veh_id, 
                                veh_params, 
                                clock_step, 
                                channel_root)
        
        if veh_id == ego_vehicle:
            ego_iv = iv
        else:
            ivs.append(iv)

    return scenarios, ego_iv, ivs
