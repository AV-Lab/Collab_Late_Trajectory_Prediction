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

from intelligent_vehicles.vehicles.basic_iv import BasicIntelligentVehicle 
from intelligent_vehicles.vehicles.aggregating_iv import AggregatingIntelligentVehicle 
from intelligent_vehicles.vehicles.broadcasting_iv import BroadcastingIntelligentVehicle 
from intelligent_vehicles.vehicles.hybrid_iv import HybridIntelligentVehicle 


logger = logging.getLogger(__name__)

def extract_vehicles_sensors_data(data, ivs):
    
    sensors_data = {}
    pkls = {k:v for k,v in data.items() if isinstance(v, str) and v.endswith("pkl")}
    
    if data["preprocessed"]:
        for iv in ivs:
            sensors_data[iv] = {}
            for prefix, pkl_path in pkls.items():
                loc = "/".join(pkl_path.split("/")[:-2])
                pickle_path = os.path.join(loc, f"{prefix}/{iv}_{prefix}_data.pkl")
                if not os.path.isfile(pickle_path):
                    logger.error(f"'{pickle_path}' does not exsist, please specify preprocessed as False in config.")
                    exit()
                sensors_data[iv][prefix] = pickle_path
    else:
        for prefix, pkl_path in pkls.items():
            logger.info(f"Processing pickle file for prefix '{prefix}': {pkl_path}")
            with open(pkl_path, "rb") as f:
                dataset = pickle.load(f)
                scenarios = dataset["scenarios"]
                for scenario, vehicles_data in scenarios.items():
                    for vehicle, ss_data in vehicles_data.items():
                        if vehicle not in sensors_data:
                            sensors_data[vehicle] = {}
                        if prefix not in sensors_data[vehicle]:
                            sensors_data[vehicle][prefix] = {}
                        sensors_data[vehicle][prefix][scenario] = ss_data
            logger.debug(f"Finished processing prefix '{prefix}'.")
              
        # save each as vehicle + prefix as pkl
        loc = "/".join(pkl_path.split("/")[:-2])
        for vehicle_name, vehicle_data in sensors_data.items():
            for prefix, data in vehicle_data.items():
                output_pickle_path = os.path.join(loc, f"{prefix}/{vehicle_name}_{prefix}_data.pkl")
                with open(output_pickle_path, 'wb') as f:
                    pickle.dump(data, f)
                    sensors_data[vehicle_name][prefix] = output_pickle_path
                    print(f"Saved dataset to {output_pickle_path}")
        # save data and return path to the dataset 
    return sensors_data
    

def initialize_vehicle(sensors_data, vehicle_params):
    logger.info(f"Initializing vehicle '{vehicle_params['name']}' of type '{vehicle_params['type']}'.")
    vehicle_type = vehicle_params["type"]
    name = vehicle_params["name"]
    detector_config = vehicle_params["detector"]
    tracker_config = vehicle_params["tracker"]
    predictor_config = vehicle_params["predictor"]
    parameters = vehicle_params["parameters"]
    sensors = vehicle_params["sensors"]
    broadcaster_config = vehicle_params.get("broadcaster", None)

    if vehicle_type == "aggregating":
        vehicle_obj = AggregatingIntelligentVehicle(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data
        )
    elif vehicle_type == "broadcasting":
        vehicle_obj = BroadcastingIntelligentVehicle(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            broadcaster_config=broadcaster_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data
        )
    elif vehicle_type == "hybrid":
        vehicle_obj = HybridIntelligentVehicle(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            broadcaster_config=broadcaster_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data
        )
    else:
        vehicle_obj = BasicIntelligentVehicle(
            name=name,
            detector_config=detector_config,
            tracker_config=tracker_config,
            predictor_config=predictor_config,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data
        )
    logger.info(f"Vehicle '{name}' initialized successfully.")
    return vehicle_obj


def initialize_vehicles(data, ego_vehicle_dict, vehicles_dict):
    logger.info("Extracting vehicles sensors data from pickle files.")
    ivs = [iv_name["name"] for iv_name in vehicles_dict.values()]
    ivs.extend([ego_vehicle_dict["name"]])
    sensors_data = extract_vehicles_sensors_data(data, ivs)
    logger.info("Initializing ego vehicle.")
    ego_vehicle = initialize_vehicle(sensors_data[ego_vehicle_dict["name"]], ego_vehicle_dict)
    
    ivs = []
    for vehicle_id, vehicle_params in vehicles_dict.items():
        logger.info(f"Initializing vehicle '{vehicle_params['name']}' with ID '{vehicle_id}'.")
        iv_obj = initialize_vehicle(sensors_data[vehicle_params["name"]], vehicle_params)
        ivs.append(iv_obj)
    logger.info("All vehicles initialized successfully.")
    return ego_vehicle, ivs
