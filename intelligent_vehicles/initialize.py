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

import pickle

from intelligent_vehicles.vehicles import BasicIV 
from intelligent_vehicles.vehicles import AggregatingIV
from intelligent_vehicles.vehicles import BroadcastingIV 
from intelligent_vehicles.vehicles import HybridIV 
from pathlib import Path



def parse_meta(meta_path):
    scenarios = {}

    meta_path = Path(meta_path)
    with meta_path.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    fps = int(lines[2].split("=")[1])
    max_vehicles = int(lines[4].split("=")[1])

    for line in lines[6:]:
        scen, n, duration = line.rsplit(",", 2)
        scenarios[scen] = (int(n), float(duration))

    return fps, max_vehicles, scenarios


def extract_vehicles_sensors_data(data, vehicle_count):
    
    meta_file = data["meta_file"]
    data_file = data["data_file"] 
    
    fps, max_vehicles, scenarios_arr = parse_meta(meta_file)
    if vehicle_count > max_vehicles:
        raise ValueError(
            f"Configuration defines {vehicle_count} vehicles, "
            f"but the dataset supports at most {max_vehicles}."
        )

    tmp_dir = Path(data_file).parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    sensors_data_paths = [tmp_dir / f"vehicle_{idx}.pkl" for idx in range(vehicle_count)]
    existing_paths = list(tmp_dir.glob("vehicle_*.pkl"))
    
    sensors_data = [{} for _ in range(vehicle_count)]
    
    if len(existing_paths) != vehicle_count or not all(path.is_file() for path in sensors_data_paths):
        for path in existing_paths:
            path.unlink()

        with open(data_file, 'rb') as f:
            dataset = pickle.load(f)
            scenarios = dataset["scenarios"]
        
            for scenario, vehicles_data in scenarios.items():
                scenario_vehicles = list(vehicles_data.items())[:vehicle_count]
                for idx, (vehicle, ss_data) in enumerate(scenario_vehicles):
                    sensors_data[idx][scenario] = ss_data 

                for idx in range(len(scenario_vehicles), vehicle_count):
                    sensors_data[idx][scenario] = "Nan"
              
        for path, vehicle_data in zip(sensors_data_paths, sensors_data):
            with path.open('wb') as f:
                pickle.dump(vehicle_data, f)
                print(f"Saved data to {path}")

    return scenarios_arr, [str(path) for path in sensors_data_paths]
    

def initialize_vehicle(sensors_data, veh_id, veh_params, clock_step, channel_root, load_lidar,
                       calibration_by_vehicle=None):
    print(f"Initializing vehicle '{veh_id}' of type '{veh_params['type']}'.")
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
            category_config={category: veh_params[category]
                             for category in ("category_l", "category_s")},
            calibration_by_vehicle=calibration_by_vehicle,
            parameters=parameters,
            sensors=sensors,
            data=sensors_data,
            clock_step=clock_step,
            channel_root=channel_root,
            load_lidar=load_lidar
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
            load_lidar=load_lidar
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
            load_lidar=load_lidar
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
            load_lidar=load_lidar
        )
    print(f"Vehicle '{name}' initialized successfully.")
    return vehicle_obj


def initialize_vehicles(config, clock_step, channel_root, visualize=False):
    print("Extracting vehicles sensors data from pickle files.")
    vehicles = config["vehicles"]
    ego_vehicle = config["ego_vehicle"]
    data = config["data"]
    scenarios, sensors_data_paths = extract_vehicles_sensors_data(data, len(vehicles))
    n = min(len(sensors_data_paths), len(vehicles))
    vehicles_upd = {k: vehicles[k] for k in list(vehicles.keys())[:n]}
    ivs = []
    
    for i, (veh_id, veh_params) in enumerate(vehicles_upd.items()):
        veh_params["parameters"]["fps"] = data["fps"]
        iv = initialize_vehicle(sensors_data_paths[i],
                                veh_id,
                                veh_params, 
                                clock_step, 
                                channel_root,
                                load_lidar=(visualize and veh_id == ego_vehicle),
                                calibration_by_vehicle=config.get("calibration_by_vehicle", {}))
        
        if veh_id == ego_vehicle:
            ego_iv = iv
        else:
            ivs.append(iv)

    return scenarios, ego_iv, ivs
