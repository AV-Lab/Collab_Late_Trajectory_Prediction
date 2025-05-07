#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:18:44 2024

@author: nadya
"""


from logging_setup import setup_logging
from parser import (load_config, 
                    parse_deepaccident_config,
                    parse_opv2v_config,
                    parse_v2v4real_config)
from intelligent_vehicles.initialize import (initialize_vehicle, 
                                            initialize_vehicles)

from visualization.bbox_visualize import BBoxVisualizer
import time                    
import numpy as np

def parse_configuration(config_path):
    try:
        config = load_config(config_path)
        
        if "dataset" not in config:
            msg = "Config must have a 'dataset' block."
            logger.error(msg)
            raise ValueError(msg)
        
        dataset_block = config["dataset"]
        required_dataset_keys = {"name", "path", "prefixes"}
    
        if not required_dataset_keys.issubset(dataset_block.keys()):
            msg = "'dataset' block must contain 'name', 'path', and 'prefixes' fields."
            logger.error(msg)
            raise ValueError(msg)
            
        if dataset_block["name"].lower() == "deepaccident":
            parsed = parse_deepaccident_config(config, logger)
        elif dataset_block["name"].lower() == "opv2v":
            parsed = parse_opv2v_config(config, logger)
        elif dataset_block["name"].lower() == "v2v4real":
            parsed = parse_v2v4real_config(config, logger)
        else:
            msg = "'dataset name' should one of [deepaccident, opv2v, v2v4real]."
            logger.error(msg)
            raise ValueError(msg)
            
        return parsed
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error parsing config: {e}")
        raise
    

if __name__ == '__main__':
    
    config_path = "configs/DeepAccident/config.yaml"
    logger = setup_logging("collaboration.log")

    
    logger.info(f"Loading configuration from: {config_path}")
    configuration = parse_configuration(config_path)
    logger.info("Config parsed successfully")
    
    ego_vehicle, vehicles = initialize_vehicles(configuration['data'], configuration["ego_vehicle"], configuration["vehicles"])
    
    scenarios = ego_vehicle.test_loader.extract_all_scenarios()
    
    # global clock 
    simulation_time = 10.0  # total sim time in seconds
    dt = 0.01               # step in seconds

    visualizer = BBoxVisualizer()
        
    for scenario in scenarios:
        # first preload all data for scenario
        ego_vehicle.test_loader.preload_data(scenario)
        logger.info(f"For {ego_vehicle.name} scnerio {scenario} is loaded")
        for iv in vehicles:
            iv.test_loader.preload_data(scenario) 
            logger.info(f"For {iv.name} scnerio {scenario} is loaded")
        
        t_global = 0.0
        # run global_clock
        while t_global < simulation_time:
            response = ego_vehicle.run(t_global)
            if response is not None:
                detections, point_cloud, ego_pose = response
                visualizer.visualize(point_cloud, detections, ego_pose)
                time.sleep(0.1)
            #for iv in vehicles:
            #    iv.run(t_global)
            t_global += dt
         
