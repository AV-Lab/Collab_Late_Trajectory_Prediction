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
from intelligent_vehicles.initialize import initialize_vehicles

from visualization.bbox_visualize import BBoxVisualizer
from visualization.trajectory_visualize import PredictorVisualizer 
from evaluation.frame_based_metrics import compute_frame_based_performance  
from evaluation.prediction_evaluation import Evaluator              
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
    
    ego_vehicle, vehicles = initialize_vehicles(configuration['data'],
                                                configuration["ego_vehicle"],
                                                configuration["vehicles"])  
    scenarios = ego_vehicle.test_loader.extract_all_scenarios()
    print("scenarios ready")
       
    # global clock 
    simulation_time = 10.0  # total sim time in seconds
    dt = 0.02               # step in seconds

    evaluator = Evaluator(logger=logger)  # NEW
    #visualizer = BBoxVisualizer()
    #visualizer = PredictorVisualizer()
     
    for scenario in scenarios:
        # first preload all data for scenario
        ego_vehicle.reset()
        ego_vehicle.test_loader.preload_data(scenario)
        logger.info(f"For {ego_vehicle.name} scnerio {scenario} is loaded")
        for iv in vehicles:
            iv.reset()
            iv.test_loader.preload_data(scenario) 
            logger.info(f"For {iv.name} scnerio {scenario} is loaded")
        
        t_global = 0.0
        evaluator.begin_scenario()  # NEW
        
        # run global_clock
        while t_global < simulation_time:
            #for iv in vehicles:
            #    iv.run(t_global)
            response = ego_vehicle.run(t_global, scenario)
            if response is not None:
                predictions, tracklets, trajectories, point_cloud, ego_pose, calibration = response
                forecasts, metrics = compute_frame_based_performance(predictions, tracklets, trajectories, use_id=False, iou_th=0.75)
                evaluator.accumulate(metrics)  
                
                #visualizer.visualize_predictions(point_cloud, tracklets, predictions, ego_pose, calibration, transform_to_global=True)
                #point_cloud, detections, ego_pose = response
                #visualizer.visualize(point_cloud, detections, ego_pose)
            
            t_global += dt
            
        evaluator.end_scenario(scenario)  # NEW
    
    evaluator.log_overall(len(scenarios))  # NEW
    
    #visualizer.close()
         
