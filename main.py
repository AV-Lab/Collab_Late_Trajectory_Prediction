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
import threading, zmq
_PROXY_THREAD = None
logger = setup_logging("collaboration.log")

def ensure_proxy_started(channel_root: str):
    global _PROXY_THREAD
    if _PROXY_THREAD and _PROXY_THREAD.is_alive():
        return
    ctx = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB); xsub.bind(f"{channel_root}.in")
    xpub = ctx.socket(zmq.XPUB); xpub.bind(f"{channel_root}.out")
    xsub.setsockopt(zmq.RCVHWM, 100)
    xpub.setsockopt(zmq.SNDHWM, 100)
    _PROXY_THREAD = threading.Thread(target=zmq.proxy, args=(xsub, xpub), daemon=True)
    _PROXY_THREAD.start()
    logger.info("[Proxy] XSUB bound %s.in | XPUB bound %s.out", channel_root, channel_root)

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
    
    channel_root = "ipc:///tmp/prediction"   # use tcp://127.0.0.1:5556/.in .out 
    ensure_proxy_started(channel_root)
    
    config_path = "configs/DeepAccident/config.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    configuration = parse_configuration(config_path)
    logger.info("Config parsed successfully")
    
    ego_vehicle, vehicles = initialize_vehicles(configuration['data'],
                                                configuration["ego_vehicle"],
                                                configuration["vehicles"],
                                                channel_root)  
    scenarios = ego_vehicle.test_loader.extract_all_scenarios()
    print("scenarios ready")
       
    # global clock 
    simulation_time = 10.0  # total sim time in seconds
    dt = 0.02               # step in seconds
    
    # parameters 
    input_size = ego_vehicle.predictor.predictor.input_size
    pred_len = ego_vehicle.predictor.predictor.prediction_horizon
    past_len = ego_vehicle.predictor.predictor.observation_length
    
    evaluator = Evaluator(logger=logger)  # NEW
    #viz = PredictorVisualizer()
     
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
            for iv in vehicles:
                iv.run(t_global)
            response = ego_vehicle.run(t_global, scenario)
            if response is not None:
                predictions, tracklets, trajectories, point_cloud, ego_pose, calibration = response
                forecasts, metrics = compute_frame_based_performance(predictions, 
                                                                     tracklets, 
                                                                     trajectories, 
                                                                     input_size, 
                                                                     pred_len, 
                                                                     past_len)
                

                evaluator.accumulate(metrics)  
                
                #viz.visualize_forecasts(
                #    point_cloud=point_cloud,
                #    ego_pose=ego_pose,
                #    calib=calibration,
                #    forecasts=forecasts,
                #    show_past=True,
                #    show_future=True,
                #    show_missing=True,      # include missed
                #    show_false=True,        # include false positives
                #    sigma_scale=1.0
                #)

            t_global += dt
            
        evaluator.end_scenario(scenario)  
    evaluator.log_overall(len(scenarios))  
    
    #visualizer.close()
         
