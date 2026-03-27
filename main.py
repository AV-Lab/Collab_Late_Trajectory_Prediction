#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:18:44 2024

@author: nadya
"""


from logging_setup import setup_logging
from parser import (load_config, parse_config)
from intelligent_vehicles.initialize import initialize_vehicles

from visualization.bbox_visualize import BBoxVisualizer
from visualization.trajectory_visualize import PredictorVisualizer 
from evaluation.frame_based_metrics import compute_frame_based_performance  
from evaluation.prediction_evaluation import Evaluator              
import numpy as np
import time
import threading, zmq
import warnings
warnings.filterwarnings("ignore")
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
        parsed = parse_config(config, logger)
        return parsed
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error parsing config: {e}")
        raise
    

if __name__ == '__main__':
    
    channel_root = "ipc:///tmp/prediction"   # use tcp://127.0.0.1:5556/.in .out 
    ensure_proxy_started(channel_root)
    
    #config_path = "configs/DeepAccident/config.yaml"
    config_path = "configs/V2V4Real/config.yaml"
    #config_path = "configs/OPV2V/config.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    configuration = parse_configuration(config_path)
    logger.info("Config parsed successfully")
    
    simulation_time = 200.0  # total sim time in seconds
    dt = 0.02               # step in seconds
    clock_step =  dt / 2
    
    scenarios, ego_vehicle, vehicles = initialize_vehicles(configuration, clock_step, channel_root)  
        
    # parameters 
    input_size = ego_vehicle.predictor.input_size
    pred_len = ego_vehicle.predictor.prediction_horizon
    past_len = ego_vehicle.predictor.observation_length
    
    evaluator = Evaluator(logger=logger)
    #viz = BBoxVisualizer()
    #viz = PredictorVisualizer()
     
    for scenario, number_of_vehicles in scenarios.items():
        # first preload all data for scenario
        ego_vehicle.reset()
        res = ego_vehicle.loader.preload_data(scenario)
        if not res:
            raise ValueError(f"Scenario '{scenario}' not found in dataset, for ego-vehicle it must be present.")
            
        logger.info(f"For ego_vehicle {ego_vehicle.name} scnerio {scenario} is loaded")
        N = min(len(vehicles), number_of_vehicles-1)
        logger.info(f"Total number of vehicles apart from ego: {N}")
        for iv in vehicles[:N]:
            iv.reset()
            iv.loader.preload_data(scenario) 
            logger.info(f"For {iv.name} scnerio {scenario} is loaded")
        
        t_global = 0.0
        evaluator.begin_scenario()
        
        # run global_clock (sequential, ego advances time)
        while t_global < simulation_time:
            # step all other vehicles at current sim-time
            for iv in vehicles[:N]:
                iv.run(t_global, scenario)
            # step ego at current sim-time
            response = ego_vehicle.run(t_global, scenario)
            if response is not None:
                
                #detections, ego_state, point_cloud = response
                #viz.visualize(point_cloud, detections, ego_state)
                
                predictions, tracklets, trajectories, point_cloud, ego_pose = response
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
                #    forecasts=forecasts,
                #    show_past=True,
                #    show_future=True,
                #    show_missing=True,      # include missed
                #    show_false=True,        # include false positives
                #   sigma_scale=1.0
                #)

            # advance sim-time 
            t_global += dt
            
        evaluator.end_scenario(scenario)  
    evaluator.log_overall(len(scenarios))  
    #evaluator.plot_hist(model_name="LSTM-23")
    
    #visualizer.close()
