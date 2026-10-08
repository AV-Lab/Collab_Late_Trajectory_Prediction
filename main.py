#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:18:44 2024

@author: nadya
"""


import argparse
from contextlib import ExitStack
import json

from parser import (load_config, parse_config)
from intelligent_vehicles.initialize import initialize_vehicles

from visualization.trajectory_visualize import PredictorVisualizer 
from evaluation import Evaluator
from evaluation.gt_calibration import GTCalibrationExperiment
from evaluation.evaluator import _json_value
from evaluation.matching import match_predictions
from evaluation.paths import OUTPUTS_DIR
import numpy as np
import time
import threading, zmq
import warnings
warnings.filterwarnings("ignore")
_PROXY_THREAD = None

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
    print(f"[Proxy] XSUB bound {channel_root}.in | XPUB bound {channel_root}.out")

def parse_configuration(config_path):
    try:
        config = load_config(config_path)
        parsed = parse_config(config)
        return parsed
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error parsing config: {e}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--viz", action="store_true", help="Enable trajectory visualization.")
    parser.add_argument("--gt-calibration", action="store_true",
                        help="Compare original fusion with pointwise GT calibration of Category I ego/sharing and Category II sharing covariance (gp_vector only).")
    parser.add_argument("--fusion-workers", type=int, default=None,
                        help="Number of CPU processes for independent vector-GP nodes (1: sequential).")
    args = parser.parse_args()
    if args.fusion_workers is not None and args.fusion_workers < 1:
        parser.error("--fusion-workers must be at least 1.")
    return args


def apply_fusion_options(ego_config, *, workers=None, gt_calibration=False):
    """Apply CLI overrides to category branches before vehicle initialization."""
    fusion_configs = [ego_config.get(category, {}).get("fusion", {})
                      for category in ("category_l", "category_s")]
    if workers is not None:
        vector_configs = [fusion for fusion in fusion_configs
                          if fusion.get("type") == "gp_vector"]
        if ego_config["type"] != "aggregating" or not vector_configs:
            raise ValueError("--fusion-workers requires an aggregating ego vehicle with gp_vector fusion.")
        for branch in vector_configs:
            branch["workers"] = workers
    if gt_calibration and (
        ego_config["type"] != "aggregating"
        or any(fusion.get("type") != "gp_vector" for fusion in fusion_configs)
    ):
        raise ValueError("--gt-calibration requires gp_vector fusion for both category_l and category_s.")


def attach_evaluation_timestamps(response, vehicle):
    """Attach exact future frame times to the inputs measured by evaluation."""
    predictions = [prediction for prediction in response["predictions"]
                   if "fusion_inputs" in prediction]
    if not predictions:
        return
    frame_index = vehicle.loader.current_frame - 1
    frame_time = vehicle.loader.timestamps[frame_index]
    horizon = vehicle.predictor.prediction_horizon
    gt_times_ms = [vehicle.last_prediction_timestamp + int(round(1000 * (t - frame_time)))
                   for t in vehicle.loader.timestamps[frame_index + 1:frame_index + 1 + horizon]]
    for prediction in predictions:
        prediction["gt_future_timestamps_ms"] = gt_times_ms


if __name__ == '__main__':
    args = parse_arguments()
    
    channel_root = "ipc:///tmp/prediction"   # use tcp://127.0.0.1:5556/.in .out 
    
    config_path = args.config
    print(f"Loading configuration from: {config_path}")
    configuration = parse_configuration(config_path)
    ego_config = configuration["vehicles"][configuration["ego_vehicle"]]
    apply_fusion_options(ego_config, workers=args.fusion_workers, gt_calibration=args.gt_calibration)
    print("Config parsed successfully")
    ensure_proxy_started(channel_root)
    
    dt = 0.02               # step in seconds
    clock_step =  dt / 2
    
    ego_vehicle = None
    vehicles = []
    visualizer = None
    try:
        scenarios, ego_vehicle, vehicles = initialize_vehicles(
            configuration,
            clock_step,
            channel_root,
            visualize=args.viz,
        )
        
        # parameters
        pred_len = ego_vehicle.predictor.prediction_horizon
        past_len = ego_vehicle.predictor.observed_past
    
        evaluator = Evaluator(
            prediction_horizon=pred_len,
            observation_length=past_len,
            sample_fps=ego_vehicle.predictor.fps,
        )
        control_evaluator = None
        calibration_experiment = None
        if args.gt_calibration:
            calibration_experiment = GTCalibrationExperiment(
                fuse_pools=ego_vehicle.fuse_pools, ego_vehicle=ego_vehicle.name,
                prediction_horizon=pred_len, covariance_jitter=evaluator.covariance_jitter,
            )
            ego_vehicle.fusion_observer = calibration_experiment.capture
            evaluator.run_metadata.update(condition="original", gt_calibration=False)
            control_evaluator = Evaluator(
                prediction_horizon=pred_len, observation_length=past_len,
                sample_fps=ego_vehicle.predictor.fps,
                iou_threshold=evaluator.iou_threshold, covariance_jitter=evaluator.covariance_jitter,
                output_dir=OUTPUTS_DIR / "gt_control",
                run_metadata=dict(condition="gt_control", gt_calibration=True,
                                  gt_calibration_categories=["category_I", "category_II"],
                                  gt_control_variance_floor_m2=1e-6),
            )
            audit_path = control_evaluator.output_dir / "covariance_changes.jsonl"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text("")
        visualizer = PredictorVisualizer() if args.viz else None
     
        for scenario, (number_of_vehicles, sim_time) in scenarios.items():
            # first preload all data for scenario
            ego_vehicle.reset()
            res = ego_vehicle.loader.preload_data(scenario)
            if not res:
                raise ValueError(f"Scenario '{scenario}' not found in dataset, for ego-vehicle it must be present.")
            
            print(f"For ego_vehicle {ego_vehicle.name} scnerio {scenario} is loaded")
            N = min(len(vehicles), number_of_vehicles-1)
            print(f"Total number of vehicles apart from ego: {N}")
            for iv in vehicles[:N]:
                iv.reset()
                iv.loader.preload_data(scenario)
                print(f"For {iv.name} scnerio {scenario} is loaded")
        
            t_global = 0.0
            prediction_frame_index = 0
            evaluator.begin_scenario(scenario)
            if control_evaluator is not None:
                control_evaluator.begin_scenario(scenario)
        
            # run global_clock (sequential, ego advances time)
            while t_global < sim_time:
                # step all other vehicles at current sim-time
                for iv in vehicles[:N]:
                    iv.run(t_global, scenario)
                # step ego at current sim-time
                response = ego_vehicle.run(t_global, scenario)
                if response is not None:
                    attach_evaluation_timestamps(response, ego_vehicle)
                    if control_evaluator is None:
                        forecasts = evaluator.compute(response)
                    else:
                        associations = match_predictions(
                            response["predictions"], response["trajectories"], evaluator.iou_threshold,
                        )
                        forecasts = evaluator.compute(response, associations=associations)
                        controlled = calibration_experiment.build_response(
                            response, associations[0],
                        )
                        control_evaluator.compute(controlled, associations=associations)
                        with audit_path.open("a") as audit_file:
                            for audit in controlled["gt_control_changes"]:
                                audit.update(scenario=scenario, frame_index=prediction_frame_index)
                                audit_file.write(json.dumps(_json_value(audit), allow_nan=False) + "\n")
                
                    if visualizer is not None:
                        visualizer.visualize_forecasts(
                            point_cloud=response["point_cloud"],
                            ego_pose=response["ego_state"],
                            calibration=response["calibration"],
                            forecasts=forecasts,
                            show_past=True,
                            show_future=True,
                            show_missing=True,
                            show_false=True,
                            sigma_scale=1.0
                        )
                    prediction_frame_index += 1

                # advance sim-time
                t_global += dt
            
            evaluator.end_scenario()
            if control_evaluator is not None:
                control_evaluator.end_scenario()
        if control_evaluator is None:
            evaluator.evaluate()
        else:
            evaluator.evaluate(gt_control=control_evaluator.evaluate(print_results=False))

    finally:
        with ExitStack() as cleanup:
            for resource in [ego_vehicle, *vehicles, visualizer]:
                close = getattr(resource, "close", None)
                if callable(close):
                    cleanup.callback(close)
