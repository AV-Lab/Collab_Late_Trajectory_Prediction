#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregating Intelligent Vehicle:
- Runs local detect/track/predict like BasicIV.
- Listens asynchronously for peer broadcasts and aggregates them into the object graph.
"""

from . import BasicIV
from intelligent_vehicles.listener import Listener
from intelligent_vehicles.filters import Filter
from intelligent_vehicles.late_fusion import GPFuser
from intelligent_vehicles.gates import KFGate
import asyncio
import time
import logging
from typing import Optional
logger = logging.getLogger(__name__)


class AggregatingIV(BasicIV):
    """
    Aggregating agent:
      - local sensing stack (detector/tracker/predictor) via BasicIV
      - async listener consuming peer predictions and updating collaboration graph
    """

    def __init__(self, name, detector_config, tracker_config, predictor_config, listener_config, parameters, sensors, data, clock_step, channel_root):
        
        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data,
                         clock_step)
        
        self._listener = Listener(root=channel_root, topic=listener_config["topic"], on_message=self.updtae_object_graph)
        self._listener.start_in_background()
        self.fuser = GPFuser()
        self.kf_gate = KFGate(mode="simple", min_streak=2, cov_ratio_thr=2.0)
        self.last_prediction_timestamp = None

    def close(self):
        """Call this when tearing down the vehicle to stop the background listener."""
        try:
            self._listener.stop_in_background()
        except Exception:
            pass

    def run_predictor(self, tracklets, sim_time_s, trajectories):
        past_trajs = self.predictor.format_input(tracklets)
        mean_trajs, cov_trajs = self.predictor.predict(past_trajs) 
        
        pred_ts_ms = int(round(sim_time_s * 1000.0))
        self.last_prediction_timestamp = pred_ts_ms  
        ego_ts = list(mean_trajs[0].keys())
        self.object_graph.update_by_predictor(tracklets, mean_trajs, cov_trajs, pred_ts_ms)
        preds_with_pools = self.object_graph.extract_pools()   
        
        ids_to_idx = {t["id"]: idx for idx,t in enumerate(tracklets)}
        gated_preds_with_pools = {}
        for k,v in preds_with_pools.items():
            if k in ids_to_idx:
                kf_features = tracklets[ids_to_idx[k]]["kf_gate"]    
                decision = self.kf_gate.decide(kf_features)
                if not decision.passed:
                    continue   
            gated_preds_with_pools[k] = v
        
        fused_predictions = self.fuser.fuse(ego_ts, gated_preds_with_pools, trajectories)
        self.object_graph.update_predictions(fused_predictions)
        self.object_graph.empty_pools()
        predictions = self.object_graph.extract_predictions()
        return predictions
        
    def updtae_object_graph(self, topic, payload):
        """
        Called immediately on message arrival by AsyncListener.
        `payload` is whatever the broadcaster sent (dict or compact list).
        Update the collaboration graph here.
        """
        
        try:
            logger.info(f"[{self.name}] recieved remote packet")
            
            # Unpacking the package
            broadcasting_timestamp = payload["timestamp_ms"]
            vehicle_parameters = {"fps": payload["fps"], 
                                  "prediction_horizon": payload["pred_hz"],
                                  "prediction_sampling": payload["pred_sampling"]}
            vehicle_location = payload["ego_position"]
            shared_predictions = payload["predictions"]
            
            # Filter predictions to match to ego step
            shared_predictions = Filter.filter_to_ego_prediction_step(shared_predictions, 
                                                                      self.last_prediction_timestamp,
                                                                      self.prediction_frequency,
                                                                      self.prediction_sampling) 
            
            if len(shared_predictions) > 0:
                # Run association for object-level category I nodes
                objs_locations = [sp["cur_location"] for sp in shared_predictions]
                matches, _, unmatched_predictions = self.object_graph.match_shared_predictions(objs_locations)
                self.object_graph.update_pools(matches, shared_predictions)
                logger.info(f"[{self.name}] processed remote packet: total {len(shared_predictions)}, associated {len(matches)}")
                
                # Add category II nodes 
                if self.cur_location:
                    added_ids = self.object_graph.add_new_objects(self.cur_location, unmatched_predictions, shared_predictions)
                    logger.info(f"Total added nodes of category II: {len(added_ids)}")
            else: 
                logger.info(f"[{self.name}] processed remote packet: no relevant shared predictions")
                 
        except Exception as e:
            logger.info(f"[{self.name}] failed to process remote packet, {e}")