#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregating Intelligent Vehicle:
- Runs local detect/track/predict like BasicIV.
- Listens asynchronously for peer broadcasts and aggregates them into the object graph.
"""

from . import BasicIV
from intelligent_vehicles.listener import Listener
import asyncio
import logging
from typing import Optional
logger = logging.getLogger(__name__)


class AggregatingIV(BasicIV):
    """
    Aggregating agent:
      - local sensing stack (detector/tracker/predictor) via BasicIV
      - async listener consuming peer predictions and updating collaboration graph
    """

    def __init__(self, name, detector_config, tracker_config, predictor_config, listener_config, parameters, sensors, data, channel_root):
        
        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data)
        
        self._listener = Listener(root=channel_root, topic=listener_config["topic"], on_message=self.updtae_object_graph)
        self._listener.start_in_background()


    def close(self):
        """Call this when tearing down the vehicle to stop the background listener."""
        try:
            self._listener.stop_in_background()
        except Exception:
            pass
        
    def updtae_object_graph(self, topic, payload):
        """
        Called immediately on message arrival by AsyncListener.
        `payload` is whatever the broadcaster sent (dict or compact list).
        Update the collaboration graph here.
        """
        
        try:
            logger.info(f"[{self.name}] recieved remote packet")
            broadcasting_timestamp = payload["timestamp_ms"]
            vehicle_parameters = {"fps": payload["fps"], 
                                  "prediction_horizon": payload["pred_hz"],
                                  "prediction_sampling": payload["pred_sampling"]}
            vehicle_location = payload["ego_position"]
            shared_predictions = payload["predictions"]
            objs_locations = [sp["cur_location"] for sp in shared_predictions]
            matches = self.object_graph.match_shared_predictions(objs_locations)
            self.object_graph.update_pools(matches, shared_predictions)
            matched_idx = set([match[0] for match in matches])
            unmatched_predictions = [sp for i, sp in enumerate(shared_predictions) if i not in matched_idx]
            self.object_graph.add_new_objects(self.cur_location, unmatched_predictions)
            logger.info(f"[{self.name}] processed remote packet: total {len(shared_predictions)}, associated {len(matches)}, unmatched {len(unmatched_predictions)}")

        except Exception as e:
            logger.info(f"[{self.name}] failed to process remote packet, {e}")

