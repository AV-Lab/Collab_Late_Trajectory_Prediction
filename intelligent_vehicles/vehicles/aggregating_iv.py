# aggregating_iv.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregating Intelligent Vehicle:
- Runs local detect/track/predict like BasicIV.
- Listens asynchronously for peer broadcasts.
- NEW: consumes received messages via Listener.pop_arrived(sim_time_ms) in the main thread,
       so object_graph updates are synchronous and delay injection is unified.
"""

from . import BasicIV
from intelligent_vehicles.listener import Listener
from intelligent_vehicles.filters import Filter
from intelligent_vehicles.late_fusion import GPFuser
from intelligent_vehicles.gates import KFGate
import time
import logging

logger = logging.getLogger(__name__)


class AggregatingIV(BasicIV):
    def __init__(self,
                 name,
                 detector_config,
                 tracker_config,
                 predictor_config,
                 listener_config,
                 parameters,
                 sensors,
                 data,
                 clock_step,
                 channel_root,
                 global_coordinates):

        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data,
                         clock_step,
                         global_coordinates)

        # ---- unified delay params (optional) ----
        delay_cfg = listener_config.get("delay", {}) if isinstance(listener_config, dict) else {}
        k = delay_cfg.get("k", None)
        mu = delay_cfg.get("mu", None)
        var = delay_cfg.get("var", None)
        # ----------------------------------------

        # Unified: listener always buffers; no async callback updates
        self._listener = Listener(
            root=channel_root,
            topic=listener_config["topic"],
            on_message=None,
            k=k, mu=mu, var=var,
            drop=listener_config["drop"]
        )
        self._listener.start_in_background()

        self.fuser = GPFuser()
        self.kf_gate = KFGate(mode="simple", min_streak=2, cov_ratio_thr=2.0)
        self.last_prediction_timestamp = None

    def close(self):
        try:
            self._listener.stop_in_background()
        except Exception:
            pass

    def run_predictor(self, tracklets, sim_time_s, trajectories):
        t_all0 = time.perf_counter()

        # arrived packets at current sim time ------------------
        sim_ms = int(round(sim_time_s * 1000.0))
        arrived = self._listener.pop_arrived(sim_ms)
        for topic, payload in arrived:
            self.updtae_object_graph(topic, payload)
        # -----------------------------------------------------------------------------------

        # predictor input + forward
        t0 = time.perf_counter()
        past_trajs = self.predictor.format_input(tracklets)
        mean_trajs, cov_trajs = self.predictor.predict(past_trajs, trajectories)
        t_pred_ms = (time.perf_counter() - t0) * 1000.0

        # timestamp + ego time indices
        pred_ts_ms = int(round(sim_time_s * 1000.0))
        self.last_prediction_timestamp = pred_ts_ms
        ego_ts = list(mean_trajs[0].keys())

        # graph update + pool extraction
        t0 = time.perf_counter()
        self.object_graph.update_by_predictor(tracklets, mean_trajs, cov_trajs, pred_ts_ms)
        preds_with_pools = self.object_graph.extract_pools()
        t_graph_ms = (time.perf_counter() - t0) * 1000.0

        # predict-only (no fusion) time up to here
        t_predict_only_ms = (time.perf_counter() - t_all0) * 1000.0

        # KF gating
        t0 = time.perf_counter()
        ids_to_idx = {t["id"]: idx for idx, t in enumerate(tracklets)}
        gated_preds_with_pools = {}
        for k_id, v in preds_with_pools.items():
            if k_id in ids_to_idx:
                kf_features = tracklets[ids_to_idx[k_id]]["kf_gate"]
                decision = self.kf_gate.decide(kf_features)
                if not decision.passed:
                    continue
            gated_preds_with_pools[k_id] = v
        t_gate_ms = (time.perf_counter() - t0) * 1000.0

        # fusion
        t0 = time.perf_counter()
        fused_predictions = self.fuser.fuse(ego_ts, gated_preds_with_pools, trajectories) ######## !!!!!!!!!!!!!!! trajectories are only passed for visualization
        t_fuse_ms = (time.perf_counter() - t0) * 1000.0

        # post-fusion graph ops
        t0 = time.perf_counter()
        self.object_graph.update_predictions(fused_predictions)
        self.object_graph.empty_pools()
        predictions = self.object_graph.extract_predictions()
        t_post_ms = (time.perf_counter() - t0) * 1000.0

        # Total
        t_all_ms = (time.perf_counter() - t_all0) * 1000.0

        # Log (CSV line)
        with open("sh2_time_breakdown.csv", "a") as f:
            f.write(
                f"{t_all_ms:.3f},{t_pred_ms:.3f},{t_graph_ms:.3f},{t_gate_ms:.3f},"
                f"{t_fuse_ms:.3f},{t_post_ms:.3f},{t_predict_only_ms:.3f},"
                f"{len(tracklets)},{len(preds_with_pools)},{len(gated_preds_with_pools)}\n"
            )

        return predictions

    def updtae_object_graph(self, topic, payload):
        """
        Now called ONLY from the main thread (run_predictor) after delay-buffer release.
        """
        try:
            logger.info(f"[{self.name}] recieved remote packet")

            broadcasting_timestamp = payload["timestamp_ms"]
            vehicle_parameters = {"fps": payload["fps"],
                                  "prediction_horizon": payload["pred_hz"],
                                  "prediction_sampling": payload["pred_sampling"]}
            vehicle_location = payload["ego_position"]
            shared_predictions = payload["predictions"]

            shared_predictions = Filter.filter_to_ego_prediction_step(
                shared_predictions,
                self.last_prediction_timestamp,
                self.prediction_frequency,
                self.prediction_sampling
            )

            if len(shared_predictions) > 0:
                objs_locations = [sp["cur_location"] for sp in shared_predictions]
                matches, _, unmatched_predictions = self.object_graph.match_shared_predictions(objs_locations)
                self.object_graph.update_pools(matches, shared_predictions)
                logger.info(f"[{self.name}] processed remote packet: total {len(shared_predictions)}, associated {len(matches)}")

                if self.cur_location:
                    added_ids = self.object_graph.add_new_objects(self.cur_location, unmatched_predictions, shared_predictions)
                    logger.info(f"Total added nodes of category II: {len(added_ids)}")
            else:
                logger.info(f"[{self.name}] processed remote packet: no relevant shared predictions")

        except Exception as e:
            logger.info(f"[{self.name}] failed to process remote packet, {e}")