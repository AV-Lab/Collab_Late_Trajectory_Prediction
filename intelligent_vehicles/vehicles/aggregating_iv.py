# aggregating_iv.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregating Intelligent Vehicle:
- Runs local detect/track/predict like BasicIV.
- Listens asynchronously for peer broadcasts.
- NEW: consumes received messages via Listener.pop_arrived(sim_time_ms) in the main thread,
       so prediction-map updates are synchronous and delay injection is unified.
"""

from . import BasicIV
from intelligent_vehicles.listener import Listener
from intelligent_vehicles.fusion.gp_fusion import GPFuser, GPFuserVector
from intelligent_vehicles.fusion.linear_fusion import LinearFusion
from intelligent_vehicles.alignment import PredictionTimeAligner
from intelligent_vehicles.gates import ConsensusGate, KFGate
from evaluation.paths import RUNTIME_TMP_DIR
from contextlib import ExitStack
from copy import deepcopy
import time


_RUNTIME_HEADER = (
    "individual_ms,collaborative_ms,fusion_ms,fused_nodes,fusion_workers,"
    "category_l_ms,category_l_candidates,category_l_fused_nodes,category_l_workers,"
    "category_s_ms,category_s_candidates,category_s_fused_nodes,category_s_workers\n"
)


class AggregatingIV(BasicIV):
    def __init__(self,
                 name,
                 detector_config,
                 tracker_config,
                 predictor_config,
                 listener_config,
                 category_config,
                 parameters,
                 sensors,
                 data,
                 clock_step,
                 channel_root,
                 load_lidar):

        super().__init__(name,
                         detector_config,
                         tracker_config,
                         predictor_config,
                         parameters,
                         sensors,
                         data,
                         clock_step,
                         load_lidar)

        # ---- unified delay params (optional) ----
        delay_cfg = listener_config.get("delay", {}) if isinstance(listener_config, dict) else {}
        k = delay_cfg.get("k", None)
        mu = delay_cfg.get("mu", None)
        var = delay_cfg.get("var", None)
        # ----------------------------------------

        # Unified: listener always buffers; no async callback updates
        self._listener = Listener(
            vehicle_name=name,
            root=channel_root,
            topic=listener_config["topic"],
            on_message=None,
            k=k, mu=mu, var=var,
            drop=listener_config["drop"]
        )
        self.category_s_enabled = category_config["category_s"].get("enabled", True)
        local_config = category_config["category_l"]["fusion"]
        shared_config = category_config["category_s"].get("fusion", {})
        local_fuser = self._create_fuser(local_config)
        same_gp = (
            self.category_s_enabled
            and local_config["type"] in ("gp", "gp_vector")
            and local_config["type"] == shared_config["type"]
            and local_config.get("workers", 1) == shared_config.get("workers", 1)
        )
        self.fusers = {
            "category_l": local_fuser,
            "category_s": (local_fuser if same_gp else self._create_fuser(shared_config))
                          if self.category_s_enabled else None,
        }
        self.kf_gate = self._create_gate(category_config["category_l"].get("gate", {}))
        self.consensus_gate = self._create_gate(category_config["category_s"].get("gate", {})) \
                              if self.category_s_enabled else None
        self.aligners = {
            category: PredictionTimeAligner(**config.get("alignment", {}))
            for category, config in category_config.items()
            if config.get("enabled", True)
        }
        self.last_prediction_timestamp = None
        self.fusion_observer = None
        RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.runtime_path = RUNTIME_TMP_DIR / f"{self.name}.csv"
        self._runtime_header_checked = False
        self._listener.start_in_background()

    def close(self):
        try:
            self._listener.stop_in_background()
        except Exception:
            pass
        finally:
            with ExitStack() as cleanup:
                for fuser in dict.fromkeys(self.fusers.values()):
                    close_fuser = getattr(fuser, "close", None)
                    if callable(close_fuser):
                        cleanup.callback(close_fuser)

    def _create_fuser(self, config):
        if config["type"] == "gp":
            return GPFuser()
        if config["type"] == "gp_vector":
            return GPFuserVector(workers=config.get("workers", 1))
        if config["type"] == "linear_fusion":
            if self.certificate is None:
                raise ValueError("Linear fusion requires this vehicle's local certificate.")
            return LinearFusion(ego_source_id=self.name,
                                gate_ratio=config.get("gate_ratio", 1.0),
                                cap=config.get("cap"))
        if config["type"] == "linear_fusion_cross":
            from intelligent_vehicles.fusion.linear_fusion_cross import LinearFusionCross

            if self.certificate is None:
                raise ValueError("Linear fusion requires this vehicle's local certificate.")
            return LinearFusionCross(ego_source_id=self.name,
                                     gate_ratio=config.get("gate_ratio", 1.0))
        raise ValueError(f"Unsupported fusion type: {config['type']}")

    @staticmethod
    def _create_gate(config):
        options = dict(config)
        kind = options.pop("type", "none")
        if kind == "none":
            return None
        if kind == "kf":
            return KFGate(**options)
        if kind == "consensus":
            options.setdefault("min_overlap", 0.5)
            return ConsensusGate(**options)
        raise ValueError(f"Unsupported gate type: {kind}")

    def _apply_gates(self, pools, tracklets):
        """Gate local and shared-only nodes independently, preserving map order."""
        local = {node_id: values for node_id, values in pools.items() if values[3] == 1}
        shared = {node_id: values for node_id, values in pools.items() if values[3] == 2} \
                 if getattr(self, "category_s_enabled", True) else {}
        if self.kf_gate is not None:
            local = self.kf_gate.apply(local, tracklets)
        if self.consensus_gate is not None:
            shared = self.consensus_gate.apply(shared)
        selected = {**local, **shared}
        return {node_id: selected[node_id] for node_id in pools if node_id in selected}

    def fuse_pools(self, ego_ts, pools):
        """Time complete category batches, including worker dispatch and wait."""
        fused = {}
        self._fusion_runtime = {}
        for category, node_type in (("category_l", 1), ("category_s", 2)):
            selected = {node_id: values for node_id, values in pools.items()
                        if values[3] == node_type}
            fuser = self.fusers[category]
            if fuser is None:
                self._fusion_runtime[category] = {
                    "fusion_ms": 0.0, "candidate_nodes": 0, "fused_nodes": 0, "workers": 1,
                }
                continue
            start = time.perf_counter()
            outputs = fuser.fuse(ego_ts, selected)
            self._fusion_runtime[category] = {
                "fusion_ms": (time.perf_counter() - start) * 1000.0,
                "candidate_nodes": sum(bool(values[2]) for values in selected.values()),
                "fused_nodes": len(outputs),
                "workers": getattr(fuser, "workers", 1),
            }
            fused.update(outputs)
        return {node_id: fused[node_id] for node_id in pools if node_id in fused}

    def run_predictor(self, tracklets, sim_time_s):
        collaborative_start = time.perf_counter()

        # arrived packets at current sim time ------------------
        sim_ms = int(round(sim_time_s * 1000.0))
        arrived = self._listener.pop_arrived(sim_ms)
        # -----------------------------------------------------------------------------------

        # predictor input + forward
        individual_start = time.perf_counter()
        past_trajs = self.predictor.format_input(tracklets)
        mean_trajs, cov_trajs = self.predictor.predict(past_trajs)
        individual_ms = (time.perf_counter() - individual_start) * 1000.0

        # timestamp + ego time indices
        pred_ts_ms = int(round(sim_time_s * 1000.0))
        ego_ts = list(mean_trajs[0].keys())

        # prediction-map update + pool extraction
        self.prediction_map.update_by_predictor(tracklets, mean_trajs, cov_trajs, pred_ts_ms)
        self._annotate_certificates(tracklets, pred_ts_ms)
        for topic, payload in arrived:
            self.update_prediction_map(topic, payload, pred_ts_ms, ego_ts)
        self.last_prediction_timestamp = pred_ts_ms
        if getattr(self, "category_s_enabled", True):
            self.prediction_map.remove_unrefreshed_category_II_nodes()
        preds_with_pools = self.prediction_map.extract_pools()

        # Gates own filtering and their decision records.
        pools = self._apply_gates(preds_with_pools, tracklets)

        # Preserve the supplied inputs for evaluation before clearing pools.
        fusion_inputs = {node_id: deepcopy(values[2])
                         for node_id, values in pools.items() if values[2]}
        if self.fusion_observer is not None:
            self.fusion_observer(ego_ts, pools)

        fusion_start = time.perf_counter()
        fused_predictions = self.fuse_pools(ego_ts, pools)
        fusion_ms = (time.perf_counter() - fusion_start) * 1000.0
        fused_nodes = len(fused_predictions)

        # post-fusion prediction-map operations
        self.prediction_map.update_predictions(fused_predictions)
        predictions = self.prediction_map.extract_predictions(
            category_II_nodes=getattr(self, "category_s_enabled", True),
        )
        for prediction in predictions:
            prediction["ego_vehicle"] = self.name
            if prediction["id"] in fusion_inputs:
                prediction["fusion_inputs"] = fusion_inputs[prediction["id"]]
        if getattr(self, "category_s_enabled", True):
            self.prediction_map.advance_category_II_nodes()
        self.prediction_map.empty_pools()

        collaborative_ms = (time.perf_counter() - collaborative_start) * 1000.0

        self._record_runtime(individual_ms, collaborative_ms, fusion_ms, fused_nodes, self._fusion_runtime)
        return predictions

    def _record_runtime(self, individual_ms, collaborative_ms, fusion_ms, fused_nodes, category_runtime):
        """Append measured category costs and candidate/output counts per frame."""
        write_header = not self.runtime_path.exists() or self.runtime_path.stat().st_size == 0
        if not self._runtime_header_checked and not write_header:
            with self.runtime_path.open() as file:
                if file.readline() != _RUNTIME_HEADER:
                    raise ValueError(
                        f"Runtime CSV header mismatch: {self.runtime_path}. "
                        "Move the existing log before starting a new run."
                    )
        with self.runtime_path.open("a") as file:
            if write_header:
                file.write(_RUNTIME_HEADER)
            workers = max(values["workers"] for values in category_runtime.values())
            category_fields = ",".join(
                f"{values['fusion_ms']:.3f},{values['candidate_nodes']},{values['fused_nodes']},{values['workers']}"
                for values in (category_runtime["category_l"], category_runtime["category_s"])
            )
            file.write(f"{individual_ms:.3f},{collaborative_ms:.3f},{fusion_ms:.3f},"
                       f"{fused_nodes},{workers},{category_fields}\n")
        self._runtime_header_checked = True

    def update_prediction_map(self, topic, payload, ego_timestamp_ms, ego_timestamps):
        """
        Now called ONLY from the main thread (run_predictor) after delay-buffer release.
        """
        try:
            print(f"[{self.name}] recieved remote packet")

            broadcasting_timestamp = payload["timestamp_ms"]
            vehicle_parameters = {"fps": payload["fps"],
                                  "prediction_horizon": payload["pred_hz"],
                                  "prediction_sampling": payload["pred_sampling"]}
            vehicle_location = payload["ego_position"]
            shared_predictions = payload["predictions"]

            shared_predictions = PredictionTimeAligner.prepare(
                shared_predictions,
                ego_timestamp_ms,
                association_timestamp_ms=self.last_prediction_timestamp,
            )

            for prediction in shared_predictions:
                prediction["prediction"].update(
                    source_vehicle=payload["sender"], source_object_id=prediction["id"],
                    origin_ms=prediction["pred_ts_ms"],
                )

            if len(shared_predictions) > 0:
                matches, _, _ = self.prediction_map.match_shared_predictions(shared_predictions)
                matched = dict(matches)
                retained, retained_matches, unmatched = [], [], []
                for index, prediction in enumerate(shared_predictions):
                    node_id = matched.get(index)
                    node_type = (self.prediction_map.G.nodes[node_id]["node_data"].type
                                 if node_id is not None else 2)
                    category = "category_l" if node_type == 1 else "category_s"
                    if category == "category_s" and not getattr(self, "category_s_enabled", True):
                        continue
                    aligned = self.aligners[category].align([prediction], ego_timestamps)
                    if not aligned:
                        continue
                    new_index = len(retained)
                    retained.append(aligned[0])
                    if node_id is None:
                        unmatched.append(new_index)
                    else:
                        retained_matches.append((new_index, node_id))
                self.prediction_map.update_pools(retained_matches, retained)
                print(f"[{self.name}] processed remote packet: total {len(retained)}, associated {len(retained_matches)}")

                if self.cur_location and getattr(self, "category_s_enabled", True):
                    added_ids = self.prediction_map.add_new_objects(self.cur_location, unmatched, retained)
                    print(f"Total added nodes of category II: {len(added_ids)}")
            else:
                print(f"[{self.name}] processed remote packet: no relevant shared predictions")

        except Exception as e:
            print(f"[{self.name}] failed to process remote packet, {e}")
