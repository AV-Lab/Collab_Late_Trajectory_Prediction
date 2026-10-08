from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.paths import OUTPUTS_DIR, RUNTIME_TMP_DIR

import numpy as np

from evaluation.matching import match_predictions
from evaluation.metrics.distance_metrics import ade, fde, hit_rate, mse_2d
from evaluation.metrics.uncertainty_metrics import (
    CONFIDENCE_LEVELS,
    chi_square_thresholds,
    ece,
    msne,
)
from evaluation.plotting import (
    plot_calibration_curve,
    plot_runtime_stats,
    plot_uncertainty_metrics,
)
from evaluation.uncertainty_accumulator import (
    UncertaintyAccumulator,
    prepare_uncertainty_batch,
)

INPUT_UNCERTAINTY_METRICS = (
    "MSNE", "NLL_mean", "CE_mean", "AUSE_2D", "Spearman_2D", "coverage_ECE_L1",
)


def _runtime_summary(runtime_dir=None):
    """Read per-vehicle runtimes, averaging fusion only over active frames."""
    runtime_dir = Path(RUNTIME_TMP_DIR if runtime_dir is None else runtime_dir)
    summaries = {}
    required = {"individual_ms", "collaborative_ms"}
    fusion_columns = {"fusion_ms", "fused_nodes", "fusion_workers"}
    for path in sorted(runtime_dir.glob("*.csv")):
        frames = active_frames = active_nodes = 0
        individual_sum = collaborative_sum = fusion_sum = 0.0
        workers = set()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            if columns and not required.issubset(columns):
                raise ValueError(f"Missing runtime columns in {path}")
            if columns & fusion_columns and not fusion_columns.issubset(columns):
                raise ValueError(f"Incomplete fusion runtime columns in {path}")
            has_fusion = fusion_columns.issubset(columns)
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Runtime row does not match its header in {path}")
                individual = float(row["individual_ms"])
                collaborative = float(row["collaborative_ms"])
                if any(not math.isfinite(value) or value < 0 for value in (individual, collaborative)):
                    raise ValueError(f"Runtime durations must be finite and nonnegative in {path}")
                frames += 1
                individual_sum += individual
                collaborative_sum += collaborative
                if has_fusion:
                    fusion = float(row["fusion_ms"])
                    nodes = int(row["fused_nodes"])
                    worker_count = int(row["fusion_workers"])
                    if not math.isfinite(fusion) or fusion < 0 or nodes < 0 or worker_count < 1:
                        raise ValueError(f"Invalid fusion runtime values in {path}")
                    workers.add(worker_count)
                    if nodes > 0:
                        active_frames += 1
                        active_nodes += nodes
                        fusion_sum += fusion
        summaries[path.stem] = {
            "frames": frames,
            "predictor_mean_ms": individual_sum / frames if frames else None,
            "collaborative_mean_ms": collaborative_sum / frames if frames else None,
            "fusion_mean_ms": fusion_sum / active_frames if active_frames else None,
            "active_fusion_frames": active_frames if has_fusion else None,
            "mean_fused_nodes_active": active_nodes / active_frames if active_frames else None,
            "fusion_workers": sorted(workers),
        }
    return summaries


def _new_totals(uncertainty_options=None):
    return {
        "uncertainty": UncertaintyAccumulator(**(uncertainty_options or {})),
        "frames_seen": 0,
        "frames_with_matches": 0,
        "ade_sum": 0.0,
        "ade_count": 0,
        "fde_sum": 0.0,
        "fde_count": 0,
        "mse_2d_sum": 0.0,
        "mse_2d_count": 0,
        "hit_05_count": 0,
        "hit_10_count": 0,
        "mahalanobis_sum": 0.0,
        "mahalanobis_count": 0,
        "mahalanobis_1s_sum": 0.0,
        "mahalanobis_1s_count": 0,
        "mahalanobis_final_sum": 0.0,
        "mahalanobis_final_count": 0,
        "calibration_hits": np.zeros(len(CONFIDENCE_LEVELS), dtype=np.int64),
        "num_gt": 0,
        "num_matched": 0,
        "num_missed": 0,
        "num_raw_unmatched": 0,
        "num_false_positives": 0,
        "num_certified_retained_unmatched": 0,
        "num_stale_matched": 0,
        "horizon_steps_sum": 0,
        "horizon_count": 0,
        "horizon_steps_min": None,
        "horizon_steps_max": None,
        "num_full_horizon": 0,
        "num_partial_horizon": 0,
    }


def _safe_mean(total, count):
    return float(total / count) if count > 0 else float("nan")


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _json_value(value):
    """Serialize report values, representing undefined metrics as JSON null."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _distance_gains(before, after):
    return {
        metric + "_gain_percent": (
            100.0 * (before[metric] - after[metric]) / before[metric]
            if np.isfinite(before[metric]) and before[metric] > 0
            and np.isfinite(after[metric]) else float("nan")
        ) for metric in ("ADE", "FDE", "MSE_2D")
    }


def _summary(totals):
    uncertainty_count = totals["mahalanobis_count"]
    if uncertainty_count:
        coverage = (
            totals["calibration_hits"].astype(np.float64) / uncertainty_count
        )
        calibration = {
            "confidence": list(CONFIDENCE_LEVELS),
            "coverage": coverage.tolist(),
            "thresholds": chi_square_thresholds().tolist(),
        }
        ece_value = ece(coverage)
    else:
        calibration = None
        ece_value = float("nan")

    result = {
        "frames_seen": int(totals["frames_seen"]),
        "frames_with_matches": int(totals["frames_with_matches"]),
        "forecast_samples": int(totals["ade_count"]),
        "hit_05_count": int(totals["hit_05_count"]),
        "hit_10_count": int(totals["hit_10_count"]),
        "ADE": _safe_mean(totals["ade_sum"], totals["ade_count"]),
        "FDE": _safe_mean(totals["fde_sum"], totals["fde_count"]),
        "MSE_2D": _safe_mean(totals["mse_2d_sum"], totals["mse_2d_count"]),
        "Hit@0.5": _safe_ratio(totals["hit_05_count"], totals["num_gt"]),
        "Hit@1.0": _safe_ratio(totals["hit_10_count"], totals["num_gt"]),
        "MSNE": (
            msne([totals["mahalanobis_sum"] / uncertainty_count])
            if uncertainty_count else float("nan")
        ),
        "MSNE@1s": (
            msne([
                totals["mahalanobis_1s_sum"]
                / totals["mahalanobis_1s_count"]
            ])
            if totals["mahalanobis_1s_count"] else float("nan")
        ),
        "MSNE@final": (
            msne([
                totals["mahalanobis_final_sum"]
                / totals["mahalanobis_final_count"]
            ])
            if totals["mahalanobis_final_count"] else float("nan")
        ),
        "ECE": ece_value,
        "coverage_ECE_L1": ece_value,
        "calibration": calibration,
        "uncertainty_samples": int(uncertainty_count),
        "num_gt": int(totals["num_gt"]),
        "num_matched": int(totals["num_matched"]),
        "num_missed": int(totals["num_missed"]),
        "num_raw_unmatched": int(totals["num_raw_unmatched"]),
        "num_false_positives": int(totals["num_false_positives"]),
        "num_certified_retained_unmatched": int(
            totals["num_certified_retained_unmatched"]
        ),
        "num_stale_matched": int(totals["num_stale_matched"]),
        "gt_coverage": _safe_ratio(totals["num_matched"], totals["num_gt"]),
        "false_positive_percentage": _safe_ratio(
            totals["num_false_positives"],
            totals["num_matched"] + totals["num_false_positives"],
        ),
        "mean_horizon_steps": _safe_mean(
            totals["horizon_steps_sum"], totals["horizon_count"]
        ),
        "min_horizon_steps": totals["horizon_steps_min"],
        "max_horizon_steps": totals["horizon_steps_max"],
        "num_full_horizon": int(totals["num_full_horizon"]),
        "num_partial_horizon": int(totals["num_partial_horizon"]),
    }
    result.update(totals["uncertainty"].compute())
    return result


class Evaluator:
    """Online trajectory evaluator using one-to-one IoU association.

    Every matched forecast appearance is one ADE/FDE/MSE_2D sample. MSE_2D
    averages dx² + dy² over valid points, then weights forecasts equally.
    Hit rates count successful matched forecasts over all ground-truth
    trajectories, so missed
    ground truth contributes a failed hit. Uncertainty metrics pool valid
    forecast points across each reporting group. Forecasts are evaluated up to
    the available horizon, capped by ``prediction_horizon``. ``ECE`` remains a
    compatibility alias for ellipse ``coverage_ECE_L1``, distinct from CDF CE.

    Pass ``output_dir`` for isolated reports that do not clear shared runtime
    files. Outputs default to the repository-root ``outputs/`` directory.
    Explicit ``output_dir=None`` disables report and plot file output.
    Vehicle responses and predictor mode selection belong to the caller.
    """

    SOURCE_NAMES = (
        "category_I_ego",
        "category_I_fused",
        "category_II_fused",
        "no_fusion",
    )

    def __init__(
        self,
        prediction_horizon,
        observation_length,
        sample_fps=None,
        iou_threshold=0.7,
        covariance_jitter=1e-6,
        store_distributions=True,
        *,
        confidence_levels=CONFIDENCE_LEVELS,
        ce_weights=None,
        vehicle_label=None,
        output_dir=OUTPUTS_DIR,
        generate_plots=True,
        run_metadata=None,
        print_by_category=False,
    ):
        self.input_dimension = 2
        self.prediction_horizon = int(prediction_horizon)
        self.observation_length = int(observation_length)
        self.sample_fps = None if sample_fps is None else float(sample_fps)
        self.iou_threshold = float(iou_threshold)
        self.covariance_jitter = float(covariance_jitter)
        self.store_distributions = bool(store_distributions)
        self.vehicle_label = vehicle_label
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.generate_plots = bool(generate_plots)
        self.print_by_category = bool(print_by_category)
        self.run_metadata = dict(run_metadata or {})
        if self.sample_fps is not None and self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive when provided.")
        if not np.isfinite(self.covariance_jitter) or self.covariance_jitter <= 0:
            raise ValueError("covariance_jitter must be finite and positive.")

        # Resolve settings once so all groups use exactly the same protocol.
        self._uncertainty_options = {
            "confidence_levels": tuple(confidence_levels),
            "ce_weights": None if ce_weights is None else tuple(ce_weights),
        }

        self.overall = self._make_totals()
        self.by_category = defaultdict(self._make_totals)
        self.by_source = {name: self._make_totals() for name in self.SOURCE_NAMES}
        self.input_calibration_totals = defaultdict(self._make_totals)
        self.input_alignment_exclusions = defaultdict(Counter)
        self.scenarios = []
        self.distributions = {"ADE": [], "FDE": [], "MSE_2D": [], "horizon_steps": []}

        self._scenario_name = None
        self._scenario = None
        self._certified_nodes = {}
        if self.output_dir == OUTPUTS_DIR and self.generate_plots:
            # The legacy caller relies on construction starting a fresh run.
            # Isolated vehicle reports must never clear another run's files.
            from evaluation.cleanup import clear_runtime_stats

            clear_runtime_stats()

    def _make_totals(self):
        return _new_totals(self._uncertainty_options)

    def begin_scenario(self, scenario_name):
        if self._scenario is not None:
            raise RuntimeError("The previous scenario has not been ended.")
        self._scenario_name = str(scenario_name)
        self._scenario = self._make_totals()
        self._certified_nodes = {}

    def _sequence_to_array(self, sequence, max_steps, reverse=False):
        if sequence is None:
            return np.empty((0, self.input_dimension), dtype=np.float64)

        if isinstance(sequence, dict):
            keys = sorted(sequence.keys(), key=float)
            keys = keys[-max_steps:] if reverse else keys[:max_steps]
            values = [sequence[key] for key in keys]
        else:
            values = list(sequence)
            values = values[-max_steps:] if reverse else values[:max_steps]

        rows = []
        for value in values:
            if hasattr(value, "x"):
                value = [value.x, value.y, getattr(value, "z", 0.0)]
            row = np.asarray(value, dtype=np.float64).reshape(-1)
            if row.size < self.input_dimension:
                raise ValueError(
                    f"Trajectory point has {row.size} dimensions; "
                    f"{self.input_dimension} are required."
                )
            rows.append(row[:self.input_dimension])

        if not rows:
            return np.empty((0, self.input_dimension), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def _prediction_arrays(self, prediction):
        if not prediction or not isinstance(prediction.get("pred"), dict):
            return np.empty((0, self.input_dimension)), None, []

        mean_map = prediction["pred"]
        keys = sorted(mean_map.keys(), key=float)[:self.prediction_horizon]
        means = self._sequence_to_array(mean_map, self.prediction_horizon)

        covariance_map = prediction.get("cov")
        covariances = None
        if isinstance(covariance_map, dict):
            covariances = np.full((len(keys), 2, 2), np.nan, dtype=np.float64)
            for index, key in enumerate(keys):
                value = covariance_map.get(key)
                if value is not None:
                    matrix = np.asarray(value, dtype=np.float64)
                    if matrix.shape != (2, 2):
                        raise ValueError("Each position covariance must have shape [2, 2].")
                    covariances[index] = matrix

        return means, covariances, keys

    def _forecast_metrics(self, prediction, ground_truth_future, identifiers=None):
        means, covariances, timestamps = self._prediction_arrays(prediction)
        target = self._sequence_to_array(ground_truth_future, self.prediction_horizon)
        horizon = min(len(means), len(target), self.prediction_horizon)
        if horizon < 1:
            return None

        means = means[:horizon]
        target = target[:horizon]
        ade_value = ade(means, target)
        fde_value = fde(means, target)

        batch = prepare_uncertainty_batch(
            means,
            target,
            None if covariances is None else covariances[:horizon],
            jitter=self.covariance_jitter,
            identifiers=identifiers,
            timestamps=np.asarray(timestamps[:horizon], dtype=np.float64),
        )
        mahalanobis = batch.mahalanobis_squared

        one_second_index = None
        if self.sample_fps is not None:
            candidate = int(round(self.sample_fps)) - 1
            if 0 <= candidate < horizon:
                one_second_index = candidate

        return {
            "ADE": ade_value,
            "FDE": fde_value,
            "MSE_2D": mse_2d(means, target),
            "Hit@0.5": hit_rate([fde_value], 0.5),
            "Hit@1.0": hit_rate([fde_value], 1.0),
            "mahalanobis_squared": mahalanobis,
            "uncertainty_batch": batch,
            "one_second_index": one_second_index,
            "horizon_steps": horizon,
            "horizon_seconds": float(timestamps[horizon - 1]) if timestamps else None,
            "full_horizon": horizon == self.prediction_horizon,
            "prediction": means,
            "target": target,
        }

    def _add_sharing_calibration(self, prediction, gt_future, identifiers, has_ego):
        """Measure received inputs on exact GT timestamps, before any GP processing."""
        target = self._sequence_to_array(gt_future, self.prediction_horizon)
        gt_times = prediction.get("gt_future_timestamps_ms", [])[:len(target)]
        targets = dict(zip(gt_times, target))
        for remote in prediction.get("fusion_inputs", []):
            group = (str(remote["source_vehicle"]), "category_I" if has_ego else "category_II")
            totals = self.input_calibration_totals[group]
            excluded = self.input_alignment_exclusions[group]
            if not (len(remote["t"]) == len(remote["xy"]) == len(remote["cov"])):
                raise ValueError("Sharing input timestamps, means and covariances differ in length.")
            times = [int(remote["origin_ms"]) + int(round(1000 * float(t)))
                     if np.isfinite(float(t)) else None for t in remote["t"]]
            counts = Counter(times)
            selected = []
            for t, mean, covariance in zip(times, remote["xy"], remote["cov"]):
                if t is None:
                    excluded["nonfinite_timestamp"] += 1
                elif counts[t] > 1:
                    excluded["duplicate_timestamp"] += 1
                elif t not in targets:
                    excluded["no_matching_gt_timestamp"] += 1
                else:
                    selected.append((t, mean, covariance))
            if not selected:
                continue
            selected.sort(key=lambda item: item[0])
            aligned = {"pred": {t / 1000: mean for t, mean, _ in selected},
                       "cov": {t / 1000: cov for t, _, cov in selected}}
            values = self._forecast_metrics(
                aligned, [targets[t] for t, _, _ in selected],
                dict(identifiers, vehicle=group[0], source=group[1],
                     source_object_id=remote["source_object_id"]),
            )
            self._add_forecast(totals, values)

    def _add_counts(
        self,
        totals,
        num_gt=0,
        num_matched=0,
        num_missed=0,
        num_raw_unmatched=0,
        num_fp=0,
        num_certified_retained_unmatched=0,
        num_stale_matched=0,
    ):
        totals["num_gt"] += int(num_gt)
        totals["num_matched"] += int(num_matched)
        totals["num_missed"] += int(num_missed)
        totals["num_raw_unmatched"] += int(num_raw_unmatched)
        totals["num_false_positives"] += int(num_fp)
        totals["num_certified_retained_unmatched"] += int(
            num_certified_retained_unmatched
        )
        totals["num_stale_matched"] += int(num_stale_matched)

    def _add_forecast(self, totals, values):
        totals["uncertainty"].add_batch(values["uncertainty_batch"])
        totals["ade_sum"] += values["ADE"]
        totals["ade_count"] += 1
        totals["fde_sum"] += values["FDE"]
        totals["fde_count"] += 1
        totals["mse_2d_sum"] += values["MSE_2D"]
        totals["mse_2d_count"] += 1
        totals["hit_05_count"] += int(values["Hit@0.5"])
        totals["hit_10_count"] += int(values["Hit@1.0"])

        mahalanobis = values["mahalanobis_squared"]
        if mahalanobis is not None:
            mahalanobis = np.asarray(mahalanobis, dtype=np.float64)
            finite = mahalanobis[np.isfinite(mahalanobis)]
            if finite.size:
                totals["mahalanobis_sum"] += float(finite.sum())
                totals["mahalanobis_count"] += int(finite.size)
                thresholds = chi_square_thresholds()
                totals["calibration_hits"] += (
                    finite[:, None] <= thresholds[None, :]
                ).sum(axis=0)

                one_second_index = values["one_second_index"]
                if one_second_index is not None and np.isfinite(
                    mahalanobis[one_second_index]
                ):
                    totals["mahalanobis_1s_sum"] += float(
                        mahalanobis[one_second_index]
                    )
                    totals["mahalanobis_1s_count"] += 1

                if np.isfinite(mahalanobis[-1]):
                    totals["mahalanobis_final_sum"] += float(mahalanobis[-1])
                    totals["mahalanobis_final_count"] += 1

        horizon = int(values["horizon_steps"])
        totals["horizon_steps_sum"] += horizon
        totals["horizon_count"] += 1
        current_min = totals["horizon_steps_min"]
        current_max = totals["horizon_steps_max"]
        totals["horizon_steps_min"] = horizon if current_min is None else min(current_min, horizon)
        totals["horizon_steps_max"] = horizon if current_max is None else max(current_max, horizon)
        if values["full_horizon"]:
            totals["num_full_horizon"] += 1
        else:
            totals["num_partial_horizon"] += 1

    @staticmethod
    def _select_predictions(prediction):
        ego = prediction.get("prediction")
        fused = prediction.get("fused_prediction")
        has_ego = bool(ego and isinstance(ego.get("pred"), dict) and ego["pred"])
        has_fused = bool(fused and isinstance(fused.get("pred"), dict) and fused["pred"])
        primary = fused if has_fused else ego if has_ego else None
        return ego, fused, primary, has_ego, has_fused

    def compute(self, response, *, associations=None):
        if self._scenario is None:
            raise RuntimeError("begin_scenario() must be called before compute().")

        predictions = response["predictions"]
        tracklets = response["tracklets"]
        ground_truth = response["trajectories"]
        track_history = {tracklet["id"]: tracklet["tracklet"] for tracklet in tracklets}
        local_streak = {
            tracklet["id"]: int(tracklet["kf_gate"]["streak"])
            for tracklet in tracklets
        }

        # Paired covariance experiments reuse one association on identical boxes.
        matches, missed_ids, false_indices = associations if associations is not None else match_predictions(
            predictions, ground_truth, self.iou_threshold,
        )

        raw_unmatched_indices = list(false_indices)
        matched_prediction_indices = {
            prediction_index for _, prediction_index in matches
        }
        prediction_indices_by_id = {
            prediction["id"]: index
            for index, prediction in enumerate(predictions)
        }

        active_prediction_ids = set(prediction_indices_by_id)
        self._certified_nodes = {
            node_id: certified
            for node_id, certified in self._certified_nodes.items()
            if node_id in active_prediction_ids
        }
        for track_id, streak in local_streak.items():
            if streak == 0:
                prediction_index = prediction_indices_by_id[track_id]
                self._certified_nodes[track_id] = (
                    prediction_index in matched_prediction_indices
                )
            elif track_id not in self._certified_nodes:
                self._certified_nodes[track_id] = False

        category_II_ids = active_prediction_ids - set(local_streak)
        for node_id in category_II_ids:
            prediction_index = prediction_indices_by_id[node_id]
            if prediction_index in matched_prediction_indices:
                self._certified_nodes[node_id] = True
            elif node_id not in self._certified_nodes:
                self._certified_nodes[node_id] = False

        certified_retained_indices = [
            index
            for index in raw_unmatched_indices
            if self._certified_nodes[predictions[index]["id"]]
            and (
                predictions[index]["id"] not in local_streak
                or local_streak[predictions[index]["id"]] > 0
            )
        ]
        certified_retained_set = set(certified_retained_indices)
        false_indices = [
            index
            for index in raw_unmatched_indices
            if index not in certified_retained_set
        ]
        stale_matched = sum(
            predictions[prediction_index]["id"] in local_streak
            and local_streak[predictions[prediction_index]["id"]] > 0
            for _, prediction_index in matches
        )

        prediction_sources = {}
        for index, prediction in enumerate(predictions):
            ego, fused, _, has_ego, has_fused = self._select_predictions(prediction)
            if has_ego and has_fused:
                prediction_sources[index] = (
                    ("category_I_ego", ego),
                    ("category_I_fused", fused),
                )
            elif has_fused:
                prediction_sources[index] = (("category_II_fused", fused),)
            elif has_ego:
                prediction_sources[index] = (("no_fusion", ego),)
            else:
                prediction_sources[index] = ()

        matched_by_source = defaultdict(int)
        stale_by_source = defaultdict(int)
        for _, prediction_index in matches:
            for source, _ in prediction_sources[prediction_index]:
                matched_by_source[source] += 1
                prediction_id = predictions[prediction_index]["id"]
                if prediction_id in local_streak and local_streak[prediction_id] > 0:
                    stale_by_source[source] += 1

        raw_unmatched_by_source = defaultdict(int)
        false_by_source = defaultdict(int)
        retained_by_source = defaultdict(int)
        for prediction_index in raw_unmatched_indices:
            for source, _ in prediction_sources[prediction_index]:
                raw_unmatched_by_source[source] += 1
        for prediction_index in false_indices:
            for source, _ in prediction_sources[prediction_index]:
                false_by_source[source] += 1
        for prediction_index in certified_retained_indices:
            for source, _ in prediction_sources[prediction_index]:
                retained_by_source[source] += 1

        available_sources = {
            source
            for pairs in prediction_sources.values()
            for source, _ in pairs
        }
        for source in available_sources:
            matched_count = matched_by_source[source]
            totals = self.by_source[source]
            totals["frames_seen"] += 1
            if matched_count:
                totals["frames_with_matches"] += 1
            self._add_counts(
                totals,
                num_gt=len(ground_truth),
                num_matched=matched_count,
                num_missed=len(ground_truth) - matched_count,
                num_raw_unmatched=raw_unmatched_by_source[source],
                num_fp=false_by_source[source],
                num_certified_retained_unmatched=retained_by_source[source],
                num_stale_matched=stale_by_source[source],
            )

        self.overall["frames_seen"] += 1
        self._scenario["frames_seen"] += 1
        if matches:
            self.overall["frames_with_matches"] += 1
            self._scenario["frames_with_matches"] += 1

        self._add_counts(
            self.overall,
            num_gt=len(ground_truth),
            num_matched=len(matches),
            num_missed=len(missed_ids),
            num_raw_unmatched=len(raw_unmatched_indices),
            num_fp=len(false_indices),
            num_certified_retained_unmatched=len(certified_retained_indices),
            num_stale_matched=stale_matched,
        )
        self._add_counts(
            self._scenario,
            num_gt=len(ground_truth),
            num_matched=len(matches),
            num_missed=len(missed_ids),
            num_raw_unmatched=len(raw_unmatched_indices),
            num_fp=len(false_indices),
            num_certified_retained_unmatched=len(certified_retained_indices),
            num_stale_matched=stale_matched,
        )

        categories_in_frame = set()
        matched_visualization = []
        missed_visualization = []
        false_visualization = []
        certified_retained_visualization = []
        label_metrics = []

        for gt_id, prediction_index in matches:
            gt_object = ground_truth[gt_id]
            prediction = predictions[prediction_index]
            category = str(gt_object["category"]).lower()
            categories_in_frame.add(category)
            self._add_counts(self.by_category[category], num_gt=1, num_matched=1)
            if prediction["id"] in local_streak and local_streak[prediction["id"]] > 0:
                self._add_counts(self.by_category[category], num_stale_matched=1)

            ego, fused, primary, has_ego, has_fused = self._select_predictions(prediction)
            primary_source = (
                "category_I_fused" if has_fused and has_ego
                else "category_II_fused" if has_fused else "no_fusion"
            )
            if has_ego and has_fused:
                horizon = min(self.prediction_horizon, len(gt_object["future"]))
                ego_times = self._prediction_arrays(ego)[2][:horizon]
                fused_times = self._prediction_arrays(fused)[2][:horizon]
                if not np.array_equal(ego_times, fused_times):
                    raise ValueError("Category I ego/fused forecast timestamps or horizons differ; paired evaluation requires identical timestamps.")
            identifiers = {
                "vehicle": self.vehicle_label,
                "scenario": self._scenario_name,
                "frame_index": self._scenario["frames_seen"] - 1,
                "prediction_timestamp": prediction.get("timestamp"),
                "prediction_id": prediction["id"],
                "ground_truth_id": gt_id,
                "category": category,
            }
            primary_values = self._forecast_metrics(
                primary, gt_object["future"], dict(identifiers, source=primary_source)
            )
            if primary_values is not None:
                self._add_forecast(self.overall, primary_values)
                self._add_forecast(self._scenario, primary_values)
                self._add_forecast(self.by_category[category], primary_values)
                label_metrics.append({
                    "id": gt_id,
                    "category": category,
                    "ADE": primary_values["ADE"],
                    "FDE": primary_values["FDE"],
                    "MSE_2D": primary_values["MSE_2D"],
                    "horizon_steps": primary_values["horizon_steps"],
                })
                if self.store_distributions:
                    self.distributions["ADE"].append(primary_values["ADE"])
                    self.distributions["FDE"].append(primary_values["FDE"])
                    self.distributions["MSE_2D"].append(primary_values["MSE_2D"])
                    self.distributions["horizon_steps"].append(primary_values["horizon_steps"])

            for source, source_prediction in prediction_sources[prediction_index]:
                source_values = primary_values if source_prediction is primary else self._forecast_metrics(
                    source_prediction, gt_object["future"], dict(identifiers, source=source)
                )
                if source_values is not None:
                    self._add_forecast(self.by_source[source], source_values)
                    if source in ("category_I_ego", "no_fusion"):
                        vehicle = prediction.get("ego_vehicle", self.vehicle_label or "ego")
                        self._add_forecast(self.input_calibration_totals[(vehicle, "ego")], source_values)

            self._add_sharing_calibration(prediction, gt_object["future"], identifiers, has_ego)

            primary_covariance = primary.get("cov") if primary else None
            matched_visualization.append({
                "id": gt_id,
                "category": category,
                "gt": {
                    "past": self._sequence_to_array(
                        gt_object["past"], self.observation_length, reverse=True
                    ),
                    "bbox": gt_object["current_state"],
                    "future": self._sequence_to_array(
                        gt_object["future"], self.prediction_horizon
                    ),
                },
                "pred": {
                    "past": self._sequence_to_array(
                        track_history[prediction["id"]],
                        self.observation_length,
                        reverse=True,
                    ) if prediction["id"] in track_history else None,
                    "bbox": prediction["cur_location"],
                    "future": primary.get("pred") if primary else {},
                    "cov": primary_covariance,
                    "timestamp": prediction["timestamp"],
                },
            })

        for gt_id in missed_ids:
            gt_object = ground_truth[gt_id]
            category = str(gt_object["category"]).lower()
            categories_in_frame.add(category)
            self._add_counts(self.by_category[category], num_gt=1, num_missed=1)
            missed_visualization.append({
                "id": gt_id,
                "category": category,
                "gt": {
                    "past": self._sequence_to_array(
                        gt_object["past"], self.observation_length, reverse=True
                    ),
                    "bbox": gt_object["current_state"],
                    "future": self._sequence_to_array(
                        gt_object["future"], self.prediction_horizon
                    ),
                },
            })

        for prediction_index in false_indices:
            prediction = predictions[prediction_index]
            category = str(prediction["category"]).lower()
            categories_in_frame.add(category)
            self._add_counts(
                self.by_category[category],
                num_raw_unmatched=1,
                num_fp=1,
            )
            _, _, primary, _, _ = self._select_predictions(prediction)
            false_visualization.append({
                "id": prediction["id"],
                "category": category,
                "pred": {
                    "past": self._sequence_to_array(
                        track_history[prediction["id"]],
                        self.observation_length,
                        reverse=True,
                    ) if prediction["id"] in track_history else None,
                    "bbox": prediction["cur_location"],
                    "future": primary.get("pred") if primary else {},
                    "cov": primary.get("cov") if primary else None,
                    "timestamp": prediction["timestamp"],
                },
            })

        for prediction_index in certified_retained_indices:
            prediction = predictions[prediction_index]
            category = str(prediction["category"]).lower()
            categories_in_frame.add(category)
            self._add_counts(
                self.by_category[category],
                num_raw_unmatched=1,
                num_certified_retained_unmatched=1,
            )
            _, _, primary, _, _ = self._select_predictions(prediction)
            certified_retained_visualization.append({
                "id": prediction["id"],
                "category": category,
                "streak": (
                    local_streak[prediction["id"]]
                    if prediction["id"] in local_streak
                    else None
                ),
                "pred": {
                    "past": self._sequence_to_array(
                        track_history[prediction["id"]],
                        self.observation_length,
                        reverse=True,
                    ) if prediction["id"] in track_history else None,
                    "bbox": prediction["cur_location"],
                    "future": primary.get("pred") if primary else {},
                    "cov": primary.get("cov") if primary else None,
                    "timestamp": prediction["timestamp"],
                },
            })

        for category in categories_in_frame:
            self.by_category[category]["frames_seen"] += 1
        return {
            "matched": matched_visualization,
            "missed": missed_visualization,
            "false_positives": false_visualization,
            "certified_retained_unmatched": certified_retained_visualization,
            "label_metrics": label_metrics,
        }

    def end_scenario(self):
        if self._scenario is None:
            raise RuntimeError("No active scenario to end.")
        scenario_summary = _summary(self._scenario)
        scenario_summary["scenario"] = self._scenario_name
        self.scenarios.append(scenario_summary)

        print(
            f"[{self._scenario_name}] evaluation complete: "
            f"frames={scenario_summary['frames_seen']}, "
            f"samples={scenario_summary['forecast_samples']}"
        )

        self._scenario = None
        self._scenario_name = None
        return scenario_summary

    def _ego_observed_summary(self, category_i_source):
        """Pool disjoint forecast groups; count the shared GT denominator once."""
        groups = (self.by_source[category_i_source], self.by_source["no_fusion"])
        # Both stages use the same associations; prediction errors alone change.
        associations = (self.by_source["category_I_ego"], self.by_source["no_fusion"])
        num_gt = self.overall["num_gt"]
        matched = sum(group["num_matched"] for group in associations)
        false_positives = sum(group["num_false_positives"] for group in associations)
        result = {
            "num_gt": num_gt,
            "num_matched": matched,
            "num_missed": num_gt - matched,
            "num_false_positives": false_positives,
            "gt_coverage": _safe_ratio(matched, num_gt),
            "false_positive_percentage": _safe_ratio(false_positives, matched + false_positives),
        }
        for metric in ("ADE", "FDE", "MSE_2D"):
            name = metric.lower()
            total = sum(group[name + "_sum"] for group in groups)
            count = sum(group[name + "_count"] for group in groups)
            result[name + "_sum"] = total
            result[name + "_count"] = count
            result[metric] = _safe_mean(total, count)
        result["forecast_samples"] = result["ade_count"]
        for metric, key in (("Hit@0.5", "hit_05_count"), ("Hit@1.0", "hit_10_count")):
            result[key] = sum(group[key] for group in groups)
            result[metric] = _safe_ratio(result[key], num_gt)
        return result

    def evaluate(self, *, gt_control=None, print_results=True):
        if self._scenario is not None:
            raise RuntimeError("end_scenario() must be called before evaluate().")

        results = {
            "vehicle_label": self.vehicle_label,
            "metadata": self._metadata(),
            "overall": _summary(self.overall),
            "ego_observed_before": self._ego_observed_summary("category_I_ego"),
            "ego_observed_after": self._ego_observed_summary("category_I_fused"),
            "by_category": {
                category: _summary(totals)
                for category, totals in sorted(self.by_category.items())
            },
            "by_source": {
                source: _summary(totals)
                for source, totals in self.by_source.items()
            },
            "scenarios": list(self.scenarios),
            "distributions": {
                name: list(values) for name, values in self.distributions.items()
            } if self.store_distributions else {},
        }

        # Compatibility with the former after-fusion distance-only field.
        results["ego_observed"] = {key: results["ego_observed_after"][key]
                                   for key in ("ADE", "FDE", "MSE_2D", "ade_count", "fde_count", "mse_2d_count")}
        results["fusion_gains"] = {
            "ego_observed": _distance_gains(results["ego_observed_before"], results["ego_observed_after"]),
            "category_I": _distance_gains(results["by_source"]["category_I_ego"], results["by_source"]["category_I_fused"]),
        }
        results["ego_observed_pairing"] = "Common capped forecast timestamps checked per matched Category I object."
        plots = {}
        calibration_plot = None
        if self.generate_plots:
            if self.output_dir == OUTPUTS_DIR:
                # Preserve the legacy caller's plot destinations. Explicit
                # report directories never read or write shared runtime plots.
                plots = plot_runtime_stats()
                calibration_plot = plot_calibration_curve(results)
                if calibration_plot is not None:
                    plots.update(plot_uncertainty_metrics(results, calibration_plot.parent))
            elif self.output_dir is not None:
                calibration_plot = plot_calibration_curve(
                    results, output_path=self.output_dir / "ellipse_coverage.png"
                )
                plots.update(plot_uncertainty_metrics(results, self.output_dir))
            if calibration_plot is not None:
                plots["calibration"] = calibration_plot
        results["plots"] = plots
        input_groups = []
        for (vehicle, group), totals in sorted(self.input_calibration_totals.items()):
            summary = _summary(totals)
            input_groups.append({
                "vehicle": vehicle, "group": group,
                **{key: summary[key] for key in INPUT_UNCERTAINTY_METRICS},
                "forecast_samples": summary["forecast_samples"],
                "valid_points": summary["uncertainty_valid_count"],
                "covariance_exclusions": summary["uncertainty_exclusion_counts"],
                "alignment_exclusions": dict(self.input_alignment_exclusions[(vehicle, group)]),
            })
        results["input_calibration"] = {
            "stage": "Ego predictions and received sharing predictions before GP processing",
            "scope": "Matched objects; sharing groups use the existing fusion-target GT association",
            "sample_unit": "Forecast point per evaluated input appearance; repeated inputs count again",
            "metadata": self._metadata(), "groups": input_groups,
        }
        if gt_control is not None:
            results["gt_control"] = gt_control
        results["runtime"] = (
            _runtime_summary() if self.output_dir == OUTPUTS_DIR and self.generate_plots else {}
        )
        if self.output_dir is not None:
            report_path = self.output_dir / "metrics.json"
            results["report_path"] = report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(_json_value(results), indent=2, allow_nan=False) + "\n")
            (self.output_dir / "input_calibration.json").write_text(
                json.dumps(_json_value(results["input_calibration"]), indent=2, allow_nan=False) + "\n")
        if print_results:
            self._print_results(results)
        return results

    def _metadata(self):
        protocol = self.overall["uncertainty"].metadata()
        protocol.update({
            "covariance_jitter": self.covariance_jitter,
            "covariance_preparation": "symmetrize_then_add_jitter_once",
            "position_units": "metres",
            "covariance_units": "metres_squared",
            "coordinate_frame": "input_xy (COLTP supplies world coordinates)",
            "prediction_horizon_steps": self.prediction_horizon,
            "observation_length_steps": self.observation_length,
            "sample_fps": self.sample_fps,
            "horizon_policy": "available_future_capped_at_prediction_horizon",
            "sample_unit": "matched_forecast_appearance_at_one_future_timestep",
            "mode_selection": "provided_by_caller; evaluator_does_not_select_modes",
            "ellipse_confidence_levels": list(CONFIDENCE_LEVELS),
            "ECE_alias": "coverage_ECE_L1 (not scalar predictive-CDF CE)",
        })
        return {"uncertainty_protocol": protocol, "run": dict(self.run_metadata)}

    def uncertainty_records(self, source=None, include_excluded=False):
        """Iterate identified samples without requiring a file or plot export."""
        totals = self.overall if source is None else self.by_source[source]
        return totals["uncertainty"].sample_records(include_excluded=include_excluded)

    @staticmethod
    def _metric(value, percentage=False):
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{100.0 * value:.2f}%" if percentage else f"{value:.4f}"

    def _print_metric_tables(self, title, rows):
        width = max([32] + [len(label) for label, _ in rows])
        print(f"\n================ {title}: ACCURACY & ASSOCIATION ================")
        print(
            f"{'Group':<{width}} {'ADE(m)':>9} {'FDE(m)':>9} {'MSE_2D(m²)':>13} "
            f"{'Hit@0.5':>10} {'Hit@1.0':>10} {'Coverage':>10} {'False Pos.':>11}"
        )
        for label, values in rows:
            print(
                f"{label:<{width}} "
                f"{self._metric(values['ADE']):>9} "
                f"{self._metric(values['FDE']):>9} "
                f"{self._metric(values['MSE_2D']):>13} "
                f"{self._metric(values['Hit@0.5'], True):>10} "
                f"{self._metric(values['Hit@1.0'], True):>10} "
                f"{self._metric(values['gt_coverage'], True):>10} "
                f"{self._metric(values['false_positive_percentage'], True):>11}"
            )

    def _print_uncertainty_table(self, title, rows):
        width = max([40] + [len(label) for label, _ in rows])
        print(f"\n================ {title}: UNCERTAINTY METRICS ====================")
        print(f"{'Group':<{width}} " + " ".join(f"{name:>16}" for name in INPUT_UNCERTAINTY_METRICS))
        for label, values in rows:
            print(f"{label:<{width}} " + " ".join(
                f"{self._metric(values[name]):>16}" for name in INPUT_UNCERTAINTY_METRICS))

    def _print_results(self, results):
        if self.vehicle_label is not None:
            print(f"\nVehicle: {self.vehicle_label}")
        variants = [("", results)]
        if "gt_control" in results:
            variants = [(" / without GT calibration", results), (" / with GT calibration", results["gt_control"])]
            print("GT calibration: per-point future-GT covariance, Category I ego and sharing inputs; Category II sharing inputs.")
        system_rows = [("Ego-observed: before", results["ego_observed_before"])]
        for key, label in (("ego_observed_after", "Ego-observed: after"),
                           ("overall", "Overall (includes Category II)")):
            system_rows.extend((label + suffix, values[key]) for suffix, values in variants)
        self._print_metric_tables("SYSTEM", system_rows)

        print("\n================ FUSION BREAKDOWN ================")
        width = max(32, len("Category II: fused") + max(len(suffix) for suffix, _ in variants))
        print(f"{'Group':<{width}} {'ADE(m)':>9} {'FDE(m)':>9} {'MSE_2D(m²)':>13} "
              f"{'ADE gain (%)':>14} {'FDE gain (%)':>14} {'MSE_2D gain (%)':>17}")
        for source, label in (("category_I_ego", "Category I: ego"),
                              ("category_I_fused", "Category I: fused"),
                              ("category_II_fused", "Category II: fused")):
            for suffix, variant in variants if source in ("category_I_fused", "category_II_fused") else [("", results)]:
                values = variant["by_source"][source]
                gain = variant["fusion_gains"]["category_I"]
                gain_text = [self._metric(gain[metric + "_gain_percent"])
                             if source == "category_I_fused" else "--" for metric in ("ADE", "FDE", "MSE_2D")]
                print(f"{label + suffix:<{width}} {self._metric(values['ADE']):>9} {self._metric(values['FDE']):>9} "
                      f"{self._metric(values['MSE_2D']):>13} "
                      f"{gain_text[0]:>14} {gain_text[1]:>14} {gain_text[2]:>17}")

        input_rows = []
        for row in results["input_calibration"]["groups"]:
            group = row["group"]
            labels = [("all ego inputs", False), ("Category I ego inputs", True)] if group == "ego" else [
                ({"category_I": "Category I", "category_II": "Category II"}[group] + " sharing inputs", False)]
            for label, category_i_ego in labels:
                for suffix, variant in variants:
                    values = variant["by_source"]["category_I_ego"] if category_i_ego else next(
                        item for item in variant["input_calibration"]["groups"]
                        if (item["vehicle"], item["group"]) == (row["vehicle"], group))
                    input_rows.append((row["vehicle"] + ": " + label + suffix, values))
        self._print_uncertainty_table("INPUT CALIBRATION", input_rows)

        category_rows = list(results["by_category"].items())
        if self.print_by_category and category_rows:
            self._print_metric_tables("PER CATEGORY", category_rows)
            self._print_uncertainty_table("PER CATEGORY", category_rows)

        calibration_plot = results.get("plots", {}).get("calibration")
        if calibration_plot is not None:
            print(f"\nCalibration curve saved to: {calibration_plot}")

        runtime = results.get("runtime", {})
        if runtime:
            width = max(12, max(map(len, runtime)))
            print("\n================ RUNTIME: MEAN MILLISECONDS ================")
            print(f"{'Vehicle':<{width}} {'Workers':>7} {'Frames':>7} {'Predictor':>11} "
                  f"{'Collaborative':>13} {'Fusion active':>13} {'Active frames':>13} {'Nodes/active':>12}")
            for vehicle, values in runtime.items():
                workers = ",".join(map(str, values["fusion_workers"])) or "n/a"
                active = values["active_fusion_frames"]
                print(f"{vehicle:<{width}} {workers:>7} {values['frames']:>7} "
                      f"{self._metric(values['predictor_mean_ms']):>11} "
                      f"{self._metric(values['collaborative_mean_ms']):>13} "
                      f"{self._metric(values['fusion_mean_ms']):>13} "
                      f"{active if active is not None else 'n/a':>13} "
                      f"{self._metric(values['mean_fused_nodes_active']):>12}")
            print("Predictor/collaborative: all frames; collaborative includes ego prediction plus collaboration.")
            print("Fusion: active frames only; startup and dispatch/wait included, no warmup removed.")
            print("Original pass only; GT-control replay excluded. Nodes/active is the mean fused-node count per active frame.")
