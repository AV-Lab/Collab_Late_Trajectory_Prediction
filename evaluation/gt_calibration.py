"""Pointwise GT covariance control and matched replay of an original fusion frame."""

from collections import Counter
from copy import deepcopy

import numpy as np
import torch

from evaluation.uncertainty_accumulator import prepare_uncertainty_batch


def pointwise_gt_covariances(means, covariances, timestamps_ms, targets_by_timestamp,
                            *, max_steps, jitter=1e-6, variance_floor=1e-6):
    """Replace supported, originally valid covariances with per-point GT error.

    Each replacement is ``diag(max((target - mean)**2, variance_floor))``;
    no residuals are averaged or centered. Absolute timestamps must match GT
    exactly. Duplicate/nonfinite timestamps, absent GT, and invalid uncertainty
    samples retain their original values. The returned list is a separate copy;
    an entirely missing covariance input returns ``None`` unchanged.
    """
    if not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, bool) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer.")
    variance_floor = float(variance_floor)
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("Variance floor must be finite and positive.")
    jitter = float(jitter)
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("Covariance jitter must be finite and nonnegative.")
    means = np.asarray(means, dtype=np.float64)
    if means.shape == (0,):
        means = means.reshape(0, 2)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("Means must have shape [N, 2].")
    count = len(means)
    timestamps_ms = list(timestamps_ms)
    if len(timestamps_ms) != count:
        raise ValueError("Means, covariances, and timestamps must have equal lengths.")
    copied = None if covariances is None else deepcopy(list(covariances))
    if copied is not None and len(copied) != count:
        raise ValueError("Means, covariances, and timestamps must have equal lengths.")
    covariance_array = np.full((count, 2, 2), np.nan, dtype=np.float64)
    for index, value in enumerate(copied or []):
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (2, 2):
                raise ValueError("Each position covariance must have shape [2, 2].")
            covariance_array[index] = matrix
    times = []
    for value in timestamps_ms:
        timestamp = float(value) if value is not None else np.nan
        times.append(timestamp if np.isfinite(timestamp) else None)
    counts = Counter(times)
    exclusions = Counter()
    selected = []
    for index, timestamp in enumerate(times):
        if timestamp is None:
            exclusions["nonfinite_timestamp"] += 1
        elif counts[timestamp] > 1:
            exclusions["duplicate_timestamp"] += 1
        elif timestamp not in targets_by_timestamp:
            exclusions["no_matching_gt_timestamp"] += 1
        else:
            selected.append(index)
    selected.sort(key=lambda index: times[index])
    selected = selected[:max_steps]
    audit = dict(total_points=count, modified_points=0, unmodified_points=count,
                 modified_indices=[], supported_timestamps_ms=[],
                 alignment_exclusions=dict(exclusions), covariance_exclusions={},
                 variance_before_floor=None, variance_after_floor=None,
                 floored_coordinates=0)
    if selected:
        batch = prepare_uncertainty_batch(
            means[selected],
            np.asarray([targets_by_timestamp[times[index]] for index in selected], dtype=np.float64),
            covariance_array[selected], jitter=jitter,
            timestamps=np.asarray([times[index] / 1000 for index in selected]),
        )
        audit["covariance_exclusions"] = batch.exclusion_counts
        indices = [selected[index] for index in batch.valid_indices]
        if indices:
            with np.errstate(over="ignore", invalid="ignore"):
                variances = (batch.targets - batch.means) ** 2
            if not np.isfinite(variances).all():
                raise ValueError("Nonfinite GT-derived variance.")
            floored = np.maximum(variances, variance_floor)
            for index, variance in zip(indices, floored):
                copied[index] = np.diag(variance).tolist()
            audit.update(modified_points=len(indices), unmodified_points=count - len(indices),
                         modified_indices=indices,
                         supported_timestamps_ms=[times[index] for index in indices],
                         variance_before_floor=variances.tolist(),
                         variance_after_floor=floored.tolist(),
                         floored_coordinates=int(np.count_nonzero(variances < variance_floor)))
    return copied, audit


class GTCalibrationExperiment:
    """Replay fusion with GT-derived covariance copies on fixed associations."""

    def __init__(self, fuse_pools, ego_vehicle, prediction_horizon, covariance_jitter=1e-6):
        self.fuse_pools = fuse_pools
        self.ego_vehicle = ego_vehicle
        self.prediction_horizon = prediction_horizon
        self.covariance_jitter = covariance_jitter
        self._frame = None

    def capture(self, ego_ts, pools):
        """Capture the complete ordered frame before the original fusion call."""
        self._frame = (list(ego_ts), deepcopy(pools), torch.get_rng_state().clone())

    def build_response(self, response, matches):
        """Fuse covariance-modified copies on the original evaluator associations.

        Category I updates ego and sharing covariance; Category II updates
        sharing covariance only because it has no ego forecast.
        The normal response and live prediction map retain the original run.
        The control uses the same GP initialization without advancing its RNG.
        """
        if self._frame is None:
            raise RuntimeError("GT calibration requires an enabled, unconsumed prediction frame.")
        ego_ts, inputs, rng_state = self._frame
        self._frame = None
        controlled = dict(response, predictions=deepcopy(response["predictions"]))
        audits, changed_ids = [], set()

        for gt_id, index in matches:
            prediction = controlled["predictions"][index]
            node_id = prediction["id"]
            if node_id not in inputs:
                continue
            timestamp, ego, pool, node_type = inputs[node_id]
            if node_type not in (1, 2) or not pool or not prediction.get("fused_prediction"):
                continue
            future = response["trajectories"][gt_id]["future"][:self.prediction_horizon]
            targets = dict(zip(prediction["gt_future_timestamps_ms"][:len(future)],
                               [np.asarray(point, dtype=np.float64)[:2] for point in future]))

            def replace(means, covariances, times, origin_ms, role, vehicle, object_id, pool_index):
                absolute_times = [int(origin_ms) + int(round(1000 * float(t)))
                                  if np.isfinite(float(t)) else None for t in times]
                new_covariances, audit = pointwise_gt_covariances(
                    means, covariances, absolute_times, targets,
                    max_steps=self.prediction_horizon, jitter=self.covariance_jitter,
                )
                audit.update(role=role, source_vehicle=vehicle, source_object_id=object_id,
                             target_id=node_id, ground_truth_id=gt_id, origin_ms=origin_ms,
                             pool_index=pool_index,
                             fusion_category="category_I" if node_type == 1 else "category_II")
                audits.append(audit)
                return new_covariances

            if node_type == 1:
                keys = sorted(ego["pred"], key=float)
                covariances = replace(
                    [ego["pred"][key] for key in keys], [ego["cov"][key] for key in keys],
                    keys, timestamp, "ego", self.ego_vehicle, node_id, None,
                )
                for key, covariance in zip(keys, covariances):
                    ego["cov"][key] = covariance
                prediction["prediction"] = ego
            for pool_index, remote in enumerate(pool):
                remote["cov"] = replace(
                    remote["xy"], remote["cov"], remote["t"], remote["origin_ms"],
                    "sharing", remote["source_vehicle"], remote["source_object_id"], pool_index,
                )
            prediction["fusion_inputs"] = pool
            changed_ids.add(node_id)

        if changed_ids:
            # Replay the whole ordered frame so every node keeps its GP RNG position.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(rng_state)
                fused = self.fuse_pools(ego_ts, inputs)
            for prediction in controlled["predictions"]:
                if prediction["id"] in changed_ids:
                    prediction["fused_prediction"] = fused[prediction["id"]]
        controlled["gt_control_changes"] = audits
        return controlled
