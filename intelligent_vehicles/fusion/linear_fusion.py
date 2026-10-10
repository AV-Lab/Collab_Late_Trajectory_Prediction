"""Category L fusion with constant scalar weights per availability interval.

For each source, aggregate upper squared-error risk over the interval, then
normalize inverse risks to obtain one weight used at every timestamp. Accept
only when mean fused upper risk is strictly below the configured fraction of
mean ego lower risk. An optional growing cone clips the accepted correction;
the risk gate describes the candidate before clipping.
L/U risks arrive in square metres with each source's covariance. The expected
squared-error argument assumes valid conditional second-moment bounds,
interpolation transfer and zero cross-source error moments; calibration does
not prove them.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibration.runtime import valid_bounds


_NUMERICAL_ERRORS = (
    ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError,
)
_PACKET_ERRORS = (*_NUMERICAL_ERRORS, KeyError, IndexError)


@dataclass
class _PreparedPoint:
    source_id: str
    mean: np.ndarray
    covariance: np.ndarray
    symmetrized: bool
    horizon_ms: Optional[int]
    is_native: Optional[bool]


class LinearFusion:
    """Fuse aligned Category L pools, preserving ego on rejected intervals.

    One coherent packet is selected per sharing source before gating. Intervals
    are consecutive ego timestamps with the same available source set. Input
    means and covariances are already aligned; this class does no interpolation.
    Rejected nodes are omitted so the prediction map retains the original ego.
    ``decisions[node_id]`` contains the selected packets and interval records.
    """

    def __init__(self, ego_source_id, *, gate_ratio=1.0, cap=None,
                 rtol=1e-10, atol=1e-12):
        if not self._valid_source_id(ego_source_id):
            raise ValueError("ego_source_id must be a nonempty string.")
        if not self._valid_factor(rtol) or not self._valid_factor(atol):
            raise ValueError("rtol and atol must be finite nonnegative scalars.")
        if not self._valid_factor(gate_ratio) or not 0 < gate_ratio <= 1:
            raise ValueError("gate_ratio must be finite and in (0, 1].")
        if cap is not None and (
            not isinstance(cap, Mapping) or set(cap) != {"radius_m", "horizon_s"}
            or any(not self._valid_factor(value) or value <= 0 for value in cap.values())
        ):
            raise ValueError("cap requires finite positive radius_m and horizon_s.")
        self.ego_source_id = ego_source_id
        self.gate_ratio = float(gate_ratio)
        self.cap = {key: float(value) for key, value in cap.items()} if cap is not None else None
        self.rtol, self.atol = float(rtol), float(atol)
        self.decisions = {}

    def fuse(self, ego_ts, local_pools):
        """Return accepted full trajectories; untouched timestamps stay ego."""
        self.decisions = {}
        queries = self._time_index(ego_ts)
        outputs = {}
        for node_id, (origin, ego, packets, node_type) in local_pools.items():
            if node_type != 1:
                raise ValueError("LinearFusion accepts Category L only (type=1).")
            if not self._valid_timestamp_ms(origin):
                raise ValueError("Ego forecast origin must be an integer millisecond timestamp.")
            output, decision = self._fuse_node(queries, origin, ego, packets)
            self.decisions[node_id] = decision
            if output is not None:
                outputs[node_id] = output
        return outputs

    @staticmethod
    def _valid_source_id(value):
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _valid_timestamp_ms(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

    @staticmethod
    def _valid_factor(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and value >= 0
        )

    @staticmethod
    def _validate_mean(value):
        try:
            mean = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Mean must be a finite vector of shape (2,).") from error
        if mean.shape != (2,) or not np.isfinite(mean).all():
            raise ValueError("Mean must be a finite vector of shape (2,).")
        return mean

    @classmethod
    def _prepare_point(cls, point):
        """Validate a finite position and SPD covariance without changing input."""
        mean = cls._validate_mean(point.get("mean"))
        covariance = np.asarray(point.get("cov"), dtype=np.float64)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise ValueError("Covariance must be a finite matrix of shape (2, 2).")

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            tolerance = 1e-12 + 1e-10 * np.max(np.abs(covariance))
            if np.max(np.abs(covariance - covariance.T)) > tolerance:
                raise ValueError("Covariance must be symmetric.")
            symmetrized = not np.array_equal(covariance, covariance.T)
            covariance = 0.5 * covariance + 0.5 * covariance.T
            np.linalg.cholesky(covariance)

        return _PreparedPoint(
            source_id=point["source_id"], mean=mean, covariance=covariance,
            symmetrized=symmetrized,
            horizon_ms=point.get("native_horizon_ms"), is_native=point.get("is_native"),
        )

    def _passes_gate(self, risk_upper_bound, ego_risk_lower_bound):
        """Require U < gate_ratio * L0 with a floating-point margin."""
        threshold = self.gate_ratio * ego_risk_lower_bound
        margin = self.atol + self.rtol * max(
            abs(threshold), abs(risk_upper_bound),
        )
        return threshold - risk_upper_bound > margin

    @staticmethod
    def _time_index(timestamps):
        """Map seconds to repository millisecond precision, rejecting collisions."""
        result = {}
        for timestamp in timestamps:
            try:
                milliseconds = float(timestamp) * 1000.0
                if not np.isfinite(milliseconds):
                    raise ValueError
                key = int(round(milliseconds))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Timestamps must be finite seconds.") from error
            if key in result:
                raise ValueError("Duplicate timestamps at millisecond precision.")
            result[key] = timestamp
        return result

    def _index_sharing_pool(self, pool):
        """Index each sharing forecast once, keeping the original pool order."""
        indexed_pool = []
        unavailable = []
        for index, prediction in enumerate(pool):
            try:
                times = prediction["t"]
                if (len(times) != len(prediction["xy"])
                        or len(times) != len(prediction["cov"])):
                    raise ValueError("Sharing arrays must have equal lengths.")
                native_mask = prediction.get("native_point_mask")
                if native_mask is not None and (
                    len(native_mask) != len(times)
                    or any(not isinstance(flag, (bool, np.bool_)) for flag in native_mask)
                ):
                    raise ValueError("Native-point flags must be booleans matching the sharing timeline.")
                time_index = self._time_index(times)
                point_indices = {time: i for i, time in enumerate(time_index)}
                indexed_pool.append((prediction, point_indices))
            except (KeyError, TypeError, ValueError, OverflowError):
                unavailable.append({"index": index, "reason": "invalid_sharing_timeline"})
        return indexed_pool, unavailable

    def _fuse_node(self, queries, origin, ego, packets):
        means = self._time_index(ego["pred"])
        if not queries.keys() <= means.keys():
            raise ValueError("Each query must exactly match an ego timestamp.")
        times = sorted(means)
        covariances = ego.get("cov")
        cov_keys = self._time_index(covariances) if isinstance(covariances, Mapping) else {}
        decision = {
            "used_sharing": False, "gate_passed": False,
            "reason": "no_usable_packet", "selected_packets": {},
            "query_times_s": [t / 1000. for t in times],
            "intervals": [], "excluded_packets": [], "max_correction_m": 0.,
        }
        prepared = self._select_packets(
            packets, queries, times, origin, ego, means, cov_keys, decision,
        )
        output = deepcopy(ego)
        for interval_times, sharing_sources in self._partition(times, prepared):
            sources = [self.ego_source_id, *sharing_sources]
            interval = {
                "query_times_s": [t / 1000. for t in interval_times],
                "source_ids": sources,
                "weights": {self.ego_source_id: 1.} if not sharing_sources else {},
                "U": None, "L0": None, "gate_passed": False,
                "gate_ratio": self.gate_ratio, "gate_threshold": None,
                "cap": self.cap, "capped_points": 0,
                "risk_applies_to": "uncapped_candidate",
                "reason": "ego_only" if not sharing_sources else "not_evaluated",
                "risk_per_point": [],
            }
            decision["intervals"].append(interval)
            if not sharing_sources or not self._evaluate_interval(interval_times, prepared, interval):
                continue
            updates = self._fused_points(interval_times, prepared, interval)
            if updates is None:
                interval.update(gate_passed=False, reason="numerical_output_failure")
                continue
            for t, mean, covariance, correction in updates:
                output["pred"][means[t]] = mean.tolist()
                output["cov"][cov_keys.get(t, means[t])] = covariance.tolist()
                decision["max_correction_m"] = max(decision["max_correction_m"], correction)
            decision["used_sharing"] = True
        decision["gate_passed"] = decision["used_sharing"]
        if decision["used_sharing"]:
            decision["reason"] = "accepted"
            for key in ("bounds", "certificate_id", "certificate_method", "motion_tag", "risk_unit"):
                output.pop(key, None)
            return output, decision
        if prepared:
            decision["reason"] = "no_interval_passed"
        return None, decision

    def _select_packets(self, packets, queries, times, origin, ego, means,
                        cov_keys, decision):
        """Keep the last usable whole packet per source in pool order."""
        prepared = {}
        for index in range(len(packets) - 1, -1, -1):
            packet = packets[index]
            source = packet.get("source_vehicle") if isinstance(packet, Mapping) else None
            if isinstance(source, str) and source in prepared:
                continue
            try:
                points = self._prepare_packet(packet, queries, times, origin, ego,
                                              means, cov_keys)
            except _PACKET_ERRORS as error:
                decision["excluded_packets"].append({"index": index, "reason": str(error)})
                continue
            prepared[source] = points
            decision["selected_packets"][source] = {
                "pool_index": index, "source_vehicle": source,
                "forecast_origin_ms": int(packet["forecast_origin_ms"]),
                "source_object_id": str(packet.get("source_object_id", "")),
                "certificate_id": packet.get("certificate_id"),
                "certificate_method": packet.get("method"),
                "motion_tag": packet.get("motion_tag"),
            }
        return prepared

    def _prepare_packet(self, packet, queries, times, origin, ego, means,
                        cov_keys):
        if not isinstance(packet, Mapping):
            raise ValueError("invalid_packet")
        source = packet.get("source_vehicle")
        if not self._valid_source_id(source) or source == self.ego_source_id:
            raise ValueError("invalid_sharing_source")
        if not self._valid_timestamp_ms(packet.get("forecast_origin_ms")):
            raise ValueError("invalid_sharing_forecast_origin")
        indexed, unavailable = self._index_sharing_pool([packet])
        if unavailable or not indexed:
            raise ValueError("invalid_sharing_timeline")
        _, indices = indexed[0]
        support = [t for t in times if t in indices and t in queries]
        if not support:
            raise ValueError("no_aligned_support")
        native = packet.get("native_point_mask")
        if native is None:
            raise ValueError("missing_native_point_metadata")
        bounds = packet.get("bounds")
        horizons = packet.get("native_horizon_ms")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != len(packet["t"]):
            raise ValueError("invalid_sharing_bounds")
        if any(value is not None and not valid_bounds(value) for value in bounds):
            raise ValueError("invalid_certificate_bounds")
        if packet.get("risk_unit") != "m2" or ego.get("risk_unit") != "m2":
            raise ValueError("invalid_certificate_risk_unit")
        if (not isinstance(horizons, (list, tuple)) or len(horizons) != len(packet["t"])
                or any(not self._valid_timestamp_ms(h) or h <= 0 for h in horizons)):
            raise ValueError("invalid_native_horizons")
        ego_bounds_map = ego.get("bounds")
        ego_bounds_map = ego_bounds_map if isinstance(ego_bounds_map, Mapping) else {}
        ego_bound_keys = self._time_index(ego_bounds_map)
        points = {}
        for t in support:
            q = indices[t]
            ego_point = self._prepare_point({
                "source_id": self.ego_source_id, "mean": ego["pred"][means[t]],
                "cov": ego["cov"][cov_keys[t]] if t in cov_keys else None,
                "native_horizon_ms": t, "is_native": True,
            })
            shared = self._prepare_point({
                "source_id": source, "mean": packet["xy"][q], "cov": packet["cov"][q],
                "is_native": native[q],
                "native_horizon_ms": horizons[q],
            })
            ego_bounds = ego_bounds_map.get(ego_bound_keys.get(t))
            shared_bounds = bounds[q]
            if not valid_bounds(ego_bounds) or shared_bounds is None:
                continue
            upper_ego = float(ego_bounds["upper"])
            upper_shared = float(shared_bounds["upper"])
            lower = float(ego_bounds["lower"])
            if not np.isfinite([upper_ego, upper_shared, lower]).all():
                raise ValueError("nonfinite_certificate_risk")
            points[t] = {"ego": ego_point, "shared": shared,
                         "ego_upper": upper_ego, "shared_upper": upper_shared,
                         "ego_lower": lower}
        if not points:
            raise ValueError("no_certified_support")
        return points

    @staticmethod
    def _partition(times, prepared):
        intervals = []
        for t in times:
            sources = tuple(sorted(source for source, points in prepared.items() if t in points))
            if not intervals or sources != intervals[-1][1]:
                intervals.append(([t], sources))
            else:
                intervals[-1][0].append(t)
        return intervals

    def _evaluate_interval(self, times, prepared, interval):
        sources = interval["source_ids"]
        rows = interval["risk_per_point"]
        for t in times:
            ego = prepared[sources[1]][t]
            rows.append({
                "time_s": t / 1000.,
                "upper_by_source": {self.ego_source_id: ego["ego_upper"],
                                    **{source: prepared[source][t]["shared_upper"]
                                       for source in sources[1:]}},
                "ego_lower": ego["ego_lower"],
                "selected_horizon_ms_by_source": {
                    self.ego_source_id: int(ego["ego"].horizon_ms),
                    **{source: int(prepared[source][t]["shared"].horizon_ms)
                       for source in sources[1:]},
                },
                "native_by_source": {
                    self.ego_source_id: True,
                    **{source: bool(prepared[source][t]["shared"].is_native)
                       for source in sources[1:]},
                },
            })
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                risks = np.array([[row["upper_by_source"][source] for source in sources]
                                  for row in rows])
                averages = risks.mean(axis=0)
                if not np.isfinite(averages).all() or np.any(averages < 0.):
                    raise FloatingPointError("Negative or nonfinite source risk.")
                zero = averages == 0.
                if zero.any():
                    weights = zero.astype(np.float64) / zero.sum()
                else:
                    inverse_relative = averages.min() / averages
                    weights = inverse_relative / inverse_relative.sum()
                upper = float(np.sum(weights ** 2 * averages))
                lower = float(np.mean([row["ego_lower"] for row in rows]))
                if not np.isfinite([upper, lower]).all():
                    raise FloatingPointError("Nonfinite mean risk.")
        except FloatingPointError:
            interval["reason"] = "numerical_risk_failure"
            return False
        accepted = self._passes_gate(upper, lower)
        interval.update(weights=dict(zip(sources, map(float, weights))),
                        U=upper, L0=lower, gate_passed=accepted,
                        gate_threshold=self.gate_ratio * lower,
                        reason="accepted" if accepted else "interval_bound_failed")
        return accepted

    def _fused_points(self, times, prepared, interval):
        """Validate every output before applying an interval to the ego copy."""
        sources, weights = interval["source_ids"], interval["weights"]
        updates = []
        capped_points = 0
        try:
            with np.errstate(over="raise", invalid="raise"):
                for t in times:
                    ego = prepared[sources[1]][t]["ego"]
                    points = {self.ego_source_id: ego,
                              **{source: prepared[source][t]["shared"] for source in sources[1:]}}
                    mean = sum(weights[source] * points[source].mean for source in sources)
                    correction = float(np.linalg.norm(mean - ego.mean))
                    effective_weights = weights
                    if self.cap is not None and 0 < t / 1000. <= self.cap["horizon_s"]:
                        radius = self.cap["radius_m"] * (t / 1000.) / self.cap["horizon_s"]
                        if correction > radius:
                            beta = radius / correction
                            mean = ego.mean + beta * (mean - ego.mean)
                            effective_weights = {source: beta * weights[source] for source in sources}
                            effective_weights[self.ego_source_id] += 1. - beta
                            correction = radius
                            capped_points += 1
                    covariance = sum(effective_weights[source] ** 2 * points[source].covariance
                                     for source in sources)
                    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not np.isfinite(correction):
                        raise FloatingPointError("Nonfinite fused output.")
                    updates.append((t, mean, covariance, correction))
        except FloatingPointError:
            return None
        interval["capped_points"] = capped_points
        if capped_points:
            interval["reason"] = "accepted_capped"
        return updates
