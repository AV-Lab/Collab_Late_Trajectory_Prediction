"""Prepare sharing forecasts on the current ego prediction timeline."""

from bisect import bisect_left
from copy import deepcopy

import numpy as np

from calibration.runtime import valid_bounds


class PredictionTimeAligner:
    def __init__(self, align=False, min_points=10):
        if not isinstance(align, bool):
            raise ValueError("align must be a boolean.")
        if not isinstance(min_points, int) or isinstance(min_points, bool) or min_points < 1:
            raise ValueError("min_points must be a positive integer.")
        self.align_enabled = align
        self.min_points = min_points

    @staticmethod
    def _milliseconds(timestamps):
        """Use the packet's millisecond precision for time comparisons."""
        times = np.asarray(timestamps, dtype=np.float64)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("Prediction timestamps must be a finite one-dimensional sequence.")
        milliseconds = [int(round(float(t) * 1000)) for t in times]
        if any(b <= a for a, b in zip(milliseconds, milliseconds[1:])):
            raise ValueError("Prediction timestamps must be strictly increasing at millisecond precision.")
        return milliseconds

    @staticmethod
    def _calibration_metadata(prediction):
        """Keep native forecast ages and interpolation provenance recoverable."""
        forecast = prediction["prediction"]
        origin = forecast.get("forecast_origin_ms", prediction["pred_ts_ms"])
        if isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)):
            raise ValueError("forecast_origin_ms must be an integer timestamp in milliseconds.")
        mask = np.asarray(forecast.get("native_point_mask", [True] * len(forecast["t"])))
        if (mask.ndim != 1 or len(mask) != len(forecast["t"])
                or any(not isinstance(value, (bool, np.bool_)) for value in mask)):
            raise ValueError("native_point_mask must contain one boolean per prediction timestamp.")
        horizons = forecast.get("native_horizon_ms")
        if horizons is None:
            horizons = [int(prediction["pred_ts_ms"]) + t - int(origin)
                        for t in PredictionTimeAligner._milliseconds(forecast["t"])]
        if (not isinstance(horizons, (list, tuple)) or len(horizons) != len(forecast["t"])
                or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
                       or h <= 0 for h in horizons)):
            raise ValueError("native_horizon_ms must contain one positive native horizon per timestamp.")
        bounds = forecast.get("bounds", [None] * len(forecast["t"]))
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != len(forecast["t"])
                or any(value is not None and not valid_bounds(value) for value in bounds)):
            raise ValueError("bounds must contain one valid L/U risk pair or None per timestamp.")
        if any(value is not None for value in bounds) and forecast.get("risk_unit") != "m2":
            raise ValueError("Available bounds require risk_unit=m2.")
        return int(origin), mask.tolist(), list(horizons), list(bounds)

    @classmethod
    def rebase(cls, prediction, ego_timestamp_ms):
        """Copy a forecast, preserving absolute times under a new origin."""
        result = deepcopy(prediction)
        forecast = result["prediction"]
        origin, native_mask, horizons, bounds = cls._calibration_metadata(prediction)
        forecast.setdefault("forecast_origin_ms", origin)
        forecast["native_point_mask"] = native_mask
        forecast["native_horizon_ms"] = horizons
        forecast["bounds"] = deepcopy(bounds)
        offsets = cls._milliseconds(forecast["t"])
        if len(offsets) != len(forecast["xy"]) or len(offsets) != len(forecast["cov"]):
            raise ValueError("Prediction timestamps, means and covariances must have equal lengths.")
        shift = int(prediction["pred_ts_ms"]) - int(ego_timestamp_ms)
        forecast["t"] = [(shift + offset) / 1000.0 for offset in offsets]
        result["pred_ts_ms"] = int(ego_timestamp_ms)
        if "origin_ms" in forecast:
            forecast["origin_ms"] = int(ego_timestamp_ms)
        return result

    @classmethod
    def remove_outdated(cls, prediction, first_query_t, keep_left_bracket=False):
        """Trim before the first query, optionally retaining interpolation support."""
        forecast = prediction["prediction"]
        origin, native_mask, horizons, bounds = cls._calibration_metadata(prediction)
        times = cls._milliseconds(forecast["t"])
        first_query_ms = cls._milliseconds([first_query_t])[0]
        start = bisect_left(times, first_query_ms)
        if keep_left_bracket and start > 0 and (start == len(times) or times[start] > first_query_ms):
            start -= 1
        return {**prediction, "prediction": {
            **forecast, **{key: forecast[key][start:] for key in ("t", "xy", "cov")},
            "forecast_origin_ms": origin, "native_point_mask": native_mask[start:],
            "native_horizon_ms": horizons[start:], "bounds": deepcopy(bounds[start:]),
        }}

    @classmethod
    def align_to_ego_timestamps(cls, prediction, ego_timestamps):
        """Interpolate positions and copy the nearest native P/L/U tuple.

        Earlier timestamps win ties. Native certificates do not automatically
        certify interpolated positions; this is an explicit transfer policy.
        """
        forecast = prediction["prediction"]
        origin, native_mask, horizons, bounds = cls._calibration_metadata(prediction)
        times = cls._milliseconds(forecast["t"])
        query_ms = cls._milliseconds(ego_timestamps)
        xy = np.asarray(forecast["xy"], dtype=np.float64)
        cov = np.asarray(forecast["cov"], dtype=np.float64)
        if times and (xy.shape != (len(times), 2) or cov.shape != (len(times), 2, 2)):
            raise ValueError("Alignment requires means [N, 2] and covariance matrices [N, 2, 2].")
        if not np.isfinite(xy).all() or not np.isfinite(cov).all():
            raise ValueError("Alignment requires finite means and covariance matrices.")
        output_t, output_xy, output_cov, output_native_mask = [], [], [], []
        output_horizons, output_bounds = [], []
        for t, query in zip(ego_timestamps, query_ms):
            right = bisect_left(times, query)
            if right < len(times) and times[right] == query:
                mean = deepcopy(forecast["xy"][right])
                nearest = right
                is_native = native_mask[right]
            elif 0 < right < len(times):
                weight = (query - times[right - 1]) / (times[right] - times[right - 1])
                mean = ((1 - weight) * xy[right - 1] + weight * xy[right]).tolist()
                nearest = right - 1 if query - times[right - 1] <= times[right] - query else right
                is_native = False
            else:
                continue
            output_t.append(float(t))
            output_xy.append(mean)
            output_cov.append(deepcopy(forecast["cov"][nearest]))
            output_native_mask.append(is_native)
            output_horizons.append(horizons[nearest])
            output_bounds.append(deepcopy(bounds[nearest]))
        return {**prediction, "prediction": {
            **forecast, "t": output_t, "xy": output_xy, "cov": output_cov,
            "forecast_origin_ms": origin, "native_point_mask": output_native_mask,
            "native_horizon_ms": output_horizons, "bounds": output_bounds,
        }}

    @classmethod
    def _update_current_location(cls, prediction, source_timestamp_ms, association_timestamp_ms):
        """Preserve the existing nearest-native-point anchor used for matching."""
        forecast = prediction["prediction"]
        absolute_times = [prediction["pred_ts_ms"] + t
                          for t in cls._milliseconds(forecast["t"])]
        candidates = [(-1, source_timestamp_ms), *enumerate(absolute_times)]
        best_index, _ = min(candidates, key=lambda item: abs(association_timestamp_ms - item[1]))
        if best_index != -1:
            x, y = forecast["xy"][best_index]
            location = prediction["cur_location"]
            if hasattr(location, "__len__") and len(location) >= 3:
                location = list(location)
                location[0], location[1] = x, y
            else:
                location = [x, y]
            prediction["cur_location"] = location

    @classmethod
    def prepare(cls, preds, ego_timestamp_ms, *, association_timestamp_ms=None):
        """Copy/rebase native forecasts and establish their matching locations."""
        if association_timestamp_ms is None:
            association_timestamp_ms = ego_timestamp_ms
        out = []
        for prediction in preds:
            prepared = cls.rebase(prediction, ego_timestamp_ms)
            cls._update_current_location(prepared, prediction["pred_ts_ms"], association_timestamp_ms)
            out.append(prepared)
        return out

    def align(self, preds, ego_timestamps):
        """Trim/resample prepared copies; min_points counts final samples."""
        ego_timestamps = list(ego_timestamps)
        if not self._milliseconds(ego_timestamps):
            return []
        out = []
        for prediction in preds:
            prepared = self.remove_outdated(deepcopy(prediction), ego_timestamps[0], self.align_enabled)
            if self.align_enabled:
                prepared = self.align_to_ego_timestamps(prepared, ego_timestamps)
            if len(prepared["prediction"]["t"]) >= self.min_points:
                out.append(prepared)
        return out

    def apply(self, preds, ego_timestamp_ms, ego_timestamps, *, association_timestamp_ms=None):
        """Prepare and align forecasts without modifying the supplied packet."""
        ego_timestamps = list(ego_timestamps)
        if not self._milliseconds(ego_timestamps):
            return []
        prepared = self.prepare(preds, ego_timestamp_ms,
                                association_timestamp_ms=association_timestamp_ms)
        return self.align(prepared, ego_timestamps)
