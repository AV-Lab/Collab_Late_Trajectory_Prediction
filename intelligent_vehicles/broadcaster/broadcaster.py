"""Transmit native forecasts with sender-selected squared-error risk bounds."""

from typing import Any, List, Dict
from collections.abc import Mapping

import numpy as np
import msgpack
import zmq
import zstandard as zstd

from calibration.motion import MOTION_TAGS
from intelligent_vehicles.prediction_codec import (
    CORRELATION_LIMIT, COVARIANCE_SCALE, PACKET_SCHEMA_VERSION, POSITION_SCALE,
    RISK_UNIT, decode_covariances, pack_covariances, quantized_risk_bounds,
    validate_bound_pair,
)


class Broadcaster:
    POSITION_SCALE = POSITION_SCALE
    COVARIANCE_SCALE = COVARIANCE_SCALE
    CORRELATION_LIMIT = CORRELATION_LIMIT

    def __init__(self, root: str, topic: str, compress_min_bytes: int = 1500,
                 zstd_level: int = 6, certificate=None):
        self.certificate = certificate
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 100)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"{root}.in")
        self.topic = topic.encode("utf-8")

        self._zc = zstd.ZstdCompressor(level=int(zstd_level))
        self._compress_min = int(compress_min_bytes)

        print(f"[Broadcaster] connected to {root}.in topic='{topic}'")

    @staticmethod
    def _np(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    @staticmethod
    def _compact_times(seconds: List[float]) -> Dict[str, int]:
        times = np.asarray(seconds, dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
            raise ValueError("Prediction timestamps must be a non-empty sequence.")
        t_ms = np.rint(times * 1000.0).astype(np.int64)
        if t_ms[0] <= 0:
            raise ValueError("Native prediction horizons must be positive.")

        if t_ms.size == 1:
            dt_ms = 0
        else:
            intervals = np.diff(t_ms)
            dt_ms = int(intervals[0])
            if dt_ms <= 0 or not np.all(intervals == dt_ms):
                raise ValueError("Prediction timestamps must be regularly spaced.")

        return {"t0": int(t_ms[0]), "dt": dt_ms, "n": int(t_ms.size)}

    @staticmethod
    def _quantize_offsets_xy(base_xy: np.ndarray,
                             xy_list: List[List[float]],
                             cm_per_unit: float = 100.0) -> bytes:
        base = np.asarray(base_xy, dtype=np.float64).reshape(-1)
        xy = np.asarray(xy_list, dtype=np.float64)
        if base.size < 2 or xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Trajectory positions must have shape [T, 2].")
        if not np.isfinite(base[:2]).all() or not np.isfinite(xy).all():
            raise ValueError("Trajectory positions must contain only finite values.")

        offsets = np.rint((xy - base[:2]) * cm_per_unit)
        limits = np.iinfo(np.int16)
        if np.any(offsets < limits.min) or np.any(offsets > limits.max):
            raise ValueError("Trajectory offset exceeds the int16 centimetre range.")
        return offsets.astype("<i2").tobytes()

    @staticmethod
    def _quantize_location_full(cur_location, scale: float = 100.0) -> list:
        location = np.asarray(cur_location, dtype=np.float64).reshape(-1)
        if location.size < 2 or not np.isfinite(location).all():
            raise ValueError("Current location must contain finite XY coordinates.")
        quantized = np.rint(location * scale)
        limits = np.iinfo(np.int32)
        if np.any(quantized < limits.min) or np.any(quantized > limits.max):
            raise ValueError("Current location exceeds the int32 centimetre range.")
        return quantized.astype(np.int32).tolist()

    _pack_covariances = staticmethod(pack_covariances)

    @staticmethod
    def _extract_ordered_series(pred_map: Dict[float, List[float]],
                                cov_map: Dict[float, List[List[float]]]) -> Dict[str, Any]:
        ts = sorted(pred_map.keys())
        xy = [pred_map[t] for t in ts]
        vv = [
            [cov_map[t][0][0], cov_map[t][0][1], cov_map[t][1][1]]
            for t in ts
        ]
        return {"t": ts, "xy": xy, "vv": vv, "cov": [cov_map[t] for t in ts]}

    def _compact_prediction_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        cat = str(entry["category"])
        tt = entry["timestamp"]
        pred_obj = entry["prediction"]
        loc = self._quantize_location_full(entry["cur_location"], scale=100.0)
        series = self._extract_ordered_series(pred_obj["pred"], pred_obj["cov"])

        base_xy = np.asarray(entry["cur_location"], dtype=np.float64)
        time_fields = self._compact_times(series["t"])
        P = self._quantize_offsets_xy(
            base_xy,
            series["xy"],
            cm_per_unit=self.POSITION_SCALE,
        )
        V = self._pack_covariances(series["vv"])
        wire_xy = np.frombuffer(P, dtype="<i2").reshape(-1, 2).astype(np.float64)
        wire_xy = wire_xy / self.POSITION_SCALE + np.asarray(loc[:2]) / self.POSITION_SCALE
        motion_tag = pred_obj.get("motion_tag")
        if self.certificate is not None:
            if (pred_obj.get("certificate_id") != self.certificate.certificate_id
                    or pred_obj.get("certificate_method") != self.certificate.method
                    or pred_obj.get("risk_unit") != RISK_UNIT):
                raise ValueError("Native prediction must carry this sender's selected risk bounds.")
            if self.certificate.method == "motion_bootstrap_raw_moment":
                native_bounds = pred_obj.get("bounds")
                if not isinstance(native_bounds, Mapping) or set(native_bounds) != set(series["t"]):
                    raise ValueError("Native risk bounds must match the original forecast timestamps.")
                if motion_tag is None:
                    if any(value is not None for value in native_bounds.values()):
                        raise ValueError("Available motion bounds require a native motion tag.")
                elif motion_tag not in MOTION_TAGS:
                    raise ValueError("Unknown native forecast motion tag.")
            elif motion_tag is not None:
                raise ValueError("Bootstrap certificate predictions do not use motion tags.")
        elif pred_obj.get("certificate_id") is not None or motion_tag is not None:
            raise ValueError("Annotated prediction requires its sender's certificate.")
        bootstrap = (self.certificate is not None
                     and self.certificate.method == "bootstrap_raw_moment")
        wire_covariances = decode_covariances(V, time_fields["n"]) if bootstrap else None
        bounds = []
        for index, timestamp in enumerate(series["t"]):
            horizon_ms = time_fields["t0"] + index * time_fields["dt"]
            if bootstrap:
                selected = self.certificate.bounds(
                    cat, horizon_ms, wire_covariances[index],
                    grouping_covariance=series["cov"][index],
                )
            elif self.certificate is not None:
                selected = pred_obj["bounds"][timestamp]
                displacement = float(np.linalg.norm(wire_xy[index] - series["xy"][index]))
                selected = quantized_risk_bounds(selected, displacement)
            else:
                selected = None
            pair = None if selected is None else [selected["lower"], selected["upper"]]
            normalized = validate_bound_pair(pair)
            bounds.append(None if normalized is None else [normalized["lower"], normalized["upper"]])

        return {
            "id": str(entry["id"]),
            "c": cat,
            "b": loc,
            "tt": tt,
            **time_fields,
            "P": P,
            "V": V,
            "B": bounds,
            "motion_tag": motion_tag,
        }

    def _build_compact_packet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ego = payload["ego_position"]
        preds = payload["predictions"]

        compact_preds = [self._compact_prediction_entry(self._np(p)) for p in preds]

        # --- FIX: use SIM TIME broadcasting_timestamp (seconds) -> ms for "ts" ---
        bt = payload["broadcasting_timestamp"]
        ts_ms = int(round(float(bt) * 1000.0))
        # ----------------------------------------------------------------------

        package = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "risk_unit": RISK_UNIT,
            "certificate_id": (self.certificate.certificate_id
                               if self.certificate is not None else None),
            "method": self.certificate.method if self.certificate is not None else None,
            "s": str(payload["sender"]),
            "ts": ts_ms,  # now aligned with simulation time axis
            "fps": float(payload["fps"]),
            "phz": float(payload["pred_hz"]),
            "ps": float(payload["pred_sampling"]),
            "ego": [
                float(ego["x"]),
                float(ego["y"]),
                float(ego["z"]),
                float(ego["yaw"]),
            ],
            "pred": compact_preds,
        }

        return package

    def send(self, payload_dict: Dict[str, Any]) -> int:
        compact = self._build_compact_packet(payload_dict)
        raw = msgpack.packb(compact, use_bin_type=True)

        data = self._zc.compress(raw)
        flag = b"z"

        size_bytes = len(self.topic) + len(flag) + len(data)
        self.sock.send_multipart([self.topic, flag, data])
        return size_bytes

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
