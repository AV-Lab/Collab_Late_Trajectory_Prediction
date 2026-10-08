# broadcaster.py  (ONLY mismatch fix: use sim-time broadcasting_timestamp for "ts")

from typing import Any, List, Dict

import numpy as np
import msgpack
import zmq
import zstandard as zstd



class Broadcaster:
    POSITION_SCALE = 100.0
    COVARIANCE_SCALE = 4096.0
    CORRELATION_LIMIT = 0.999

    def __init__(self, root: str, topic: str, compress_min_bytes: int = 1500, zstd_level: int = 6):
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
        t_ms = np.rint(np.asarray(seconds, dtype=np.float64) * 1000.0).astype(np.int64)
        if t_ms.ndim != 1 or t_ms.size == 0:
            raise ValueError("Prediction timestamps must be a non-empty sequence.")

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
        arr = np.asarray(cur_location, dtype=float).reshape(-1)
        out = []
        for v in arr:
            qv = int(round(v * scale))
            qv = max(min(qv, 2_147_483_647), -2_147_483_648)
            out.append(qv)
        return out

    @staticmethod
    def _pack_covariances(covariance_list: List[List[float]]) -> bytes:
        covariance = np.asarray(covariance_list, dtype=np.float64)
        if covariance.ndim != 2 or covariance.shape[1] != 3:
            raise ValueError("Covariance entries must have shape [T, 3].")
        if not np.isfinite(covariance).all():
            raise ValueError("Covariance entries must contain only finite values.")

        variance_x = covariance[:, 0]
        covariance_xy = covariance[:, 1]
        variance_y = covariance[:, 2]
        determinant = variance_x * variance_y - covariance_xy ** 2
        if np.any(variance_x <= 0.0) or np.any(variance_y <= 0.0) or np.any(determinant <= 0.0):
            raise ValueError("Covariance entries must be positive definite.")

        std_x = np.sqrt(variance_x)
        std_y = np.sqrt(variance_y)
        correlation = covariance_xy / (std_x * std_y)
        correlation = np.clip(
            correlation,
            -Broadcaster.CORRELATION_LIMIT,
            Broadcaster.CORRELATION_LIMIT,
        )
        transformed = np.column_stack((
            np.log(std_x),
            np.log(std_y),
            np.arctanh(correlation),
        ))
        quantized = np.rint(transformed * Broadcaster.COVARIANCE_SCALE)
        limits = np.iinfo(np.int16)
        if np.any(quantized < limits.min) or np.any(quantized > limits.max):
            raise ValueError("Covariance exceeds the supported quantization range.")
        return quantized.astype("<i2").tobytes()

    @staticmethod
    def _extract_ordered_series(pred_map: Dict[float, List[float]],
                                cov_map: Dict[float, List[List[float]]]) -> Dict[str, Any]:
        ts = sorted(pred_map.keys())
        xy = [pred_map[t] for t in ts]
        vv = [
            [cov_map[t][0][0], cov_map[t][0][1], cov_map[t][1][1]]
            for t in ts
        ]
        return {"t": ts, "xy": xy, "vv": vv}

    def _compact_prediction_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        cat = str(entry["category"])
        tt = entry["timestamp"]
        pred_obj = entry["prediction"]
        loc = self._quantize_location_full(entry["cur_location"], scale=100.0)
        series = self._extract_ordered_series(pred_obj["pred"], pred_obj["cov"])

        base_xy = np.asarray(entry["cur_location"], dtype=np.float32)
        time_fields = self._compact_times(series["t"])
        P = self._quantize_offsets_xy(
            base_xy,
            series["xy"],
            cm_per_unit=self.POSITION_SCALE,
        )
        V = self._pack_covariances(series["vv"])

        return {
            "id": str(entry["id"]),
            "c": cat,
            "b": loc,
            "tt": tt,
            **time_fields,
            "P": P,
            "V": V,
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
