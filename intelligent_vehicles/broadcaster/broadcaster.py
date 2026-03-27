# broadcaster.py  (ONLY mismatch fix: use sim-time broadcasting_timestamp for "ts")

from typing import Any, List, Dict
import time
import logging

import numpy as np
import msgpack
import zmq
import zstandard as zstd

logger = logging.getLogger(__name__)


class Broadcaster:
    def __init__(self, root: str, topic: str, compress_min_bytes: int = 1500, zstd_level: int = 6):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 100)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"{root}.in")
        self.topic = topic.encode("utf-8")

        self._zc = zstd.ZstdCompressor(level=int(zstd_level))
        self._compress_min = int(compress_min_bytes)

        logger.info("[Broadcaster] connected to %s.in topic='%s'", root, topic)

    @staticmethod
    def _np(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    @staticmethod
    def _quantize_times(seconds: List[float]) -> List[int]:
        t_ms = [int(round(s * 1000.0)) for s in seconds]
        return [max(min(v, 32767), -32768) for v in t_ms]

    @staticmethod
    def _quantize_offsets_xy(base_xy: np.ndarray,
                             xy_list: List[List[float]],
                             cm_per_unit: float = 100.0) -> List[List[int]]:
        bx, by = float(base_xy[0]), float(base_xy[1])
        out = []
        for x, y in xy_list:
            dx_cm = int(round((x - bx) * cm_per_unit))
            dy_cm = int(round((y - by) * cm_per_unit))
            dx_cm = max(min(dx_cm, 32767), -32768)
            dy_cm = max(min(dy_cm, 32767), -32768)
            out.append([dx_cm, dy_cm])
        return out

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
    def _quantize_diag_cov(diag_list: List[List[float]],
                           scale: float = 100.0) -> List[List[int]]:
        out = []
        for vx, vy in diag_list:
            qx = int(round(vx * scale))
            qy = int(round(vy * scale))
            qx = max(min(qx, 32767), 0)
            qy = max(min(qy, 32767), 0)
            out.append([qx, qy])
        return out

    @staticmethod
    def _extract_ordered_series(pred_map: Dict[float, List[float]],
                                cov_map: Dict[float, List[List[float]]]) -> Dict[str, Any]:
        ts = sorted(pred_map.keys())
        xy = [pred_map[t] for t in ts]
        vv = [[cov_map[t][0][0], cov_map[t][1][1]] for t in ts]
        return {"t": ts, "xy": xy, "vv": vv}

    def _compact_prediction_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        cat = str(entry["category"])
        tt = entry["timestamp"]
        pred_obj = entry["prediction"]
        loc = self._quantize_location_full(entry["cur_location"], scale=100.0)
        series = self._extract_ordered_series(pred_obj["pred"], pred_obj["cov"])

        base_xy = np.asarray(entry["cur_location"], dtype=np.float32)
        t_ms = self._quantize_times(series["t"])
        P = self._quantize_offsets_xy(base_xy, series["xy"], cm_per_unit=100.0)
        V = self._quantize_diag_cov(series["vv"], scale=100.0)

        return {
            "id": str(entry["id"]),
            "c": cat,
            "b": loc,
            "tt": tt,
            "T": t_ms,
            "P": P,
            "V": V,
        }

    def _build_compact_packet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ego = payload.get("ego_position", {}) or {}
        preds = payload.get("predictions", []) or []

        compact_preds = [self._compact_prediction_entry(self._np(p)) for p in preds]

        # --- FIX: use SIM TIME broadcasting_timestamp (seconds) -> ms for "ts" ---
        bt = payload.get("broadcasting_timestamp", None)
        if bt is None:
            # fallback for safety (keeps old behavior if caller didn't provide sim time)
            ts_ms = (time.time_ns() // 1_000_000)
        else:
            ts_ms = int(round(float(bt) * 1000.0))
        # ----------------------------------------------------------------------

        package = {
            "s": str(payload.get("sender", "")),
            "ts": ts_ms,  # now aligned with simulation time axis
            "fps": float(payload.get("fps", 0.0)),
            "phz": float(payload.get("pred_hz", 0.0)),
            "ps": float(payload.get("pred_sampling", 0.0)),
            "ego": [
                float(ego.get("x", 0.0)),
                float(ego.get("y", 0.0)),
                float(ego.get("z", 0.0)),
                float(ego.get("yaw", 0.0)),
            ],
            "pred": compact_preds,
        }

        return package

    def send(self, payload_dict: Dict[str, Any]) -> int:
        compact = self._build_compact_packet(payload_dict)
        raw = msgpack.packb(compact, use_bin_type=True)

        data = self._zc.compress(raw)
        flag = b"z"

        size_bytes = len(self.topic) + 1 + len(data)
        self.sock.send_multipart([self.topic, flag, data])
        return size_bytes

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass