import asyncio
import logging
import threading
import heapq
import math
import itertools

import msgpack
import numpy as np
import zmq
import zmq.asyncio as azmq
import zstandard as zstd
import os

logger = logging.getLogger(__name__)


class Listener:
    """
    Unified strategy:
      - Always receives messages asynchronously and buffers them with an arrival timestamp.
      - Delay model uses (k, mu, var):
            Δ(b) = k * b + LogNormal(mu, sigma),  sigma = sqrt(var)
        If any of (mu, var) is None -> stochastic term = 0
        If k is None -> k = 0
      - Optional packet drop based on packet size.
      - Main thread calls pop_arrived(sim_time_ms) to get messages that are available.

    NOTE: on_message is kept in the signature for backward compatibility,
          but is NOT used (no async thread graph updates).
    """

    def __init__(self, root: str, topic: str, on_message=None, k=None, mu=None, var=None, drop=False):
        # config
        self._root = root
        self._topic = topic
        self._topic_b = topic.encode("utf-8")
        self._on_message = on_message  # kept but unused by design
        self._drop = bool(drop)

        # runtime
        self._ctx = None
        self._sock = None
        self._loop = None
        self._task = None
        self._thread = None
        self._running = False
        self._started_evt = threading.Event()

        # decompress once, reuse
        self._zd = zstd.ZstdDecompressor()

        # -------------------- delay params (optional) --------------------
        self._k = 0.0 if k is None else float(k)  # ms/byte
        self._mu = None if mu is None else float(mu)
        self._sigma = None if var is None else math.sqrt(float(var))

        # Buffer: (arrival_ms:int, seq:int, topic:bytes, payload:dict)
        self._buf_lock = threading.Lock()
        self._buf_heap = []
        self._seq = itertools.count()

        # RNG for sampling (deterministic seed for reproducibility)
        self._rng = np.random.default_rng(0)

        if (self._mu is None) or (self._sigma is None):
            logger.info("[Listener] delay: k=%.6f ms/B, LogNormal disabled (mu/var not provided) -> Δ=b*k", self._k)
        else:
            logger.info("[Listener] delay: k=%.6f ms/B, LogNormal(mu=%.3f, sigma=%.3f)", self._k, self._mu, self._sigma)

        logger.info("[Listener] packet drop: %s", "enabled" if self._drop else "disabled")

        self._delay_log_path = "listener_delay_log.txt"
        if os.path.exists(self._delay_log_path):
            os.remove(self._delay_log_path)

    @staticmethod
    def _expand_entry(entry):
        loc = [v / 100.0 for v in entry["b"]]
        bx, by = loc[0], loc[1]
        t_s = [tm / 1000.0 for tm in entry["T"]]
        xy = [[bx + dx / 100.0, by + dy / 100.0] for dx, dy in entry["P"]]
        cov = [[[vx / 100.0, 0.0], [0.0, vy / 100.0]] for vx, vy in entry["V"]]
        pred_ts_ms = int(entry.get("tt", 0))

        return {
            "id": entry["id"],
            "category": str(entry["c"]),
            "cur_location": loc,
            "pred_ts_ms": pred_ts_ms,
            "prediction": {
                "t": t_s,
                "xy": xy,
                "cov": cov
            },
        }

    @staticmethod
    def _expand_packet(pkt):
        ego = pkt.get("ego", [0.0, 0.0, 0.0, 0.0])
        expanded = {
            "sender": str(pkt.get("s", "")),
            "timestamp_ms": int(pkt.get("ts", 0)),
            "fps": float(pkt.get("fps", 0.0)),
            "pred_hz": float(pkt.get("phz", 0.0)),
            "pred_sampling": float(pkt.get("ps", 0.0)),
            "ego_position": {
                "x": float(ego[0]),
                "y": float(ego[1]),
                "z": float(ego[2]),
                "yaw": float(ego[3]),
            },
            "predictions": [Listener._expand_entry(e) for e in (pkt.get("pred") or [])],
        }
        return expanded

    # -------------------- buffering API --------------------

    def _sample_delay_ms(self, bytes_len: int) -> int:
        d = self._k * float(bytes_len)
        if (self._mu is not None) and (self._sigma is not None):
            d += float(self._rng.lognormal(mean=self._mu, sigma=self._sigma))
        if d < 0.0:
            print(d)
            print(self._k * float(bytes_len))
            exit()
            d = 0.0
        return int(round(d))

    def _should_drop_packet(self, bytes_len: int) -> bool:
        if not self._drop:
            return False

        if bytes_len <= 400:
            p_drop = 0.0
        elif bytes_len <= 900:
            p_drop = 0.08
        else:
            p_drop = 0.1

        return bool(self._rng.random() < p_drop)

    def pop_arrived(self, sim_time_ms: int):
        out = []
        with self._buf_lock:
            while self._buf_heap and self._buf_heap[0][0] <= int(sim_time_ms):
                _, _, topic, payload = heapq.heappop(self._buf_heap)
                out.append((topic, payload))
        return out

    # -------------------- async receive loop --------------------

    async def _loop_coro(self):
        logger.info("[Listener] loop started; subscribed to '%s'", self._topic)
        self._running = True
        try:
            while self._running:
                topic, flag, data = await self._sock.recv_multipart()  # [topic, flag, data]
                if topic != self._topic_b:
                    continue

                if flag != b"z":
                    logger.warning("[Listener] unexpected flag=%r (expected b'z')", flag)
                try:
                    raw = self._zd.decompress(data)
                except Exception:
                    logger.exception("[Listener] zstd decompress failed")
                    continue

                try:
                    pkt = msgpack.unpackb(raw, raw=False)
                    if isinstance(pkt, dict) and "pred" in pkt and "s" in pkt:
                        payload = self._expand_packet(pkt)
                    else:
                        payload = pkt
                except Exception:
                    logger.exception("[Listener] msgpack unpack/expand failed")
                    continue

                # ALWAYS buffer with arrival time
                try:
                    send_ms = int(payload.get("timestamp_ms", 0))
                    bytes_len = int(len(data))  # compressed payload size (bytes)

                    if self._should_drop_packet(bytes_len):
                        continue

                    delay_ms = self._sample_delay_ms(bytes_len)
                    arrival_ms = send_ms + delay_ms

                    with open(self._delay_log_path, "a") as f:
                        f.write(f"{bytes_len},{delay_ms}\n")
                    # -------------------------------------------------------------------

                    with self._buf_lock:
                        heapq.heappush(self._buf_heap, (arrival_ms, next(self._seq), topic, payload))
                except Exception:
                    logger.exception("[Listener] buffering failed")

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[Listener] loop error")
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            self._running = False
            logger.info("[Listener] loop stopped")

    # -------------------- thread management --------------------

    def start_in_background(self):
        if self._thread and self._thread.is_alive():
            logger.info("[Listener] already running")
            return

        def _run():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                self._ctx = azmq.Context.instance()
                self._sock = self._ctx.socket(zmq.SUB)
                self._sock.setsockopt(zmq.LINGER, 0)
                self._sock.setsockopt(zmq.RCVHWM, 1000)
                self._sock.connect(f"{self._root}.out")
                self._sock.setsockopt(zmq.SUBSCRIBE, self._topic_b)

                self._task = self._loop.create_task(self._loop_coro())
                self._started_evt.set()
                self._loop.run_forever()
            except Exception:
                logger.exception("[Listener] background thread failed to start")
            finally:
                try:
                    if self._sock is not None:
                        self._sock.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if self._started_evt.wait(timeout=1.0):
            logger.info("[Listener] background thread started")
        else:
            logger.error("[Listener] failed to signal start")

    def stop_in_background(self):
        if not self._loop:
            return

        def _stop():
            if self._task and not self._task.done():
                self._task.cancel()
            self._loop.stop()

        self._loop.call_soon_threadsafe(_stop)
        if self._thread:
            self._thread.join(timeout=1.0)
        self._task = None
        self._loop = None
        self._thread = None
        self._started_evt.clear()