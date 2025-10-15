# listener/listener.py
import asyncio
import logging
import threading

import msgpack
import numpy as np
import zmq
import zmq.asyncio as azmq
import zstandard as zstd

logger = logging.getLogger(__name__)


# listener/listener.py
import asyncio, msgpack, zmq, zmq.asyncio as azmq, threading, logging
import zstandard as zstd
logger = logging.getLogger(__name__)

class Listener:
    def __init__(self, root: str, topic: str, on_message):
        # config
        self._root = root
        self._topic = topic
        self._on_message = on_message

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

    # -------------------- decoding helpers --------------------

    @staticmethod
    def _expand_entry(entry):
        """
        Compact entry:
          { "c": str, "b": [bx,by], "T": [t_ms...], "P": [[dx_cm,dy_cm]...],
            "V": [[varx_centi,vary_centi]...], "tt": <pred_ts_ms> }
        """
        
        bx, by = entry["b"]
        t_s = [tm / 1000.0 for tm in entry["T"]]
        xy = [[bx + dx / 100.0, by + dy / 100.0] for dx, dy in entry["P"]]
        cov = [[[vx / 100.0, 0.0], [0.0, vy / 100.0]] for vx, vy in entry["V"]]
        pred_ts_ms = int(entry.get("tt", 0))  # <-- include prediction timestamp (ms)

        return {
            "id":entry["id"],
            "category": str(entry["c"]),
            "cur_location": [float(bx), float(by)],
            "pred_ts_ms": pred_ts_ms,
            "prediction": {
                "t": t_s,
                "xy": xy,
                "cov": cov
            },
        }

    @staticmethod
    def _expand_packet(pkt):
        """
        Compact packet ->
        {
          "sender": str, "timestamp_ms": int,
          "fps": float, "pred_hz": float, "pred_sampling": float,
          "ego_position": {"x":..., "y":..., "z":..., "yaw":...},
          "predictions": [expanded_entry, ...]
        }
        """
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

    # -------------------- async receive loop --------------------

    async def _loop_coro(self):
        logger.info("[Listener] loop started; subscribed to '%s'", self._topic)
        self._running = True
        try:
            while self._running:
                topic, flag, data = await self._sock.recv_multipart()   # always 3 parts
                if topic != self._topic.encode("utf-8"):
                    continue
    
                # always compressed → decompress
                if flag != b"z":
                    logger.warning("[Listener] unexpected flag=%r (expected b'z')", flag)
                try:
                    raw = self._zd.decompress(data)
                except Exception:
                    logger.exception("[Listener] zstd decompress failed")
                    continue

                # unpack and expand
                try:
                    pkt = msgpack.unpackb(raw, raw=False)
                    # detect compact; compact has "pred" & short keys
                    if isinstance(pkt, dict) and "pred" in pkt and "s" in pkt:
                        payload = self._expand_packet(pkt)
                    else:
                        payload = pkt
                except Exception:
                    logger.exception("[Listener] msgpack unpack/expand failed")
                    continue

                # dispatch (supports async or sync callback)
                try:
                    res = self._on_message(topic, payload)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    logger.exception("[Listener] on_message handler failed")

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

                # context & socket must be created in this thread
                self._ctx = azmq.Context.instance()
                self._sock = self._ctx.socket(zmq.SUB)
                self._sock.setsockopt(zmq.LINGER, 0)
                self._sock.setsockopt(zmq.RCVHWM, 1000)
                self._sock.connect(f"{self._root}.out")
                self._sock.setsockopt(zmq.SUBSCRIBE, self._topic.encode("utf-8"))

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
