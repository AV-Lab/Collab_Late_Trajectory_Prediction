from __future__ import annotations
import numpy as np
import logging
from typing import Dict, List, Optional
import math


class Evaluator:
    """
    Frame-by-frame aggregator.
    Use:
        ev = Evaluator(logger)
        for scenario in scenarios:
            ev.begin_scenario()
            while ...:
                forecasts, metrics, by_cat = compute_frame_based_performance(...)
                ev.accumulate(metrics, by_cat)
            ev.end_scenario(scenario)
        ev.log_overall(len(scenarios))
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.overall_ade  = 0.0
        self.overall_fde  = 0.0
        self.overall_missed = 0
        self.overall_msne = 0.0
        self.overall_msne_count = 0  # scenarios with finite MSNE
        self.cat_totals: Dict[str, Dict[str, float]] = {}
        self.scenario_summaries: List[Dict] = []

    # -------- per-scenario lifecycle -------- #

    def begin_scenario(self):
        self._sc_ade_sum = 0.0
        self._sc_fde_sum = 0.0
        self._sc_frames = 0
        self._sc_msne_sum = 0.0
        self._sc_msne_frames = 0

    def accumulate(self, metrics):
        """
        Add one frame’s metrics into the running scenario totals and global category totals.
        Expects `metrics` shaped like:
          {"overall": {...}, "by_cat": {...}}
        """
        overall = metrics.get("overall", {})
        by_cat  = metrics.get("by_cat", {})
    
        # overall ADE/FDE means (frame-level)
        ade_mean = overall.get("ADE_mean")
        fde_mean = overall.get("FDE_mean")
        
        if ade_mean is not None and not math.isnan(ade_mean):
            self._sc_ade_sum += ade_mean
        if fde_mean is not None and not math.isnan(fde_mean):
            self._sc_fde_sum += fde_mean
        self._sc_frames  += 1
    
        # MSNE mean (only counted if provided and not NaN)
        msne_val = overall.get("MSNE_mean")
        if msne_val is not None and not math.isnan(msne_val):
            self._sc_msne_sum    += msne_val
            self._sc_msne_frames += 1
    
        # per-category frame-averaged metrics + counts
        for cat, m in by_cat.items():
            ent = self.cat_totals.setdefault(
                cat, {"ade_sum": 0.0, "fde_sum": 0.0, "frames": 0,
                      "num_matched": 0, "num_missed": 0, "num_false_positives": 0}
            )
            ade_c = m.get("ADE_mean")
            fde_c = m.get("FDE_mean")
            if ade_c is not None and not math.isnan(ade_c):
                ent["ade_sum"] += ade_c
            if fde_c is not None and not math.isnan(fde_c):
                ent["fde_sum"] += fde_c
            ent["frames"]  += 1
    
            ent["num_matched"]        += int(m.get("num_matched", 0))
            ent["num_missed"]         += int(m.get("num_missed", 0))
            ent["num_false_positives"]+= int(m.get("num_false_positives", 0))
    
        # global counts
        self.overall_missed += int(overall.get("num_missed", 0))


    def end_scenario(self, name: str):
        """Finalize this scenario: compute averages, add to overall, and log."""
        print(self._sc_frames, self._sc_ade_sum, self._sc_fde_sum)
        if self._sc_frames:
            sc_ade = self._sc_ade_sum / self._sc_frames
            sc_fde = self._sc_fde_sum / self._sc_frames
            sc_msne = (self._sc_msne_sum / self._sc_msne_frames) if self._sc_msne_frames > 0 else float("nan")

            self.logger.info(f"[{name}] frames={self._sc_frames:3d}   ADE={sc_ade}  FDE={sc_fde}  MSNE={sc_msne:.3f}")

            self.overall_ade += sc_ade
            self.overall_fde += sc_fde
            if sc_msne is not None and not math.isnan(sc_msne):
                self.overall_msne += sc_msne
                self.overall_msne_count += 1
        else:
            sc_ade = float("nan")
            sc_fde = float("nan")
            sc_msne = float("nan")
            self.logger.info(f"[{name}] no valid matches – ADE/FDE/MSNE undefined")

        self.scenario_summaries.append({
            "scenario": name,
            "frames": self._sc_frames,
            "ade": sc_ade,
            "fde": sc_fde,
            "msne": sc_msne
        })

    # -------- reporting -------- #

    def _per_category_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for cat, v in self.cat_totals.items():
            if v["frames"]:
                ade_c = v["ade_sum"] / v["frames"]
                fde_c = v["fde_sum"] / v["frames"]
                out[cat] = {"frames": int(v["frames"]), "ΔADE_mean": float(ade_c), "ΔFDE_mean": float(fde_c)}
            else:
                out[cat] = {"frames": 0, "ΔADE_mean": float("nan"), "ΔFDE_mean": float("nan")}
        return out

    def log_overall(self, num_scenarios: int):
        mean_ade = self.overall_ade / max(1, num_scenarios)
        mean_fde = self.overall_fde / max(1, num_scenarios)
        mean_msne = (self.overall_msne / self.overall_msne_count) if self.overall_msne_count > 0 else float("nan")

        self.logger.info("\n===========  OVERALL  ===========")
        self.logger.info(f"Scenarios evaluated : {num_scenarios}")
        self.logger.info(f"Mean ADE            : {mean_ade}")
        self.logger.info(f"Mean FDE            : {mean_fde}")
        self.logger.info(f"Mean MSNE           : {mean_msne:.3f}")

        self.logger.info("\n=======  PER CATEGORY  ==========")
        for cat, v in self._per_category_summary().items():
            if v["frames"]:
                self.logger.info(f"[{cat:<12}] frames={v['frames']:3d}  ADE={v['ΔADE_mean']:.4f}  FDE={v['ΔFDE_mean']:.4f}")
            else:
                self.logger.info(f"[{cat:<12}] no matched frames")

        self.logger.info(f"Overall missed : {self.overall_missed}")
