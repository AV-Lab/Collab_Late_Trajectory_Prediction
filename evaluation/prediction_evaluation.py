from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})


class Evaluator:
    """
    Frame-by-frame aggregator.

    Updated to support:
      - exact TSR accumulation via (num_success_tau, N_gt) instead of tsr*N
      - storing per-object FDE values (FDE_list) for histogram/CDF sanity checks
      - matched-only TSR diagnostic accumulation
      - NEW: per-category MR/TSR accumulation via integer counts from metrics["by_cat"]
    """

    def __init__(self, logger: Optional[logging.Logger] = None, store_distributions: bool = True):
        self.logger = logger or logging.getLogger(__name__)
        self.store_distributions = store_distributions

        self.overall_ade = 0.0
        self.overall_fde = 0.0
        self.overall_missed = 0
        self.overall_msne = 0.0
        self.overall_msne_count = 0  # scenarios with finite MSNE

        # Global matched / GT counts (for miss rate & TSR)
        self.overall_matched = 0
        self.total_gt_objects = 0  # sum over frames of N_gt

        # Global TSR accumulators (exact, integer counts preferred)
        self.overall_tsr_success = 0  # count of successes over all frames
        self.overall_tsr_N = 0        # sum of N_gt over all frames

        # matched-only TSR diagnostic
        self.overall_tsr_success_matched = 0  # successes among matched
        self.overall_tsr_N_matched = 0        # matched count (valid FDE count)

        # Last seen TSR threshold (read from per-frame metrics)
        self.traj_success_threshold: Optional[float] = None

        # ---------- UPDATED: category totals now track integer counts for MR/TSR ----------
        # We keep frame-averaged ADE/FDE sums as you had, but we compute MR/TSR from counts.
        self.cat_totals: Dict[str, Dict[str, float]] = {}
        # --------------------------------------------------------------------------------

        self.scenario_summaries: List[Dict] = []

        # Global FDE distribution store (for histogram/CDF)
        self.fde_all: List[float] = []

        # Per-scenario distributions
        self._sc_fde_all: List[float] = []

        # Global accumulators by source
        self.source_totals: Dict[str, Dict[str, float]] = {
            "category_I_ego":    {"ade_sum": 0.0, "fde_sum": 0.0, "frames": 0, "num": 0},
            "category_I_fused":  {"ade_sum": 0.0, "fde_sum": 0.0, "frames": 0, "num": 0},
            "category_II_fused": {"ade_sum": 0.0, "fde_sum": 0.0, "frames": 0, "num": 0},
            "no_fusion":         {"ade_sum": 0.0, "fde_sum": 0.0, "frames": 0, "num": 0},
        }

    def plot_hist(
        self,
        model_name: str = "",
        tau: float = 0.5,
        clip_max: float = 20.0,
        bins: str | int = "fd",
        use_counts: bool = True,   # True = Count (#objects). False = density.
        log_hist_y: bool = True,   # log-scale ONLY for histogram y-axis
        save_hist_path: str | None = None,
        save_cdf_path: str | None = None,
        show: bool = True,
    ):
        """
        Same function, outputs TWO SEPARATE images:
          - Histogram figure (full, clipped to clip_max)
          - CDF figure (with TSR annotation at x=tau)
    
        Styling: black + red only (no blue), white annotation box with black text.
        Axis labels are removed (you will add them yourself).
    
        Returns:
          (fig_hist, fig_cdf, tsr_tau)
        """
    
        if not self.store_distributions or len(self.fde_all) == 0:
            raise RuntimeError("No FDE samples stored. Use Evaluator(store_distributions=True).")
    
        fde = np.asarray(self.fde_all, dtype=float)
        fde = fde[np.isfinite(fde)]
        if fde.size == 0:
            raise RuntimeError("FDE list contains no finite values.")
    
        # ---------- TSR at tau (empirical CDF value over matched objects) ----------
        tau = float(tau)
        tsr_tau = float(np.mean(fde <= tau))  # P(FDE <= tau)
    
        # ---------- Histogram data (clipped for consistent axes) ----------
        if clip_max is not None:
            xmax = float(clip_max)
            fde_hist = np.clip(fde, 0.0, xmax)
        else:
            xmax = float(np.max(fde))
            fde_hist = fde
    
        # ===================== 1) HIST FIG =====================
        fig_h, ax_h = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    
        # black histogram bars
        ax_h.hist(
            fde_hist,
            bins=bins,
            density=(not use_counts),
            color="black",
            edgecolor="black",
            linewidth=0.6,
        )
    
        # red tau line
        ax_h.axvline(tau, color="red", linestyle="--", linewidth=2.5, label=f"tau={tau:.2f} m")
    
        ax_h.set_xlim(0, xmax)
    
        # remove axis labels (you will add them)
        ax_h.set_xlabel("")
        ax_h.set_ylabel("")
    
        # y-label intent (kept as title note only; no axis label)
        if use_counts:
            ydesc = "Count"
        else:
            ydesc = "Density"
    
        if log_hist_y:
            ax_h.set_yscale("log")
            ydesc += " (log-y)"
    
        ax_h.set_title(
            (f"{model_name} | Histogram (clipped to {xmax:.0f} m) | {ydesc}")
            if clip_max is not None
            else (f"{model_name} | Histogram | {ydesc}"),
            fontsize=14,
            color="black",
        )
    
        ax_h.legend(fontsize=12, frameon=True, facecolor="white", edgecolor="black")
        ax_h.tick_params(axis="both", labelsize=12, colors="black")
        for spine in ax_h.spines.values():
            spine.set_color("black")
    
        if save_hist_path is not None:
            fig_h.savefig(save_hist_path, dpi=200)
    
        # ===================== 2) CDF FIG =====================
        fde_sorted = np.sort(fde)
        cdf_y = np.arange(1, fde_sorted.size + 1) / float(fde_sorted.size)
    
        fig_c, ax_c = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    
        # black CDF curve
        ax_c.plot(fde_sorted, cdf_y, color="black", linewidth=2.2, label="CDF")
    
        # red tau line
        ax_c.axvline(tau, color="red", linestyle="--", linewidth=2.5, label=f"tau={tau:.2f} m")
    
        # intersection marker (black) + horizontal guide (black dotted)
        ax_c.plot([tau], [tsr_tau], marker="o", markersize=5.5,
                  markerfacecolor="black", markeredgecolor="black")
        ax_c.axhline(tsr_tau, color="black", linestyle=":", linewidth=1.8)
    
        # annotate TSR (white box, black text)
        ax_c.annotate(
            f"TSR@{tau:.2f} = {tsr_tau*100:.2f}%",
            xy=(tau, tsr_tau),
            xytext=(10, -18),
            textcoords="offset points",
            fontsize=12,
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=1.0),
        )
    
        if clip_max is not None:
            ax_c.set_xlim(0, xmax)
        ax_c.set_ylim(0.0, 1.0)
    
        # remove axis labels (you will add them)
        ax_c.set_xlabel("")
        ax_c.set_ylabel("")
    
        ax_c.set_title(f"{model_name} | CDF", fontsize=14, color="black")
        ax_c.legend(fontsize=12, frameon=True, facecolor="white", edgecolor="black")
        ax_c.tick_params(axis="both", labelsize=12, colors="black")
        for spine in ax_c.spines.values():
            spine.set_color("black")
    
        if save_cdf_path is not None:
            fig_c.savefig(save_cdf_path, dpi=600)
    
        if show:
            plt.show()
        else:
            plt.close(fig_h)
            plt.close(fig_c)
    
        return fig_h, fig_c, tsr_tau



    # ------------------------- scenario lifecycle ------------------------- #
    def begin_scenario(self):
        self._sc_ade_sum = 0.0
        self._sc_fde_sum = 0.0
        self._sc_frames = 0
        self._sc_msne_sum = 0.0
        self._sc_msne_frames = 0

        self._sc_matched = 0
        self._sc_missed = 0

        self._sc_tsr_success = 0
        self._sc_tsr_N = 0

        self._sc_tsr_success_matched = 0
        self._sc_tsr_N_matched = 0

        self._sc_fde_all = []

    def accumulate(self, metrics: Dict):
        """
        Add one frame’s metrics into the running scenario totals and global category totals.
        Expects: {"overall": {...}, "by_cat": {...}, "by_source": {...} (optional)}
        """
        overall = metrics.get("overall", {})
        by_cat = metrics.get("by_cat", {})
        by_source = metrics.get("by_source", {})

        # --- frame-level ADE/FDE means ---
        ade_mean = overall.get("ADE_mean")
        fde_mean = overall.get("FDE_mean")

        if ade_mean is not None and not math.isnan(ade_mean):
            self._sc_ade_sum += float(ade_mean)
        if fde_mean is not None and not math.isnan(fde_mean):
            self._sc_fde_sum += float(fde_mean)
        self._sc_frames += 1

        # --- MSNE ---
        msne_val = overall.get("MSNE_mean")
        if msne_val is not None and not math.isnan(msne_val):
            self._sc_msne_sum += float(msne_val)
            self._sc_msne_frames += 1

        # --- per-frame GT counts ---
        num_matched = int(overall.get("num_matched", 0))
        num_missed = int(overall.get("num_missed", 0))

        N_gt_frame = overall.get("N_gt")
        N_gt_frame = int(N_gt_frame) if N_gt_frame is not None else (num_matched + num_missed)

        self._sc_matched += num_matched
        self._sc_missed += num_missed

        self.overall_matched += num_matched
        self.overall_missed += num_missed
        self.total_gt_objects += N_gt_frame

        # --- TSR accumulation (overall) ---
        num_success_tau = overall.get("num_success_tau")
        num_valid_fde = overall.get("num_valid_fde")

        if num_success_tau is not None and num_valid_fde is not None:
            s = int(num_success_tau)

            if N_gt_frame > 0:
                self._sc_tsr_success += s
                self._sc_tsr_N += N_gt_frame
                self.overall_tsr_success += s
                self.overall_tsr_N += N_gt_frame

            mN = int(num_valid_fde)
            if mN > 0:
                self._sc_tsr_success_matched += s
                self._sc_tsr_N_matched += mN
                self.overall_tsr_success_matched += s
                self.overall_tsr_N_matched += mN
        else:
            tsr = overall.get("trajectory_success_rate")
            if tsr is not None and not math.isnan(tsr) and N_gt_frame > 0:
                successes = float(tsr) * N_gt_frame
                self._sc_tsr_success += int(round(successes))
                self._sc_tsr_N += N_gt_frame
                self.overall_tsr_success += int(round(successes))
                self.overall_tsr_N += N_gt_frame

        th = overall.get("traj_success_threshold")
        if th is not None:
            self.traj_success_threshold = float(th)

        # --- store FDE distribution ---
        if self.store_distributions:
            fde_list = overall.get("FDE_list", None)
            if isinstance(fde_list, list) and len(fde_list) > 0:
                clean = [float(x) for x in fde_list if x is not None and np.isfinite(x)]
                if clean:
                    self.fde_all.extend(clean)
                    self._sc_fde_all.extend(clean)

        # -------------------- UPDATED: per-category accumulation (MR/TSR exact) --------------------
        for cat, m in by_cat.items():
            cat = str(cat).lower()

            ent = self.cat_totals.setdefault(
                cat,
                {
                    # keep your old frame-averaged ADE/FDE sums
                    "ade_sum": 0.0,
                    "fde_sum": 0.0,
                    "frames": 0,

                    # counts (accumulated)
                    "num_matched": 0,
                    "num_missed": 0,
                    "num_false_positives": 0,

                    # NEW: exact TSR accumulators per category
                    "num_success_tau": 0,
                    "N_gt": 0,
                },
            )

            # frame-mean ADE/FDE (as before)
            ade_c = m.get("ADE_mean")
            fde_c = m.get("FDE_mean")
            if ade_c is not None and not math.isnan(ade_c):
                ent["ade_sum"] += float(ade_c)
            if fde_c is not None and not math.isnan(fde_c):
                ent["fde_sum"] += float(fde_c)
            ent["frames"] += 1

            # counts (as before)
            ent["num_matched"] += int(m.get("num_matched", 0))
            ent["num_missed"] += int(m.get("num_missed", 0))
            ent["num_false_positives"] += int(m.get("num_false_positives", 0))

            # NEW: exact per-category TSR counts (preferred)
            # These are produced by your updated compute_by_category_statistics()
            ent["num_success_tau"] += int(m.get("num_success_tau", 0))
            ent["N_gt"] += int(m.get("N_gt", int(m.get("num_matched", 0)) + int(m.get("num_missed", 0))))
        # ------------------------------------------------------------------------------------------

        # --- per-source accumulation (unchanged) ---
        def _acc_source(name: str, entry: Dict):
            if not entry:
                return
            num = int(entry.get("num", 0))
            if num <= 0:
                return
            ade_s = entry.get("ADE_mean")
            fde_s = entry.get("FDE_mean")
            ent = self.source_totals[name]
            if ade_s is not None and not math.isnan(ade_s):
                ent["ade_sum"] += float(ade_s)
            if fde_s is not None and not math.isnan(fde_s):
                ent["fde_sum"] += float(fde_s)
            ent["frames"] += 1
            ent["num"] += num

        cat1 = by_source.get("category_I", {})
        _acc_source("category_I_ego", cat1.get("ego", {}))
        _acc_source("category_I_fused", cat1.get("fused", {}))

        cat2 = by_source.get("category_II", {})
        _acc_source("category_II_fused", cat2.get("fused", {}))

        nof = by_source.get("no_fusion", {})
        _acc_source("no_fusion", nof)

    def end_scenario(self, name: str):
        if self._sc_frames:
            sc_ade = self._sc_ade_sum / self._sc_frames
            sc_fde = self._sc_fde_sum / self._sc_frames
            sc_msne = (self._sc_msne_sum / self._sc_msne_frames) if self._sc_msne_frames > 0 else float("nan")

            sc_N_gt = self._sc_matched + self._sc_missed
            sc_miss_rate = (self._sc_missed / sc_N_gt) if sc_N_gt > 0 else float("nan")

            sc_tsr = (self._sc_tsr_success / self._sc_tsr_N) if self._sc_tsr_N > 0 else float("nan")
            sc_tsr_matched = (
                self._sc_tsr_success_matched / self._sc_tsr_N_matched
                if self._sc_tsr_N_matched > 0 else float("nan")
            )

            self.logger.info(
                f"[{name}] frames={self._sc_frames:3d}   "
                f"ADE={sc_ade:.4f}  FDE={sc_fde:.4f}  MSNE={sc_msne:.3f}  "
                f"MissRate={sc_miss_rate:.4f}  TSR={sc_tsr:.4f}  TSR_matched={sc_tsr_matched:.4f}"
            )

            self.overall_ade += sc_ade
            self.overall_fde += sc_fde
            if sc_msne is not None and not math.isnan(sc_msne):
                self.overall_msne += sc_msne
                self.overall_msne_count += 1
        else:
            sc_ade = float("nan")
            sc_fde = float("nan")
            sc_msne = float("nan")
            sc_miss_rate = float("nan")
            sc_tsr = float("nan")
            sc_tsr_matched = float("nan")
            self.logger.info(f"[{name}] no valid frames – ADE/FDE/MSNE/TSR undefined")

        self.scenario_summaries.append(
            {
                "scenario": name,
                "frames": self._sc_frames,
                "ade": sc_ade,
                "fde": sc_fde,
                "msne": sc_msne,
                "miss_rate": sc_miss_rate,
                "traj_success_rate": sc_tsr,
                "traj_success_rate_matched": sc_tsr_matched,
                "fde_count": len(self._sc_fde_all) if self.store_distributions else 0,
            }
        )

    # ------------------------- reporting ------------------------- #
    def _per_category_summary(self) -> Dict[str, Dict[str, float]]:
        """
        UPDATED: returns per-category ADE/FDE (frame-avg, as before) PLUS exact MR/TSR computed from counts.
        """
        out: Dict[str, Dict[str, float]] = {}
        for cat, v in self.cat_totals.items():
            frames = int(v.get("frames", 0))

            ade_c = (v["ade_sum"] / frames) if frames > 0 else float("nan")
            fde_c = (v["fde_sum"] / frames) if frames > 0 else float("nan")

            N_gt = int(v.get("N_gt", int(v.get("num_matched", 0)) + int(v.get("num_missed", 0))))
            num_missed = int(v.get("num_missed", 0))
            num_succ = int(v.get("num_success_tau", 0))

            mr = (num_missed / N_gt) if N_gt > 0 else float("nan")
            tsr = (num_succ / N_gt) if N_gt > 0 else float("nan")

            out[cat] = {
                "frames": frames,
                "ADE_mean": float(ade_c),
                "FDE_mean": float(fde_c),
                "N_gt": int(N_gt),
                "MR": float(mr),
                "TSR": float(tsr),
            }
        return out

    def log_overall(self, num_scenarios: int):
        total_frames = sum(sc["frames"] for sc in self.scenario_summaries)

        if total_frames > 0:
            ade_sum = sum(
                sc["ade"] * sc["frames"]
                for sc in self.scenario_summaries
                if sc["frames"] > 0 and not math.isnan(sc["ade"])
            )
            fde_sum = sum(
                sc["fde"] * sc["frames"]
                for sc in self.scenario_summaries
                if sc["frames"] > 0 and not math.isnan(sc["fde"])
            )
            mean_ade = ade_sum / total_frames
            mean_fde = fde_sum / total_frames
        else:
            mean_ade = float("nan")
            mean_fde = float("nan")

        mean_msne = (self.overall_msne / self.overall_msne_count) if self.overall_msne_count > 0 else float("nan")

        overall_miss_rate = (self.overall_missed / self.total_gt_objects) if self.total_gt_objects > 0 else float("nan")

        overall_tsr = (self.overall_tsr_success / self.overall_tsr_N) if self.overall_tsr_N > 0 else float("nan")
        overall_tsr_matched = (
            self.overall_tsr_success_matched / self.overall_tsr_N_matched
            if self.overall_tsr_N_matched > 0 else float("nan")
        )

        th = self.traj_success_threshold if self.traj_success_threshold is not None else float("nan")

        self.logger.info("\n===========  OVERALL  ===========")
        self.logger.info(f"Scenarios evaluated            : {num_scenarios}")
        self.logger.info(f"Mean ADE                       : {mean_ade:.4f}")
        self.logger.info(f"Mean FDE                       : {mean_fde:.4f}")
        self.logger.info(f"Mean MSNE                      : {mean_msne:.3f}")
        self.logger.info(f"Mean Miss Rate (overall)       : {overall_miss_rate:.4f}")
        self.logger.info(f"Traj Success Rate (GT)         : {overall_tsr:.4f} @ thresh={th}")
        self.logger.info(f"Traj Success Rate (matched-only): {overall_tsr_matched:.4f}")
        if self.store_distributions:
            self.logger.info(f"Stored FDE samples             : {len(self.fde_all)}")

        # -------- PER CATEGORY (UPDATED: show MR/TSR) --------
        self.logger.info("\n=======  PER CATEGORY  ==========")
        cat_sum = self._per_category_summary()
        for cat, v in cat_sum.items():
            totals = self.cat_totals.get(cat, {})
            num_fp = int(totals.get("num_false_positives", 0))
            num_matched = int(totals.get("num_matched", 0))
            num_missed = int(totals.get("num_missed", 0))
            N_gt = int(v.get("N_gt", 0))

            if v["frames"] > 0:
                self.logger.info(
                    f"[{cat:<12}] frames={v['frames']:3d}  "
                    f"ADE={v['ADE_mean']:.4f}  FDE={v['FDE_mean']:.4f}  "
                    f"N_gt={N_gt:5d}  matched={num_matched:5d}  missed={num_missed:5d}  FP={num_fp:5d}  "
                    f"MR={v['MR']:.4f}  TSR={v['TSR']:.4f}"
                )
            else:
                self.logger.info(
                    f"[{cat:<12}] no matched frames  "
                    f"N_gt={N_gt:5d}  matched={num_matched:5d}  missed={num_missed:5d}  FP={num_fp:5d}  "
                    f"MR={v['MR']:.4f}  TSR={v['TSR']:.4f}"
                )

        # ------- BY SOURCE (unchanged) ------- #
        self.logger.info("\n=======  BY SOURCE (FUSION)  ==========")

        label_map = {
            "category_I_ego": "Category I (ego)",
            "category_I_fused": "Category I (fused)",
            "category_II_fused": "Category II (fused)",
            "no_fusion": "No fusion (ego only)",
        }

        for key, ent in self.source_totals.items():
            frames = ent.get("frames", 0)
            num = ent.get("num", 0)
            if frames > 0:
                ade_mean_s = ent["ade_sum"] / frames
                fde_mean_s = ent["fde_sum"] / frames
            else:
                ade_mean_s = float("nan")
                fde_mean_s = float("nan")

            self.logger.info(
                f"[{label_map[key]:<22}] frames_with_data={frames:3d}  "
                f"objects={num:5d}  ADE={ade_mean_s:.4f}  FDE={fde_mean_s:.4f}"
            )

        self.logger.info(f"\nOverall missed : {self.overall_missed}")
