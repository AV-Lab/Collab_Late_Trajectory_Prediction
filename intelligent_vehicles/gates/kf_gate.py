from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np


@dataclass
class KFDecision:
    passed: bool
    reason: str                 # e.g., "simple:streak", "simple:cov", "simple:nis", "kfri", "none"
    rho_cov: float              # u_now / u_med
    rho_miss: float             # normalized predict-only streak
    rho_nis: float              # med_nis / nis_thr
    kfri: Optional[float]       # combined score (None in simple mode)
    streak: int
    u_now: float
    u_med: float
    med_nis: Optional[float]

    def as_dict(self) -> Dict:
        return {
            "passed": self.passed, "reason": self.reason,
            "rho_cov": self.rho_cov, "rho_miss": self.rho_miss, "rho_nis": self.rho_nis,
            "kfri": self.kfri, "streak": self.streak, "u_now": self.u_now,
            "u_med": self.u_med, "med_nis": self.med_nis
        }


class KFGate:
    """
    Lightweight KF-only gate.
    Decide whether to fuse shared predictions based on the ego KF's recent reliability.

    Modes:
      - mode="simple": OR of three checks:
            (1) predict-only streak >= min_streak
            (2) covariance growth u_now / u_med >= cov_ratio_thr
            (3) median recent NIS >= nis_thr (scaled as rho_nis >= 1.0)
      - mode="kfri": single scalar score (KFRI) over [0, ~2]; fuse if KFRI >= kfri_thr.
            KFRI = w_cov*clip(rho_cov, cov_cap) + w_miss*rho_miss + w_nis*clip(rho_nis, nis_cap)

    Inputs (features): dict with keys produced by tracker.Track.get_kf_gate_features():
        u_now: float          # trace of predicted pos cov at current frame
        u_med: float          # median trace over recent window
        streak: int           # consecutive predict-only frames
        med_nis: Optional[float]  # median NIS over last L updates; None if no updates

    Usage:
        gate = KFOnlyReliabilityGate(mode="simple")  # or mode="kfri"
        decision = gate.decide(kf_features)          # -> KFDecision
        if decision.passed:
            # proceed to fuse; else keep ego-only
    """

    def __init__(self,
                 mode: str = "simple",
                 # --- simple mode thresholds ---
                 min_streak: int = 2,
                 cov_ratio_thr: float = 2.0,
                 nis_thr: float = 5.99,
                 # --- kfri mode weights/thresholds ---
                 w_cov: float = 0.5,
                 w_miss: float = 0.3,
                 w_nis: float = 0.2,
                 kfri_thr: float = 0.6,
                 cov_cap: float = 2.0,
                 nis_cap: float = 2.0):
        assert mode in ("simple", "kfri")
        self.mode = mode

        # simple thresholds
        self.min_streak = int(min_streak)
        self.cov_ratio_thr = float(cov_ratio_thr)
        self.nis_thr = float(nis_thr)

        # kfri settings
        self.w_cov = float(w_cov)
        self.w_miss = float(w_miss)
        self.w_nis = float(w_nis)
        self.kfri_thr = float(kfri_thr)
        self.cov_cap = float(cov_cap)
        self.nis_cap = float(nis_cap)

    def decide(self, features: Dict) -> KFDecision:
        eps = 1e-9

        u_now = float(features.get("u_now", 0.0))
        u_med = float(features.get("u_med", max(u_now, eps)))
        streak = int(features.get("streak", 0))
        med_nis = features.get("med_nis", None)

        # ratios
        rho_cov = u_now / max(u_med, eps)
        rho_miss = min(streak / 3.0, 1.0)  # 0, 1/3, 2/3, 1 for 0,1,2,>=3 consecutive predicts
        if med_nis is None:
            # If we haven't had an update recently, treat as slightly above 95% threshold
            med_nis = self.nis_thr * 1.01
        rho_nis = float(med_nis) / self.nis_thr

        if self.mode == "simple":
            passed, reason = False, "none"
            if streak >= self.min_streak:
                passed, reason = True, "simple:streak"
            elif rho_cov >= self.cov_ratio_thr:
                passed, reason = True, "simple:cov"
            elif rho_nis >= 1.0:
                passed, reason = True, "simple:nis"

            return KFDecision(
                passed=passed, reason=reason,
                rho_cov=rho_cov, rho_miss=rho_miss, rho_nis=rho_nis,
                kfri=None, streak=streak, u_now=u_now, u_med=u_med, med_nis=med_nis
            )

        # mode == "kfri"
        kfri = (self.w_cov * min(rho_cov, self.cov_cap) +
                self.w_miss * rho_miss +
                self.w_nis * min(rho_nis, self.nis_cap))
        passed = kfri >= self.kfri_thr
        return KFDecision(
            passed=passed, reason="kfri" if passed else "none",
            rho_cov=rho_cov, rho_miss=rho_miss, rho_nis=rho_nis,
            kfri=kfri, streak=streak, u_now=u_now, u_med=u_med, med_nis=med_nis
        )
