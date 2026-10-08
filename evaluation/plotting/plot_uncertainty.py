"""Plots of scalar CDF calibration and normalized sparsification curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _title(title, results):
    vehicle_label = results.get("vehicle_label")
    return title if vehicle_label is None else f"{title} — {vehicle_label}"


def _metric(value):
    return "undefined" if value is None else f"{value:.4f}"


def _save(figure, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def plot_uncertainty_metrics(results, output_dir):
    """Save overall CE and AUSE curves to an explicit output directory.

    Return a mapping of available plot names to their ``Path`` destinations.
    Curves with no valid samples are omitted. Each vehicle can be directed to
    its own directory, and ``vehicle_label`` is included in figure titles.
    The caller supplies normalized sparsification curves; this function does
    not recompute metrics or aggregate curves across groups.
    """
    output_dir = Path(output_dir)
    values = results.get("overall", {})
    paths = {}
    calibration = values.get("cdf_calibration") or {}
    calibration_series = []
    for key in ("x", "y"):
        curve = calibration.get(key)
        if curve is None or curve.get("count", 1) <= 0:
            continue
        confidence = np.asarray(curve.get("confidence", []), dtype=float)
        coverage = np.asarray(curve.get("coverage", []), dtype=float)
        if confidence.ndim != 1 or confidence.shape != coverage.shape:
            continue
        finite = np.isfinite(confidence) & np.isfinite(coverage)
        if np.any(finite):
            calibration_series.append((key, {
                **curve, "confidence": confidence[finite], "coverage": coverage[finite],
            }))
    if calibration_series:
        figure, axes = plt.subplots(
            1, len(calibration_series), figsize=(5.5 * len(calibration_series), 4.5),
            squeeze=False, constrained_layout=True,
        )
        for axis, (key, curve) in zip(axes.flat, calibration_series):
            axis.plot([0, 1], [0, 1], "k--", label="Ideal")
            axis.plot(
                curve["confidence"], curve["coverage"], marker="o",
                label=f"CE_{key}={_metric(curve.get('CE'))}",
            )
            axis.set(
                xlim=(0, 1), ylim=(0, 1),
                xlabel="Nominal CDF probability",
                ylabel="Empirical CDF coverage",
                title=f"{key} coordinate",
            )
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")
        figure.suptitle(_title("Gaussian marginal CDF calibration", results))
        paths["cdf_calibration"] = _save(
            figure, output_dir / "cdf_calibration.png",
        )

    sparsification = values.get("sparsification") or {}
    sparsification_series = [
        (key, sparsification[key])
        for key in ("x", "y", "position")
        if key in sparsification
        and sparsification[key] is not None
        and len(sparsification[key].get("removal_fraction", [])) > 0
        and len(sparsification[key].get("uncertainty", [])) > 0
        and len(sparsification[key].get("oracle", [])) > 0
    ]
    if sparsification_series:
        figure, axes = plt.subplots(
            1, len(sparsification_series),
            figsize=(5.5 * len(sparsification_series), 4.5),
            squeeze=False, constrained_layout=True,
        )
        for axis, (key, curve) in zip(axes.flat, sparsification_series):
            fractions = curve["removal_fraction"]
            axis.plot(fractions, curve["uncertainty"], label="Uncertainty ranking")
            axis.plot(fractions, curve["oracle"], label="Oracle error ranking")
            axis.fill_between(
                fractions, curve["oracle"], curve["uncertainty"], alpha=0.15,
            )
            if len(curve.get("error", [])) == len(fractions):
                axis.plot(
                    fractions, curve["error"], linestyle="--",
                    label="Sparsification error",
                )
            suffix = "2D" if key == "position" else key
            coordinate = "2D position" if key == "position" else f"{key} coordinate"
            axis.set(
                xlim=(0, 1), xlabel="Fraction removed",
                ylabel="Error / full-set mean error",
                title=f"{coordinate}: AUSE_{suffix}={_metric(curve.get('AUSE'))}",
            )
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", fontsize=8)
        figure.suptitle(_title("Normalized sparsification", results))
        paths["sparsification"] = _save(
            figure, output_dir / "sparsification.png",
        )
    return paths
