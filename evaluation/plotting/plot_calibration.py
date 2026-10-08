from pathlib import Path

import matplotlib.pyplot as plt


SOURCE_LABELS = {
    "category_I_ego": "Category I ego",
    "category_I_fused": "Category I fused",
    "category_II_fused": "Category II fused",
    "no_fusion": "No fusion ego",
}


def plot_calibration_curve(results, output_path=None):
    """Plot overall and prediction-source Gaussian coverage curves."""
    series = [("Overall", results["overall"])]
    series.extend(
        (SOURCE_LABELS.get(source, source), values)
        for source, values in results.get("by_source", {}).items()
    )
    series = [
        (label, values)
        for label, values in series
        if values.get("calibration") is not None
    ]
    if not series:
        return None

    figure, axis = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Ideal")
    for label, values in series:
        calibration = values["calibration"]
        axis.plot(
            calibration["confidence"],
            calibration["coverage"],
            marker="o",
            linewidth=2,
            label=(
                f"{label} (coverage_ECE_L1="
                f"{values.get('coverage_ECE_L1', values.get('ECE', float('nan'))):.4f})"
            ),
        )

    title = "Gaussian ellipse coverage"
    if results.get("vehicle_label") is not None:
        title += f" — {results['vehicle_label']}"
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Expected confidence",
        ylabel="Empirical coverage",
        title=title,
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    if output_path is None:
        from evaluation.paths import CALIBRATION_PLOT

        output_path = CALIBRATION_PLOT
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path
