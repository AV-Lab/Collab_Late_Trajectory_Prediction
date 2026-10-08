import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.paths import OUTPUTS_PLOTS_DIR
from evaluation.paths import RUNTIME_PLOT, RUNTIME_TMP_DIR


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


def plot_runtime():
    runtime_files = sorted(RUNTIME_TMP_DIR.glob("*.csv"), key=_natural_key)
    if not runtime_files:
        raise FileNotFoundError(f"No runtime files were found in {RUNTIME_TMP_DIR}")

    positions = []
    series = []
    means = []
    centers = []
    labels = []

    pair_gap = 0.25
    group_gap = 0.75
    current_position = 1.0

    for runtime_file in runtime_files:
        values = pd.read_csv(runtime_file)
        individual = pd.to_numeric(
            values["individual_ms"], errors="coerce"
        ).dropna().to_numpy()
        collaborative = pd.to_numeric(
            values["collaborative_ms"], errors="coerce"
        ).dropna().to_numpy()

        if individual.size == 0 or collaborative.size == 0:
            raise ValueError(f"Runtime file contains no measurements: {runtime_file}")

        positions.extend(
            [current_position - pair_gap / 2, current_position + pair_gap / 2]
        )
        series.extend([individual, collaborative])
        means.extend([float(np.mean(individual)), float(np.mean(collaborative))])
        centers.append(current_position)
        labels.append(runtime_file.stem.replace("_", " ").title())
        current_position += group_gap

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12.8, 5.2))
    boxplot = axis.boxplot(
        series,
        positions=positions,
        widths=0.25,
        showfliers=True,
        patch_artist=True,
    )

    palette = sns.color_palette("Blues", n_colors=6)
    individual_color = palette[2]
    collaborative_color = palette[4]
    colors = [individual_color, collaborative_color] * len(runtime_files)

    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.9)

    for element in ("whiskers", "caps", "medians"):
        for line in boxplot[element]:
            line.set_linewidth(1.2)

    axis.set_yscale("log")
    axis.set_ylabel("Time (ms), log scale", fontsize=18)
    axis.set_xticks(centers)
    axis.set_xticklabels(labels, fontsize=18)

    for position, mean in zip(positions, means):
        axis.text(
            position,
            mean,
            f"μ={mean:.1f}",
            fontsize=14,
            ha="center",
            va="center",
        )

    axis.plot([], [], color=individual_color, linewidth=8, label="Individual")
    axis.plot([], [], color=collaborative_color, linewidth=8, label="Collaborative")
    axis.legend(loc="upper left", fontsize=16, frameon=True)

    figure.tight_layout()
    OUTPUTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(RUNTIME_PLOT, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return RUNTIME_PLOT
