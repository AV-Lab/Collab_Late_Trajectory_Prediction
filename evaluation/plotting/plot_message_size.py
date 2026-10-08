import re

import matplotlib.pyplot as plt
import numpy as np

from evaluation.paths import OUTPUTS_PLOTS_DIR
from evaluation.paths import MESSAGE_SIZE_PLOT, MESSAGE_SIZE_TMP_DIR


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


def plot_message_size():
    message_files = sorted(MESSAGE_SIZE_TMP_DIR.glob("*.txt"), key=_natural_key)
    if not message_files:
        raise FileNotFoundError(
            f"No message-size files were found in {MESSAGE_SIZE_TMP_DIR}"
        )

    series = []
    means = []
    labels = []

    for message_file in message_files:
        sizes = np.loadtxt(message_file, dtype=np.float64)
        sizes = np.atleast_1d(sizes)
        sizes = sizes[np.isfinite(sizes)]
        if sizes.size == 0:
            raise ValueError(f"Message-size file is empty: {message_file}")

        series.append(np.rint(sizes).astype(np.int64))
        means.append(float(np.mean(sizes)))
        labels.append(message_file.stem.replace("_", " ").title())

    positions = np.arange(1, len(series) + 1)
    figure, axis = plt.subplots(figsize=(13.5, 5.2))
    boxplot = axis.boxplot(
        series,
        positions=positions,
        widths=0.45,
        showfliers=True,
        patch_artist=True,
    )

    box_color = plt.get_cmap("Greens")(0.70)
    for box in boxplot["boxes"]:
        box.set_facecolor(box_color)
        box.set_alpha(0.90)
        box.set_linewidth(1.2)

    for element in ("whiskers", "caps", "medians"):
        for line in boxplot[element]:
            line.set_linewidth(1.2)

    axis.set_ylabel("Message size (bytes)", fontsize=18)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=16)
    axis.tick_params(axis="y", labelsize=14)

    for position, mean in zip(positions, means):
        axis.text(
            position,
            mean,
            f"μ={mean:.0f}",
            fontsize=14,
            ha="center",
            va="center",
        )

    axis.grid(True, which="major", axis="y", linewidth=0.8, alpha=0.35)
    figure.tight_layout()
    OUTPUTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(MESSAGE_SIZE_PLOT, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return MESSAGE_SIZE_PLOT
