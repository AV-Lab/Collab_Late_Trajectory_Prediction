import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.paths import OUTPUTS_PLOTS_DIR
from evaluation.paths import (
    DELAY_DISTRIBUTION_PLOT,
    DELAY_SCATTER_PLOT,
    DELAY_TMP_DIR,
)


def plot_delay():
    delay_files = sorted(DELAY_TMP_DIR.glob("*.txt"))
    if not delay_files:
        raise FileNotFoundError(f"No delay files were found in {DELAY_TMP_DIR}")

    measurements = []
    for delay_file in delay_files:
        values = np.loadtxt(delay_file, delimiter=",", dtype=np.float64)
        values = np.atleast_2d(values)
        if values.shape[1] != 2:
            raise ValueError(
                f"Delay file must contain message_size_bytes,delay_ms: {delay_file}"
            )

        frame = pd.DataFrame(values, columns=["size_bytes", "delay_ms"])
        frame["vehicle"] = delay_file.stem
        measurements.append(frame)

    data = pd.concat(measurements, ignore_index=True)
    data["size_group"] = pd.cut(
        data["size_bytes"],
        bins=[0, 400, 900, np.inf],
        labels=["≤400 B", "401–900 B", ">900 B"],
        include_lowest=True,
    )

    sns.set_theme(style="whitegrid")

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x="size_bytes",
        y="delay_ms",
        s=30,
        alpha=0.6,
        ax=axis,
    )
    axis.set_xlabel("Packet size (bytes)")
    axis.set_ylabel("Delay (ms)")
    axis.set_title("Packet size vs delay")
    figure.tight_layout()
    OUTPUTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(DELAY_SCATTER_PLOT, dpi=600, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=data,
        x="size_group",
        y="delay_ms",
        inner="quartile",
        ax=axis,
    )
    axis.set_xlabel("Packet size range")
    axis.set_ylabel("Delay (ms)")
    axis.set_title("Delay distribution")
    figure.tight_layout()
    OUTPUTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(DELAY_DISTRIBUTION_PLOT, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return DELAY_SCATTER_PLOT, DELAY_DISTRIBUTION_PLOT
