import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = np.loadtxt("listener_delay_log.txt", delimiter=",")
df = pd.DataFrame(data, columns=["size_bytes", "delay_ms"])


df["size_group"] = pd.cut(
    df["size_bytes"],
    bins=[0, 400, 900, np.inf],
    labels=["<=400 B", "401-900 B", ">900 B"],
    include_lowest=True
)


sns.set_style("whitegrid")


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="size_bytes", y="delay_ms", s=30, alpha=0.6)
plt.xlabel("Packet size (bytes)")
plt.ylabel("Delay (ms)")
plt.title("Packet size vs delay")
plt.tight_layout()
plt.savefig("scatter_packet_delay.png", dpi=600, bbox_inches="tight")
plt.close()


plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="size_group", y="delay_ms", inner="quartile")
plt.xlabel("Packet size range")
plt.ylabel("Delay (ms)")
plt.title("Delay distribution")
plt.tight_layout()
plt.savefig("delay_distribution.png", dpi=600, bbox_inches="tight")
plt.close()

summary = df.groupby("size_group")["delay_ms"].agg(["count", "mean", "median", "min", "max"])
print(summary)