import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from uav_planner.paths import FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/results.csv")
df_s = df[df["success"] == True].copy()

ENV_ORDER  = [e for e in ["random", "canyon", "clustered", "maze"]
              if e in df["env_class"].unique()]
ENV_LABELS = [e.capitalize() for e in ENV_ORDER]

ALGO_COLORS = {"astar": "#2196F3", "rrt_star": "#FF9800"}
ALGO_LABELS = {"astar": "A*", "rrt_star": "RRT*"}
ALGOS = [a for a in ["astar", "rrt_star"] if a in df["algorithm"].unique()]


def savefig(name):
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# success rate heatmap
sr = (
    df.groupby(["algorithm", "env_class"])["success"]
    .mean()
    .unstack()
    .reindex(columns=ENV_ORDER)
    .rename(columns=dict(zip(ENV_ORDER, ENV_LABELS)))
    .rename(index=ALGO_LABELS)
)

fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(ENV_ORDER)), 2.8))
im = ax.imshow(sr.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(sr.columns))); ax.set_xticklabels(sr.columns, fontsize=11)
ax.set_yticks(range(len(sr.index)));   ax.set_yticklabels(sr.index, fontsize=11)
for i in range(sr.shape[0]):
    for j in range(sr.shape[1]):
        v = sr.values[i, j]
        if np.isnan(v):
            continue
        ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if v < 0.4 or v > 0.85 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Success rate")
ax.set_title("Success Rate by Algorithm × Environment", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig("success_rate_heatmap.png")


# helper: violin panel
def violin_panel(metric, title, ylabel, log_scale=False, sharey=True):
    fig, axes = plt.subplots(1, len(ENV_ORDER),
                             figsize=(4 * len(ENV_ORDER), 5),
                             sharey=sharey)
    if len(ENV_ORDER) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, env, label in zip(axes, ENV_ORDER, ENV_LABELS):
        parts_data, parts_algos = [], []
        for algo in ALGOS:
            data = df_s[(df_s.env_class == env) & (df_s.algorithm == algo)][metric].dropna()
            if len(data) >= 5:
                parts_data.append(data.values)
                parts_algos.append(algo)

        if parts_data:
            vp = ax.violinplot(parts_data, positions=range(len(parts_data)),
                               showmedians=True, showextrema=False)
            for pc, algo in zip(vp["bodies"], parts_algos):
                pc.set_facecolor(ALGO_COLORS[algo])
                pc.set_alpha(0.75)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")

        ax.set_xticks(range(len(parts_algos)))
        ax.set_xticklabels([ALGO_LABELS[a] for a in parts_algos], fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylabel(ylabel, fontsize=11)
    fig.tight_layout()
    return fig


# time violin
violin_panel("time_ms", "Planning Time (ms) — A* vs RRT*",
             "Time (ms, log scale)", log_scale=True, sharey=True)
savefig("time_ms_violin.png")

# path length violin
violin_panel("path_length", "Path Length (voxels) — A* vs RRT*",
             "Path length (voxels)", sharey=False)
savefig("path_length_violin.png")

# smoothness violin
violin_panel("smoothness",
             "Path Smoothness (direction changes) — A* vs RRT*  [lower = smoother]",
             "Direction changes", sharey=False)
savefig("smoothness_violin.png")


# panel bar summary
metrics = [
    ("time_ms",             "Mean Planning Time (ms)",            True),
    ("path_length",         "Mean Path Length (voxels)",          False),
    ("smoothness",          "Mean Direction Changes (smoothness)", False),
    ("altitude_utilisation","Mean Altitude Utilisation",          False),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle("Algorithm Comparison: Mean Metrics by Environment",
             fontsize=13, fontweight="bold")

x = np.arange(len(ENV_LABELS))
width = 0.35

for ax, (metric, ylabel, log_scale) in zip(axes, metrics):
    for i, algo in enumerate(ALGOS):
        means, errs = [], []
        for env in ENV_ORDER:
            subset = df_s[(df_s.algorithm == algo) & (df_s.env_class == env)][metric].dropna()
            means.append(subset.mean() if len(subset) else np.nan)
            errs.append(subset.sem() * 1.96 if len(subset) > 1 else 0)
        offset = (i - (len(ALGOS) - 1) / 2) * width
        ax.bar(x + offset, means, width, label=ALGO_LABELS[algo],
               color=ALGO_COLORS[algo], alpha=0.85,
               yerr=errs, capsize=4, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(ENV_LABELS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

fig.tight_layout()
savefig("metric_bars.png")


# success rate bar chart
fig, ax = plt.subplots(figsize=(max(6, 2 * len(ENV_ORDER)), 4))
x = np.arange(len(ENV_LABELS))
for i, algo in enumerate(ALGOS):
    vals = [df[(df.algorithm == algo) & (df.env_class == env)]["success"].mean()
            for env in ENV_ORDER]
    offset = (i - (len(ALGOS) - 1) / 2) * width
    ax.bar(x + offset, vals, width, label=ALGO_LABELS[algo],
           color=ALGO_COLORS[algo], alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(ENV_LABELS, fontsize=11)
ax.set_ylabel("Success rate", fontsize=11)
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend(fontsize=11)
ax.set_title("Success Rate by Algorithm × Environment", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
savefig("success_rate_bar.png")

print("\nAll plots saved to", FIGURES_DIR)