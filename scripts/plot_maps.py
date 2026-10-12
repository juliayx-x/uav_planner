import matplotlib.pyplot as plt
import numpy as np
from uav_planner.paths import MAPS_DIR, FIGURES_DIR

random_map    = np.load(MAPS_DIR / "random_seed42.npy")
canyon_map    = np.load(MAPS_DIR / "canyon_seed42.npy")
clustered_map = np.load(MAPS_DIR / "clustered_seed42.npy")
maze_map      = np.load(MAPS_DIR / "maze_seed42.npy")

fig = plt.figure(figsize=(12, 12))

specs = [
    (random_map,    "Random",    "steelblue"),
    (canyon_map,    "Canyon",    "indianred"),
    (clustered_map, "Clustered", "seagreen"),
    (maze_map,      "Maze",      "goldenrod"),
]

for i, (grid, title, color) in enumerate(specs):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.voxels(grid, facecolor=color, edgecolor='k', alpha=0.6)
    ax.set_title(title)

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "all_maps_comparison.png", dpi=150)
plt.show()