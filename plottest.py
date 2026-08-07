import matplotlib.pyplot as plt
import numpy as np
import os
from CanyonMap import CanyonMapGenerator
from ClusteredMap import ClusteredMapGenerator
from MazeMap import MazeMapGenerator

print(os.getcwd())
print(os.listdir("maps"))

random_map    = np.load("maps/random_seed42.npy")
canyon_map    = np.load("maps/canyon_seed42.npy")
clustered_map = np.load("maps/clustered_seed42.npy")
maze_map      = np.load("maps/maze_seed42.npy")

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
plt.savefig("all_maps_comparison.png", dpi=150)
plt.show()