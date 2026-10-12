from uav_planner.voxel_grid import VoxelGrid
import numpy as np
from uav_planner.maps.bfs import path_exists

class ClusteredMapGenerator:
    def __init__(self, seed=42):
        self.seed = seed

    def _place_clusters(self, rng, shape, n_clusters):
        vg = VoxelGrid(*shape)
        for _ in range(n_clusters):
            cx = rng.integers(5, shape[0]-5)
            cy = rng.integers(5, shape[1]-5)
            cz = rng.integers(2, shape[2]-2)
            r  = rng.integers(3, 7)
            for x in range(max(0, cx-r), min(shape[0], cx+r)):
                for y in range(max(0, cy-r), min(shape[1], cy+r)):
                    for z in range(max(0, cz-r), min(shape[2], cz+r)):
                        if np.linalg.norm([x-cx, y-cy, z-cz]) <= r:
                            vg.grid[x, y, z] = 1
        return vg

    def generate(self, shape=(50, 50, 20), start=(0, 0, 0), goal=None, n_clusters=None):
        if goal is None:
            goal = (shape[0]-1, shape[1]-1, shape[2]-1)

        for attempt in range(10):
            rng = np.random.default_rng(self.seed + attempt)

            if n_clusters != None:
                n = n_clusters
            else:
                n = rng.integers(8,16)

            vg = self._place_clusters(rng, shape, n)

            vg.grid[start] = 0
            vg.grid[goal] = 0

            if path_exists(vg.grid, start, goal):
                return vg

        raise ValueError("Could not generate a connected clustered map after 10 attempts")

