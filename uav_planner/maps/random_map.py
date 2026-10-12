import numpy as np  
from uav_planner.voxel_grid import VoxelGrid
from uav_planner.maps.bfs import path_exists

class RandomMapGenerator:
    def __init__(self, seed=42, density=0.15):
        self.seed = seed
        self.density = density

    def _place_obstacles(self, rng, shape):
        vg = VoxelGrid(*shape)
        n_voxels = shape[0] * shape[1] * shape[2]
        n_obstacles = int(n_voxels * self.density)
        xs = rng.integers(0, shape[0], n_obstacles)
        ys = rng.integers(0, shape[1], n_obstacles)
        zs = rng.integers(0, shape[2], n_obstacles)
        vg.grid[xs, ys, zs] = 1
        return vg

    def generate(self, shape=(50, 50, 20), start=(0, 0, 0), goal=None):
        if goal is None:
            goal = (shape[0]-1, shape[1]-1, shape[2]-1)

        for attempt in range(10):
            rng = np.random.default_rng(self.seed + attempt)  # new internal seed each retry
            vg = self._place_obstacles(rng, shape)
            vg.grid[start] = 0
            vg.grid[goal] = 0

            if path_exists(vg.grid, start, goal):
                return vg

        raise ValueError(f"Could not generate a connected map after 10 attempts (density={self.density})")

