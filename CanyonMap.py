from voxel_grid import VoxelGrid
from bfs import path_exists
import numpy as np

class CanyonMapGenerator:
    def __init__(self, seed=42):
        self.seed = seed

    def generate(self, shape=(50, 50, 20), start=(1, 1, 0), goal=None,
                 wall_gap_min=8, wall_gap_max=12):
        if goal is None:
            goal = (shape[0]-2, shape[1]-2, 0)

        for attempt in range(10):
            rng = np.random.default_rng(self.seed + attempt)
            vg = VoxelGrid(*shape)

            # place walls at irregular x-intervals, not a fixed step
            wx = wall_gap_min
            while wx < shape[0] - 2:
                vg.grid[wx, :, :15] = 1  # full y-width, z=0..14 (half-height)

                n_gaps = rng.integers(2, 4)          # 2-3 gaps
                for _ in range(n_gaps):
                    gw = rng.integers(3, 6)            # width 3-5
                    gy = rng.integers(0, shape[1] - gw)  # stay in-bounds
                    vg.grid[wx, gy:gy+gw, :15] = 0 

                wx += rng.integers(wall_gap_min, wall_gap_max + 1)

            vg.grid[start] = 0
            vg.grid[goal] = 0

            if path_exists(vg.grid, start, goal):
                return vg

        raise ValueError("Could not generate a connected canyon map after 10 attempts")