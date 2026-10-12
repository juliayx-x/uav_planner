import numpy as np
from uav_planner.voxel_grid import VoxelGrid

class MazeMapGenerator:
    def __init__(self, seed=42):
        self.seed = seed

    def generate(self, shape=(50, 50, 20)):
        rng = np.random.default_rng(self.seed)
        vg = VoxelGrid(*shape)
        vg.grid[:, :, :] = 1                            # start with every cell as obstacle

        step = 2
        cell_xs = list(range(0, shape[0], step))        # gives every even coordinate for x, y, z
        cell_xy = list(range(0, shape[1], step))
        cell_xz = list(range(0, shape[2], step))    

        start = (int(rng.choice(cell_xs)), int(rng.choice(cell_xy)), int(rng.choice(cell_xz)))

        visited = set()
        vg.grid[start] = 0
        visited.add(start)
        stack = [start]
        directions = [(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)]

        while stack:
            cx, cy, cz = stack[-1]          # look at (don't remove) current cell

            neighbors = []
            for dx, dy, dz in directions:
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]
                        and (nx, ny, nz) not in visited):
                    neighbors.append((nx, ny, nz))

            if neighbors:
                idx = rng.integers(0, len(neighbors))
                nx, ny, nz = neighbors[idx]

                wall = ((cx+nx)//2, (cy+ny)//2, (cz+nz)//2)
                vg.grid[wall] = 0
                vg.grid[nx, ny, nz] = 0

                visited.add((nx, ny, nz))
                stack.append((nx, ny, nz))
            else:
                stack.pop()

        return vg

