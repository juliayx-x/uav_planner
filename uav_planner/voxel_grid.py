import numpy as np
import matplotlib.pyplot as plt

class VoxelGrid:
    def __init__(self, sx=50, sy=50, sz=20):
        self.grid = np.zeros((sx, sy, sz), dtype=np.uint8)
        self.shape = (sx, sy, sz)

    def in_bounds(self, x, y, z):
        return 0 <= x < self.shape[0] and \
               0 <= y < self.shape[1] and \
               0 <= z < self.shape[2]

    def is_free(self, x, y, z):
        return self.in_bounds(x, y, z) and self.grid[x, y, z] == 0

    def set_obstacle(self, x, y, z):
        self.grid[x, y, z] = 1

    def get_neighbours(self, x, y, z):
        neighbours = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if self.is_free(nx, ny, nz):
                        neighbours.append((nx, ny, nz))
        return neighbours

    def visualise(self, start=None, goal=None, path=None):
        obstacles = np.argwhere(self.grid == 1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2],
                   c='gray', alpha=0.3, s=10, label='Obstacles')
        if start:
            ax.scatter(*start, c='green', s=60, zorder=5, label='Start')
        if goal:
            ax.scatter(*goal, c='red', s=60, zorder=5, label='Goal')
        if path:
            px, py, pz = zip(*path)
            ax.plot3D(px, py, pz, c='blue', linewidth=2, label='Path')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    vg = VoxelGrid()

    # load random obstacles into the class
    obstacle_positions = np.argwhere(
        (rng.random((50, 50, 20)) < 0.15).astype(np.uint8) == 1
    )
    for x, y, z in obstacle_positions:
        vg.set_obstacle(x, y, z)

    # guarantee start and goal are free
    vg.grid[3, 3, 2] = 0
    vg.grid[47, 47, 18] = 0

    # test methods
    print(vg.in_bounds(3, 3, 2))          # → True
    print(vg.in_bounds(99, 0, 0))         # → False
    print(vg.is_free(3, 3, 2))            # → True
    print(vg.get_neighbours(3, 3, 2))     # → list of free adjacent cells

    # visualise
    vg.visualise(
        start=(3, 3, 2),
        goal=(47, 47, 18),
        path=[(3,3,2), (10,10,5), (25,25,10), (47,47,18)]
    )