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


# --- usage ---
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

import numpy as np
import os

class RandomMapGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate(self, density=0.15, shape=(50, 50, 20)):
        vg = VoxelGrid(*shape)
        mask = self.rng.random(shape) < density
        vg.grid[mask] = 1
        vg.grid[3, 3, 2] = 0       # start always free
        vg.grid[47, 47, 18] = 0    # goal always free
        return vg

    def load(self, filepath):
        vg = VoxelGrid()
        vg.grid = np.load(filepath)
        vg.shape = vg.grid.shape
        return vg


# --- generate and save one map ---
os.makedirs("maps", exist_ok=True)

gen = RandomMapGenerator(seed=0)
vg = gen.generate(density=0.15)

print(f"Grid shape:     {vg.grid.shape}")
print(f"Obstacle count: {vg.grid.sum()}")
print(f"Actual density: {vg.grid.sum() / vg.grid.size:.4f}")
print(f"Start free:     {vg.is_free(3, 3, 2)}")
print(f"Goal free:      {vg.is_free(47, 47, 18)}")

np.save("maps/random_d15_01.npy", vg.grid)
print("Map saved.")

# --- load it back and verify ---
gen2 = RandomMapGenerator()
loaded_vg = gen2.load("maps/random_d15_01.npy")

print(f"Identical:   {np.array_equal(vg.grid, loaded_vg.grid)}")
print(f"dtype:       {loaded_vg.grid.dtype}")
print(f"shape:       {loaded_vg.grid.shape}")

# --- generate a batch of 5 maps ---
for i in range(5):
    g = RandomMapGenerator(seed=i)
    m = g.generate(density=0.15)
    path = f"maps/random_d15_{i:02d}.npy"
    np.save(path, m.grid)
    print(f"Saved {path} — obstacles: {m.grid.sum()}")