import numpy as np

rng = np.random.default_rng(seed=42)
grid = (rng.random((50, 50, 20)) < 0.15).astype(np.uint8)  # 15% obstacle density
total_voxels = grid.size
obstacle_count = grid.sum()
density = obstacle_count / total_voxels
print(f"Obstacle count: {obstacle_count}")
print(f"Actual density: {density:.4f}")                    # verifying density
obstacle_coords = np.argwhere(grid == 1)
print(f"Total obstacles: {len(obstacle_coords)}")

grid[3, 3, 2] = 0                                          # clearing start and goal positions
grid[47, 47, 18] = 0

print("Start and goal cleared.")