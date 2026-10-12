import numpy as np
import matplotlib.pyplot as plt

# regenerate grid
rng = np.random.default_rng(seed=42)
grid = (rng.random((50, 50, 20)) < 0.15).astype(np.uint8)
grid[3, 3, 2] = 0
grid[47, 47, 18] = 0

obstacles = np.argwhere(grid == 1)

# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# obstacles
ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2],
           c='gray', alpha=0.3, s=10, label='Obstacles')

# start and goal
ax.scatter([3],  [3],  [2],  c='green', s=60, zorder=5, label='Start')
ax.scatter([47], [47], [18], c='red',   s=60, zorder=5, label='Goal')

# make example path
waypoints = [(3,3,2), (10,10,5), (20,20,10), (35,35,15), (47,47,18)]
wx = [p[0] for p in waypoints]
wy = [p[1] for p in waypoints]
wz = [p[2] for p in waypoints]
ax.plot3D(wx, wy, wz, c='blue', linewidth=2, label='Path')

# labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (height)')
ax.set_title('UAV Voxel Map')
plt.legend()
plt.show()
