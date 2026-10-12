import numpy as np
from uav_planner.planners.rrt_star import RRTStarPlanner, dist_3d


class SimpleGrid:
    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.shape = (size, size, size)
        self.obstacles = obstacles or set()

    def is_free(self, x, y, z):
        if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size):
            return False
        return (x, y, z) not in self.obstacles


def test_finds_path_in_empty_grid():
    grid = SimpleGrid(size=20)
    rng = np.random.default_rng(seed=1)
    planner = RRTStarPlanner(grid, (0, 0, 0), (15, 15, 15), rng=rng)
    path = planner.search(max_iters=1000)

    assert path is not None
    assert len(path) > 0


def test_returns_none_when_blocked():
    size = 10
    obstacles = set()
    gx, gy, gz = 8, 8, 8
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = gx+dx, gy+dy, gz+dz
                if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                    obstacles.add((nx, ny, nz))

    grid = SimpleGrid(size=size, obstacles=obstacles)
    rng = np.random.default_rng(seed=1)
    planner = RRTStarPlanner(grid, (0, 0, 0), (gx, gy, gz), rng=rng)
    path = planner.search(max_iters=1000, goal_threshold=1.0)

    assert path is None


def test_path_starts_at_start():
    grid = SimpleGrid(size=20)
    rng = np.random.default_rng(seed=2)
    start, goal = (1, 1, 1), (15, 15, 15)
    planner = RRTStarPlanner(grid, start, goal, rng=rng)
    path = planner.search(max_iters=1000)

    assert path is not None
    assert path[0] == start
    assert dist_3d(path[-1], goal) <= 3.0


def test_path_avoids_obstacles():
    size = 20
    obstacles = {(x, 10, 10) for x in range(size)}
    grid = SimpleGrid(size=size, obstacles=obstacles)
    rng = np.random.default_rng(seed=3)
    planner = RRTStarPlanner(grid, (0, 0, 0), (19, 19, 19), rng=rng)
    path = planner.search(max_iters=1000)

    assert path is not None
    for voxel in path:
        assert voxel not in obstacles


def test_deterministic_with_same_seed():
    grid = SimpleGrid(size=20)

    rng1 = np.random.default_rng(seed=42)
    planner1 = RRTStarPlanner(grid, (0, 0, 0), (15, 15, 15), rng=rng1)
    path1 = planner1.search(max_iters=1000)

    rng2 = np.random.default_rng(seed=42)
    planner2 = RRTStarPlanner(grid, (0, 0, 0), (15, 15, 15), rng=rng2)
    path2 = planner2.search(max_iters=1000)

    assert path1 == path2