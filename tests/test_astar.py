import pytest
from uav_planner.planners.astar import AStarPlanner


class SimpleGrid:
    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.obstacles = obstacles or set()

    def is_free(self, x, y, z):
        if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size):
            return False
        return (x, y, z) not in self.obstacles


def test_finds_path_in_empty_grid():
    grid = SimpleGrid(size=10)
    planner = AStarPlanner(grid, (0, 0, 0), (5, 5, 5))
    path = planner.search()

    assert path is not None
    assert len(path) > 0


def test_returns_none_when_blocked():
    size = 5
    obstacles = set()
    gx, gy, gz = 4, 4, 4
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = gx+dx, gy+dy, gz+dz
                if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                    obstacles.add((nx, ny, nz))

    grid = SimpleGrid(size=size, obstacles=obstacles)
    planner = AStarPlanner(grid, (0, 0, 0), (gx, gy, gz))
    path = planner.search()

    assert path is None


def test_path_endpoints_correct():
    grid = SimpleGrid(size=10)
    start, goal = (1, 1, 1), (8, 8, 8)
    planner = AStarPlanner(grid, start, goal)
    path = planner.search()

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal


def test_path_avoids_obstacles():
    size = 10
    obstacles = {(x, 5, 5) for x in range(size)}
    grid = SimpleGrid(size=size, obstacles=obstacles)
    planner = AStarPlanner(grid, (0, 0, 0), (9, 9, 9))
    path = planner.search()

    assert path is not None
    for voxel in path:
        assert voxel not in obstacles