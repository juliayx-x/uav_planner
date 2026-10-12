import numpy as np
from scipy.spatial import KDTree
import math


def dist_3d(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def bresenham_3d(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    xs = 1 if dx >= 0 else -1
    ys = 1 if dy >= 0 else -1
    zs = 1 if dz >= 0 else -1
    dx, dy, dz = abs(dx), abs(dy), abs(dz)

    points = [(x1, y1, z1)]
    x, y, z = x1, y1, z1

    if dx >= dy and dx >= dz:
        p1, p2 = 2*dy - dx, 2*dz - dx
        while x != x2:
            x += xs
            if p1 >= 0:
                y += ys
                p1 -= 2*dx
            if p2 >= 0:
                z += zs
                p2 -= 2*dx
            p1 += 2*dy
            p2 += 2*dz
            points.append((x, y, z))
    elif dy >= dx and dy >= dz:
        p1, p2 = 2*dx - dy, 2*dz - dy
        while y != y2:
            y += ys
            if p1 >= 0:
                x += xs
                p1 -= 2*dy
            if p2 >= 0:
                z += zs
                p2 -= 2*dy
            p1 += 2*dx
            p2 += 2*dz
            points.append((x, y, z))
    else:
        p1, p2 = 2*dy - dz, 2*dx - dz
        while z != z2:
            z += zs
            if p1 >= 0:
                y += ys
                p1 -= 2*dz
            if p2 >= 0:
                x += xs
                p2 -= 2*dz
            p1 += 2*dy
            p2 += 2*dx
            points.append((x, y, z))

    return points


def is_segment_free(grid, a, b):
    for (x, y, z) in bresenham_3d(a, b):
        if not grid.is_free(x, y, z):
            return False
    return True


class RRTStarPlanner:
    def __init__(self, grid, start, goal, rng=None,
                 goal_bias=0.1, kd_rebuild_interval=50):
        self.grid = grid
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.goal_bias = goal_bias
        self.kd_rebuild_interval = kd_rebuild_interval

        self.nodes = [{"pos": self.start, "parent": None, "cost": 0.0}]

        self._kd_tree = None
        self._kd_tree_size = 0

    def _sample(self):
        if self.rng.random() < self.goal_bias:
            return self.goal

        sx, sy, sz = self.grid.shape
        while True:
            x = int(self.rng.integers(0, sx))
            y = int(self.rng.integers(0, sy))
            z = int(self.rng.integers(0, sz))
            if self.grid.is_free(x, y, z):
                return (x, y, z)

    def _build_kd_tree(self):
        positions = [n["pos"] for n in self.nodes]
        self._kd_tree = KDTree(positions)
        self._kd_tree_size = len(self.nodes)

    def _ensure_kd_tree(self, force=False):
        stale_count = len(self.nodes) - self._kd_tree_size
        if force or self._kd_tree is None or stale_count >= self.kd_rebuild_interval:
            self._build_kd_tree()

    def _nearest(self, sample_pos):
        self._ensure_kd_tree()
        dist, idx = self._kd_tree.query(sample_pos)
        return idx, dist

    def _near(self, sample_pos, radius):
        self._ensure_kd_tree()
        return self._kd_tree.query_ball_point(sample_pos, r=radius)

    def _steer(self, from_pos, to_pos, step=3):
        from_pos = np.array(from_pos, dtype=float)
        to_pos = np.array(to_pos, dtype=float)

        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)

        if dist <= step:
            new_pos = to_pos
        else:
            new_pos = from_pos + (direction / dist) * step

        return tuple(int(round(c)) for c in new_pos)

    def _extend(self, nearest_idx, sample_pos, step=3):
        nearest_pos = self.nodes[nearest_idx]["pos"]
        new_pos = self._steer(nearest_pos, sample_pos, step=step)

        if new_pos == nearest_pos:
            return None  # steering produced no movement — degenerate case

        if not self.grid.is_free(*new_pos):
            return None

        if not is_segment_free(self.grid, nearest_pos, new_pos):
            return None

        return new_pos

    def _choose_parent(self, new_pos, near_idxs, nearest_idx):
        best_idx = nearest_idx
        best_cost = self.nodes[nearest_idx]["cost"] + dist_3d(self.nodes[nearest_idx]["pos"], new_pos)

        for idx in near_idxs:
            candidate = self.nodes[idx]
            if not is_segment_free(self.grid, candidate["pos"], new_pos):
                continue  # can't actually connect through this node
            candidate_cost = candidate["cost"] + dist_3d(candidate["pos"], new_pos)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_idx = idx

        return best_idx, best_cost

    def _add_node(self, pos, parent_idx, cost):
        self.nodes.append({"pos": pos, "parent": parent_idx, "cost": cost})
        return len(self.nodes) - 1

    def _rewire(self, new_idx, near_idxs):
        new_node = self.nodes[new_idx]

        for near_idx in near_idxs:
            if near_idx == new_idx:
                continue

            near = self.nodes[near_idx]
            if not is_segment_free(self.grid, new_node["pos"], near["pos"]):
                continue

            new_cost = new_node["cost"] + dist_3d(new_node["pos"], near["pos"])
            if new_cost < near["cost"]:
                near["parent"] = new_idx
                near["cost"] = new_cost

    def _step(self, step_size=3, rewire_radius=5):
        sample = self._sample()
        nearest_idx, _ = self._nearest(sample)

        new_pos = self._extend(nearest_idx, sample, step=step_size)
        if new_pos is None:
            return None  # extension failed (blocked or degenerate) — caller tries again

        near_idxs = self._near(new_pos, radius=rewire_radius)

        parent_idx, cost = self._choose_parent(new_pos, near_idxs, nearest_idx)
        new_idx = self._add_node(new_pos, parent_idx, cost)

        self._rewire(new_idx, near_idxs)

        return new_idx

    def search(self, max_iters=2000, step_size=3, rewire_radius=10, goal_threshold=3.0):
        for _ in range(max_iters):
            self._step(step_size=step_size, rewire_radius=rewire_radius)

        return self._extract_path(goal_threshold=goal_threshold)

    def _extract_path(self, goal_threshold=3.0):
        best_idx = None
        best_dist = math.inf

        for idx, node in enumerate(self.nodes):
            d = dist_3d(node["pos"], self.goal)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx is None or best_dist > goal_threshold:
            return None

        return self._walk_to_root(best_idx)

    def _walk_to_root(self, idx):
        path = []
        while idx is not None:
            node = self.nodes[idx]
            path.append(node["pos"])
            idx = node["parent"]
        path.reverse()
        return path