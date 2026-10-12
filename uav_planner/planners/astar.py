import heapq
import math
import numpy as np
import time
import tracemalloc


def path_smoothness(path):
    if len(path) < 3:
        return 0
    changes = 0
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i])   - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        cos_a = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        )
        if np.arccos(np.clip(cos_a, -1, 1)) > np.pi / 4:
            changes += 1
    return changes

STEP_COSTS = {}
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == dy == dz == 0:
                continue
            nonzero = sum(c != 0 for c in (dx, dy, dz))
            STEP_COSTS[(dx, dy, dz)] = {1: 1.0, 2: math.sqrt(2), 3: math.sqrt(3)}[nonzero]

def octile_3d(a, b):                                                # heuristic (h(n))
    dx, dy, dz = abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2])

    s = sorted([dx, dy, dz])  # s[0]=min, s[1]=mid, s[2]=max

    return (
        s[2]
        + (math.sqrt(2) - 1) * s[1]
        + (math.sqrt(3) - math.sqrt(2)) * s[0]
    )

class AStarPlanner:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.metrics = {}

    def _neighbors(self, node):
        x, y, z = node
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    n = (x+dx, y+dy, z+dz)
                    if self.grid.is_free(*n):
                        yield n, STEP_COSTS[(dx, dy, dz)]

    def _finalize_metrics(self, t0, nodes_expanded, path):
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if path:
            path_length = sum(
                STEP_COSTS[(b[0]-a[0], b[1]-a[1], b[2]-a[2])]
                for a, b in zip(path, path[1:])
            )
        else:
            path_length = None

        self.metrics = {
            "nodes_expanded": nodes_expanded,
            "time_sec": elapsed,
            "path_length": path_length,
            "peak_memory_bytes": peak,
            "path_smoothness": path_smoothness(path) if path else None,
        }

    
    def search(self) -> list | None:
        start, goal = self.start, self.goal

        tracemalloc.start()
        t0 = time.perf_counter()

        open_heap = [(octile_3d(start, goal), 0.0, start)]
        g_score = {start: 0.0}
        came_from = {}
        closed = set()
        nodes_expanded = 0

        while open_heap:
            f, g, node = heapq.heappop(open_heap)

            if node in closed:
                continue
            if node == goal:
                path = self._reconstruct_path(came_from, node)
                self._finalize_metrics(t0, nodes_expanded, path)
                return path

            closed.add(node)
            nodes_expanded += 1

            for nbr, step_cost in self._neighbors(node):
                if nbr in closed:
                    continue
                tentative_g = g + step_cost
                if tentative_g < g_score.get(nbr, math.inf):
                    g_score[nbr] = tentative_g
                    came_from[nbr] = node
                    f_nbr = tentative_g + octile_3d(nbr, goal)
                    heapq.heappush(open_heap, (f_nbr, tentative_g, nbr))

        self._finalize_metrics(t0, nodes_expanded, None)
        return None
    
    @staticmethod
    def _reconstruct_path(came_from, node):
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path


                        
   