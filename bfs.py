from collections import deque
import numpy as np

def path_exists(grid, start, goal):
    shape = grid.shape
    if grid[start] != 0 or grid[goal] != 0:
        return False 

    visited = np.zeros(shape, dtype=bool)
    visited[start] = True
    q = deque([start])
    neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]        # no diagonal steps

    while q:
            x, y, z = q.popleft()
            if (x, y, z) == goal:
                return True
            for dx, dy, dz in neighbors:
                nx, ny, nz = x+dx, y+dy, z+dz
                if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]
                        and not visited[nx, ny, nz] and grid[nx, ny, nz] == 0):
                    visited[nx, ny, nz] = True
                    q.append((nx, ny, nz))
    return False

