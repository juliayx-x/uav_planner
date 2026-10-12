import time
import tracemalloc
import math
import numpy as np
import pandas as pd

from uav_planner.voxel_grid import VoxelGrid
from uav_planner.planners.astar import AStarPlanner
from uav_planner.planners.rrt_star import RRTStarPlanner, dist_3d
from uav_planner.paths import RESULTS_DIR
from tqdm import tqdm

def run_condition(condition, n_maps=10, n_trials_per_map=50, base_seed=0):
    rows = []
    algo = condition["algo"]
    env = condition["env"]
    density = condition["density"]

    total = n_maps * n_trials_per_map
    with tqdm(total=total, desc=f"{algo}/{env}/{density}") as pbar:
        for map_id in range(n_maps):
            map_seed = base_seed + map_id
            grid = generate_map(env, density, map_id, seed=map_seed)
            start, goal = _pick_start_goal(grid, seed=map_seed)

            for trial_id in range(n_trials_per_map):
                trial_seed = map_seed * 1000 + trial_id
                row = run_single_trial(
                    algo, grid, start, goal, env, density,
                    map_id, trial_id, seed=trial_seed
                )
                rows.append(row)
                pbar.update(1)

    return pd.DataFrame(rows, columns=COLUMNS)

CONDITIONS = [
    {"algo": algo, "env": env, "density": density}
    for algo in ["astar", "rrt_star"]
    for env in ["random", "canyon", "clustered", "maze"]
    for density in ["low", "medium", "high"]
]

COLUMNS = [
    "algorithm", "env_class", "density", "map_id", "trial_id",
    "path_length", "time_ms", "nodes_explored", "smoothness",
    "success", "memory_kb", "altitude_utilisation",
]

def generate_map(env_class, density, map_id, sx=50, sy=50, sz=20, seed=None):
    """Generate one VoxelGrid instance for the given env_class/density combo.

    env_class controls obstacle *pattern* (random scatter, canyon walls,
    clustered blobs, maze corridors). density controls obstacle *fraction*.
    """
    rng = np.random.default_rng(seed)
    grid = VoxelGrid(sx=sx, sy=sy, sz=sz)

    density_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
    p = density_map[density]

    if env_class == "random":
        mask = rng.random((sx, sy, sz)) < p
        for (x, y, z) in np.argwhere(mask):
            grid.set_obstacle(x, y, z)
    else:
        # TODO: implement canyon / clustered / maze generators
        raise NotImplementedError(f"env_class '{env_class}' not yet implemented")

    return grid

def run_single_trial(algo, grid, start, goal, env_class, density, map_id, trial_id, seed=None):
    tracemalloc.start()
    t0 = time.perf_counter()

    if algo == "astar":
        planner = AStarPlanner(grid, start, goal)
        path = planner.search()
        nodes_explored = planner.metrics.get("nodes_expanded")
    elif algo == "rrt_star":
        rng = np.random.default_rng(seed)
        planner = RRTStarPlanner(grid, start, goal, rng=rng)
        path = planner.search(max_iters=3000)
        nodes_explored = len(planner.nodes)
    else:
        raise ValueError(f"unknown algo '{algo}'")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    success = path is not None
    path_length = _compute_path_length(path) if success else None
    altitude_utilisation_val = altitude_utilisation(path, grid) if success else None

    return {
        "algorithm": algo,
        "env_class": env_class,
        "density": density,
        "map_id": map_id,
        "trial_id": trial_id,
        "path_length": path_length,
        "time_ms": elapsed_ms,
        "nodes_explored": nodes_explored,
        "smoothness": smoothness,
        "success": success,
        "memory_kb": peak_bytes / 1024,
        "altitude_utilisation": altitude_utilisation,
    }

def _compute_path_length(path):
    return sum(dist_3d(path[i], path[i+1]) for i in range(len(path)-1))


def path_smoothness(path):
    if len(path) < 3:
        return 0

    changes = 0
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i])   - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if np.arccos(np.clip(cos_a, -1, 1)) > np.pi/4:
            changes += 1
    return changes


def altitude_utilisation(path, grid):
    z_threshold = grid.shape[2] // 2
    if not path:
        return 0.0
    above = sum(1 for p in path if p[2] >= z_threshold)
    return above / len(path)


def _pick_start_goal(grid, seed=None, min_dist=None):
    rng = np.random.default_rng(seed)
    sx, sy, sz = grid.shape
    min_dist = min_dist or (sx * 0.6)  # require a reasonably long path

    while True:
        start = tuple(int(rng.integers(0, s)) for s in (sx, sy, sz))
        goal = tuple(int(rng.integers(0, s)) for s in (sx, sy, sz))
        if not grid.is_free(*start) or not grid.is_free(*goal):
            continue
        if dist_3d(start, goal) >= min_dist:
            return start, goal

RESULTS_PATH = RESULTS_DIR / "results.csv"


def append_condition_results(df):
    file_exists = RESULTS_PATH.exists()
    df.to_csv(
        RESULTS_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )

def run_all_conditions(conditions=CONDITIONS, n_maps=10, n_trials_per_map=50):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, condition in enumerate(conditions):
        print(f"[{i+1}/{len(conditions)}] running {condition} ...")
        try:
            df = run_condition(condition, n_maps=n_maps, n_trials_per_map=n_trials_per_map)
        except NotImplementedError as e:
            print(f"  skipped: {e}")
            continue

        append_condition_results(df)
        print(f"  wrote {len(df)} rows")

    print("done.")


if __name__ == "__main__":
    dry_run_condition = {"algo": "astar", "env": "random", "density": "low"}

    t0 = time.perf_counter()
    df = run_condition(dry_run_condition, n_maps=10, n_trials_per_map=50)
    elapsed = time.perf_counter() - t0

    print(f"\ndry run took {elapsed:.1f}s for 1 condition (500 trials)")
    print(f"estimated total for 24 conditions: {elapsed * 24 / 60:.1f} minutes")

    append_condition_results(df)