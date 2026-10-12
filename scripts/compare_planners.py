"""
Side-by-side comparison of AStarPlanner vs RRTStarPlanner across one map
from each environment class (Random, Canyon, Clustered, Maze).

Produces a 2x4 figure: each column is one environment, top row A*, bottom
row RRT*, all sharing the same view so shapes are easy to compare. Also
saves per-environment overlay plots (both paths in the same 3D axes).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from uav_planner.maps.random_map import RandomMapGenerator
from uav_planner.maps.canyon_map import CanyonMapGenerator
from uav_planner.maps.clustered_map import ClusteredMapGenerator
from uav_planner.maps.maze_map import MazeMapGenerator
from uav_planner.planners.astar import AStarPlanner
from uav_planner.planners.rrt_star import RRTStarPlanner
from uav_planner.paths import FIGURES_DIR

SEED = 42
SHAPE = (50, 50, 20)
START = (3, 3, 2)
GOAL = (47, 47, 18)


def build_environments():
    envs = {}

    random_vg = RandomMapGenerator(seed=SEED, density=0.15).generate(
        shape=SHAPE, start=START, goal=GOAL
    )
    envs["Random"] = (random_vg, START, GOAL)

    canyon_start, canyon_goal = (1, 1, 0), (SHAPE[0] - 2, SHAPE[1] - 2, 0)
    canyon_vg = CanyonMapGenerator(seed=SEED).generate(
        shape=SHAPE, start=canyon_start, goal=canyon_goal
    )
    envs["Canyon"] = (canyon_vg, canyon_start, canyon_goal)

    clustered_vg = ClusteredMapGenerator(seed=SEED).generate(
        shape=SHAPE, start=START, goal=GOAL
    )
    envs["Clustered"] = (clustered_vg, START, GOAL)

    maze_vg = MazeMapGenerator(seed=SEED).generate(shape=SHAPE)
    maze_start, maze_goal = (0, 0, 0), (48, 48, 18)
    envs["Maze"] = (maze_vg, maze_start, maze_goal)

    return envs


def run_astar(vg, start, goal):
    planner = AStarPlanner(vg, start, goal)
    t0 = time.perf_counter()
    path = planner.search()
    elapsed = time.perf_counter() - t0
    n_expanded = planner.metrics.get("nodes_expanded")
    length = planner.metrics.get("path_length")
    return path, elapsed, n_expanded, length


def run_rrtstar(vg, start, goal, seed=SEED, max_iters=3000, step_size=3,
                 rewire_radius=10, goal_threshold=3.0, goal_bias=0.1):
    rng = np.random.default_rng(seed)
    planner = RRTStarPlanner(vg, start, goal, rng=rng, goal_bias=goal_bias)
    t0 = time.perf_counter()
    path = planner.search(max_iters=max_iters, step_size=step_size,
                           rewire_radius=rewire_radius, goal_threshold=goal_threshold)
    elapsed = time.perf_counter() - t0
    n_nodes = len(planner.nodes)
    length = None
    if path:
        length = sum(
            np.linalg.norm(np.array(a) - np.array(b))
            for a, b in zip(path, path[1:])
        )
    return path, elapsed, n_nodes, length


def plot_obstacles(ax, grid, color="gray", alpha=0.12):
    obstacles = np.argwhere(grid == 1)
    if len(obstacles):
        ax.scatter(
            obstacles[:, 0], obstacles[:, 1], obstacles[:, 2],
            c=color, alpha=alpha, s=4, linewidths=0,
        )


def plot_path(ax, path, color, label, marker_size=40):
    if not path:
        return
    px, py, pz = zip(*path)
    ax.plot3D(px, py, pz, c=color, linewidth=2.2, label=label)
    ax.scatter(px, py, pz, c=color, s=8, alpha=0.6)


def main():
    envs = build_environments()
    results = {}

    # RRT* param overrides per environment. Maze has 1-voxel-wide axis-aligned
    # corridors, which is a poor fit for continuous straight-line steering, so
    # give it a much smaller step size and more iterations for a fair shot.
    rrt_params = {
        "Random":    dict(max_iters=3000,  step_size=3, rewire_radius=10, goal_threshold=3.0, goal_bias=0.1),
        "Canyon":    dict(max_iters=3000,  step_size=3, rewire_radius=10, goal_threshold=3.0, goal_bias=0.1),
        "Clustered": dict(max_iters=3000,  step_size=3, rewire_radius=10, goal_threshold=3.0, goal_bias=0.1),
        "Maze":      dict(max_iters=40000, step_size=1, rewire_radius=2,  goal_threshold=1.5, goal_bias=0.2),
    }

    print(f"{'Env':<10} {'Planner':<8} {'Success':<8} {'Time(s)':<9} {'NodesExp':<10} {'PathLen':<8}")
    for name, (vg, start, goal) in envs.items():
        a_path, a_time, a_nodes, a_len = run_astar(vg, start, goal)
        r_path, r_time, r_nodes, r_len = run_rrtstar(vg, start, goal, **rrt_params[name])
        results[name] = {
            "vg": vg, "start": start, "goal": goal,
            "astar": (a_path, a_time, a_nodes, a_len),
            "rrtstar": (r_path, r_time, r_nodes, r_len),
        }
        a_len_str = "n/a" if a_len is None else f"{a_len:.2f}"
        r_len_str = "n/a" if r_len is None else f"{r_len:.2f}"
        print(f"{name:<10} {'A*':<8} {str(a_path is not None):<8} {a_time:<9.4f} {a_nodes:<10} {a_len_str:<8}")
        print(f"{name:<10} {'RRT*':<8} {str(r_path is not None):<8} {r_time:<9.4f} {r_nodes:<10} {r_len_str:<8}")

    # 2 x 4 grid: rows = planner, cols = environment
    env_names = list(envs.keys())
    fig = plt.figure(figsize=(22, 11))

    for col, name in enumerate(env_names):
        r = results[name]
        vg, start, goal = r["vg"], r["start"], r["goal"]

        # Row 0: A*
        ax_a = fig.add_subplot(2, 4, col + 1, projection="3d")
        plot_obstacles(ax_a, vg.grid)
        ax_a.scatter(*start, c="green", s=70, zorder=5)
        ax_a.scatter(*goal, c="red", s=70, zorder=5)
        plot_path(ax_a, r["astar"][0], color="dodgerblue", label="A*")
        ax_a.set_title(f"{name} — A*\n{len(r['astar'][0]) if r['astar'][0] else 0} waypoints, "
                        f"{r['astar'][2]} nodes expanded")
        ax_a.set_xlabel("X"); ax_a.set_ylabel("Y"); ax_a.set_zlabel("Z")

        # Row 1: RRT*
        ax_r = fig.add_subplot(2, 4, col + 5, projection="3d")
        plot_obstacles(ax_r, vg.grid)
        ax_r.scatter(*start, c="green", s=70, zorder=5)
        ax_r.scatter(*goal, c="red", s=70, zorder=5)
        plot_path(ax_r, r["rrtstar"][0], color="darkorange", label="RRT*")
        rrt_path = r["rrtstar"][0]
        rrt_title = (f"{name} — RRT*\n{len(rrt_path)} waypoints, {r['rrtstar'][2]} tree nodes"
                     if rrt_path else
                     f"{name} — RRT*\nNO PATH FOUND ({r['rrtstar'][2]} tree nodes)")
        ax_r.set_title(rrt_title)
        ax_r.set_xlabel("X"); ax_r.set_ylabel("Y"); ax_r.set_zlabel("Z")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "astar_vs_rrtstar_grid.png", dpi=150)
    print(f"\nSaved {FIGURES_DIR / 'astar_vs_rrtstar_grid.png'}")

    # Overlay figure: both paths on the same axes, per environment
    fig2 = plt.figure(figsize=(22, 6))
    for col, name in enumerate(env_names):
        r = results[name]
        vg, start, goal = r["vg"], r["start"], r["goal"]
        ax = fig2.add_subplot(1, 4, col + 1, projection="3d")
        plot_obstacles(ax, vg.grid)
        ax.scatter(*start, c="green", s=70, zorder=5, label="Start")
        ax.scatter(*goal, c="red", s=70, zorder=5, label="Goal")
        plot_path(ax, r["astar"][0], color="dodgerblue", label="A*")
        plot_path(ax, r["rrtstar"][0], color="darkorange", label="RRT*")
        ax.set_title(name)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "astar_vs_rrtstar_overlay.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'astar_vs_rrtstar_overlay.png'}")


if __name__ == "__main__":
    main()