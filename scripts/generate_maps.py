import numpy as np
from uav_planner.maps.random_map import RandomMapGenerator
from uav_planner.maps.canyon_map import CanyonMapGenerator
from uav_planner.maps.clustered_map import ClusteredMapGenerator
from uav_planner.maps.maze_map import MazeMapGenerator
from uav_planner.paths import MAPS_DIR

START = (3, 3, 2)
GOAL  = (47, 47, 18)


def generate_and_save(name, generator, **generate_kwargs):
    print(f"\nGenerating {name} map...")
    vg = generator.generate(**generate_kwargs)
    print(f"  Obstacles : {vg.grid.sum()}")
    print(f"  Density   : {vg.grid.sum() / vg.grid.size:.4f}")

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    path = MAPS_DIR / f"{name}_seed{generator.seed}.npy"
    np.save(path, vg.grid)
    print(f"  Saved to  : {path}")
    return vg


def main():
    random_vg = generate_and_save("random", RandomMapGenerator(seed=42, density=0.15), start=START, goal=GOAL)
    canyon_vg    = generate_and_save("canyon",     CanyonMapGenerator(seed=42),    start=START, goal=GOAL)
    clustered_vg = generate_and_save("clustered",  ClusteredMapGenerator(seed=42), start=START, goal=GOAL)
    maze_vg      = generate_and_save("maze",       MazeMapGenerator(seed=42))

    # Reload check on the random map, as a sanity test that save/load round-trips correctly
    print("\nReloading random map to verify save/load integrity...")
    loaded = np.load(MAPS_DIR / "random_seed42.npy")
    identical = np.array_equal(random_vg.grid, loaded)
    print(f"  Identical : {identical}")
    print(f"  dtype     : {loaded.dtype}")
    print(f"  shape     : {loaded.shape}")

    # Visualise the random map with an example path
    print("\nVisualising random map...")
    example_path = [START, (10, 10, 5), (20, 20, 10), (35, 35, 15), GOAL]
    clustered_vg.visualise(start=START, goal=GOAL, path=example_path)

if __name__ == "__main__":
    main()