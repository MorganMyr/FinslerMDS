"""Benchmark path-frozen pair-selection heuristics on branching datasets.

The script reuses the cached branching geodesic datasets and contracted init
from ``main_branching.py``. It records the verbose>=2 path-frozen history:
logged outer iteration, optimization elapsed time, masked stress, full
geodesic stress, and normalized full geodesic stress.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds
from scripts import main_branching as branching


N_SAMPLES_LIST = [1000]
SEEDS = [branching.SEED]
N_LANDMARK_FRACTION = 0.20
TARGETS_PER_LANDMARK_FRACTION = 0.35

SCHEDULES = [
    {"inner_iter": 5, "outer_iter": 100, "log_frequency": 2},
    {"inner_iter": 20, "outer_iter": 35, "log_frequency": 3},
    {"inner_iter": 50, "outer_iter": 15, "log_frequency": 1},
]

ACTIVE_STRATEGIES = [
    "all_pair",
    "landmark_all_targets",
]

OVERWRITE = True


@dataclass(frozen=True)
class Strategy:
    name: str
    n_landmark_fraction: float | None = None
    targets_per_landmark_fraction: float | None = None
    n_local_pairs: int | None = None
    local_pair_mode: str = "direct"


STRATEGIES = {
    "all_pair": Strategy("all_pair"),
    "landmark_all_targets": Strategy(
        "landmark_all_targets",
        n_landmark_fraction=N_LANDMARK_FRACTION,
    ),
    "landmark_limited_targets": Strategy(
        "landmark_limited_targets",
        n_landmark_fraction=N_LANDMARK_FRACTION,
        targets_per_landmark_fraction=TARGETS_PER_LANDMARK_FRACTION,
    ),
    "landmark_limited_targets_local_geodesic": Strategy(
        "landmark_limited_targets_local_geodesic",
        n_landmark_fraction=N_LANDMARK_FRACTION,
        targets_per_landmark_fraction=TARGETS_PER_LANDMARK_FRACTION,
        n_local_pairs=branching.PATH_FROZEN_OPTIONS["n_local_pairs"],
        local_pair_mode="geodesic",
    ),
    "landmark_limited_targets_local_direct": Strategy(
        "landmark_limited_targets_local_direct",
        n_landmark_fraction=N_LANDMARK_FRACTION,
        targets_per_landmark_fraction=TARGETS_PER_LANDMARK_FRACTION,
        n_local_pairs=branching.PATH_FROZEN_OPTIONS["n_local_pairs"],
        local_pair_mode="direct",
    ),
}


def main():
    dir_res = SCRIPT_DIR / "res" / "branching_geodesic"
    dir_raw = dir_res / "raw"
    dir_bench = dir_res / "benchmarks"
    dir_raw.mkdir(parents=True, exist_ok=True)
    dir_bench.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for n_samples in N_SAMPLES_LIST:
        dataset, init = load_dataset_and_init(dir_raw, n_samples)
        for seed in SEEDS:
            for strategy_name in ACTIVE_STRATEGIES:
                strategy = STRATEGIES[strategy_name]
                for schedule in SCHEDULES:
                    run_row = run_benchmark(
                        dataset,
                        init,
                        n_samples=n_samples,
                        seed=seed,
                        strategy=strategy,
                        schedule=schedule,
                        dir_bench=dir_bench,
                    )
                    summary_rows.append(run_row)

    write_summary(dir_bench / "summary.csv", summary_rows)
    print(f"Saved benchmark results in {dir_bench}")


def load_dataset_and_init(dir_raw, n_samples):
    stem = (
        f"branching_n{n_samples}_k{branching.GRAPH_NEIGHBORS}_"
        f"w{branching.format_float(branching.CORRIDOR_WIDTH)}_s{branching.SEED}"
    )
    dataset = branching.load_or_create_dataset(
        dir_raw / f"{stem}_dataset.npz",
        n_samples=n_samples,
        graph_neighbors=branching.GRAPH_NEIGHBORS,
        corridor_width=branching.CORRIDOR_WIDTH,
        seed=branching.SEED,
        force=False,
    )
    init = branching.load_or_create_contracted_init(
        dir_raw / f"{stem}_contracted_init.npz",
        dataset,
        force=False,
    )
    return dataset, init


def run_benchmark(dataset, init, *, n_samples, seed, strategy, schedule, dir_bench):
    options = benchmark_options(n_samples, seed, strategy, schedule)
    filename = (
        f"n{n_samples}_{strategy.name}_"
        f"inner{schedule['inner_iter']}_outer{schedule['outer_iter']}_seed{seed}.csv"
    )
    path = dir_bench / filename
    if path.exists() and not OVERWRITE:
        print(f"Skipping existing benchmark: {path}")
        return read_last_summary_row(path)

    print(
        f"Running n={n_samples}, strategy={strategy.name}, "
        f"inner={schedule['inner_iter']}, outer={schedule['outer_iter']}, seed={seed}"
    )
    wall_start = perf_counter()
    result = fit_finsler_mds(
        dataset.D,
        metric=RandersMetric(alpha=0.0),
        optimizer="path_frozen",
        n_components=branching.N_COMPONENTS,
        init=init,
        print_time=True,
        return_result=True,
        **options,
    )
    wall_time = perf_counter() - wall_start
    write_history_csv(
        path,
        result.history,
        n_samples=n_samples,
        seed=seed,
        strategy=strategy,
        schedule=schedule,
        options=options,
        wall_time=wall_time,
    )
    final = result.history[-1] if result.history else {}
    return {
        "path": str(path),
        "n_samples": n_samples,
        "seed": seed,
        "strategy": strategy.name,
        "inner_iter": schedule["inner_iter"],
        "outer_iter": schedule["outer_iter"],
        "log_frequency": schedule["log_frequency"],
        "wall_time": wall_time,
        "final_outer_iter": final.get("outer_iter", ""),
        "final_elapsed": final.get("elapsed", ""),
        "final_full_geodesic_stress": final.get("full_geodesic_stress", ""),
        "final_normalized_full_geodesic_stress": final.get("normalized_full_geodesic_stress", ""),
        "final_masked_stress": final.get("masked_stress", ""),
    }


def benchmark_options(n_samples, seed, strategy, schedule):
    options = dict(branching.PATH_FROZEN_OPTIONS)
    options.update(schedule)
    options["verbose"] = 0
    options["record_history"] = True
    options["random_state"] = seed
    options["mask_random_state"] = seed
    options["target_random_state"] = seed + 10_000
    options["n_landmark"] = count_from_fraction(
        n_samples,
        strategy.n_landmark_fraction,
        default=0,
    )
    options["targets_per_landmark"] = count_from_fraction(
        n_samples,
        strategy.targets_per_landmark_fraction,
    )
    options["n_local_pairs"] = strategy.n_local_pairs
    options["local_pair_mode"] = strategy.local_pair_mode
    if strategy.name == "all_pair":
        options["local_global_reweighting"] = "none"
        options["local_weight"] = 1.0
    return options


def count_from_fraction(n_samples, fraction, *, default=None):
    if fraction is None:
        return default
    return max(1, int(round(float(fraction) * n_samples)))


def write_history_csv(
        path,
        history,
        *,
        n_samples,
        seed,
        strategy,
        schedule,
        options,
        wall_time,
):
    fieldnames = [
        "n_samples",
        "seed",
        "strategy",
        "inner_iter",
        "outer_iter_total",
        "log_frequency",
        "graph_neighbors",
        "n_landmark",
        "targets_per_landmark",
        "n_local_pairs",
        "local_pair_mode",
        "local_global_reweighting",
        "local_weight",
        "wall_time",
        "outer_iter",
        "elapsed",
        "masked_stress",
        "full_geodesic_stress",
        "normalized_full_geodesic_stress",
        "nit",
        "nfev",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "n_samples": n_samples,
                    "seed": seed,
                    "strategy": strategy.name,
                    "inner_iter": schedule["inner_iter"],
                    "outer_iter_total": schedule["outer_iter"],
                    "log_frequency": schedule["log_frequency"],
                    "graph_neighbors": options["graph_neighbors"],
                    "n_landmark": options["n_landmark"],
                    "targets_per_landmark": options["targets_per_landmark"],
                    "n_local_pairs": options["n_local_pairs"],
                    "local_pair_mode": options["local_pair_mode"],
                    "local_global_reweighting": options["local_global_reweighting"],
                    "local_weight": options["local_weight"],
                    "wall_time": wall_time,
                    **row,
                }
            )


def read_last_summary_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"path": str(path)}
    last = rows[-1]
    return {
        "path": str(path),
        "n_samples": last["n_samples"],
        "seed": last["seed"],
        "strategy": last["strategy"],
        "inner_iter": last["inner_iter"],
        "outer_iter": last["outer_iter_total"],
        "log_frequency": last["log_frequency"],
        "wall_time": last["wall_time"],
        "final_outer_iter": last["outer_iter"],
        "final_elapsed": last["elapsed"],
        "final_full_geodesic_stress": last["full_geodesic_stress"],
        "final_normalized_full_geodesic_stress": last["normalized_full_geodesic_stress"],
        "final_masked_stress": last["masked_stress"],
    }


def write_summary(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
