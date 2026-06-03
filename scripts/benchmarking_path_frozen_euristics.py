"""Benchmark path-frozen pair-selection heuristics on toy datasets.

Set ``DATASET_NAME`` to choose between the branching geodesic dataset and the
asymmetric Swiss roll dataset. The script records path-frozen history: logged
outer iteration, optimization elapsed time, masked stress, full geodesic
stress, and normalized full geodesic stress.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds, utils
from scripts import main_branching as branching


DATASET_NAME = "branching"  # "branching" | "swiss_roll"
N_SAMPLES_LIST = [1000]
SEEDS = [42]
N_LANDMARK_FRACTION = 0.10
TARGETS_PER_LANDMARK_FRACTION = 0.35
LOCAL_GLOBAL_REWEIGHTING = "energy" # "count" or "energy"
LANDMARK_SAMPLING = "random"  # "random" or "farthest"
VERBOSE = 2

SCHEDULES = [
    {"inner_iter": 20, "outer_iter": 50, "log_frequency": 5},
]


ACTIVE_STRATEGIES = [
    "landmark_limited_targets",
]

OVERWRITE = True

BENCHMARK_ROOT = SCRIPT_DIR / "res" / "path_frozen_benchmarks"
DATASET_DIRS = {
    "branching": {
        "raw": "branching_raw",
        "benchmarks": "branching_benchmarks",
    },
    "swiss_roll": {
        "raw": "swiss_roll_raw",
        "benchmarks": "swiss_roll_benchmarks",
    },
}

SWISS_ROLL_K = 10
SWISS_ROLL_N_COMPONENTS = 3
SWISS_ROLL_NOISE_LEVEL = 0.0
SWISS_ROLL_ALPHA_MANIFOLD = 0.35
SWISS_ROLL_ALPHA_EMBEDDING = 0.5
SWISS_ROLL_LOCAL_PAIRS = 8


@dataclass(frozen=True)
class Strategy:
    name: str
    n_landmark_fraction: float | None = None
    targets_per_landmark_fraction: float | None = None
    n_local_pairs: int | str | None = None
    local_pair_mode: str = "direct"


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    D: np.ndarray
    init: np.ndarray
    metric: object
    n_components: int
    graph_neighbors: int
    default_n_local_pairs: int

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
        n_local_pairs="default",
        local_pair_mode="geodesic",
    ),
    "landmark_limited_targets_local_direct": Strategy(
        "landmark_limited_targets_local_direct",
        n_landmark_fraction=N_LANDMARK_FRACTION,
        targets_per_landmark_fraction=TARGETS_PER_LANDMARK_FRACTION,
        n_local_pairs="default",
        local_pair_mode="direct",
    ),
}

def main():
    dir_raw, dir_bench = dataset_dirs(DATASET_NAME)
    dir_csv = dir_bench / "csv"
    dir_raw.mkdir(parents=True, exist_ok=True)
    dir_csv.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for n_samples in N_SAMPLES_LIST:
        dataset = load_benchmark_dataset(DATASET_NAME, dir_raw, n_samples)
        for seed in SEEDS:
            for strategy_name in ACTIVE_STRATEGIES:
                strategy = STRATEGIES[strategy_name]
                for schedule in SCHEDULES:
                    run_row = run_benchmark(
                        dataset,
                        n_samples=n_samples,
                        seed=seed,
                        strategy=strategy,
                        schedule=schedule,
                        dir_csv=dir_csv,
                    )
                    summary_rows.append(run_row)

    write_summary(dir_csv / "summary.csv", summary_rows)
    print(f"Saved benchmark results in {dir_bench}")


def dataset_dirs(dataset_name):
    try:
        names = DATASET_DIRS[dataset_name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_DIRS))
        raise ValueError(f"Unknown DATASET_NAME={dataset_name!r}. Choose one of: {known}.") from exc
    return BENCHMARK_ROOT / names["raw"], BENCHMARK_ROOT / names["benchmarks"]


def load_benchmark_dataset(dataset_name, dir_raw, n_samples):
    if dataset_name == "branching":
        return load_branching_dataset(dir_raw, n_samples)
    if dataset_name == "swiss_roll":
        return load_swiss_roll_dataset(dir_raw, n_samples)
    raise RuntimeError(f"Unhandled dataset {dataset_name!r}.")


def load_branching_dataset(dir_raw, n_samples):
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
    return BenchmarkDataset(
        name="branching",
        D=dataset.D,
        init=init,
        metric=RandersMetric(alpha=0.0),
        n_components=branching.N_COMPONENTS,
        graph_neighbors=branching.PATH_FROZEN_OPTIONS["graph_neighbors"],
        default_n_local_pairs=branching.PATH_FROZEN_OPTIONS["n_local_pairs"],
    )


def load_swiss_roll_dataset(dir_raw, n_samples):
    stem = (
        f"swiss_n{n_samples}_k{SWISS_ROLL_K}_"
        f"ma{branching.format_float(SWISS_ROLL_ALPHA_MANIFOLD)}_"
        f"ea{branching.format_float(SWISS_ROLL_ALPHA_EMBEDDING)}_s{branching.SEED}"
    )
    path = dir_raw / f"{stem}.npz"
    if path.exists():
        print(f"Loading cached Swiss roll benchmark data: {path}")
        data = np.load(path)
        return swiss_roll_benchmark_dataset(data["D"], data["init"])

    print(f"Creating Swiss roll benchmark data: n={n_samples}, k={SWISS_ROLL_K}")
    rng = np.random.default_rng(branching.SEED)
    params = rng.random((n_samples, 2))
    x = np.empty((n_samples, 2), dtype=float)
    x[:, 0] = params[:, 0] * 3 * np.pi + 1.5 * np.pi
    x[:, 1] = params[:, 1] * 20

    X_noiseless = np.empty((n_samples, 3), dtype=float)
    X_noiseless[:, 0] = x[:, 0] * np.cos(x[:, 0])
    X_noiseless[:, 1] = x[:, 1]
    X_noiseless[:, 2] = x[:, 0] * np.sin(x[:, 0])
    X = X_noiseless + SWISS_ROLL_NOISE_LEVEL * rng.normal(size=X_noiseless.shape)

    isomap = utils.IsomapWithPreds(n_components=SWISS_ROLL_N_COMPONENTS, n_neighbors=SWISS_ROLL_K)
    init = isomap.fit_transform(X)

    tangent_x = np.cos(x[:, 0]) - x[:, 0] * np.sin(x[:, 0])
    tangent_y = np.zeros(n_samples)
    tangent_z = np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 0])
    randers_field = np.stack([tangent_x, tangent_y, tangent_z], axis=1)
    randers_field /= np.linalg.norm(randers_field, axis=1)[:, None]
    randers_field *= SWISS_ROLL_ALPHA_MANIFOLD
    D, _ = utils.compute_dist_matrix(
        X,
        n_neighbors=SWISS_ROLL_K,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=randers_field,
    )
    np.savez_compressed(
        path,
        X=X,
        X_noiseless=X_noiseless,
        init=init,
        D=D,
        randers_field=randers_field,
        x=x,
    )
    print(f"Saved Swiss roll benchmark data: {path}")
    return swiss_roll_benchmark_dataset(D, init)


def swiss_roll_benchmark_dataset(D, init):
    return BenchmarkDataset(
        name="swiss_roll",
        D=D,
        init=init,
        metric=RandersMetric(alpha=SWISS_ROLL_ALPHA_EMBEDDING),
        n_components=SWISS_ROLL_N_COMPONENTS,
        graph_neighbors=SWISS_ROLL_K,
        default_n_local_pairs=SWISS_ROLL_LOCAL_PAIRS,
    )


def run_benchmark(dataset, *, n_samples, seed, strategy, schedule, dir_csv):
    options = benchmark_options(n_samples, seed, strategy, schedule, dataset)
    path = dir_csv / benchmark_filename(n_samples, strategy.name, schedule, seed, options)
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
        metric=dataset.metric,
        optimizer="path_frozen",
        n_components=dataset.n_components,
        init=dataset.init,
        print_time=True,
        return_result=True,
        **options,
    )
    wall_time = perf_counter() - wall_start
    write_history_csv(
        path,
        result.history,
        dataset_name=dataset.name,
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
        "dataset": dataset.name,
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


def benchmark_options(n_samples, seed, strategy, schedule, dataset):
    options = dict(branching.PATH_FROZEN_OPTIONS)
    options.update(schedule)
    options["graph_neighbors"] = dataset.graph_neighbors
    options["verbose"] = VERBOSE
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
    options["n_local_pairs"] = (
        dataset.default_n_local_pairs
        if strategy.n_local_pairs == "default"
        else strategy.n_local_pairs
    )
    options["local_pair_mode"] = strategy.local_pair_mode
    options["local_global_reweighting"] = LOCAL_GLOBAL_REWEIGHTING
    options["landmark_sampling"] = LANDMARK_SAMPLING
    if strategy.name == "all_pair":
        options["local_global_reweighting"] = "none"
        options["local_weight"] = 1.0
    return options


def benchmark_filename(n_samples, strategy_name, schedule, seed, options):
    return (
        f"n{n_samples}_{strategy_code(strategy_name, n_samples, options)}_"
        f"ii{schedule['inner_iter']}_oi{schedule['outer_iter']}_s{seed}.csv"
    )


def strategy_code(strategy_name, n_samples, options):
    if strategy_name == "all_pair":
        return "AP"

    parts = [f"land{percent_code(options['n_landmark'], n_samples)}"]
    if options["targets_per_landmark"] is None:
        parts.append("targ_all")
    else:
        parts.append(f"targ{percent_code(options['targets_per_landmark'], n_samples)}")

    if options["n_local_pairs"]:
        local = "geo" if options["local_pair_mode"] == "geodesic" else "dir"
        reweighting = options["local_global_reweighting"]
        suffix = "count" if reweighting == "count" else ""
        parts.append(f"loc{local}{suffix}")
    return "_".join(parts)


def percent_code(count, total):
    percent = int(round(100 * float(count) / float(total)))
    return f"{percent}p"


def count_from_fraction(n_samples, fraction, *, default=None):
    if fraction is None:
        return default
    return max(1, int(round(float(fraction) * n_samples)))


def write_history_csv(
        path,
        history,
        *,
        dataset_name,
        n_samples,
        seed,
        strategy,
        schedule,
        options,
        wall_time,
):
    fieldnames = [
        "dataset",
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
                    "dataset": dataset_name,
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
        "dataset": last.get("dataset", ""),
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
