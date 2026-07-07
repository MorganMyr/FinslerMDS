"""Benchmark Soft Bellman-Ford on the shared branching/Swiss-roll datasets.

The parameter grid is a Cartesian product. Keep singleton tuples for parameters
that should remain fixed in a given campaign. Histories contain all-pairs hard
geodesic stress against optimization time, excluding stress-evaluation time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import fit_finsler_mds  # noqa: E402
from scripts import benchmark_branching_path_frozen as shared  # noqa: E402


DATASETS = ("branching",)
N_SAMPLES = 1000
SEEDS = (42,)

LANDMARK_SAMPLINGS = ("farthest",)
LANDMARK_FRACTIONS = (0.10, 0.05)
TARGETS_PER_LANDMARK_FRACTION = 0.35
LOCAL_PAIR_MODE = "direct"
LOCAL_GLOBAL_REWEIGHTING = "energy"
LOCAL_WEIGHT = 1.0

BETA_VALUES = (20,)
N_RELAXATIONS_VALUES = (32,)
SOURCE_BATCH_SIZE_VALUES = (128,)
MAX_ITER = 15
N_GRAPH_UPDATES = 25 
LOG_FREQUENCY = 1

OVERWRITE = False
RUN_BENCHMARKS = True
MAKE_PLOTS = False


@dataclass(frozen=True)
class SoftBFConfig:
    beta: float
    n_relaxations: int | None
    source_batch_size: int | None


def main():
    summary_rows = []
    for dataset_name in DATASETS:
        raw_dir, csv_dir, fig_dir = output_dirs(dataset_name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        dataset = shared.load_benchmark_dataset(
            dataset_name,
            raw_dir,
            n_samples=N_SAMPLES,
        )

        if RUN_BENCHMARKS:
            for seed, sampling, fraction, config in campaign_grid():
                summary_rows.append(
                    run_benchmark(
                        dataset,
                        seed=seed,
                        landmark_sampling=sampling,
                        landmark_fraction=fraction,
                        config=config,
                        csv_dir=csv_dir,
                    )
                )
        if MAKE_PLOTS:
            plot_dataset_curves(dataset_name, csv_dir=csv_dir, fig_dir=fig_dir)

    if summary_rows:
        shared.write_summary(shared.BENCHMARK_ROOT / "soft_bf_summary.csv", summary_rows)
    print(f"Soft-BF benchmark files are in {shared.BENCHMARK_ROOT}")


def campaign_grid():
    configs = (
        SoftBFConfig(float(beta), relaxations, batch_size)
        for beta, relaxations, batch_size in product(
            BETA_VALUES,
            N_RELAXATIONS_VALUES,
            SOURCE_BATCH_SIZE_VALUES,
        )
    )
    return product(SEEDS, LANDMARK_SAMPLINGS, LANDMARK_FRACTIONS, tuple(configs))


def output_dirs(dataset_name):
    raw_dir, _, shared_fig_dir = shared.dataset_dirs(dataset_name)
    benchmark_dir = shared_fig_dir.parent
    return (
        raw_dir,
        benchmark_dir / "csv" / "soft_bf",
        shared_fig_dir / "soft_bf",
    )


def run_benchmark(
        dataset,
        *,
        seed,
        landmark_sampling,
        landmark_fraction,
        config,
        csv_dir,
):
    options = benchmark_options(
        dataset,
        seed=seed,
        landmark_sampling=landmark_sampling,
        landmark_fraction=landmark_fraction,
        config=config,
    )
    path = csv_dir / benchmark_filename(
        seed=seed,
        landmark_sampling=landmark_sampling,
        landmark_fraction=landmark_fraction,
        config=config,
    )
    if path.exists() and not OVERWRITE:
        print(f"Skipping existing benchmark: {path}")
        return read_last_summary_row(path)

    print(
        f"Running Soft-BF {dataset.name}: beta={config.beta:g}, "
        f"relax={value_token(config.n_relaxations)}, "
        f"batch={value_token(config.source_batch_size)}, "
        f"landmarks={shared.percent_code(options['n_global_landmarks'], N_SAMPLES)}, "
        f"targets={shared.percent_code(options['max_global_targets_per_source'], N_SAMPLES)}, "
        f"inner={MAX_ITER}, graph_updates={N_GRAPH_UPDATES}, seed={seed}"
    )
    wall_start = perf_counter()
    result = fit_finsler_mds(
        dataset.D,
        metric=dataset.metric,
        optimizer="soft_bellman_ford",
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
        dataset=dataset,
        seed=seed,
        landmark_sampling=landmark_sampling,
        landmark_fraction=landmark_fraction,
        config=config,
        options=options,
        wall_time=wall_time,
    )
    return read_last_summary_row(path)


def benchmark_options(dataset, *, seed, landmark_sampling, landmark_fraction, config):
    solver = shared.branching.PATH_FROZEN_OPTIONS
    return {
        "graph_neighbors": dataset.graph_neighbors,
        "beta": config.beta,
        "n_relaxations": config.n_relaxations,
        "max_iter": MAX_ITER,
        "n_graph_updates": N_GRAPH_UPDATES,
        "log_frequency": LOG_FREQUENCY,
        "record_history": True,
        "verbose": 0,
        "random_state": seed,
        "mask_random_state": seed,
        "target_random_state": seed + 10_000,
        "n_global_landmarks": shared.count_from_fraction(N_SAMPLES, landmark_fraction),
        "random_landmark_fraction": shared.landmark_random_fraction(landmark_sampling),
        "max_global_targets_per_source": shared.count_from_fraction(
            N_SAMPLES,
            TARGETS_PER_LANDMARK_FRACTION,
        ),
        "n_local_neighbors": dataset.n_local_pairs,
        "local_pair_mode": LOCAL_PAIR_MODE,
        "local_global_reweighting": LOCAL_GLOBAL_REWEIGHTING,
        "local_weight": LOCAL_WEIGHT,
        "source_batch_size": config.source_batch_size,
        "on_unreachable": "raise",
        "device": solver["device"],
        "method": solver["method"],
        "optimizer_options": dict(solver["optimizer_options"]),
        "eps": solver["eps"],
    }


def benchmark_filename(*, seed, landmark_sampling, landmark_fraction, config):
    landmarks = shared.count_from_fraction(N_SAMPLES, landmark_fraction)
    targets = shared.count_from_fraction(N_SAMPLES, TARGETS_PER_LANDMARK_FRACTION)
    return (
        f"n{N_SAMPLES}_lm{shared.sampling_code(landmark_sampling)}"
        f"{shared.percent_code(landmarks, N_SAMPLES)}_"
        f"targ{shared.percent_code(targets, N_SAMPLES)}_"
        f"loc{local_mode_code()}_ii{MAX_ITER}_gu{N_GRAPH_UPDATES}_"
        f"b{shared.branching.format_float(config.beta)}_"
        f"rel{value_token(config.n_relaxations)}_"
        f"sb{value_token(config.source_batch_size)}_s{seed}.csv"
    )


def local_mode_code():
    return {"direct": "dir", "geodesic": "geo"}[LOCAL_PAIR_MODE]


def value_token(value):
    return "all" if value is None else str(int(value))


def write_history_csv(
        path,
        history,
        *,
        dataset,
        seed,
        landmark_sampling,
        landmark_fraction,
        config,
        options,
        wall_time,
):
    fixed = {
        "dataset": dataset.name,
        "n_samples": N_SAMPLES,
        "seed": seed,
        "landmark_sampling": landmark_sampling,
        "landmark_fraction": landmark_fraction,
        "beta": config.beta,
        "n_relaxations": config.n_relaxations,
        "source_batch_size": config.source_batch_size,
        "max_iter": MAX_ITER,
        "n_graph_updates_total": N_GRAPH_UPDATES,
        "log_frequency": LOG_FREQUENCY,
        "graph_neighbors": dataset.graph_neighbors,
        "n_landmark": options["n_global_landmarks"],
        "targets_per_landmark": options["max_global_targets_per_source"],
        "n_local_pairs": dataset.n_local_pairs,
        "local_pair_mode": LOCAL_PAIR_MODE,
        "local_global_reweighting": LOCAL_GLOBAL_REWEIGHTING,
        "local_weight": LOCAL_WEIGHT,
        "wall_time": wall_time,
    }
    history_keys = [
        "graph_update",
        "elapsed",
        "masked_stress",
        "full_geodesic_stress",
        "normalized_full_geodesic_stress",
        "nit",
        "nfev",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fixed, *history_keys])
        writer.writeheader()
        for row in history:
            writer.writerow({**fixed, **{key: row[key] for key in history_keys}})


def read_last_summary_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"path": str(path)}
    final = rows[-1]
    return {
        "path": str(path),
        "dataset": final["dataset"],
        "n_samples": final["n_samples"],
        "seed": final["seed"],
        "landmark_sampling": final["landmark_sampling"],
        "landmark_fraction": final["landmark_fraction"],
        "beta": final["beta"],
        "n_relaxations": final["n_relaxations"],
        "source_batch_size": final["source_batch_size"],
        "wall_time": final["wall_time"],
        "final_graph_update": final["graph_update"],
        "final_elapsed": final["elapsed"],
        "final_full_geodesic_stress": final["full_geodesic_stress"],
        "final_normalized_full_geodesic_stress": final[
            "normalized_full_geodesic_stress"
        ],
        "final_masked_stress": final["masked_stress"],
    }


def plot_dataset_curves(dataset_name, *, csv_dir, fig_dir):
    curves = load_average_curves(csv_dir)
    if not curves:
        print(f"No Soft-BF curves to plot for {dataset_name}: {csv_dir}")
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(curves)))
    for color, curve in zip(colors, curves):
        ax.plot(
            curve["elapsed"],
            curve["stress"],
            color=color,
            linewidth=1.8,
            label=curve_label(curve),
        )
    ax.set_xlabel("Optimization time excluding full-stress evaluations (s)")
    ax.set_ylabel("Full hard geodesic stress")
    ax.set_yscale("log")
    ax.set_title(f"Soft-BF: {dataset_name}, n={N_SAMPLES}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(fig_dir / f"soft_bf_{dataset_name}_n{N_SAMPLES}.{suffix}")
    plt.close(fig)


def load_average_curves(csv_dir):
    curves = []
    for sampling, fraction, config in product(
            LANDMARK_SAMPLINGS,
            LANDMARK_FRACTIONS,
            tuple(parameter_configs()),
    ):
        seed_curves = []
        for seed in SEEDS:
            path = csv_dir / benchmark_filename(
                seed=seed,
                landmark_sampling=sampling,
                landmark_fraction=fraction,
                config=config,
            )
            if path.exists():
                seed_curves.append(load_curve(path))
        if seed_curves:
            curves.append(average_seed_curves(seed_curves))
    return curves


def parameter_configs():
    return (
        SoftBFConfig(float(beta), relaxations, batch_size)
        for beta, relaxations, batch_size in product(
            BETA_VALUES,
            N_RELAXATIONS_VALUES,
            SOURCE_BATCH_SIZE_VALUES,
        )
    )


def load_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty benchmark CSV: {path}")
    first = rows[0]
    return {
        "beta": float(first["beta"]),
        "n_relaxations": parse_optional_int(first["n_relaxations"]),
        "source_batch_size": parse_optional_int(first["source_batch_size"]),
        "sampling": first["landmark_sampling"],
        "fraction": float(first["landmark_fraction"]),
        "graph_update": np.array([int(row["graph_update"]) for row in rows]),
        "elapsed": np.array([float(row["elapsed"]) for row in rows]),
        "stress": np.array([float(row["full_geodesic_stress"]) for row in rows]),
    }


def average_seed_curves(curves):
    common_updates = set(curves[0]["graph_update"])
    for curve in curves[1:]:
        common_updates &= set(curve["graph_update"])
    updates = np.array(sorted(common_updates), dtype=int)
    elapsed = []
    stress = []
    for curve in curves:
        index = {update: i for i, update in enumerate(curve["graph_update"])}
        positions = [index[int(update)] for update in updates]
        elapsed.append(curve["elapsed"][positions])
        stress.append(curve["stress"][positions])
    return {
        **{key: curves[0][key] for key in (
            "beta",
            "n_relaxations",
            "source_batch_size",
            "sampling",
            "fraction",
        )},
        "graph_update": updates,
        "elapsed": np.mean(elapsed, axis=0),
        "stress": np.mean(stress, axis=0),
        "n_seeds": len(curves),
    }


def curve_label(curve):
    return (
        f"beta={curve['beta']:g}, rel={value_token(curve['n_relaxations'])}, "
        f"batch={value_token(curve['source_batch_size'])}, "
        f"land={int(round(100 * curve['fraction']))}% "
        f"({curve['sampling']}, n={curve['n_seeds']})"
    )


def parse_optional_int(value):
    return None if value in ("", "None") else int(value)


if __name__ == "__main__":
    main()
