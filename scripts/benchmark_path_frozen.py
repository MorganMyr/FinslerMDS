"""Benchmark Path-Frozen convergence on branching and Swiss roll data.

The campaign compares landmark mixes and direct-MDS regularization weights. It
records the Path-Frozen history at the selected log frequency into CSV files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds, utils  # noqa: E402
from finsler_mds.utils.branching import load_branching_problem  # noqa: E402


@dataclass(frozen=True)
class LandmarkStrategy:
    name: str
    random_landmark_fraction: float


DATASETS = ("swiss_roll", "branching")
N_SAMPLES = 1000
SEEDS = (42, 43, 44, 45, 46)
LANDMARK_STRATEGIES = (
    LandmarkStrategy("farthest", 0.0),
)
DIRECT_STRESS_WEIGHTS = (0.0, 1.0, 0.5, 0.1)
LANDMARK_FRACTIONS = (0.10,)
TARGETS_PER_LANDMARK_FRACTION = 0.35
INNER_ITER = 50
OUTER_STEP_SIZE = 1.0
OUTER_STEP_SIZES = (OUTER_STEP_SIZE,)
OUTER_ITER_BY_LANDMARK_FRACTION = {
    0.10: 30,
}
LOG_FREQUENCY_BY_LANDMARK_FRACTION = {
    0.10: 2,
}

LOCAL_WEIGHT = 1.0
OVERWRITE = False
RUN_BENCHMARKS = True
MAKE_PLOTS = False
CSV_SUBDIR = "direct_stress"
SUMMARY_NAME = "direct_stress_summary.csv"

BENCHMARK_ROOT = SCRIPT_DIR / "res" / "path_frozen_benchmarks"
DATASET_DIRS = {
    "branching": {"raw": "branching_raw", "benchmarks": "branching_benchmarks"},
    "swiss_roll": {"raw": "swiss_roll_raw", "benchmarks": "swiss_roll_benchmarks"},
}

SWISS_ROLL_K = 10
SWISS_ROLL_N_COMPONENTS = 3
SWISS_ROLL_NOISE_LEVEL = 0.0
SWISS_ROLL_ALPHA_MANIFOLD = 0.35
SWISS_ROLL_ALPHA_EMBEDDING = 0.5
SWISS_ROLL_LOCAL_PAIRS = 8
DATASET_SEED = 42
BRANCHING_K = 10
BRANCHING_CORRIDOR_WIDTH = 0.11
BRANCHING_N_COMPONENTS = 2
BRANCHING_EMBEDDING_K = 20
BRANCHING_LOCAL_PAIRS = 20


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    D: np.ndarray
    init: np.ndarray
    metric: object
    n_components: int
    graph_neighbors: int
    n_local_pairs: int


def main():
    summary_rows = []
    for dataset_name in DATASETS:
        raw_dir, csv_dir, fig_dir = dataset_dirs(dataset_name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_benchmark_dataset(dataset_name, raw_dir)

        if RUN_BENCHMARKS:
            for seed in SEEDS:
                for strategy in LANDMARK_STRATEGIES:
                    for landmark_fraction in LANDMARK_FRACTIONS:
                        for outer_step_size in OUTER_STEP_SIZES:
                            for direct_stress_weight in DIRECT_STRESS_WEIGHTS:
                                summary_rows.append(
                                    run_benchmark(
                                        dataset,
                                        seed=seed,
                                        strategy=strategy,
                                        direct_stress_weight=direct_stress_weight,
                                        landmark_fraction=landmark_fraction,
                                        outer_step_size=outer_step_size,
                                        csv_dir=csv_dir,
                                    )
                                )
        if MAKE_PLOTS:
            plot_dataset_curves(dataset_name, csv_dir=csv_dir, fig_dir=fig_dir)

    if summary_rows:
        write_summary(BENCHMARK_ROOT / SUMMARY_NAME, summary_rows)
    print(f"Benchmark files are in {BENCHMARK_ROOT}")


def dataset_dirs(dataset_name):
    if dataset_name not in DATASET_DIRS:
        known = ", ".join(sorted(DATASET_DIRS))
        raise ValueError(f"Unknown dataset {dataset_name!r}. Choose one of: {known}.")
    dirs = DATASET_DIRS[dataset_name]
    bench_dir = BENCHMARK_ROOT / dirs["benchmarks"]
    return (
        BENCHMARK_ROOT / dirs["raw"],
        bench_dir / "csv" / CSV_SUBDIR,
        bench_dir / "figures",
    )


def load_benchmark_dataset(dataset_name, raw_dir, *, n_samples=None):
    n_samples = N_SAMPLES if n_samples is None else int(n_samples)
    if dataset_name == "branching":
        return load_branching_dataset(raw_dir, n_samples=n_samples)
    if dataset_name == "swiss_roll":
        return load_swiss_roll_dataset(raw_dir, n_samples=n_samples)
    raise RuntimeError(f"Unhandled dataset {dataset_name!r}.")


def load_branching_dataset(raw_dir, *, n_samples):
    problem = load_branching_problem(
        raw_dir,
        n_samples=n_samples,
        graph_neighbors=BRANCHING_K,
        corridor_width=BRANCHING_CORRIDOR_WIDTH,
        seed=DATASET_SEED,
        force=False,
    )
    return BenchmarkDataset(
        name="branching",
        D=problem.dissimilarities,
        init=problem.init,
        metric=RandersMetric(alpha=0.0),
        n_components=BRANCHING_N_COMPONENTS,
        graph_neighbors=BRANCHING_EMBEDDING_K,
        n_local_pairs=BRANCHING_LOCAL_PAIRS,
    )


def load_swiss_roll_dataset(raw_dir, *, n_samples):
    stem = (
        f"swiss_n{n_samples}_k{SWISS_ROLL_K}_"
        f"ma{float_code(SWISS_ROLL_ALPHA_MANIFOLD)}_"
        f"ea{float_code(SWISS_ROLL_ALPHA_EMBEDDING)}_s{DATASET_SEED}"
    )
    path = raw_dir / f"{stem}.npz"
    if path.exists():
        print(f"Loading cached Swiss roll benchmark data: {path}")
        data = np.load(path)
        return swiss_roll_benchmark_dataset(data["D"], data["init"])

    print(f"Creating Swiss roll benchmark data: n={n_samples}, k={SWISS_ROLL_K}")
    rng = np.random.default_rng(DATASET_SEED)
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
        n_local_pairs=SWISS_ROLL_LOCAL_PAIRS,
    )


def run_benchmark(dataset, *, seed, strategy, direct_stress_weight, landmark_fraction, outer_step_size, csv_dir):
    options = benchmark_options(
        dataset,
        seed,
        strategy,
        direct_stress_weight,
        landmark_fraction,
        outer_step_size,
    )
    n_samples = len(dataset.D)
    path = csv_dir / benchmark_filename(
        seed,
        strategy,
        direct_stress_weight,
        landmark_fraction,
        outer_step_size,
        n_samples=n_samples,
    )
    if path.exists() and not OVERWRITE:
        print(f"Skipping existing benchmark: {path}")
        return read_last_summary_row(path)

    print(
        f"Running {dataset.name}: {strategy.name}, "
        f"random_landmark_fraction={strategy.random_landmark_fraction:g}, "
        f"landmarks={percent_code(options['n_landmark'], n_samples)}, "
        f"targets={percent_code(options['targets_per_landmark'], n_samples)}, "
        f"inner={INNER_ITER}, outer={options['outer_iter']}, "
        f"direct_stress={direct_stress_code(direct_stress_weight)}, "
        f"outer_step={outer_step_size:g}, seed={seed}"
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
        dataset=dataset,
        seed=seed,
        landmark_strategy=strategy,
        direct_stress_weight=direct_stress_weight,
        landmark_fraction=landmark_fraction,
        outer_step_size=outer_step_size,
        options=options,
        wall_time=wall_time,
    )
    return read_last_summary_row(path)


def benchmark_options(dataset, seed, strategy, direct_stress_weight, landmark_fraction, outer_step_size):
    outer_iter = OUTER_ITER_BY_LANDMARK_FRACTION[landmark_fraction]
    log_frequency = LOG_FREQUENCY_BY_LANDMARK_FRACTION[landmark_fraction]
    n_samples = len(dataset.D)
    return dict(
        graph_neighbors=dataset.graph_neighbors,
        outer_iter=outer_iter,
        inner_iter=INNER_ITER,
        log_frequency=log_frequency,
        verbose=0,
        record_history=True,
        random_state=seed,
        n_landmark=count_from_fraction(n_samples, landmark_fraction),
        random_landmark_fraction=strategy.random_landmark_fraction,
        targets_per_landmark=count_from_fraction(
            n_samples, TARGETS_PER_LANDMARK_FRACTION
        ),
        n_local_pairs=dataset.n_local_pairs,
        local_weight=LOCAL_WEIGHT,
        direct_stress_weight=direct_stress_weight,
        outer_step_size=outer_step_size,
        device="auto",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
    )


def benchmark_filename(
    seed,
    strategy,
    direct_stress_weight,
    landmark_fraction,
    outer_step_size=1.0,
    *,
    n_samples=N_SAMPLES,
):
    n_landmark = count_from_fraction(n_samples, landmark_fraction)
    targets = count_from_fraction(n_samples, TARGETS_PER_LANDMARK_FRACTION)
    outer_iter = OUTER_ITER_BY_LANDMARK_FRACTION[landmark_fraction]
    step_tag = "" if float(outer_step_size) == 1.0 else f"_oss{float_code(outer_step_size)}"
    return (
        f"n{n_samples}_lm{percent_code(n_landmark, n_samples)}_"
        f"{landmark_strategy_code(strategy)}_"
        f"{direct_stress_code(direct_stress_weight)}_"
        f"targ{percent_code(targets, n_samples)}_"
        f"locdir_ii{INNER_ITER}_oi{outer_iter}{step_tag}_s{seed}.csv"
    )


def landmark_strategy_code(strategy):
    return f"rf{percent_code(int(round(100 * strategy.random_landmark_fraction)), 100)}"


def direct_stress_code(weight):
    if weight == 0.0:
        return "dsnone"
    return f"dsmds_w{float_code(weight)}"


def count_from_fraction(n_samples, fraction):
    return max(1, int(round(float(fraction) * n_samples)))


def percent_code(count, total):
    return f"{int(round(100 * float(count) / float(total)))}p"


def float_code(value):
    return ("%g" % float(value)).replace(".", "p").replace("-", "m")


def write_history_csv(
        path,
        history,
        *,
        dataset,
        seed,
        landmark_strategy,
        direct_stress_weight,
        landmark_fraction,
        outer_step_size,
        options,
        wall_time,
):
    fieldnames = [
        "dataset",
        "n_samples",
        "seed",
        "landmark_strategy",
        "random_landmark_fraction",
        "direct_stress_weight",
        "landmark_fraction",
        "inner_iter",
        "outer_iter_total",
        "log_frequency",
        "graph_neighbors",
        "n_landmark",
        "targets_per_landmark",
        "n_local_pairs",
        "local_weight",
        "outer_step_size",
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
                    "dataset": dataset.name,
                    "n_samples": len(dataset.D),
                    "seed": seed,
                    "landmark_strategy": landmark_strategy.name,
                    "random_landmark_fraction": landmark_strategy.random_landmark_fraction,
                    "direct_stress_weight": direct_stress_weight,
                    "landmark_fraction": landmark_fraction,
                    "inner_iter": INNER_ITER,
                    "outer_iter_total": options["outer_iter"],
                    "log_frequency": options["log_frequency"],
                    "graph_neighbors": options["graph_neighbors"],
                    "n_landmark": options["n_landmark"],
                    "targets_per_landmark": options["targets_per_landmark"],
                    "n_local_pairs": options["n_local_pairs"],
                    "local_weight": options["local_weight"],
                    "outer_step_size": options["outer_step_size"],
                    "wall_time": wall_time,
                    **row,
                }
            )


def read_last_summary_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return summary_from_history(path, rows, rows[-1].get("wall_time", "") if rows else "")


def summary_from_history(path, history, wall_time):
    if not history:
        return {"path": str(path), "wall_time": wall_time}
    final = history[-1]
    return {
        "path": str(path),
        "dataset": final.get("dataset", ""),
        "n_samples": final.get("n_samples", N_SAMPLES),
        "seed": final.get("seed", ""),
        "landmark_strategy": final.get("landmark_strategy", ""),
        "random_landmark_fraction": final.get("random_landmark_fraction", ""),
        "direct_stress_weight": final.get("direct_stress_weight", ""),
        "landmark_fraction": final.get("landmark_fraction", ""),
        "n_landmark": final.get("n_landmark", ""),
        "targets_per_landmark": final.get("targets_per_landmark", ""),
        "inner_iter": final.get("inner_iter", INNER_ITER),
        "outer_step_size": final.get("outer_step_size", "1.0"),
        "outer_iter_total": final.get("outer_iter_total", ""),
        "wall_time": wall_time,
        "final_outer_iter": final.get("outer_iter", ""),
        "final_elapsed": final.get("elapsed", ""),
        "final_full_geodesic_stress": final.get("full_geodesic_stress", ""),
        "final_normalized_full_geodesic_stress": final.get("normalized_full_geodesic_stress", ""),
        "final_masked_stress": final.get("masked_stress", ""),
    }


def write_summary(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset_curves(dataset_name, *, csv_dir, fig_dir):
    curves = load_curves(csv_dir)
    if not curves:
        print(f"No curves to plot for {dataset_name}: {csv_dir}")
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = {0.0: "tab:blue", 0.1: "tab:orange", 0.5: "tab:green", 1.0: "tab:red"}
    for curve in sorted(curves, key=direct_stress_curve_key):
        step = curve["outer_step_size"]
        label = (
            direct_stress_curve_label(curve)
            + ("" if step == 1.0 else f", step {step:g}")
            + f" (n={curve.get('n_seeds', 1)})"
        )
        ax.plot(
            curve["elapsed"],
            curve["stress"],
            color=colors.get(curve["direct_stress_weight"], "0.2"),
            linewidth=1.9,
            label=label,
        )
    ax.set_xlabel("Optimization time excluding full-stress evaluations (s)")
    ax.set_ylabel("Full geodesic stress")
    ax.set_yscale("log")
    ax.set_title(f"Path-frozen direct stress: {dataset_name}, n={N_SAMPLES}, ii={INNER_ITER}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(fig_dir / f"direct_stress_{dataset_name}_n{N_SAMPLES}_ii{INNER_ITER}.{suffix}")
    plt.close(fig)


def load_curves(csv_dir):
    curves = []
    for strategy in LANDMARK_STRATEGIES:
        for direct_stress_weight in DIRECT_STRESS_WEIGHTS:
            for fraction in LANDMARK_FRACTIONS:
                for outer_step_size in OUTER_STEP_SIZES:
                    seed_curves = []
                    for seed in SEEDS:
                        path = csv_dir / benchmark_filename(
                            seed,
                            strategy,
                            direct_stress_weight,
                            fraction,
                            outer_step_size,
                        )
                        if path.exists():
                            seed_curves.append(load_curve(path))
                    if seed_curves:
                        curves.append(average_seed_curves(seed_curves))
    return curves


def load_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty benchmark CSV: {path}")
    first = rows[0]
    return {
        "path": path,
        "strategy": first["landmark_strategy"],
        "random_landmark_fraction": float(first["random_landmark_fraction"]),
        "direct_stress_weight": float(first["direct_stress_weight"]),
        "fraction": float(first["landmark_fraction"]),
        "outer_step_size": float(first.get("outer_step_size", 1.0)),
        "outer_iter": np.array([int(row["outer_iter"]) for row in rows], dtype=int),
        "elapsed": np.array([float(row["elapsed"]) for row in rows], dtype=float),
        "stress": np.array([float(row["full_geodesic_stress"]) for row in rows], dtype=float),
    }


def average_seed_curves(curves):
    common_outer = set(curves[0]["outer_iter"])
    for curve in curves[1:]:
        common_outer &= set(curve["outer_iter"])
    outer = np.array(sorted(common_outer), dtype=int)
    elapsed_stack = []
    stress_stack = []
    for curve in curves:
        index = {outer_iter: i for i, outer_iter in enumerate(curve["outer_iter"])}
        idx = [index[int(outer_iter)] for outer_iter in outer]
        elapsed_stack.append(curve["elapsed"][idx])
        stress_stack.append(curve["stress"][idx])
    return {
        "strategy": curves[0]["strategy"],
        "random_landmark_fraction": curves[0]["random_landmark_fraction"],
        "direct_stress_weight": curves[0]["direct_stress_weight"],
        "fraction": curves[0]["fraction"],
        "outer_step_size": curves[0]["outer_step_size"],
        "outer_iter": outer,
        "elapsed": np.mean(elapsed_stack, axis=0),
        "stress": np.mean(stress_stack, axis=0),
        "n_seeds": len(curves),
    }


def direct_stress_curve_key(curve):
    return curve["direct_stress_weight"]


def direct_stress_curve_label(curve):
    weight = curve["direct_stress_weight"]
    return "none" if weight == 0 else f"direct MDS, w={weight:g}"


if __name__ == "__main__":
    main()
