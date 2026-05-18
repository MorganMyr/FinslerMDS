"""Alpha sweeps for sea current-map experiments.

This script is intentionally experiment-oriented: it keeps ``main_sea.py` as
the clean single-run entry point, while this file runs small continuations and
records comparable full geodesic stresses.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
from time import perf_counter
from time import strftime

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finsler_mds import (  # noqa: E402
    MatsumotoMetric,
    RandersMetric,
    fit_finsler_mds,
    geodesic_embedding_stress,
)
from main_sea import (  # noqa: E402
    adapt_embedding_dimension,
    alpha_tag,
    classical_mds_initialization,
)
from sea_datasets import current_map_distances, make_sea_dataset  # noqa: E402


SEED = 0
DATASET = "sea2"
N_SAMPLES = 2000
N_COMPONENTS = 3
ALPHA_CURRENT = 0.8
TARGET_GRAPH_NEIGHBORS = 10
EMBEDDING_GRAPH_NEIGHBORS = 12

RESULT_DIR = SCRIPT_DIR / "res" / DATASET / "alpha_search"
EMBED_DIR = SCRIPT_DIR / "res" / DATASET / "embeddings"
CSV_PATH = RESULT_DIR / "sea2_alpha_search.csv"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    experiment = os.environ.get("SEA_ALPHA_EXPERIMENT", "smacof_randers")
    alphas = parse_alphas(os.environ.get("SEA_ALPHA_VALUES", "0.0,0.2,0.4,0.6,0.8,0.9"))
    max_iter = int(os.environ.get("SEA_ALPHA_MAX_ITER", "100"))
    n_graph_updates = int(os.environ.get("SEA_ALPHA_GRAPH_UPDATES", "4"))
    init_mode = os.environ.get("SEA_ALPHA_INIT", "target_mds")
    continuation = env_bool("SEA_ALPHA_CONTINUATION", default=False)

    print(f"Experiment: {experiment}")
    print(f"Alphas: {', '.join(f'{a:g}' for a in alphas)}")
    print("Building sea2 target distances")
    rng = np.random.default_rng(SEED)
    dataset = make_sea_dataset(
        DATASET,
        n_samples=N_SAMPLES,
        alpha_current=ALPHA_CURRENT,
        rng=rng,
        graph_neighbors=TARGET_GRAPH_NEIGHBORS,
    )
    target_distances, _ = current_map_distances(
        dataset,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        path_method="auto",
    )
    target_mds = classical_mds_initialization(target_distances, n_components=N_COMPONENTS)

    previous_embedding = None
    for alpha in alphas:
        metric_name, metric, optimizer, kwargs = experiment_config(
            experiment,
            alpha=alpha,
            max_iter=max_iter,
            n_graph_updates=n_graph_updates,
        )
        init = choose_init(
            init_mode,
            target_mds=target_mds,
            previous_embedding=previous_embedding if continuation else None,
            experiment=experiment,
            alpha=alpha,
        )
        print(
            f"\n=== {experiment}, alpha={alpha:g}, max_iter={max_iter}"
            + (f", graph_updates={n_graph_updates}" if optimizer == "soft_bellman_ford" else "")
            + f", init={init_mode} ==="
        )
        start = perf_counter()
        embedding, optimizer_stress = fit_finsler_mds(
            target_distances,
            optimizer=optimizer,
            metric=metric,
            init=init,
            n_components=N_COMPONENTS,
            random_state=SEED,
            print_time=True,
            **kwargs,
        )
        elapsed = perf_counter() - start
        print("Evaluating full hard geodesic stress")
        full_geodesic_stress = geodesic_embedding_stress(
            embedding,
            target_distances,
            metric=metric,
            n_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
            on_unreachable="inf",
        )
        run_key = (
            f"sea_a{alpha_tag(ALPHA_CURRENT)}_"
            f"{experiment_abbrev(experiment)}_{metric_abbrev(metric_name)}_"
            f"a{alpha_tag(alpha)}_search_"
            f"it{max_iter}_u{n_graph_updates}_"
            f"{strftime('%Y%m%d_%H%M%S')}"
        )
        np.savez(
            EMBED_DIR / f"{run_key}.npz",
            embedding=embedding,
            stress=np.asarray(optimizer_stress, dtype=float),
            full_geodesic_stress=np.asarray(full_geodesic_stress, dtype=float),
            X=dataset.X,
            current_field=dataset.current_field,
            randers_field=dataset.randers_field,
            target_distances=target_distances,
            dataset=np.asarray(DATASET),
            optimizer=np.asarray(experiment),
            metric=np.asarray(metric_name),
            alpha_current=np.asarray(ALPHA_CURRENT, dtype=float),
            alpha_embedding=np.asarray(alpha, dtype=float),
            max_iter=np.asarray(max_iter, dtype=int),
            n_graph_updates=np.asarray(n_graph_updates, dtype=int),
        )
        append_result(
            {
                "experiment": experiment,
                "metric": metric_name,
                "alpha": alpha,
                "max_iter": max_iter,
                "n_graph_updates": n_graph_updates if optimizer == "soft_bellman_ford" else "",
                "init_mode": init_mode,
                "optimizer_stress": optimizer_stress,
                "full_geodesic_stress": full_geodesic_stress,
                "elapsed_seconds": elapsed,
                "embedding": str(EMBED_DIR / f"{run_key}.npz"),
            }
        )
        print(
            f"alpha={alpha:g}: optimizer stress={optimizer_stress:.6g}, "
            f"full geodesic stress={full_geodesic_stress:.6g}, "
            f"elapsed={elapsed:.1f}s"
        )
        previous_embedding = embedding

    print(f"\nWrote summary: {CSV_PATH}")


def experiment_config(experiment, *, alpha, max_iter, n_graph_updates):
    name = experiment.lower().replace("-", "_")
    if name in {"smacof", "smacof_randers"}:
        return (
            "randers",
            RandersMetric(alpha=alpha),
            "smacof_randers",
            {
                "max_iter": max_iter,
                "n_init": 1,
                "n_jobs": 1,
                "pseudo_inv_solver": "gmres",
                "project_on_V": True,
                "check_monotony": False,
                "device": "auto",
            },
        )
    if name in {"soft_bf_randers", "sbf_randers"}:
        return (
            "randers",
            RandersMetric(alpha=alpha),
            "soft_bellman_ford",
            soft_bf_kwargs(max_iter=max_iter, n_graph_updates=n_graph_updates),
        )
    if name in {"soft_bf_matsumoto", "sbf_matsumoto", "soft_bf_mats"}:
        return (
            "matsumoto",
            MatsumotoMetric(alpha=alpha),
            "soft_bellman_ford",
            soft_bf_kwargs(max_iter=max_iter, n_graph_updates=n_graph_updates),
        )
    raise ValueError(
        "SEA_ALPHA_EXPERIMENT must be one of "
        "{'smacof_randers', 'soft_bf_randers', 'soft_bf_matsumoto'}."
    )


def soft_bf_kwargs(*, max_iter, n_graph_updates):
    return {
        "graph_neighbors": EMBEDDING_GRAPH_NEIGHBORS,
        "beta": float(os.environ.get("SEA_ALPHA_BETA", "35.0")),
        "n_relaxations": int(os.environ.get("SEA_ALPHA_RELAXATIONS", "40")),
        "max_iter": max_iter,
        "n_graph_updates": n_graph_updates,
        "eps": float(os.environ.get("SEA_ALPHA_EPS", "1e-6")),
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": float(os.environ.get("SEA_ALPHA_FTOL", "1e-8")), "maxls": 40},
        "n_global_landmarks": int(os.environ.get("SEA_ALPHA_LANDMARKS", "100")),
        "n_local_neighbors": int(os.environ.get("SEA_ALPHA_LOCAL_NEIGHBORS", "12")),
        "local_pair_mode": "direct",
        "max_global_targets_per_source": int(os.environ.get("SEA_ALPHA_TARGETS", "160")),
        "local_global_reweighting": os.environ.get("SEA_ALPHA_REWEIGHTING", "count"),
        "local_weight": float(os.environ.get("SEA_ALPHA_LOCAL_WEIGHT", "1.0")),
        "source_batch_size": int(os.environ.get("SEA_ALPHA_BATCH_SIZE", "8")),
        "device": os.environ.get("SEA_ALPHA_DEVICE", "auto"),
        "on_unreachable": "warn_skip",
        "verbose": 1,
    }


def choose_init(init_mode, *, target_mds, previous_embedding, experiment, alpha):
    if previous_embedding is not None:
        return previous_embedding
    mode = init_mode.lower().replace("-", "_")
    if mode == "target_mds":
        return target_mds
    if mode == "latest_same":
        path = latest_search_embedding(experiment=experiment, alpha=alpha)
        return load_embedding(path)
    if mode == "latest_any":
        path = latest_search_embedding(experiment=experiment, alpha=None)
        return load_embedding(path)
    if mode == "smacof":
        path = latest_embedding("sea_a0p8_smacof_r_*.npz")
        return load_embedding(path)
    if mode == "path_frozen":
        path = latest_embedding("sea_a0p8_pf_*.npz")
        return load_embedding(path)
    raise ValueError(
        "SEA_ALPHA_INIT must be one of "
        "{'target_mds', 'latest_same', 'latest_any', 'smacof', 'path_frozen'}."
    )


def latest_search_embedding(*, experiment, alpha):
    prefix = f"sea_a{alpha_tag(ALPHA_CURRENT)}_{experiment_abbrev(experiment)}_"
    if alpha is not None:
        prefix += f"*a{alpha_tag(alpha)}_search*.npz"
    else:
        prefix += "*_search*.npz"
    return latest_embedding(prefix)


def latest_embedding(pattern):
    candidates = sorted(EMBED_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No embedding matching {pattern} in {EMBED_DIR}.")
    print(f"Loaded init embedding: {candidates[0]}")
    return candidates[0]


def load_embedding(path):
    with np.load(path) as data:
        embedding = np.asarray(data["embedding"], dtype=float)
    return adapt_embedding_dimension(embedding, N_COMPONENTS)


def append_result(row):
    fieldnames = [
        "experiment",
        "metric",
        "alpha",
        "max_iter",
        "n_graph_updates",
        "init_mode",
        "optimizer_stress",
        "full_geodesic_stress",
        "elapsed_seconds",
        "embedding",
    ]
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_alphas(text):
    return [float(value) for value in text.replace(";", ",").split(",") if value.strip()]


def env_bool(name, *, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def experiment_abbrev(experiment):
    name = experiment.lower().replace("-", "_")
    if name in {"smacof", "smacof_randers"}:
        return "smacof"
    if "mats" in name:
        return "sbf"
    return "sbf"


def metric_abbrev(metric_name):
    return {"randers": "r", "matsumoto": "mats"}[metric_name]


if __name__ == "__main__":
    main()
