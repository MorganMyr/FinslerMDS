"""Search embedding alpha values on the three-mountain dataset."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from finsler_mds import ConvexifiedMatsumotoMetric, RandersMetric, fit_finsler_mds, geodesic_embedding_stress, utils
from main_mountains import (
    make_metric,
    make_mountain_surface,
    mountains_run_key,
    normalize_metric_name,
    normalize_optimizer,
    optimizer_kwargs,
)


def main():
    seed = 42
    alpha_target = 1.0
    n_neighbors = 10
    n_components = 3
    grid = {"nx": 60, "ny": 32, "xlim": (-15.0, 15.0), "ylim": (-8.0, 8.0), "xy_noise": 0.08}
    mountains = {
        "left": {"center": (-10.0, 0.0), "height": 4.0, "sigma": (2.2, 2.2)},
        "middle": {"center": (0.0, 0.0), "height": 3.5, "sigma": (0.8, 4.0)},
        "right": {"center": (10.0, 0.0), "height": 4.0, "sigma": (2.2, 2.2)},
    }
    short_alphas = {
        ("smacof", "randers"): [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
        ("gd", "c_matsumoto"): [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        ("path_frozen", "c_matsumoto"): [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        ("path_frozen", "randers"): [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
    }
    refine_radius = {
        "randers": 0.1,
        "c_matsumoto": 0.25,
    }

    dir_res = SCRIPT_DIR / "res" / "mountains"
    dir_embeddings = dir_res / "embeddings"
    dir_search = dir_res / "alpha_search"
    dir_embeddings.mkdir(parents=True, exist_ok=True)
    dir_search.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    X = make_mountain_surface(grid, mountains, rng)
    target_metric = ConvexifiedMatsumotoMetric(alpha=alpha_target)
    print("Precomputing target distances")
    D, _ = utils.compute_metric_dist_matrix(X, target_metric, n_neighbors=n_neighbors, directed=True)
    if not np.all(np.isfinite(D)):
        raise ValueError("Target graph disconnected.")
    np.fill_diagonal(D, 0.0)
    terrain_init = X - X.mean(axis=0, keepdims=True)

    rows = []
    for case, alphas in short_alphas.items():
        optimizer, metric = case
        print(f"\nShort sweep: {optimizer} / {metric}")
        short_rows = [
            run_case(
                X,
                D,
                init=terrain_init,
                alpha_target=alpha_target,
                alpha_embedding=alpha,
                optimizer=optimizer,
                metric_name=metric,
                params=short_params(),
                n_components=n_components,
                seed=seed,
                save_dir=dir_embeddings,
                stage="short",
            )
            for alpha in alphas
        ]
        rows.extend(short_rows)
        best = min(short_rows, key=lambda row: row["eval_stress"])
        refined = refined_alphas(best["alpha_embedding"], metric, radius=refine_radius[normalize_metric_name(metric)])
        print(f"Refine around alpha={best['alpha_embedding']:.4g}: {refined}")
        init = np.load(best["embedding_path"])["embedding"]
        rows.extend(
            run_case(
                X,
                D,
                init=init,
                alpha_target=alpha_target,
                alpha_embedding=alpha,
                optimizer=optimizer,
                metric_name=metric,
                params=refined_params(),
                n_components=n_components,
                seed=seed,
                save_dir=dir_embeddings,
                stage="refine",
            )
            for alpha in refined
        )
        write_rows(dir_search / "mountains_alpha_search.csv", rows)

    write_rows(dir_search / "mountains_alpha_search.csv", rows)
    print_summary(rows)


def run_case(X, D, *, init, alpha_target, alpha_embedding, optimizer, metric_name, params, n_components, seed, save_dir, stage):
    optimizer = normalize_optimizer(optimizer)
    effective_metric = "randers" if optimizer == "smacof" else normalize_metric_name(metric_name)
    metric = make_metric(effective_metric, alpha_embedding)
    kwargs = optimizer_kwargs(
        optimizer,
        metric=metric,
        init=init,
        n_components=n_components,
        smacof=params["smacof"],
        gd=params["gd"],
        path_frozen=params["path_frozen"],
        seed=seed,
    )
    start = perf_counter()
    embedding, stress = fit_finsler_mds(D, print_time=False, **kwargs)
    elapsed = perf_counter() - start
    eval_stress = evaluate_embedding(embedding, D, metric=metric, optimizer=optimizer, graph_neighbors=params["path_frozen"]["graph_neighbors"])
    key = f"{stage}_{mountains_run_key(optimizer, effective_metric, alpha_target, alpha_embedding)}"
    path = save_dir / f"{key}.npz"
    np.savez(
        path,
        embedding=embedding,
        stress=np.asarray(stress, dtype=float),
        eval_stress=np.asarray(eval_stress, dtype=float),
        X=X,
        init=init,
        optimizer=np.asarray(optimizer),
        embedding_metric=np.asarray(effective_metric),
        alpha_target=np.asarray(alpha_target, dtype=float),
        alpha_embedding=np.asarray(alpha_embedding, dtype=float),
        stage=np.asarray(stage),
    )
    row = {
        "stage": stage,
        "optimizer": optimizer,
        "metric": effective_metric,
        "alpha_embedding": float(alpha_embedding),
        "optimizer_stress": float(stress),
        "eval_stress": float(eval_stress),
        "seconds": float(elapsed),
        "embedding_path": str(path),
    }
    print(
        f"  alpha={alpha_embedding:g}: optimizer={float(stress):.6g}, "
        f"eval={float(eval_stress):.6g}, time={elapsed:.1f}s"
    )
    return row


def evaluate_embedding(embedding, D, *, metric, optimizer, graph_neighbors):
    if optimizer == "path_frozen":
        return geodesic_embedding_stress(
            embedding,
            D,
            metric=metric,
            n_neighbors=graph_neighbors,
            on_unreachable="warn_skip",
        )
    embedded = metric.pairwise(embedding)
    active = np.isfinite(D)
    np.fill_diagonal(active, False)
    return float(np.sum((embedded[active] - D[active]) ** 2))


def short_params():
    return {
        "smacof": {"max_iter": 8, "pseudo_inv_solver": "gmres", "project_on_V": True, "check_monotony": False},
        "gd": {"max_iter": 60, "eps": 1e-6, "method": "L-BFGS-B", "optimizer_options": {"ftol": 1e-8, "maxls": 30}, "verbose": 0},
        "path_frozen": {
            "graph_neighbors": 12,
            "outer_iter": 3,
            "inner_iter": 20,
            "eps": 1e-6,
            "method": "L-BFGS-B",
            "optimizer_options": {"ftol": 1e-8, "maxls": 30},
            "n_landmark": 70,
            "n_local_pairs": 8,
            "local_pair_mode": "direct",
            "targets_per_landmark": 120,
            "local_global_reweighting": "count",
            "local_weight": 1.0,
            "device": "auto",
            "verbose": 0,
        },
    }


def refined_params():
    params = short_params()
    params["smacof"]["max_iter"] = 25
    params["gd"]["max_iter"] = 220
    params["gd"]["optimizer_options"] = {"ftol": 1e-9, "maxls": 40}
    params["path_frozen"].update({
        "outer_iter": 8,
        "inner_iter": 35,
        "n_landmark": 120,
        "targets_per_landmark": 180,
        "optimizer_options": {"ftol": 1e-8, "maxls": 40},
    })
    return params


def refined_alphas(center, metric, *, radius):
    if normalize_metric_name(metric) == "randers":
        lo, hi = max(0.0, center - radius), min(0.97, center + radius)
    else:
        lo, hi = max(0.0, center - radius), center + radius
    return [float(x) for x in np.linspace(lo, hi, 5)]


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nBest refined results")
    for optimizer, metric in [
        ("smacof", "randers"),
        ("gd", "c_matsumoto"),
        ("path_frozen", "c_matsumoto"),
        ("path_frozen", "randers"),
    ]:
        subset = [r for r in rows if r["stage"] == "refine" and r["optimizer"] == optimizer and r["metric"] == normalize_metric_name(metric)]
        best = min(subset, key=lambda row: row["eval_stress"])
        print(f"  {optimizer}/{metric}: alpha={best['alpha_embedding']:.4g}, eval={best['eval_stress']:.6g}")


if __name__ == "__main__":
    main()
