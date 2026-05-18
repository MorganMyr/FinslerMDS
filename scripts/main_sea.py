"""Sea-current Finsler-MDS experiments.

This script is a compact version of ``main_2D_maps.main_sea`` meant for
comparing direct and embedding-geodesic Finsler-MDS variants on a synthetic
current map.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np

from finsler_mds import (
    ConvexifiedMatsumotoMetric,
    ConvexifiedToblerMetric,
    MatsumotoMetric,
    RandersMetric,
    fit_finsler_mds,
    utils,
)
from sea_datasets import (
    add_obstacles_to_axis,
    current_map_distances,
    make_sea_dataset,
    normalize_sea_dataset_name,
    obstacle_array,
)


def main_sea():
    seed = 0
    script_dir = Path(__file__).resolve().parent

    dataset_name = normalize_sea_dataset_name(os.environ.get("SEA_DATASET", "sea2"))

    dir_res = script_dir / "res" / dataset_name
    dir_fig = dir_res / "figures"
    dir_embeddings = dir_res / "embeddings"

    optimizer = os.environ.get("SEA_OPTIMIZER", "path_frozen")  # one of {"smacof", "gd", "path_frozen", "soft_bf"}
    metric_name = os.environ.get("SEA_METRIC", "randers")
    alpha_current = 0.8
    alpha_embedding = 0.8
    tobler = {"a": 7.0, "b": 0.04}
    init_source = os.environ.get(
        "SEA_INIT",
        "isomap",
    )  # one of {"isomap", "target_mds", "latest_same"}

    n_samples = 2000
    n_components = 3
    n_neighbors = 10
    sea_length = 10.0
    sea_width = 10.0
    current_frequency = 2.0

    smacof = {
        "max_iter": 100,
        "pseudo_inv_solver": "gmres",
        "project_on_V": True,
        "check_monotony": False,
    }
    gd = {
        "max_iter": 350,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-9, "maxls": 40},
        "verbose": 0,
    }
    path_frozen = {
        "graph_neighbors": 12,
        "max_iter": 300,
        "inner_iter": 2,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 40},
        "n_global_landmarks": 180,
        "n_local_neighbors": 12,
        "local_pair_mode": "direct",
        "max_global_targets_per_source": 100,
        "local_global_reweighting": "count",
        "local_weight": 1.0,
        "device": "auto",
        "verbose": 1,
    }
    soft_bf = {
        "graph_neighbors": 12,
        "beta": 30.0,
        "n_relaxations": 30,
        "max_iter": 150,
        "n_graph_updates": 10,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 40},
        "n_global_landmarks": 150,
        "n_local_neighbors": 12,
        "local_pair_mode": "direct",
        "max_global_targets_per_source": 200,
        "local_global_reweighting": "count",
        "local_weight": 1.0,
        "source_batch_size": 8,
        "device": "auto",
        "verbose": 1,
    }
    for directory in (dir_fig, dir_embeddings):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dataset = make_sea_dataset(
        dataset_name,
        n_samples=n_samples,
        alpha_current=alpha_current,
        rng=rng,
        graph_neighbors=n_neighbors,
        sea_length=sea_length,
        sea_width=sea_width,
        current_frequency=current_frequency,
    )
    X = dataset.X
    randers_field = dataset.randers_field
    current_field = dataset.current_field
    optimizer_kind = normalize_optimizer(optimizer)
    effective_metric_name = "randers" if optimizer_kind == "smacof" else normalize_metric_name(metric_name)
    run_key = sea_run_key(
        optimizer_kind,
        effective_metric_name,
        alpha_current,
        alpha_embedding,
        tobler=tobler,
    )

    print(f"Dataset: {dataset.key} ({dataset.title})")
    print("Target Randers geodesic distances on the current map")
    target_distances, target_predecessors = current_map_distances(
        dataset,
        n_neighbors=n_neighbors,
        path_method="auto",
    )

    init = make_initialization(
        X,
        n_components=n_components,
        n_neighbors=n_neighbors,
        target_distances=target_distances,
        init_source=init_source,
        dir_embeddings=dir_embeddings,
        optimizer=optimizer_kind,
        metric_name=effective_metric_name,
        alpha_current=alpha_current,
    )
    metric = make_metric(effective_metric_name, alpha_embedding, tobler=tobler)
    fit_kwargs = optimizer_kwargs(
        optimizer_kind,
        metric=metric,
        init=init,
        n_components=n_components,
        smacof=smacof,
        gd=gd,
        path_frozen=path_frozen,
        soft_bf=soft_bf,
        seed=seed,
    )

    print(
        f"Running {optimizer_kind} with metric={display_metric_name(effective_metric_name)} "
        f"({metric_parameter_summary(effective_metric_name, alpha_embedding, tobler=tobler)}, "
        f"target alpha={alpha_current:g})"
    )
    embedding, stress = fit_finsler_mds(target_distances, print_time=True, **fit_kwargs)
    print(f"  stress: {stress}")
    extrema = utils.get_extrema(embedding) if embedding.shape[1] == 3 else None

    np.savez(
        dir_embeddings / f"{run_key}.npz",
        embedding=embedding,
        stress=np.asarray(stress, dtype=float),
        X=X,
        randers_field=randers_field,
        current_field=current_field,
        target_distances=target_distances,
        target_predecessors=target_predecessors,
        init=init,
        init_source=np.asarray(init_source),
        dataset=np.asarray(dataset.key),
        optimizer=np.asarray(optimizer_kind),
        metric=np.asarray(display_metric_name(effective_metric_name)),
        alpha_current=np.asarray(alpha_current, dtype=float),
        alpha_embedding=np.asarray(alpha_embedding, dtype=float),
        tobler_a=np.asarray(tobler["a"], dtype=float),
        tobler_b=np.asarray(tobler["b"], dtype=float),
        obstacles=obstacle_array(dataset.obstacles),
        bounds=np.asarray(dataset.bounds, dtype=float),
    )

    plot_current_map(
        X,
        current_field,
        title=f"{dataset.title}, alpha={alpha_current:g}",
        save_path=dir_fig / f"sea_a{alpha_tag(alpha_current)}_current.pdf",
        extrema=extrema,
        obstacles=dataset.obstacles,
        bounds=dataset.bounds,
    )

    plot_embedding(
        embedding,
        X,
        extrema=extrema,
        title=(
            f"Sea {display_optimizer_name(optimizer_kind)}, "
            f"{display_metric_name(effective_metric_name)}, "
            f"target alpha={alpha_current:g}, "
            f"{metric_parameter_summary(effective_metric_name, alpha_embedding, tobler=tobler)}"
        ),
        save_path=dir_fig / f"{run_key}.pdf",
    )
    print(f"Saved figures in: {dir_fig}")
    print(f"Saved embedding: {dir_embeddings / f'{run_key}.npz'}")


def make_metric(metric_name, alpha, *, tobler):
    name = normalize_metric_name(metric_name)
    if name == "randers":
        return RandersMetric(alpha=alpha)
    if name == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    if name == "convexified_matsumoto":
        return ConvexifiedMatsumotoMetric(alpha=alpha)
    if name == "convexified_tobler":
        return ConvexifiedToblerMetric(a=tobler["a"], b=tobler["b"])
    raise ValueError(f"Unknown metric {metric_name!r}.")


def make_initialization(
    X,
    *,
    n_components,
    n_neighbors,
    target_distances,
    init_source,
    dir_embeddings,
    optimizer,
    metric_name,
    alpha_current,
):
    source = normalize_init_source(init_source)
    if source == "isomap":
        return utils.IsomapWithPreds(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
    if source == "target_mds":
        return classical_mds_initialization(target_distances, n_components=n_components)

    path = latest_embedding_path(
        dir_embeddings,
        optimizer=optimizer,
        metric_name=metric_name,
        alpha_current=alpha_current,
    )
    with np.load(path) as data:
        init = np.asarray(data["embedding"], dtype=float)
    if init.shape[0] != X.shape[0]:
        raise ValueError(
            f"Saved init {path} has {init.shape[0]} points, but current dataset has {X.shape[0]}."
        )
    init = adapt_embedding_dimension(init, n_components)
    print(f"Loaded initialization from: {path}")
    return init


def classical_mds_initialization(distances, *, n_components):
    D = np.asarray(distances, dtype=float)
    if not np.all(np.isfinite(D)):
        raise ValueError("target_mds initialization requires finite target distances.")
    sym = 0.5 * (D + D.T)
    n = sym.shape[0]
    squared = sym**2
    row_mean = squared.mean(axis=1, keepdims=True)
    col_mean = squared.mean(axis=0, keepdims=True)
    total_mean = squared.mean()
    gram = -0.5 * (squared - row_mean - col_mean + total_mean)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.maximum(eigvals[order], 0.0)
    return eigvecs[:, order] * np.sqrt(eigvals)


def latest_embedding_path(dir_embeddings, *, optimizer, metric_name, alpha_current):
    optimizer = normalize_optimizer(optimizer)
    metric = normalize_metric_name(metric_name)
    prefix = f"sea_a{alpha_tag(alpha_current)}_{optimizer_abbrev(optimizer)}_{metric_abbrev(metric)}_"
    candidates = sorted(
        dir_embeddings.glob(f"{prefix}*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No saved embedding matching {prefix}*.npz in {dir_embeddings}. "
            "Set init_source='isomap' to start from scratch."
        )
    return candidates[0]


def adapt_embedding_dimension(embedding, n_components):
    if embedding.shape[1] == n_components:
        return embedding
    if embedding.shape[1] > n_components:
        return embedding[:, :n_components]
    padding = np.zeros((embedding.shape[0], n_components - embedding.shape[1]), dtype=embedding.dtype)
    return np.hstack([embedding, padding])


def optimizer_kwargs(optimizer, *, metric, init, n_components, smacof, gd, path_frozen, soft_bf, seed):
    if optimizer == "smacof":
        return {
            "optimizer": "smacof_randers",
            "metric": RandersMetric(alpha=metric.alpha),
            "init": init,
            "n_components": n_components,
            "n_init": 1,
            "n_jobs": 1,
            **smacof,
        }
    if optimizer == "gd":
        return {
            "optimizer": "gradient_descent",
            "metric": metric,
            "init": init,
            "n_components": n_components,
            "random_state": seed,
            **gd,
        }
    if optimizer == "path_frozen":
        return {
            "optimizer": "path_frozen",
            "metric": metric,
            "init": init,
            "n_components": n_components,
            "random_state": seed,
            **path_frozen,
        }
    if optimizer == "soft_bf":
        return {
            "optimizer": "soft_bellman_ford",
            "metric": metric,
            "init": init,
            "n_components": n_components,
            "random_state": seed,
            **soft_bf,
        }
    raise ValueError(f"Unknown optimizer {optimizer!r}.")


def plot_current_map(X, current_field, *, title, save_path, extrema=None, obstacles=(), bounds=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sea_colors(X)
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=45, lw=0)
    ax.quiver(
        X[:, 0],
        X[:, 1],
        current_field[:, 0],
        current_field[:, 1],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1.4,
        width=0.004,
        alpha=0.75,
    )
    add_obstacles_to_axis(ax, obstacles)
    add_extrema_to_2d_axis(ax, X, extrema)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box" if bounds is not None else "datalim")
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_embedding(embedding, X_original, *, title, save_path, extrema=None):
    embedding = np.asarray(embedding, dtype=float)
    colors = sea_colors(X_original)
    if embedding.shape[1] == 3:
        fig = plt.figure(figsize=(11, 9))
        views = [("front", 20, -60), ("side", 20, 30), ("top", 90, -90), ("diagonal", 35, 135)]
        for view_id, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, view_id + 1, projection="3d")
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=18, lw=0)
            add_extrema_to_3d_axis(ax, extrema)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(view_name)
            ax.set_box_aspect([1, 1, 1])
            utils.set_axes_equal(ax)
            ax.set_xlabel("Embedding 1")
            ax.set_ylabel("Embedding 2")
            ax.set_zlabel("Embedding 3")
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=32, lw=0)
        add_extrema_to_2d_axis(ax, embedding, extrema)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def add_extrema_to_2d_axis(ax, points, extrema):
    if extrema is None:
        return
    max_idx = np.asarray(extrema["maxima"]["indices"], dtype=int)
    min_idx = np.asarray(extrema["minima"]["indices"], dtype=int)
    if len(max_idx):
        ax.scatter(points[max_idx, 0], points[max_idx, 1], c="black", s=200, lw=0, alpha=1, zorder=1_000_000)
    if len(min_idx):
        ax.scatter(points[min_idx, 0], points[min_idx, 1], c="gray", s=200, lw=0, alpha=1, zorder=1_000_000)


def add_extrema_to_3d_axis(ax, extrema):
    if extrema is None:
        return
    maxima = extrema["maxima"]["values"]
    minima = extrema["minima"]["values"]
    if len(maxima):
        ax.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color="black", s=70, alpha=1, marker="o", zorder=1_000_000)
    if len(minima):
        ax.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color="gray", s=70, alpha=1, marker="o", zorder=1_000_000)


def sea_colors(X):
    values = (X[:, 0] - X[:, 0].min()) / max(np.ptp(X[:, 0]), 1e-12)
    return plt.cm.jet(values)


def sea_run_key(optimizer, metric_name, alpha_current, alpha_embedding, *, tobler=None):
    optimizer = normalize_optimizer(optimizer)
    metric = "randers" if optimizer == "smacof" else normalize_metric_name(metric_name)
    metric_tag = metric_abbrev(metric)
    parameter_tag = f"a{alpha_tag(alpha_embedding)}"
    if metric == "convexified_tobler":
        tobler = {} if tobler is None else tobler
        parameter_tag = f"ta{alpha_tag(tobler.get('a', 3.5))}_tb{alpha_tag(tobler.get('b', 0.05))}"
    return (
        f"sea_a{alpha_tag(alpha_current)}_"
        f"{optimizer_abbrev(optimizer)}_{metric_tag}_"
        f"{parameter_tag}"
    )


def normalize_optimizer(optimizer):
    name = str(optimizer).lower().replace("-", "_")
    aliases = {
        "smacof": "smacof",
        "randers_smacof": "smacof",
        "smacof_randers": "smacof",
        "gd": "gd",
        "gradient_descent": "gd",
        "path_frozen": "path_frozen",
        "pf": "path_frozen",
        "soft_bf": "soft_bf",
        "sbf": "soft_bf",
        "soft_bellman_ford": "soft_bf",
    }
    if name not in aliases:
        raise ValueError("optimizer must be one of {'smacof', 'gd', 'path_frozen', 'soft_bf'}.")
    return aliases[name]


def normalize_metric_name(metric_name):
    name = str(metric_name).lower().replace("-", "_")
    aliases = {
        "randers": "randers",
        "r": "randers",
        "matsumoto": "matsumoto",
        "mats": "matsumoto",
        "convexified_matsumoto": "convexified_matsumoto",
        "convexifiedmatsumoto": "convexified_matsumoto",
        "cmats": "convexified_matsumoto",
        "convmats": "convexified_matsumoto",
        "convexified_tobler": "convexified_tobler",
        "convexifiedtobler": "convexified_tobler",
        "tobler_convexified": "convexified_tobler",
        "ctobler": "convexified_tobler",
        "ctobl": "convexified_tobler",
    }
    if name not in aliases:
        raise ValueError(
            "metric_name must be one of "
            "{'randers', 'matsumoto', 'convexified_matsumoto', 'convexified_tobler'}."
        )
    return aliases[name]


def normalize_init_source(init_source):
    name = str(init_source).lower().replace("-", "_")
    aliases = {
        "isomap": "isomap",
        "target_mds": "target_mds",
        "mds": "target_mds",
        "classical_mds": "target_mds",
        "latest": "latest_same",
        "latest_same": "latest_same",
        "saved": "latest_same",
    }
    if name not in aliases:
        raise ValueError("init_source must be one of {'isomap', 'target_mds', 'latest_same'}.")
    return aliases[name]


def optimizer_abbrev(optimizer):
    return {"smacof": "smacof", "gd": "gd", "path_frozen": "pf", "soft_bf": "sbf"}[optimizer]


def metric_abbrev(metric_name):
    return {
        "randers": "r",
        "matsumoto": "mats",
        "convexified_matsumoto": "cmats",
        "convexified_tobler": "ctobl",
    }[metric_name]


def display_optimizer_name(optimizer):
    return {"smacof": "SMACOF", "gd": "GD", "path_frozen": "Path-frozen", "soft_bf": "Soft-BF"}[optimizer]


def display_metric_name(metric_name):
    return {
        "randers": "Randers",
        "matsumoto": "Matsumoto",
        "convexified_matsumoto": "Convexified Matsumoto",
        "convexified_tobler": "Convexified Tobler",
    }[normalize_metric_name(metric_name)]


def metric_parameter_summary(metric_name, alpha_embedding, *, tobler):
    if normalize_metric_name(metric_name) == "convexified_tobler":
        return f"Tobler a={tobler['a']:g}, b={tobler['b']:g}"
    return f"embedding alpha={alpha_embedding:g}"


def alpha_tag(alpha):
    alpha = float(alpha)
    if alpha.is_integer():
        return str(int(alpha))
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main_sea()
