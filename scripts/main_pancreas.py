"""Compute and visualize Finsler embeddings of the pancreas velocity dataset."""

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds  # noqa: E402
from finsler_mds.utils import plot_3d_embedding_views, plot_categorical_embedding  # noqa: E402
from finsler_mds.utils.pancreas import (  # noqa: E402
    PANCREAS_CLUSTER_COLORS,
    PANCREAS_PREPROCESSING,
    embedding_metric_tag,
    ensure_pancreas_reference_embedding,
    load_or_compute_pancreas_inputs,
    make_embedding_metric,
    metric_display_name,
    normalize_embedding_dim,
    velocity_distance_formula_tag,
)
from finsler_mds.utils.pancreas_files import (  # noqa: E402
    latest_pancreas_embedding_path,
    load_pancreas_embedding,
    pancreas_embedding_path,
    pancreas_figure_path,
    pancreas_result_stem,
    resolve_pancreas_embedding_path,
    save_pancreas_embedding,
)


CONFIG = {
    "seed": 42,
    "finsler_optimizer": "gradient_descent",  # None, "smacof", "gradient_descent", "finsler_umap", "path_frozen"
    "init_finsler_mds": "umap_2d",  # None, "spectral", UMAP/Isomap variant, saved method, or NPZ path
    "embedding_dim": 3,  # 2 or 3
    "finsler_metric": "randers",  # "randers", "matsumoto", "convexified_matsumoto"
    "alpha_embedding": 0,
    "preprocessing": dict(PANCREAS_PREPROCESSING),
    "velocity": {
        "mode": "dynamical",  # "deterministic", "stochastic", or "dynamical"
        "distance_formula": "randers",  # "randers" or "matsumoto"
        "alpha": 0.0,
        "cos_clip": 1.0,
        "velocity_neighbors": 30,
        "kNN_euclid": 30,
        "kNN_finsler": 0,
        "average_velocity": True,
        "symmetrize_support": True,
        "graph_n_jobs": -1,
        "recover_dynamics_max_iter": 20,
        "recover_dynamics_n_jobs": -1,
    },
    "umap": {
        "n_neighbors": 50,
        "min_dist": 0.5,
        "spread": 1.0,
        "maxiter": 1500,
        "negative_sample_rate": 10,
        "init_pos": "spectral",  # "spectral", "random", "paga", or an obsm key
    },
    "isomap": {"n_neighbors": 30},
    "smacof": {
        "max_iter": 1000,
        "eps": 1e-7,
        "project_on_V": True,
        "device": "auto",  # "cpu", "auto", "gpu", or "cuda"
        "verbose": 1,
    },
    "gradient_descent": {
        "max_iter": 100,
        "eps": 1e-8,
        "optimizer_options": {"ftol": 1e-10, "maxls": 80, "maxcor": 30},
        "device": "auto",  # "cpu", "auto", "gpu", or "cuda"
        "verbose": 1,
    },
    "finsler_umap": {
        "n_neighbors": 30,
        "symmetrize_support": True,
        "symmetrize_rho": False,
        "symmetrize_sigma": True,
        "min_dist": 0.5,
        "spread": 1.0,
        "max_iter": 1000,
        "negative_sample_rate": 10,
        "gradient_clip": 4.0,
        "verbose": 1,
        "log_frequency": 200,
    },
    "path_frozen": {
        "graph_neighbors": 30,
        "outer_iter": 10,
        "inner_iter": 50,
        "n_landmark": 300,
        "random_landmark_fraction": 0.9,
        "n_local_pairs": 30,
        "targets_per_landmark": 1000,
        "local_weight": 0.1,
        "direct_stress_weight": 0.1,
        "outer_step_size": 1.0,
        "device": "auto",  # "cpu", "auto", "gpu", or "cuda"
        "verbose": 1,
    },
}


def main_pancreas(config_overrides=None):
    config = merged_config(config_overrides)
    seed = int(config["seed"])
    n_components = normalize_embedding_dim(config["embedding_dim"])
    optimizer_kind = normalize_finsler_optimizer(config["finsler_optimizer"])
    np.random.seed(seed)

    pancreas_dir = Path(__file__).parent / "res" / "pancreas"
    raw_dir = pancreas_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_or_compute_pancreas_inputs(
        raw_dir,
        preprocessing=config["preprocessing"],
        velocity=config["velocity"],
        seed=seed,
    )
    print(
        f"Directed velocity dissimilarities: {inputs.dissimilarities.shape}, "
        f"finite={np.isfinite(inputs.dissimilarities).mean():.3f}"
    )

    references = {}

    def reference(method, dimension):
        key = method, int(dimension)
        if key not in references:
            embedding, path = ensure_pancreas_reference_embedding(
                inputs,
                raw_dir,
                method=method,
                n_components=dimension,
                preprocessing=config["preprocessing"],
                velocity=config["velocity"],
                options=config[method],
                seed=seed,
            )
            references[key] = embedding, path
            save_pancreas_plot(
                embedding,
                labels=inputs.labels,
                title=f"Pancreas {method.upper()} {dimension}D",
                path=pancreas_figure_path(pancreas_dir, f"{path.stem}.pdf"),
                seed=seed,
            )
        return references[key]

    reference("umap", 2)
    if optimizer_kind is None:
        plt.close("all")
        return inputs.adata, inputs.dissimilarities

    init, init_description = resolve_pancreas_init(
        config["init_finsler_mds"],
        optimizer_kind=optimizer_kind,
        inputs=inputs,
        raw_dir=raw_dir,
        n_components=n_components,
        reference=reference,
    )
    metric = make_embedding_metric(config["finsler_metric"], config["alpha_embedding"])
    optimizer_name, optimizer_metric, optimizer_options = optimizer_spec(
        optimizer_kind,
        metric,
        config,
    )
    print(
        f"Running {n_components}D {optimizer_kind} "
        f"with {metric_display_name(optimizer_metric)} from {init_description}"
    )
    embedding, objective = fit_finsler_mds(
        inputs.dissimilarities,
        metric=optimizer_metric,
        optimizer=optimizer_name,
        init=init,
        n_components=n_components,
        random_state=seed,
        print_time=True,
        **optimizer_options,
    )

    velocity_tag = velocity_distance_formula_tag(
        config["velocity"]["distance_formula"],
        alpha=config["velocity"]["alpha"],
    )
    stem = pancreas_result_stem(
        optimizer_kind,
        n_components=n_components,
        velocity_tag=velocity_tag,
        metric_tag=embedding_metric_tag(optimizer_metric),
    )
    save_pancreas_embedding(
        pancreas_embedding_path(raw_dir, stem),
        embedding,
        inputs.cell_ids,
        objective=objective,
    )
    save_pancreas_plot(
        embedding,
        labels=inputs.labels,
        title=f"Pancreas {optimizer_kind.replace('_', ' ')} — {metric_display_name(optimizer_metric)}",
        path=pancreas_figure_path(pancreas_dir, f"{stem}.pdf"),
        seed=seed,
    )
    print(f"{optimizer_kind} objective: {objective}")
    plt.close("all")
    return inputs.adata, inputs.dissimilarities


def optimizer_spec(kind, metric, config):
    if kind == "smacof":
        options = {
            "n_init": 1,
            "n_jobs": 1,
            **config["smacof"],
        }
        return "smacof", RandersMetric(alpha=config["alpha_embedding"]), options
    return kind, metric, dict(config[kind])


def normalize_finsler_optimizer(value):
    if value is None:
        return None
    aliases = {
        "smacof": "smacof",
        "smacof_randers": "smacof",
        "gradient_descent": "gradient_descent",
        "gd": "gradient_descent",
        "finsler_umap": "finsler_umap",
        "fumap": "finsler_umap",
        "path_frozen": "path_frozen",
        "frozen_paths": "path_frozen",
    }
    try:
        return aliases[str(value).lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(f"Unknown finsler_optimizer: {value!r}.") from exc


def resolve_pancreas_init(
    value,
    *,
    optimizer_kind,
    inputs,
    raw_dir,
    n_components,
    reference,
):
    if value is None:
        return None, "random initialization"
    normalized = str(value).lower().replace("-", "_")
    if normalized in {"spectral", "spec"}:
        if optimizer_kind != "finsler_umap":
            raise ValueError("The spectral init is only available for Finsler-UMAP.")
        return "spectral", "internal spectral initialization"

    reference_aliases = {
        "umap": ("umap", 2),
        "umap_2d": ("umap", 2),
        "umap2d": ("umap", 2),
        "umap_3d": ("umap", 3),
        "umap3d": ("umap", 3),
        "isomap": ("isomap", 2),
        "isomap_2d": ("isomap", 2),
        "isomap2d": ("isomap", 2),
        "isomap_3d": ("isomap", 3),
        "isomap3d": ("isomap", 3),
    }
    method_aliases = {
        "smacof": "smacof",
        "gradient_descent": "gradient_descent",
        "gd": "gradient_descent",
        "finsler_umap": "finsler_umap",
        "fumap": "finsler_umap",
        "path_frozen": "path_frozen",
        "pf": "path_frozen",
    }
    if normalized in reference_aliases:
        method, dimension = reference_aliases[normalized]
        if dimension > n_components:
            raise ValueError(f"A {dimension}D {method} cannot initialize a {n_components}D run.")
        embedding, path = reference(method, dimension)
        description = f"{dimension}D {method.upper()} ({path})"
    elif normalized in method_aliases:
        method = method_aliases[normalized]
        path = latest_pancreas_embedding_path(raw_dir, method, n_components)
        if path is None:
            raise FileNotFoundError(f"No saved {method} embedding can initialize this run.")
        embedding = load_pancreas_embedding(path, cell_ids=inputs.cell_ids)
        description = f"latest {method} embedding ({path})"
    else:
        path = resolve_pancreas_embedding_path(value, raw_dir)
        embedding = load_pancreas_embedding(path, cell_ids=inputs.cell_ids)
        description = f"custom embedding ({path})"

    embedding = promote_embedding(embedding, n_components)
    if embedding.shape[0] != len(inputs.cell_ids):
        raise ValueError(
            f"Initialization has {embedding.shape[0]} cells, expected {len(inputs.cell_ids)}."
        )
    return embedding, description


def promote_embedding(embedding, n_components):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError("Initialization must be a 2D array.")
    if embedding.shape[1] > n_components:
        raise ValueError(
            f"A {embedding.shape[1]}D embedding cannot initialize a {n_components}D run."
        )
    if embedding.shape[1] == n_components:
        return embedding
    return np.column_stack(
        (embedding, np.zeros((len(embedding), n_components - embedding.shape[1])))
    )


def save_pancreas_plot(embedding, *, labels, title, path, seed):
    if embedding.shape[1] == 2:
        figure, _ = plot_categorical_embedding(
            embedding,
            labels=labels,
            title=title,
            save_path=path,
            cmap=PANCREAS_CLUSTER_COLORS,
        )
    elif embedding.shape[1] == 3:
        figure, _ = plot_3d_embedding_views(
            embedding,
            labels=labels,
            title=title,
            save_path=path,
            point_fraction=1.0,
            random_state=seed,
            cmap=PANCREAS_CLUSTER_COLORS,
        )
    else:
        raise ValueError("Only 2D and 3D pancreas embeddings can be plotted.")
    plt.close(figure)


def merged_config(overrides):
    config = deepcopy(CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(target, updates):
    for key, value in updates.items():
        if key not in target:
            raise KeyError(f"Unknown main_pancreas option: {key!r}.")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Override {key!r} must be a dictionary.")
            _deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    main_pancreas()
