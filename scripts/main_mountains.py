"""Embed geodesic distances measured on a three-mountain surface."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
    fit_finsler_mds,
    utils,
)
from finsler_mds.utils.embedding_io import adapt_embedding_dimension, metric_alpha_tag


# Main choices -----------------------------------------------------------------
OPTIMIZER = "path_frozen"  # "smacof", "gradient_descent", or "path_frozen"
FINSLER_METRIC = "randers"  # "randers", "matsumoto", or "convexified_matsumoto"
INIT = "terrain"  # "terrain" or "latest"
N_COMPONENTS = 3  # 2 or 3
TARGET_ALPHA = 1.0
FINSLER_ALPHA = 0.9

SEED = 42
TARGET_GRAPH_NEIGHBORS = 10
GRID = {
    "nx": 60,
    "ny": 40,
    "xlim": (-15.0, 15.0),
    "ylim": (-10.0, 10.0),
    "xy_noise": 0.08,
}
MOUNTAINS = {
    "left": {"center": (-10.0, 0.0), "height": 4.0, "sigma": (2.2, 2.2)},
    "middle": {"center": (0.0, 0.0), "height": 4.0, "sigma": (0.8, 6.0)},
    "right": {"center": (10.0, 0.0), "height": 4.0, "sigma": (2.2, 2.2)},
}
OPTIMIZER_OPTIONS = {
    "smacof": {
        "max_iter": 20,
        "project_on_V": True,
        "check_monotony": False,
        "device": "auto",
        "verbose": 1,
    },
    "gradient_descent": {
        "max_iter": 300,
        "eps": 1e-6,
        "optimizer_options": {"ftol": 1e-9, "maxls": 40},
        "device": "auto",
        "verbose": 0,
    },
    "path_frozen": {
        "graph_neighbors": 12,
        "outer_iter": 20,
        "inner_iter": 10,
        "n_landmark": 150,
        "random_landmark_fraction": 1.0,
        "n_local_pairs": 12,
        "targets_per_landmark": 300,
        "local_weight": 1.0,
        "direct_stress_weight": 0.0,
        "outer_step_size": 1.0,
        "device": "auto",
        "verbose": 1,
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "res" / "mountains"
FIGURE_DIR = RESULT_DIR / "figures"
EMBEDDING_DIR = RESULT_DIR / "embeddings"

METRICS = {
    "randers": RandersMetric,
    "matsumoto": MatsumotoMetric,
    "convexified_matsumoto": ConvexifiedMatsumotoMetric,
}
METRIC_TAGS = {
    "randers": "r",
    "matsumoto": "mats",
    "convexified_matsumoto": "cmats",
}
OPTIMIZER_TAGS = {"smacof": "smacof", "gradient_descent": "gd", "path_frozen": "pf"}


def main_mountains():
    if OPTIMIZER not in OPTIMIZER_OPTIONS:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER!r}")
    if FINSLER_METRIC not in METRICS:
        raise ValueError(f"Unknown Finsler metric: {FINSLER_METRIC!r}")
    if N_COMPONENTS not in (2, 3):
        raise ValueError("N_COMPONENTS must be 2 or 3.")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    data = make_mountain_surface(np.random.default_rng(SEED))
    source, target = mountain_summit_pair(data)
    target_metric = ConvexifiedMatsumotoMetric(TARGET_ALPHA)
    print("Building Convexified Matsumoto geodesic target distances")
    dissimilarities, predecessors = utils.compute_metric_dist_matrix(
        data,
        target_metric,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        directed=True,
    )
    if not np.all(np.isfinite(dissimilarities)):
        raise RuntimeError("The target kNN graph is not connected.")
    np.fill_diagonal(dissimilarities, 0.0)
    surface_path = checked_path(utils.path_from_predecessors(predecessors, source, target))

    embedding_metric_name = "randers" if OPTIMIZER == "smacof" else FINSLER_METRIC
    embedding_metric = METRICS[embedding_metric_name](FINSLER_ALPHA)
    stem = result_stem(embedding_metric_name)
    init = load_initialization(data, stem)
    fit_options = {
        "optimizer": OPTIMIZER,
        "metric": embedding_metric,
        "init": init,
        "n_components": N_COMPONENTS,
        "random_state": SEED,
        **OPTIMIZER_OPTIONS[OPTIMIZER],
    }
    if OPTIMIZER == "smacof":
        fit_options["n_init"] = 1

    print(f"Running {OPTIMIZER} with {embedding_metric_name}, alpha={FINSLER_ALPHA:g}")
    embedding, objective = fit_finsler_mds(dissimilarities, print_time=True, **fit_options)
    np.savez(EMBEDDING_DIR / f"{stem}.npz", embedding=embedding, objective=objective)

    embedding_path = (
        np.asarray([source, target])
        if OPTIMIZER in {"smacof", "gradient_descent"}
        else checked_path(
            utils.geodesic_path_indices(
                embedding,
                source,
                target,
                embedding_metric,
                n_neighbors=OPTIMIZER_OPTIONS["path_frozen"]["graph_neighbors"],
                directed=True,
            )
        )
    )
    plot_paths(
        data,
        surface_colors(data),
        [(surface_path, "surface geodesic", "crimson")],
        "Three-mountain surface",
        FIGURE_DIR / f"surface_cmats{metric_alpha_tag(TARGET_ALPHA)}.pdf",
    )
    plot_paths(
        embedding,
        surface_colors(data),
        [
            (surface_path, "target-space path", "crimson"),
            (embedding_path, "embedding path", "deepskyblue"),
        ],
        f"{OPTIMIZER} — {embedding_metric_name}",
        FIGURE_DIR / f"{stem}.pdf",
    )
    print(f"Objective: {objective:.6g}")
    print(f"Saved results in {RESULT_DIR}")


def make_mountain_surface(rng):
    xs = np.linspace(*GRID["xlim"], GRID["nx"])
    ys = np.linspace(*GRID["ylim"], GRID["ny"])
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    xy += rng.normal(scale=GRID["xy_noise"], size=xy.shape)
    z = sum(gaussian_bump(xy, **parameters) for parameters in MOUNTAINS.values())
    return np.column_stack([xy, z])


def gaussian_bump(xy, *, center, height, sigma):
    scaled = (xy - np.asarray(center)) / np.asarray(sigma)
    return height * np.exp(-0.5 * np.sum(scaled**2, axis=1))


def mountain_summit_pair(data):
    return tuple(
        int(np.argmin(np.linalg.norm(data[:, :2] - MOUNTAINS[name]["center"], axis=1)))
        for name in ("left", "right")
    )


def result_stem(metric_name):
    return "_".join(
        [
            OPTIMIZER_TAGS[OPTIMIZER],
            f"{N_COMPONENTS}d",
            f"cmats{metric_alpha_tag(TARGET_ALPHA)}",
            f"{METRIC_TAGS[metric_name]}{metric_alpha_tag(FINSLER_ALPHA)}",
        ]
    )


def load_initialization(data, stem):
    terrain_init = adapt_embedding_dimension(data - data.mean(axis=0), N_COMPONENTS)
    if INIT == "terrain":
        return terrain_init
    if INIT != "latest":
        raise ValueError("INIT must be 'terrain' or 'latest'.")

    prefix = "_".join(stem.split("_")[:2])
    candidates = sorted(
        EMBEDDING_DIR.glob(f"{prefix}_*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No saved {prefix} embedding found in {EMBEDDING_DIR}.")
    with np.load(candidates[0]) as saved:
        init = adapt_embedding_dimension(saved["embedding"], N_COMPONENTS)
    if init.shape != terrain_init.shape:
        raise ValueError(f"Incompatible initialization shape in {candidates[0]}: {init.shape}")
    print(f"Loaded initialization from {candidates[0]}")
    return init


def checked_path(path):
    if path is None or len(path) == 0:
        raise RuntimeError("No directed path connects the selected mountain summits.")
    return np.asarray(path, dtype=int)


def surface_colors(data):
    height = data[:, 2]
    return plt.cm.terrain((height - height.min()) / max(np.ptp(height), 1e-12))


def plot_paths(points, colors, paths, title, save_path):
    points = np.asarray(points)
    views = [
        ("front", 20, -60),
        ("side", 20, 30),
        ("top", 90, -90),
        ("diagonal", 35, 135),
    ]
    if points.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        axes = [ax]
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=16, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
    else:
        fig = plt.figure(figsize=(11, 9))
        axes = []
        for index, (name, elevation, azimuth) in enumerate(views, start=1):
            ax = fig.add_subplot(2, 2, index, projection="3d")
            axes.append(ax)
            ax.scatter(*points.T, c=colors, s=16, linewidths=0)
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_title(name)
            utils.set_axes_equal(ax)

    for ax in axes:
        for path, label, color in paths:
            utils.add_index_path(ax, points, path, color=color, label=label)
        ax.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    main_mountains()
