"""Visualize source-target paths on the synthetic sea-current map and embeddings."""

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
    utils,
)
from main_sea import alpha_tag, metric_abbrev, normalize_metric_name, sea_colors
from sea_datasets import add_obstacles_to_axis, normalize_sea_dataset_name, obstacles_from_array


def main_sea_paths():
    script_dir = Path(__file__).resolve().parent
    dataset_name = normalize_sea_dataset_name(os.environ.get("SEA_DATASET", "sea2"))
    dir_res = script_dir / "res" / dataset_name
    dir_fig = dir_res / "figures"
    dir_embeddings = dir_res / "embeddings"

    embedding_kind = os.environ.get("SEA_PATH_EMBEDDING", "path_frozen")
    metric_name = os.environ.get("SEA_PATH_METRIC", "matsumoto")
    embedding_variant = os.environ.get("SEA_PATH_VARIANT", "latest")
    if embedding_variant.lower() in {"", "none", "latest"}:
        embedding_variant = None
    path_direction = normalize_path_direction(os.environ.get("SEA_PATH_DIRECTION", "top_left_to_bottom_right"))
    tobler = {"a": 7.0, "b": 0.04}
    alpha_current = 0.8
    embedding_neighbors = 12
    current_neighbors = 10

    embedding_path = latest_embedding(
        dir_embeddings,
        embedding_pattern(
            embedding_kind,
            metric_name=metric_name,
            alpha_current=alpha_current,
            variant=embedding_variant,
        ),
    )
    payload = load_embedding_payload(embedding_path)
    X = payload["X"]
    embedding = payload["embedding"]
    randers_field = payload["randers_field"]
    current_field = payload["current_field"]
    target_predecessors = payload.get("target_predecessors")
    obstacles = obstacles_from_payload(payload)
    bounds = tuple(float(x) for x in payload["bounds"]) if "bounds" in payload else None
    alpha_embedding = float(payload["alpha_embedding"]) if "alpha_embedding" in payload else None
    tobler = tobler_params_from_payload(payload, fallback=tobler)

    source, target = corner_pair(X, direction=path_direction)
    print(f"Loaded {embedding_kind} embedding: {embedding_path}")
    print(f"Path direction: {path_direction}")
    print(f"Path source index={source}, target index={target}")

    current_path = current_map_shortest_path(
        X,
        randers_field,
        source=source,
        target=target,
        n_neighbors=current_neighbors,
        predecessors=target_predecessors,
    )
    print(f"Current-map path length in vertices: {len(current_path)}")

    if normalize_embedding_kind(embedding_kind) == "smacof":
        embedding_path_indices = np.asarray([source, target], dtype=int)
        embedding_path_label = "SMACOF direct chord"
        embedding_metric_label = metric_parameter_label("randers", alpha_embedding, tobler=tobler)
    else:
        metric = make_embedding_metric(metric_name, alpha_embedding, tobler=tobler)
        embedding_path_indices = embedding_geodesic_path(
            embedding,
            metric,
            source=source,
            target=target,
            n_neighbors=embedding_neighbors,
        )
        embedding_path_label = f"{display_embedding_kind(embedding_kind)} geodesic path"
        embedding_metric_label = metric_parameter_label(metric_name, alpha_embedding, tobler=tobler)
        print(f"Embedding geodesic path length in vertices: {len(embedding_path_indices)}")

    current_path_name, embedding_path_name = path_figure_names(
        embedding_kind,
        metric_name=metric_name,
        alpha_current=alpha_current,
        direction=path_direction,
    )
    plot_current_path(
        X,
        current_field,
        current_path=current_path,
        embedding_path=embedding_path_indices,
        embedding_path_label=embedding_path_label,
        source=source,
        target=target,
        title=f"Sea path comparison on current map, alpha={alpha_current:g}",
        save_path=dir_fig / current_path_name,
        obstacles=obstacles,
        bounds=bounds,
    )
    plot_embedding_paths(
        embedding,
        X,
        current_path=current_path,
        embedding_path=embedding_path_indices,
        source=source,
        target=target,
        embedding_path_label=embedding_path_label,
        title=(
            f"Sea {display_embedding_kind(embedding_kind)}: embedding path and current-map shortest path "
            f"(target alpha={alpha_current:g}, {embedding_metric_label})"
        ),
        save_path=dir_fig / embedding_path_name,
    )
    print(f"Saved path visualizations in: {dir_fig}")


def embedding_pattern(embedding_kind, *, metric_name, alpha_current, variant=None):
    kind = normalize_embedding_kind(embedding_kind)
    alpha = alpha_tag(alpha_current)
    if kind == "smacof":
        if variant:
            return [f"sea_a{alpha}_smacof_*_{variant}.npz", f"sea_a{alpha}_smacof_*.npz"]
        return f"sea_a{alpha}_smacof_*.npz"

    metric = normalize_metric_name(metric_name)
    patterns = []
    if variant:
        patterns.append(f"sea_a{alpha}_pf_{metric_abbrev(metric)}_*_{variant}.npz")
    patterns.append(f"sea_a{alpha}_pf_{metric_abbrev(metric)}_*.npz")
    if metric == "matsumoto":
        patterns.append(f"sea_a{alpha}_pf_cmats_*.npz")
    return patterns


def latest_embedding(directory, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        candidates = sorted(
            set(Path(directory).glob(pattern)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No embedding found for patterns {patterns!r} in {directory}.")


def load_embedding_payload(path):
    with np.load(path, allow_pickle=True) as data:
        required = ["embedding", "X", "randers_field", "current_field"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Saved sea embedding {path} is missing arrays: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def corner_pair(X, *, direction):
    X = np.asarray(X, dtype=float)
    x = normalize01(X[:, 0])
    y = normalize01(X[:, 1])
    top_left = int(np.argmin(x + (1.0 - y)))
    bottom_right = int(np.argmin((1.0 - x) + y))
    if direction == "top_left_to_bottom_right":
        source, target = top_left, bottom_right
    elif direction == "bottom_right_to_top_left":
        source, target = bottom_right, top_left
    else:  # pragma: no cover - guarded by normalize_path_direction
        raise RuntimeError(f"Unhandled path direction {direction!r}.")
    if source == target:
        raise ValueError("Could not select distinct top-left and bottom-right points.")
    return source, target


def normalize01(values):
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    if span <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def current_map_shortest_path(X, randers_field, *, source, target, n_neighbors, predecessors=None):
    if predecessors is not None:
        path = path_from_predecessors(np.asarray(predecessors)[source], source, target)
        if path is None:
            raise ValueError("The selected target is unreachable from the selected source in the saved current graph.")
        return np.asarray(path, dtype=int)

    _, predecessors = utils.compute_dist_matrix(
        X,
        n_neighbors=n_neighbors,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        randers_field=randers_field,
    )
    path = path_from_predecessors(predecessors[source], source, target)
    if path is None:
        raise ValueError("The selected target is unreachable from the selected source in the current map.")
    return np.asarray(path, dtype=int)


def embedding_geodesic_path(embedding, metric, *, source, target, n_neighbors):
    path = utils.geodesic_path_indices(
        embedding,
        source=source,
        target=target,
        metric=metric,
        n_neighbors=n_neighbors,
        directed=True,
    )
    if path is None:
        raise ValueError("The selected target is unreachable from the selected source in the embedding graph.")
    return np.asarray(path, dtype=int)


def path_from_predecessors(predecessors, source, target):
    if source == target:
        return [source]

    path = [target]
    current = int(target)
    while current != source:
        previous = int(predecessors[current])
        if previous < 0:
            return None
        path.append(previous)
        current = previous
    path.reverse()
    return path


def make_embedding_metric(metric_name, alpha, *, tobler):
    metric = normalize_metric_name(metric_name)
    if metric == "randers":
        if alpha is None:
            raise ValueError("Randers path visualization needs alpha_embedding in the saved embedding.")
        return RandersMetric(alpha=alpha)
    if metric == "matsumoto":
        if alpha is None:
            raise ValueError("Matsumoto path visualization needs alpha_embedding in the saved embedding.")
        return MatsumotoMetric(alpha=alpha)
    if metric == "convexified_matsumoto":
        if alpha is None:
            raise ValueError("Convexified Matsumoto path visualization needs alpha_embedding in the saved embedding.")
        return ConvexifiedMatsumotoMetric(alpha=alpha)
    if metric == "convexified_tobler":
        return ConvexifiedToblerMetric(a=tobler["a"], b=tobler["b"])
    raise ValueError(f"Unknown metric {metric_name!r}.")


def tobler_params_from_payload(payload, *, fallback):
    params = dict(fallback)
    if "tobler_a" in payload:
        params["a"] = float(payload["tobler_a"])
    if "tobler_b" in payload:
        params["b"] = float(payload["tobler_b"])
    return params


def obstacles_from_payload(payload):
    if "obstacles" not in payload:
        return ()
    values = np.asarray(payload["obstacles"], dtype=float)
    if values.size == 0:
        return ()
    return obstacles_from_array(values)


def plot_current_path(
    X,
    current_field,
    *,
    current_path,
    embedding_path,
    embedding_path_label,
    source,
    target,
    title,
    save_path,
    obstacles=(),
    bounds=None,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(
        X[:, 0],
        X[:, 1],
        current_field[:, 0],
        current_field[:, 1],
        color="0.25",
        angles="xy",
        scale_units="xy",
        scale=1.4,
        width=0.0032,
        alpha=0.65,
    )
    ax.plot(
        X[current_path, 0],
        X[current_path, 1],
        color="crimson",
        lw=2.8,
        label="current-map shortest path",
        zorder=20,
    )
    ax.plot(
        X[embedding_path, 0],
        X[embedding_path, 1],
        color="deepskyblue",
        lw=2.5,
        ls="--",
        label=embedding_path_label,
        zorder=25,
    )
    ax.scatter(X[source, 0], X[source, 1], c="black", s=120, lw=0, label="start", zorder=30)
    ax.scatter(X[target, 0], X[target, 1], c="gray", s=120, lw=0, label="target", zorder=30)
    add_obstacles_to_axis(ax, obstacles)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box" if bounds is not None else "datalim")
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_embedding_paths(
    embedding,
    X_original,
    *,
    current_path,
    embedding_path,
    source,
    target,
    embedding_path_label,
    title,
    save_path,
):
    embedding = np.asarray(embedding, dtype=float)
    colors = darker_colors(sea_colors(X_original), factor=0.55)
    if embedding.shape[1] != 3:
        raise ValueError(f"Expected a 3D embedding, got shape {embedding.shape}.")

    fig = plt.figure(figsize=(11, 9))
    views = [("front", 20, -60), ("side", 20, 30), ("top", 90, -90), ("diagonal", 35, 135)]
    for view_id, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_id + 1, projection="3d")
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=10, lw=0, alpha=0.75)
        plot_path_3d(
            ax,
            embedding,
            embedding_path,
            color="deepskyblue",
            lw=2.5,
            ls="--",
            label=embedding_path_label if view_id == 0 else None,
            zorder=30,
        )
        plot_path_3d(
            ax,
            embedding,
            current_path,
            color="crimson",
            lw=2.8,
            ls="-",
            label="current-map shortest path" if view_id == 0 else None,
            zorder=25,
        )
        ax.scatter(*embedding[source], c="black", s=80, lw=0, zorder=40)
        ax.scatter(*embedding[target], c="gray", s=80, lw=0, zorder=40)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name)
        ax.set_box_aspect([1, 1, 1])
        utils.set_axes_equal(ax)
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        ax.set_zlabel("Embedding 3")
        if view_id == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path)
    plt.close(fig)


def plot_path_3d(ax, embedding, path, *, color, lw, ls, label, zorder):
    path = np.asarray(path, dtype=int)
    ax.plot(
        embedding[path, 0],
        embedding[path, 1],
        embedding[path, 2],
        color=color,
        lw=lw,
        ls=ls,
        label=label,
        zorder=zorder,
    )


def darker_colors(colors, *, factor):
    colors = np.asarray(colors, dtype=float).copy()
    colors[:, :3] *= float(factor)
    return np.clip(colors, 0.0, 1.0)


def normalize_embedding_kind(embedding_kind):
    kind = str(embedding_kind).lower().replace("-", "_")
    aliases = {
        "smacof": "smacof",
        "randers_smacof": "smacof",
        "smacof_randers": "smacof",
        "path_frozen": "path_frozen",
        "pf": "path_frozen",
        "frozen_paths": "path_frozen",
    }
    if kind not in aliases:
        raise ValueError("embedding_kind must be one of {'smacof', 'path_frozen'}.")
    return aliases[kind]


def normalize_path_direction(direction):
    key = str(direction).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "top_left_to_bottom_right": "top_left_to_bottom_right",
        "tl_to_br": "top_left_to_bottom_right",
        "tl_br": "top_left_to_bottom_right",
        "forward": "top_left_to_bottom_right",
        "normal": "top_left_to_bottom_right",
        "bottom_right_to_top_left": "bottom_right_to_top_left",
        "br_to_tl": "bottom_right_to_top_left",
        "br_tl": "bottom_right_to_top_left",
        "reverse": "bottom_right_to_top_left",
        "reversed": "bottom_right_to_top_left",
    }
    if key not in aliases:
        raise ValueError(
            "SEA_PATH_DIRECTION must be one of "
            "{'top_left_to_bottom_right', 'bottom_right_to_top_left'}."
        )
    return aliases[key]


def direction_abbrev(direction):
    direction = normalize_path_direction(direction)
    return {
        "top_left_to_bottom_right": "tl_br",
        "bottom_right_to_top_left": "br_tl",
    }[direction]


def path_figure_names(embedding_kind, *, metric_name, alpha_current, direction):
    kind = normalize_embedding_kind(embedding_kind)
    alpha = alpha_tag(alpha_current)
    direction_tag = direction_abbrev(direction)
    if kind == "smacof":
        stem = f"path_a{alpha}_smacof_{direction_tag}"
        return f"{stem}_current.pdf", f"{stem}.pdf"
    metric = normalize_metric_name(metric_name)
    stem = f"path_a{alpha}_pf_{metric_abbrev(metric)}_{direction_tag}"
    return f"{stem}_current.pdf", f"{stem}.pdf"


def display_embedding_kind(embedding_kind):
    kind = normalize_embedding_kind(embedding_kind)
    return {"smacof": "SMACOF", "path_frozen": "Path-frozen"}[kind]


def display_metric_name(metric_name):
    return {
        "randers": "Randers",
        "matsumoto": "Matsumoto",
        "convexified_matsumoto": "Convexified Matsumoto",
        "convexified_tobler": "Convexified Tobler",
    }[normalize_metric_name(metric_name)]


def metric_parameter_label(metric_name, alpha, *, tobler):
    metric = normalize_metric_name(metric_name)
    if metric == "convexified_tobler":
        return f"{display_metric_name(metric)} a={tobler['a']:g}, b={tobler['b']:g}"
    if alpha is None:
        return display_metric_name(metric)
    return f"{display_metric_name(metric)} alpha={alpha:g}"


if __name__ == "__main__":
    main_sea_paths()
