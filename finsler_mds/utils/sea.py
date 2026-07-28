"""Shared data and metric helpers for the Sea experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra

from finsler_mds import ConvexifiedMatsumotoMetric, MatsumotoMetric, RandersMetric
from .embedding_io import metric_alpha_tag
from .graph import symmetric_knn_graph


@dataclass(frozen=True)
class SeaInputs:
    X: np.ndarray
    randers_field: np.ndarray
    current_field: np.ndarray
    dissimilarities: np.ndarray
    predecessors: np.ndarray
    path: Path
    metadata: dict


def load_or_compute_sea_inputs(
        raw_dir,
        *,
        n_samples,
        seed,
        n_neighbors,
        data_metric,
        alpha_current,
        sea_length,
        sea_width,
        current_frequency,
):
    """Load one reusable Sea dataset and its directed graph distances."""
    metadata = {
        "dataset": "sea",
        "n_samples": int(n_samples),
        "seed": int(seed),
        "n_neighbors": int(n_neighbors),
        "data_metric": normalize_metric_name(data_metric),
        "alpha_current": float(alpha_current),
        "sea_length": float(sea_length),
        "sea_width": float(sea_width),
        "current_frequency": float(current_frequency),
    }
    path = sea_input_path(raw_dir, metadata)
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            saved = json.loads(str(cache["metadata_json"].item()))
            if saved == metadata:
                print(f"Loading Sea inputs: {path}")
                return SeaInputs(
                    X=np.asarray(cache["X"], dtype=float),
                    randers_field=np.asarray(cache["randers_field"], dtype=float),
                    current_field=np.asarray(cache["current_field"], dtype=float),
                    dissimilarities=np.asarray(cache["dissimilarities"], dtype=float),
                    predecessors=np.asarray(cache["predecessors"], dtype=int),
                    path=path,
                    metadata=saved,
                )

    X, randers_field, current_field = make_sea(
        n_samples,
        seed=seed,
        alpha_current=alpha_current,
        sea_length=sea_length,
        sea_width=sea_width,
        current_frequency=current_frequency,
    )
    dissimilarities, predecessors = sea_graph_distances(
        X,
        randers_field,
        metric=data_metric,
        n_neighbors=n_neighbors,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        randers_field=randers_field,
        current_field=current_field,
        dissimilarities=dissimilarities,
        predecessors=predecessors,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved Sea inputs: {path}")
    return SeaInputs(
        X, randers_field, current_field, dissimilarities, predecessors, path, metadata
    )


def load_sea_inputs(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        return SeaInputs(
            X=np.asarray(cache["X"], dtype=float),
            randers_field=np.asarray(cache["randers_field"], dtype=float),
            current_field=np.asarray(cache["current_field"], dtype=float),
            dissimilarities=np.asarray(cache["dissimilarities"], dtype=float),
            predecessors=np.asarray(cache["predecessors"], dtype=int),
            path=path,
            metadata=json.loads(str(cache["metadata_json"].item())),
        )


def make_sea(
        n_samples,
        *,
        seed,
        alpha_current,
        sea_length=10.0,
        sea_width=10.0,
        current_frequency=2.0,
):
    if float(alpha_current) < 0:
        raise ValueError("alpha_current must be non-negative.")
    rng = np.random.default_rng(seed)
    X = rng.random((int(n_samples), 2)) * (sea_length, sea_width)
    field = np.column_stack(
        (
            np.sin(current_frequency * X[:, 0]) + np.cos(current_frequency * X[:, 1]),
            np.cos(current_frequency * X[:, 0]) - np.sin(current_frequency * X[:, 1]),
        )
    )
    field /= max(float(np.linalg.norm(field, axis=1).max()), 1e-12)

    # The Randers/beta field enters the metric cost. The physical current points
    # in the opposite direction, so travelling with the current is cheaper.
    randers_field = float(alpha_current) * field
    current_field = -randers_field
    return X, randers_field, current_field


def sea_graph_distances(X, randers_field, *, metric, n_neighbors):
    """Compute shortest paths on the directed, pointwise Finsler k-NN graph."""
    metric = normalize_metric_name(metric)
    X = np.asarray(X, dtype=float)
    randers_field = np.asarray(randers_field, dtype=float)
    if metric == "randers" and np.linalg.norm(randers_field, axis=1).max() >= 1:
        raise ValueError("A Randers data metric requires alpha_current < 1.")
    support = symmetric_knn_graph(X, n_neighbors=n_neighbors, ensure_connected=True).tocoo()
    vectors = X[support.col] - X[support.row]
    lengths = np.linalg.norm(vectors, axis=1)
    beta = np.divide(
        np.einsum("ij,ij->i", randers_field[support.row], vectors),
        lengths,
        out=np.zeros_like(lengths),
        where=lengths > 1e-12,
    )
    if metric == "randers":
        weights = lengths * (1.0 + beta)
    elif metric == "matsumoto":
        denominator = 1.0 - beta
        weights = np.divide(
            lengths,
            denominator,
            out=np.full_like(lengths, np.inf),
            where=denominator > 0,
        )
    else:
        weights = lengths / (1.0 - beta)
        linear = beta > 0.5
        weights[linear] = 4.0 * lengths[linear] * beta[linear]

    finite = np.isfinite(weights) & (weights > 0)
    graph = scipy.sparse.csr_matrix(
        (weights[finite], (support.row[finite], support.col[finite])),
        shape=support.shape,
    )
    distances, predecessors = dijkstra(graph, directed=True, return_predecessors=True)
    if not np.all(np.isfinite(distances)):
        raise ValueError(
            "The directed Sea graph is not strongly connected for this metric/alpha. "
            "Lower alpha_current or increase n_neighbors."
        )
    return distances, predecessors


def make_metric(name, alpha):
    return {
        "randers": RandersMetric,
        "matsumoto": MatsumotoMetric,
        "convexified_matsumoto": ConvexifiedMatsumotoMetric,
    }[normalize_metric_name(name)](alpha=float(alpha))


def normalize_metric_name(name):
    key = str(name).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "r": "randers",
        "rand": "randers",
        "randers": "randers",
        "m": "matsumoto",
        "mats": "matsumoto",
        "matsumoto": "matsumoto",
        "cm": "convexified_matsumoto",
        "cmats": "convexified_matsumoto",
        "convexified_matsumoto": "convexified_matsumoto",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "metric must be 'randers', 'matsumoto', or 'convexified_matsumoto'."
        ) from exc


def normalize_optimizer(value):
    key = str(value).lower().replace("-", "_")
    aliases = {
        "smacof": "smacof",
        "smacof_randers": "smacof",
        "gd": "gradient_descent",
        "gradient_descent": "gradient_descent",
        "pf": "path_frozen",
        "path_frozen": "path_frozen",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "optimizer must be 'smacof', 'gradient_descent', or 'path_frozen'."
        ) from exc


def metric_tag(name, alpha):
    prefix = {
        "randers": "rand",
        "matsumoto": "mats",
        "convexified_matsumoto": "cmats",
    }[normalize_metric_name(name)]
    return f"{prefix}{metric_alpha_tag(alpha)}"


def sea_input_path(raw_dir, metadata):
    name = (
        f"sea_n{metadata['n_samples']}_s{metadata['seed']}_k{metadata['n_neighbors']}_"
        f"{metric_tag(metadata['data_metric'], metadata['alpha_current'])}.npz"
    )
    return Path(raw_dir) / name


def normalized_x(X):
    X = np.asarray(X, dtype=float)
    span = max(float(np.ptp(X[:, 0])), 1e-12)
    return (X[:, 0] - X[:, 0].min()) / span
