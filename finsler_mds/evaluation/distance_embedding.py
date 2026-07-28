"""Joint distance-based evaluation for Finsler-MDS embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse.csgraph import dijkstra

from finsler_mds.metrics import AlphaBetaMetric
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph

from .asymmetry import AsymmetryPreservationResult, asymmetry_score, summarize_asymmetry_preservation


@dataclass(frozen=True)
class StretchSummary:
    n_pairs: int
    mean: float
    std: float
    median: float
    q05: float
    q25: float
    q75: float
    q95: float
    min: float
    max: float
    ratios: np.ndarray


@dataclass(frozen=True)
class DistanceEmbeddingEvaluation:
    stress: float
    normalized_stress: float
    n_active_pairs: int
    n_unreachable_pairs: int
    stretch: StretchSummary
    asymmetry: AsymmetryPreservationResult
    data_distances: np.ndarray
    embedding_distances: np.ndarray
    active_mask: np.ndarray


def compute_embedding_distances(
        embedding,
        *,
        metric,
        mode="geodesic",
        n_neighbors=10,
        support_graph=None,
        neighbors_algorithm="auto",
        n_jobs=None,
):
    """Compute all-pairs directed distances in an embedding.

    ``mode="direct"`` uses the pairwise metric distance directly. ``mode="geodesic"``
    first builds a symmetric kNN support graph, evaluates directed metric edge
    lengths on it, then runs Dijkstra.
    """
    if not isinstance(metric, AlphaBetaMetric):
        raise TypeError("metric must be an AlphaBetaMetric instance.")
    X = np.asarray(embedding, dtype=float)
    if X.ndim != 2:
        raise ValueError("embedding must be a 2D array.")

    mode = _normalize_distance_mode(mode)
    if mode == "direct":
        return metric.pairwise(X)

    if support_graph is None:
        support_graph = symmetric_knn_graph(
            X,
            n_neighbors=n_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )
    graph = metric_graph_from_support(X, support_graph, metric)
    return dijkstra(graph, directed=True, return_predecessors=False)


def evaluate_distance_embedding(
        data_distances,
        embedding,
        *,
        metric,
        mode="geodesic",
        n_neighbors=10,
        support_graph=None,
        weight=None,
        pairs=None,
        asymmetry_tau=0.02,
        on_unreachable="warn_skip",
        neighbors_algorithm="auto",
        n_jobs=None,
        return_distances=True,
):
    """Evaluate stress, stretch ratios, and asymmetry after one distance pass."""
    embedding_distances = compute_embedding_distances(
        embedding,
        metric=metric,
        mode=mode,
        n_neighbors=n_neighbors,
        support_graph=support_graph,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    )
    return evaluate_precomputed_embedding_distances(
        data_distances,
        embedding_distances,
        weight=weight,
        pairs=pairs,
        asymmetry_tau=asymmetry_tau,
        on_unreachable=on_unreachable,
        return_distances=return_distances,
    )


def evaluate_precomputed_embedding_distances(
        data_distances,
        embedding_distances,
        *,
        weight=None,
        pairs=None,
        asymmetry_tau=0.02,
        on_unreachable="warn_skip",
        return_distances=True,
):
    """Evaluate metrics from already computed all-pairs embedding distances."""
    D = _validate_square_matrix(data_distances, "data_distances")
    E = _validate_square_matrix(embedding_distances, "embedding_distances")
    if D.shape != E.shape:
        raise ValueError(f"data and embedding distances have different shapes: {D.shape} != {E.shape}.")

    W = np.ones_like(D, dtype=float) if weight is None else np.asarray(weight, dtype=float)
    if W.shape != D.shape:
        raise ValueError("weight must have the same shape as data_distances.")
    if on_unreachable not in {"inf", "raise", "warn_skip"}:
        raise ValueError("on_unreachable must be one of {'inf', 'raise', 'warn_skip'}.")

    active = _active_pair_mask(D, W, pairs=pairs)
    unreachable = active & ~np.isfinite(E)
    n_unreachable = int(np.count_nonzero(unreachable))
    if n_unreachable:
        if on_unreachable == "raise":
            raise ValueError(f"Embedding distances contain {n_unreachable} unreachable active pairs.")
        if on_unreachable == "inf":
            stress = normalized = np.inf
        else:
            warnings.warn(
                f"Skipping {n_unreachable} unreachable active pairs during embedding evaluation.",
                RuntimeWarning,
                stacklevel=2,
            )
            active = active & np.isfinite(E)
            stress, normalized = stress_from_distance_matrices(D, E, weight=W, active_mask=active)
    else:
        stress, normalized = stress_from_distance_matrices(D, E, weight=W, active_mask=active)

    sources, targets = np.nonzero(active)
    stretch = stretch_summary(D[sources, targets], E[sources, targets])
    data_asymmetry = asymmetry_score(D[sources, targets], D[targets, sources])
    emb_asymmetry = asymmetry_score(E[sources, targets], E[targets, sources])
    asymmetry = summarize_asymmetry_preservation(
        sources,
        targets,
        data_asymmetry,
        emb_asymmetry,
        tau=asymmetry_tau,
    )

    if not return_distances:
        D_out = np.empty((0, 0), dtype=float)
        E_out = np.empty((0, 0), dtype=float)
    else:
        D_out = D
        E_out = E

    return DistanceEmbeddingEvaluation(
        stress=float(stress),
        normalized_stress=float(normalized),
        n_active_pairs=int(np.count_nonzero(active)),
        n_unreachable_pairs=n_unreachable,
        stretch=stretch,
        asymmetry=asymmetry,
        data_distances=D_out,
        embedding_distances=E_out,
        active_mask=active,
    )


def stress_from_distance_matrices(data_distances, embedding_distances, *, weight=None, active_mask=None):
    D = np.asarray(data_distances, dtype=float)
    E = np.asarray(embedding_distances, dtype=float)
    W = np.ones_like(D, dtype=float) if weight is None else np.asarray(weight, dtype=float)
    active = _active_pair_mask(D, W) if active_mask is None else np.asarray(active_mask, dtype=bool)
    residual = E[active] - D[active]
    stress = float(np.sum(W[active] * residual**2))
    denom = float(np.sum(W[active] * D[active] ** 2))
    normalized = float(np.sqrt(stress / denom)) if denom > 0 else np.inf
    return stress, normalized


def stretch_summary(data_forward, embedding_forward, *, eps=1e-12):
    data_forward = np.asarray(data_forward, dtype=float)
    embedding_forward = np.asarray(embedding_forward, dtype=float)
    ratios = np.divide(
        embedding_forward,
        data_forward,
        out=np.full_like(embedding_forward, np.nan, dtype=float),
        where=np.isfinite(data_forward) & np.isfinite(embedding_forward) & (data_forward > eps),
    )
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return StretchSummary(0, *(np.nan,) * 9, ratios)
    q05, q25, q75, q95 = np.quantile(ratios, [0.05, 0.25, 0.75, 0.95])
    return StretchSummary(
        n_pairs=int(len(ratios)),
        mean=float(np.mean(ratios)),
        std=float(np.std(ratios)),
        median=float(np.median(ratios)),
        q05=float(q05),
        q25=float(q25),
        q75=float(q75),
        q95=float(q95),
        min=float(np.min(ratios)),
        max=float(np.max(ratios)),
        ratios=ratios,
    )


def _active_pair_mask(data_distances, weight, *, pairs=None):
    D = np.asarray(data_distances, dtype=float)
    W = np.asarray(weight, dtype=float)
    active = (W != 0) & np.isfinite(D)
    np.fill_diagonal(active, False)
    if pairs is None:
        return active

    pair_mask = np.zeros_like(active, dtype=bool)
    sources, targets = pairs
    pair_mask[np.asarray(sources, dtype=int), np.asarray(targets, dtype=int)] = True
    return active & pair_mask


def _validate_square_matrix(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return arr


def _normalize_distance_mode(mode):
    key = str(mode).lower().replace("-", "_")
    aliases = {
        "direct": "direct",
        "pairwise": "direct",
        "smacof": "direct",
        "gd": "direct",
        "geodesic": "geodesic",
        "path_frozen": "geodesic",
    }
    if key not in aliases:
        raise ValueError("mode must be one of {'direct', 'geodesic'}.")
    return aliases[key]


__all__ = [
    "DistanceEmbeddingEvaluation",
    "StretchSummary",
    "compute_embedding_distances",
    "evaluate_distance_embedding",
    "evaluate_precomputed_embedding_distances",
    "stress_from_distance_matrices",
    "stretch_summary",
]
