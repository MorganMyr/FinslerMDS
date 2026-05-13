"""Evaluation utilities for Finsler-MDS embeddings."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse.csgraph import dijkstra

from finsler_mds.metrics import AlphaBetaMetric
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph


def geodesic_embedding_stress(
        embedding,
        dissimilarities,
        *,
        metric,
        n_neighbors=10,
        weight=None,
        normalized_stress=False,
        neighbors_algorithm="auto",
        n_jobs=None,
        on_unreachable="inf",
        return_distances=False,
):
    """Evaluate full graph-geodesic stress for a fixed embedding.

    The embedding is converted to a symmetric kNN support graph. Directed edge
    lengths are then evaluated with ``metric`` and exact shortest-path distances
    are computed with Dijkstra. The returned stress uses all finite, nonzero
    weighted, off-diagonal target dissimilarities.
    """
    if not isinstance(metric, AlphaBetaMetric):
        raise TypeError("metric must be an AlphaBetaMetric instance.")
    X = np.asarray(embedding, dtype=float)
    if X.ndim != 2:
        raise ValueError("embedding must be a 2D array of shape (n_samples, n_components).")
    D = np.asarray(dissimilarities, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")
    if weight is None:
        W = np.ones_like(D, dtype=float)
    else:
        W = np.asarray(weight, dtype=float)
        if W.shape != D.shape:
            raise ValueError("weight must have the same shape as dissimilarities.")
    if np.any((W != 0) & ~np.isfinite(D)):
        raise ValueError("Active dissimilarities must be finite.")
    if D.shape[0] != X.shape[0]:
        raise ValueError(
            "embedding and dissimilarities disagree on n_samples: "
            f"{X.shape[0]} != {D.shape[0]}."
        )
    if on_unreachable not in {"inf", "raise", "warn_skip"}:
        raise ValueError("on_unreachable must be one of {'inf', 'raise', 'warn_skip'}.")

    support = symmetric_knn_graph(
        X,
        n_neighbors=n_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    )
    graph = metric_graph_from_support(X, support, metric)
    embedded = dijkstra(graph, directed=True, return_predecessors=False)

    active = (W != 0) & np.isfinite(D)
    np.fill_diagonal(active, False)
    unreachable = active & ~np.isfinite(embedded)
    if np.any(unreachable):
        n_unreachable = int(np.count_nonzero(unreachable))
        if on_unreachable == "raise":
            raise ValueError(
                "The embedding graph has unreachable active pairs during "
                f"geodesic stress evaluation ({n_unreachable} pairs). "
                "Increase n_neighbors or use on_unreachable='inf'/'warn_skip'."
            )
        if on_unreachable == "inf":
            stress = np.inf
            if return_distances:
                return stress, embedded
            return stress
        warnings.warn(
            "Skipping unreachable active pairs during geodesic stress "
            f"evaluation ({n_unreachable} pairs).",
            RuntimeWarning,
            stacklevel=2,
        )
        active = active & np.isfinite(embedded)

    denom = float(np.sum(W[active] * D[active] ** 2))
    residual = embedded[active] - D[active]
    raw_stress = float(np.sum(W[active] * residual ** 2))
    if normalized_stress:
        stress = np.sqrt(raw_stress / denom) if denom > 0 else np.inf
    else:
        stress = raw_stress

    if return_distances:
        return float(stress), embedded
    return float(stress)


evaluate_geodesic_stress = geodesic_embedding_stress


__all__ = [
    "geodesic_embedding_stress",
    "evaluate_geodesic_stress",
]
