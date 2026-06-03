"""Gap-distance metrics for trajectory embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist, pdist


@dataclass(frozen=True)
class GapDistanceResult:
    distance: float
    normalized_distance: float
    before_representative: int
    after_representative: int
    max_pairwise_distance: float
    n_before: int
    n_after: int


def normalized_gap_distance(
        embedding,
        before_indices,
        after_indices,
        *,
        representative="medoid",
        eps=1e-12,
):
    """Distance between before/after trajectory-gap populations.

    The value is normalized by the maximum pairwise distance in the embedding,
    matching the VeloViz gap-distance convention.
    """
    X = np.asarray(embedding, dtype=float)
    if X.ndim != 2:
        raise ValueError("embedding must be a 2D array.")
    before = _validate_indices(before_indices, len(X), "before_indices")
    after = _validate_indices(after_indices, len(X), "after_indices")
    if before.size == 0 or after.size == 0:
        raise ValueError("before_indices and after_indices must both be non-empty.")

    before_rep = _representative_index(X, before, representative)
    after_rep = _representative_index(X, after, representative)
    distance = float(np.linalg.norm(X[before_rep] - X[after_rep]))
    max_distance = _max_pairwise_distance(X)
    normalized = distance / max(max_distance, eps)
    return GapDistanceResult(
        distance=distance,
        normalized_distance=float(normalized),
        before_representative=int(before_rep),
        after_representative=int(after_rep),
        max_pairwise_distance=float(max_distance),
        n_before=int(before.size),
        n_after=int(after.size),
    )


def _validate_indices(indices, n_samples, name):
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if indices.size and (np.min(indices) < 0 or np.max(indices) >= n_samples):
        raise ValueError(f"{name} contains indices outside [0, {n_samples}).")
    return np.unique(indices)


def _representative_index(X, indices, representative):
    if representative == "medoid":
        distances = cdist(X[indices], X[indices])
        return int(indices[np.argmin(distances.sum(axis=1))])
    if representative == "centroid":
        centroid = np.mean(X[indices], axis=0, keepdims=True)
        distances = cdist(X[indices], centroid).ravel()
        return int(indices[np.argmin(distances)])
    raise ValueError("representative must be one of {'medoid', 'centroid'}.")


def _max_pairwise_distance(X):
    if len(X) < 2:
        return 0.0
    return float(np.max(pdist(X)))
