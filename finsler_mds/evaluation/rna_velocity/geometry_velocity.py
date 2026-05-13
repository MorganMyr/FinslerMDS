"""Geometry-induced transition directions for Finsler embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class GeometryVelocityResult:
    """Velocity-like vectors induced by the local Finsler geometry."""

    vectors: np.ndarray
    neighbor_indices: np.ndarray
    weights: np.ndarray
    phi: np.ndarray
    confidence: np.ndarray


def finsler_induced_velocity_field(
        embedding,
        metric,
        *,
        n_neighbors=20,
        beta=8.0,
        mode="soft",
        eps=1e-12,
):
    """Construct velocity-like vectors from preferred local metric directions.

    Candidate directions are the Euclidean k-nearest neighbors in the embedding.
    Their weight depends only on the angular cost profile ``phi(s)`` of the
    Finsler metric, not on distance. The local neighbor graph therefore provides
    the admissible tangent directions, while the metric decides which of these
    directions is preferred.

    For the default soft mode,

    ``v_i = sum_j softmax(-beta * phi_ij) * unit(x_j - x_i) / phi_ij``.

    The ``1 / phi`` factor gives faster preferred directions larger vector
    norm before averaging, so ambiguous downward branches partially cancel.
    """
    X = _validate_embedding(embedding)
    if not hasattr(metric, "phi"):
        raise TypeError("metric must expose a phi(s) method.")

    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    beta = float(beta)
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'.")

    neighbors = _embedding_neighbors(X, n_neighbors=n_neighbors)
    vectors = np.zeros_like(X)
    weights = np.zeros(neighbors.shape, dtype=float)
    phi_values = np.full(neighbors.shape, np.nan, dtype=float)

    for cell in range(len(X)):
        targets = neighbors[cell]
        valid_targets = targets[targets >= 0]
        if len(valid_targets) == 0:
            continue

        displacements = X[valid_targets] - X[cell]
        norms = np.linalg.norm(displacements, axis=1)
        nonzero = norms > eps
        if not np.any(nonzero):
            continue

        valid_targets = valid_targets[nonzero]
        displacements = displacements[nonzero]
        norms = norms[nonzero]
        unit = displacements / norms[:, None]
        s = unit[:, -1]
        phi = np.asarray(metric.phi(s), dtype=float)
        usable = np.isfinite(phi) & (phi > eps)
        if not np.any(usable):
            continue

        unit = unit[usable]
        phi = phi[usable]
        if mode == "hard":
            local_weights = np.zeros(len(phi), dtype=float)
            local_weights[int(np.argmin(phi))] = 1.0
        else:
            shifted = phi - np.min(phi)
            scores = np.exp(-beta * shifted)
            total = float(np.sum(scores))
            if total <= 0 or not np.isfinite(total):
                continue
            local_weights = scores / total

        candidate_vectors = unit / phi[:, None]
        velocity = local_weights @ candidate_vectors
        vectors[cell] = velocity

        keep_pos = np.flatnonzero(nonzero)[usable]
        weights[cell, keep_pos] = local_weights
        phi_values[cell, keep_pos] = phi

    confidence = np.linalg.norm(vectors, axis=1)
    return GeometryVelocityResult(
        vectors=vectors,
        neighbor_indices=neighbors,
        weights=weights,
        phi=phi_values,
        confidence=confidence,
    )


def _validate_embedding(embedding):
    X = np.asarray(embedding, dtype=float)
    if X.ndim != 2 or X.shape[1] not in {2, 3}:
        raise ValueError("embedding must have shape (n_samples, 2) or (n_samples, 3).")
    return X


def _embedding_neighbors(X, *, n_neighbors):
    if len(X) <= 1:
        return np.empty((len(X), 0), dtype=int)
    n_query = min(n_neighbors + 1, len(X))
    nbrs = NearestNeighbors(n_neighbors=n_query)
    nbrs.fit(X)
    raw = nbrs.kneighbors(X, return_distance=False)

    out = np.full((len(X), min(n_neighbors, len(X) - 1)), -1, dtype=int)
    for cell, row in enumerate(raw):
        row = row[row != cell][:out.shape[1]]
        out[cell, :len(row)] = row
    return out
