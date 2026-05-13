"""RNA-velocity adapters for generic asymmetry-preservation metrics."""

from __future__ import annotations

import numpy as np

from ..asymmetry import (
    asymmetry_score,
    neighbor_pairs,
    summarize_asymmetry_preservation,
)


__all__ = [
    "velocity_field_asymmetry_preservation_from_neighbors",
    "velocity_field_asymmetry_preservation_from_pairs",
    "velocity_field_pair_costs",
]


def velocity_field_asymmetry_preservation_from_neighbors(
        data_dissimilarities,
        embedding,
        velocity_vectors,
        neighbor_indices,
        *,
        alpha=1.0,
        tau=0.02,
        unique_pairs=True,
        eps=1e-12,
):
    """Compare data asymmetry to an embedding plus projected velocity field.

    This is intended for baselines such as UMAP + scVelo projected velocities.
    The embedding-side directed cost on a pair ``i -> j`` is

    ``||y_j - y_i|| * exp(-alpha * cos(v_i, y_j - y_i))``.
    """
    sources, targets = neighbor_pairs(neighbor_indices, unique_pairs=unique_pairs)
    return velocity_field_asymmetry_preservation_from_pairs(
        data_dissimilarities,
        embedding,
        velocity_vectors,
        sources,
        targets,
        alpha=alpha,
        tau=tau,
        eps=eps,
    )


def velocity_field_asymmetry_preservation_from_pairs(
        data_dissimilarities,
        embedding,
        velocity_vectors,
        sources,
        targets,
        *,
        alpha=1.0,
        tau=0.02,
        eps=1e-12,
):
    """Compare data asymmetry to velocity-field costs on explicit pairs."""
    D = np.asarray(data_dissimilarities, dtype=float)
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    data_asymmetry = asymmetry_score(D[sources, targets], D[targets, sources], eps=eps)
    forward, backward = velocity_field_pair_costs(
        embedding,
        velocity_vectors,
        sources,
        targets,
        alpha=alpha,
        eps=eps,
    )
    embedding_asymmetry = asymmetry_score(forward, backward, eps=eps)
    return summarize_asymmetry_preservation(
        sources,
        targets,
        data_asymmetry,
        embedding_asymmetry,
        tau=tau,
    )


def velocity_field_pair_costs(
        embedding,
        velocity_vectors,
        sources,
        targets,
        *,
        alpha=1.0,
        eps=1e-12,
):
    """Return forward and reverse velocity-biased costs on embedding pairs."""
    X = np.asarray(embedding, dtype=float)
    V = np.asarray(velocity_vectors, dtype=float)
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    if X.shape != V.shape:
        raise ValueError("embedding and velocity_vectors must have the same shape.")
    if sources.shape != targets.shape:
        raise ValueError("sources and targets must have the same shape.")

    forward = _velocity_cost(X, V, sources, targets, alpha=alpha, eps=eps)
    backward = _velocity_cost(X, V, targets, sources, alpha=alpha, eps=eps)
    return forward, backward


def _velocity_cost(X, V, sources, targets, *, alpha, eps):
    displacements = X[targets] - X[sources]
    distances = np.linalg.norm(displacements, axis=1)
    velocity_norms = np.linalg.norm(V[sources], axis=1)
    denom = distances * velocity_norms
    cosine = np.divide(
        np.sum(V[sources] * displacements, axis=1),
        denom,
        out=np.zeros_like(distances, dtype=float),
        where=denom > eps,
    )
    return distances * np.exp(-float(alpha) * np.clip(cosine, -1.0, 1.0))
