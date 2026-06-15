"""Local RNA-velocity alignment preservation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


__all__ = [
    "VelocityAlignmentPreservationResult",
    "velocity_alignment_cosines_from_pairs",
    "velocity_alignment_preservation_from_neighbors",
    "velocity_alignment_preservation_from_pairs",
]


@dataclass(frozen=True)
class VelocityAlignmentPreservationResult:
    """Preservation of local velocity/displacement cosine similarities."""

    spearman: float
    sign_accuracy: float
    n_pairs: int
    n_sign_pairs: int
    data_cosines: np.ndarray
    embedding_cosines: np.ndarray


def velocity_alignment_preservation_from_neighbors(
        data_embedding,
        data_velocity,
        embedding,
        embedding_velocity,
        neighbor_indices,
        *,
        eps=1e-12,
):
    """Compare local velocity/displacement cosines in data and embedding.

    Pairs are oriented: for each cell ``i`` and each fixed neighbor ``j`` from
    ``neighbor_indices``, compare ``cos(v_i, x_j - x_i)`` in the data space to
    ``cos(v_i^emb, y_j - y_i)`` in the embedding.
    """
    sources, targets = _oriented_neighbor_pairs(neighbor_indices)
    return velocity_alignment_preservation_from_pairs(
        data_embedding,
        data_velocity,
        embedding,
        embedding_velocity,
        sources,
        targets,
        eps=eps,
    )


def velocity_alignment_preservation_from_pairs(
        data_embedding,
        data_velocity,
        embedding,
        embedding_velocity,
        sources,
        targets,
        *,
        eps=1e-12,
):
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    if sources.shape != targets.shape:
        raise ValueError("sources and targets must have the same shape.")

    data_cosines = velocity_alignment_cosines_from_pairs(
        data_embedding,
        data_velocity,
        sources,
        targets,
        eps=eps,
    )
    embedding_cosines = velocity_alignment_cosines_from_pairs(
        embedding,
        embedding_velocity,
        sources,
        targets,
        eps=eps,
    )
    valid = np.isfinite(data_cosines) & np.isfinite(embedding_cosines)
    data_cosines = data_cosines[valid]
    embedding_cosines = embedding_cosines[valid]

    spearman = _safe_spearman(data_cosines, embedding_cosines)
    sign_mask = (np.abs(data_cosines) > eps) & (np.abs(embedding_cosines) > eps)
    if np.any(sign_mask):
        sign_accuracy = float(np.mean(np.sign(data_cosines[sign_mask]) == np.sign(embedding_cosines[sign_mask])))
    else:
        sign_accuracy = np.nan

    return VelocityAlignmentPreservationResult(
        spearman=spearman,
        sign_accuracy=sign_accuracy,
        n_pairs=int(len(data_cosines)),
        n_sign_pairs=int(np.count_nonzero(sign_mask)),
        data_cosines=data_cosines,
        embedding_cosines=embedding_cosines,
    )


def velocity_alignment_cosines_from_pairs(points, velocity_vectors, sources, targets, *, eps=1e-12):
    points = np.asarray(points, dtype=float)
    velocity_vectors = np.asarray(velocity_vectors, dtype=float)
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    if points.shape != velocity_vectors.shape:
        raise ValueError("points and velocity_vectors must have the same shape.")
    displacements = points[targets] - points[sources]
    velocities = velocity_vectors[sources]
    displacement_norms = np.linalg.norm(displacements, axis=1)
    velocity_norms = np.linalg.norm(velocities, axis=1)
    denom = displacement_norms * velocity_norms
    cosines = np.full(len(sources), np.nan, dtype=float)
    valid = denom > eps
    cosines[valid] = np.sum(velocities[valid] * displacements[valid], axis=1) / denom[valid]
    return np.clip(cosines, -1.0, 1.0)


def _oriented_neighbor_pairs(neighbor_indices):
    neighbors = np.asarray(neighbor_indices, dtype=int)
    if neighbors.ndim != 2:
        raise ValueError("neighbor_indices must be a 2D array.")
    sources = np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])
    targets = neighbors.reshape(-1)
    valid = targets >= 0
    return sources[valid], targets[valid]


def _safe_spearman(x, y):
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(stats.spearmanr(x, y).statistic)
