"""Embedding orientation helpers."""

from __future__ import annotations

import numpy as np


def rotate_embedding_to_mean_velocity_down(embedding, velocity_vectors):
    """Rotate a 2D/3D embedding so the mean projected velocity points downward.

    Down means negative y in 2D and negative z in 3D.  The same orthogonal
    rotation is applied to the embedding; translations are left unchanged.
    """
    embedding = np.asarray(embedding, dtype=float)
    velocity_vectors = np.asarray(velocity_vectors, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] not in {2, 3}:
        raise ValueError("embedding must have shape (n_samples, 2) or (n_samples, 3).")
    if velocity_vectors.shape != embedding.shape:
        raise ValueError("velocity_vectors must have the same shape as embedding.")

    mean_velocity = np.nanmean(velocity_vectors, axis=0)
    rotation = rotation_to_down_axis(mean_velocity)
    return embedding @ rotation.T, rotation, mean_velocity


def rotation_to_down_axis(vector):
    """Return an orthogonal matrix mapping vector to the downward axis."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or vector.size not in {2, 3}:
        raise ValueError("vector must be 2D or 3D.")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.eye(vector.size)

    source = vector / norm
    target = np.zeros_like(source)
    target[-1] = -1.0
    if source.size == 2:
        return _rotation_2d(source, target)
    return _rotation_3d(source, target)


def _rotation_2d(source, target):
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = float(source[0] * target[1] - source[1] * target[0])
    angle = np.arctan2(cross, dot)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def _rotation_3d(source, target):
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        axis = _orthogonal_unit_vector(source)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    axis = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _orthogonal_unit_vector(vector):
    basis = np.eye(vector.size)
    axis = basis[np.argmin(np.abs(vector))]
    axis = axis - vector * np.dot(axis, vector)
    return axis / np.linalg.norm(axis)
