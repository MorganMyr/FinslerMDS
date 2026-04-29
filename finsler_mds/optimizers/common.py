"""Shared helpers for Finsler-MDS optimizers."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from finsler_mds.metrics import AlphaBetaMetric


def validate_metric(metric):
    if not isinstance(metric, AlphaBetaMetric):
        raise TypeError("optimizer requires an AlphaBetaMetric instance.")
    return metric


def initial_embedding(dissimilarities, n_components, init, random_state):
    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)
    if init is None:
        X = random_state.uniform(size=n_samples * n_components)
        return X.reshape((n_samples, n_components))

    X = np.asarray(init, dtype=float).copy()
    if X.ndim != 2 or X.shape[0] != n_samples:
        raise ValueError(
            "init must have shape (n_samples, n_components), "
            f"got {X.shape} for n_samples={n_samples}."
        )
    return X


def prepare_weights_and_mask(dissimilarities, weight):
    D = np.asarray(dissimilarities, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")

    if weight is None:
        W = np.ones_like(D, dtype=float)
    else:
        W = np.asarray(weight, dtype=float)
        if W.shape != D.shape:
            raise ValueError("weight must have the same shape as dissimilarities.")

    active = W != 0
    if np.any(active & ~np.isfinite(D)):
        raise ValueError("Active dissimilarities must be finite.")
    W = np.where(active, W, 0.0)
    D = np.where(active, D, 0.0)
    return D, W


def normalized_stress_scale(raw_stress, dissimilarities, weight):
    active = weight != 0
    denom = np.sum(weight[active] * dissimilarities[active] ** 2)
    if denom <= 0:
        return np.inf, 0.0
    stress = np.sqrt(raw_stress / denom)
    if raw_stress <= 0:
        return stress, 0.0
    return stress, 1.0 / np.sqrt(raw_stress * denom)


__all__ = [
    "validate_metric",
    "initial_embedding",
    "prepare_weights_and_mask",
    "normalized_stress_scale",
]
