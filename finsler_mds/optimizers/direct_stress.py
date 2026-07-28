"""Direct-distance objectives shared by Path-Frozen components."""

from __future__ import annotations

import numpy as np

from finsler_mds.optimizers.metric_kernels import cupy_metric_length_and_grad


def build_direct_pairs_objective(
    pairs,
    *,
    shape,
    metric,
    gpu_backend=None,
):
    """Build a direct-distance stress over an existing collection of pairs."""
    if pairs.n_pairs == 0:
        return None
    sources, targets, weights, dissimilarities = _flatten_pairs(pairs)
    cls = _GpuDirectPairsObjective if gpu_backend is not None else _DirectPairsObjective
    kwargs = {"cp": gpu_backend} if gpu_backend is not None else {}
    return cls(
        sources=sources,
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        shape=shape,
        metric=metric,
        **kwargs,
    )


def build_direct_stress_objective(
    *,
    D,
    W,
    shape,
    metric,
    weight,
    gpu_backend=None,
):
    """Build the optional all-pairs direct Finsler-MDS regularizer."""
    weight = float(weight)
    if weight < 0:
        raise ValueError("direct_stress_weight must be non-negative.")
    if weight == 0:
        return None

    active = (W != 0) & np.isfinite(D)
    np.fill_diagonal(active, False)
    sources, targets = np.nonzero(active)
    cls = _GpuDirectPairsObjective if gpu_backend is not None else _DirectPairsObjective
    kwargs = {"cp": gpu_backend} if gpu_backend is not None else {}
    return cls(
        sources=sources,
        targets=targets,
        weights=weight * W[sources, targets],
        dissimilarities=D[sources, targets],
        shape=shape,
        metric=metric,
        **kwargs,
    )


class _DirectPairsObjective:
    def __init__(
        self,
        *,
        sources,
        targets,
        weights,
        dissimilarities,
        shape,
        metric,
    ):
        self.sources = sources
        self.targets = targets
        self.weights = weights
        self.dissimilarities = dissimilarities
        self.shape = shape
        self.metric = metric

    def __call__(self, X_flat):
        X = X_flat.reshape(self.shape)
        vectors = X[self.targets] - X[self.sources]
        lengths = self.metric.length(vectors)
        pair_grads = self.metric.grad_u(vectors)
        if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(pair_grads)):
            raise ValueError("The metric produced non-finite direct-pair lengths or gradients.")

        residual = lengths - self.dissimilarities
        stress = float(np.sum(self.weights * residual**2))
        contributions = (2.0 * self.weights * residual)[:, None] * pair_grads
        grad = np.zeros_like(X)
        np.add.at(grad, self.sources, -contributions)
        np.add.at(grad, self.targets, contributions)
        return stress, grad.ravel()


class _GpuDirectPairsObjective:
    def __init__(
        self,
        *,
        cp,
        sources,
        targets,
        weights,
        dissimilarities,
        shape,
        metric,
    ):
        self.cp = cp
        self.sources = cp.asarray(sources, dtype=cp.int32)
        self.targets = cp.asarray(targets, dtype=cp.int32)
        self.weights = cp.asarray(weights, dtype=cp.float64)
        self.dissimilarities = cp.asarray(dissimilarities, dtype=cp.float64)
        self.shape = shape
        self.metric = metric

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape))
        vectors = X[self.targets] - X[self.sources]
        lengths, pair_grads = cupy_metric_length_and_grad(cp, vectors, self.metric)
        if not cp.all(cp.isfinite(lengths)).item() or not cp.all(cp.isfinite(pair_grads)).item():
            raise ValueError("The metric produced non-finite direct-pair lengths or gradients.")

        residual = lengths - self.dissimilarities
        stress = cp.sum(self.weights * residual**2)
        contributions = (2.0 * self.weights * residual)[:, None] * pair_grads
        grad = cp.zeros_like(X)
        cp.add.at(grad, self.sources, -contributions)
        cp.add.at(grad, self.targets, contributions)
        return float(stress.get()), cp.asnumpy(grad).ravel()


def _flatten_pairs(pairs):
    counts = np.fromiter((len(targets) for targets in pairs.targets), dtype=int)
    return (
        np.repeat(pairs.sources, counts).astype(int, copy=False),
        np.concatenate(pairs.targets).astype(int, copy=False),
        np.concatenate(pairs.weights).astype(float, copy=False),
        np.concatenate(pairs.dissimilarities).astype(float, copy=False),
    )


__all__ = ["build_direct_pairs_objective", "build_direct_stress_objective"]
