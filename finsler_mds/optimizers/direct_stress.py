"""Direct-distance stress regularizers for geodesic optimizers."""

from __future__ import annotations

import numpy as np

from finsler_mds.optimizers.metric_kernels import cupy_metric_length_and_grad


def build_direct_stress_objective(
        *,
        D,
        W,
        shape,
        metric,
        mode,
        weight,
        margin,
        gpu_backend=None,
):
    weight = float(weight)
    mode = _validate_mode(mode)
    if mode == "none" or weight == 0.0:
        return None
    if weight < 0:
        raise ValueError("direct_stress_weight must be non-negative.")
    if margin <= 0:
        raise ValueError("direct_stress_margin must be positive.")

    cls = _GpuDirectStressObjective if gpu_backend is not None else _DirectStressObjective
    kwargs = {"cp": gpu_backend} if gpu_backend is not None else {}
    return cls(
        D=D,
        W=W,
        shape=shape,
        metric=metric,
        mode=mode,
        weight=weight,
        margin=float(margin),
        **kwargs,
    )


class _DirectStressObjective:
    def __init__(self, *, D, W, shape, metric, mode, weight, margin):
        self.shape = shape
        self.metric = metric
        self.mode = mode
        self.margin = margin

        active = (W != 0) & np.isfinite(D)
        np.fill_diagonal(active, False)
        self.sources, self.targets = np.nonzero(active)
        self.weights = (weight * W[self.sources, self.targets]).astype(float, copy=False)
        self.dissimilarities = D[self.sources, self.targets].astype(float, copy=False)
        self.n_pairs = len(self.sources)
        target = self._target_distance()
        self.denom = float(np.sum(self.weights * target**2))

    def _target_distance(self):
        if self.mode == "hinge":
            return self.margin * self.dissimilarities
        return self.dissimilarities

    def __call__(self, X_flat):
        X = X_flat.reshape(self.shape)
        grad = np.zeros_like(X)
        if self.n_pairs == 0:
            return 0.0, grad.ravel()

        vectors = X[self.targets] - X[self.sources]
        lengths = self.metric.length(vectors)
        edge_grads = self.metric.grad_u(vectors)
        if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(edge_grads)):
            raise ValueError("The metric produced non-finite direct stress lengths or gradients.")

        if self.mode == "mds":
            residual = lengths - self.dissimilarities
            coeff = 2.0 * self.weights * residual
            raw_stress = float(np.sum(self.weights * residual**2))
        else:
            residual = self.margin * self.dissimilarities - lengths
            active = residual > 0
            raw_stress = float(np.sum(self.weights[active] * residual[active] ** 2))
            coeff = np.zeros_like(residual)
            coeff[active] = -2.0 * self.weights[active] * residual[active]

        contrib = coeff[:, None] * edge_grads
        np.add.at(grad, self.sources, -contrib)
        np.add.at(grad, self.targets, contrib)
        return raw_stress, grad.ravel()


class _GpuDirectStressObjective:
    def __init__(self, *, cp, D, W, shape, metric, mode, weight, margin):
        self.cp = cp
        self.shape = shape
        self.metric = metric
        self.mode = mode
        self.margin = margin

        active = (W != 0) & np.isfinite(D)
        np.fill_diagonal(active, False)
        sources, targets = np.nonzero(active)
        weights = weight * W[sources, targets]
        dissimilarities = D[sources, targets]
        target = margin * dissimilarities if mode == "hinge" else dissimilarities

        self.sources = cp.asarray(sources, dtype=cp.int32)
        self.targets = cp.asarray(targets, dtype=cp.int32)
        self.weights = cp.asarray(weights, dtype=cp.float64)
        self.dissimilarities = cp.asarray(dissimilarities, dtype=cp.float64)
        self.n_pairs = len(sources)
        self.denom = float(np.sum(weights * target**2))

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape))
        grad = cp.zeros_like(X)
        if self.n_pairs == 0:
            return 0.0, cp.asnumpy(grad).ravel()

        vectors = X[self.targets] - X[self.sources]
        lengths, edge_grads = cupy_metric_length_and_grad(cp, vectors, self.metric)
        if not cp.all(cp.isfinite(lengths)).item() or not cp.all(cp.isfinite(edge_grads)).item():
            raise ValueError("The metric produced non-finite direct stress lengths or gradients.")

        if self.mode == "mds":
            residual = lengths - self.dissimilarities
            coeff = 2.0 * self.weights * residual
            raw_stress = cp.sum(self.weights * residual**2)
        else:
            residual = self.margin * self.dissimilarities - lengths
            active = residual > 0
            raw_stress = cp.sum(cp.where(active, self.weights * residual**2, 0.0))
            coeff = cp.where(active, -2.0 * self.weights * residual, 0.0)

        contrib = coeff[:, None] * edge_grads
        cp.add.at(grad, self.sources, -contrib)
        cp.add.at(grad, self.targets, contrib)
        return float(raw_stress.get()), cp.asnumpy(grad).ravel()


def _validate_mode(mode):
    if mode not in {"none", "mds", "hinge"}:
        raise ValueError("direct_stress_mode must be one of 'none', 'mds', or 'hinge'.")
    return mode


__all__ = ["build_direct_stress_objective"]
