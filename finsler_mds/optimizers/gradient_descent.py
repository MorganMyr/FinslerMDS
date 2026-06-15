"""Generic gradient-based optimizer for Finsler-MDS stress."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize

from finsler_mds.optimizers.common import (
    initial_embedding,
    normalized_stress_scale,
    prepare_weights_and_mask,
    validate_metric,
)
from finsler_mds.optimizers.path_frozen import (
    _cupy_metric_length_and_grad,
    _gpu_metric_supported,
    _load_cupy,
)


@dataclass(frozen=True)
class GradientDescentResult:
    embedding: np.ndarray
    stress: float
    n_iter: int
    optimizer_result: scipy.optimize.OptimizeResult


def _stress_and_grad(X_flat, *, shape, dissimilarities, weight, metric, normalized_stress):
    X = X_flat.reshape(shape)
    active = weight != 0
    embedded = metric.pairwise(X)
    embedded_active = np.where(active, embedded, 0.0)
    if not np.all(np.isfinite(embedded_active)):
        raise ValueError(
            "The metric produced non-finite embedded distances on active pairs. "
            "For direct gradient descent, use a finite metric variant, clipping, "
            "or a surrogate gradient outside forbidden directions."
        )

    residual = np.zeros_like(dissimilarities, dtype=float)
    residual[active] = weight[active] * (embedded[active] - dissimilarities[active])
    stress = np.sum(weight[active] * (embedded[active] - dissimilarities[active]) ** 2)

    if normalized_stress:
        stress, _ = normalized_stress_scale(stress, dissimilarities, weight)

    diff = X[None, :, :] - X[:, None, :]
    grad_u = metric.grad_u(diff)
    if not np.all(np.isfinite(np.where(active[..., None], grad_u, 0.0))):
        raise ValueError(
            "The metric produced non-finite gradients on active pairs. "
            "If using Matsumoto with forbidden directions, set a finite "
            "forbidden_grad_norm or use a convexified/clipped metric."
        )

    if normalized_stress:
        raw_stress = np.sum(weight[active] * (embedded[active] - dissimilarities[active]) ** 2)
        _, scale = normalized_stress_scale(raw_stress, dissimilarities, weight)
    else:
        scale = 2.0

    pair_grad = scale * residual[..., None] * grad_u
    grad = pair_grad.sum(axis=0) - pair_grad.sum(axis=1)
    return float(stress), grad.ravel()


def _resolve_gpu_backend(device, metric, verbose):
    if device not in {"cpu", "auto", "gpu", "cuda"}:
        raise ValueError("device must be one of {'cpu', 'auto', 'gpu', 'cuda'}.")
    if device == "cpu":
        return None
    if not _gpu_metric_supported(metric):
        message = (
            "gradient_descent GPU backend currently supports RandersMetric, "
            "MatsumotoMetric, ConvexifiedMatsumotoMetric, and ConvexifiedToblerMetric only."
        )
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise ValueError(message)

    cp, error = _load_cupy()
    if cp is None:
        message = f"gradient_descent GPU backend unavailable: {error}"
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise RuntimeError(message) from error

    if verbose:
        device_id = cp.cuda.Device().id
        device_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
        if hasattr(device_name, "decode"):
            device_name = device_name.decode()
        print(f"gradient_descent GPU backend enabled on CUDA device {device_id}: {device_name}")
    return cp


class _GpuDenseStressObjective:
    def __init__(
            self,
            cp,
            *,
            shape,
            dissimilarities,
            weight,
            metric,
            normalized_stress,
            block_size,
    ):
        self.cp = cp
        self.shape = shape
        self.metric = metric
        self.normalized_stress = normalized_stress
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("gpu_block_size must be positive.")
        self.D = cp.asarray(dissimilarities, dtype=cp.float64)
        self.W = cp.asarray(weight, dtype=cp.float64)
        if normalized_stress:
            active = self.W != 0
            self.denom = float(cp.asnumpy(cp.sum(self.W[active] * self.D[active] ** 2)))
        else:
            self.denom = None

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape), dtype=cp.float64)
        raw_stress = self._raw_stress(X)
        if self.normalized_stress:
            if self.denom is None or self.denom <= 0:
                return np.inf, np.zeros_like(X_flat)
            stress = np.sqrt(raw_stress / self.denom)
            scale = 0.0 if raw_stress <= 0 else 1.0 / np.sqrt(raw_stress * self.denom)
        else:
            stress = raw_stress
            scale = 2.0
        grad = self._grad(X, scale)
        return float(stress), cp.asnumpy(grad).ravel()

    def _raw_stress(self, X):
        cp = self.cp
        stress = cp.asarray(0.0, dtype=cp.float64)
        n_samples = X.shape[0]
        for start in range(0, n_samples, self.block_size):
            stop = min(start + self.block_size, n_samples)
            vectors = X[None, :, :] - X[start:stop, None, :]
            lengths, _ = _cupy_metric_length_and_grad(
                cp,
                vectors.reshape(-1, X.shape[1]),
                self.metric,
            )
            lengths = lengths.reshape(stop - start, n_samples)
            active_lengths = cp.where(self.W[start:stop] != 0, lengths, 0.0)
            if not cp.all(cp.isfinite(active_lengths)).item():
                raise ValueError(
                    "The metric produced non-finite embedded distances on active pairs."
                )
            residual = lengths - self.D[start:stop]
            stress += cp.sum(self.W[start:stop] * residual * residual)
        return float(cp.asnumpy(stress))

    def _grad(self, X, scale):
        cp = self.cp
        grad = cp.zeros_like(X)
        n_samples, n_components = X.shape
        for start in range(0, n_samples, self.block_size):
            stop = min(start + self.block_size, n_samples)
            vectors = X[None, :, :] - X[start:stop, None, :]
            lengths, grad_u = _cupy_metric_length_and_grad(
                cp,
                vectors.reshape(-1, n_components),
                self.metric,
            )
            block_shape = (stop - start, n_samples)
            lengths = lengths.reshape(block_shape)
            grad_u = grad_u.reshape(block_shape + (n_components,))
            active_grad = cp.where(self.W[start:stop, :, None] != 0, grad_u, 0.0)
            if not cp.all(cp.isfinite(active_grad)).item():
                raise ValueError(
                    "The metric produced non-finite gradients on active pairs."
                )
            residual = self.W[start:stop] * (lengths - self.D[start:stop])
            pair_grad = scale * residual[:, :, None] * grad_u
            grad += cp.sum(pair_grad, axis=0)
            grad[start:stop] -= cp.sum(pair_grad, axis=1)
        return grad


def gradient_descent(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-6,
    random_state=None,
    normalized_stress=False,
    weight=None,
    method="L-BFGS-B",
    optimizer_options=None,
    device="cpu",
    gpu_block_size=256,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with a generic gradient-based optimizer.

    The objective is the usual weighted pairwise stress
    ``sum_ij w_ij (F(X_j - X_i) - D_ij)^2``. Any metric implementing the
    ``AlphaBetaMetric`` interface can be used.
    """
    metric = validate_metric(metric)
    D, W = prepare_weights_and_mask(dissimilarities, weight)
    X0 = initial_embedding(D, n_components, init, random_state)
    shape = X0.shape
    gpu_backend = _resolve_gpu_backend(device, metric, verbose)

    options = {"maxiter": max_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    if gpu_backend is not None:
        objective = _GpuDenseStressObjective(
            gpu_backend,
            shape=shape,
            dissimilarities=D,
            weight=W,
            metric=metric,
            normalized_stress=normalized_stress,
            block_size=gpu_block_size,
        )
    else:
        def objective(x_flat):
            return _stress_and_grad(
                x_flat,
                shape=shape,
                dissimilarities=D,
                weight=W,
                metric=metric,
                normalized_stress=normalized_stress,
            )

    result = scipy.optimize.minimize(
        objective,
        X0.ravel(),
        jac=True,
        method=method,
        options=options,
    )

    X = result.x.reshape(shape)
    stress = float(result.fun)
    n_iter = int(getattr(result, "nit", max_iter))
    gd_result = GradientDescentResult(
        embedding=X,
        stress=stress,
        n_iter=n_iter,
        optimizer_result=result,
    )

    if return_result:
        return gd_result
    if return_n_iter:
        return gd_result.embedding, gd_result.stress, gd_result.n_iter
    return gd_result.embedding, gd_result.stress


def optimize_gradient_descent(*args, **kwargs):
    """Alias used by the higher-level API layer."""
    return gradient_descent(*args, **kwargs)


__all__ = [
    "GradientDescentResult",
    "gradient_descent",
    "optimize_gradient_descent",
]
