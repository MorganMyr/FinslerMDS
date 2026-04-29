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

    options = {"maxiter": max_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

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
