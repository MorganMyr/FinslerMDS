"""Directed Finsler-UMAP optimizer.

This optimizer builds a directed fuzzy kNN graph from an asymmetric
dissimilarity matrix, then optimizes a UMAP-like cross-entropy where embedded
edge probabilities are computed from a Finsler metric.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import find_ab_params, make_epochs_per_sample

from finsler_mds.metrics import ConvexifiedMatsumotoMetric, MatsumotoMetric, RandersMetric
from finsler_mds.optimizers.common import initial_embedding, validate_metric
from finsler_mds.utils.umap import (
    directed_fuzzy_graph_from_dense,
    spectral_initial_embedding,
)

_EPS = 1e-12
_METRIC_RANDERS = 0
_METRIC_MATSUMOTO = 1
_METRIC_CONVEXIFIED_MATSUMOTO = 2


def _metric_params(metric):
    if isinstance(metric, RandersMetric):
        return _METRIC_RANDERS, float(metric.alpha)
    if isinstance(metric, MatsumotoMetric):
        if metric.max_phi is not None or metric.forbidden_grad_norm is not None:
            raise ValueError(
                "finsler_umap only supports plain MatsumotoMetric "
                "(max_phi=None and forbidden_grad_norm=None)."
            )
        return _METRIC_MATSUMOTO, float(metric.alpha)
    if isinstance(metric, ConvexifiedMatsumotoMetric):
        return _METRIC_CONVEXIFIED_MATSUMOTO, float(metric.alpha)
    raise ValueError(
        "finsler_umap supports RandersMetric, MatsumotoMetric, "
        "and ConvexifiedMatsumotoMetric."
    )


@nb.njit(cache=True)
def _clip(value, bound):
    if bound >= 0.0:
        if value > bound:
            return bound
        if value < -bound:
            return -bound
    return value


@nb.njit(cache=True)
def _length_grad(X, src, dst, metric_kind, metric_alpha, grad_u):
    n_dim = X.shape[1]
    r2 = 0.0
    for d in range(n_dim):
        u = X[dst, d] - X[src, d]
        grad_u[d] = u
        r2 += u * u
    r = math.sqrt(r2)
    if r <= _EPS:
        for d in range(n_dim):
            grad_u[d] = 0.0
        return _EPS

    s = grad_u[n_dim - 1] / r
    if metric_kind == _METRIC_RANDERS:
        phi = 1.0 + metric_alpha * s
        dphi = metric_alpha
    elif metric_kind == _METRIC_MATSUMOTO:
        denominator = 1.0 - metric_alpha * s
        if denominator <= 0.0:
            raise ValueError("Matsumoto metric produced a forbidden direction.")
        phi = 1.0 / denominator
        dphi = metric_alpha / (denominator * denominator)
    else:
        if metric_alpha != 0.0 and s > 1.0 / (2.0 * metric_alpha):
            phi = 4.0 * metric_alpha * s
            dphi = 4.0 * metric_alpha
        else:
            denominator = 1.0 - metric_alpha * s
            phi = 1.0 / denominator
            dphi = metric_alpha / (denominator * denominator)

    coeff = phi - s * dphi
    for d in range(n_dim):
        grad_u[d] = coeff * grad_u[d] / r
    grad_u[n_dim - 1] += dphi
    f = r * phi
    if f > _EPS:
        return f
    return _EPS


@nb.njit(cache=True)
def _euclidean_length_grad(X, src, dst, grad_u):
    r2 = 0.0
    for d in range(X.shape[1]):
        u = X[dst, d] - X[src, d]
        grad_u[d] = u
        r2 += u * u
    r = math.sqrt(r2)
    if r <= _EPS:
        for d in range(X.shape[1]):
            grad_u[d] = 0.0
        return _EPS
    for d in range(X.shape[1]):
        grad_u[d] /= r
    return r


@nb.njit(cache=True)
def _apply_update(
        X, src, dst, coeff, grad_u, alpha, move_dst, gradient_clip
):
    for d in range(X.shape[1]):
        value = alpha * _clip(coeff * grad_u[d], gradient_clip)
        X[src, d] += value
        if move_dst:
            X[dst, d] -= value


@nb.njit(cache=True)
def _optimize_umap_core(
        X,
        row,
        col,
        epochs_per_sample,
        metric_kind,
        metric_alpha,
        max_iter,
        learning_rate,
        negative_sample_rate,
        negative_sample_weight,
        a,
        b,
        gradient_clip,
        negative_metric_is_finsler,
        random_seed,
        verbose,
        log_frequency,
):
    np.random.seed(random_seed)
    n_edges = len(row)
    n_samples, n_dim = X.shape
    epoch_of_next_sample = epochs_per_sample.copy()
    if negative_sample_rate > 0.0:
        epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
        epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    else:
        epochs_per_negative_sample = np.empty(n_edges, dtype=np.float64)
        epoch_of_next_negative_sample = np.empty(n_edges, dtype=np.float64)
        for i in range(n_edges):
            epochs_per_negative_sample[i] = math.inf
            epoch_of_next_negative_sample[i] = math.inf
    grad_u = np.empty(n_dim, dtype=np.float64)
    last_loss = np.nan

    for epoch in range(1, max_iter + 1):
        epoch_loss = 0.0
        alpha_lr = learning_rate * (1.0 - (epoch - 1) / max(1, max_iter))
        for edge_idx in range(n_edges):
            if epoch_of_next_sample[edge_idx] > epoch:
                continue
            src = row[edge_idx]
            dst = col[edge_idx]
            f = _length_grad(X, src, dst, metric_kind, metric_alpha, grad_u)
            y = a * f ** (2.0 * b)
            q = 1.0 / (1.0 + y)
            if q < _EPS:
                q = _EPS
            elif q > 1.0 - _EPS:
                q = 1.0 - _EPS
            epoch_loss -= math.log(q)
            coeff = 2.0 * b * (1.0 - q) / f
            _apply_update(
                X, src, dst, coeff, grad_u, alpha_lr, True, gradient_clip
            )
            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx]

            n_negative = int(
                (epoch - epoch_of_next_negative_sample[edge_idx])
                / epochs_per_negative_sample[edge_idx]
            )
            for _ in range(n_negative):
                neg_dst = np.random.randint(0, n_samples - 1)
                if neg_dst >= src:
                    neg_dst += 1
                if negative_metric_is_finsler:
                    f = _length_grad(X, src, neg_dst, metric_kind, metric_alpha, grad_u)
                else:
                    f = _euclidean_length_grad(X, src, neg_dst, grad_u)
                y = a * f ** (2.0 * b)
                q = 1.0 / (1.0 + y)
                if q < _EPS:
                    q = _EPS
                elif q > 1.0 - _EPS:
                    q = 1.0 - _EPS
                epoch_loss -= negative_sample_weight * math.log(1.0 - q)
                coeff = -negative_sample_weight * 2.0 * b * q / f
                _apply_update(
                    X, src, neg_dst, coeff, grad_u, alpha_lr, False, gradient_clip
                )
            epoch_of_next_negative_sample[edge_idx] += (
                n_negative * epochs_per_negative_sample[edge_idx]
            )

        last_loss = epoch_loss
        if verbose and (epoch == 1 or epoch % log_frequency == 0 or epoch == max_iter):
            print("finsler_umap epoch", epoch, ": sampled loss", last_loss)
    return X, last_loss


def _optimize_umap(
        X,
        *,
        metric,
        row,
        col,
        epochs_per_sample,
        max_iter,
        learning_rate,
        negative_sample_rate,
        negative_sample_weight,
        a,
        b,
        rng,
        gradient_clip,
        negative_metric,
        verbose,
        log_frequency,
):
    metric_kind, metric_alpha = _metric_params(metric)
    clip = -1.0 if gradient_clip is None else float(gradient_clip)
    random_seed = int(rng.randint(0, 2**31 - 1))
    return _optimize_umap_core(
        X,
        row.astype(np.int64, copy=False),
        col.astype(np.int64, copy=False),
        epochs_per_sample.astype(float, copy=False),
        metric_kind,
        metric_alpha,
        int(max_iter),
        float(learning_rate),
        float(negative_sample_rate),
        float(negative_sample_weight),
        float(a),
        float(b),
        clip,
        negative_metric == "finsler",
        random_seed,
        int(verbose),
        int(log_frequency),
    )


def finsler_umap(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    n_neighbors=50,
    symmetrize_support=False,
    symmetrize_rho=True,
    symmetrize_sigma=True,
    local_connectivity=1.0,
    bandwidth=1.0,
    min_dist=0.5,
    spread=1.0,
    max_iter=1500,
    learning_rate=1.0,
    negative_sample_rate=10,
    negative_sample_weight=1.0,
    negative_metric="euclidean",
    gradient_clip=4.0,
    weight=None,
    random_state=None,
    verbose=0,
    log_frequency=None,
    return_n_iter=False,
):
    """Optimize a directed Finsler-UMAP objective.

    ``dissimilarities`` may be asymmetric and must be dense. Directed
    high-dimensional edge weights are
    ``exp(-max(0, d_ij - rho_ij) / sigma_ij)`` on the outgoing kNN graph.
    ``symmetrize_support``, ``symmetrize_rho``, and ``symmetrize_sigma`` control
    the support and local-scale symmetrization independently. If ``init`` is
    ``None`` or ``"spectral"``, the embedding is initialized with a UMAP-style
    spectral initialization on the symmetrized fuzzy graph. By default,
    negative samples     use Euclidean repulsion; set ``negative_metric="finsler"`` to reproduce the
    fully Finsler negative-sampling behavior.
    """
    metric = validate_metric(metric)
    D = np.asarray(dissimilarities, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")
    rng = check_random_state(random_state)

    a, b = find_ab_params(spread, min_dist)
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive.")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if negative_sample_rate < 0:
        raise ValueError("negative_sample_rate must be non-negative.")
    negative_metric = str(negative_metric).lower()
    if negative_metric not in {"euclidean", "finsler"}:
        raise ValueError("negative_metric must be either 'euclidean' or 'finsler'.")
    if log_frequency is None:
        log_frequency = max(1, int(verbose)) if verbose else 1
    log_frequency = int(log_frequency)
    if log_frequency <= 0:
        raise ValueError("log_frequency must be positive.")

    n_samples = D.shape[0]
    graph = directed_fuzzy_graph_from_dense(
        D,
        n_neighbors,
        weight=weight,
        symmetrize_support=symmetrize_support,
        symmetrize_rho=symmetrize_rho,
        symmetrize_sigma=symmetrize_sigma,
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
    )
    row, col = graph.row, graph.col
    p, p_graph = graph.probability, graph.graph_probability

    if init is None or (isinstance(init, str) and init == "spectral"):
        X = spectral_initial_embedding(row, col, p_graph, n_samples, n_components, rng)
    elif isinstance(init, str) and init == "random":
        X = initial_embedding(D, n_components, None, random_state)
    else:
        X = initial_embedding(D, n_components, init, random_state)
    X = X.astype(float, copy=True)

    epochs_per_sample = make_epochs_per_sample(p, int(max_iter))
    sampled = epochs_per_sample <= int(max_iter)
    row, col, epochs_per_sample = row[sampled], col[sampled], epochs_per_sample[sampled]
    if len(row) == 0:
        raise ValueError("No Finsler-UMAP edges are sampled with the requested max_iter.")
    X, last_loss = _optimize_umap(
        X,
        metric=metric,
        row=row,
        col=col,
        epochs_per_sample=epochs_per_sample,
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        negative_sample_rate=float(negative_sample_rate),
        negative_sample_weight=float(negative_sample_weight),
        a=a,
        b=b,
        rng=rng,
        gradient_clip=gradient_clip,
        negative_metric=negative_metric,
        verbose=verbose,
        log_frequency=log_frequency,
    )

    if return_n_iter:
        return X, float(last_loss), int(max_iter)
    return X, float(last_loss)


__all__ = ["finsler_umap"]
