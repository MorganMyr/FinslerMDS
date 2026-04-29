"""Soft Bellman-Ford optimizer for Finsler-MDS.

This is an all-sources, differentiable Bellman-Ford algorithm. Hard edge
minimum updates are replaced by softmin updates, and gradients are propagated
explicitly through the relaxation trace.
"""

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
from finsler_mds.utils.graph import softmin_with_probs, symmetric_knn_graph


@dataclass(frozen=True)
class SoftBellmanFordResult:
    embedding: np.ndarray
    stress: float
    n_iter: int
    n_graph_updates: int
    optimizer_results: list


@dataclass(frozen=True)
class _GraphSupport:
    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]


def _support_from_embedding(X, *, n_neighbors, neighbors_algorithm, n_jobs):
    support = symmetric_knn_graph(
        X,
        n_neighbors=n_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    ).tocoo()
    return _GraphSupport(
        rows=support.row.astype(int, copy=False),
        cols=support.col.astype(int, copy=False),
        shape=support.shape,
    )


def _metric_edges(X, support, metric):
    edge_vectors = X[support.cols] - X[support.rows]
    edge_lengths = metric.length(edge_vectors)
    finite = np.isfinite(edge_lengths)
    return (
        support.rows[finite],
        support.cols[finite],
        edge_lengths[finite],
        edge_vectors[finite],
    )


def soft_bellman_ford_distances(
        n_samples,
        tails,
        heads,
        costs,
        *,
        beta,
        n_relaxations=None,
        prob_dtype=np.float32,
):
    """Return soft all-pairs distances and a trace for backpropagation.

    The update is synchronous: after ``t`` passes, paths with at most ``t``
    edges have contributed. With a hard min, ``n_samples - 1`` passes recover
    Bellman-Ford distances on graphs without negative cycles.
    """
    if n_relaxations is None:
        n_relaxations = n_samples - 1
    if n_relaxations < 0:
        raise ValueError("n_relaxations must be non-negative.")

    distances = np.full((n_samples, n_samples), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    incoming_edges = [np.flatnonzero(heads == v) for v in range(n_samples)]

    trace = []
    for _ in range(n_relaxations):
        next_distances = np.empty_like(distances)
        keep_probs = np.zeros((n_samples, n_samples), dtype=prob_dtype)
        edge_probs = np.zeros((len(costs), n_samples), dtype=prob_dtype)

        for target in range(n_samples):
            incoming = incoming_edges[target]
            if incoming.size == 0:
                next_distances[:, target] = distances[:, target]
                keep_probs[:, target] = 1.0
                continue

            candidates = np.empty((n_samples, incoming.size + 1), dtype=float)
            candidates[:, 0] = distances[:, target]
            candidates[:, 1:] = distances[:, tails[incoming]] + costs[incoming][None, :]
            values, probs = softmin_with_probs(
                candidates,
                beta=beta,
                axis=1,
                prob_dtype=prob_dtype,
            )

            # Keep exact zero self-distances. Otherwise the entropy term can
            # make zero cycles slightly negative when beta is finite.
            values[target] = 0.0
            probs[target, :] = 0.0
            probs[target, 0] = 1.0

            next_distances[:, target] = values
            keep_probs[:, target] = probs[:, 0]
            edge_probs[incoming, :] = probs[:, 1:].T

        distances = next_distances
        trace.append((keep_probs, edge_probs))

    return distances, trace


def _soft_bellman_ford_pullback(final_adjoint, trace, tails, heads):
    adjoint = np.asarray(final_adjoint, dtype=float)
    cost_adjoint = np.zeros(len(tails), dtype=float)

    for keep_probs, edge_probs in reversed(trace):
        keep_probs = keep_probs.astype(float, copy=False)
        edge_probs = edge_probs.astype(float, copy=False)
        previous = keep_probs * adjoint

        edge_to_head_adjoint = adjoint[:, heads].T
        edge_contrib = edge_probs * edge_to_head_adjoint
        cost_adjoint += edge_contrib.sum(axis=1)
        for edge_id, tail in enumerate(tails):
            previous[:, tail] += edge_contrib[edge_id]

        adjoint = previous

    return cost_adjoint


def _soft_bf_stress_and_grad(
        X_flat,
        *,
        shape,
        support,
        dissimilarities,
        weight,
        metric,
        beta,
        n_relaxations,
        prob_dtype,
        normalized_stress,
):
    X = X_flat.reshape(shape)
    tails, heads, costs, edge_vectors = _metric_edges(X, support, metric)
    distances, trace = soft_bellman_ford_distances(
        X.shape[0],
        tails,
        heads,
        costs,
        beta=beta,
        n_relaxations=n_relaxations,
        prob_dtype=prob_dtype,
    )

    active = weight != 0
    if np.any(active & ~np.isfinite(distances)):
        raise ValueError(
            "The soft Bellman-Ford graph has unreachable active pairs. "
            "Increase n_neighbors or use a metric without infinite local edges."
        )

    residual = np.zeros_like(dissimilarities, dtype=float)
    residual[active] = weight[active] * (distances[active] - dissimilarities[active])
    raw_stress = np.sum(weight[active] * (distances[active] - dissimilarities[active]) ** 2)

    final_adjoint = 2.0 * residual
    stress = raw_stress
    if normalized_stress:
        stress, norm_scale = normalized_stress_scale(raw_stress, dissimilarities, weight)
        final_adjoint *= norm_scale / 2.0

    edge_adjoint = _soft_bellman_ford_pullback(final_adjoint, trace, tails, heads)
    edge_grads = metric.grad_u(edge_vectors)
    used_edges = edge_adjoint != 0
    if not np.all(np.isfinite(edge_grads[used_edges])):
        raise ValueError(
            "The metric produced non-finite gradients on active graph edges. "
            "If using Matsumoto with forbidden directions, set a finite "
            "forbidden_grad_norm or use a convexified/clipped metric."
        )

    grad = np.zeros_like(X)
    scaled_edge_grads = edge_adjoint[used_edges, None] * edge_grads[used_edges]
    np.add.at(grad, tails[used_edges], -scaled_edge_grads)
    np.add.at(grad, heads[used_edges], scaled_edge_grads)
    return float(stress), grad.ravel()


def soft_bellman_ford(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    n_neighbors=10,
    beta=10.0,
    n_relaxations=None,
    max_iter=100,
    n_graph_updates=1,
    verbose=0,
    eps=1e-6,
    random_state=None,
    normalized_stress=False,
    weight=None,
    method="L-BFGS-B",
    optimizer_options=None,
    neighbors_algorithm="auto",
    n_jobs=None,
    prob_dtype=np.float32,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with soft Bellman-Ford geodesics."""
    metric = validate_metric(metric)
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if n_graph_updates < 1:
        raise ValueError("n_graph_updates must be at least 1.")

    D, W = prepare_weights_and_mask(dissimilarities, weight)
    X = initial_embedding(D, n_components, init, random_state)
    shape = X.shape

    options = {"maxiter": max_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    optimizer_results = []
    total_iter = 0
    stress = np.inf

    for graph_update in range(n_graph_updates):
        support = _support_from_embedding(
            X,
            n_neighbors=n_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )

        def objective(x_flat):
            return _soft_bf_stress_and_grad(
                x_flat,
                shape=shape,
                support=support,
                dissimilarities=D,
                weight=W,
                metric=metric,
                beta=beta,
                n_relaxations=n_relaxations,
                prob_dtype=prob_dtype,
                normalized_stress=normalized_stress,
            )

        result = scipy.optimize.minimize(
            objective,
            X.ravel(),
            jac=True,
            method=method,
            options=options,
        )
        optimizer_results.append(result)
        X = result.x.reshape(shape)
        stress = float(result.fun)
        total_iter += int(getattr(result, "nit", max_iter))

        if verbose:
            print(f"soft_bellman_ford graph update {graph_update}: stress {stress}")

    sbf_result = SoftBellmanFordResult(
        embedding=X,
        stress=float(stress),
        n_iter=total_iter,
        n_graph_updates=n_graph_updates,
        optimizer_results=optimizer_results,
    )

    if return_result:
        return sbf_result
    if return_n_iter:
        return sbf_result.embedding, sbf_result.stress, sbf_result.n_iter
    return sbf_result.embedding, sbf_result.stress


def optimize_soft_bellman_ford(*args, **kwargs):
    """Alias used by the higher-level API layer."""
    return soft_bellman_ford(*args, **kwargs)


# Backward-compatible aliases for code written before the rename.
RelaxedBellmanFordResult = SoftBellmanFordResult
relaxed_bellman_ford_distances = soft_bellman_ford_distances
relaxed_bellman_ford = soft_bellman_ford
optimize_relaxed_bellman_ford = optimize_soft_bellman_ford


__all__ = [
    "SoftBellmanFordResult",
    "soft_bellman_ford_distances",
    "soft_bellman_ford",
    "optimize_soft_bellman_ford",
    "RelaxedBellmanFordResult",
    "relaxed_bellman_ford_distances",
    "relaxed_bellman_ford",
    "optimize_relaxed_bellman_ford",
]
