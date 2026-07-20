"""Differentiable Floyd-Warshall optimizer for Finsler-MDS.

This implements the part of DataSP that is useful for Finsler-MDS: replace
the hard minimum in Floyd-Warshall by a temperature-controlled soft minimum,
then backpropagate explicitly through the dynamic program.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize
from sklearn.utils import check_random_state

from finsler_mds.evaluation import geodesic_embedding_stress
from finsler_mds.optimizers.common import (
    initial_embedding,
    normalized_stress_scale,
    prepare_weights_and_mask,
    validate_metric,
)
from finsler_mds.optimizers.pair_groups import (
    build_local_global_pairs,
    empty_active_pairs,
    merge_active_pairs,
    sample_active_pairs,
)
from finsler_mds.optimizers.path_frozen import (
    _DirectPairsObjective,
    _add_raw_objective,
)
from finsler_mds.utils.graph import softmin_with_probs, symmetric_knn_graph


@dataclass(frozen=True)
class DataSPResult:
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


def _support_from_embedding(X, *, graph_neighbors, neighbors_algorithm, n_jobs):
    support = symmetric_knn_graph(
        X,
        n_neighbors=graph_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    ).tocoo()
    return _GraphSupport(
        rows=support.row.astype(int, copy=False),
        cols=support.col.astype(int, copy=False),
        shape=support.shape,
    )


def _metric_cost_matrix(X, support, metric):
    n_samples = support.shape[0]
    edge_vectors = X[support.cols] - X[support.rows]
    edge_lengths = metric.length(edge_vectors)
    finite = np.isfinite(edge_lengths)

    costs = np.full((n_samples, n_samples), np.inf, dtype=float)
    np.fill_diagonal(costs, 0.0)
    costs[support.rows[finite], support.cols[finite]] = edge_lengths[finite]
    return costs, edge_vectors, edge_lengths, finite


def _softmin_pair(old, via, *, beta, update_mask):
    finite = update_mask & (np.isfinite(old) | np.isfinite(via))
    new = old.copy()
    via_prob = np.zeros_like(old, dtype=float)
    if not np.any(finite):
        return new, via_prob

    candidates = np.stack([old[finite], via[finite]], axis=-1)
    soft_values, probs = softmin_with_probs(candidates, beta=beta, axis=-1)
    new[finite] = soft_values
    via_prob[finite] = probs[:, 1]
    return new, via_prob


def soft_floyd_warshall(costs, *, beta, prob_dtype=np.float32):
    """Return soft all-pairs distances and local via probabilities.

    ``via_probs[k, i, j]`` is the derivative of the kth update
    ``softmin(M[i, j], M[i, k] + M[k, j])`` with respect to the second
    candidate. Rows, columns, and the diagonal touched by k are skipped; with
    non-negative edge costs, these self-loop updates are redundant and can make
    the entropy-smoothed distances drift below zero.
    """
    distances = np.asarray(costs, dtype=float).copy()
    n_samples = distances.shape[0]
    via_probs = []
    base_mask = ~np.eye(n_samples, dtype=bool)

    for k in range(n_samples):
        update_mask = base_mask.copy()
        update_mask[k, :] = False
        update_mask[:, k] = False
        via = distances[:, [k]] + distances[[k], :]
        distances, via_prob = _softmin_pair(
            distances,
            via,
            beta=beta,
            update_mask=update_mask,
        )
        via_probs.append(via_prob.astype(prob_dtype, copy=False))

    return distances, via_probs


def _soft_fw_pullback(final_adjoint, via_probs):
    """Backpropagate an adjoint through ``soft_floyd_warshall``."""
    adjoint = np.asarray(final_adjoint, dtype=float)
    for k in range(len(via_probs) - 1, -1, -1):
        via_prob = via_probs[k].astype(float, copy=False)
        contribution = via_prob * adjoint
        previous = (1.0 - via_prob) * adjoint
        previous[:, k] += contribution.sum(axis=1)
        previous[k, :] += contribution.sum(axis=0)
        adjoint = previous
    return adjoint


def _datasp_stress_and_grad(
        X_flat,
        *,
        shape,
        support,
        dissimilarities,
        weight,
        metric,
        beta,
        prob_dtype,
        normalized_stress,
):
    X = X_flat.reshape(shape)
    costs, edge_vectors, edge_lengths, finite_edges = _metric_cost_matrix(X, support, metric)
    soft_distances, via_probs = soft_floyd_warshall(costs, beta=beta, prob_dtype=prob_dtype)

    active = weight != 0
    if np.any(active & ~np.isfinite(soft_distances)):
        raise ValueError(
            "The soft Floyd-Warshall graph has unreachable active pairs. "
            "Increase n_neighbors or use a metric without infinite local edges."
        )

    residual = np.zeros_like(dissimilarities, dtype=float)
    residual[active] = weight[active] * (soft_distances[active] - dissimilarities[active])
    raw_stress = np.sum(weight[active] * (soft_distances[active] - dissimilarities[active]) ** 2)

    final_adjoint = 2.0 * residual
    stress = raw_stress
    if normalized_stress:
        stress, norm_scale = normalized_stress_scale(raw_stress, dissimilarities, weight)
        final_adjoint *= norm_scale / 2.0

    cost_adjoint = _soft_fw_pullback(final_adjoint, via_probs)
    edge_adjoint = cost_adjoint[support.rows, support.cols]

    edge_grads = metric.grad_u(edge_vectors)
    used_edges = finite_edges & (edge_adjoint != 0)
    if not np.all(np.isfinite(edge_grads[used_edges])):
        raise ValueError(
            "The metric produced non-finite gradients on active graph edges. "
            "If using Matsumoto with forbidden directions, set a finite "
            "forbidden_grad_norm or use a convexified/clipped metric."
        )

    grad = np.zeros_like(X)
    scaled_edge_grads = edge_adjoint[used_edges, None] * edge_grads[used_edges]
    np.add.at(grad, support.rows[used_edges], -scaled_edge_grads)
    np.add.at(grad, support.cols[used_edges], scaled_edge_grads)
    return float(stress), grad.ravel()


def _active_pairs_to_weight_matrix(shape, active_pairs):
    W = np.zeros(shape, dtype=float)
    for source_pos, source in enumerate(active_pairs.sources):
        W[source, active_pairs.targets[source_pos]] = active_pairs.weights[source_pos]
    return W


def datasp(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    graph_neighbors=10,
    beta=10.0,
    max_iter=100,
    n_graph_updates=1,
    verbose=0,
    eps=1e-6,
    random_state=None,
    normalized_stress=False,
    weight=None,
    pair_mask=None,
    n_local_neighbors=None,
    local_pair_mode="geodesic",
    landmark_indices=None,
    n_global_landmarks=0,
    random_landmark_fraction=1.0,
    fps_init="random",
    mask_random_state=None,
    max_global_targets_per_source=None,
    target_random_state=None,
    local_weight=1.0,
    local_global_reweighting="none",
    method="L-BFGS-B",
    optimizer_options=None,
    neighbors_algorithm="auto",
    n_jobs=None,
    prob_dtype=np.float32,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with soft Floyd-Warshall geodesics.

    Each graph update rebuilds the kNN support from the current embedding, then
    runs up to ``max_iter`` optimizer iterations with that support fixed. Edge
    lengths, soft all-pairs distances, and their gradients are still recomputed
    at every objective call. Local/global pair options match ``path_frozen``;
    they only sparsify the stress adjoint, not the Floyd-Warshall dynamic
    program itself.
    """
    metric = validate_metric(metric)
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if n_graph_updates < 1:
        raise ValueError("n_graph_updates must be at least 1.")

    D, W = prepare_weights_and_mask(dissimilarities, weight)
    if mask_random_state is None:
        mask_random_state = random_state
    if target_random_state is None:
        target_random_state = random_state
    target_random_state = check_random_state(target_random_state)
    pair_groups = build_local_global_pairs(
        D,
        W,
        pair_mask=pair_mask,
        n_local_neighbors=n_local_neighbors,
        local_pair_mode=local_pair_mode,
        landmark_indices=landmark_indices,
        n_global_landmarks=n_global_landmarks,
        random_landmark_fraction=random_landmark_fraction,
        fps_init=fps_init,
        random_state=mask_random_state,
        local_weight=local_weight,
        local_global_reweighting=local_global_reweighting,
    )
    global_pairs = pair_groups.global_pairs
    local_pairs = pair_groups.local_pairs
    local_geodesic_pairs = local_pairs if local_pair_mode == "geodesic" else empty_active_pairs()
    direct_pairs = local_pairs if local_pair_mode == "direct" else empty_active_pairs()

    X = initial_embedding(D, n_components, init, random_state)
    shape = X.shape
    direct_objective = (
        _DirectPairsObjective(shape=shape, direct_pairs=direct_pairs, metric=metric)
        if direct_pairs.n_pairs > 0
        else None
    )

    options = {"maxiter": max_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    optimizer_results = []
    total_iter = 0
    stress = np.inf

    for graph_update in range(n_graph_updates):
        iteration_global_pairs = sample_active_pairs(
            global_pairs,
            max_targets_per_source=max_global_targets_per_source,
            random_state=target_random_state,
        )
        iteration_pairs = merge_active_pairs(iteration_global_pairs, local_geodesic_pairs)
        iteration_weight = _active_pairs_to_weight_matrix(D.shape, iteration_pairs)
        support = _support_from_embedding(
            X,
            graph_neighbors=graph_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )

        def raw_objective(x_flat):
            return _datasp_stress_and_grad(
                x_flat,
                shape=shape,
                support=support,
                dissimilarities=D,
                weight=iteration_weight,
                metric=metric,
                beta=beta,
                prob_dtype=prob_dtype,
                normalized_stress=False,
            )

        def objective(x_flat):
            return _add_raw_objective(
                x_flat,
                raw_objective if iteration_pairs.n_pairs > 0 else None,
                shape=shape,
                direct_pairs=direct_pairs,
                direct_objective=direct_objective,
                metric=metric,
                denom=iteration_pairs.denom + direct_pairs.denom,
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
            n_active_sources = len(np.union1d(iteration_pairs.sources, direct_pairs.sources))
            print(
                f"datasp graph update {graph_update}: stress {stress} "
                f"({iteration_global_pairs.n_pairs} global, "
                f"{local_pairs.n_pairs} local-{local_pair_mode}, "
                f"{n_active_sources} active sources)"
            )

    if verbose:
        full_stress = geodesic_embedding_stress(
            X,
            D,
            metric=metric,
            n_neighbors=graph_neighbors,
            weight=W,
            normalized_stress=normalized_stress,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            on_unreachable="inf",
        )
        print(f"datasp final full geodesic stress: {full_stress}")

    ds_result = DataSPResult(
        embedding=X,
        stress=float(stress),
        n_iter=total_iter,
        n_graph_updates=n_graph_updates,
        optimizer_results=optimizer_results,
    )

    if return_result:
        return ds_result
    if return_n_iter:
        return ds_result.embedding, ds_result.stress, ds_result.n_iter
    return ds_result.embedding, ds_result.stress


def dataSP(*args, **kwargs):
    """Compatibility alias matching the module name."""
    return datasp(*args, **kwargs)


def optimize_datasp(*args, **kwargs):
    """Alias used by the higher-level API layer."""
    return datasp(*args, **kwargs)


__all__ = [
    "DataSPResult",
    "soft_floyd_warshall",
    "datasp",
    "dataSP",
    "optimize_datasp",
]
