"""Path-frozen geodesic optimizer for Finsler-MDS.

The optimizer alternates between:
1. building a kNN graph on the current embedding and computing all-pairs
   shortest paths with Dijkstra;
2. freezing those paths and optimizing the stress for a few gradient-based
   steps, treating each path length as the sum of metric lengths of its edges.
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
from finsler_mds.utils.graph import (
    dijkstra_all_pairs,
    metric_graph_from_support,
    symmetric_knn_graph,
)


@dataclass(frozen=True)
class PathFrozenResult:
    embedding: np.ndarray
    stress: float
    n_iter: int
    n_path_updates: int
    optimizer_results: list


@dataclass(frozen=True)
class _FrozenForest:
    parents: np.ndarray
    orders: list[np.ndarray]
    active_targets: list[np.ndarray]
    edge_nodes: list[np.ndarray]
    edge_ids: list[np.ndarray]
    edge_tails: np.ndarray
    edge_heads: np.ndarray


def _source_tree_order(parents, source, finite):
    """Return a parent-before-child order for one Dijkstra predecessor tree."""
    children = [[] for _ in range(parents.shape[0])]
    for target, parent in enumerate(parents):
        if target == source or not finite[target]:
            continue
        if parent < 0:
            continue
        children[parent].append(target)

    order = []
    stack = [source]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])
    return np.asarray(order, dtype=int)


def _path_frozen_stress_and_grad(
        X_flat,
        *,
        shape,
        forest,
        dissimilarities,
        weight,
        metric,
        normalized_stress,
):
    X = X_flat.reshape(shape)
    n_samples, n_components = X.shape
    grad = np.zeros_like(X)
    raw_stress = 0.0

    edge_vectors = X[forest.edge_heads] - X[forest.edge_tails]
    unique_edge_lengths = metric.length(edge_vectors)
    unique_edge_grads = metric.grad_u(edge_vectors)
    if (
        not np.all(np.isfinite(unique_edge_lengths))
        or not np.all(np.isfinite(unique_edge_grads))
    ):
        raise ValueError(
            "The metric produced a non-finite edge length or gradient "
            "on a frozen shortest-path tree."
        )

    for source in range(n_samples):
        edge_nodes = forest.edge_nodes[source]
        parent = forest.parents[source]
        edge_parents = parent[edge_nodes]
        edge_lengths = unique_edge_lengths[forest.edge_ids[source]]
        edge_grads = unique_edge_grads[forest.edge_ids[source]]

        path_lengths = np.zeros(n_samples, dtype=float)
        edge_grad_by_node = np.zeros((n_samples, n_components), dtype=float)
        edge_grad_by_node[edge_nodes] = edge_grads
        for node, parent_node, edge_length in zip(edge_nodes, edge_parents, edge_lengths):
            path_lengths[node] = path_lengths[parent_node] + edge_length

        targets = forest.active_targets[source]
        residual = path_lengths[targets] - dissimilarities[source, targets]
        raw_stress += np.sum(weight[source, targets] * residual ** 2)

        # Reverse-mode pass on the frozen tree. The adjoint at a vertex is the
        # sum of all residual contributions whose frozen path uses the edge
        # from its parent to that vertex.
        adjoint = np.zeros(n_samples, dtype=float)
        adjoint[targets] = 2.0 * weight[source, targets] * residual
        for node in edge_nodes[::-1]:
            parent_node = parent[node]
            scale = adjoint[node]
            if scale != 0:
                edge_grad = edge_grad_by_node[node]
                grad[parent_node] -= scale * edge_grad
                grad[node] += scale * edge_grad
            adjoint[parent_node] += scale

    stress = raw_stress
    if normalized_stress:
        stress, norm_scale = normalized_stress_scale(raw_stress, dissimilarities, weight)
        grad *= norm_scale / 2.0

    return float(stress), grad.ravel()


def _frozen_forest_from_embedding(
        X,
        *,
        metric,
        n_neighbors,
        active,
        neighbors_algorithm,
        n_jobs,
):
    support = symmetric_knn_graph(
        X,
        n_neighbors=n_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    )
    graph = metric_graph_from_support(X, support, metric)
    dist_matrix, predecessors = dijkstra_all_pairs(graph, directed=True)
    if np.any(active & ~np.isfinite(dist_matrix)):
        raise ValueError(
            "The current embedding graph has unreachable active pairs. "
            "Increase n_neighbors or use a metric without infinite local edges."
        )

    orders = [
        _source_tree_order(predecessors[source], source, np.isfinite(dist_matrix[source]))
        for source in range(active.shape[0])
    ]
    for source, order in enumerate(orders):
        in_tree = np.zeros(active.shape[0], dtype=bool)
        in_tree[order] = True
        if np.any(active[source] & ~in_tree):
            raise ValueError(
                "An active pair is finite in the Dijkstra matrix but is missing "
                "from the predecessor tree."
            )

    edge_nodes = []
    edge_ids = []
    edge_to_id = {}
    edge_tails = []
    edge_heads = []
    for source, order in enumerate(orders):
        nodes = order[order != source]
        ids = np.empty(nodes.shape[0], dtype=int)
        for pos, node in enumerate(nodes):
            parent = predecessors[source, node]
            key = (int(parent), int(node))
            edge_id = edge_to_id.get(key)
            if edge_id is None:
                edge_id = len(edge_tails)
                edge_to_id[key] = edge_id
                edge_tails.append(key[0])
                edge_heads.append(key[1])
            ids[pos] = edge_id
        edge_nodes.append(nodes)
        edge_ids.append(ids)

    forest = _FrozenForest(
        parents=predecessors,
        orders=orders,
        active_targets=[np.flatnonzero(row) for row in active],
        edge_nodes=edge_nodes,
        edge_ids=edge_ids,
        edge_tails=np.asarray(edge_tails, dtype=int),
        edge_heads=np.asarray(edge_heads, dtype=int),
    )
    return forest, dist_matrix


def path_frozen(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    n_neighbors=10,
    max_iter=20,
    inner_iter=5,
    verbose=0,
    eps=1e-6,
    random_state=None,
    normalized_stress=False,
    weight=None,
    method="L-BFGS-B",
    optimizer_options=None,
    neighbors_algorithm="auto",
    n_jobs=None,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with path-frozen graph geodesics."""
    metric = validate_metric(metric)
    D, W = prepare_weights_and_mask(dissimilarities, weight)
    active = W != 0
    X = initial_embedding(D, n_components, init, random_state)
    shape = X.shape

    options = {"maxiter": inner_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    optimizer_results = []
    old_stress = None
    total_inner_iter = 0

    for outer_it in range(max_iter):
        forest, _ = _frozen_forest_from_embedding(
            X,
            metric=metric,
            n_neighbors=n_neighbors,
            active=active,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )

        def objective(x_flat):
            return _path_frozen_stress_and_grad(
                x_flat,
                shape=shape,
                forest=forest,
                dissimilarities=D,
                weight=W,
                metric=metric,
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
        total_inner_iter += int(getattr(result, "nit", inner_iter))

        if verbose:
            print(f"path_frozen outer {outer_it}: stress {stress}")

        if old_stress is not None and old_stress != 0:
            if np.abs(1 - stress / old_stress) < eps:
                break
        old_stress = stress

    pf_result = PathFrozenResult(
        embedding=X,
        stress=float(stress),
        n_iter=total_inner_iter,
        n_path_updates=outer_it + 1,
        optimizer_results=optimizer_results,
    )

    if return_result:
        return pf_result
    if return_n_iter:
        return pf_result.embedding, pf_result.stress, pf_result.n_iter
    return pf_result.embedding, pf_result.stress


def optimize_path_frozen(*args, **kwargs):
    """Alias used by the higher-level API layer."""
    return path_frozen(*args, **kwargs)


__all__ = [
    "PathFrozenResult",
    "path_frozen",
    "optimize_path_frozen",
]
