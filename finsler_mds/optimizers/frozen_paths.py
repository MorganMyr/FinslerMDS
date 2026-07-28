"""Frozen shortest-path construction and stress objectives."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse.csgraph import dijkstra

from finsler_mds.optimizers.metric_kernels import cupy_metric_length_and_grad
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph


_GPU_MAX_PATH_EDGES = 100_000_000


@dataclass(frozen=True)
class _FrozenForest:
    sources: np.ndarray
    parents: np.ndarray
    edge_nodes: list[np.ndarray]
    edge_ids: list[np.ndarray]
    edge_tails: np.ndarray
    edge_heads: np.ndarray


def build_frozen_path_objective(
    X,
    *,
    shape,
    active_pairs,
    metric,
    graph_neighbors,
    n_jobs,
    verbose,
    gpu_backend,
    device,
):
    """Freeze current shortest paths and build their stress objective."""
    if active_pairs.n_pairs == 0:
        return None

    forest = _frozen_forest_from_embedding(
        X,
        metric=metric,
        graph_neighbors=graph_neighbors,
        active_pairs=active_pairs,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    if gpu_backend is not None:
        try:
            return _GpuFrozenPathObjective(
                gpu_backend,
                shape=shape,
                forest=forest,
                active_pairs=active_pairs,
                metric=metric,
            )
        except MemoryError:
            if device in {"gpu", "cuda"}:
                raise
            warnings.warn(
                "Falling back to the CPU Path-Frozen objective because "
                "the flattened paths exceed the GPU safety limit.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _CpuFrozenPathObjective(
        shape=shape,
        forest=forest,
        active_pairs=active_pairs,
        metric=metric,
    )


class _CpuFrozenPathObjective:
    def __init__(self, *, shape, forest, active_pairs, metric):
        self.shape = shape
        self.forest = forest
        self.active_pairs = active_pairs
        self.metric = metric

    def __call__(self, X_flat):
        X = X_flat.reshape(self.shape)
        forest = self.forest
        edge_vectors = X[forest.edge_heads] - X[forest.edge_tails]
        edge_lengths = self.metric.length(edge_vectors)
        edge_grads = self.metric.grad_u(edge_vectors)
        if not np.all(np.isfinite(edge_lengths)) or not np.all(np.isfinite(edge_grads)):
            raise ValueError(
                "The metric produced a non-finite edge length or gradient "
                "on a frozen shortest-path tree."
            )

        grad = np.zeros_like(X)
        stress = 0.0
        for source_pos, source in enumerate(forest.sources):
            nodes = forest.edge_nodes[source_pos]
            parent = forest.parents[source_pos]
            parents = parent[nodes]
            lengths = edge_lengths[forest.edge_ids[source_pos]]
            grads = edge_grads[forest.edge_ids[source_pos]]

            path_lengths = np.zeros(len(X), dtype=float)
            for node, parent_node, length in zip(nodes, parents, lengths):
                path_lengths[node] = path_lengths[parent_node] + length

            targets = self.active_pairs.targets[source_pos]
            weights = self.active_pairs.weights[source_pos]
            residual = path_lengths[targets] - self.active_pairs.dissimilarities[source_pos]
            stress += np.sum(weights * residual**2)

            adjoint = np.zeros(len(X), dtype=float)
            adjoint[targets] = 2.0 * weights * residual
            for node, parent_node, edge_grad in zip(
                nodes[::-1],
                parents[::-1],
                grads[::-1],
            ):
                scale = adjoint[node]
                if scale != 0:
                    grad[parent_node] -= scale * edge_grad
                    grad[node] += scale * edge_grad
                adjoint[parent_node] += scale

        return float(stress), grad.ravel()


class _GpuFrozenPathObjective:
    def __init__(self, cp, *, shape, forest, active_pairs, metric):
        self.cp = cp
        self.shape = shape
        self.metric = metric

        path_offsets, path_edge_ids, weights, dissimilarities = _flatten_paths(
            forest,
            active_pairs,
        )
        path_counts = np.diff(path_offsets)
        self.path_offsets = cp.asarray(path_offsets[:-1], dtype=cp.int64)
        self.path_edge_ids = cp.asarray(path_edge_ids, dtype=cp.int32)
        self.path_ids = cp.asarray(
            np.repeat(np.arange(len(path_counts), dtype=np.int32), path_counts),
            dtype=cp.int32,
        )
        self.weights = cp.asarray(weights, dtype=cp.float64)
        self.dissimilarities = cp.asarray(dissimilarities, dtype=cp.float64)
        self.edge_tails = cp.asarray(forest.edge_tails, dtype=cp.int32)
        self.edge_heads = cp.asarray(forest.edge_heads, dtype=cp.int32)
        self.n_edges = len(forest.edge_tails)

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape))
        edge_vectors = X[self.edge_heads] - X[self.edge_tails]
        edge_lengths, edge_grads = cupy_metric_length_and_grad(cp, edge_vectors, self.metric)
        if not cp.all(cp.isfinite(edge_lengths)).item() or not cp.all(cp.isfinite(edge_grads)).item():
            raise ValueError(
                "The metric produced a non-finite edge length or gradient "
                "on a frozen shortest-path tree."
            )

        path_lengths = cp.add.reduceat(
            edge_lengths[self.path_edge_ids],
            self.path_offsets,
        )
        residual = path_lengths - self.dissimilarities
        stress = cp.sum(self.weights * residual**2)

        usage_coeff = (2.0 * self.weights * residual)[self.path_ids]
        edge_adjoint = cp.bincount(
            self.path_edge_ids,
            weights=usage_coeff,
            minlength=self.n_edges,
        )
        contributions = edge_adjoint[:, None] * edge_grads
        grad = cp.zeros_like(X)
        cp.add.at(grad, self.edge_tails, -contributions)
        cp.add.at(grad, self.edge_heads, contributions)
        return float(stress.get()), cp.asnumpy(grad).ravel()


def _flatten_paths(forest, active_pairs):
    path_offsets = [0]
    path_edge_ids = []
    weights = []
    dissimilarities = []
    n_samples = forest.parents.shape[1]

    for source_pos, source in enumerate(forest.sources):
        source = int(source)
        node_to_edge = np.full(n_samples, -1, dtype=np.int32)
        node_to_edge[forest.edge_nodes[source_pos]] = forest.edge_ids[source_pos]
        parent = forest.parents[source_pos]

        for target, weight, dissimilarity in zip(
            active_pairs.targets[source_pos],
            active_pairs.weights[source_pos],
            active_pairs.dissimilarities[source_pos],
        ):
            current = int(target)
            path_start = len(path_edge_ids)
            while current != source:
                edge_id = int(node_to_edge[current])
                if edge_id < 0:
                    raise ValueError("An active target path is missing from the frozen forest.")
                path_edge_ids.append(edge_id)
                current = int(parent[current])
                if current < 0:
                    raise ValueError("An active target is unreachable in the predecessor tree.")
            if len(path_edge_ids) == path_start:
                raise ValueError("Diagonal active pairs are not supported.")
            if len(path_edge_ids) > _GPU_MAX_PATH_EDGES:
                raise MemoryError("Flattened frozen paths exceed the GPU safety limit.")
            path_offsets.append(len(path_edge_ids))
            weights.append(float(weight))
            dissimilarities.append(float(dissimilarity))

    return (
        np.asarray(path_offsets, dtype=np.int64),
        np.asarray(path_edge_ids, dtype=np.int32),
        np.asarray(weights, dtype=float),
        np.asarray(dissimilarities, dtype=float),
    )


def _frozen_forest_from_embedding(
    X,
    *,
    metric,
    graph_neighbors,
    active_pairs,
    n_jobs,
    verbose,
):
    support = symmetric_knn_graph(
        X,
        n_neighbors=graph_neighbors,
        neighbors_algorithm="auto",
        n_jobs=n_jobs,
        ensure_connected=True,
        warn_on_connect=verbose >= 1,
    )
    graph = metric_graph_from_support(X, support, metric)
    distances, predecessors = dijkstra(
        graph,
        directed=True,
        indices=active_pairs.sources,
        return_predecessors=True,
    )
    distances = np.atleast_2d(distances)
    predecessors = np.atleast_2d(predecessors)

    edge_nodes = []
    edge_ids = []
    edge_to_id = {}
    edge_tails = []
    edge_heads = []
    for source_pos, source in enumerate(active_pairs.sources):
        targets = active_pairs.targets[source_pos]
        if np.any(~np.isfinite(distances[source_pos, targets])):
            raise ValueError(
                "The current embedding graph has unreachable active pairs. "
                "Increase graph_neighbors or use a metric without infinite local edges."
            )

        order = _pruned_tree_order(predecessors[source_pos], source, targets)
        nodes = order[order != source]
        ids = np.empty(len(nodes), dtype=int)
        for pos, node in enumerate(nodes):
            parent = predecessors[source_pos, node]
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

    return _FrozenForest(
        sources=active_pairs.sources,
        parents=predecessors,
        edge_nodes=edge_nodes,
        edge_ids=edge_ids,
        edge_tails=np.asarray(edge_tails, dtype=int),
        edge_heads=np.asarray(edge_heads, dtype=int),
    )


def _pruned_tree_order(parents, source, targets):
    source = int(source)
    included = {source}
    for target in targets:
        current = int(target)
        while current != source:
            parent = int(parents[current])
            if parent < 0:
                raise ValueError("An active target is unreachable in the predecessor tree.")
            included.add(current)
            included.add(parent)
            current = parent

    children = {node: [] for node in included}
    for node in included:
        if node != source:
            children[int(parents[node])].append(node)

    order = []
    stack = [source]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])
    return np.asarray(order, dtype=int)


__all__ = ["build_frozen_path_objective"]
