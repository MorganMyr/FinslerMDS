"""Path-frozen geodesic optimizer for Finsler-MDS.

The optimizer alternates between:
1. building a kNN graph on the current embedding and computing shortest-path
   trees from the active sources with Dijkstra;
2. freezing those paths and optimizing the stress for a few gradient-based
   steps, treating each path length as the sum of metric lengths of its edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
import scipy.optimize
from scipy.sparse.csgraph import dijkstra
from sklearn.utils import check_random_state

from finsler_mds.metrics import (
    ConvexifiedMatsumotoMetric,
    ConvexifiedToblerMetric,
    MatsumotoMetric,
    RandersMetric,
)
from finsler_mds.evaluation.distance_embedding import compute_embedding_distances
from finsler_mds.optimizers.common import (
    initial_embedding,
    prepare_weights_and_mask,
    validate_metric,
)
from finsler_mds.optimizers.pair_groups import (
    ActivePairs as _ActivePairs,
    build_local_global_pairs,
    empty_active_pairs,
    merge_active_pairs,
    sample_active_pairs,
)
from finsler_mds.utils.graph import (
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
    history: list
    final_full_geodesic_stress: float | None = None
    final_normalized_full_geodesic_stress: float | None = None


@dataclass(frozen=True)
class _FrozenForest:
    sources: np.ndarray
    parents: np.ndarray
    edge_nodes: list[np.ndarray]
    edge_ids: list[np.ndarray]
    edge_tails: np.ndarray
    edge_heads: np.ndarray


@dataclass(frozen=True)
class _FlatPairs:
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    dissimilarities: np.ndarray
    n_pairs: int


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        return None, exc

    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return None, RuntimeError("CuPy did not find a CUDA device.")
        # CuPy can see a device while still missing runtime compiler libraries
        # such as libnvrtc.so. Exercise indexing and an elementwise operation so
        # device="auto" falls back before scipy.optimize enters its first call.
        values = cp.arange(4, dtype=cp.float64)
        indices = cp.asarray([0, 2], dtype=cp.int32)
        cp.asnumpy(values[indices] + 1.0)
    except Exception as exc:
        return None, exc
    return cp, None


def _default_log_frequency(max_iter):
    max_iter = max(1, int(max_iter))
    if max_iter < 30:
        return 1
    decade = 10 ** int(np.floor(np.log10(max_iter)))
    if max_iter < 3 * decade:
        return max(1, decade // 10)
    return max(1, 5 * decade // 10)


def _resolve_log_frequency(log_frequency, max_iter):
    if log_frequency is None:
        return _default_log_frequency(max_iter)
    log_frequency = int(log_frequency)
    if log_frequency < 0:
        raise ValueError("log_frequency must be non-negative or None.")
    return log_frequency


def _should_log_iteration(iteration, max_iter, log_frequency):
    if log_frequency == 0:
        return False
    return iteration == 0 or iteration == max_iter - 1 or iteration % log_frequency == 0


def _sampled_pair_count(active_pairs, max_targets_per_source):
    if max_targets_per_source is None:
        return active_pairs.n_pairs
    max_targets_per_source = int(max_targets_per_source)
    return int(sum(min(len(targets), max_targets_per_source) for targets in active_pairs.targets))


def _sampled_sources(active_pairs, max_targets_per_source):
    if max_targets_per_source is None:
        return active_pairs.sources
    max_targets_per_source = int(max_targets_per_source)
    keep = [len(targets) > 0 and max_targets_per_source > 0 for targets in active_pairs.targets]
    return active_pairs.sources[np.asarray(keep, dtype=bool)]


def _geodesic_source_count(global_pairs, local_geodesic_pairs, max_global_targets_per_source):
    global_sources = _sampled_sources(global_pairs, max_global_targets_per_source)
    return len(np.union1d(global_sources, local_geodesic_pairs.sources))


def _full_stress_active_mask_and_denominator(D, W):
    active = (W != 0) & np.isfinite(D)
    np.fill_diagonal(active, False)
    denom = float(np.sum(W[active] * D[active] ** 2))
    return active, denom


def _full_geodesic_stress(
        X,
        D,
        W,
        *,
        metric,
        active_mask,
        denominator,
        graph_neighbors,
        neighbors_algorithm,
        n_jobs,
        ensure_connected_graph=False,
        warn_on_connect=False,
):
    support_graph = symmetric_knn_graph(
        X,
        n_neighbors=graph_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
        ensure_connected=ensure_connected_graph,
        warn_on_connect=warn_on_connect,
    )
    embedded = compute_embedding_distances(
        X,
        metric=metric,
        mode="geodesic",
        support_graph=support_graph,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    )
    active = active_mask & np.isfinite(embedded)
    if not np.array_equal(active, active_mask):
        return np.inf, np.inf

    residual = embedded[active] - D[active]
    raw_stress = float(np.sum(W[active] * residual ** 2))
    normalized = np.sqrt(raw_stress / denominator) if denominator > 0 else np.inf
    return raw_stress, float(normalized)


def _gpu_metric_supported(metric):
    if isinstance(metric, MatsumotoMetric) and metric.forbidden_grad_norm is not None:
        return False
    return isinstance(
        metric,
        (
            RandersMetric,
            MatsumotoMetric,
            ConvexifiedMatsumotoMetric,
            ConvexifiedToblerMetric,
        ),
    )


def _resolve_gpu_backend(device, metric, verbose):
    if device not in {"cpu", "auto", "gpu", "cuda"}:
        raise ValueError("device must be one of 'cpu', 'auto', 'gpu', or 'cuda'.")
    if device == "cpu":
        return None
    if not _gpu_metric_supported(metric):
        message = (
            "path_frozen GPU backend currently supports RandersMetric, "
            "MatsumotoMetric, ConvexifiedMatsumotoMetric, and "
            "ConvexifiedToblerMetric only."
        )
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise ValueError(message)

    cp, error = _load_cupy()
    if cp is None:
        message = f"path_frozen GPU backend unavailable: {error}"
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
        print(f"path_frozen GPU backend enabled on CUDA device {device_id}: {device_name}")
    return cp


def _cupy_metric_length_and_grad(cp, edge_vectors, metric):
    r = cp.linalg.norm(edge_vectors, axis=1)
    z = edge_vectors[:, -1]
    nonzero = r > 1e-12
    safe_r = cp.where(nonzero, r, 1.0)
    s = cp.where(nonzero, z / safe_r, 0.0)

    if isinstance(metric, RandersMetric):
        length = r + metric.alpha * z
        grad = cp.where(nonzero[:, None], edge_vectors / safe_r[:, None], 0.0)
        grad[:, -1] += metric.alpha
        grad = cp.where(nonzero[:, None], grad, 0.0)
        return length, grad

    if isinstance(metric, MatsumotoMetric):
        denominator = 1 - metric.alpha * s
        allowed = denominator > 0
        phi = cp.where(allowed, 1.0 / denominator, cp.inf)
        dphi = cp.where(allowed, metric.alpha / denominator**2, cp.nan)
        if metric.max_phi is not None:
            clipped = phi >= metric.max_phi
            phi = cp.minimum(phi, metric.max_phi)
            dphi = cp.where(clipped, 0.0, dphi)
        if metric.forbidden_grad_norm is not None and cp.any(~allowed & nonzero).item():
            raise ValueError(
                "The path_frozen GPU backend does not support MatsumotoMetric "
                "with forbidden_grad_norm. Use device='cpu' for this metric."
            )
    elif isinstance(metric, ConvexifiedMatsumotoMetric):
        if metric.alpha == 0:
            phi = cp.ones_like(s)
            dphi = cp.zeros_like(s)
        else:
            linear = s > 1 / (2 * metric.alpha)
            denominator = 1 - metric.alpha * s
            phi = cp.where(linear, 4 * metric.alpha * s, 1 / denominator)
            dphi = cp.where(linear, 4 * metric.alpha, metric.alpha / denominator**2)
    elif isinstance(metric, ConvexifiedToblerMetric):
        slope_denominator = cp.sqrt(cp.maximum(1 - s**2, 0.0))
        finite_slope = slope_denominator > 1e-12
        slope = cp.where(finite_slope, s / slope_denominator, cp.sign(s) * cp.inf)
        dslope = cp.where(finite_slope, 1.0 / slope_denominator**3, cp.inf)
        shifted = slope + metric.b
        base_phi = cp.exp(metric.a * cp.abs(shifted)) / metric.speed
        base_dphi = base_phi * metric.a * cp.sign(shifted) * dslope

        uphill = s > metric.s_uphill
        downhill = s < metric.s_downhill
        phi = cp.where(uphill, s / metric.z_max, base_phi)
        phi = cp.where(downhill, s / metric.z_min, phi)
        dphi = cp.where(uphill, 1.0 / metric.z_max, base_dphi)
        dphi = cp.where(downhill, 1.0 / metric.z_min, dphi)
    else:
        raise TypeError(f"Unsupported GPU metric {type(metric).__name__}.")

    length = r * phi
    coeff = phi - s * dphi
    direction = cp.where(nonzero[:, None], edge_vectors / safe_r[:, None], 0.0)
    grad = coeff[:, None] * direction
    grad[:, -1] += dphi
    grad = cp.where(nonzero[:, None], grad, 0.0)
    return length, grad


class _GpuPathFrozenObjective:
    def __init__(
            self,
            cp,
            *,
            shape,
            forest,
            active_pairs,
            metric,
            normalized_stress,
            max_path_edges,
            direct_pairs=None,
    ):
        self.cp = cp
        self.shape = shape
        self.metric = metric
        self.normalized_stress = normalized_stress
        direct_pairs = empty_active_pairs() if direct_pairs is None else direct_pairs
        self.denom = active_pairs.denom + direct_pairs.denom

        path_offsets, path_edge_ids, weights, dissimilarities = _flatten_active_paths(
            forest,
            active_pairs,
            max_path_edges=max_path_edges,
        )
        edge_tails = forest.edge_tails
        edge_heads = forest.edge_heads
        flat_direct = _flatten_pairs(direct_pairs)
        if flat_direct.n_pairs > 0:
            direct_edge_ids = np.arange(
                len(edge_tails),
                len(edge_tails) + flat_direct.n_pairs,
                dtype=np.int32,
            )
            path_offsets = np.concatenate([
                path_offsets,
                path_offsets[-1] + np.arange(1, flat_direct.n_pairs + 1, dtype=np.int64),
            ])
            path_edge_ids = np.concatenate([path_edge_ids, direct_edge_ids])
            weights = np.concatenate([weights, flat_direct.weights])
            dissimilarities = np.concatenate([dissimilarities, flat_direct.dissimilarities])
            edge_tails = np.concatenate([edge_tails, flat_direct.sources])
            edge_heads = np.concatenate([edge_heads, flat_direct.targets])
            if max_path_edges is not None and len(path_edge_ids) > max_path_edges:
                raise MemoryError(
                    "Flattened frozen paths exceed gpu_max_path_edges="
                    f"{max_path_edges}. Increase the limit or use device='cpu'."
                )
        path_counts = np.diff(path_offsets)
        path_ids = np.repeat(np.arange(len(path_counts), dtype=np.int32), path_counts)

        self.path_offsets = cp.asarray(path_offsets[:-1], dtype=cp.int64)
        self.path_edge_ids = cp.asarray(path_edge_ids, dtype=cp.int32)
        self.path_ids = cp.asarray(path_ids, dtype=cp.int32)
        self.weights = cp.asarray(weights, dtype=cp.float64)
        self.dissimilarities = cp.asarray(dissimilarities, dtype=cp.float64)
        self.edge_tails = cp.asarray(edge_tails, dtype=cp.int32)
        self.edge_heads = cp.asarray(edge_heads, dtype=cp.int32)
        self.n_edges = len(edge_tails)

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape))
        grad = cp.zeros_like(X)

        edge_vectors = X[self.edge_heads] - X[self.edge_tails]
        edge_lengths, edge_grads = _cupy_metric_length_and_grad(cp, edge_vectors, self.metric)
        if (
            not cp.all(cp.isfinite(edge_lengths)).item()
            or not cp.all(cp.isfinite(edge_grads)).item()
        ):
            raise ValueError(
                "The metric produced a non-finite edge length or gradient "
                "on a frozen shortest-path tree."
            )

        path_lengths = cp.add.reduceat(edge_lengths[self.path_edge_ids], self.path_offsets)
        residual = path_lengths - self.dissimilarities
        raw_stress = cp.sum(self.weights * residual**2)

        pair_coeff = 2.0 * self.weights * residual
        usage_coeff = pair_coeff[self.path_ids]
        edge_adjoint = cp.bincount(
            self.path_edge_ids,
            weights=usage_coeff,
            minlength=self.n_edges,
        )
        edge_contrib = edge_adjoint[:, None] * edge_grads
        cp.add.at(grad, self.edge_tails, -edge_contrib)
        cp.add.at(grad, self.edge_heads, edge_contrib)

        stress = raw_stress
        if self.normalized_stress:
            if self.denom <= 0:
                stress = cp.asarray(cp.inf)
                grad *= 0.0
            elif (raw_stress <= 0).item():
                stress = cp.asarray(0.0)
                grad *= 0.0
            else:
                stress = cp.sqrt(raw_stress / self.denom)
                grad *= 1.0 / (2.0 * cp.sqrt(raw_stress * self.denom))

        return float(stress.get()), cp.asnumpy(grad).ravel()


class _DirectPairsObjective:
    def __init__(self, *, shape, direct_pairs, metric):
        self.shape = shape
        self.metric = metric
        self.flat_pairs = _flatten_pairs(direct_pairs)

    def __call__(self, X_flat):
        X = X_flat.reshape(self.shape)
        grad = np.zeros_like(X)
        pairs = self.flat_pairs
        if pairs.n_pairs == 0:
            return 0.0, grad.ravel()

        vectors = X[pairs.targets] - X[pairs.sources]
        lengths = self.metric.length(vectors)
        edge_grads = self.metric.grad_u(vectors)
        if (
            not np.all(np.isfinite(lengths))
            or not np.all(np.isfinite(edge_grads))
        ):
            raise ValueError("The metric produced non-finite direct local-pair lengths or gradients.")

        residual = lengths - pairs.dissimilarities
        raw_stress = float(np.sum(pairs.weights * residual**2))
        contrib = (2.0 * pairs.weights * residual)[:, None] * edge_grads
        np.add.at(grad, pairs.sources, -contrib)
        np.add.at(grad, pairs.targets, contrib)
        return raw_stress, grad.ravel()


class _GpuDirectPairsObjective:
    def __init__(self, cp, *, shape, direct_pairs, metric):
        self.cp = cp
        self.shape = shape
        self.metric = metric
        pairs = _flatten_pairs(direct_pairs)
        self.n_pairs = pairs.n_pairs
        self.sources = cp.asarray(pairs.sources, dtype=cp.int32)
        self.targets = cp.asarray(pairs.targets, dtype=cp.int32)
        self.weights = cp.asarray(pairs.weights, dtype=cp.float64)
        self.dissimilarities = cp.asarray(pairs.dissimilarities, dtype=cp.float64)

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape))
        grad = cp.zeros_like(X)
        if self.n_pairs == 0:
            return 0.0, cp.asnumpy(grad).ravel()

        vectors = X[self.targets] - X[self.sources]
        lengths, edge_grads = _cupy_metric_length_and_grad(cp, vectors, self.metric)
        if (
            not cp.all(cp.isfinite(lengths)).item()
            or not cp.all(cp.isfinite(edge_grads)).item()
        ):
            raise ValueError("The metric produced non-finite direct local-pair lengths or gradients.")

        residual = lengths - self.dissimilarities
        raw_stress = cp.sum(self.weights * residual**2)
        contrib = (2.0 * self.weights * residual)[:, None] * edge_grads
        cp.add.at(grad, self.sources, -contrib)
        cp.add.at(grad, self.targets, contrib)
        return float(raw_stress.get()), cp.asnumpy(grad).ravel()


def _flatten_pairs(active_pairs):
    if active_pairs.n_pairs == 0:
        return _FlatPairs(
            sources=np.array([], dtype=int),
            targets=np.array([], dtype=int),
            weights=np.array([], dtype=float),
            dissimilarities=np.array([], dtype=float),
            n_pairs=0,
        )

    counts = np.fromiter((len(targets) for targets in active_pairs.targets), dtype=int)
    return _FlatPairs(
        sources=np.repeat(active_pairs.sources, counts).astype(int, copy=False),
        targets=np.concatenate(active_pairs.targets).astype(int, copy=False),
        weights=np.concatenate(active_pairs.weights).astype(float, copy=False),
        dissimilarities=np.concatenate(active_pairs.dissimilarities).astype(float, copy=False),
        n_pairs=active_pairs.n_pairs,
    )


def _flatten_active_paths(forest, active_pairs, *, max_path_edges):
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
                raise ValueError("Diagonal active pairs are not supported in flattened GPU paths.")
            if max_path_edges is not None and len(path_edge_ids) > max_path_edges:
                raise MemoryError(
                    "Flattened frozen paths exceed gpu_max_path_edges="
                    f"{max_path_edges}. Increase the limit or use device='cpu'."
                )
            path_offsets.append(len(path_edge_ids))
            weights.append(float(weight))
            dissimilarities.append(float(dissimilarity))

    return (
        np.asarray(path_offsets, dtype=np.int64),
        np.asarray(path_edge_ids, dtype=np.int32),
        np.asarray(weights, dtype=float),
        np.asarray(dissimilarities, dtype=float),
    )


def _path_frozen_stress_and_grad(
        X_flat,
        *,
        shape,
        forest,
        active_pairs,
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

    for source_pos, source in enumerate(forest.sources):
        edge_nodes = forest.edge_nodes[source_pos]
        parent = forest.parents[source_pos]
        edge_parents = parent[edge_nodes]
        edge_lengths = unique_edge_lengths[forest.edge_ids[source_pos]]
        edge_grads = unique_edge_grads[forest.edge_ids[source_pos]]

        path_lengths = np.zeros(n_samples, dtype=float)
        for node, parent_node, edge_length in zip(edge_nodes, edge_parents, edge_lengths):
            path_lengths[node] = path_lengths[parent_node] + edge_length

        targets = active_pairs.targets[source_pos]
        weights = active_pairs.weights[source_pos]
        dissimilarities = active_pairs.dissimilarities[source_pos]
        residual = path_lengths[targets] - dissimilarities
        raw_stress += np.sum(weights * residual ** 2)

        # Reverse-mode pass on the frozen tree. The adjoint at a vertex is the
        # sum of all residual contributions whose frozen path uses the edge
        # from its parent to that vertex.
        adjoint = np.zeros(n_samples, dtype=float)
        adjoint[targets] = 2.0 * weights * residual
        for node, parent_node, edge_grad in zip(edge_nodes[::-1], edge_parents[::-1], edge_grads[::-1]):
            scale = adjoint[node]
            if scale != 0:
                grad[parent_node] -= scale * edge_grad
                grad[node] += scale * edge_grad
            adjoint[parent_node] += scale

    stress = raw_stress
    if normalized_stress:
        if active_pairs.denom <= 0:
            stress = np.inf
            grad[:] = 0.0
        elif raw_stress <= 0:
            stress = 0.0
            grad[:] = 0.0
        else:
            stress = np.sqrt(raw_stress / active_pairs.denom)
            grad *= 1.0 / (2.0 * np.sqrt(raw_stress * active_pairs.denom))

    return float(stress), grad.ravel()


def _direct_pairs_stress_and_grad(X_flat, *, shape, direct_pairs, metric):
    return _DirectPairsObjective(
        shape=shape,
        direct_pairs=direct_pairs,
        metric=metric,
    )(X_flat)


def _normalize_stress_and_grad(raw_stress, grad_flat, denom, *, normalized_stress):
    if not normalized_stress:
        return float(raw_stress), grad_flat
    if denom <= 0:
        return np.inf, np.zeros_like(grad_flat)
    if raw_stress <= 0:
        return 0.0, np.zeros_like(grad_flat)
    stress = np.sqrt(raw_stress / denom)
    return float(stress), grad_flat * (1.0 / (2.0 * np.sqrt(raw_stress * denom)))


def _add_raw_objective(
        X_flat,
        raw_objective,
        *,
        shape,
        direct_pairs=None,
        direct_objective=None,
        metric=None,
        denom,
        normalized_stress,
):
    raw_stress = 0.0
    grad = np.zeros(shape, dtype=float).ravel()

    if raw_objective is not None:
        geodesic_stress, geodesic_grad = raw_objective(X_flat)
        raw_stress += float(geodesic_stress)
        grad += geodesic_grad

    if direct_objective is not None:
        direct_stress, direct_grad = direct_objective(X_flat)
        raw_stress += direct_stress
        grad += direct_grad
    elif direct_pairs is not None and direct_pairs.n_pairs > 0:
        if metric is None:
            raise ValueError("metric is required when direct_objective is not provided.")
        direct_stress, direct_grad = _direct_pairs_stress_and_grad(
            X_flat,
            shape=shape,
            direct_pairs=direct_pairs,
            metric=metric,
        )
        raw_stress += direct_stress
        grad += direct_grad

    denom_value = denom() if callable(denom) else denom
    return _normalize_stress_and_grad(
        raw_stress,
        grad,
        denom_value,
        normalized_stress=normalized_stress,
    )


def _pruned_tree_order(parents, source, targets):
    """Return a parent-before-child order restricted to active target paths."""
    source = int(source)
    included = {source}
    for target in targets:
        current = int(target)
        while current != source:
            parent = int(parents[current])
            if parent < 0:
                raise ValueError("An active target is unreachable in the Dijkstra predecessor tree.")
            included.add(current)
            included.add(parent)
            current = parent

    children = {node: [] for node in included}
    for node in included:
        if node == source:
            continue
        parent = int(parents[node])
        children[parent].append(node)

    order = []
    stack = [source]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])
    return np.asarray(order, dtype=int)


def _frozen_forest_from_embedding(
        X,
        *,
        metric,
        graph_neighbors,
        active_pairs,
        neighbors_algorithm,
        n_jobs,
        verbose=0,
):
    support = symmetric_knn_graph(
        X,
        n_neighbors=graph_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
        ensure_connected=True,
        warn_on_connect=verbose >= 1,
    )
    graph = metric_graph_from_support(X, support, metric)
    dist_matrix, predecessors = dijkstra(
        graph,
        directed=True,
        indices=active_pairs.sources,
        return_predecessors=True,
    )
    dist_matrix = np.atleast_2d(dist_matrix)
    predecessors = np.atleast_2d(predecessors)

    edge_nodes = []
    edge_ids = []
    edge_to_id = {}
    edge_tails = []
    edge_heads = []
    for source_pos, source in enumerate(active_pairs.sources):
        targets = active_pairs.targets[source_pos]
        if np.any(~np.isfinite(dist_matrix[source_pos, targets])):
            raise ValueError(
                "The current embedding graph has unreachable active pairs. "
                "Increase graph_neighbors, reduce the active-pair mask, or use a "
                "metric without infinite local edges."
            )

        order = _pruned_tree_order(predecessors[source_pos], source, targets)
        nodes = order[order != source]
        ids = np.empty(nodes.shape[0], dtype=int)
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

    forest = _FrozenForest(
        sources=active_pairs.sources,
        parents=predecessors,
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
    graph_neighbors=10,
    outer_iter=20,
    inner_iter=5,
    verbose=0,
    eps=1e-6,
    random_state=None,
    normalized_stress=False,
    weight=None,
    pair_mask=None,
    n_local_pairs=None,
    local_pair_mode="geodesic",
    landmark_indices=None,
    n_landmark=0,
    mask_random_state=None,
    targets_per_landmark=None,
    global_target_sampling="random",
    target_random_state=None,
    local_weight=1.0,
    local_global_reweighting="none",
    device="cpu",
    gpu_max_path_edges=100_000_000,
    method="L-BFGS-B",
    optimizer_options=None,
    log_frequency=None,
    record_history=False,
    neighbors_algorithm="auto",
    n_jobs=None,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with path-frozen graph geodesics.

    By default, all non-diagonal pairs with nonzero weight are active, matching
    the original full-stress behavior. For larger data sets, pass one or more
    sparse-pair options:

    ``n_local_pairs``
        Keep the closest target dissimilarities in each row. With the default
        ``local_pair_mode="geodesic"``, these local constraints use the frozen
        graph-geodesic objective. Pass ``local_pair_mode="direct"`` to use
        direct Finsler distances for local pairs instead.
    ``n_landmark`` or ``landmark_indices``
        Keep all valid outgoing pairs from selected landmark sources.
    ``pair_mask``
        Restrict all active-pair choices to a user-provided boolean mask.
    ``targets_per_landmark``
        At each outer iteration, sample at most this many global targets for
        each landmark source. Sampled weights are internally corrected so the
        global term estimates the full landmark-source objective.
    ``local_global_reweighting``
        ``"none"`` keeps the input weights. ``"count"`` balances ``sum w_ij``
        between local and global groups. ``"energy"`` balances
        ``sum w_ij D_ij^2``. ``local_weight`` then multiplies the local group.
    ``device``
        ``"cpu"`` keeps the historical implementation. ``"auto"`` uses a
        CuPy/CUDA stress-gradient backend when available and falls back to CPU.
        ``"gpu"``/``"cuda"`` require the CuPy backend to be available.
    ``log_frequency``
        Print one progress line every ``log_frequency`` outer iterations. The
        default adapts to ``outer_iter``: 1 below 30, 5 below 100, 10 below
        300, 50 below 1000, 100 below 3000, and so on. Pass 0 to suppress
        per-iteration progress lines. With ``verbose >= 2`` or
        ``record_history=True``, logged iterations also evaluate and record the
        full geodesic stress; this evaluation time is excluded from the
        reported elapsed optimization time.
    ``record_history``
        If True, record full-stress history at the same frequency as
        ``log_frequency`` without requiring terminal logs.
    """
    metric = validate_metric(metric)
    gpu_backend = _resolve_gpu_backend(device, metric, verbose)
    D, W = prepare_weights_and_mask(dissimilarities, weight)
    full_active_mask, full_denominator = _full_stress_active_mask_and_denominator(D, W)
    if mask_random_state is None:
        mask_random_state = random_state
    if target_random_state is None:
        target_random_state = random_state
    target_random_state = check_random_state(target_random_state)
    pair_groups = build_local_global_pairs(
        D,
        W,
        pair_mask=pair_mask,
        n_local_neighbors=n_local_pairs,
        local_pair_mode=local_pair_mode,
        landmark_indices=landmark_indices,
        n_global_landmarks=n_landmark,
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
    direct_objective = None
    if direct_pairs.n_pairs > 0:
        if gpu_backend is not None:
            direct_objective = _GpuDirectPairsObjective(
                gpu_backend,
                shape=shape,
                direct_pairs=direct_pairs,
                metric=metric,
            )
        else:
            direct_objective = _DirectPairsObjective(
                shape=shape,
                direct_pairs=direct_pairs,
                metric=metric,
            )

    options = {"maxiter": inner_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    optimizer_results = []
    history = []
    last_logged_outer_iter = None
    last_full_stress = None
    last_normalized_full_stress = None
    old_stress = None
    total_inner_iter = 0
    log_frequency = _resolve_log_frequency(log_frequency, outer_iter)
    optimization_start = perf_counter()
    logging_elapsed = 0.0

    if verbose:
        sampled_global_n_pairs = _sampled_pair_count(global_pairs, targets_per_landmark)
        n_geodesic_sources = _geodesic_source_count(
            global_pairs,
            local_geodesic_pairs,
            targets_per_landmark,
        )
        print(
            "path_frozen: "
            f"{sampled_global_n_pairs + local_pairs.n_pairs} pairs "
            f"({sampled_global_n_pairs} global, {local_pairs.n_pairs} local-{local_pair_mode}) "
            f"over "
            f"{D.shape[0] * (D.shape[0] - 1)} off-diagonal pairs; "
            f"{n_geodesic_sources} active sources"
        )
        if local_global_reweighting != "none" or local_weight != 1.0:
            print(
                "path_frozen pair weights: "
                f"reweighting={local_global_reweighting}, "
                f"global_factor={pair_groups.global_factor:.6g}, "
                f"local_factor={pair_groups.local_factor:.6g}"
            )
        if targets_per_landmark is not None:
            print(
                "path_frozen global target sampling: "
                f"targets_per_landmark={targets_per_landmark}, "
                f"global_target_sampling={global_target_sampling}, "
                f"global_sources={len(_sampled_sources(global_pairs, targets_per_landmark))}"
            )
        if log_frequency != 1:
            print(f"path_frozen logging every {log_frequency} outer iterations")

    for outer_it in range(outer_iter):
        iteration_global_pairs = sample_active_pairs(
            global_pairs,
            max_targets_per_source=targets_per_landmark,
            target_sampling=global_target_sampling,
            random_state=target_random_state,
        )
        iteration_pairs = merge_active_pairs(iteration_global_pairs, local_geodesic_pairs)

        raw_objective = None
        direct_in_raw_objective = False
        if iteration_pairs.n_pairs > 0:
            forest, _ = _frozen_forest_from_embedding(
                X,
                metric=metric,
                graph_neighbors=graph_neighbors,
                active_pairs=iteration_pairs,
                neighbors_algorithm=neighbors_algorithm,
                n_jobs=n_jobs,
                verbose=verbose,
            )

            if gpu_backend is not None:
                try:
                    raw_objective = _GpuPathFrozenObjective(
                        gpu_backend,
                        shape=shape,
                        forest=forest,
                        active_pairs=iteration_pairs,
                        metric=metric,
                        normalized_stress=False,
                        max_path_edges=gpu_max_path_edges,
                    )
                except MemoryError:
                    if device in {"gpu", "cuda"}:
                        raise
                    warnings.warn(
                        "Falling back to the CPU path_frozen objective because "
                        "the flattened paths are too large for the configured GPU limit.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    raw_objective = None
            else:
                direct_in_raw_objective = False

            if raw_objective is None:
                direct_in_raw_objective = False
                def raw_objective(x_flat):
                    return _path_frozen_stress_and_grad(
                        x_flat,
                        shape=shape,
                        forest=forest,
                        active_pairs=iteration_pairs,
                        metric=metric,
                        normalized_stress=False,
                    )

        def objective(x_flat):
            return _add_raw_objective(
                x_flat,
                raw_objective,
                shape=shape,
                direct_pairs=direct_pairs,
                direct_objective=None if direct_in_raw_objective else direct_objective,
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
        total_inner_iter += int(getattr(result, "nit", inner_iter))

        should_log = _should_log_iteration(outer_it, outer_iter, log_frequency)
        should_record_full = (record_history or verbose >= 2) and should_log
        should_print = verbose and should_log
        if should_record_full or should_print:
            nit = getattr(result, "nit", "?")
            nfev = getattr(result, "nfev", "?")
            elapsed = perf_counter() - optimization_start - logging_elapsed
            if should_record_full:
                log_start = perf_counter()
                full_stress, normalized_full_stress = _full_geodesic_stress(
                    X,
                    D,
                    W,
                    metric=metric,
                    active_mask=full_active_mask,
                    denominator=full_denominator,
                    graph_neighbors=graph_neighbors,
                    neighbors_algorithm=neighbors_algorithm,
                    n_jobs=n_jobs,
                    ensure_connected_graph=True,
                    warn_on_connect=verbose >= 1,
                )
                logging_elapsed += perf_counter() - log_start
                history.append(
                    {
                        "outer_iter": outer_it,
                        "elapsed": elapsed,
                        "masked_stress": stress,
                        "full_geodesic_stress": full_stress,
                        "normalized_full_geodesic_stress": normalized_full_stress,
                        "nit": nit,
                        "nfev": nfev,
                    }
                )
                last_logged_outer_iter = outer_it
                last_full_stress = full_stress
                last_normalized_full_stress = normalized_full_stress
            if verbose >= 2:
                print(
                    f"path_frozen outer {outer_it}: masked stress {stress}, "
                    f"full geodesic stress {full_stress}, "
                    f"normalized {normalized_full_stress} "
                    f"(elapsed={elapsed:.3f}s, nit={nit}, nfev={nfev})"
                )
            elif should_print:
                print(
                    f"path_frozen outer {outer_it}: masked stress {stress} "
                    f"(nit={nit}, nfev={nfev})"
                )

        if targets_per_landmark is None and old_stress is not None and old_stress != 0:
            if np.abs(1 - stress / old_stress) < eps:
                break
        old_stress = stress

    final_full_stress = None
    final_normalized_full_stress = None
    if verbose or record_history:
        if last_logged_outer_iter == outer_it:
            final_full_stress = last_full_stress
            final_normalized_full_stress = last_normalized_full_stress
        else:
            final_full_stress, final_normalized_full_stress = _full_geodesic_stress(
                X,
                D,
                W,
                metric=metric,
                active_mask=full_active_mask,
                denominator=full_denominator,
                graph_neighbors=graph_neighbors,
                neighbors_algorithm=neighbors_algorithm,
                n_jobs=n_jobs,
                ensure_connected_graph=True,
                warn_on_connect=verbose >= 1,
            )
        if verbose:
            displayed_full_stress = final_normalized_full_stress if normalized_stress else final_full_stress
            print(f"path_frozen final full geodesic stress: {displayed_full_stress}")

    pf_result = PathFrozenResult(
        embedding=X,
        stress=float(stress),
        n_iter=total_inner_iter,
        n_path_updates=outer_it + 1,
        optimizer_results=optimizer_results,
        history=history,
        final_full_geodesic_stress=final_full_stress,
        final_normalized_full_geodesic_stress=final_normalized_full_stress,
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
