"""Path-frozen geodesic optimizer for Finsler-MDS.

The optimizer alternates between:
1. building a kNN graph on the current embedding and computing shortest-path
   trees from the active sources with Dijkstra;
2. freezing those paths and optimizing the stress for a few gradient-based
   steps, treating each path length as the sum of metric lengths of its edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize
from scipy.sparse.csgraph import dijkstra
from sklearn.utils import check_random_state

from finsler_mds.optimizers.common import (
    initial_embedding,
    prepare_weights_and_mask,
    validate_metric,
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


@dataclass(frozen=True)
class _FrozenForest:
    sources: np.ndarray
    parents: np.ndarray
    edge_nodes: list[np.ndarray]
    edge_ids: list[np.ndarray]
    edge_tails: np.ndarray
    edge_heads: np.ndarray


@dataclass(frozen=True)
class _ActivePairs:
    sources: np.ndarray
    targets: list[np.ndarray]
    weights: list[np.ndarray]
    dissimilarities: list[np.ndarray]
    sampleable: np.ndarray
    denom: float
    n_pairs: int


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


def _active_pairs_from_mask(D, W, active_mask, sampleable_sources):
    sources = np.flatnonzero(np.any(active_mask, axis=1))
    if len(sources) == 0:
        raise ValueError("No active pair remains for path_frozen optimization.")

    targets = []
    weights = []
    dissimilarities = []
    denom = 0.0
    n_pairs = 0
    for source in sources:
        source_targets = np.flatnonzero(active_mask[source])
        source_weights = W[source, source_targets].astype(float, copy=False)
        source_dissimilarities = D[source, source_targets].astype(float, copy=False)
        targets.append(source_targets)
        weights.append(source_weights)
        dissimilarities.append(source_dissimilarities)
        denom += float(np.sum(source_weights * source_dissimilarities ** 2))
        n_pairs += len(source_targets)

    return _ActivePairs(
        sources=sources.astype(int, copy=False),
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        sampleable=sampleable_sources[sources].astype(bool, copy=False),
        denom=denom,
        n_pairs=n_pairs,
    )


def _build_active_pairs(
        D,
        W,
        *,
        pair_mask,
        local_neighbors,
        landmark_indices,
        n_landmarks,
        landmark_mode,
        n_random_pairs,
        random_state,
):
    allowed = (W != 0) & np.isfinite(D)
    np.fill_diagonal(allowed, False)

    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != D.shape:
            raise ValueError("pair_mask must have the same shape as dissimilarities.")
        allowed &= pair_mask

    use_sparse_builder = (
        local_neighbors is not None
        or landmark_indices is not None
        or n_landmarks > 0
        or n_random_pairs > 0
    )

    if use_sparse_builder:
        active = np.zeros_like(allowed, dtype=bool)
        sampleable_sources = np.zeros(D.shape[0], dtype=bool)
        if local_neighbors is not None and local_neighbors > 0:
            _add_local_pairs(active, allowed, D, int(local_neighbors))
        landmarks = _select_landmarks(
            D.shape[0],
            landmark_indices=landmark_indices,
            n_landmarks=n_landmarks,
            random_state=random_state,
        )
        if len(landmarks) > 0:
            _add_landmark_pairs(active, allowed, landmarks, landmark_mode)
            if landmark_mode in {"sources", "both"}:
                sampleable_sources[landmarks] = True
        if n_random_pairs > 0:
            _add_random_pairs(active, allowed, int(n_random_pairs), random_state)
    else:
        active = allowed
        sampleable_sources = np.ones(D.shape[0], dtype=bool)

    return _active_pairs_from_mask(D, W, active, sampleable_sources)


def _add_local_pairs(active, allowed, D, local_neighbors):
    n_samples = D.shape[0]
    for source in range(n_samples):
        candidates = np.flatnonzero(allowed[source])
        if len(candidates) == 0:
            continue
        k = min(local_neighbors, len(candidates))
        distances = D[source, candidates]
        chosen = candidates[np.argpartition(distances, k - 1)[:k]]
        active[source, chosen] = True


def _select_landmarks(n_samples, *, landmark_indices, n_landmarks, random_state):
    if landmark_indices is not None:
        landmarks = np.asarray(landmark_indices, dtype=int)
        if landmarks.ndim != 1:
            raise ValueError("landmark_indices must be a 1D array-like.")
        if np.any((landmarks < 0) | (landmarks >= n_samples)):
            raise ValueError("landmark_indices contains an out-of-range index.")
        return np.unique(landmarks)

    if n_landmarks <= 0:
        return np.array([], dtype=int)

    rng = check_random_state(random_state)
    n_landmarks = min(int(n_landmarks), n_samples)
    return np.sort(rng.choice(n_samples, size=n_landmarks, replace=False))


def _add_landmark_pairs(active, allowed, landmarks, landmark_mode):
    if landmark_mode not in {"sources", "targets", "both"}:
        raise ValueError("landmark_mode must be 'sources', 'targets', or 'both'.")
    if landmark_mode in {"sources", "both"}:
        active[landmarks, :] = active[landmarks, :] | allowed[landmarks, :]
    if landmark_mode in {"targets", "both"}:
        active[:, landmarks] = active[:, landmarks] | allowed[:, landmarks]


def _add_random_pairs(active, allowed, n_random_pairs, random_state):
    rng = check_random_state(random_state)
    n_samples = allowed.shape[0]
    added = 0
    attempts = 0
    max_attempts = max(100, 20 * n_random_pairs)
    while added < n_random_pairs and attempts < max_attempts:
        attempts += 1
        source = rng.randint(n_samples)
        target = rng.randint(n_samples)
        if allowed[source, target] and not active[source, target]:
            active[source, target] = True
            added += 1

    if added >= n_random_pairs:
        return

    remaining = np.argwhere(allowed & ~active)
    if len(remaining) == 0:
        return
    chosen = rng.choice(len(remaining), size=min(n_random_pairs - added, len(remaining)), replace=False)
    active[remaining[chosen, 0], remaining[chosen, 1]] = True


def _sample_active_pairs(
        active_pairs,
        *,
        max_targets_per_source,
        target_sampling,
        random_state,
        rescale_sampled_weights,
):
    if max_targets_per_source is None:
        return active_pairs

    max_targets_per_source = int(max_targets_per_source)
    if max_targets_per_source <= 0:
        raise ValueError("max_targets_per_source must be positive or None.")
    if target_sampling not in {"random", "farthest", "mixed"}:
        raise ValueError("target_sampling must be 'random', 'farthest', or 'mixed'.")

    rng = check_random_state(random_state)
    targets = []
    weights = []
    dissimilarities = []
    denom = 0.0
    n_pairs = 0

    for source_pos in range(len(active_pairs.sources)):
        source_targets = active_pairs.targets[source_pos]
        source_weights = active_pairs.weights[source_pos]
        source_dissimilarities = active_pairs.dissimilarities[source_pos]
        n_available = len(source_targets)

        if active_pairs.sampleable[source_pos] and n_available > max_targets_per_source:
            chosen = _sample_target_indices(
                source_dissimilarities,
                max_targets_per_source,
                target_sampling,
                rng,
            )
            sampled_targets = source_targets[chosen]
            sampled_weights = source_weights[chosen].copy()
            sampled_dissimilarities = source_dissimilarities[chosen]
            if rescale_sampled_weights:
                sampled_weights *= n_available / max_targets_per_source
        else:
            sampled_targets = source_targets
            sampled_weights = source_weights
            sampled_dissimilarities = source_dissimilarities

        targets.append(sampled_targets)
        weights.append(sampled_weights)
        dissimilarities.append(sampled_dissimilarities)
        denom += float(np.sum(sampled_weights * sampled_dissimilarities ** 2))
        n_pairs += len(sampled_targets)

    return _ActivePairs(
        sources=active_pairs.sources,
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        sampleable=active_pairs.sampleable,
        denom=denom,
        n_pairs=n_pairs,
    )


def _sample_target_indices(dissimilarities, n_keep, target_sampling, rng):
    n_available = len(dissimilarities)
    if target_sampling == "random":
        chosen = rng.choice(n_available, size=n_keep, replace=False)
    elif target_sampling == "farthest":
        chosen = np.argpartition(dissimilarities, n_available - n_keep)[-n_keep:]
    else:
        n_far = n_keep // 2
        far = np.argpartition(dissimilarities, n_available - n_far)[-n_far:] if n_far > 0 else np.array([], dtype=int)
        remaining_mask = np.ones(n_available, dtype=bool)
        remaining_mask[far] = False
        remaining = np.flatnonzero(remaining_mask)
        n_random = n_keep - len(far)
        random = rng.choice(remaining, size=n_random, replace=False)
        chosen = np.concatenate([far, random])
    return np.sort(chosen)


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
        n_neighbors,
        active_pairs,
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
                "Increase n_neighbors, reduce the active-pair mask, or use a "
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


def _full_geodesic_stress(
        X,
        D,
        *,
        metric,
        n_neighbors,
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
    embedded = dijkstra(graph, directed=True, return_predecessors=False)
    active = np.isfinite(D)
    np.fill_diagonal(active, False)
    if np.any(active & ~np.isfinite(embedded)):
        return np.inf
    residual = embedded[active] - D[active]
    return float(np.sum(residual ** 2))


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
    pair_mask=None,
    local_neighbors=None,
    landmark_indices=None,
    n_landmarks=0,
    landmark_mode="sources",
    n_random_pairs=0,
    mask_random_state=None,
    max_targets_per_source=None,
    target_sampling="random",
    target_random_state=None,
    rescale_sampled_weights=True,
    method="L-BFGS-B",
    optimizer_options=None,
    neighbors_algorithm="auto",
    n_jobs=None,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with path-frozen graph geodesics.

    By default, all non-diagonal pairs with nonzero weight are active, matching
    the original full-stress behavior. For larger data sets, pass one or more
    sparse-pair options:

    ``local_neighbors``
        Keep the closest target dissimilarities in each row.
    ``n_landmarks`` or ``landmark_indices``
        Keep pairs involving selected landmarks. ``landmark_mode="sources"``
        only launches Dijkstra from landmarks and is the most scalable option.
    ``n_random_pairs``
        Add directed random pairs for extra long-range constraints.
    ``pair_mask``
        Restrict all active-pair choices to a user-provided boolean mask.
    ``max_targets_per_source``
        At each outer iteration, sample at most this many targets for each
        sampleable source. Landmark sources are sampleable; if no sparse mask
        is requested, all sources are treated as sampleable.
    """
    metric = validate_metric(metric)
    D, W = prepare_weights_and_mask(dissimilarities, weight)
    if mask_random_state is None:
        mask_random_state = random_state
    if target_random_state is None:
        target_random_state = random_state
    target_random_state = check_random_state(target_random_state)
    active_pairs = _build_active_pairs(
        D,
        W,
        pair_mask=pair_mask,
        local_neighbors=local_neighbors,
        landmark_indices=landmark_indices,
        n_landmarks=n_landmarks,
        landmark_mode=landmark_mode,
        n_random_pairs=n_random_pairs,
        random_state=mask_random_state,
    )
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

    if verbose:
        print(
            "path_frozen active pairs: "
            f"{active_pairs.n_pairs} over {D.shape[0] * (D.shape[0] - 1)} off-diagonal pairs; "
            f"{len(active_pairs.sources)} active sources"
        )
        if max_targets_per_source is not None:
            print(
                "path_frozen target sampling: "
                f"max_targets_per_source={max_targets_per_source}, "
                f"target_sampling={target_sampling}, "
                f"sampleable_sources={int(np.sum(active_pairs.sampleable))}"
            )

    for outer_it in range(max_iter):
        iteration_pairs = _sample_active_pairs(
            active_pairs,
            max_targets_per_source=max_targets_per_source,
            target_sampling=target_sampling,
            random_state=target_random_state,
            rescale_sampled_weights=rescale_sampled_weights,
        )
        forest, _ = _frozen_forest_from_embedding(
            X,
            metric=metric,
            n_neighbors=n_neighbors,
            active_pairs=iteration_pairs,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )

        def objective(x_flat):
            return _path_frozen_stress_and_grad(
                x_flat,
                shape=shape,
                forest=forest,
                active_pairs=iteration_pairs,
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
            print(f"path_frozen outer {outer_it}: masked stress {stress} ({iteration_pairs.n_pairs} sampled pairs)")

        if max_targets_per_source is None and old_stress is not None and old_stress != 0:
            if np.abs(1 - stress / old_stress) < eps:
                break
        old_stress = stress

    if verbose:
        full_stress = _full_geodesic_stress(
            X,
            D,
            metric=metric,
            n_neighbors=n_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )
        print(f"path_frozen final full stress: {full_stress}")

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
