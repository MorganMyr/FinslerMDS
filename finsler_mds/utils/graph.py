import warnings

import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.utils.graph import _fix_connected_components


def nearest_neighbors(X, k):
    # we use k+1 here since Xi will have the shortest distance to itself
    knn_matrix = np.zeros((len(X), k))
    # compute pairwise distances
    dist_matrix = pairwise_distances(X)
    # for each row find indices of k nearest neighbors
    for i in range(len(X)):
        knn_matrix[i] = dist_matrix[i, :].argsort()[1:k + 1]
    return knn_matrix


def symmetric_knn_graph(
        X,
        n_neighbors=5,
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        ensure_connected=False,
        warn_on_connect=False,
):
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=neighbors_algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    nbrs.fit(X)
    graph = kneighbors_graph(
        nbrs,
        n_neighbors,
        metric=metric,
        p=p,
        metric_params=metric_params,
        mode="distance",
        n_jobs=n_jobs,
    )
    graph = graph.maximum(graph.T).tocsr()
    if ensure_connected:
        n_components, labels = connected_components(graph, directed=False)
        if n_components > 1:
            if warn_on_connect:
                print(f"Warning: reconnected embedding kNN graph ({n_components} components).")
            graph = _fix_connected_components(
                X=nbrs._fit_X,
                graph=graph.tolil(),
                n_connected_components=n_components,
                component_labels=labels,
                mode="distance",
                metric=nbrs.effective_metric_,
                **nbrs.effective_metric_params_,
            )
            graph = graph.maximum(graph.T).tocsr()
    return graph


def softmin_with_probs(values, beta, axis=-1, prob_dtype=None):
    """Return softmin values and the associated probabilities.

    The soft minimum is ``-log(sum(exp(-beta*x))) / beta``. Non-finite
    candidates are ignored; if all candidates are infinite, the softmin is
    infinite and all probabilities are zero.
    """
    values = np.asarray(values, dtype=float)
    if beta <= 0:
        raise ValueError("beta must be positive.")

    finite = np.isfinite(values)
    any_finite = np.any(finite, axis=axis, keepdims=True)
    shifted_min = np.min(np.where(finite, values, np.inf), axis=axis, keepdims=True)

    scores = np.zeros_like(values, dtype=float)
    broadcast_min = np.broadcast_to(shifted_min, values.shape)
    valid = finite & np.broadcast_to(any_finite, values.shape)
    scores[valid] = np.exp(-beta * (values[valid] - broadcast_min[valid]))

    denom = np.sum(scores, axis=axis, keepdims=True)
    probs = np.divide(scores, denom, out=np.zeros_like(scores), where=denom > 0)

    soft = np.full(np.squeeze(shifted_min, axis=axis).shape, np.inf, dtype=float)
    finite_out = np.squeeze(any_finite, axis=axis)
    soft[finite_out] = (
        np.squeeze(shifted_min, axis=axis)[finite_out]
        - np.log(np.squeeze(denom, axis=axis)[finite_out]) / beta
    )
    if prob_dtype is not None:
        probs = probs.astype(prob_dtype, copy=False)
    return soft, probs


def metric_graph_from_support(X, support_graph, finsler_metric):
    support = support_graph.tocoo()
    edge_vectors = X[support.col] - X[support.row]
    edge_lengths = finsler_metric.length(edge_vectors)
    finite = np.isfinite(edge_lengths)
    return scipy.sparse.csr_matrix(
        (edge_lengths[finite], (support.row[finite], support.col[finite])),
        shape=support_graph.shape,
    )


def compute_metric_dist_matrix(
        X,
        finsler_metric,
        support_graph=None,
        n_neighbors=5,
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        directed=True,
):
    if support_graph is None:
        support_graph = symmetric_knn_graph(
            X,
            n_neighbors=n_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            metric=metric,
            p=p,
            metric_params=metric_params,
        )
    graph = metric_graph_from_support(X, support_graph, finsler_metric)
    return dijkstra(
        graph,
        directed=directed,
        return_predecessors=True,
    )


def velocity_directed_graph(
        X,
        velocity,
        n_neighbors=None,
        kNN_euclid=30,
        kNN_finsler=0,
        alpha=1.0,
        distance_formula="exponential",
        velocity_neighbors=None,
        average_velocity=True,
        symmetrize_support=True,
        cos_clip=None,
        neighbors_algorithm="auto",
        n_jobs=None,
):
    """Build a directed kNN graph whose edge weights follow a velocity field.

    For an edge ``i -> j``, ``theta`` is the angle between the local velocity at
    ``i`` and ``X_j - X_i``. Supported formulas are:

    - ``"exponential"``: ``||X_j - X_i|| * exp(-alpha * cos(theta))``.
    - ``"randers"``: ``||X_j - X_i|| * (1 - alpha * cos(theta))``.
    - ``"matsumoto"``: ``||X_j - X_i|| / (1 + alpha * cos(theta))``.

    If ``cos_clip`` is not ``None``, cosines are clipped to
    ``[-cos_clip, cos_clip]`` before applying either formula.

    Moving with the velocity is cheaper, moving against it is more expensive.
    """
    if n_neighbors is not None:
        kNN_euclid = n_neighbors
    X = np.asarray(X, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if X.shape != velocity.shape:
        raise ValueError("X and velocity must have the same shape.")
    distance_formula = _normalize_velocity_distance_formula(distance_formula)
    if cos_clip is not None:
        cos_clip = float(cos_clip)
        if not 0 <= cos_clip <= 1:
            raise ValueError("cos_clip must be None or a float in [0, 1].")
    if distance_formula in {"randers", "matsumoto"}:
        max_cos = 1.0 if cos_clip is None else cos_clip
        if alpha < 0 or alpha * max_cos >= 1:
            raise ValueError(
                f"{distance_formula.title()} velocity distances require alpha >= 0 and "
                "alpha * max(|cos|) < 1. Lower alpha or set a smaller cos_clip."
            )

    support = symmetric_knn_graph(
        X,
        n_neighbors=kNN_euclid,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    )
    if not symmetrize_support:
        nbrs = NearestNeighbors(
            n_neighbors=kNN_euclid,
            algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )
        nbrs.fit(X)
        support = kneighbors_graph(nbrs, kNN_euclid, mode="distance", n_jobs=n_jobs).tocsr()

    velocity_used = velocity
    if average_velocity:
        velocity_used = average_vectors_on_graph(
            velocity,
            support,
            include_self=True,
            n_neighbors=velocity_neighbors,
        )

    if kNN_finsler:
        finsler_support = _velocity_finsler_knn_support(
            X,
            velocity_used,
            n_neighbors=kNN_finsler,
            alpha=alpha,
            distance_formula=distance_formula,
            cos_clip=cos_clip,
        )
        support = support.maximum(finsler_support).tocsr()
        if symmetrize_support:
            support = support.maximum(support.T).tocsr()

    support_coo = support.tocoo()
    edge_vectors = X[support_coo.col] - X[support_coo.row]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    source_velocity = velocity_used[support_coo.row]
    velocity_norms = np.linalg.norm(source_velocity, axis=1)

    denom = velocity_norms * edge_lengths
    cosines = np.divide(
        np.sum(source_velocity * edge_vectors, axis=1),
        denom,
        out=np.zeros_like(edge_lengths),
        where=denom > 1e-12,
    )
    cosines = np.clip(cosines, -1.0, 1.0)
    if cos_clip is not None:
        cosines = np.clip(cosines, -cos_clip, cos_clip)
    if distance_formula == "exponential":
        edge_weights = edge_lengths * np.exp(-alpha * cosines)
    elif distance_formula == "randers":
        edge_weights = edge_lengths * (1 - alpha * cosines)
    elif distance_formula == "matsumoto":
        edge_weights = edge_lengths / (1 + alpha * cosines)
    else:  # pragma: no cover - guarded by normalization
        raise RuntimeError(f"Unhandled velocity distance formula {distance_formula!r}.")

    graph = scipy.sparse.csr_matrix(
        (edge_weights, (support_coo.row, support_coo.col)),
        shape=support.shape,
    )
    return graph, velocity_used


def _velocity_finsler_knn_support(
        X,
        velocity,
        *,
        n_neighbors,
        alpha,
        distance_formula,
        cos_clip,
        batch_size=64,
):
    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        return scipy.sparse.csr_matrix((len(X), len(X)))
    n_samples = len(X)
    if n_neighbors >= n_samples:
        raise ValueError("kNN_finsler must be smaller than the number of samples.")

    rows = []
    cols = []
    data = []
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        vectors = X[None, :, :] - X[start:stop, None, :]
        lengths = np.linalg.norm(vectors, axis=2)
        source_velocity = velocity[start:stop]
        velocity_norms = np.linalg.norm(source_velocity, axis=1)
        denom = velocity_norms[:, None] * lengths
        cosines = np.divide(
            np.einsum("bd,bnd->bn", source_velocity, vectors),
            denom,
            out=np.zeros_like(lengths),
            where=denom > 1e-12,
        )
        cosines = np.clip(cosines, -1.0, 1.0)
        if cos_clip is not None:
            cosines = np.clip(cosines, -cos_clip, cos_clip)
        if distance_formula == "exponential":
            weights = lengths * np.exp(-alpha * cosines)
        elif distance_formula == "randers":
            weights = lengths * (1 - alpha * cosines)
        elif distance_formula == "matsumoto":
            weights = lengths / (1 + alpha * cosines)
        else:  # pragma: no cover - guarded by normalization
            raise RuntimeError(f"Unhandled velocity distance formula {distance_formula!r}.")
        weights[np.arange(stop - start), np.arange(start, stop)] = np.inf
        selected = np.argpartition(weights, n_neighbors - 1, axis=1)[:, :n_neighbors]
        batch_rows = np.repeat(np.arange(start, stop), n_neighbors)
        batch_cols = selected.reshape(-1)
        rows.append(batch_rows)
        cols.append(batch_cols)
        data.append(weights[np.arange(stop - start)[:, None], selected].reshape(-1))

    return scipy.sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_samples, n_samples),
    )


def average_vectors_on_graph(vectors, graph, include_self=True, n_neighbors=None):
    """Average vectors over graph outgoing neighborhoods."""
    vectors = np.asarray(vectors, dtype=float)
    graph = graph.tocsr()
    averaged = np.zeros_like(vectors)

    for i in range(graph.shape[0]):
        neighbors = graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
        if n_neighbors is not None and len(neighbors) > n_neighbors:
            distances = graph.data[graph.indptr[i]:graph.indptr[i + 1]]
            keep = np.argpartition(distances, int(n_neighbors) - 1)[:int(n_neighbors)]
            neighbors = neighbors[keep]
        if include_self:
            neighbors = np.concatenate(([i], neighbors))
        if len(neighbors) == 0:
            averaged[i] = vectors[i]
        else:
            averaged[i] = np.mean(vectors[neighbors], axis=0)

    return averaged


def compute_velocity_dist_matrix(
        X,
        velocity,
        n_neighbors=None,
        kNN_euclid=30,
        kNN_finsler=0,
        alpha=1.0,
        distance_formula="exponential",
        velocity_neighbors=None,
        average_velocity=True,
        symmetrize_support=True,
        cos_clip=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
):
    """Compute directed shortest-path distances induced by a velocity field."""
    graph, velocity_used = velocity_directed_graph(
        X,
        velocity,
        n_neighbors=n_neighbors,
        kNN_euclid=kNN_euclid,
        kNN_finsler=kNN_finsler,
        alpha=alpha,
        distance_formula=distance_formula,
        velocity_neighbors=velocity_neighbors,
        average_velocity=average_velocity,
        symmetrize_support=symmetrize_support,
        cos_clip=cos_clip,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    )
    dist_matrix, predecessors = shortest_path(
        graph,
        method=path_method,
        directed=True,
        return_predecessors=True,
    )
    return dist_matrix, predecessors, graph, velocity_used


def _normalize_velocity_distance_formula(distance_formula):
    if not isinstance(distance_formula, str):
        raise TypeError("distance_formula must be 'exponential', 'randers', or 'matsumoto'.")
    formula = distance_formula.lower()
    aliases = {
        "exp": "exponential",
        "exponential": "exponential",
        "softmax": "exponential",
        "randers": "randers",
        "local_randers": "randers",
        "linear_randers": "randers",
        "mats": "matsumoto",
        "matsumoto": "matsumoto",
        "local_matsumoto": "matsumoto",
    }
    if formula not in aliases:
        raise ValueError("distance_formula must be 'exponential', 'randers', or 'matsumoto'.")
    return aliases[formula]


def dijkstra_all_pairs(graph, directed=True):
    return dijkstra(
        graph,
        directed=directed,
        return_predecessors=True,
    )


def predecessor_path_edges(predecessors, source, target):
    if source == target:
        return []

    edges = []
    current = target
    while current != source:
        previous = predecessors[source, current]
        if previous < 0:
            return None
        edges.append((previous, current))
        current = previous
    edges.reverse()
    return edges


def compute_dist_matrix(
        X,
        n_neighbors=5,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=None,
        nn_riemannian_precomputed=None,
):
    nbrs_ = None
    if nn_riemannian_precomputed is None:
        # Compute in the same way as in Isomap and change at the end the edge distances
        nbrs_ = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=neighbors_algorithm,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        nbrs_.fit(X)

        if n_neighbors is not None:
            nbg = kneighbors_graph(
                nbrs_,
                n_neighbors,
                metric=metric,
                p=p,
                metric_params=metric_params,
                mode="distance",
                n_jobs=n_jobs,
            )
        else:
            nbg = radius_neighbors_graph(
                nbrs_,
                radius=radius,
                metric=metric,
                p=p,
                metric_params=metric_params,
                mode="distance",
                n_jobs=n_jobs,
            )
    else:
        if not scipy.sparse.issparse(nn_riemannian_precomputed):
            raise TypeError("nn_riemannian_precomputed must be a scipy sparse matrix.")
        nbg = nn_riemannian_precomputed

    # Compute the number of connected components, and connect the different
    # components to be able to compute a shortest path between all pairs.
    n_connected_components, labels = connected_components(nbg)
    if n_connected_components > 1:
        if metric == "precomputed" and issparse(X):
            raise RuntimeError(
                "The number of connected components of the neighbors graph"
                f" is {n_connected_components} > 1. The graph cannot be "
                "completed with metric='precomputed', and Isomap cannot be"
                "fitted. Increase the number of neighbors to avoid this "
                "issue, or precompute the full distance matrix instead "
                "of passing a sparse neighbors graph."
            )
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " Isomap might be slow. Increase the number of neighbors to "
                "avoid this issue."
            ),
            stacklevel=2,
        )

        nbg = _fix_connected_components(
            X=nbrs_._fit_X,
            graph=nbg,
            n_connected_components=n_connected_components,
            component_labels=labels,
            mode="distance",
            metric=nbrs_.effective_metric_,
            **nbrs_.effective_metric_params_,
        )

    # Update the nbg graph with the Randers field.
    # Modification formula is:
    # d(x, y) = d(x, y) + <randers_field, y - x>
    if randers_field is not None:
        # The Randers lengths are directed, but the local neighborhood support
        # should still represent an undirected manifold adjacency. Without this
        # symmetrization, directed shortest paths can contain unreachable pairs
        # even when the underlying kNN graph is connected.
        nbg = nbg.maximum(nbg.T)
        edges_mask = nbg.toarray() != 0
        for i in range(len(X)):
            randers_update = np.dot(X - X[i], randers_field[i]) * edges_mask[i]
            nbg[i, edges_mask[i]] = nbg[i, edges_mask[i]] + randers_update[edges_mask[i]]
        nbg = nbg.tocsr()
        directed = True
    else:
        directed = False

    dist_matrix_, preds_ = shortest_path(
        nbg,
        method=path_method,
        directed=directed,
        return_predecessors=True,
    )

    if nbrs_ is not None and nbrs_._fit_X.dtype == np.float32:
        dist_matrix_ = dist_matrix_.astype(nbrs_._fit_X.dtype, copy=False)

    return dist_matrix_, preds_


__all__ = [
    "nearest_neighbors",
    "symmetric_knn_graph",
    "softmin_with_probs",
    "average_vectors_on_graph",
    "velocity_directed_graph",
    "compute_velocity_dist_matrix",
    "metric_graph_from_support",
    "compute_metric_dist_matrix",
    "dijkstra_all_pairs",
    "predecessor_path_edges",
    "compute_dist_matrix",
]
