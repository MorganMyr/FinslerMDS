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
    return graph.maximum(graph.T).tocsr()


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
    "metric_graph_from_support",
    "compute_metric_dist_matrix",
    "dijkstra_all_pairs",
    "predecessor_path_edges",
    "compute_dist_matrix",
]
