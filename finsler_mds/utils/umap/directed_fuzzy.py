"""Directed fuzzy graph utilities adapted from UMAP's graph construction.

The public UMAP implementation builds local fuzzy neighborhoods with
``smooth_knn_dist`` and samples edges according to ``make_epochs_per_sample``.
Finsler-UMAP needs the same ingredients, but keeps the graph directed instead
of applying UMAP's fuzzy union symmetrization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
from sklearn.manifold import SpectralEmbedding
from umap.umap_ import smooth_knn_dist as umap_smooth_knn_dist


_EPS = 1e-12


@dataclass(frozen=True)
class DirectedFuzzyGraph:
    row: np.ndarray
    col: np.ndarray
    probability: np.ndarray
    graph_probability: np.ndarray
    rho: np.ndarray
    sigma: np.ndarray


def directed_fuzzy_graph_from_dense(
        dissimilarities,
        n_neighbors,
        *,
        weight=None,
        symmetrize_support=False,
        symmetrize_rho=True,
        symmetrize_sigma=True,
        local_connectivity=1.0,
        bandwidth=1.0,
):
    """Build directed UMAP-style edge probabilities from a dense distance matrix."""
    D = np.asarray(dissimilarities, dtype=float)
    indices, distances, edge_weight = directed_knn_from_dense(
        D,
        n_neighbors,
        weight=weight,
    )
    sigma, rho = _smooth_knn_dist(
        distances,
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
        bandwidth=float(bandwidth),
    )
    row, col, dist, active_edge_weight = directed_support_edges(
        D,
        indices,
        distances,
        edge_weight,
        weight=weight,
        symmetrize_support=bool(symmetrize_support),
    )
    row, col, graph_probability, active_edge_weight = directed_membership_strengths(
        row,
        col,
        dist,
        sigma,
        rho,
        edge_weight=active_edge_weight,
        symmetrize_rho=bool(symmetrize_rho),
        symmetrize_sigma=bool(symmetrize_sigma),
    )
    probability = graph_probability * active_edge_weight
    active = probability > 0
    if not np.any(active):
        raise ValueError("All directed Finsler-UMAP edge weights are zero.")
    return DirectedFuzzyGraph(
        row=row[active].astype(np.int64, copy=False),
        col=col[active].astype(np.int64, copy=False),
        probability=probability[active].astype(float, copy=False),
        graph_probability=graph_probability[active].astype(float, copy=False),
        rho=rho.astype(float, copy=False),
        sigma=sigma.astype(float, copy=False),
    )


def directed_knn_from_dense(dissimilarities, n_neighbors, *, weight=None):
    """Return outgoing non-self kNN rows from a dense directed distance matrix."""
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    D = np.asarray(dissimilarities, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")
    n_samples = D.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be smaller than the number of samples.")

    valid = np.isfinite(D)
    valid[np.arange(n_samples), np.arange(n_samples)] = False
    if weight is None:
        W = None
    else:
        W = np.asarray(weight, dtype=float)
        if W.shape != D.shape:
            raise ValueError("weight must have the same shape as dissimilarities.")
        valid &= W != 0

    indices = np.full((n_samples, n_neighbors), -1, dtype=np.int64)
    distances = np.full((n_samples, n_neighbors), np.inf, dtype=float)
    edge_weight = np.zeros((n_samples, n_neighbors), dtype=float)

    for i in range(n_samples):
        candidates = np.flatnonzero(valid[i])
        if len(candidates) == 0:
            continue
        candidate_dist = D[i, candidates]
        n_chosen = min(int(n_neighbors), len(candidates))
        keep = np.argpartition(candidate_dist, n_chosen - 1)[:n_chosen]
        keep = keep[np.argsort(candidate_dist[keep], kind="mergesort")]
        chosen = candidates[keep]
        indices[i, :n_chosen] = chosen
        distances[i, :n_chosen] = D[i, chosen]
        edge_weight[i, :n_chosen] = 1.0 if W is None else W[i, chosen]

    if not np.any(indices >= 0):
        raise ValueError("No finite directed neighbor edges were found.")
    return indices, distances, edge_weight


def directed_support_edges(
        dissimilarities,
        indices,
        distances,
        edge_weight,
        *,
        weight=None,
        symmetrize_support=False,
):
    D = np.asarray(dissimilarities, dtype=float)
    valid = indices >= 0
    row = np.repeat(np.arange(indices.shape[0]), indices.shape[1])[valid.ravel()]
    col = indices.ravel()[valid.ravel()]
    dist = distances.ravel()[valid.ravel()]
    active_weight = edge_weight.ravel()[valid.ravel()]

    if symmetrize_support:
        pairs = np.vstack((np.column_stack((row, col)), np.column_stack((col, row))))
        row, col = np.unique(pairs, axis=0).T
        dist = D[row, col]
        if weight is None:
            active_weight = np.ones(len(row), dtype=float)
        else:
            W = np.asarray(weight, dtype=float)
            active_weight = W[row, col]

    keep = (row != col) & np.isfinite(dist) & (active_weight != 0)
    return row[keep], col[keep], dist[keep], active_weight[keep]


def _smooth_knn_dist(distances, k, *, local_connectivity, bandwidth):
    """Call UMAP with the leading self-distance column it expects."""
    distances = np.asarray(distances, dtype=float)
    sigma = np.ones(distances.shape[0], dtype=float)
    rho = np.zeros(distances.shape[0], dtype=float)

    complete = np.all(np.isfinite(distances), axis=1)
    if np.any(complete):
        with_self = np.column_stack(
            (np.zeros(np.count_nonzero(complete)), distances[complete])
        )
        sigma[complete], rho[complete] = umap_smooth_knn_dist(
            with_self, k, local_connectivity=local_connectivity, bandwidth=bandwidth
        )

    for i in np.flatnonzero(~complete):
        finite = distances[i, np.isfinite(distances[i])]
        if len(finite) == 0:
            continue
        with_self = np.concatenate(([0.0], finite))[None, :]
        sigma[i : i + 1], rho[i : i + 1] = umap_smooth_knn_dist(
            with_self, k, local_connectivity=local_connectivity, bandwidth=bandwidth
        )
    return sigma, rho


def directed_membership_strengths(
        row,
        col,
        dist,
        sigma,
        rho,
        *,
        edge_weight,
        symmetrize_rho,
        symmetrize_sigma,
):
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    dist = np.asarray(dist, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    rho = np.asarray(rho, dtype=float)
    edge_weight = np.asarray(edge_weight, dtype=float)

    edge_rho = 0.5 * (rho[row] + rho[col]) if symmetrize_rho else rho[row]
    if symmetrize_sigma:
        edge_sigma = _harmonic_mean(sigma[row], sigma[col])
    else:
        edge_sigma = sigma[row]

    probability = np.exp(-np.maximum(0.0, dist - edge_rho) / np.maximum(edge_sigma, _EPS))
    probability[dist - edge_rho <= 0.0] = 1.0
    return row, col, probability, edge_weight


def spectral_initial_embedding(row, col, probability, n_samples, n_components, rng):
    """UMAP-style spectral initialization on the symmetrized fuzzy graph."""
    if n_samples <= n_components + 1:
        return rng.normal(scale=1e-4, size=(n_samples, n_components))

    graph = symmetric_fuzzy_union(row, col, probability, n_samples)
    try:
        from umap.spectral import spectral_layout

        embedding = spectral_layout(
            np.zeros((n_samples, 1), dtype=float),
            graph,
            n_components,
            rng,
            metric="euclidean",
            metric_kwds={},
        )
        embedding = np.asarray(embedding, dtype=float)
    except Exception:
        embedding = SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",
            eigen_solver="arpack",
            random_state=rng,
        ).fit_transform(graph)
        embedding = np.asarray(embedding, dtype=float)

    embedding = _scale_coords(embedding, max_coord=10.0)
    return embedding + rng.normal(scale=1e-4, size=embedding.shape)


def symmetric_fuzzy_union(row, col, probability, n_samples):
    directed = scipy.sparse.coo_matrix(
        (probability, (row, col)),
        shape=(n_samples, n_samples),
    ).tocsr()
    directed.eliminate_zeros()
    transpose = directed.T.tocsr()
    graph = directed + transpose - directed.multiply(transpose)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def _harmonic_mean(a, b):
    denom = a + b
    return np.divide(
        2.0 * a * b,
        denom,
        out=np.full_like(denom, _EPS, dtype=float),
        where=denom > _EPS,
    )


def _scale_coords(coords, *, max_coord):
    coords = np.asarray(coords, dtype=float)
    max_abs = float(np.max(np.abs(coords))) if coords.size else 0.0
    if max_abs <= _EPS:
        return coords.copy()
    return coords * (float(max_coord) / max_abs)
