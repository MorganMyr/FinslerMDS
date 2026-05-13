"""Directional RNA-velocity metrics on 2D or 3D embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class ClusterTransitionScore:
    source: object
    target: object
    score: float
    n_boundary_cells: int
    n_neighbor_pairs: int
    cell_indices: np.ndarray
    cell_scores: np.ndarray


@dataclass(frozen=True)
class CBDirResult:
    score: float
    transitions: dict[tuple[object, object], ClusterTransitionScore]


@dataclass(frozen=True)
class ClusterCoherenceScore:
    cluster: object
    score: float
    n_cells: int
    n_neighbor_pairs: int
    cell_indices: np.ndarray
    cell_scores: np.ndarray


@dataclass(frozen=True)
class ICVCohResult:
    score: float
    clusters: dict[object, ClusterCoherenceScore]


def project_velocity_graph_to_embedding(
        embedding,
        transition_graph,
        *,
        normalize_rows=True,
        normalize_displacements=True,
        positive_only=True,
        eps=1e-12,
):
    """Project transition weights to displacement vectors in an embedding.

    ``transition_graph[i, j]`` is interpreted as a transition strength from
    cell ``i`` to cell ``j``. Larger values should mean stronger transitions;
    distance/cost graphs should not be passed directly without conversion.
    """
    X = _validate_embedding(embedding)
    graph = scipy.sparse.csr_matrix(transition_graph, dtype=float)
    if graph.shape != (len(X), len(X)):
        raise ValueError(
            "transition_graph must have shape (n_samples, n_samples), "
            f"got {graph.shape} for n_samples={len(X)}."
        )

    velocities = np.zeros_like(X)
    for source in range(graph.shape[0]):
        start, end = graph.indptr[source], graph.indptr[source + 1]
        targets = graph.indices[start:end]
        weights = graph.data[start:end].astype(float, copy=True)
        if len(targets) == 0:
            continue
        if positive_only:
            keep = weights > 0
            targets = targets[keep]
            weights = weights[keep]
            if len(targets) == 0:
                continue
        if normalize_rows:
            total = float(np.sum(np.abs(weights)))
            if total <= 0:
                continue
            weights = weights / total
        displacements = X[targets] - X[source]
        if normalize_displacements:
            norms = np.linalg.norm(displacements, axis=1)
            valid = norms > eps
            if not np.any(valid):
                continue
            displacements = displacements[valid] / norms[valid, None]
            weights = weights[valid]
        velocities[source] = weights @ displacements
    return velocities


def cross_boundary_direction_correctness(
        embedding,
        labels,
        cluster_edges,
        *,
        velocity_vectors=None,
        transition_graph=None,
        n_neighbors=30,
        neighbor_indices=None,
        eps=1e-12,
):
    """Compute cross-boundary direction correctness (CBDir).

    For each expected transition ``source -> target``, source cells whose
    embedding-neighborhood touches the target cluster are treated as boundary
    cells. The score is the mean cosine similarity between each boundary
    velocity and displacements toward target-cluster neighbors.
    """
    X = _validate_embedding(embedding)
    labels = _validate_labels(labels, len(X))
    velocities = _resolve_embedding_velocities(
        X,
        velocity_vectors=velocity_vectors,
        transition_graph=transition_graph,
    )
    neighbors = _resolve_neighbors(
        X,
        n_neighbors=n_neighbors,
        neighbor_indices=neighbor_indices,
    )

    transition_scores = {}
    all_cell_scores = []
    for source, target in cluster_edges:
        score = _score_cross_boundary_transition(
            X,
            labels,
            velocities,
            neighbors,
            source=source,
            target=target,
            eps=eps,
        )
        transition_scores[(source, target)] = score
        all_cell_scores.append(score.cell_scores)

    score = _nanmean_concatenated(all_cell_scores)
    return CBDirResult(score=score, transitions=transition_scores)


def in_cluster_velocity_coherence(
        embedding,
        labels,
        *,
        velocity_vectors=None,
        transition_graph=None,
        clusters=None,
        n_neighbors=30,
        neighbor_indices=None,
        eps=1e-12,
):
    """Compute in-cluster velocity coherence (ICVCoh/ICCoh).

    For each cluster, the score is the mean cosine similarity between each
    cell's embedding velocity and the velocities of its same-cluster neighbors.
    """
    X = _validate_embedding(embedding)
    labels = _validate_labels(labels, len(X))
    velocities = _resolve_embedding_velocities(
        X,
        velocity_vectors=velocity_vectors,
        transition_graph=transition_graph,
    )
    neighbors = _resolve_neighbors(
        X,
        n_neighbors=n_neighbors,
        neighbor_indices=neighbor_indices,
    )
    if clusters is None:
        clusters = np.unique(labels)

    cluster_scores = {}
    all_cell_scores = []
    for cluster in clusters:
        score = _score_cluster_coherence(
            labels,
            velocities,
            neighbors,
            cluster=cluster,
            eps=eps,
        )
        cluster_scores[cluster] = score
        all_cell_scores.append(score.cell_scores)

    score = _nanmean_concatenated(all_cell_scores)
    return ICVCohResult(score=score, clusters=cluster_scores)


def _score_cross_boundary_transition(
        X,
        labels,
        velocities,
        neighbors,
        *,
        source,
        target,
        eps,
):
    source_cells = np.flatnonzero(labels == source)
    cell_indices = []
    cell_scores = []
    n_neighbor_pairs = 0

    for cell in source_cells:
        cell_neighbors = _valid_neighbors(neighbors[cell])
        target_neighbors = cell_neighbors[labels[cell_neighbors] == target]
        if len(target_neighbors) == 0:
            continue
        displacements = X[target_neighbors] - X[cell]
        cosines = _cosine_to_many(velocities[cell], displacements, eps=eps)
        finite = np.isfinite(cosines)
        if not np.any(finite):
            continue
        cell_indices.append(cell)
        cell_scores.append(float(np.mean(cosines[finite])))
        n_neighbor_pairs += int(np.count_nonzero(finite))

    cell_indices = np.asarray(cell_indices, dtype=int)
    cell_scores = np.asarray(cell_scores, dtype=float)
    return ClusterTransitionScore(
        source=source,
        target=target,
        score=float(np.mean(cell_scores)) if len(cell_scores) else np.nan,
        n_boundary_cells=len(cell_scores),
        n_neighbor_pairs=n_neighbor_pairs,
        cell_indices=cell_indices,
        cell_scores=cell_scores,
    )


def _score_cluster_coherence(labels, velocities, neighbors, *, cluster, eps):
    cluster_cells = np.flatnonzero(labels == cluster)
    cell_indices = []
    cell_scores = []
    n_neighbor_pairs = 0

    for cell in cluster_cells:
        cell_neighbors = _valid_neighbors(neighbors[cell])
        same_cluster_neighbors = cell_neighbors[labels[cell_neighbors] == cluster]
        same_cluster_neighbors = same_cluster_neighbors[same_cluster_neighbors != cell]
        if len(same_cluster_neighbors) == 0:
            continue
        cosines = _cosine_to_many(velocities[cell], velocities[same_cluster_neighbors], eps=eps)
        finite = np.isfinite(cosines)
        if not np.any(finite):
            continue
        cell_indices.append(cell)
        cell_scores.append(float(np.mean(cosines[finite])))
        n_neighbor_pairs += int(np.count_nonzero(finite))

    cell_indices = np.asarray(cell_indices, dtype=int)
    cell_scores = np.asarray(cell_scores, dtype=float)
    return ClusterCoherenceScore(
        cluster=cluster,
        score=float(np.mean(cell_scores)) if len(cell_scores) else np.nan,
        n_cells=len(cell_scores),
        n_neighbor_pairs=n_neighbor_pairs,
        cell_indices=cell_indices,
        cell_scores=cell_scores,
    )


def _validate_embedding(embedding):
    X = np.asarray(embedding, dtype=float)
    if X.ndim != 2 or X.shape[1] not in {2, 3}:
        raise ValueError("embedding must have shape (n_samples, 2) or (n_samples, 3).")
    return X


def _validate_labels(labels, n_samples):
    labels = np.asarray(labels)
    if labels.shape[0] != n_samples:
        raise ValueError(f"labels must have length {n_samples}, got {labels.shape[0]}.")
    return labels


def _resolve_embedding_velocities(X, *, velocity_vectors, transition_graph):
    if velocity_vectors is None:
        if transition_graph is None:
            raise ValueError("Pass either velocity_vectors or transition_graph.")
        return project_velocity_graph_to_embedding(X, transition_graph)
    velocities = np.asarray(velocity_vectors, dtype=float)
    if velocities.shape != X.shape:
        raise ValueError(
            "velocity_vectors must have the same shape as embedding, "
            f"got {velocities.shape} and {X.shape}."
        )
    return velocities


def _resolve_neighbors(X, *, n_neighbors, neighbor_indices):
    if neighbor_indices is not None:
        neighbors = np.asarray(neighbor_indices, dtype=int)
        if neighbors.ndim != 2 or neighbors.shape[0] != len(X):
            raise ValueError(
                "neighbor_indices must have shape (n_samples, n_neighbors)."
            )
        return neighbors

    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    if len(X) <= 1:
        return np.empty((len(X), 0), dtype=int)
    n_query = min(n_neighbors + 1, len(X))
    nbrs = NearestNeighbors(n_neighbors=n_query)
    nbrs.fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)
    cleaned = []
    for cell, row in enumerate(indices):
        row = row[row != cell]
        cleaned.append(row[:n_neighbors])
    max_len = max(len(row) for row in cleaned)
    out = np.full((len(X), max_len), -1, dtype=int)
    for cell, row in enumerate(cleaned):
        out[cell, :len(row)] = row
    return out


def _valid_neighbors(neighbors):
    neighbors = np.asarray(neighbors, dtype=int)
    return neighbors[neighbors >= 0]


def _cosine_to_many(vector, others, *, eps):
    others = np.asarray(others, dtype=float)
    vector_norm = float(np.linalg.norm(vector))
    other_norms = np.linalg.norm(others, axis=1)
    denom = vector_norm * other_norms
    cosines = np.full(len(others), np.nan, dtype=float)
    valid = denom > eps
    cosines[valid] = (others[valid] @ vector) / denom[valid]
    return np.clip(cosines, -1.0, 1.0)


def _nanmean_concatenated(arrays):
    arrays = [np.asarray(array, dtype=float) for array in arrays if len(array)]
    if not arrays:
        return np.nan
    values = np.concatenate(arrays)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
