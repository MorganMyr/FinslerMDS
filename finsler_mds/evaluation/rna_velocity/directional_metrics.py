"""Directional RNA-velocity metrics on 2D or 3D embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

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
class BoundaryTransitionNeighbors:
    source: object
    target: object
    cell_indices: np.ndarray
    target_indptr: np.ndarray
    target_indices: np.ndarray


@dataclass(frozen=True)
class BoundaryNeighborPlan:
    n_neighbors: int
    n_samples: int
    labels: np.ndarray
    transitions: dict[tuple[object, object], BoundaryTransitionNeighbors]
    neighbor_indices_hash: str | None = None


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


@dataclass(frozen=True)
class GVCohResult:
    score: float
    mean_direction: np.ndarray
    n_vectors: int


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
        boundary_plan=None,
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
    if boundary_plan is None:
        neighbors = _resolve_neighbors(
            X,
            n_neighbors=n_neighbors,
            neighbor_indices=neighbor_indices,
        )
        boundary_plan = build_boundary_neighbor_plan(
            labels,
            cluster_edges,
            neighbors,
            n_neighbors=neighbors.shape[1],
        )
    else:
        boundary_plan = _validate_boundary_plan(
            boundary_plan,
            labels=labels,
            cluster_edges=cluster_edges,
            n_neighbors=n_neighbors,
        )

    transition_scores = {}
    all_cell_scores = []
    for source, target in cluster_edges:
        score = _score_cross_boundary_transition_from_plan(
            X,
            velocities,
            boundary_plan.transitions[(source, target)],
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


def global_velocity_coherence(
        embedding,
        *,
        velocity_vectors=None,
        transition_graph=None,
        eps=1e-12,
):
    """Compute global velocity coherence as a mean resultant length.

    Each nonzero projected velocity is normalized to unit length, then the
    score is the norm of their mean vector. It is close to 1 when projected
    velocities share a common direction and close to 0 when they cancel out.
    """
    X = _validate_embedding(embedding)
    velocities = _resolve_embedding_velocities(
        X,
        velocity_vectors=velocity_vectors,
        transition_graph=transition_graph,
    )
    norms = np.linalg.norm(velocities, axis=1)
    valid = np.isfinite(norms) & (norms > eps)
    if not np.any(valid):
        return GVCohResult(score=np.nan, mean_direction=np.full(X.shape[1], np.nan), n_vectors=0)
    unit_velocities = velocities[valid] / norms[valid, None]
    mean_direction = np.mean(unit_velocities, axis=0)
    return GVCohResult(
        score=float(np.linalg.norm(mean_direction)),
        mean_direction=mean_direction,
        n_vectors=int(np.count_nonzero(valid)),
    )


def build_boundary_neighbor_plan(labels, cluster_edges, neighbor_indices, *, n_neighbors=None):
    """Precompute boundary cells and target-cluster neighbors for CBDir.

    The result depends on ``n_neighbors``. If a smaller value than the number of
    columns in ``neighbor_indices`` is supplied, only the first ``n_neighbors``
    columns are used.
    """
    labels = np.asarray(labels)
    neighbors = np.asarray(neighbor_indices, dtype=int)
    if neighbors.ndim != 2 or neighbors.shape[0] != len(labels):
        raise ValueError("neighbor_indices must have shape (n_samples, n_neighbors).")
    if n_neighbors is None:
        n_neighbors = neighbors.shape[1]
    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    if n_neighbors > neighbors.shape[1]:
        raise ValueError(
            f"Requested n_neighbors={n_neighbors}, but neighbor_indices only "
            f"has {neighbors.shape[1]} columns."
        )
    neighbors = neighbors[:, :n_neighbors]
    neighbor_indices_hash = _neighbor_indices_hash(neighbors)

    transitions = {}
    for source, target in cluster_edges:
        source_cells = np.flatnonzero(labels == source)
        cell_indices = []
        target_indptr = [0]
        target_indices = []
        for cell in source_cells:
            cell_neighbors = _valid_neighbors(neighbors[cell])
            target_neighbors = cell_neighbors[labels[cell_neighbors] == target]
            if len(target_neighbors) == 0:
                continue
            cell_indices.append(cell)
            target_indices.extend(target_neighbors.tolist())
            target_indptr.append(len(target_indices))
        transitions[(source, target)] = BoundaryTransitionNeighbors(
            source=source,
            target=target,
            cell_indices=np.asarray(cell_indices, dtype=int),
            target_indptr=np.asarray(target_indptr, dtype=int),
            target_indices=np.asarray(target_indices, dtype=int),
        )

    return BoundaryNeighborPlan(
        n_neighbors=n_neighbors,
        n_samples=len(labels),
        labels=np.asarray(labels, dtype=str),
        transitions=transitions,
        neighbor_indices_hash=neighbor_indices_hash,
    )


def save_boundary_neighbor_plan(path, plan):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = list(plan.transitions)
    arrays = {
        "labels": np.asarray(plan.labels, dtype=str),
        "transition_sources": np.asarray([source for source, _ in edges], dtype=str),
        "transition_targets": np.asarray([target for _, target in edges], dtype=str),
        "metadata_json": np.asarray(json.dumps({
            "n_neighbors": int(plan.n_neighbors),
            "n_samples": int(plan.n_samples),
            "n_transitions": len(edges),
            "neighbor_indices_hash": plan.neighbor_indices_hash,
        }, sort_keys=True)),
    }
    for idx, edge in enumerate(edges):
        transition = plan.transitions[edge]
        prefix = f"transition_{idx}"
        arrays[f"{prefix}_cell_indices"] = transition.cell_indices
        arrays[f"{prefix}_target_indptr"] = transition.target_indptr
        arrays[f"{prefix}_target_indices"] = transition.target_indices
    np.savez(path, **arrays)


def load_boundary_neighbor_plan(
        path,
        *,
        labels=None,
        cluster_edges=None,
        n_neighbors=None,
        neighbor_indices=None,
):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        cached_n_neighbors = int(metadata["n_neighbors"])
        if n_neighbors is not None and cached_n_neighbors != int(n_neighbors):
            raise ValueError(
                f"Boundary-neighbor cache was built with n_neighbors={cached_n_neighbors}, "
                f"but n_neighbors={int(n_neighbors)} was requested: {path}"
            )
        cached_neighbor_hash = metadata.get("neighbor_indices_hash")
        if neighbor_indices is not None and cached_neighbor_hash is not None:
            neighbors = np.asarray(neighbor_indices, dtype=int)
            if int(n_neighbors or cached_n_neighbors) > neighbors.shape[1]:
                raise ValueError(
                    f"Requested n_neighbors={int(n_neighbors or cached_n_neighbors)}, but "
                    f"neighbor_indices only has {neighbors.shape[1]} columns."
                )
            requested_hash = _neighbor_indices_hash(neighbors[:, :cached_n_neighbors])
            if requested_hash != cached_neighbor_hash:
                raise ValueError(f"Boundary-neighbor cache neighbor graph does not match: {path}")

        cached_labels = np.asarray(cache["labels"], dtype=str)
        if labels is not None and not np.array_equal(cached_labels, np.asarray(labels, dtype=str)):
            raise ValueError(f"Boundary-neighbor cache labels do not match requested labels: {path}")

        sources = np.asarray(cache["transition_sources"], dtype=str)
        targets = np.asarray(cache["transition_targets"], dtype=str)
        edges = [(source, target) for source, target in zip(sources, targets)]
        if cluster_edges is not None:
            requested_edges = [(str(source), str(target)) for source, target in cluster_edges]
            if edges != requested_edges:
                raise ValueError(
                    f"Boundary-neighbor cache transitions do not match requested transitions: {path}"
                )

        transitions = {}
        for idx, edge in enumerate(edges):
            prefix = f"transition_{idx}"
            source, target = edge
            transitions[edge] = BoundaryTransitionNeighbors(
                source=source,
                target=target,
                cell_indices=np.asarray(cache[f"{prefix}_cell_indices"], dtype=int),
                target_indptr=np.asarray(cache[f"{prefix}_target_indptr"], dtype=int),
                target_indices=np.asarray(cache[f"{prefix}_target_indices"], dtype=int),
            )

    return BoundaryNeighborPlan(
        n_neighbors=cached_n_neighbors,
        n_samples=len(cached_labels),
        labels=cached_labels,
        transitions=transitions,
        neighbor_indices_hash=cached_neighbor_hash,
    )


def load_or_compute_boundary_neighbor_plan(
        path,
        labels,
        *,
        cluster_edges,
        neighbor_indices,
        n_neighbors,
        on_mismatch="error",
):
    path = Path(path)
    if path.exists():
        try:
            return load_boundary_neighbor_plan(
                path,
                labels=labels,
                cluster_edges=cluster_edges,
                n_neighbors=n_neighbors,
                neighbor_indices=neighbor_indices,
            )
        except ValueError:
            if on_mismatch != "recompute":
                raise
    elif on_mismatch not in {"error", "recompute"}:
        raise ValueError("on_mismatch must be 'error' or 'recompute'.")

    plan = build_boundary_neighbor_plan(
        labels,
        cluster_edges,
        neighbor_indices,
        n_neighbors=n_neighbors,
    )
    save_boundary_neighbor_plan(path, plan)
    return plan


def _score_cross_boundary_transition_from_plan(
        X,
        velocities,
        transition,
        *,
        eps,
):
    cell_indices = []
    cell_scores = []
    n_neighbor_pairs = 0

    for pos, cell in enumerate(transition.cell_indices):
        start, end = transition.target_indptr[pos], transition.target_indptr[pos + 1]
        target_neighbors = transition.target_indices[start:end]
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
        source=transition.source,
        target=transition.target,
        score=float(np.mean(cell_scores)) if len(cell_scores) else np.nan,
        n_boundary_cells=len(cell_scores),
        n_neighbor_pairs=n_neighbor_pairs,
        cell_indices=cell_indices,
        cell_scores=cell_scores,
    )


def _validate_boundary_plan(plan, *, labels, cluster_edges, n_neighbors):
    if not isinstance(plan, BoundaryNeighborPlan):
        raise TypeError("boundary_plan must be a BoundaryNeighborPlan.")
    if plan.n_samples != len(labels):
        raise ValueError(
            f"boundary_plan has n_samples={plan.n_samples}, expected {len(labels)}."
        )
    if n_neighbors is not None and int(n_neighbors) != int(plan.n_neighbors):
        raise ValueError(
            f"boundary_plan was built with n_neighbors={plan.n_neighbors}, "
            f"but n_neighbors={int(n_neighbors)} was requested."
        )
    if not np.array_equal(np.asarray(labels, dtype=str), np.asarray(plan.labels, dtype=str)):
        raise ValueError("boundary_plan labels do not match labels.")
    missing = [edge for edge in cluster_edges if edge not in plan.transitions]
    if missing:
        raise ValueError(f"boundary_plan is missing transitions: {missing!r}.")
    return plan


def _neighbor_indices_hash(neighbor_indices):
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    contiguous = np.ascontiguousarray(neighbors)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


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
