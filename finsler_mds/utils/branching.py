"""Branching geodesic dataset and deterministic Path-Frozen initialization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.metrics import pairwise_distances


_INIT_MAIN_FACTOR = 0.82
_INIT_MAIN_RADIAL_DROP = 0.24
_INIT_SECONDARY_FACTOR = 0.46
_INIT_SECONDARY_RADIAL_DROP = 0.26
_INIT_NOISE_SCALE = 0.095


@dataclass(frozen=True)
class Segment:
    start: np.ndarray
    end: np.ndarray
    branch: int


@dataclass(frozen=True)
class BranchingProblem:
    X: np.ndarray
    graph: csr_matrix
    dissimilarities: np.ndarray
    branch_labels: np.ndarray
    init: np.ndarray


def load_branching_problem(
        cache_dir,
        *,
        n_samples=1000,
        graph_neighbors=10,
        corridor_width=0.11,
        seed=42,
        force=False,
):
    """Load or create the benchmark dataset and its fixed contracted init."""
    cache_path = Path(cache_dir) / (
        f"branching_n{n_samples}_k{graph_neighbors}_"
        f"w{_float_tag(corridor_width)}_s{seed}_dataset.npz"
    )
    if cache_path.exists() and not force:
        print(f"Loading cached branching dataset: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            graph = csr_matrix(
                (data["graph_data"], data["graph_indices"], data["graph_indptr"]),
                shape=tuple(data["graph_shape"]),
            )
            X = np.asarray(data["X"], dtype=float)
            D = np.asarray(data["D"], dtype=float)
            labels = np.asarray(data["branch_labels"], dtype=int)
    else:
        X, graph, D, labels = _make_dataset(
            n_samples, graph_neighbors, corridor_width, seed
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            D=D,
            branch_labels=labels,
            graph_data=graph.data,
            graph_indices=graph.indices,
            graph_indptr=graph.indptr,
            graph_shape=np.asarray(graph.shape, dtype=int),
        )
        print(f"Saved branching dataset: {cache_path}")

    init = _contracted_init(X, seed=seed + 17)
    return BranchingProblem(X, graph, D, labels, init)


def _make_dataset(n_samples, graph_neighbors, corridor_width, seed):
    print(f"Creating branching dataset: n={n_samples}, k={graph_neighbors}")
    rng = np.random.default_rng(seed)
    segments = _tree_segments()
    X, labels = _sample_tree(
        n_samples, segments, width=corridor_width, rng=rng
    )
    graph = _visible_knn_graph(
        X, segments, width=corridor_width, k=graph_neighbors
    )
    n_components, component_labels = connected_components(graph, directed=False)
    if n_components != 1:
        largest = np.bincount(component_labels).max()
        raise RuntimeError(
            f"Target graph has {n_components} components; largest has "
            f"{largest} points. Increase graph_neighbors or corridor_width."
        )
    D = shortest_path(graph, directed=False, return_predecessors=False)
    np.fill_diagonal(D, 0.0)
    return X, graph, D, labels


def _contracted_init(X, *, seed):
    secondary = _secondary_branch_mask(X)
    radius = np.linalg.norm(X, axis=1)
    radius_scale = radius / np.max(radius) if np.max(radius) > 0 else radius
    main = _INIT_MAIN_FACTOR - _INIT_MAIN_RADIAL_DROP * radius_scale
    secondary_factor = (
        _INIT_SECONDARY_FACTOR - _INIT_SECONDARY_RADIAL_DROP * radius_scale
    )
    factors = np.clip(np.where(secondary, secondary_factor, main), 0.12, None)
    noise_scale = _INIT_NOISE_SCALE * (0.45 + 0.9 * radius_scale)
    rng = np.random.default_rng(seed)
    return X * factors[:, None] + rng.normal(
        scale=noise_scale[:, None], size=X.shape
    )


def _secondary_branch_mask(X):
    segments = _tree_segments()
    primary = [s for s in segments if np.linalg.norm(s.start) < 1e-12]
    secondary = [s for s in segments if np.linalg.norm(s.start) >= 1e-12]

    def distance_to(group):
        return np.min(
            np.vstack([_point_segment_distances(X, s.start, s.end) for s in group]),
            axis=0,
        )

    return distance_to(secondary) < distance_to(primary)


def _tree_segments():
    root = np.zeros(2)
    segments = []
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + 0.15
    for branch, angle in enumerate(angles):
        direction = _unit_vector(angle)
        end = 3.0 * direction
        segments.append(Segment(root, end, branch))
        first, second = 0.58 * end, 0.82 * end
        for sign in (-1.0, 1.0):
            segments.append(
                Segment(
                    first,
                    first + 1.25 * _unit_vector(angle + sign * 0.55),
                    branch,
                )
            )
            segments.append(
                Segment(
                    second,
                    second + 0.85 * _unit_vector(angle + sign * 0.32),
                    branch,
                )
            )
    return segments


def _sample_tree(n_samples, segments, *, width, rng):
    lengths = np.array([np.linalg.norm(s.end - s.start) for s in segments])
    choices = rng.choice(len(segments), size=n_samples, p=lengths / lengths.sum())
    points = np.empty((n_samples, 2))
    labels = np.empty(n_samples, dtype=int)
    for i, segment_index in enumerate(choices):
        segment = segments[segment_index]
        vector = segment.end - segment.start
        tangent = vector / np.linalg.norm(vector)
        normal = np.array([-tangent[1], tangent[0]])
        t = rng.uniform(0.0, 1.0)
        offset = rng.uniform(-0.55 * width, 0.55 * width)
        points[i] = segment.start + t * vector + offset * normal
        labels[i] = segment.branch
    return points, labels


def _visible_knn_graph(X, segments, *, width, k):
    distances = pairwise_distances(X)
    rows, cols, weights = [], [], []
    for i in range(len(X)):
        visible = list(
            islice(
                (
                    j
                    for j in np.argsort(distances[i])
                    if i != j
                    and _segment_inside_tree(X[i], X[j], segments, width=width)
                ),
                k,
            )
        )
        if len(visible) < max(4, k // 2):
            raise RuntimeError(f"Point {i} only has {len(visible)} visible neighbors.")
        rows.extend([i] * len(visible))
        cols.extend(visible)
        weights.extend(distances[i, visible])

    directed = csr_matrix((weights, (rows, cols)), shape=(len(X), len(X)))
    graph = directed.maximum(directed.T).tolil()
    return _connect_components(
        graph, X, segments, width=width, distances=distances
    ).tocsr()


def _connect_components(graph, X, segments, *, width, distances):
    n_components, labels = connected_components(graph.tocsr(), directed=False)
    while n_components > 1:
        best = None
        source = np.flatnonzero(labels == 0)
        for component in range(1, n_components):
            target = np.flatnonzero(labels == component)
            sub_distances = distances[np.ix_(source, target)]
            for flat_index in np.argsort(sub_distances, axis=None):
                local_i, local_j = np.unravel_index(flat_index, sub_distances.shape)
                i, j = int(source[local_i]), int(target[local_j])
                if _segment_inside_tree(X[i], X[j], segments, width=width):
                    candidate = (float(distances[i, j]), i, j)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                    break
        if best is None:
            break
        distance, i, j = best
        graph[i, j] = graph[j, i] = distance
        n_components, labels = connected_components(graph.tocsr(), directed=False)
    return graph


def _segment_inside_tree(p, q, segments, *, width):
    t = np.linspace(0.0, 1.0, 25)[:, None]
    samples = (1.0 - t) * p + t * q
    inside = np.zeros(len(samples), dtype=bool)
    for segment in segments:
        inside |= (
            _point_segment_distances(samples, segment.start, segment.end) <= width
        )
    return bool(np.all(inside))


def _point_segment_distances(points, start, end):
    vector = end - start
    t = np.clip(((points - start) @ vector) / float(vector @ vector), 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * vector), axis=1)


def _unit_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def _float_tag(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


__all__ = ["BranchingProblem", "load_branching_problem"]
