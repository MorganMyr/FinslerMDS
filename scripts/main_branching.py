"""Path-frozen tests for a branching geodesic dataset.

The target dissimilarities are shortest-path distances on a thin 2D branching
tree. Path-frozen starts from a deliberately deformed version of the ground
truth geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.metrics import pairwise_distances

from finsler_mds import RandersMetric, fit_finsler_mds, geodesic_embedding_stress


SEED = 42
N_SAMPLES = 1000
GRAPH_NEIGHBORS = 10
N_COMPONENTS = 2
CORRIDOR_WIDTH = 0.11
RUN_PATH_FROZEN = True
FORCE_RECOMPUTE_DATASET = False
FORCE_RECOMPUTE_INIT = False
CONTRACTED_INIT_SEED = SEED + 17
CONTRACTED_INIT_MAIN_FACTOR = 0.82
CONTRACTED_INIT_MAIN_RADIAL_DROP = 0.24
CONTRACTED_INIT_SECONDARY_FACTOR = 0.46
CONTRACTED_INIT_SECONDARY_RADIAL_DROP = 0.26
CONTRACTED_INIT_NOISE_SCALE = 0.095

PATH_FROZEN_OPTIONS = {
    "graph_neighbors": 20,
    "outer_iter": 50,
    "inner_iter": 10,
    "n_landmark": int(N_SAMPLES * 0.2),
    "targets_per_landmark": int(N_SAMPLES * 0.35),
    "n_local_pairs": 20,
    "local_pair_mode": "direct",
    "local_global_reweighting": "energy", # count or energy
    "local_weight": 1.0,
    "device": "auto",
    "method": "L-BFGS-B",
    "optimizer_options": {"ftol": 1e-9, "maxls": 50},
    "log_frequency": 10,
    "verbose": 2,
    "eps": 1e-6,
}


@dataclass(frozen=True)
class Segment:
    start: np.ndarray
    end: np.ndarray
    branch: int


@dataclass(frozen=True)
class BranchingDataset:
    X: np.ndarray
    graph: csr_matrix
    D: np.ndarray
    branch_labels: np.ndarray


def main_branching():
    dir_res = SCRIPT_DIR / "res" / "branching_geodesic"
    dir_raw = dir_res / "raw"
    dir_fig = dir_res / "figures"
    dir_raw.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)

    cache_stem = f"branching_n{N_SAMPLES}_k{GRAPH_NEIGHBORS}_w{format_float(CORRIDOR_WIDTH)}_s{SEED}"
    dataset = load_or_create_dataset(
        dir_raw / f"{cache_stem}_dataset.npz",
        n_samples=N_SAMPLES,
        graph_neighbors=GRAPH_NEIGHBORS,
        corridor_width=CORRIDOR_WIDTH,
        seed=SEED,
        force=FORCE_RECOMPUTE_DATASET,
    )
    init = load_or_create_contracted_init(
        dir_raw / f"{cache_stem}_contracted_init.npz",
        dataset,
        force=FORCE_RECOMPUTE_INIT,
    )

    plot_dataset_and_init(dataset, init, dir_fig=dir_fig, n_samples=N_SAMPLES)
    if RUN_PATH_FROZEN:
        run_path_frozen(dataset, init, dir_fig=dir_fig)
    print(f"Saved figures in {dir_fig}")


def load_or_create_dataset(cache_path, *, n_samples, graph_neighbors, corridor_width, seed, force):
    if cache_path.exists() and not force:
        print(f"Loading cached branching dataset: {cache_path}")
        data = np.load(cache_path)
        graph = csr_matrix(
            (data["graph_data"], data["graph_indices"], data["graph_indptr"]),
            shape=tuple(data["graph_shape"]),
        )
        return BranchingDataset(
            X=data["X"],
            graph=graph,
            D=data["D"],
            branch_labels=data["branch_labels"],
        )

    print(f"Creating branching dataset: n={n_samples}, k={graph_neighbors}")
    rng = np.random.default_rng(seed)
    segments = make_branching_tree_segments()
    X, branch_labels = sample_branching_tree(n_samples, segments, width=corridor_width, rng=rng)
    graph = visible_knn_graph(X, segments, width=corridor_width, k=graph_neighbors)
    n_graph_components, labels = connected_components(graph, directed=False)
    if n_graph_components != 1:
        largest = np.bincount(labels).max()
        raise RuntimeError(
            f"Target graph has {n_graph_components} components; largest has {largest} points. "
            "Increase graph_neighbors or corridor_width."
        )
    D = shortest_path(graph, directed=False, return_predecessors=False)
    np.fill_diagonal(D, 0.0)
    np.savez_compressed(
        cache_path,
        X=X,
        D=D,
        branch_labels=branch_labels,
        graph_data=graph.data,
        graph_indices=graph.indices,
        graph_indptr=graph.indptr,
        graph_shape=np.array(graph.shape, dtype=int),
    )
    print(f"Saved branching dataset: {cache_path}")
    return BranchingDataset(X=X, graph=graph, D=D, branch_labels=branch_labels)


def load_or_create_contracted_init(cache_path, dataset, *, force):
    params = contracted_init_params()
    if cache_path.exists() and not force:
        data = np.load(cache_path)
        cached_params = data.get("params")
        if cached_params is not None and np.allclose(cached_params, params):
            print(f"Loading cached contracted init: {cache_path}")
            return data["init"]
        print(f"Cached contracted init parameters changed; regenerating: {cache_path}")

    print(f"Creating contracted init: {cache_path}")
    init = contracted_ground_truth_init(dataset.X)
    np.savez_compressed(cache_path, init=init, params=params)
    print(f"Saved contracted init: {cache_path}")
    return init


def run_path_frozen(dataset, init, *, dir_fig):
    metric = RandersMetric(alpha=0.0)
    print("Running path-frozen from contracted ground-truth init")
    result = fit_finsler_mds(
        dataset.D,
        metric=metric,
        optimizer="path_frozen",
        n_components=N_COMPONENTS,
        init=init,
        random_state=SEED,
        print_time=True,
        return_result=True,
        **PATH_FROZEN_OPTIONS,
    )
    X_pf = result.embedding
    masked_stress = result.stress
    geodesic_stress = result.final_normalized_full_geodesic_stress
    if geodesic_stress is None:
        geodesic_stress = geodesic_embedding_stress(
            X_pf,
            dataset.D,
            metric=metric,
            n_neighbors=PATH_FROZEN_OPTIONS["graph_neighbors"],
            normalized_stress=True,
            on_unreachable="inf",
        )
    print(f"Path-frozen masked stress: {masked_stress:.6g}")
    print(f"Path-frozen normalized full geodesic stress: {geodesic_stress:.6g}")

    plot_three_embeddings(
        dataset.X,
        init,
        X_pf,
        graph=dataset.graph,
        values=dataset.branch_labels,
        cmap="tab10",
        value_label="primary branch",
        titles=(
            "ground truth branching tree",
            "contracted/noisy init",
            "path-frozen, same branch colors",
        ),
        save_path=dir_fig / path_frozen_figure_name(),
        discrete=True,
    )


def path_frozen_figure_name():
    outer_iter = PATH_FROZEN_OPTIONS["outer_iter"]
    inner_iter = PATH_FROZEN_OPTIONS["inner_iter"]
    return f"pf_n{N_SAMPLES}_oi{outer_iter}_ii{inner_iter}.pdf"


def contracted_ground_truth_init(X):
    X = np.asarray(X, dtype=float)
    secondary = secondary_branch_mask(X)
    radius = np.linalg.norm(X, axis=1)
    radius_scale = radius / np.max(radius) if np.max(radius) > 0 else radius
    main_factors = CONTRACTED_INIT_MAIN_FACTOR - CONTRACTED_INIT_MAIN_RADIAL_DROP * radius_scale
    secondary_factors = (
        CONTRACTED_INIT_SECONDARY_FACTOR
        - CONTRACTED_INIT_SECONDARY_RADIAL_DROP * radius_scale
    )
    factors = np.where(secondary, secondary_factors, main_factors)
    factors = np.clip(factors, 0.12, None)
    rng = np.random.default_rng(CONTRACTED_INIT_SEED)
    noise_scale = CONTRACTED_INIT_NOISE_SCALE * (0.45 + 0.9 * radius_scale)
    return X * factors[:, None] + rng.normal(scale=noise_scale[:, None], size=X.shape)


def contracted_init_params():
    return np.array(
        [
            CONTRACTED_INIT_SEED,
            CONTRACTED_INIT_MAIN_FACTOR,
            CONTRACTED_INIT_MAIN_RADIAL_DROP,
            CONTRACTED_INIT_SECONDARY_FACTOR,
            CONTRACTED_INIT_SECONDARY_RADIAL_DROP,
            CONTRACTED_INIT_NOISE_SCALE,
        ],
        dtype=float,
    )


def secondary_branch_mask(X):
    segments = make_branching_tree_segments()
    primary_segments = [segment for segment in segments if np.linalg.norm(segment.start) < 1e-12]
    secondary_segments = [segment for segment in segments if np.linalg.norm(segment.start) >= 1e-12]
    primary_distance = min_distance_to_segments(X, primary_segments)
    secondary_distance = min_distance_to_segments(X, secondary_segments)
    return secondary_distance < primary_distance


def min_distance_to_segments(X, segments):
    distances = [
        point_segment_distances(np.asarray(X, dtype=float), segment.start, segment.end)
        for segment in segments
    ]
    return np.min(np.vstack(distances), axis=0)


def plot_dataset_and_init(dataset, init, *, dir_fig, n_samples):
    secondary = secondary_branch_mask(dataset.X)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    panels = (
        (dataset.X, dataset.branch_labels, "tab10", "ground truth branching tree"),
        (init, dataset.branch_labels, "tab10", "contracted/noisy init"),
        (init, secondary.astype(int), "coolwarm", f"secondary mask ({int(np.count_nonzero(secondary))} points)"),
    )
    for ax, (embedding, values, cmap, title) in zip(axes, panels):
        plot_graph_edges(ax, embedding, dataset.graph, alpha=0.045)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=values, s=9, cmap=cmap, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(dir_fig / f"branching_n{n_samples}_dataset_init.pdf", bbox_inches="tight")
    plt.close(fig)


def make_branching_tree_segments():
    root = np.array([0.0, 0.0])
    segments = []
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + 0.15
    for branch, angle in enumerate(angles):
        direction = unit_vector(angle)
        end = 3.0 * direction
        segments.append(Segment(root, end, branch))

        first_split = 0.58 * end
        second_split = 0.82 * end
        for sign in (-1.0, 1.0):
            segments.append(Segment(first_split, first_split + 1.25 * unit_vector(angle + sign * 0.55), branch))
            segments.append(Segment(second_split, second_split + 0.85 * unit_vector(angle + sign * 0.32), branch))
    return segments


def sample_branching_tree(n_samples, segments, *, width, rng):
    lengths = np.array([np.linalg.norm(segment.end - segment.start) for segment in segments])
    choices = rng.choice(len(segments), size=n_samples, p=lengths / np.sum(lengths))
    points = np.empty((n_samples, 2), dtype=float)
    labels = np.empty(n_samples, dtype=int)
    for i, segment_idx in enumerate(choices):
        segment = segments[segment_idx]
        vector = segment.end - segment.start
        length = np.linalg.norm(vector)
        tangent = vector / length
        normal = np.array([-tangent[1], tangent[0]])
        t = rng.uniform(0.0, 1.0)
        offset = rng.uniform(-0.55 * width, 0.55 * width)
        points[i] = segment.start + t * vector + offset * normal
        labels[i] = segment.branch
    return points, labels


def visible_knn_graph(X, segments, *, width, k):
    distances = pairwise_distances(X)
    rows = []
    cols = []
    data = []
    for i in range(len(X)):
        n_added = 0
        for j in np.argsort(distances[i]):
            if i == j:
                continue
            if not segment_inside_tree(X[i], X[j], segments, width=width):
                continue
            rows.append(i)
            cols.append(j)
            data.append(distances[i, j])
            n_added += 1
            if n_added >= k:
                break
        if n_added < max(4, k // 2):
            raise RuntimeError(f"Point {i} only has {n_added} visible neighbors.")
    graph = csr_matrix((data, (rows, cols)), shape=(len(X), len(X))).maximum(
        csr_matrix((data, (cols, rows)), shape=(len(X), len(X)))
    )
    return connect_visible_components(graph.tolil(), X, segments, width=width, distances=distances).tocsr()


def connect_visible_components(graph, X, segments, *, width, distances):
    n_components, labels = connected_components(graph.tocsr(), directed=False)
    while n_components > 1:
        best = None
        for component in range(1, n_components):
            source_idx = np.flatnonzero(labels == 0)
            target_idx = np.flatnonzero(labels == component)
            sub_distances = distances[np.ix_(source_idx, target_idx)]
            for flat_index in np.argsort(sub_distances, axis=None):
                local_i, local_j = np.unravel_index(flat_index, sub_distances.shape)
                i = int(source_idx[local_i])
                j = int(target_idx[local_j])
                if segment_inside_tree(X[i], X[j], segments, width=width):
                    candidate = (float(distances[i, j]), i, j)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                    break
        if best is None:
            break
        distance, i, j = best
        graph[i, j] = distance
        graph[j, i] = distance
        n_components, labels = connected_components(graph.tocsr(), directed=False)
    return graph


def segment_inside_tree(p, q, segments, *, width):
    t = np.linspace(0.0, 1.0, 25)
    samples = (1.0 - t[:, None]) * p[None, :] + t[:, None] * q[None, :]
    return bool(np.all(inside_tree(samples, segments, width=width)))


def inside_tree(points, segments, *, width):
    points = np.asarray(points, dtype=float)
    inside = np.zeros(len(points), dtype=bool)
    for segment in segments:
        inside |= point_segment_distances(points, segment.start, segment.end) <= width
    return inside


def point_segment_distances(points, start, end):
    vector = end - start
    denom = float(vector @ vector)
    t = np.clip(((points - start) @ vector) / denom, 0.0, 1.0)
    projection = start + t[:, None] * vector
    return np.linalg.norm(points - projection, axis=1)


def unit_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def format_float(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def plot_three_embeddings(
        X_true,
        X_init,
        X_result,
        *,
        graph,
        values,
        cmap,
        value_label,
        titles,
        save_path,
        discrete=False,
):
    plot_embedding_triptych(
        (X_true, X_init, X_result),
        graph=graph,
        values=values,
        cmap=cmap,
        value_label=value_label,
        titles=titles,
        save_path=save_path,
        discrete=discrete,
        edge_alpha=0.045,
    )


def plot_embedding_triptych(
        embeddings,
        *,
        graph,
        values,
        cmap,
        value_label,
        titles,
        save_path,
        discrete=False,
        edge_alpha=0.05,
):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, embedding, title in zip(axes, embeddings, titles):
        plot_graph_edges(ax, embedding, graph, alpha=edge_alpha)
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=values, s=9, cmap=cmap, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
    ticks = np.unique(values) if discrete else None
    fig.colorbar(scatter, ax=axes, label=value_label, ticks=ticks, shrink=0.86)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_graph_edges(ax, X, graph, *, alpha):
    coo = graph.tocoo()
    for i, j in zip(coo.row, coo.col):
        if i < j:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="0.2", alpha=alpha, linewidth=0.35)


if __name__ == "__main__":
    main_branching()
