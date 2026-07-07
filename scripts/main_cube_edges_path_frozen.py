from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.sparse.csgraph import shortest_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.api import fit_finsler_mds
from finsler_mds.metrics import RandersMetric
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph


SEED = 42
POINTS_PER_EDGE = 10
INIT_RADIAL_Z_OFFSET = 1.5

TARGET_GRAPH_NEIGHBORS = 6
EMBEDDING_GRAPH_NEIGHBORS = 6
PATH_FROZEN_INIT = "flattened"  # one of {"flattened", "smacof"}
OUTER_STEP_SIZE = 0.1

PF_STAGE1 = {"outer_iter": 100, "inner_iter": 1, "log_frequency": 5}
NO_STAGE_2 = True
PF_STAGE2 = {"outer_iter": 10, "inner_iter": 1, "log_frequency": 1}
SMACOF_MAX_ITER = 300

RES_DIR = Path(__file__).resolve().parent / "res" / "cube_edges_path_frozen"
FIG_DIR = RES_DIR / "figures"


def make_cube_edges():
    points = []
    edge_ids = []
    edge_groups = []
    edge_names = []
    values = np.linspace(-1.0, 1.0, POINTS_PER_EDGE)
    edge_id = 0

    for axis in range(3):
        fixed_axes = [a for a in range(3) if a != axis]
        for first in (-1.0, 1.0):
            for second in (-1.0, 1.0):
                X_edge = np.empty((POINTS_PER_EDGE, 3), dtype=float)
                X_edge[:, axis] = values
                X_edge[:, fixed_axes[0]] = first
                X_edge[:, fixed_axes[1]] = second
                points.append(X_edge)
                edge_ids.extend([edge_id] * POINTS_PER_EDGE)
                if axis == 2:
                    group = 2
                elif second < 0:
                    group = 0
                else:
                    group = 1
                edge_groups.extend([group] * POINTS_PER_EDGE)
                edge_names.append(
                    f"edge {edge_id}: axis {axis}, "
                    f"{fixed_axes[0]}={first:g}, {fixed_axes[1]}={second:g}"
                )
                edge_id += 1

    return np.vstack(points), np.asarray(edge_ids, dtype=int), np.asarray(edge_groups, dtype=int), edge_names


def flattened_init(X):
    X2 = X[:, :2].copy()
    radial_norm = np.linalg.norm(X2, axis=1)
    radial = np.divide(X2, radial_norm[:, None], out=np.zeros_like(X2), where=radial_norm[:, None] > 0)
    z_range = np.ptp(X[:, 2])
    z01 = (X[:, 2] - X[:, 2].min()) / (z_range if z_range > 0 else 1.0)
    X2 = X2 + INIT_RADIAL_Z_OFFSET * z01[:, None] * radial
    X2 -= X2.mean(axis=0, keepdims=True)
    return X2


def isomap_dissimilarities(X):
    support = symmetric_knn_graph(
        X,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        ensure_connected=True,
        warn_on_connect=True,
    )
    D = shortest_path(support, directed=False, return_predecessors=False)
    if not np.all(np.isfinite(D)):
        raise RuntimeError("The cube-edge target kNN graph is not connected.")
    return support, D


def geodesic_distances_on_support(X_embedding, support):
    graph = metric_graph_from_support(X_embedding, support, RandersMetric(alpha=0.0))
    return shortest_path(graph, directed=False, return_predecessors=False)


def stress_against_target(embedding_distances, target_dissimilarities):
    mask = ~np.eye(len(target_dissimilarities), dtype=bool)
    residual = embedding_distances[mask] - target_dissimilarities[mask]
    stress = float(np.sum(residual * residual))
    denom = float(np.sum(target_dissimilarities[mask] ** 2))
    normalized = np.sqrt(stress / denom) if denom > 0 else np.inf
    return stress, normalized


def full_embedding_knn_stress(X_embedding, D):
    support = symmetric_knn_graph(
        X_embedding,
        n_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
        ensure_connected=True,
        warn_on_connect=True,
    )
    embedding_distances = geodesic_distances_on_support(X_embedding, support)
    return stress_against_target(embedding_distances, D)


def print_stress(label, stress):
    raw, normalized = stress
    print(f"{label}: stress={raw:.6g}, normalized={normalized:.6g}")


def run_smacof(D, init):
    print("\nRunning Euclidean SMACOF from flattened init")
    start = perf_counter()
    result = fit_finsler_mds(
        D,
        metric=RandersMetric(alpha=0.0),
        optimizer="smacof",
        init=init,
        n_components=2,
        n_init=1,
        max_iter=SMACOF_MAX_ITER,
        eps=1e-6,
        check_monotony=True,
        project_on_V=True,
        device="auto",
        random_state=SEED,
        verbose=1,
        return_result=True,
    )
    print(f"SMACOF finished in {perf_counter() - start:.3f} s")
    return result


def run_path_frozen(D, init):
    common = dict(
        metric=RandersMetric(alpha=0.0),
        optimizer="path_frozen",
        init=init,
        n_components=2,
        graph_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
        n_landmark=0,
        targets_per_landmark=None,
        n_local_pairs=None,
        random_state=SEED,
        target_random_state=SEED,
        method="L-BFGS-B",
        outer_step_size=OUTER_STEP_SIZE,
        device="auto",
        verbose=1,
        record_history=True,
        return_result=True,
    )

    print(f"\nRunning path-frozen stage 1 from {PATH_FROZEN_INIT} init")
    stage1 = fit_finsler_mds(D, print_time=True, **common, **PF_STAGE1)
    print("\nRunning path-frozen stage 2")
    if NO_STAGE_2:
        stage2 = stage1
    else:
        stage2 = fit_finsler_mds(D, print_time=True, **{**common, "init": stage1.embedding}, **PF_STAGE2)
    return stage1, stage2


GROUP_LABELS = ["bottom z=-1", "top z=1", "vertical"]
GROUP_COLORS = ["#377eb8", "#e41a1c", "#4daf4a"]


def colors_for_groups(edge_groups):
    return [GROUP_COLORS[group] for group in edge_groups]


def add_group_legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=7,
            label=label,
        )
        for color, label in zip(GROUP_COLORS, GROUP_LABELS)
    ]
    ax.legend(handles=handles, loc="best", frameon=True)


def save_3d_plot(X, edge_groups, path):
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors_for_groups(edge_groups),
        s=28, linewidths=0,
    )
    ax.set_title("Original cube edges")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    add_group_legend(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_2d_plot(X, edge_groups, path, title):
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=colors_for_groups(edge_groups),
        s=28, linewidths=0,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    add_group_legend(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X3, edge_ids, edge_groups, edge_names = make_cube_edges()
    X_init = flattened_init(X3)
    _, D = isomap_dissimilarities(X3)

    print(f"Cube-edge dataset: {len(X3)} points, {POINTS_PER_EDGE} points per edge")
    print(f"target_k={TARGET_GRAPH_NEIGHBORS}, embedding_k={EMBEDDING_GRAPH_NEIGHBORS}")

    print_stress("Flattened init, kNN rebuilt in 2D", full_embedding_knn_stress(X_init, D))

    smacof = run_smacof(D, X_init)
    X_smacof = smacof.embedding
    print_stress("SMACOF, kNN rebuilt in 2D", full_embedding_knn_stress(X_smacof, D))
    print(f"SMACOF optimizer stress: {smacof.stress:.6g}, n_iter={smacof.n_iter}")

    if PATH_FROZEN_INIT == "flattened":
        X_pf_init = X_init
    elif PATH_FROZEN_INIT == "smacof":
        X_pf_init = X_smacof
    else:
        raise ValueError("PATH_FROZEN_INIT must be 'flattened' or 'smacof'.")

    _, path_frozen = run_path_frozen(D, X_pf_init)
    X_pf = path_frozen.embedding
    print_stress("Path-frozen, kNN rebuilt in 2D", full_embedding_knn_stress(X_pf, D))
    if path_frozen.final_full_geodesic_stress is not None:
        print(
            "Path-frozen recorded final full geodesic stress: "
            f"{path_frozen.final_full_geodesic_stress:.6g}, "
            f"normalized={path_frozen.final_normalized_full_geodesic_stress:.6g}"
        )

    save_3d_plot(X3, edge_groups, FIG_DIR / "cube_edges_original_3d.pdf")
    save_2d_plot(X_init, edge_groups, FIG_DIR / "cube_edges_smacof_init_2d.pdf", "Flattened SMACOF init")
    save_2d_plot(X_smacof, edge_groups, FIG_DIR / "cube_edges_smacof_2d.pdf", "SMACOF result")
    save_2d_plot(X_pf, edge_groups, FIG_DIR / "cube_edges_path_frozen_2d.pdf", "Path-frozen result")
    print(f"Saved PDF figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
