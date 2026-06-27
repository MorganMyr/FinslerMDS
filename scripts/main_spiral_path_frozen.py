from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.api import fit_finsler_mds
from finsler_mds.metrics import RandersMetric
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph


N_SAMPLES = 1000
SEED = 1
T_MAX = 1.1
XY_NOISE = 0.015
Z_SCALE = 0.3

TARGET_GRAPH_NEIGHBORS = 15
EMBEDDING_GRAPH_NEIGHBORS = 15
PATH_FROZEN_LANDMARKS = 50
PATH_FROZEN_TARGETS = 400
PATH_FROZEN_LOCAL_PAIRS = 15

PF_STAGE1 = {"outer_iter": 10, "inner_iter": 30, "log_frequency": 5}
PF_STAGE2 = {"outer_iter": 10, "inner_iter": 1, "log_frequency": 5}

SMACOF_MAX_ITER = 1000

RES_DIR = Path(__file__).resolve().parent / "res" / "spiral_path_frozen"
FIG_DIR = RES_DIR / "figures"
RAW_DIR = RES_DIR / "raw"


def make_spiral_dataset():
    rng = np.random.default_rng(SEED)
    t = np.sort(rng.uniform(0.0, T_MAX, size=N_SAMPLES))
    X = np.column_stack(
        [
            np.cos(2.0 * np.pi * t),
            np.sin(2.0 * np.pi * t),
            Z_SCALE * t,
        ]
    )
    X[:, :2] += rng.normal(scale=XY_NOISE, size=(N_SAMPLES, 2))
    X_init = X[:, :2].copy()
    return t, X, X_init


def isomap_dissimilarities(X):
    support = symmetric_knn_graph(
        X,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        ensure_connected=True,
        warn_on_connect=True,
    )
    D = shortest_path(support, directed=False, return_predecessors=False)
    if not np.all(np.isfinite(D)):
        raise RuntimeError("The original 3D kNN graph is not connected.")
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


def original_support_stress(X_embedding, original_support, D):
    embedding_distances = geodesic_distances_on_support(X_embedding, original_support)
    return stress_against_target(embedding_distances, D)


def run_path_frozen(D, init):
    common = dict(
        metric=RandersMetric(alpha=0.0),
        optimizer="path_frozen",
        init=init,
        n_components=2,
        graph_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
        n_landmark=PATH_FROZEN_LANDMARKS,
        landmark_sampling="farthest",
        targets_per_landmark=PATH_FROZEN_TARGETS,
        n_local_pairs=PATH_FROZEN_LOCAL_PAIRS,
        local_pair_mode="direct",
        local_global_reweighting="count",
        local_weight=1.0,
        random_state=SEED,
        target_random_state=SEED,
        method="L-BFGS-B",
        device="auto",
        verbose=1,
        record_history=True,
        return_result=True,
    )

    print("\nRunning path-frozen stage 1")
    stage1 = fit_finsler_mds(D, print_time=True, **common, **PF_STAGE1)
    print("\nRunning path-frozen stage 2")
    stage2_options = {**common, "init": stage1.embedding}
    stage2 = fit_finsler_mds(D, print_time=True, **stage2_options, **PF_STAGE2)
    return stage1, stage2


def run_smacof(D, init):
    print("\nRunning one-step Euclidean SMACOF from path-frozen init")
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


def save_3d_plot(X, t, path):
    fig = plt.figure(figsize=(5.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="viridis", s=10, linewidths=0)
    ax.set_title("Original 3D spiral")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="t")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_2d_plot(X, t, path, title):
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    sc = ax.scatter(X[:, 0], X[:, 1], c=t, cmap="viridis", s=10, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, label="t")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_stress(label, stress):
    raw, normalized = stress
    print(f"{label}: stress={raw:.6g}, normalized={normalized:.6g}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    t, X3, X_init = make_spiral_dataset()
    original_support, D = isomap_dissimilarities(X3)

    print(
        f"Dataset: n={N_SAMPLES}, t in [0, {T_MAX}], "
        f"target_k={TARGET_GRAPH_NEIGHBORS}, embedding_k={EMBEDDING_GRAPH_NEIGHBORS}"
    )
    print(f"3D kNN graph edges: {original_support.nnz // 2} undirected")

    init_embedding_knn = full_embedding_knn_stress(X_init, D)
    init_original_support = original_support_stress(X_init, original_support, D)
    print_stress("Initial flattened embedding, kNN rebuilt in 2D", init_embedding_knn)
    print_stress("Initial flattened embedding, fixed original 3D kNN support", init_original_support)

    pf_stage1, pf_stage2 = run_path_frozen(D, X_init)
    X_pf_stage1 = pf_stage1.embedding
    X_pf = pf_stage2.embedding
    pf_embedding_knn = full_embedding_knn_stress(X_pf, D)
    print_stress("Path-frozen final, kNN rebuilt in 2D", pf_embedding_knn)
    if pf_stage2.final_full_geodesic_stress is not None:
        print(
            "Path-frozen recorded final full geodesic stress: "
            f"{pf_stage2.final_full_geodesic_stress:.6g}, "
            f"normalized={pf_stage2.final_normalized_full_geodesic_stress:.6g}"
        )

    smacof = run_smacof(D, X_pf)
    X_smacof = smacof.embedding
    smacof_embedding_knn = full_embedding_knn_stress(X_smacof, D)
    print_stress("SMACOF embedding, kNN rebuilt in 2D", smacof_embedding_knn)
    print(f"SMACOF optimizer stress: {smacof.stress:.6g}, n_iter={smacof.n_iter}")

    np.savez(
        RAW_DIR / "spiral_path_frozen_results.npz",
        t=t,
        X3=X3,
        init=X_init,
        D=D,
        path_frozen_stage1=X_pf_stage1,
        path_frozen=X_pf,
        smacof=X_smacof,
    )

    save_3d_plot(X3, t, FIG_DIR / "spiral_original_3d.pdf")
    save_2d_plot(X_init, t, FIG_DIR / "spiral_flattened_init_2d.pdf", "Flattened init")
    save_2d_plot(X_pf_stage1, t, FIG_DIR / "spiral_path_frozen_stage1_2d.pdf", "Path-frozen before finisher")
    save_2d_plot(X_pf, t, FIG_DIR / "spiral_path_frozen_2d.pdf", "Path-frozen result")
    save_2d_plot(X_smacof, t, FIG_DIR / "spiral_smacof_2d.pdf", "SMACOF result")

    save_3d_plot(X3, t, FIG_DIR / "spiral_original_3d.png")
    save_2d_plot(X_init, t, FIG_DIR / "spiral_flattened_init_2d.png", "Flattened init")
    save_2d_plot(X_pf_stage1, t, FIG_DIR / "spiral_path_frozen_stage1_2d.png", "Path-frozen before finisher")
    save_2d_plot(X_pf, t, FIG_DIR / "spiral_path_frozen_2d.png", "Path-frozen result")
    save_2d_plot(X_smacof, t, FIG_DIR / "spiral_smacof_2d.png", "SMACOF result")
    print(f"Saved figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
