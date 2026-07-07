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
from finsler_mds import utils
from finsler_mds.utils.graph import symmetric_knn_graph


SEED = 42
POINTS_PER_UNIT = 7
INIT_NOISE_SCALE = 0.2

OUTER_X = (-5.0, 5.0)
OUTER_Z = (-2.0, 2.0)
INNER_Y = (-3.0, 3.0)
INNER_Z = (-1.0, 1.0)
BAR_Z = (-2.0, 2.0)

TARGET_GRAPH_NEIGHBORS = 4
EMBEDDING_GRAPH_NEIGHBORS = 4
PATH_FROZEN_INIT = "diagonal"  # one of {"diagonal", "smacof"}

SMACOF_MAX_ITER = 500
PF_STAGE1 = {"outer_iter": 50, "inner_iter": 10, "log_frequency": 10, "outer_step_size": 0.1}
PF_STAGE2 = {"outer_iter": 50, "inner_iter": 1, "log_frequency": 10, "outer_step_size": 0.1}

RES_DIR = Path(__file__).resolve().parent / "res" / "nested_rectangles_path_frozen"
FIG_DIR = RES_DIR / "figures"

GROUP_LABELS = ("outer rectangle", "inner rectangle", "central bar")
GROUP_COLORS = ("#377eb8", "#e41a1c", "#4daf4a")


def n_points_for_segment(a, b):
    length = np.linalg.norm(np.asarray(b) - np.asarray(a))
    return max(2, int(round(POINTS_PER_UNIT * length)) + 1)


def segment_points(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = np.linspace(0.0, 1.0, n_points_for_segment(a, b))
    return a[None, :] + t[:, None] * (b - a)[None, :]


def add_segment(points, groups, a, b, group):
    for point in segment_points(a, b):
        key = tuple(np.round(point, 12))
        if key not in points:
            points[key] = point
            groups[key] = group


def make_nested_rectangles():
    points = {}
    groups = {}

    x0, x1 = OUTER_X
    z0, z1 = OUTER_Z
    outer = [
        ((x0, 0.0, z0), (x1, 0.0, z0)),
        ((x0, 0.0, z1), (x1, 0.0, z1)),
        ((x0, 0.0, z0), (x0, 0.0, z1)),
        ((x1, 0.0, z0), (x1, 0.0, z1)),
    ]
    for a, b in outer:
        add_segment(points, groups, a, b, 0)

    y0, y1 = INNER_Y
    z0, z1 = INNER_Z
    inner = [
        ((0.0, y0, z0), (0.0, y1, z0)),
        ((0.0, y0, z1), (0.0, y1, z1)),
        ((0.0, y0, z0), (0.0, y0, z1)),
        ((0.0, y1, z0), (0.0, y1, z1)),
    ]
    for a, b in inner:
        add_segment(points, groups, a, b, 1)

    add_segment(points, groups, (0.0, 0.0, BAR_Z[0]), (0.0, 0.0, BAR_Z[1]), 2)

    keys = sorted(points)
    X = np.vstack([points[key] for key in keys])
    group = np.asarray([groups[key] for key in keys], dtype=int)
    return X, group


def diagonal_view_init(X):
    theta = np.deg2rad(45.0)
    x_rot = (np.cos(theta) * X[:, 0] - np.sin(theta) * X[:, 1]) * 2**0.5
    X2 = np.column_stack([x_rot, X[:, 2]])
    X2 -= X2.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(SEED)
    step = 1.0 / POINTS_PER_UNIT
    X2 += rng.normal(scale=INIT_NOISE_SCALE * step, size=X2.shape)
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
        raise RuntimeError("The target kNN graph is not connected.")
    return D


def run_smacof(D, init):
    print("\nRunning Euclidean SMACOF from diagonal-view init")
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
    print(f"SMACOF optimizer stress: {result.stress:.6g}, n_iter={result.n_iter}")
    return result


def run_path_frozen(D, init):
    common = dict(
        metric=RandersMetric(alpha=0.0),
        optimizer="path_frozen",
        init=init,
        n_components=2,
        graph_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
        n_landmark=150,
        targets_per_landmark=200,
        n_local_pairs=EMBEDDING_GRAPH_NEIGHBORS,
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
    print(f"\nRunning path-frozen stage 1 from {PATH_FROZEN_INIT} init")
    stage1 = fit_finsler_mds(D, print_time=True, **common, **PF_STAGE1)
    print("\nRunning path-frozen stage 2")
    stage2 = fit_finsler_mds(D, print_time=True, **{**common, "init": stage1.embedding}, **PF_STAGE2)
    if stage2.final_full_geodesic_stress is not None:
        print(
            "Path-frozen final full geodesic stress: "
            f"{stage2.final_full_geodesic_stress:.6g}, "
            f"normalized={stage2.final_normalized_full_geodesic_stress:.6g}"
        )
    return stage1, stage2


def group_colors(group):
    return [GROUP_COLORS[index] for index in group]


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


def save_3d_views(X, group, path, title):
    views = [
        ("oblique", 25, -60),
        ("along x", 0, 0),
        ("along y", 0, 90),
        ("along z", 90, -90),
    ]
    fig = plt.figure(figsize=(10.2, 8.8))
    for index, (view_name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=group_colors(group), s=20, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} - {view_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        utils.set_axes_equal(ax)
        add_group_legend(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_2d_plot(X, group, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(X[:, 0], X[:, 1], c=group_colors(group), s=22, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    add_group_legend(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, group = make_nested_rectangles()
    D = isomap_dissimilarities(X)
    X_init = diagonal_view_init(X)

    print(
        "Nested rectangles dataset: "
        f"n={len(X)}, target_k={TARGET_GRAPH_NEIGHBORS}, "
        f"embedding_k={EMBEDDING_GRAPH_NEIGHBORS}"
    )

    smacof = run_smacof(D, X_init)

    if PATH_FROZEN_INIT == "diagonal":
        path_frozen_init = X_init
    elif PATH_FROZEN_INIT == "smacof":
        path_frozen_init = smacof.embedding
    else:
        raise ValueError("PATH_FROZEN_INIT must be 'diagonal' or 'smacof'.")
    _, path_frozen = run_path_frozen(D, path_frozen_init)

    save_3d_views(X, group, FIG_DIR / "nested_rectangles_original_3d.pdf", "Original dataset")
    save_2d_plot(
        X_init,
        group,
        FIG_DIR / "nested_rectangles_diagonal_init_2d.pdf",
        "Diagonal-view initialization",
    )
    save_2d_plot(smacof.embedding, group, FIG_DIR / "nested_rectangles_smacof_2d.pdf", "SMACOF result")
    save_2d_plot(
        path_frozen.embedding,
        group,
        FIG_DIR / "nested_rectangles_path_frozen_2d.pdf",
        "Path-frozen result",
    )
    print(f"Saved PDF figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
