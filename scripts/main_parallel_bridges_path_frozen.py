from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.api import fit_finsler_mds
from finsler_mds.evaluation.distance_embedding import compute_embedding_distances
from finsler_mds.metrics import RandersMetric
from finsler_mds import utils


SEED = 42
RAIL_X = (-0.5, 0.5)
Y_MIN = -4.0
Y_MAX = 4.0
BRIDGE_Y = (-4.0, 0.0, 4.0)
RAIL_POINTS = 61
BRIDGE_POINTS = 8

MANIFOLD_RANDERS_ALPHA = 0.5
EMBEDDING_RANDERS_ALPHA = 0.5
TARGET_GRAPH_NEIGHBORS = 4
EMBEDDING_GRAPH_NEIGHBORS = 4

SMACOF_MAX_ITER = 500
PATH_FROZEN_INIT = "smacof"  # one of {"data", "smacof"}
PF_STAGE1 = {"outer_iter": 70, "inner_iter": 50, "log_frequency": 5, "outer_step_size": 0.5}
PF_STAGE2 = {"outer_iter": 40, "inner_iter": 3, "log_frequency": 10, "outer_step_size": 0.5}

RES_DIR = Path(__file__).resolve().parent / "res" / "parallel_bridges_path_frozen"
FIG_DIR = RES_DIR / "figures"


def make_parallel_bridges():
    points = []
    rail_ys = np.linspace(Y_MIN, Y_MAX, RAIL_POINTS)
    bridge_xs = np.linspace(RAIL_X[0], RAIL_X[1], BRIDGE_POINTS)

    for x in RAIL_X:
        for y in rail_ys:
            points.append((x, y))
    for y in BRIDGE_Y:
        for x in bridge_xs:
            points.append((x, y))

    X = np.unique(np.asarray(points, dtype=float), axis=0)
    current = current_field(X)
    return X, current


def current_field(X):
    current = np.zeros_like(X)
    on_rail = np.isclose(X[:, 0], RAIL_X[0]) | np.isclose(X[:, 0], RAIL_X[1])
    current[on_rail & (X[:, 1] > 0), 1] = -1.0
    current[on_rail & (X[:, 1] < 0), 1] = 1.0
    return current


def flattened_3d_init(X):
    return np.column_stack([X, np.zeros(len(X))])


def randers_dissimilarities(X, current):
    # compute_dist_matrix uses F_i(u) = ||u|| + <randers_field_i, u>.
    # Use -current so movement with the displayed current is cheaper.
    randers_field = -MANIFOLD_RANDERS_ALPHA * current
    D, _ = utils.compute_dist_matrix(
        X,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        randers_field=randers_field,
    )
    if not np.all(np.isfinite(D)):
        raise RuntimeError("Directed Randers dissimilarities contain unreachable pairs.")
    return D


def asymmetry_factors(D):
    denom = D + D.T
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (D - D.T) / denom
    np.fill_diagonal(A, np.nan)
    return A


def asymmetry_preservation_scores(data_distances, embedding_distances):
    A_data = asymmetry_factors(data_distances)
    A_emb = asymmetry_factors(embedding_distances)
    mask = np.isfinite(A_data) & np.isfinite(A_emb)
    a = A_data[mask]
    b = A_emb[mask]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    pearson = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    beta = float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) > 0 else np.nan
    rmse = float(np.sqrt(np.mean((b - a) ** 2)))
    return pearson, beta, rmse


def log_asymmetry_preservation(label, data_distances, embedding, metric, *, mode):
    embedding_distances = compute_embedding_distances(
        embedding,
        metric=metric,
        mode=mode,
        n_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
    )
    pearson, beta, rmse = asymmetry_preservation_scores(data_distances, embedding_distances)
    print(
        f"{label} asymmetry preservation ({mode} distances): "
        f"Pearson={pearson:.6g}, beta={beta:.6g}, RMSE={rmse:.6g}"
    )


def log_asymmetry_preservation_both_modes(label, data_distances, embedding, metric):
    for mode in ("direct", "geodesic"):
        log_asymmetry_preservation(label, data_distances, embedding, metric, mode=mode)


def run_smacof(D, init, metric):
    print("\nRunning Randers SMACOF")
    start = perf_counter()
    result = fit_finsler_mds(
        D,
        metric=metric,
        optimizer="smacof",
        init=init,
        n_components=3,
        n_init=1,
        max_iter=SMACOF_MAX_ITER,
        eps=1e-6,
        check_monotony=False,
        version="corrected",
        project_on_V=True,
        device="cpu",
        random_state=SEED,
        verbose=1,
        return_result=True,
    )
    print(f"SMACOF finished in {perf_counter() - start:.3f} s")
    print(f"SMACOF optimizer stress: {result.stress:.6g}, n_iter={result.n_iter}")
    return result


def run_path_frozen(D, init, metric):
    common = dict(
        metric=metric,
        optimizer="path_frozen",
        init=init,
        n_components=3,
        graph_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
        n_landmark=0,
        targets_per_landmark=None,
        n_local_pairs=None,
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


def save_original_plot(X, current, path):
    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    ax.scatter(X[:, 0], X[:, 1], color="#333333", s=22, linewidths=0)
    arrows = np.linalg.norm(current, axis=1) > 0
    step = max(1, int(np.ceil(np.count_nonzero(arrows) / 28)))
    idx = np.flatnonzero(arrows)[2::step]
    ax.quiver(
        X[idx, 0],
        X[idx, 1],
        current[idx, 0],
        current[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=3,
        color="#d62728",
        width=0.03,
        headwidth=3,
        headlength=2,
        headaxislength=2,
    )
    ax.set_title("Original 2D graph with current")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(RAIL_X[0] - 1, RAIL_X[1] + 1)
    ax.set_ylim(Y_MIN - 0.25, Y_MAX + 0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_3d_plot(X, path, title):
    views = [
        ("oblique", 25, -60),
        ("along x", 0, 0),
        ("along y", 0, 90),
        ("along z", 90, -90),
    ]
    fig = plt.figure(figsize=(10.2, 8.8))
    for index, (view_name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="#333333", s=18, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} - {view_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        utils.set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, current = make_parallel_bridges()
    D = randers_dissimilarities(X, current)
    init = flattened_3d_init(X)
    metric = RandersMetric(alpha=EMBEDDING_RANDERS_ALPHA)

    print(
        "Parallel bridges dataset: "
        f"n={len(X)}, manifold_alpha={MANIFOLD_RANDERS_ALPHA}, "
        f"embedding_alpha={EMBEDDING_RANDERS_ALPHA}, "
        f"target_k={TARGET_GRAPH_NEIGHBORS}, embedding_k={EMBEDDING_GRAPH_NEIGHBORS}"
    )

    smacof = run_smacof(D, init, metric)
    if PATH_FROZEN_INIT == "data":
        path_frozen_init = init
    elif PATH_FROZEN_INIT == "smacof":
        path_frozen_init = smacof.embedding
    else:
        raise ValueError("PATH_FROZEN_INIT must be 'data' or 'smacof'.")
    _, path_frozen = run_path_frozen(D, path_frozen_init, metric)
    log_asymmetry_preservation_both_modes("SMACOF", D, smacof.embedding, metric)
    log_asymmetry_preservation_both_modes("Path-frozen", D, path_frozen.embedding, metric)

    save_original_plot(X, current, FIG_DIR / "parallel_bridges_original_2d.pdf")
    save_3d_plot(init, FIG_DIR / "parallel_bridges_init_3d.pdf", "3D initialization")
    save_3d_plot(smacof.embedding, FIG_DIR / "parallel_bridges_smacof_3d.pdf", "SMACOF result")
    save_3d_plot(path_frozen.embedding, FIG_DIR / "parallel_bridges_path_frozen_3d.pdf", "Path-frozen result")
    print(f"Saved PDF figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
