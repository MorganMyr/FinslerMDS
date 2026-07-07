from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.api import fit_finsler_mds
from finsler_mds.metrics import MatsumotoMetric, RandersMetric
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph
from finsler_mds import utils

import main_hysteresis_paths as hysteresis


SEED = 42
DATA_ALPHA = 0.9
EMBEDDING_ALPHA = 0.9
RUN_METRICS = ("randers",)  # use ("matsumoto",) while tuning

GRAPH_NEIGHBORS = 15
N_LANDMARK_FRACTION = 0.25
TARGETS_PER_LANDMARK_FRACTION = 0.40
N_LOCAL_PAIRS = 15
LOCAL_PAIR_MODE = "direct"
LOCAL_GLOBAL_REWEIGHTING = "count"
LOCAL_WEIGHT = 1

PF_STAGES = {
    "matsumoto": {
        "stage1": {"outer_iter": 40, "inner_iter": 10, "log_frequency": 8, "outer_step_size": 0.5},
        "stage2": {"outer_iter": 15, "inner_iter": 1, "log_frequency": 6, "outer_step_size": 0.5},
    },
    "randers": {
        "stage1": {"outer_iter": 70, "inner_iter": 10, "log_frequency": 5, "outer_step_size": 0.5},
        "stage2": {"outer_iter": 30, "inner_iter": 3, "log_frequency": 6, "outer_step_size": 0.3},
    },
}

RES_DIR = Path(__file__).resolve().parent / "res" / "hysteresis_paths"
FIG_DIR = RES_DIR / "figures"
RAW_DIR = RES_DIR / "raw"


def make_embedding_metric(kind):
    if kind == "matsumoto":
        return MatsumotoMetric(alpha=EMBEDDING_ALPHA)
    if kind == "randers":
        return RandersMetric(alpha=EMBEDDING_ALPHA)
    raise ValueError("kind must be 'matsumoto' or 'randers'.")


def endpoint_indices(X):
    start = int(np.argmin(np.linalg.norm(X - np.array([0.0, 0.0, 0.0]), axis=1)))
    end = int(np.argmin(np.linalg.norm(X - np.array([hysteresis.END_X, 0.0, hysteresis.HEIGHT]), axis=1)))
    return start, end


def shortest_path_indices(predecessors, start, end):
    if predecessors[end] < 0 and start != end:
        return np.array([], dtype=int)
    path = [end]
    node = end
    while node != start:
        node = predecessors[node]
        if node < 0:
            return np.array([], dtype=int)
        path.append(node)
    return np.asarray(path[::-1], dtype=int)


def branch_counts(path, groups):
    counts = Counter(hysteresis.GROUP_NAMES[group] for group in groups[path])
    return ", ".join(f"{name}={counts.get(name, 0)}" for name in hysteresis.GROUP_NAMES)


def embedding_directed_paths(X_emb, groups, metric):
    start, end = endpoint_indices(X_emb)
    support = symmetric_knn_graph(
        X_emb,
        n_neighbors=GRAPH_NEIGHBORS,
        ensure_connected=True,
        warn_on_connect=True,
    )
    graph = metric_graph_from_support(X_emb, support, metric)
    dist_up, pred_up = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    dist_down, pred_down = dijkstra(graph, directed=True, indices=end, return_predecessors=True)
    path_up = shortest_path_indices(pred_up, start, end)
    path_down = shortest_path_indices(pred_down, end, start)
    return dist_up[end], path_up, dist_down[start], path_down


def run_path_frozen(D, init, metric, label):
    stages = PF_STAGES[label]
    stage1_options = stages["stage1"]
    stage2_options = stages["stage2"]
    n = len(init)
    n_landmark = max(1, int(round(N_LANDMARK_FRACTION * n)))
    targets_per_landmark = max(1, int(round(TARGETS_PER_LANDMARK_FRACTION * n)))
    common = dict(
        metric=metric,
        optimizer="path_frozen",
        init=init,
        n_components=3,
        graph_neighbors=GRAPH_NEIGHBORS,
        n_landmark=n_landmark,
        random_landmark_fraction=0.0,
        targets_per_landmark=targets_per_landmark,
        n_local_pairs=N_LOCAL_PAIRS,
        local_pair_mode=LOCAL_PAIR_MODE,
        local_global_reweighting=LOCAL_GLOBAL_REWEIGHTING,
        local_weight=LOCAL_WEIGHT,
        random_state=SEED,
        target_random_state=SEED,
        method="L-BFGS-B",
        device="auto",
        verbose=1,
        record_history=True,
        return_result=True,
    )
    print(
        f"\nRunning Path-Frozen {label}: "
        f"n_landmark={n_landmark}, targets_per_landmark={targets_per_landmark}, "
        f"local_pairs={N_LOCAL_PAIRS}, "
        f"stage1={stage1_options}, stage2={stage2_options}"
    )
    start_time = perf_counter()
    stage1 = fit_finsler_mds(D, print_time=True, **common, **stage1_options)
    if stage2_options["outer_iter"] > 0:
        stage2 = fit_finsler_mds(D, print_time=True, **{**common, "init": stage1.embedding}, **stage2_options)
    else:
        stage2 = stage1
    print(f"Path-Frozen {label} total wall time: {perf_counter() - start_time:.3f} s")
    print(
        f"Path-Frozen {label} final full geodesic stress: "
        f"{stage2.final_full_geodesic_stress:.6g}, "
        f"normalized={stage2.final_normalized_full_geodesic_stress:.6g}"
    )
    return stage2


def group_colors(groups):
    return [hysteresis.GROUP_COLORS[group] for group in groups]


def save_embedding_views(X_emb, groups, path, title, path_up=None, path_down=None):
    views = [
        ("oblique", 24, -62),
        ("along x", 0, 0),
        ("along y", 0, 90),
        ("along z", 90, -90),
    ]
    fig = plt.figure(figsize=(11.0, 9.0))
    for index, (view_name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.scatter(X_emb[:, 0], X_emb[:, 1], X_emb[:, 2], c=group_colors(groups), s=5, alpha=0.38, linewidths=0)
        if path_up is not None and len(path_up) > 0:
            P = X_emb[path_up]
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#111111", linewidth=5.0, zorder=10)
        if path_down is not None and len(path_down) > 0:
            P = X_emb[path_down]
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#ffd400", linewidth=5.0, zorder=11)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} - {view_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        utils.set_axes_equal(ax)
        if index == 1:
            hysteresis.add_legend(ax, include_paths=path_up is not None or path_down is not None)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    hysteresis.DATA_METRIC = "matsumoto"
    hysteresis.MATSUMOTO_ALPHA = DATA_ALPHA
    X, groups, support = hysteresis.make_dataset()
    data_metric = hysteresis.make_metric()
    D, data_path_up, data_path_down = hysteresis.compute_distances_and_paths(X, groups, support, data_metric)
    print(
        f"Hysteresis Path-Frozen comparison: n={len(X)}, data_alpha={DATA_ALPHA}, "
        f"embedding_alpha={EMBEDDING_ALPHA}, graph_neighbors={GRAPH_NEIGHBORS}"
    )
    hysteresis.save_3d_views(
        X,
        groups,
        FIG_DIR / "hysteresis_path_frozen_data_paths.pdf",
        "Data optimal directed paths",
        path_up=data_path_up,
        path_down=data_path_down,
    )

    for kind in RUN_METRICS:
        metric = make_embedding_metric(kind)
        result = run_path_frozen(D, X, metric, kind)
        dist_up, path_up, dist_down, path_down = embedding_directed_paths(result.embedding, groups, metric)
        print(
            f"{kind} embedding start->end distance={dist_up:.6g}; "
            f"path counts: {branch_counts(path_up, groups)}"
        )
        print(
            f"{kind} embedding end->start distance={dist_down:.6g}; "
            f"path counts: {branch_counts(path_down, groups)}"
        )
        np.savez(
            RAW_DIR / f"hysteresis_pf_{kind}_a{EMBEDDING_ALPHA:g}.npz",
            embedding=result.embedding,
            groups=groups,
            full_geodesic_stress=result.final_full_geodesic_stress,
            normalized_full_geodesic_stress=result.final_normalized_full_geodesic_stress,
            path_up=path_up,
            path_down=path_down,
            metric=kind,
            alpha=EMBEDDING_ALPHA,
        )
        paths_figure = FIG_DIR / f"hysteresis_pf_{kind}_a{EMBEDDING_ALPHA:g}.pdf"
        nopaths_figure = FIG_DIR / f"hysteresis_pf_{kind}_a{EMBEDDING_ALPHA:g}_nopaths.pdf"
        save_embedding_views(
            result.embedding,
            groups,
            paths_figure,
            f"Path-Frozen {kind}",
            path_up=path_up,
            path_down=path_down,
        )
        save_embedding_views(
            result.embedding,
            groups,
            nopaths_figure,
            f"Path-Frozen {kind}",
        )
        print(f"Saved {kind} figures: {paths_figure} and {nopaths_figure}")
    print(f"Saved Path-Frozen comparison figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
