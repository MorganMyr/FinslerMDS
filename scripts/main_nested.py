"""Compare direct MDS and Path-Frozen on nested rectangles."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds, utils
from finsler_mds.utils.graph import symmetric_knn_graph


SEED = 42
POINTS_PER_UNIT = 7
INIT_NOISE_SCALE = 0.2
TARGET_GRAPH_NEIGHBORS = 4
PATH_FROZEN_INIT = "diagonal"  # "diagonal" or "smacof"

SMACOF_OPTIONS = {
    "max_iter": 500,
    "eps": 1e-6,
    "check_monotony": True,
    "project_on_V": True,
    "device": "auto",
    "verbose": 1,
}
PATH_FROZEN_OPTIONS = {
    "graph_neighbors": 4,
    "n_landmark": 150,
    "random_landmark_fraction": 1.0,
    "targets_per_landmark": 200,
    "n_local_pairs": 4,
    "local_weight": 1.0,
    "direct_stress_weight": 0.0,
    "device": "auto",
    "verbose": 1,
}
PATH_FROZEN_STAGES = (
    {"outer_iter": 50, "inner_iter": 10, "log_frequency": 10, "outer_step_size": 0.1},
    {"outer_iter": 50, "inner_iter": 1, "log_frequency": 10, "outer_step_size": 0.1},
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "res" / "nested"
FIGURE_DIR = RESULT_DIR / "figures"
EMBEDDING_DIR = RESULT_DIR / "embeddings"

GROUP_COLORS = {
    "outer rectangle": "#377eb8",
    "inner rectangle": "#e41a1c",
    "central bar": "#4daf4a",
}


def main_nested():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    data, labels = make_nested_rectangles()
    dissimilarities = graph_distances(data)
    diagonal_init = make_diagonal_init(data)
    metric = RandersMetric(alpha=0.0)

    print(f"Nested rectangles: n={len(data)}, target_k={TARGET_GRAPH_NEIGHBORS}")
    smacof = fit_finsler_mds(
        dissimilarities,
        optimizer="smacof",
        metric=metric,
        init=diagonal_init,
        n_components=2,
        n_init=1,
        random_state=SEED,
        return_result=True,
        print_time=True,
        **SMACOF_OPTIONS,
    )

    initializations = {"diagonal": diagonal_init, "smacof": smacof.embedding}
    try:
        path_frozen_init = initializations[PATH_FROZEN_INIT]
    except KeyError as exc:
        raise ValueError("PATH_FROZEN_INIT must be 'diagonal' or 'smacof'.") from exc
    path_frozen = fit_path_frozen_stages(dissimilarities, path_frozen_init, metric)

    save_result(EMBEDDING_DIR / "smacof_2d.npz", smacof)
    save_result(EMBEDDING_DIR / "pf_2d.npz", path_frozen)
    save_figures(data, labels, diagonal_init, smacof.embedding, path_frozen.embedding)

    print(f"SMACOF stress: {smacof.stress:.6g}")
    print(f"Path-Frozen stress: {path_frozen.stress:.6g}")
    print(
        "Path-Frozen normalized full geodesic stress: "
        f"{path_frozen.final_normalized_full_geodesic_stress:.6g}"
    )
    print(f"Saved results in {RESULT_DIR}")


def fit_path_frozen_stages(dissimilarities, init, metric):
    result = None
    for stage_number, stage_options in enumerate(PATH_FROZEN_STAGES, start=1):
        print(f"\nRunning Path-Frozen stage {stage_number}")
        result = fit_finsler_mds(
            dissimilarities,
            optimizer="path_frozen",
            metric=metric,
            init=init if result is None else result.embedding,
            n_components=2,
            random_state=SEED,
            return_result=True,
            print_time=True,
            **PATH_FROZEN_OPTIONS,
            **stage_options,
        )
    return result


def make_nested_rectangles():
    segments = [
        ((-5, 0, -2), (5, 0, -2), "outer rectangle"),
        ((-5, 0, 2), (5, 0, 2), "outer rectangle"),
        ((-5, 0, -2), (-5, 0, 2), "outer rectangle"),
        ((5, 0, -2), (5, 0, 2), "outer rectangle"),
        ((0, -3, -1), (0, 3, -1), "inner rectangle"),
        ((0, -3, 1), (0, 3, 1), "inner rectangle"),
        ((0, -3, -1), (0, -3, 1), "inner rectangle"),
        ((0, 3, -1), (0, 3, 1), "inner rectangle"),
        ((0, 0, -2), (0, 0, 2), "central bar"),
    ]
    points = {}
    for start, end, label in segments:
        for point in segment_points(start, end):
            points.setdefault(tuple(np.round(point, 12)), (point, label))

    ordered = [points[key] for key in sorted(points)]
    return (
        np.vstack([point for point, _ in ordered]),
        np.asarray([label for _, label in ordered]),
    )


def segment_points(start, end):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    count = max(2, round(POINTS_PER_UNIT * np.linalg.norm(end - start)) + 1)
    return np.linspace(start, end, count)


def make_diagonal_init(data):
    theta = np.deg2rad(45)
    init = np.column_stack(
        [
            np.sqrt(2) * (np.cos(theta) * data[:, 0] - np.sin(theta) * data[:, 1]),
            data[:, 2],
        ]
    )
    init -= init.mean(axis=0)
    noise_scale = INIT_NOISE_SCALE / POINTS_PER_UNIT
    init += np.random.default_rng(SEED).normal(scale=noise_scale, size=init.shape)
    return init - init.mean(axis=0)


def graph_distances(data):
    graph = symmetric_knn_graph(
        data,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        ensure_connected=True,
        warn_on_connect=True,
    )
    distances = shortest_path(graph, directed=False)
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("The target kNN graph is not connected.")
    return distances


def save_result(path, result):
    np.savez(path, embedding=result.embedding, objective=result.stress)


def save_figures(data, labels, diagonal_init, smacof, path_frozen):
    fig, _ = utils.plot_3d_embedding_views(
        data,
        labels=labels,
        title="Nested rectangles",
        cmap=GROUP_COLORS,
        s=20,
        save_path=FIGURE_DIR / "data_3d.pdf",
    )
    plt.close(fig)

    embeddings = {
        "diagonal_init": ("Diagonal-view initialization", diagonal_init),
        "smacof": ("SMACOF", smacof),
        "path_frozen": ("Path-Frozen", path_frozen),
    }
    for name, (title, embedding) in embeddings.items():
        fig, ax = utils.plot_categorical_embedding(
            embedding,
            labels=labels,
            title=title,
            xlabel="",
            ylabel="",
            cmap=GROUP_COLORS,
            s=22,
        )
        ax.set_aspect("equal", adjustable="datalim")
        fig.savefig(FIGURE_DIR / f"{name}_2d.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main_nested()
