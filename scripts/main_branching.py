"""Test Path-Frozen on a branching geodesic dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import (  # noqa: E402
    RandersMetric,
    fit_finsler_mds,
    geodesic_embedding_stress,
)
from finsler_mds.utils.branching import load_branching_problem  # noqa: E402


SEED = 42
N_SAMPLES = 1000
GRAPH_NEIGHBORS = 10
CORRIDOR_WIDTH = 0.11
FORCE_RECOMPUTE_DATASET = False

PATH_FROZEN_OPTIONS = {
    "graph_neighbors": 20,
    "outer_iter": 50,
    "inner_iter": 10,
    "n_landmark": int(N_SAMPLES * 0.2),
    "random_landmark_fraction": 1.0,
    "targets_per_landmark": int(N_SAMPLES * 0.35),
    "n_local_pairs": 20,
    "local_weight": 1.0,
    "direct_stress_weight": 0.0,
    "outer_step_size": 1.0,
    "device": "auto",
    "optimizer_options": {"ftol": 1e-9, "maxls": 50},
    "log_frequency": 10,
    "verbose": 2,
}


def main_branching():
    result_dir = Path(__file__).parent / "res" / "path_frozen_benchmarks"
    figure_dir = result_dir / "branching_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    problem = load_branching_problem(
        result_dir / "branching_raw",
        n_samples=N_SAMPLES,
        graph_neighbors=GRAPH_NEIGHBORS,
        corridor_width=CORRIDOR_WIDTH,
        seed=SEED,
        force=FORCE_RECOMPUTE_DATASET,
    )

    metric = RandersMetric(alpha=0.0)
    print("Running Path-Frozen from the contracted ground-truth init")
    result = fit_finsler_mds(
        problem.dissimilarities,
        metric=metric,
        optimizer="path_frozen",
        n_components=2,
        init=problem.init,
        random_state=SEED,
        print_time=True,
        return_result=True,
        **PATH_FROZEN_OPTIONS,
    )
    full_stress = result.final_normalized_full_geodesic_stress
    if full_stress is None:
        full_stress = geodesic_embedding_stress(
            result.embedding,
            problem.dissimilarities,
            metric=metric,
            n_neighbors=PATH_FROZEN_OPTIONS["graph_neighbors"],
            normalized_stress=True,
            on_unreachable="inf",
        )
    print(f"Path-Frozen masked stress: {result.stress:.6g}")
    print(f"Path-Frozen normalized full geodesic stress: {full_stress:.6g}")

    outer_iter = PATH_FROZEN_OPTIONS["outer_iter"]
    inner_iter = PATH_FROZEN_OPTIONS["inner_iter"]
    plot_comparison(
        problem,
        result.embedding,
        figure_dir / f"pf_n{N_SAMPLES}_oi{outer_iter}_ii{inner_iter}.pdf",
    )
    print(f"Saved figure in {figure_dir}")


def plot_comparison(problem, embedding, path):
    embeddings = (problem.X, problem.init, embedding)
    titles = ("ground-truth tree", "contracted/noisy init", "Path-Frozen")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, points, title in zip(axes, embeddings, titles):
        _plot_graph_edges(ax, points, problem.graph)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=problem.branch_labels,
            s=9,
            cmap="tab10",
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    fig.colorbar(
        scatter,
        ax=axes,
        label="primary branch",
        ticks=sorted(set(problem.branch_labels)),
        shrink=0.86,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_graph_edges(ax, points, graph):
    graph = graph.tocoo()
    for i, j in zip(graph.row, graph.col):
        if i < j:
            ax.plot(
                points[[i, j], 0],
                points[[i, j], 1],
                color="0.2",
                alpha=0.045,
                linewidth=0.35,
            )


if __name__ == "__main__":
    main_branching()
