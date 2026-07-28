"""Compare direct Finsler-MDS and Path-Frozen on a converging flow."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, asymmetry_score, fit_finsler_mds, utils
from finsler_mds.evaluation.distance_embedding import compute_embedding_distances
from finsler_mds.utils.embedding_io import metric_alpha_tag


SEED = 42
RAIL_X = (-0.5, 0.5)
Y_RANGE = (-4.0, 4.0)
BRIDGE_Y = (-4.0, 0.0, 4.0)
RAIL_POINTS = 61
BRIDGE_POINTS = 8

DATA_ALPHA = 0.5
EMBEDDING_ALPHA = 0.5
TARGET_GRAPH_NEIGHBORS = 4
EMBEDDING_GRAPH_NEIGHBORS = 4
PATH_FROZEN_INIT = "smacof"  # "data" or "smacof"

SMACOF_OPTIONS = {
    "max_iter": 500,
    "eps": 1e-6,
    "check_monotony": False,
    "project_on_V": True,
    "device": "cpu",
    "verbose": 1,
}
PATH_FROZEN_OPTIONS = {
    "graph_neighbors": EMBEDDING_GRAPH_NEIGHBORS,
    "n_landmark": 0,
    "targets_per_landmark": None,
    "n_local_pairs": None,
    "device": "auto",
    "verbose": 1,
}
PATH_FROZEN_STAGES = (
    {"outer_iter": 70, "inner_iter": 50, "log_frequency": 5, "outer_step_size": 0.5},
    {"outer_iter": 40, "inner_iter": 3, "log_frequency": 10, "outer_step_size": 0.5},
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "res" / "converging_flow"
FIGURE_DIR = RESULT_DIR / "figures"
EMBEDDING_DIR = RESULT_DIR / "embeddings"


def main_converging_flow():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    data, current = make_converging_flow()
    dissimilarities = randers_dissimilarities(data, current)
    data_init = np.column_stack([data, np.zeros(len(data))])
    metric = RandersMetric(alpha=EMBEDDING_ALPHA)

    print(
        f"Converging flow: n={len(data)}, data_alpha={DATA_ALPHA:g}, "
        f"embedding_alpha={EMBEDDING_ALPHA:g}"
    )
    smacof = fit_finsler_mds(
        dissimilarities,
        optimizer="smacof",
        metric=metric,
        init=data_init,
        n_components=3,
        n_init=1,
        random_state=SEED,
        return_result=True,
        print_time=True,
        **SMACOF_OPTIONS,
    )

    initializations = {"data": data_init, "smacof": smacof.embedding}
    try:
        path_frozen_init = initializations[PATH_FROZEN_INIT]
    except KeyError as exc:
        raise ValueError("PATH_FROZEN_INIT must be 'data' or 'smacof'.") from exc
    path_frozen = fit_path_frozen_stages(dissimilarities, path_frozen_init, metric)

    for name, result in {"smacof": smacof, "pf": path_frozen}.items():
        np.savez(
            EMBEDDING_DIR / f"{result_stem(name)}.npz",
            embedding=result.embedding,
            objective=result.stress,
        )
        log_asymmetry(name.upper(), dissimilarities, result.embedding, metric)
    print(
        "Path-Frozen normalized full geodesic stress: "
        f"{path_frozen.final_normalized_full_geodesic_stress:.6g}"
    )
    save_figures(data, current, data_init, smacof.embedding, path_frozen.embedding)
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
            n_components=3,
            random_state=SEED,
            return_result=True,
            print_time=True,
            **PATH_FROZEN_OPTIONS,
            **stage_options,
        )
    return result


def make_converging_flow():
    rail_y = np.linspace(*Y_RANGE, RAIL_POINTS)
    bridge_x = np.linspace(*RAIL_X, BRIDGE_POINTS)
    rails = [[x, y] for x in RAIL_X for y in rail_y]
    bridges = [[x, y] for y in BRIDGE_Y for x in bridge_x]
    data = np.unique(np.asarray(rails + bridges), axis=0)
    return data, current_field(data)


def current_field(data):
    current = np.zeros_like(data)
    on_rail = np.isclose(data[:, 0], RAIL_X[0]) | np.isclose(data[:, 0], RAIL_X[1])
    current[on_rail, 1] = -np.sign(data[on_rail, 1])
    return current


def randers_dissimilarities(data, current):
    # compute_dist_matrix uses F_i(u) = ||u|| + <randers_field_i, u>.
    # The minus sign makes motion along the displayed current less expensive.
    distances, _ = utils.compute_dist_matrix(
        data,
        n_neighbors=TARGET_GRAPH_NEIGHBORS,
        randers_field=-DATA_ALPHA * current,
    )
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("The directed graph contains unreachable pairs.")
    return distances


def log_asymmetry(label, data_distances, embedding, metric):
    data_asymmetry = asymmetry_score(data_distances, data_distances.T)
    for mode in ("direct", "geodesic"):
        embedding_distances = compute_embedding_distances(
            embedding,
            metric=metric,
            mode=mode,
            n_neighbors=EMBEDDING_GRAPH_NEIGHBORS,
        )
        embedding_asymmetry = asymmetry_score(embedding_distances, embedding_distances.T)
        target, observed = finite_pairs(data_asymmetry, embedding_asymmetry)
        pearson = (
            np.corrcoef(target, observed)[0, 1]
            if np.std(target) and np.std(observed)
            else np.nan
        )
        beta = np.dot(target, observed) / np.dot(target, target) if np.any(target) else np.nan
        rmse = np.sqrt(np.mean((observed - target) ** 2))
        print(
            f"{label} asymmetry ({mode}): Pearson={pearson:.6g}, "
            f"beta={beta:.6g}, RMSE={rmse:.6g}"
        )

def finite_pairs(first, second):
    mask = np.isfinite(first) & np.isfinite(second)
    if not np.any(mask):
        raise RuntimeError("No finite pair is available for the asymmetry evaluation.")
    return first[mask], second[mask]


def result_stem(method):
    return (
        f"{method}_3d_vrand{metric_alpha_tag(DATA_ALPHA)}_"
        f"r{metric_alpha_tag(EMBEDDING_ALPHA)}"
    )


def save_figures(data, current, data_init, smacof, path_frozen):
    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    ax.scatter(data[:, 0], data[:, 1], color="#333333", s=22, linewidths=0)
    moving = np.flatnonzero(np.linalg.norm(current, axis=1) > 0)
    moving = moving[2 :: max(1, int(np.ceil(len(moving) / 28)))]
    ax.quiver(
        data[moving, 0],
        data[moving, 1],
        current[moving, 0],
        current[moving, 1],
        angles="xy",
        scale_units="xy",
        scale=3,
        color="#d62728",
        width=0.03,
    )
    ax.set(title="Converging flow", xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"data_2d_vrand{metric_alpha_tag(DATA_ALPHA)}.pdf")
    plt.close(fig)

    for name, title, embedding in (
        ("init", "3D initialization", data_init),
        ("smacof", "SMACOF", smacof),
        ("pf", "Path-Frozen", path_frozen),
    ):
        filename = "init_3d" if name == "init" else result_stem(name)
        fig, _ = utils.plot_3d_embedding_views(
            embedding,
            title=title,
            s=18,
            save_path=FIGURE_DIR / f"{filename}.pdf",
        )
        plt.close(fig)


if __name__ == "__main__":
    main_converging_flow()
