"""Swiss-roll experiment with a hole and the Wormhole weighting criterion."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
    fit_finsler_mds,
    utils,
)
from finsler_mds.utils.embedding_io import metric_alpha_tag


# Main choices -----------------------------------------------------------------
OPTIMIZER = "path_frozen"  # "smacof", "gradient_descent", or "path_frozen"
FINSLER_METRIC = "matsumoto"  # "randers", "matsumoto", or "convexified_matsumoto"
INIT = "isomap"  # "isomap" or "random"
N_COMPONENTS = 2  # 2 or 3

N_SAMPLES = 800
N_NEIGHBORS = 8
NOISE_LEVEL = 0.0
DATA_ALPHA = 0.5
EMBEDDING_ALPHA = 0.5
WORMHOLE_DISTANCE_THRESHOLD = 3.0
SEED = 3

OPTIMIZER_OPTIONS = {
    "smacof": {
        "max_iter": 150,
        "project_on_V": True,
        "check_monotony": False,
        "device": "auto",
        "verbose": 1,
    },
    "gradient_descent": {
        "max_iter": 300,
        "optimizer_options": {"ftol": 1e-9, "maxls": 40},
        "device": "auto",
        "verbose": 1,
    },
    "path_frozen": {
        "graph_neighbors": N_NEIGHBORS,
        "outer_iter": 12,
        "inner_iter": 4,
        "direct_stress_weight": 0.0,
        "outer_step_size": 1.0,
        "device": "auto",
        "verbose": 1,
    },
}
BASELINE_SMACOF_OPTIONS = {
    "max_iter": 20,
    "device": "auto",
    "verbose": 1,
}

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "res" / "swiss_roll_hole"
FIGURE_DIR = RESULT_DIR / "figures"
EMBEDDING_DIR = RESULT_DIR / "embeddings"

METRIC_CLASSES = {
    "randers": RandersMetric,
    "matsumoto": MatsumotoMetric,
    "convexified_matsumoto": ConvexifiedMatsumotoMetric,
}
METRIC_TAGS = {
    "randers": "r",
    "matsumoto": "mats",
    "convexified_matsumoto": "cmats",
}
OPTIMIZER_TAGS = {
    "smacof": "smacof",
    "gradient_descent": "gd",
    "path_frozen": "pf",
}


def main_swiss_roll_hole():
    validate_configuration()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    parameters, noiseless, data, noise = make_swiss_roll()
    colors = noiseless
    plot_data(data, colors, "Full Swiss roll", "data_full.pdf")
    plot_neighbors = utils.nearest_neighbors(data, N_NEIGHBORS)

    print("Isomap on the full Swiss roll")
    isomap = utils.IsomapWithPreds(n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS)
    isomap_full = isomap.fit_transform(data)
    plot_embedding(isomap_full, data, colors, plot_neighbors, "Full Isomap", "isomap_full.pdf")

    randers_field = make_randers_field(parameters)
    plot_data(
        data,
        colors,
        f"Full Randers field, alpha={DATA_ALPHA:g}",
        "field_full.pdf",
        field=randers_field,
    )
    full_dissimilarities, _ = utils.compute_dist_matrix(
        data,
        n_neighbors=N_NEIGHBORS,
        randers_field=randers_field,
    )
    full_init = make_initialization(isomap_full, len(data), SEED + 1)
    full_embedding, full_objective = fit_selected(full_dissimilarities, full_init)
    save_embedding("full", full_embedding, full_objective)
    plot_embedding(
        full_embedding,
        data,
        colors,
        plot_neighbors,
        f"Full {optimizer_display_name()} — {metric_display_name()}",
        f"{result_stem()}_full.pdf",
    )

    # Remove the central strip and identify its boundary in the remaining data.
    removed = (
        (parameters[:, 1] > 0.4)
        & (parameters[:, 1] < 0.6)
        & (parameters[:, 0] > 0.1)
        & (parameters[:, 0] < 0.9)
    )
    border = np.any(
        np.isin(utils.nearest_neighbors(data, 6), np.flatnonzero(removed)),
        axis=1,
    )[~removed]
    hole_parameters = parameters[~removed]
    hole_noiseless = swiss_roll_coordinates(hole_parameters)
    hole_data = hole_noiseless + noise[~removed]
    hole_field = randers_field[~removed]
    hole_neighbors = utils.nearest_neighbors(hole_data, N_NEIGHBORS)

    plot_data(hole_data, hole_noiseless, "Swiss roll with a hole", "data_hole.pdf")
    plot_data(
        hole_data,
        hole_noiseless,
        f"Hole Randers field, alpha={DATA_ALPHA:g}",
        "field_hole.pdf",
        field=hole_field,
    )

    print("Isomap on the Swiss roll with a hole")
    isomap_hole = isomap.fit_transform(hole_data)
    symmetric_hole_distances = isomap.dist_matrix_
    hole_init = make_initialization(isomap_hole, len(hole_data), SEED + 3)
    plot_embedding(
        isomap_hole,
        hole_data,
        hole_noiseless,
        hole_neighbors,
        "Hole Isomap",
        "isomap_hole.pdf",
    )

    print("Euclidean Wormhole baseline")
    euclidean_mask, _, _ = utils.wormhole_mask(
        symmetric_hole_distances,
        border,
        hole_data,
        randers_drift_upper_bound=0.0,
        small_dists_threshold=WORMHOLE_DISTANCE_THRESHOLD,
    )
    baseline, baseline_objective = fit_finsler_mds(
        symmetric_hole_distances,
        optimizer="smacof",
        metric=RandersMetric(0.0),
        init=hole_init,
        n_components=N_COMPONENTS,
        n_init=1,
        weight=symmetrized_mask(euclidean_mask),
        random_state=SEED,
        print_time=True,
        **BASELINE_SMACOF_OPTIONS,
    )
    baseline_stem = f"smacof_{N_COMPONENTS}d"
    save_embedding("hole_wormhole_baseline", baseline, baseline_objective, stem=baseline_stem)
    plot_embedding(
        baseline,
        hole_data,
        hole_noiseless,
        hole_neighbors,
        "Euclidean Wormhole SMACOF",
        f"{baseline_stem}_hole_wormhole_baseline.pdf",
    )

    hole_dissimilarities, _ = utils.compute_dist_matrix(
        hole_data,
        n_neighbors=N_NEIGHBORS,
        randers_field=hole_field,
    )
    plot_distance_matrix(hole_dissimilarities)
    finsler_mask, _, _ = utils.wormhole_mask(
        symmetric_hole_distances,
        border,
        hole_data,
        randers_drift_upper_bound=DATA_ALPHA,
        small_dists_threshold=WORMHOLE_DISTANCE_THRESHOLD,
    )

    print(f"Uniform {optimizer_display_name()} embedding on the hole dataset")
    uniform_embedding, uniform_objective = fit_selected(hole_dissimilarities, hole_init)
    save_embedding("hole_uniform", uniform_embedding, uniform_objective)
    plot_embedding(
        uniform_embedding,
        hole_data,
        hole_noiseless,
        hole_neighbors,
        f"Hole uniform — {optimizer_display_name()}, {metric_display_name()}",
        f"{result_stem()}_hole_uniform.pdf",
    )

    print(f"Wormhole-weighted {optimizer_display_name()} embedding")
    wormhole_embedding, wormhole_objective = fit_selected(
        hole_dissimilarities,
        hole_init,
        weight=finsler_mask,
        eps=1e-3,
    )
    save_embedding("hole_wormhole", wormhole_embedding, wormhole_objective)
    plot_embedding(
        wormhole_embedding,
        hole_data,
        hole_noiseless,
        hole_neighbors,
        f"Hole Wormhole — {optimizer_display_name()}, {metric_display_name()}",
        f"{result_stem()}_hole_wormhole.pdf",
    )

    print(f"Saved results in {RESULT_DIR}")
    plt.show()


def validate_configuration():
    if OPTIMIZER not in OPTIMIZER_OPTIONS:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER!r}")
    if FINSLER_METRIC not in METRIC_CLASSES:
        raise ValueError(f"Unknown Finsler metric: {FINSLER_METRIC!r}")
    if OPTIMIZER == "smacof" and FINSLER_METRIC != "randers":
        raise ValueError("SMACOF only supports the Randers metric.")
    if INIT not in {"isomap", "random"}:
        raise ValueError("INIT must be 'isomap' or 'random'.")
    if N_COMPONENTS not in {2, 3}:
        raise ValueError("N_COMPONENTS must be 2 or 3.")


def make_swiss_roll():
    rng = np.random.RandomState(SEED)
    parameters = rng.rand(N_SAMPLES, 2)
    noiseless = swiss_roll_coordinates(parameters)
    noise = NOISE_LEVEL * rng.randn(*noiseless.shape)
    return parameters, noiseless, noiseless + noise, noise


def swiss_roll_coordinates(parameters):
    angle = parameters[:, 0] * 3 * np.pi + 1.5 * np.pi
    height = parameters[:, 1] * 20
    return np.column_stack([angle * np.cos(angle), height, angle * np.sin(angle)])


def make_randers_field(parameters):
    angle = parameters[:, 0] * 3 * np.pi + 1.5 * np.pi
    field = np.column_stack(
        [
            np.cos(angle) - angle * np.sin(angle),
            np.zeros(len(angle)),
            np.sin(angle) + angle * np.cos(angle),
        ]
    )
    return DATA_ALPHA * field / np.linalg.norm(field, axis=1, keepdims=True)


def make_initialization(isomap_embedding, n_samples, seed):
    if INIT == "random":
        return np.random.RandomState(seed).rand(n_samples, N_COMPONENTS)
    init = isomap_embedding.copy()
    if N_COMPONENTS in {2, 3}:
        init[:, 0], init[:, 1] = -isomap_embedding[:, 1], isomap_embedding[:, 0]
    return init


def fit_selected(dissimilarities, init, *, weight=None, eps=None):
    metric = METRIC_CLASSES[FINSLER_METRIC](EMBEDDING_ALPHA)
    options = dict(OPTIMIZER_OPTIONS[OPTIMIZER])
    if eps is not None:
        options["eps"] = eps
    if OPTIMIZER == "smacof":
        options["n_init"] = 1
        if weight is not None:
            weight = symmetrized_mask(weight)

    return fit_finsler_mds(
        dissimilarities,
        optimizer=OPTIMIZER,
        metric=metric,
        init=init,
        n_components=N_COMPONENTS,
        weight=weight,
        random_state=SEED,
        print_time=True,
        **options,
    )


def symmetrized_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    return mask & mask.T


def result_stem():
    return (
        f"{OPTIMIZER_TAGS[OPTIMIZER]}_{N_COMPONENTS}d_"
        f"vrand{metric_alpha_tag(DATA_ALPHA)}_"
        f"{METRIC_TAGS[FINSLER_METRIC]}{metric_alpha_tag(EMBEDDING_ALPHA)}"
    )


def save_embedding(suffix, embedding, objective, *, stem=None):
    stem = result_stem() if stem is None else stem
    np.savez(
        EMBEDDING_DIR / f"{stem}_{suffix}.npz",
        embedding=embedding,
        objective=objective,
    )


def plot_data(data, colors, title, filename, *, field=None):
    fig, _ = utils.plot_points(
        data,
        X_noiseless=colors,
        shape_type="swiss_roll",
        quiver_field=field,
        step_quiver=1,
    )
    utils.set_window_title(fig, title)
    fig.savefig(FIGURE_DIR / filename)


def plot_embedding(embedding, data, colors, neighbors, title, filename):
    fig, ax = utils.plot_proj_points(
        embedding,
        X=data,
        knn=neighbors,
        X_noiseless=colors,
        shape_type="swiss_roll",
        edge_alpha=1.0,
    )
    utils.set_window_title(fig, title)
    ax.axis("off")
    fig.savefig(FIGURE_DIR / filename)


def plot_distance_matrix(dissimilarities):
    fig, ax = plt.subplots()
    utils.set_window_title(fig, f"Hole Finsler distances, alpha={DATA_ALPHA:g}")
    image = ax.imshow(dissimilarities)
    fig.colorbar(image, ax=ax)
    fig.savefig(FIGURE_DIR / f"distances_hole_vrand{metric_alpha_tag(DATA_ALPHA)}.pdf")


def optimizer_display_name():
    return {
        "smacof": "SMACOF",
        "gradient_descent": "gradient descent",
        "path_frozen": "Path-Frozen",
    }[OPTIMIZER]


def metric_display_name():
    return METRIC_CLASSES[FINSLER_METRIC].__name__.removesuffix("Metric")


if __name__ == "__main__":
    main_swiss_roll_hole()
