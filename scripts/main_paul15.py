"""Embed Paul15 from PCA or diffusion-map coordinates.

Edit the short configuration block below, then run this file from the project
root. Every run saves the optimized embedding and cluster/DPT figures, and
creates the corresponding UMAP reference if needed.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import (  # noqa: E402
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
    fit_finsler_mds,
)
from finsler_mds.utils.embedding_io import scale_embedding_to_dissimilarities  # noqa: E402
from finsler_mds.utils.paul15 import (  # noqa: E402
    COMBINED_LINEAGE_DPT_KEY,
    PAUL15_CLUSTER_KEY,
    PAUL15_DPT_KEY,
    ensure_paul15_umap,
    latest_paul15_embedding,
    paul15_dissimilarities,
    paul15_method_tag,
    paul15_result_stem,
    prepare_paul15,
    save_paul15_embedding,
    save_paul15_plots,
)


# Main choices -----------------------------------------------------------------
REPRESENTATION = "diffmap"  # "pca" or "diffmap"
OPTIMIZER = "gradient_descent"  # "finsler_umap", "gradient_descent", "path_frozen"
FINSLER_METRIC = "randers"  # "randers", "matsumoto", "convexified_matsumoto"
INIT = "umap"  # "umap"/"umap_2d", "umap_3d", or an optimizer above
N_COMPONENTS = 3  # 2 or 3
LAMBDA_TIME = 0.5  # DPT asymmetry in the input graph
FINSLER_ALPHA = 0.3  # Finsler asymmetry in the embedding
PSEUDOTIME_KEY = COMBINED_LINEAGE_DPT_KEY  # or PAUL15_DPT_KEY

SEED = 42
EXCLUDED_CLUSTERS = ("19Lymph", "11DC")
TARGET_NEIGHBORS = 12
UMAP_NEIGHBORS = 20

OPTIMIZER_OPTIONS = {
    "gradient_descent": dict(
        max_iter=300,
        optimizer_options={"ftol": 1e-8, "maxls": 30},
        device="auto",  # "cpu", "auto", "gpu", or "cuda"
        verbose=1,
    ),
    "path_frozen": dict(
        graph_neighbors=20,
        outer_iter=50,
        inner_iter=5,
        n_local_pairs=20,
        n_landmark=200,
        random_landmark_fraction=1.0,
        targets_per_landmark=200,
        local_weight=1.0,
        direct_stress_weight=0.0,
        outer_step_size=1.0,
        device="auto",  # "cpu", "auto", "gpu", or "cuda"
        verbose=1,
    ),
    "finsler_umap": dict(
        n_neighbors=30,
        max_iter=1500,
        negative_sample_rate=10,
        verbose=1,
    ),
}


def main():
    if OPTIMIZER not in OPTIMIZER_OPTIONS:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER!r}")
    if N_COMPONENTS not in (2, 3):
        raise ValueError("N_COMPONENTS must be 2 or 3.")
    metric = make_finsler_metric(FINSLER_METRIC, FINSLER_ALPHA)
    umap_components = requested_umap_components(INIT, N_COMPONENTS)

    np.random.seed(SEED)
    sc.settings.autoshow = False
    result_dir = Path(__file__).parent / "res" / "paul15" / "main"
    embedding_dir = result_dir / "embeddings"
    figure_dir = result_dir / "figures"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    adata = prepare_paul15(
        excluded_clusters=EXCLUDED_CLUSTERS,
        n_pcs=20,
        pseudotime_key=PSEUDOTIME_KEY,
        seed=SEED,
    )
    cell_ids = np.asarray(adata.obs_names.astype(str))
    labels = np.asarray(adata.obs[PAUL15_CLUSTER_KEY].astype(str))
    pseudotime = np.asarray(adata.obs[PSEUDOTIME_KEY], dtype=float)

    dissimilarities = paul15_dissimilarities(
        adata,
        representation=REPRESENTATION,
        n_neighbors=TARGET_NEIGHBORS,
        lambda_time=LAMBDA_TIME,
        pseudotime_key=PSEUDOTIME_KEY,
        seed=SEED,
    )
    plotted_umap_components = 2 if umap_components is None else umap_components
    umap_stem = paul15_result_stem(
        "umap",
        representation=REPRESENTATION,
        n_components=plotted_umap_components,
        n_neighbors=UMAP_NEIGHBORS,
    )
    umap = ensure_paul15_umap(
        adata,
        embedding_dir / f"{umap_stem}.npz",
        representation=REPRESENTATION,
        n_components=plotted_umap_components,
        n_neighbors=UMAP_NEIGHBORS,
        seed=SEED,
    )
    save_paul15_plots(
        umap,
        labels=labels,
        pseudotime=pseudotime,
        pseudotime_key=PSEUDOTIME_KEY,
        directory=figure_dir,
        stem=umap_stem,
        title=f"Paul15 UMAP {plotted_umap_components}D ({REPRESENTATION})",
    )
    if umap_components is not None:
        if N_COMPONENTS == 3 and umap_components == 2:
            umap = np.column_stack((umap, np.zeros(adata.n_obs)))
        init, scale = scale_embedding_to_dissimilarities(
            umap,
            dissimilarities,
            random_state=SEED,
        )
        print(f"Using UMAP {umap_components}D init (scale={scale:.4g})")
    else:
        init = latest_paul15_embedding(
            embedding_dir,
            INIT,
            expected_shape=(adata.n_obs, N_COMPONENTS),
            cell_ids=cell_ids,
        )

    embedding, objective = fit_finsler_mds(
        dissimilarities,
        metric=metric,
        optimizer=OPTIMIZER,
        n_components=N_COMPONENTS,
        init=init,
        random_state=SEED,
        print_time=True,
        **OPTIMIZER_OPTIONS[OPTIMIZER],
    )
    if embedding.shape != (adata.n_obs, N_COMPONENTS):
        raise RuntimeError(
            f"{OPTIMIZER} returned shape {embedding.shape}; "
            f"expected {(adata.n_obs, N_COMPONENTS)}."
        )
    stem = paul15_result_stem(
        OPTIMIZER,
        representation=REPRESENTATION,
        metric=FINSLER_METRIC,
        lambda_time=LAMBDA_TIME,
        pseudotime_key=PSEUDOTIME_KEY,
        alpha=FINSLER_ALPHA,
        n_components=N_COMPONENTS,
    )
    save_paul15_embedding(
        embedding_dir / f"{stem}.npz",
        embedding,
        cell_ids,
        objective=objective,
    )
    save_paul15_plots(
        embedding,
        labels=labels,
        pseudotime=pseudotime,
        pseudotime_key=PSEUDOTIME_KEY,
        directory=figure_dir,
        stem=stem,
        title=(
            f"Paul15 {OPTIMIZER.replace('_', ' ')}, "
            f"{FINSLER_METRIC.replace('_', ' ')} ({REPRESENTATION})"
        ),
    )
    plt.close("all")


def requested_umap_components(init_kind, n_components):
    init_kind = str(init_kind).lower().replace("-", "_")
    if init_kind in {"umap", "umap_2d"}:
        return 2
    if init_kind == "umap_3d":
        if n_components != 3:
            raise ValueError("INIT='umap_3d' requires N_COMPONENTS=3.")
        return 3
    allowed_tags = {paul15_method_tag(method) for method in OPTIMIZER_OPTIONS}
    if paul15_method_tag(init_kind) not in allowed_tags:
        raise ValueError(
            f"INIT must be a UMAP variant or one of {tuple(OPTIMIZER_OPTIONS)}."
        )
    return None


def make_finsler_metric(kind, alpha):
    kind = str(kind).lower().replace("-", "_")
    metric_classes = {
        "randers": RandersMetric,
        "matsumoto": MatsumotoMetric,
        "convexified_matsumoto": ConvexifiedMatsumotoMetric,
    }
    try:
        return metric_classes[kind](alpha=alpha)
    except KeyError as exc:
        raise ValueError(f"Unknown FINSLER_METRIC: {kind!r}.") from exc


if __name__ == "__main__":
    main()
