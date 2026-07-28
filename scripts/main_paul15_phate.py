"""Replace PHATE's final MDS by gradient descent or Path-Frozen on Paul15."""

from pathlib import Path
import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist, squareform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds  # noqa: E402
from finsler_mds.utils.embedding_io import scale_embedding_to_dissimilarities  # noqa: E402
from finsler_mds.utils.paul15 import (  # noqa: E402
    COMBINED_LINEAGE_DPT_KEY,
    PAUL15_CLUSTER_KEY,
    PAUL15_DPT_KEY,
    ensure_paul15_umap,
    latest_paul15_embedding,
    paul15_method_tag,
    paul15_result_stem,
    prepare_paul15,
    save_paul15_embedding,
    save_paul15_plots,
)


# Main choices -----------------------------------------------------------------
OPTIMIZER = "gradient_descent"        # "gradient_descent" or "path_frozen"
INIT = "umap"                  # "classic_mds", "umap"/"umap_2d", "gradient_descent", "path_frozen", "phate"
N_LANDMARK = 500
PSEUDOTIME_KEY = COMBINED_LINEAGE_DPT_KEY  # or PAUL15_DPT_KEY for global DPT

SEED = 42
EXCLUDED_CLUSTERS = ("19Lymph", "11DC")
UMAP_NEIGHBORS = 20

PHATE_OPTIONS = dict(
    n_components=2,
    knn=15,
    decay=40,
    n_landmark=N_LANDMARK,
    t="auto",
    gamma=1,
    mds="metric",
    mds_solver="smacof",
    n_jobs=-1,
    random_state=SEED,
    verbose=1,
)

OPTIMIZER_OPTIONS = {
    "gradient_descent": dict(
        max_iter=300,
        optimizer_options={"ftol": 1e-8, "maxls": 30},
        device="auto",
        verbose=1,
    ),
    "path_frozen": dict(
        graph_neighbors=30,
        outer_iter=50,
        inner_iter=10,
        n_local_pairs=30,
        n_landmark=150,
        random_landmark_fraction=1.0,
        targets_per_landmark=200,
        local_weight=0.3,
        direct_stress_weight=0.01,
        outer_step_size=1,
        device="auto",
        verbose=1,
    ),
}


def main():
    if OPTIMIZER not in OPTIMIZER_OPTIONS:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER!r}")
    validate_init(INIT)

    np.random.seed(SEED)
    sc.settings.autoshow = False
    result_dir = Path(__file__).parent / "res" / "paul15" / "phate"
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

    phate = import_phate()
    operator = phate.PHATE(**PHATE_OPTIONS)
    phate_embedding = np.asarray(operator.fit_transform(adata.obsm["X_pca"][:, :20]), dtype=float)
    landmark_embedding = np.asarray(operator.embedding, dtype=float)
    potential = np.asarray(operator._calculate_potential(), dtype=float)
    dissimilarities = squareform(pdist(potential, metric=operator.mds_dist))
    classic_mds_init = phate.mds.classic(
        dissimilarities,
        n_components=2,
        random_state=SEED,
    )

    phate_stem = paul15_result_stem("phate", n_landmark=N_LANDMARK)
    save_paul15_embedding(
        embedding_dir / f"{phate_stem}.npz",
        phate_embedding,
        cell_ids,
        landmark_embedding=landmark_embedding,
    )
    save_paul15_plots(
        phate_embedding,
        labels=labels,
        pseudotime=pseudotime,
        pseudotime_key=PSEUDOTIME_KEY,
        directory=figure_dir,
        stem=phate_stem,
        title="Paul15 PHATE",
    )

    init = select_init(
        INIT,
        adata=adata,
        operator=operator,
        classic_mds_init=classic_mds_init,
        phate_landmarks=landmark_embedding,
        dissimilarities=dissimilarities,
        embedding_dir=embedding_dir,
        cell_ids=cell_ids,
    )
    optimized_landmarks, objective = fit_finsler_mds(
        dissimilarities,
        metric=RandersMetric(alpha=0.0),
        optimizer=OPTIMIZER,
        n_components=2,
        init=init,
        random_state=SEED,
        print_time=True,
        **OPTIMIZER_OPTIONS[OPTIMIZER],
    )
    embedding = np.asarray(operator.graph.interpolate(optimized_landmarks), dtype=float)

    stem = paul15_result_stem(OPTIMIZER, n_landmark=N_LANDMARK)
    save_paul15_embedding(
        embedding_dir / f"{stem}.npz",
        embedding,
        cell_ids,
        objective=objective,
        landmark_embedding=optimized_landmarks,
    )
    save_paul15_plots(
        embedding,
        labels=labels,
        pseudotime=pseudotime,
        pseudotime_key=PSEUDOTIME_KEY,
        directory=figure_dir,
        stem=stem,
        title=f"Paul15 PHATE + {OPTIMIZER.replace('_', ' ')}",
    )
    plt.close("all")


def select_init(
    init_kind,
    *,
    adata,
    operator,
    classic_mds_init,
    phate_landmarks,
    dissimilarities,
    embedding_dir,
    cell_ids,
):
    init_kind = str(init_kind).lower().replace("-", "_")
    if init_kind == "classic_mds":
        print("Using PHATE's classical MDS init")
        return classic_mds_init
    if init_kind == "phate":
        print("Using PHATE's standard MDS as init")
        return phate_landmarks
    if init_kind in {"umap", "umap_2d"}:
        stem = paul15_result_stem(
            "umap",
            representation="pca",
            n_neighbors=UMAP_NEIGHBORS,
        )
        umap = ensure_paul15_umap(
            adata,
            embedding_dir / f"{stem}.npz",
            representation="pca",
            n_components=2,
            n_neighbors=UMAP_NEIGHBORS,
            seed=SEED,
        )
        init = cell_embedding_to_landmarks(umap, operator.graph.clusters, len(dissimilarities))
        init, scale = scale_embedding_to_dissimilarities(
            init,
            dissimilarities,
            random_state=SEED,
        )
        print(f"Using landmark-averaged UMAP init (scale={scale:.4g})")
        return init
    if init_kind == "umap_3d":
        raise ValueError("INIT='umap_3d' is unavailable because the PHATE optimizer is 2D.")
    return latest_paul15_embedding(
        embedding_dir,
        init_kind,
        key="landmark_embedding",
        expected_shape=dissimilarities.shape[:1] + (2,),
        cell_ids=cell_ids,
        n_landmark=N_LANDMARK,
    )


def validate_init(init_kind):
    init_kind = str(init_kind).lower().replace("-", "_")
    if init_kind in {"classic_mds", "phate", "umap", "umap_2d"}:
        return
    if init_kind == "umap_3d":
        raise ValueError("INIT='umap_3d' is unavailable because the PHATE optimizer is 2D.")
    allowed_tags = {paul15_method_tag(method) for method in OPTIMIZER_OPTIONS}
    if paul15_method_tag(init_kind) not in allowed_tags:
        raise ValueError(
            f"INIT must be a PHATE/UMAP variant or one of {tuple(OPTIMIZER_OPTIONS)}."
        )


def cell_embedding_to_landmarks(embedding, clusters, n_landmarks):
    """Average cell coordinates within PHATE's landmark assignments."""
    clusters = np.asarray(clusters, dtype=int)
    if clusters.shape != (len(embedding),):
        raise ValueError("PHATE landmark assignments do not match the cells.")
    counts = np.bincount(clusters, minlength=n_landmarks)
    if len(counts) != n_landmarks or np.any(counts == 0):
        raise ValueError("PHATE produced an empty or unexpected landmark assignment.")
    result = np.empty((n_landmarks, embedding.shape[1]), dtype=float)
    for coordinate in range(embedding.shape[1]):
        result[:, coordinate] = np.bincount(
            clusters,
            weights=embedding[:, coordinate],
            minlength=n_landmarks,
        ) / counts
    return result


def import_phate():
    try:
        import phate
        return phate
    except ModuleNotFoundError as exc:
        if exc.name != "s_gd2":
            raise RuntimeError("Install PHATE from requirements.txt to run this script.") from exc

        # Some PHATE releases import s_gd2 even when SMACOF is requested.
        stub = types.ModuleType("s_gd2")
        stub.__version__ = "999.0"
        stub.mds_direct = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("s_gd2 is unavailable; use PHATE's SMACOF solver.")
        )
        sys.modules["s_gd2"] = stub
    try:
        import phate
    except ImportError as exc:
        raise RuntimeError("Install PHATE from requirements.txt to run this script.") from exc
    return phate


if __name__ == "__main__":
    main()
