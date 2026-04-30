import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds
from finsler_mds import utils
from finsler_mds.utils import plot_categorical_embedding


def main_pancreas():
    # Hyperparameters
    min_shared_counts = 20
    n_top_genes = 2000
    n_pcs = 30
    n_neighbors = 30
    velocity_mode = "stochastic"
    velocity_alpha = 1.0
    velocity_neighbors = 30
    run_randers_smacof = False
    randers_alpha_embedding = 0.4
    max_iter_smacof = 300
    seed = 42
    dir_res = "res/pancreas"

    os.makedirs(dir_res, exist_ok=True)
    np.random.seed(seed)

    # Keep these imports local: the rest of the project should remain usable
    # even when scvelo/scanpy are not installed.
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")

    print("Loading scVelo pancreas dataset")
    adata = scv.datasets.pancreas()
    print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

    print("Preprocessing")
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
        subset=True,
    )
    scv.pp.moments(
        adata,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
    )
    print(f"Preprocessed pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

    print("Computing RNA velocity")
    scv.tl.velocity(adata, mode=velocity_mode)

    print("Computing UMAP")
    sc.tl.umap(adata, random_state=seed)
    labels = adata.obs["clusters"] if "clusters" in adata.obs else None
    plot_categorical_embedding(
        adata.obsm["X_umap"],
        labels=labels,
        title="scVelo pancreas UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=os.path.join(dir_res, "pancreas_umap.pdf"),
    )

    print("Building directed velocity dissimilarities")
    X_pca = adata.obsm["X_pca"][:, :n_pcs]
    velocity_pca = project_velocity_to_pca(adata, n_pcs)
    dists_velocity, preds_velocity, velocity_graph, velocity_pca_smooth = utils.compute_velocity_dist_matrix(
        X_pca,
        velocity_pca,
        n_neighbors=n_neighbors,
        alpha=velocity_alpha,
        velocity_neighbors=velocity_neighbors,
        average_velocity=True,
        symmetrize_support=True,
    )
    adata.uns["finsler_mds_velocity_graph"] = velocity_graph
    adata.uns["finsler_mds_velocity_predecessors"] = preds_velocity
    adata.obsm["velocity_pca_smoothed"] = velocity_pca_smooth
    print(
        "Directed velocity distances: "
        f"{dists_velocity.shape[0]} x {dists_velocity.shape[1]}, "
        f"finite={np.isfinite(dists_velocity).mean():.3f}"
    )

    init_3d = np.column_stack([
        adata.obsm["X_umap"][:, 0],
        adata.obsm["X_umap"][:, 1],
        np.zeros(adata.n_obs),
    ])
    adata.obsm["X_umap_3d_init"] = init_3d

    if run_randers_smacof:
        print("Running Randers SMACOF warm start")
        proj_randers, stress_randers = fit_finsler_mds(
            dists_velocity,
            metric=RandersMetric(alpha=randers_alpha_embedding),
            optimizer="smacof_randers",
            init=init_3d,
            n_components=3,
            n_init=1,
            n_jobs=1,
            max_iter=max_iter_smacof,
            pseudo_inv_solver="gmres",
            project_on_V=True,
            check_monotony=False,
            print_time=True,
        )
        adata.obsm["X_finsler_randers"] = proj_randers
        print(f"Randers SMACOF stress: {stress_randers}")

    plt.show()
    return adata, dists_velocity


def project_velocity_to_pca(adata, n_pcs):
    """Project scVelo gene-space velocities onto Scanpy/scVelo PCA axes."""
    if "velocity" not in adata.layers:
        raise ValueError("adata.layers['velocity'] is missing. Run scv.tl.velocity first.")
    if "PCs" not in adata.varm:
        raise ValueError("adata.varm['PCs'] is missing. Run PCA/moments before projecting velocity.")

    velocity = adata.layers["velocity"]
    if sparse.issparse(velocity):
        velocity = velocity.toarray()
    velocity = np.asarray(velocity, dtype=float)
    velocity = np.nan_to_num(velocity, copy=False)

    pcs = np.asarray(adata.varm["PCs"][:, :n_pcs], dtype=float)
    velocity_pca = velocity @ pcs
    return np.asarray(velocity_pca, dtype=float)


if __name__ == "__main__":
    main_pancreas()
