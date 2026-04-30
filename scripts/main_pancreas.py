import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds
from finsler_mds.utils import plot_categorical_embedding


def main_pancreas():
    # Hyperparameters
    min_shared_counts = 20
    n_top_genes = 2000
    n_pcs = 30
    n_neighbors = 30
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

    # Sanity check for future Finsler-MDS calls: these imports should resolve.
    _ = fit_finsler_mds, RandersMetric

    plt.show()
    return adata


if __name__ == "__main__":
    main_pancreas()
