"""Plot Paul15 PAGA UMAP clusters on the original full-UMAP coordinates.

This keeps the embedding used by ``umap_paul15_clusters.pdf`` while applying the
cleaner no-19Lymph/no-11DC cluster-plot style.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "res" / "paul15" / "raw"
FIG_DIR = SCRIPT_DIR / "res" / "paul15" / "baseline" / "figures"


def main():
    full_umap = pd.read_csv(RAW_DIR / "paul15_biological_paga_full_umap.csv")
    full_umap["paul15_clusters"] = full_umap["paul15_clusters"].astype(str)
    keep = ~full_umap["paul15_clusters"].isin(["19Lymph", "11DC"])
    full_umap = full_umap.loc[keep].copy()

    reference = sc.read_h5ad(RAW_DIR / "paul15_biological_paga_no19lymph_no11dc.h5ad")
    categories = list(reference.obs["paul15_clusters"].cat.categories)
    colors = list(reference.uns["paul15_clusters_colors"])

    obs = pd.DataFrame(index=full_umap["cell_id"].astype(str))
    obs["paul15_clusters"] = pd.Categorical(
        full_umap["paul15_clusters"].to_numpy(),
        categories=categories,
        ordered=True,
    )
    adata = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    adata.obsm["X_umap"] = full_umap[["umap1", "umap2"]].to_numpy(dtype=float)
    adata.uns["paul15_clusters_colors"] = colors

    out_pdf = FIG_DIR / "umap_paga_full_embedding_no19lymph_no11dc_clusters.pdf"
    sc.pl.umap(
        adata,
        color="paul15_clusters",
        legend_loc="on data",
        legend_fontsize=8,
        frameon=False,
        size=14,
        title="Paul15 PAGA UMAP without 19Lymph/11DC",
        show=False,
    )
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close("all")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
