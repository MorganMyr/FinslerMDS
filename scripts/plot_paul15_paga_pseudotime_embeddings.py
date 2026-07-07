"""Plot Paul15 embeddings colored by PAGA lineage pseudotime."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.utils import plot_continuous_embedding


UMAP_REMOVED_CLUSTERS = ("19Lymph",)
MDS_REMOVED_CLUSTERS = ("19Lymph", "11DC")
PSEUDOTIME_KEY = "dpt_lineage_pseudotime"
POINT_SIZE = 7
CMAP = "viridis"
SEED = 42

LINEAGES = {
    "erythrocyte": ["10GMP", "7MEP", "8Mk", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
    "monocyte": ["10GMP", "9GMP", "14Mo", "15Mo"],
}

PSEUDOTIME = {
    "root_cluster": "10GMP",
    "n_dcs": 10,
    "subset_neighbors": 15,
    "lineages": LINEAGES,
}

UMAP = {
    "init_pos": "paga",
    "min_dist": 0.5,
    "spread": 1.0,
    "maxiter": 1000,
    "negative_sample_rate": 10,
}


def main():
    script_dir = Path(__file__).resolve().parent
    res_dir = script_dir / "res" / "paul15"
    out_dir = res_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_npz(res_dir / "baseline" / "embeddings" / "paul15_embeddings_metadata.npz")
    full_ids = baseline["cell_ids"].astype(str)
    full_labels = baseline["labels"].astype(str)

    original_pt = pd.read_csv(
        res_dir / "raw" / "paul15_biological_paga_original_combined_lineage_pseudotime.csv"
    )
    filtered_mds_pt = pd.read_csv(
        res_dir / "raw" / "paul15_biological_paga_no19lymph_no11dc_combined_lineage_pseudotime.csv"
    )
    no19_pt, no19_umap = load_or_build_no19lymph_paga(
        res_dir / "raw",
        full_ids=full_ids,
        full_labels=full_labels,
    )

    keep_no19 = ~np.isin(full_labels, UMAP_REMOVED_CLUSTERS)
    no19_ids = full_ids[keep_no19]
    keep_mds_filtered = ~np.isin(full_labels, MDS_REMOVED_CLUSTERS)
    filtered_mds_ids = full_ids[keep_mds_filtered]
    require_order(original_pt["cell_id"].astype(str).to_numpy(), full_ids, "full PAGA pseudotime")
    require_order(no19_pt["cell_id"].astype(str).to_numpy(), no19_ids, "no19Lymph PAGA pseudotime")
    require_order(no19_umap["cell_id"].astype(str).to_numpy(), no19_ids, "no19Lymph PAGA UMAP")
    require_order(
        filtered_mds_pt["cell_id"].astype(str).to_numpy(),
        filtered_mds_ids,
        "no19Lymph/no11DC PAGA pseudotime",
    )

    mds = load_npz(res_dir / "monocle3" / "experiments_MDS" / "classic_mds_alpha0_embedding_raw.npz")[
        "embedding"
    ]
    mds_geo = load_npz(res_dir / "monocle3" / "experiments" / "best_deep_refined_embedding_raw.npz")[
        "embedding"
    ]
    if len(mds) != len(filtered_mds_ids) or len(mds_geo) != len(filtered_mds_ids):
        raise ValueError(
            "Filtered Paul15 pseudotime and MDS embeddings have inconsistent lengths: "
            f"pseudotime={len(filtered_mds_ids)}, MDS={len(mds)}, MDS-geo={len(mds_geo)}"
        )

    stale_umap = out_dir / "umap_no19lymph_no11dc_paga_lineage_pseudotime.pdf"
    if stale_umap.exists():
        stale_umap.unlink()

    plots = [
        (
            baseline["umap"][keep_no19],
            original_pt[PSEUDOTIME_KEY].to_numpy(dtype=float)[keep_no19],
            "UMAP (without 19Lymph, full fit), PAGA lineage pseudotime",
            out_dir / "umap_full_paga_lineage_pseudotime.pdf",
        ),
        (
            no19_umap[["umap1", "umap2"]].to_numpy(dtype=float),
            no19_pt[PSEUDOTIME_KEY].to_numpy(dtype=float),
            "UMAP (without 19Lymph), PAGA lineage pseudotime",
            out_dir / "umap_no19lymph_paga_lineage_pseudotime.pdf",
        ),
        (
            mds,
            filtered_mds_pt[PSEUDOTIME_KEY].to_numpy(dtype=float),
            "MDS, PAGA lineage pseudotime",
            out_dir / "mds_paga_lineage_pseudotime.pdf",
        ),
        (
            mds_geo,
            filtered_mds_pt[PSEUDOTIME_KEY].to_numpy(dtype=float),
            "Geodesic MDS, PAGA lineage pseudotime",
            out_dir / "mds_geo_paga_lineage_pseudotime.pdf",
        ),
    ]

    for embedding, values, title, path in plots:
        plot_pseudotime(embedding, values, title=title, path=path)
        print(f"Saved {path}")


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def require_order(actual, expected, label):
    if len(actual) != len(expected) or not np.array_equal(actual, expected):
        raise ValueError(f"Cell order mismatch for {label}.")


def load_or_build_no19lymph_paga(raw_dir, *, full_ids, full_labels):
    pt_path = raw_dir / "paul15_biological_paga_no19lymph_combined_lineage_pseudotime.csv"
    umap_path = raw_dir / "paul15_biological_paga_no19lymph_pseudotime.csv"
    if pt_path.exists() and umap_path.exists():
        return pd.read_csv(pt_path), pd.read_csv(umap_path)

    print("Building PAGA UMAP/pseudotime without 19Lymph, keeping 11DC.")
    import scanpy as sc

    from finsler_mds.utils.paul15 import (
        COMBINED_LINEAGE_DPT_KEY,
        compute_global_and_lineage_pseudotimes,
        ensure_combined_lineage_pseudotime,
        lineage_pseudotime_keys,
    )

    adata = sc.read_h5ad(raw_dir / "paul15_biological_paga.h5ad")
    keep = ~np.isin(np.asarray(adata.obs["paul15_clusters"].astype(str)), UMAP_REMOVED_CLUSTERS)
    adata = adata[keep].copy()
    require_order(np.asarray(adata.obs_names.astype(str)), full_ids[~np.isin(full_labels, UMAP_REMOVED_CLUSTERS)], "AnnData no19Lymph")

    sc.settings.autoshow = False
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_diffmap", random_state=SEED)
    sc.tl.paga(adata, groups="paul15_clusters")
    sc.pl.paga(adata, threshold=0.03, show=False)
    plt.close("all")
    sc.tl.umap(
        adata,
        init_pos=UMAP["init_pos"],
        min_dist=UMAP["min_dist"],
        spread=UMAP["spread"],
        maxiter=UMAP["maxiter"],
        negative_sample_rate=UMAP["negative_sample_rate"],
        random_state=SEED,
    )

    lineage_keys = lineage_pseudotime_keys(PSEUDOTIME)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        compute_global_and_lineage_pseudotimes(
            adata,
            root_cluster=PSEUDOTIME["root_cluster"],
            n_dcs=PSEUDOTIME["n_dcs"],
            subset_neighbors=PSEUDOTIME["subset_neighbors"],
            lineages=PSEUDOTIME["lineages"],
            cluster_key="paul15_clusters",
        )
        ensure_combined_lineage_pseudotime(
            adata,
            lineage_keys=lineage_keys,
            key=COMBINED_LINEAGE_DPT_KEY,
        )

    pt = pd.DataFrame(
        {
            "cell_id": np.asarray(adata.obs_names.astype(str)),
            "paul15_clusters": np.asarray(adata.obs["paul15_clusters"].astype(str)),
            **{key: np.asarray(adata.obs[key], dtype=float) for key in lineage_keys},
            PSEUDOTIME_KEY: np.asarray(adata.obs[PSEUDOTIME_KEY], dtype=float),
        }
    )
    umap = pd.DataFrame(
        {
            "cell_id": np.asarray(adata.obs_names.astype(str)),
            "paul15_clusters": np.asarray(adata.obs["paul15_clusters"].astype(str)),
            "umap1": np.asarray(adata.obsm["X_umap"][:, 0], dtype=float),
            "umap2": np.asarray(adata.obsm["X_umap"][:, 1], dtype=float),
            "dpt_pseudotime_finite": np.asarray(adata.obs["dpt_pseudotime_finite"], dtype=float),
        }
    )
    pt.to_csv(pt_path, index=False)
    umap.to_csv(umap_path, index=False)
    return pt, umap


def plot_pseudotime(embedding, values, *, title, path):
    embedding = np.asarray(embedding, dtype=float)
    values = np.asarray(values, dtype=float)
    if embedding.shape != (len(values), 2):
        raise ValueError(f"Expected a 2D embedding aligned with values, got {embedding.shape}.")

    plot_continuous_embedding(
        embedding,
        values,
        title=title,
        save_path=path,
        s=POINT_SIZE,
        cmap=CMAP,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
