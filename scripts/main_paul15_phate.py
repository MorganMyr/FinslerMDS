"""Run PHATE on Paul15 and save a cluster-colored visualization."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import anndata as ad
import matplotlib
import types

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds  # noqa: E402


def main_paul15_phate():
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15" / "phate"
    dir_res.mkdir(parents=True, exist_ok=True)

    group_key = "paul15_clusters"
    excluded_clusters = ("19Lymph", "11DC")
    n_pcs = 20
    phate_params = {
        "n_components": 2,
        "knn": 15,
        "decay": 40,
        "n_landmark": 500,
        "t": "auto",
        "gamma": 1,
        "mds": "metric",
        "mds_solver": "smacof",
        "n_jobs": -1,
        "random_state": seed,
        "verbose": 1,
    }
    path_frozen_params = {
        "graph_neighbors": 30,
        "outer_iter": 50,
        "inner_iter": 10,
        "n_local_pairs": 30,
        "local_pair_mode": "direct",
        "n_landmark": 150,
        "targets_per_landmark": 300,
        "local_weight": 1.0,
        "local_global_reweighting": "count",
        "random_state": seed,
        "mask_random_state": seed,
        "target_random_state": seed,
        "verbose": 1,
        "log_frequency": 1,
        "device": "auto",
    }
    # "classic" uses PHATE's normal MDS init. "previous" continues from the
    # last saved path-frozen landmark embedding when available.
    path_frozen_init = "classic"

    suffix = "_no19lymph_no11dc" if excluded_clusters else ""
    embedding_npz = dir_res / f"paul15_phate{suffix}_embedding.npz"
    figure_phate_pdf = dir_res / f"paul15_phate{suffix}_clusters.pdf"
    figure_path_frozen_pdf = dir_res / f"paul15_phate_path_frozen_alpha0{suffix}_clusters.pdf"

    np.random.seed(seed)
    sc.settings.autoshow = False
    sc.set_figure_params(dpi=110, frameon=False, figsize=(5, 5), facecolor="white")

    adata = load_preprocessed_paul15(
        group_key=group_key,
        excluded_clusters=excluded_clusters,
        n_pcs=n_pcs,
        seed=seed,
    )

    phate_embedding, path_frozen_embedding, payload = compute_phate_and_path_frozen_embeddings(
        adata.obsm["X_pca"][:, :n_pcs],
        phate_params,
        path_frozen_params,
        path_frozen_init=path_frozen_init,
        previous_embedding_path=embedding_npz,
    )
    np.savez(
        embedding_npz,
        phate_embedding=phate_embedding,
        path_frozen_alpha0_embedding=path_frozen_embedding,
        cell_ids=np.asarray(adata.obs_names.astype(str)),
        labels=np.asarray(adata.obs[group_key].astype(str), dtype=str),
        excluded_clusters=np.asarray(excluded_clusters, dtype=str),
        n_pcs=np.asarray(n_pcs),
        path_frozen_init=np.asarray(path_frozen_init),
        **payload,
        **{key: np.asarray(value) for key, value in phate_params.items()},
        **{f"path_frozen_{key}": np.asarray(value) for key, value in path_frozen_params.items()},
    )
    print(f"Saved PHATE embedding: {embedding_npz}")

    plot_cluster_embedding(
        phate_embedding,
        labels=adata.obs[group_key],
        colors=adata.uns.get(f"{group_key}_colors"),
        out_pdf=figure_phate_pdf,
        title="Paul15 PHATE clusters",
        basis="phate",
    )
    plot_cluster_embedding(
        path_frozen_embedding,
        labels=adata.obs[group_key],
        colors=adata.uns.get(f"{group_key}_colors"),
        out_pdf=figure_path_frozen_pdf,
        title="Paul15 PHATE preprocessing + path-frozen alpha=0",
        basis="mds",
    )
    print(f"Saved PHATE cluster plots: {figure_phate_pdf}, {figure_path_frozen_pdf}")


def load_preprocessed_paul15(*, group_key, excluded_clusters, n_pcs, seed):
    print("Loading Scanpy Paul15 mouse hematopoiesis dataset")
    adata = sc.datasets.paul15()
    print(f"Raw Paul15 shape: {adata.n_obs} cells x {adata.n_vars} genes")

    adata.X = adata.X.astype("float64")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Use sc.pp.highly_variable_genes instead",
            category=FutureWarning,
            module="scanpy.preprocessing._recipes",
        )
        sc.pp.recipe_zheng17(adata)

    if excluded_clusters:
        labels = adata.obs[group_key].astype(str)
        keep = ~labels.isin(excluded_clusters).to_numpy()
        adata = adata[keep].copy()
        print(
            "Excluded clusters "
            f"{', '.join(excluded_clusters)}; kept {adata.n_obs} cells."
        )

    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return adata


def compute_phate_and_path_frozen_embeddings(
    X,
    phate_params,
    path_frozen_params,
    *,
    path_frozen_init,
    previous_embedding_path,
):
    # PHATE imports s_gd2 unconditionally, even when mds_solver="smacof".
    # On Python 3.13/Windows, s-gd2 often has no installable wheel, so this
    # stub lets us use PHATE's diffusion/potential pipeline without its SGD MDS.
    if phate_params.get("mds_solver") != "sgd" and "s_gd2" not in sys.modules:
        stub = types.ModuleType("s_gd2")
        stub.__version__ = "999.0"

        def _missing_sgd2(*args, **kwargs):
            raise RuntimeError("s_gd2 is unavailable; use mds_solver='smacof'.")

        stub.mds_direct = _missing_sgd2
        sys.modules["s_gd2"] = stub

    try:
        import phate
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'phate' is required. Install it with "
            "`pip install phate` or from requirements.txt."
        ) from exc

    operator = phate.PHATE(**phate_params)
    phate_embedding = np.asarray(operator.fit_transform(X), dtype=float)

    # PHATE itself runs MDS on this matrix. With landmark PHATE this is the
    # landmark potential; cell embeddings are obtained by interpolation after MDS.
    diff_potential = np.asarray(operator._calculate_potential(), dtype=float)
    potential_distances = squareform(pdist(diff_potential, operator.mds_dist))
    classic_init = phate.mds.classic(
        potential_distances,
        n_components=operator.n_components,
        random_state=operator.random_state,
    )
    init, init_kind = select_path_frozen_init(
        path_frozen_init,
        classic_init=classic_init,
        previous_embedding_path=previous_embedding_path,
    )
    landmark_embedding, stress = fit_finsler_mds(
        potential_distances,
        metric=RandersMetric(alpha=0.0),
        optimizer="path_frozen",
        n_components=operator.n_components,
        init=init,
        print_time=True,
        **path_frozen_params,
    )

    if hasattr(operator.graph, "interpolate"):
        path_frozen_embedding = np.asarray(
            operator.graph.interpolate(landmark_embedding),
            dtype=float,
        )
    else:
        path_frozen_embedding = np.asarray(landmark_embedding, dtype=float)

    payload = {
        "phate_potential_distances": potential_distances,
        "phate_classical_mds_init": classic_init,
        "path_frozen_alpha0_landmark_embedding": landmark_embedding,
        "path_frozen_init_used": np.asarray(init_kind),
        "path_frozen_alpha0_stress": np.asarray(stress),
    }
    return phate_embedding, path_frozen_embedding, payload


def select_path_frozen_init(path_frozen_init, *, classic_init, previous_embedding_path):
    if path_frozen_init == "classic":
        print("Using PHATE classical-MDS init for path-frozen.")
        return classic_init, "classic"
    if path_frozen_init != "previous":
        raise ValueError("path_frozen_init must be 'classic' or 'previous'.")

    previous = load_previous_path_frozen_landmark_embedding(previous_embedding_path)
    if previous is None:
        print("No previous path-frozen landmark init found; falling back to PHATE classical-MDS init.")
        return classic_init, "classic_fallback"
    if previous.shape != classic_init.shape:
        print(
            "Previous path-frozen landmark init has incompatible shape "
            f"{previous.shape}; expected {classic_init.shape}. Falling back to PHATE classical-MDS init."
        )
        return classic_init, "classic_fallback"

    print(f"Using previous path-frozen landmark init from: {previous_embedding_path}")
    return previous, "previous"


def load_previous_path_frozen_landmark_embedding(path):
    if not path.exists():
        return None
    try:
        with np.load(path) as cache:
            if "path_frozen_alpha0_landmark_embedding" not in cache.files:
                return None
            return np.asarray(cache["path_frozen_alpha0_landmark_embedding"], dtype=float)
    except Exception as exc:
        print(f"Could not load previous path-frozen init from {path}: {exc}")
        return None


def plot_cluster_embedding(embedding, *, labels, colors, out_pdf, title, basis):
    obs = pd.DataFrame(index=np.arange(len(embedding)).astype(str))
    if hasattr(labels, "cat"):
        categories = list(labels.cat.categories)
    else:
        categories = sorted(pd.unique(labels.astype(str)))
    obs["clusters"] = pd.Categorical(labels.astype(str).to_numpy(), categories=categories)

    plot_adata = ad.AnnData(X=np.zeros((len(embedding), 1)), obs=obs)
    plot_adata.obsm[f"X_{basis}"] = np.asarray(embedding, dtype=float)
    if colors is not None:
        plot_adata.uns["clusters_colors"] = list(colors)

    sc.pl.embedding(
        plot_adata,
        basis=basis,
        color="clusters",
        legend_loc="on data",
        legend_fontsize=8,
        frameon=False,
        size=14,
        title=title,
        show=False,
    )
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main_paul15_phate()
