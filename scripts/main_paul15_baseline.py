from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


COMBINED_LINEAGE_DPT_KEY = "dpt_lineage_pseudotime"


def main_paul15_baseline():
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15"
    dir_raw = dir_res / "raw"
    dir_fig = dir_res / "figures"
    dir_embeddings = dir_res / "embeddings"

    preprocessing = {
        "n_pcs": 20,
        "initial_neighbors": 4,
        "trajectory_neighbors": 10,
        # The current Scanpy Paul15 tutorial recommends float64 for more
        # reproducible trajectories across platforms.
        "use_float64": True,
    }
    cell_graph = {
        # DrivAER rebuilds the single-cell graph with a much denser diffusion
        # neighborhood before ForceAtlas2; this is what turns the PAGA result
        # into the smooth left/right trajectory map in their Figure 2.
        "neighbors": 100,
        "use_rep": "X_diffmap",
    }
    group_key = "paul15_clusters"
    paga = {
        "groups": group_key,
        "threshold": 0.03,
    }
    draw_graph = {
        # DrivAER used ForceAtlas2 (`fa`) with PAGA initialization. With the
        # modern fa2-modified/Scanpy stack used here, the Paul15 graph can
        # occasionally blow up numerically because of a tiny disconnected
        # component. We try FA first and fall back to FR when that happens.
        "preferred_layout": "fa",
        "fallback_layout": "fr",
        "init_pos": "paga",
        "max_abs_coordinate": 1e4,
    }
    umap = {
        "init_pos": "paga",
        "min_dist": 0.5,
        "spread": 1.0,
        "maxiter": 1000,
        "negative_sample_rate": 10,
    }
    pseudotime = {
        # Paul15's built-in biological annotations are more stable than
        # Louvain IDs across Scanpy/louvain versions. 10GMP sits at the
        # common granulocyte-macrophage/progenitor side of the map and gives
        # clean left/right lineage pseudotimes in this environment.
        "root_cluster": "10GMP",
        "n_dcs": 10,
        "subset_neighbors": 15,
        "lineages": {
            "erythrocyte": ["10GMP", "7MEP", "8Mk", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
            "monocyte": ["10GMP", "9GMP", "14Mo", "15Mo"],
        },
    }
    cache = {
        "use_cache": True,
        "adata_path": dir_raw / "paul15_biological_paga.h5ad",
        "embedding_path": dir_raw / "paul15_biological_paga_embedding.npz",
        "embedding_dir": dir_embeddings,
    }

    dir_raw.mkdir(parents=True, exist_ok=True)
    dir_fig.mkdir(parents=True, exist_ok=True)
    dir_embeddings.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    sc.settings.verbosity = 2
    sc.settings.autoshow = False
    sc.set_figure_params(dpi=110, frameon=False, figsize=(5, 5), facecolor="white")

    if cache["use_cache"] and cache["adata_path"].exists():
        print(f"Loading cached Paul15 Scanpy/PAGA baseline: {cache['adata_path']}")
        adata = sc.read_h5ad(cache["adata_path"])
    else:
        _require_trajectory_dependencies()
        adata = build_scanpy_paul15_baseline(
            seed=seed,
            preprocessing=preprocessing,
            group_key=group_key,
            paga=paga,
            cell_graph=cell_graph,
            draw_graph=draw_graph,
            pseudotime=pseudotime,
        )

    lineage_keys = lineage_pseudotime_keys(pseudotime)
    ensure_combined_lineage_pseudotime(adata, lineage_keys=lineage_keys, key=COMBINED_LINEAGE_DPT_KEY)

    draw_layout = ensure_stable_draw_graph(adata, seed=seed, draw_graph=draw_graph)
    ensure_paga_umap(adata, seed=seed, umap=umap)
    adata.write_h5ad(cache["adata_path"])
    print(f"Saved Paul15 baseline AnnData: {cache['adata_path']}")

    saved_embeddings = save_paul15_baseline_arrays(
        adata,
        embedding_key=_draw_graph_key(draw_layout),
        group_key=group_key,
        pseudotime_keys=["dpt_pseudotime_finite", COMBINED_LINEAGE_DPT_KEY, *lineage_keys],
        save_path=cache["embedding_path"],
        embedding_dir=cache["embedding_dir"],
    )
    print(f"Saved baseline embedding arrays: {cache['embedding_path']}")

    clear_old_figures(dir_fig)
    saved_figures = save_paul15_baseline_figures(
        adata,
        draw_layout=draw_layout,
        group_key=group_key,
        paga_threshold=paga["threshold"],
        lineage_keys=lineage_keys,
        combined_lineage_key=COMBINED_LINEAGE_DPT_KEY,
        dir_fig=dir_fig,
    )
    validate_saved_outputs([cache["adata_path"], cache["embedding_path"], *saved_embeddings, *saved_figures])
    print(f"Saved figures in: {dir_fig}")


def build_scanpy_paul15_baseline(*, seed, preprocessing, group_key, paga, cell_graph, draw_graph, pseudotime):
    print("Loading Scanpy Paul15 mouse hematopoiesis dataset")
    adata = sc.datasets.paul15()
    print(f"Raw Paul15 shape: {adata.n_obs} cells x {adata.n_vars} genes")

    if preprocessing["use_float64"]:
        adata.X = adata.X.astype("float64")

    print("Preprocessing with Scanpy recipe_zheng17")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Use sc.pp.highly_variable_genes instead",
            category=FutureWarning,
            module="scanpy.preprocessing._recipes",
        )
        sc.pp.recipe_zheng17(adata)

    print("Computing PCA and diffusion-map neighborhood")
    sc.tl.pca(adata, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(
        adata,
        n_neighbors=preprocessing["initial_neighbors"],
        n_pcs=preprocessing["n_pcs"],
        random_state=seed,
    )
    sc.tl.diffmap(adata)
    sc.pp.neighbors(
        adata,
        n_neighbors=preprocessing["trajectory_neighbors"],
        use_rep="X_diffmap",
        random_state=seed,
    )

    print(f"Computing PAGA groups={paga['groups']!r}")
    sc.tl.paga(adata, groups=paga["groups"])
    sc.pl.paga(adata, threshold=paga["threshold"], show=False)
    plt.close("all")

    print(f"Recomputing dense cell graph for single-cell trajectory layout (k={cell_graph['neighbors']})")
    sc.pp.neighbors(
        adata,
        n_neighbors=cell_graph["neighbors"],
        use_rep=cell_graph["use_rep"],
        random_state=seed,
    )
    ensure_stable_draw_graph(adata, seed=seed, draw_graph=draw_graph, force=True)

    compute_global_and_lineage_pseudotimes(
        adata,
        root_cluster=pseudotime["root_cluster"],
        n_dcs=pseudotime["n_dcs"],
        subset_neighbors=pseudotime["subset_neighbors"],
        lineages=pseudotime["lineages"],
        cluster_key=group_key,
    )
    return adata


def compute_global_and_lineage_pseudotimes(
    adata,
    *,
    root_cluster,
    n_dcs,
    subset_neighbors,
    lineages,
    cluster_key,
):
    root_idx = np.flatnonzero(np.asarray(adata.obs[cluster_key].astype(str)) == root_cluster)
    if len(root_idx) == 0:
        raise ValueError(f"Could not find DPT root cluster {root_cluster!r}.")

    adata.uns["iroot"] = int(root_idx[0])
    print(f"Global DPT root: {adata.obs_names[root_idx[0]]} ({root_cluster})")
    sc.tl.dpt(adata, n_dcs=n_dcs)
    adata.obs["dpt_pseudotime_finite"] = finite_rescaled(adata.obs["dpt_pseudotime"])

    for lineage, clusters in lineages.items():
        key = lineage_pseudotime_key(lineage)
        adata.obs[key] = lineage_dpt(
            adata,
            clusters=clusters,
            root_cluster=root_cluster,
            cluster_key=cluster_key,
            subset_neighbors=subset_neighbors,
            n_dcs=n_dcs,
        )
        finite = np.isfinite(np.asarray(adata.obs[key], dtype=float))
        print(f"{lineage} DPT cells: {int(finite.sum())} / {adata.n_obs}")


def ensure_combined_lineage_pseudotime(adata, *, lineage_keys, key):
    missing = [lineage_key for lineage_key in lineage_keys if lineage_key not in adata.obs]
    if missing:
        raise KeyError(f"Cannot build combined lineage pseudotime, missing columns: {missing}")

    values = np.vstack([np.asarray(adata.obs[lineage_key], dtype=float) for lineage_key in lineage_keys])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        combined = np.nanmean(values, axis=0)
    adata.obs[key] = combined


def lineage_dpt(adata, *, clusters, root_cluster, cluster_key, subset_neighbors, n_dcs):
    labels = np.asarray(adata.obs[cluster_key].astype(str))
    clusters = [str(cluster) for cluster in clusters]
    mask = np.isin(labels, clusters)
    if not np.any(mask):
        raise ValueError(f"No cells selected for lineage clusters={clusters}.")
    if not np.any(labels[mask] == str(root_cluster)):
        raise ValueError(f"Lineage clusters={clusters} do not contain root cluster {root_cluster!r}.")

    sub = adata[mask].copy()
    n_neighbors = min(int(subset_neighbors), max(1, sub.n_obs - 1))
    sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep="X_diffmap", random_state=0)
    sc.tl.diffmap(sub)
    sub_labels = np.asarray(sub.obs[cluster_key].astype(str))
    sub.uns["iroot"] = int(np.flatnonzero(sub_labels == str(root_cluster))[0])
    sc.tl.dpt(sub, n_dcs=n_dcs)

    out = np.full(adata.n_obs, np.nan, dtype=float)
    out[np.flatnonzero(mask)] = finite_rescaled(sub.obs["dpt_pseudotime"])
    return out


def finite_rescaled(values):
    values = np.asarray(values, dtype=float)
    out = values.copy()
    out[~np.isfinite(out)] = np.nan
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    lo = float(np.nanmin(out[finite]))
    hi = float(np.nanmax(out[finite]))
    if hi > lo:
        out[finite] = (out[finite] - lo) / (hi - lo)
    else:
        out[finite] = 0.0
    return out


def ensure_paga_umap(adata, *, seed, umap):
    if "X_umap" in adata.obsm:
        return

    print("Computing UMAP initialized from PAGA")
    if "paga" not in adata.uns or "pos" not in adata.uns["paga"]:
        sc.pl.paga(adata, show=False)
        plt.close("all")
    sc.tl.umap(
        adata,
        init_pos=umap["init_pos"],
        min_dist=umap["min_dist"],
        spread=umap["spread"],
        maxiter=umap["maxiter"],
        negative_sample_rate=umap["negative_sample_rate"],
        random_state=seed,
    )


def ensure_stable_draw_graph(adata, *, seed, draw_graph, force=False):
    preferred = draw_graph["preferred_layout"]
    fallback = draw_graph["fallback_layout"]

    if not force and _layout_is_available_and_stable(
        adata,
        layout=preferred,
        max_abs_coordinate=draw_graph["max_abs_coordinate"],
    ):
        adata.uns["paul15_draw_graph_layout"] = preferred
        return preferred

    if force or _draw_graph_key(preferred) not in adata.obsm:
        print(f"Computing {preferred} graph initialized from PAGA")
        sc.tl.draw_graph(
            adata,
            layout=preferred,
            init_pos=draw_graph["init_pos"],
            random_state=seed,
        )

    if _layout_is_available_and_stable(
        adata,
        layout=preferred,
        max_abs_coordinate=draw_graph["max_abs_coordinate"],
    ):
        adata.uns["paul15_draw_graph_layout"] = preferred
        return preferred

    if not force and _layout_is_available_and_stable(
        adata,
        layout=fallback,
        max_abs_coordinate=draw_graph["max_abs_coordinate"],
    ):
        adata.uns["paul15_draw_graph_layout"] = fallback
        return fallback

    print(
        f"Paul15 {preferred} layout is numerically unstable in this environment; "
        f"falling back to {fallback}."
    )
    sc.tl.draw_graph(
        adata,
        layout=fallback,
        init_pos=draw_graph["init_pos"],
        random_state=seed,
    )
    if not _layout_is_available_and_stable(
        adata,
        layout=fallback,
        max_abs_coordinate=draw_graph["max_abs_coordinate"],
    ):
        raise RuntimeError(f"Could not compute a stable Paul15 draw_graph layout ({preferred} or {fallback}).")

    adata.uns["paul15_draw_graph_layout"] = fallback
    return fallback


def _layout_is_available_and_stable(adata, *, layout, max_abs_coordinate):
    key = _draw_graph_key(layout)
    if key not in adata.obsm:
        return False

    coords = np.asarray(adata.obsm[key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return False
    if not np.all(np.isfinite(coords)):
        return False

    max_abs = float(np.max(np.abs(coords))) if coords.size else 0.0
    max_span = float(np.max(np.ptp(coords, axis=0))) if coords.size else 0.0
    return max_abs <= max_abs_coordinate and max_span <= max_abs_coordinate


def save_paul15_baseline_figures(
    adata,
    *,
    draw_layout,
    group_key,
    paga_threshold,
    lineage_keys,
    combined_lineage_key,
    dir_fig,
):
    saved = []

    sc.pl.paga_compare(
        adata,
        threshold=paga_threshold,
        title="",
        right_margin=0.2,
        size=10,
        edge_width_scale=0.5,
        legend_fontsize=10,
        fontsize=10,
        frameon=False,
        edges=True,
        show=False,
    )
    saved.append(_save_current_figure(dir_fig / "paga_compare_paul15_clusters.pdf"))

    sc.pl.draw_graph(
        adata,
        layout=draw_layout,
        color=group_key,
        legend_loc="on data",
        legend_fontsize=8,
        title="Paul15 clusters / PAGA cell graph",
        show=False,
    )
    saved.append(_save_current_figure(dir_fig / "cell_graph_paul15_clusters.pdf"))

    sc.pl.draw_graph(
        adata,
        layout=draw_layout,
        color="dpt_pseudotime_finite",
        color_map="viridis",
        na_color="lightgray",
        title="All-cell DPT from 10GMP",
        show=False,
    )
    saved.append(_save_current_figure(dir_fig / "cell_graph_global_dpt.pdf"))

    sc.pl.draw_graph(
        adata,
        layout=draw_layout,
        color=combined_lineage_key,
        color_map="viridis",
        na_color="lightgray",
        title="Combined lineage DPT pseudotime",
        show=False,
    )
    saved.append(_save_current_figure(dir_fig / "cell_graph_lineage_pseudotime.pdf"))

    for key in lineage_keys:
        sc.pl.draw_graph(
            adata,
            layout=draw_layout,
            color=key,
            color_map="viridis",
            na_color="lightgray",
            title=pretty_pseudotime_title(key),
            show=False,
        )
        saved.append(_save_current_figure(dir_fig / f"cell_graph_{key}.pdf"))

    sc.pl.draw_graph(
        adata,
        layout=draw_layout,
        color=[group_key, combined_lineage_key, *lineage_keys],
        color_map="viridis",
        na_color="lightgray",
        legend_loc="on data",
        legend_fontsize=7,
        title=["Paul15 clusters / PAGA cell graph", "Lineage DPT", "Erythrocyte", "Monocyte"],
        show=False,
    )
    saved.append(_save_current_figure(dir_fig / "cell_graph_drivaer_summary.pdf"))

    sc.pl.umap(
        adata,
        color=group_key,
        legend_loc="on data",
        legend_fontsize=8,
        title="Paul15 clusters / PAGA UMAP",
        show=False,
    )
    saved.append(_save_current_figure(dir_fig / "umap_paul15_clusters.pdf"))

    for key in ["dpt_pseudotime_finite", combined_lineage_key, *lineage_keys]:
        sc.pl.umap(
            adata,
            color=key,
            color_map="viridis",
            na_color="lightgray",
            title=pretty_pseudotime_title(key),
            show=False,
        )
        saved.append(_save_current_figure(dir_fig / f"umap_{key}.pdf"))

    return saved


def save_paul15_baseline_arrays(adata, *, embedding_key, group_key, pseudotime_keys, save_path, embedding_dir):
    embedding_dir.mkdir(parents=True, exist_ok=True)
    cell_graph = np.asarray(adata.obsm[embedding_key], dtype=float)
    umap = np.asarray(adata.obsm["X_umap"], dtype=float)
    payload = {
        "embedding": cell_graph,
        "cell_graph": cell_graph,
        "umap": umap,
        "labels": np.asarray(adata.obs[group_key].astype(str)),
        "cell_ids": np.asarray(adata.obs_names.astype(str)),
        "embedding_key": np.asarray(embedding_key),
        "group_key": np.asarray(group_key),
    }
    for key in pseudotime_keys:
        payload[key] = np.asarray(adata.obs[key], dtype=float)
    np.savez(save_path, **payload)

    cell_graph_path = embedding_dir / f"paul15_{embedding_key}.npy"
    umap_path = embedding_dir / "paul15_X_umap.npy"
    metadata_path = embedding_dir / "paul15_embeddings_metadata.npz"
    np.save(cell_graph_path, cell_graph)
    np.save(umap_path, umap)
    np.savez(metadata_path, **payload)
    return [cell_graph_path, umap_path, metadata_path]


def clear_old_figures(dir_fig):
    for path in dir_fig.glob("*.pdf"):
        path.unlink()


def lineage_pseudotime_keys(pseudotime):
    return [lineage_pseudotime_key(lineage) for lineage in pseudotime["lineages"]]


def lineage_pseudotime_key(lineage):
    return f"dpt_{lineage}_pseudotime"


def pretty_pseudotime_title(key):
    titles = {
        "dpt_pseudotime_finite": "All-cell DPT from 10GMP",
        COMBINED_LINEAGE_DPT_KEY: "Combined lineage DPT pseudotime",
        "dpt_erythrocyte_pseudotime": "Erythrocyte trajectory pseudotime",
        "dpt_monocyte_pseudotime": "Monocyte trajectory pseudotime",
    }
    return titles.get(key, key.replace("_", " "))


def _draw_graph_key(layout):
    key = f"X_draw_graph_{layout}"
    if layout == "fa":
        return key
    return key


def validate_saved_outputs(paths):
    missing = [Path(path) for path in paths if not Path(path).exists() or Path(path).stat().st_size == 0]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Paul15 output generation failed, missing or empty files:\n{formatted}")


def _save_current_figure(save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _require_trajectory_dependencies():
    missing = []
    for module_name in ("igraph", "fa2_modified"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        packages = {
            "igraph": "igraph==0.11.9",
            "fa2_modified": "fa2-modified==0.4",
        }
        install = " ".join(packages[name] for name in missing)
        raise ModuleNotFoundError(
            "Paul15 PAGA baseline requires missing packages: "
            f"{', '.join(missing)}. Install with: pip install {install}"
        )


if __name__ == "__main__":
    main_paul15_baseline()
