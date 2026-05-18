"""Shared helpers for Paul15 trajectory and Finsler-MDS scripts."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from .dissimilarity_graphs import normalize_asymmetry_type
from .embedding_io import cache_token


COMBINED_LINEAGE_DPT_KEY = "dpt_lineage_pseudotime"


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


def ensure_combined_lineage_pseudotime(adata, *, lineage_keys, key=COMBINED_LINEAGE_DPT_KEY):
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

    lo, hi = float(np.nanmin(out[finite])), float(np.nanmax(out[finite]))
    out[finite] = (out[finite] - lo) / (hi - lo) if hi > lo else 0.0
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

    def stable(layout):
        return _layout_is_available_and_stable(
            adata,
            layout=layout,
            max_abs_coordinate=draw_graph["max_abs_coordinate"],
        )

    if not force and stable(preferred):
        return _remember_draw_layout(adata, preferred)

    if force or _draw_graph_key(preferred) not in adata.obsm:
        print(f"Computing {preferred} graph initialized from PAGA")
        sc.tl.draw_graph(adata, layout=preferred, init_pos=draw_graph["init_pos"], random_state=seed)

    if stable(preferred):
        return _remember_draw_layout(adata, preferred)
    if not force and stable(fallback):
        return _remember_draw_layout(adata, fallback)

    print(
        f"Paul15 {preferred} layout is numerically unstable in this environment; "
        f"falling back to {fallback}."
    )
    sc.tl.draw_graph(adata, layout=fallback, init_pos=draw_graph["init_pos"], random_state=seed)
    if not stable(fallback):
        raise RuntimeError(f"Could not compute a stable Paul15 draw_graph layout ({preferred} or {fallback}).")
    return _remember_draw_layout(adata, fallback)


def _remember_draw_layout(adata, layout):
    adata.uns["paul15_draw_graph_layout"] = layout
    return layout


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


def restrict_to_lineage_union(
    adata,
    *,
    lineages,
    include_non_lineage_cells,
    exclude_19lymph_when_all_cells=False,
    cluster_key,
):
    if include_non_lineage_cells:
        if exclude_19lymph_when_all_cells:
            labels = np.asarray(adata.obs[cluster_key].astype(str))
            mask = labels != "19Lymph"
            print(f"Using all Paul15 cells except 19Lymph: {int(np.sum(mask))} / {adata.n_obs} cells")
            return adata[mask].copy()
        print(f"Using all Paul15 cells: {adata.n_obs}")
        return adata

    labels = np.asarray(adata.obs[cluster_key].astype(str))
    lineage_clusters = sorted({str(cluster) for clusters in lineages.values() for cluster in clusters})
    mask = np.isin(labels, lineage_clusters)
    if not np.any(mask):
        raise ValueError("Lineage-only mode selected no cells. Check pseudotime['lineages'].")
    print(
        "Using erythrocyte/monocyte lineage union only: "
        f"{int(np.sum(mask))} / {adata.n_obs} cells"
    )
    return adata[mask].copy()


def paul15_inputs_cache_name(
    target_graph,
    *,
    include_non_lineage_cells,
    exclude_19lymph_when_all_cells=False,
    seed,
):
    return (
        f"inputs_k{target_graph['neighbors']}"
        f"_dg{cache_token(target_graph['density_gamma'])}"
        f"_{asymmetry_cache_tag(target_graph)}"
        f"{cell_scope_cache_tag(include_non_lineage_cells, exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells)}"
        f"_s{seed}.npz"
    )


def paul15_output_family(target_graph):
    asymmetry_type = normalize_asymmetry_type(target_graph.get("asymmetry_type"))
    if asymmetry_type is None:
        return "symmetric"
    if asymmetry_type == "pseudotime":
        return "pseudotime_asym"
    if asymmetry_type == "density":
        return "density_asym"
    raise AssertionError(f"Unhandled asymmetry type: {asymmetry_type!r}")


def asymmetry_cache_tag(target_graph):
    asymmetry_type = normalize_asymmetry_type(target_graph.get("asymmetry_type"))
    if asymmetry_type is None:
        return "sym"
    if asymmetry_type == "density":
        return "den"
    config = target_graph["time_asymmetry"]
    return (
        f"pt_l{cache_token(config['lambda'])}"
        f"_f{cache_token(config['min_factor'])}"
    )


def cell_scope_cache_tag(include_non_lineage_cells, *, exclude_19lymph_when_all_cells=False):
    if not include_non_lineage_cells:
        return "_lineages"
    return "_no19lymph" if exclude_19lymph_when_all_cells else ""


def scoped_method_key(method_key, include_non_lineage_cells):
    return method_key if include_non_lineage_cells else f"{method_key}_lin"


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
    return f"X_draw_graph_{layout}"


def save_current_figure(save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def require_trajectory_dependencies():
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
