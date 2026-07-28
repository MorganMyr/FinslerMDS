"""Shared helpers for Paul15 trajectory and Finsler-MDS scripts."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.sparse.csgraph import connected_components, shortest_path

from .dissimilarity_graphs import temporally_asymmetric_knn_distances
from .embedding_io import cache_token
from .plotting import (
    plot_3d_continuous_embedding_views,
    plot_3d_embedding_views,
    plot_categorical_embedding,
    plot_continuous_embedding,
)


COMBINED_LINEAGE_DPT_KEY = "dpt_lineage_pseudotime"
PAUL15_CLUSTER_KEY = "paul15_clusters"
PAUL15_DPT_KEY = "dpt_pseudotime_finite"
PAUL15_LINEAGES = {
    "erythrocyte": [
        "10GMP",
        "7MEP",
        "8Mk",
        "1Ery",
        "2Ery",
        "3Ery",
        "4Ery",
        "5Ery",
        "6Ery",
    ],
    "monocyte": ["10GMP", "9GMP", "14Mo", "15Mo"],
}

_METHOD_TAGS = {
    "umap": "umap",
    "phate": "phate",
    "gradient_descent": "gd",
    "gd": "gd",
    "path_frozen": "pf",
    "pf": "pf",
    "finsler_umap": "fumap",
    "fumap": "fumap",
}


def paul15_method_tag(method):
    """Return the short filename tag associated with a Paul15 method."""
    key = str(method).lower().replace("-", "_")
    try:
        return _METHOD_TAGS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown Paul15 method: {method!r}.") from exc


def prepare_paul15(
    *,
    excluded_clusters=(),
    n_pcs=20,
    pseudotime_key=PAUL15_DPT_KEY,
    seed=42,
):
    """Load Paul15 and compute PCA, diffusion components, and DPT."""
    _validate_pseudotime_key(pseudotime_key)
    print("Loading and preprocessing Paul15")
    adata = sc.datasets.paul15()
    if excluded_clusters:
        labels = np.asarray(adata.obs[PAUL15_CLUSTER_KEY].astype(str))
        adata = adata[~np.isin(labels, excluded_clusters)].copy()

    adata.X = adata.X.astype("float64")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Use sc.pp.highly_variable_genes instead",
            category=FutureWarning,
            module="scanpy.preprocessing._recipes",
        )
        sc.pp.recipe_zheng17(adata)

    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=n_pcs, random_state=seed)
    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_diffmap", random_state=seed)

    _compute_global_dpt(
        adata,
        root_cluster="10GMP",
        cluster_key=PAUL15_CLUSTER_KEY,
        n_dcs=10,
    )
    if pseudotime_key == COMBINED_LINEAGE_DPT_KEY:
        for lineage, clusters in PAUL15_LINEAGES.items():
            adata.obs[lineage_pseudotime_key(lineage)] = lineage_dpt(
                adata,
                clusters=clusters,
                root_cluster="10GMP",
                cluster_key=PAUL15_CLUSTER_KEY,
                subset_neighbors=15,
                n_dcs=10,
            )
        ensure_combined_lineage_pseudotime(
            adata,
            lineage_keys=[lineage_pseudotime_key(lineage) for lineage in PAUL15_LINEAGES],
        )
    return adata


def paul15_dissimilarities(
    adata,
    *,
    representation,
    n_neighbors=12,
    lambda_time=0.0,
    min_factor=0.1,
    pseudotime_key=PAUL15_DPT_KEY,
    seed=42,
):
    """Return (possibly DPT-asymmetric) geodesic dissimilarities."""
    _validate_pseudotime_key(pseudotime_key)
    if pseudotime_key not in adata.obs:
        raise KeyError(f"Pseudotime {pseudotime_key!r} was not computed by prepare_paul15().")
    use_rep = paul15_representation_key(representation)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=seed)
    graph = adata.obsp["distances"].maximum(adata.obsp["distances"].T).tocsr()
    directed = float(lambda_time) != 0.0
    if directed:
        graph, info = temporally_asymmetric_knn_distances(
            graph,
            pseudotime=np.asarray(adata.obs[pseudotime_key], dtype=float),
            lambda_time=lambda_time,
            min_factor=min_factor,
        )
        print(
            f"Temporal asymmetry: lambda={lambda_time:g}, "
            f"floored edges={info['floored_edges']}"
        )

    n_components, labels = connected_components(
        graph,
        directed=directed,
        connection="strong" if directed else "weak",
    )
    if n_components != 1:
        raise ValueError(
            f"The {representation} kNN graph has {n_components} components "
            f"(largest: {np.bincount(labels).max()}); increase n_neighbors."
        )

    dissimilarities = shortest_path(graph, directed=directed)
    if not np.all(np.isfinite(dissimilarities)):
        raise ValueError("The geodesic dissimilarities contain non-finite values.")
    np.fill_diagonal(dissimilarities, 0.0)
    return np.asarray(dissimilarities, dtype=float)


def paul15_representation_key(representation):
    try:
        return {"pca": "X_pca", "diffmap": "X_diffmap"}[str(representation).lower()]
    except KeyError as exc:
        raise ValueError("representation must be 'pca' or 'diffmap'.") from exc


def ensure_paul15_umap(
    adata,
    path,
    *,
    representation,
    n_components,
    n_neighbors=20,
    seed=42,
):
    """Load a minimal UMAP cache or compute it in the requested space."""
    path = Path(path)
    cell_ids = np.asarray(adata.obs_names.astype(str))
    if path.exists():
        try:
            return load_paul15_embedding(
                path,
                cell_ids=cell_ids,
                expected_shape=(adata.n_obs, n_components),
            )
        except (KeyError, OSError, ValueError):
            print(f"Ignoring incompatible UMAP cache: {path}")

    use_rep = paul15_representation_key(representation)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=seed)
    init_pos = "spectral"
    if representation == "diffmap":
        sc.tl.paga(adata, groups=PAUL15_CLUSTER_KEY)
        sc.pl.paga(adata, show=False)
        plt.close("all")
        init_pos = "paga"
        if n_components == 3:
            from scanpy.tools._utils import get_init_pos_from_paga

            paga_init = get_init_pos_from_paga(adata, random_state=seed)
            scale = max(float(np.max(np.ptp(paga_init, axis=0))), 1.0) * 1e-3
            z = np.random.default_rng(seed).normal(scale=scale, size=(adata.n_obs, 1))
            init_pos = np.column_stack((paga_init, z))
    sc.tl.umap(
        adata,
        n_components=n_components,
        init_pos=init_pos,
        min_dist=0.5,
        spread=1.0,
        maxiter=1000,
        negative_sample_rate=10,
        random_state=seed,
    )
    embedding = np.asarray(adata.obsm["X_umap"], dtype=float)
    if embedding.shape != (adata.n_obs, n_components):
        raise RuntimeError(
            f"Scanpy returned a {embedding.shape[1]}D UMAP while {n_components}D was requested."
        )
    save_paul15_embedding(path, embedding, cell_ids)
    return embedding


def paul15_result_stem(
    method,
    *,
    representation=None,
    metric=None,
    lambda_time=0.0,
    pseudotime_key=None,
    alpha=0.0,
    n_components=2,
    n_landmark=None,
    n_neighbors=None,
):
    """Build short filenames from the parameters that identify an experiment."""
    parts = [paul15_method_tag(method)]
    if representation is not None:
        parts.append(str(representation).lower())
    if metric is not None:
        metric = str(metric).lower().replace("-", "_")
        metric_tags = {
            "randers": None,
            "matsumoto": "matsumoto",
            "convexified_matsumoto": "convexified_matsumoto",
        }
        try:
            metric_tag = metric_tags[metric]
        except KeyError as exc:
            raise ValueError(f"Unknown Finsler metric: {metric!r}.") from exc
        if metric_tag is not None:
            parts.append(metric_tag)
    if n_landmark is not None:
        parts.append(f"lm{int(n_landmark)}")
    if n_neighbors is not None:
        parts.append(f"k{int(n_neighbors)}")
    if float(lambda_time) != 0.0:
        parts.append(f"l{cache_token(lambda_time)}")
        if pseudotime_key is not None:
            _validate_pseudotime_key(pseudotime_key)
            if pseudotime_key == COMBINED_LINEAGE_DPT_KEY:
                parts.append("line")
    if float(alpha) != 0.0:
        parts.append(f"a{cache_token(alpha)}")
    if int(n_components) != 2:
        parts.append(f"{int(n_components)}d")
    return "_".join(parts)


def save_paul15_embedding(
    path,
    embedding,
    cell_ids,
    *,
    objective=None,
    landmark_embedding=None,
):
    """Save only the arrays needed to plot or continue an optimization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "embedding": np.asarray(embedding, dtype=float),
        "cell_ids": np.asarray(cell_ids, dtype=str),
    }
    if objective is not None:
        arrays["objective"] = np.asarray(objective, dtype=float)
    if landmark_embedding is not None:
        arrays["landmark_embedding"] = np.asarray(landmark_embedding, dtype=float)
    np.savez(path, **arrays)
    print(f"Saved {path}")


def load_paul15_embedding(path, *, key="embedding", cell_ids=None, expected_shape=None):
    path = Path(path)
    with np.load(path) as data:
        embedding = np.asarray(data[key], dtype=float)
        if cell_ids is not None:
            saved_ids = np.asarray(data["cell_ids"]).astype(str)
            if not np.array_equal(saved_ids, np.asarray(cell_ids).astype(str)):
                raise ValueError(f"Cell order does not match the saved embedding: {path}")
    if expected_shape is not None and embedding.shape != tuple(expected_shape):
        raise ValueError(f"Saved embedding has shape {embedding.shape}, expected {expected_shape}: {path}")
    return embedding


def latest_paul15_embedding(
    directory,
    method,
    *,
    expected_shape,
    cell_ids,
    key="embedding",
    n_landmark=None,
):
    """Load the newest compatible result of one method."""
    tag = paul15_method_tag(method)
    pattern = f"{tag}_lm{int(n_landmark)}*.npz" if n_landmark is not None else f"{tag}*.npz"
    candidates = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            embedding = load_paul15_embedding(
                path,
                key=key,
                cell_ids=cell_ids,
                expected_shape=expected_shape,
            )
        except (KeyError, OSError, ValueError):
            continue
        print(f"Using init from {path}")
        return embedding
    raise FileNotFoundError(f"No compatible {method} embedding found in {directory}.")


def save_paul15_plots(
    embedding,
    *,
    labels,
    pseudotime,
    pseudotime_key,
    directory,
    stem,
    title,
):
    """Save cluster- and DPT-colored views for a 2D or 3D embedding."""
    dpt_suffix, dpt_title = _pseudotime_plot_info(pseudotime_key)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cluster_path = directory / f"{stem}_clusters.pdf"
    dpt_path = directory / f"{stem}_{dpt_suffix}.pdf"
    if embedding.shape[1] == 2:
        cluster_fig, _ = plot_categorical_embedding(
            embedding,
            labels,
            title=f"{title} — clusters",
            save_path=cluster_path,
        )
        dpt_fig, _ = plot_continuous_embedding(
            embedding,
            pseudotime,
            title=f"{title} — {dpt_title}",
            save_path=dpt_path,
        )
    elif embedding.shape[1] == 3:
        cluster_fig, _ = plot_3d_embedding_views(
            embedding,
            labels,
            title=f"{title} — clusters",
            save_path=cluster_path,
            views=[("", 25, -60)],
        )
        dpt_fig, _ = plot_3d_continuous_embedding_views(
            embedding,
            values=pseudotime,
            title=f"{title} — {dpt_title}",
            save_path=dpt_path,
            views=[("", 25, -60)],
        )
    else:
        raise ValueError("Only 2D and 3D Paul15 plots are supported.")
    plt.close(cluster_fig)
    plt.close(dpt_fig)


def _validate_pseudotime_key(key):
    if key not in {PAUL15_DPT_KEY, COMBINED_LINEAGE_DPT_KEY}:
        raise ValueError(
            f"pseudotime_key must be {PAUL15_DPT_KEY!r} or "
            f"{COMBINED_LINEAGE_DPT_KEY!r}."
        )


def _pseudotime_plot_info(key):
    _validate_pseudotime_key(key)
    return (
        ("dpt_full", "global DPT pseudotime")
        if key == PAUL15_DPT_KEY
        else ("dpt_line", "lineage DPT pseudotime")
    )


def _compute_global_dpt(adata, *, root_cluster, cluster_key, n_dcs):
    labels = np.asarray(adata.obs[cluster_key].astype(str))
    root = np.flatnonzero(labels == root_cluster)
    if len(root) == 0:
        raise ValueError(f"Could not find DPT root cluster {root_cluster!r}.")
    adata.uns["iroot"] = int(root[0])
    sc.tl.dpt(adata, n_dcs=n_dcs)
    adata.obs[PAUL15_DPT_KEY] = finite_rescaled(adata.obs["dpt_pseudotime"])


def compute_global_and_lineage_pseudotimes(
    adata,
    *,
    root_cluster,
    n_dcs,
    subset_neighbors,
    lineages,
    cluster_key,
):
    _compute_global_dpt(
        adata,
        root_cluster=root_cluster,
        cluster_key=cluster_key,
        n_dcs=n_dcs,
    )

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
