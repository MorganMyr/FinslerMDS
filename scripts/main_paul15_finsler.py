"""Fast Paul15 Finsler-MDS experiments from diffusion-map geodesics."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.sparse.csgraph import connected_components, shortest_path

from finsler_mds import RandersMetric, fit_finsler_mds, geodesic_embedding_stress
from finsler_mds.utils import plot_3d_embedding_views, plot_categorical_embedding, set_axes_equal
from main_paul15_baseline import (
    COMBINED_LINEAGE_DPT_KEY,
    compute_global_and_lineage_pseudotimes,
    ensure_combined_lineage_pseudotime,
    lineage_pseudotime_keys,
)


def main_paul15_finsler_mds():
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15_finsler"
    dir_raw = dir_res / "raw"
    dir_fig = dir_res / "figures"
    dir_embeddings = dir_res / "embeddings"

    finsler_optimizer = "path_frozen"  # one of {"smacof", "path_frozen"}
    init_finsler_mds = "path_frozen"  # one of {"umap", "smacof", "path_frozen"}
    n_components = 2
    include_non_lineage_cells = True
    alpha_embedding = 0.0

    preprocessing = {
        "n_pcs": 20,
        "initial_neighbors": 4,
        "trajectory_neighbors": 10,
        "use_float64": True,
    }
    target_graph = {
        "neighbors": 35,
        "use_rep": "X_diffmap",
        "density_gamma": 0.7,
        "time_asymmetry": {
            "enabled": False,
            "lambda": 0.13,
            "min_factor": 0.1,
            "pseudotime_key": COMBINED_LINEAGE_DPT_KEY,
        },
    }
    umap = {
        "neighbors": 300,
        "use_rep": "X_diffmap",
        "init_pos": "paga",
        "min_dist": 0.5,
        "spread": 1.0,
        "maxiter": 1000,
        "negative_sample_rate": 10,
    }
    pseudotime = {
        "root_cluster": "10GMP",
        "n_dcs": 10,
        "subset_neighbors": 15,
        "lineages": {
            "erythrocyte": ["10GMP", "7MEP", "8Mk", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
            "monocyte": ["10GMP", "9GMP", "14Mo", "15Mo"],
        },
    }
    metric = RandersMetric(alpha=alpha_embedding)
    smacof = {
        "max_iter": 100,
        "pseudo_inv_solver": "gmres",
        "project_on_V": True,
        "check_monotony": False,
        "device": "auto",
    }
    path_frozen = {
        "graph_neighbors": 30,
        "max_iter": 30,
        "inner_iter": 10,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 30},
        "n_global_landmarks": 500,
        "n_local_neighbors": 30,
        "local_pair_mode": "direct",
        "max_global_targets_per_source": 500,
        "global_target_sampling": "random",
        "local_global_reweighting": "count",
        "local_weight": 1,
        "device": "auto",
        "verbose": 1,
    }
    cache = {
        "use_cache": True,
        "inputs_path": dir_raw / (
            f"paul15_diffmap_inputs_k{target_graph['neighbors']}"
            f"_dgamma{cache_token(target_graph['density_gamma'])}"
            f"_{time_asymmetry_cache_tag(target_graph['time_asymmetry'])}"
            f"{cell_scope_cache_tag(include_non_lineage_cells)}_seed{seed}.npz"
        ),
    }

    for directory in (dir_raw, dir_fig, dir_embeddings):
        directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    sc.settings.verbosity = 2
    sc.settings.autoshow = False
    sc.set_figure_params(dpi=110, frameon=False, figsize=(5, 5), facecolor="white")

    if cache["use_cache"] and cache["inputs_path"].exists():
        print(f"Loading cached Paul15 diffusion-map inputs: {cache['inputs_path']}")
        inputs = load_inputs(cache["inputs_path"])
    else:
        inputs = build_paul15_diffmap_inputs(
            seed=seed,
            preprocessing=preprocessing,
            target_graph=target_graph,
            umap=umap,
            pseudotime=pseudotime,
            include_non_lineage_cells=include_non_lineage_cells,
        )
        save_inputs(cache["inputs_path"], inputs)
        print(f"Saved Paul15 diffusion-map inputs: {cache['inputs_path']}")

    D = inputs["dissimilarities"]
    labels = inputs["labels"]
    cell_ids = inputs["cell_ids"]
    lineage_pt = inputs[COMBINED_LINEAGE_DPT_KEY]
    scope_suffix = cell_scope_cache_tag(include_non_lineage_cells)
    umap_init, init_scale = scale_embedding_to_dissimilarities(inputs["umap"], D, random_state=seed)
    print(f"Rescaled UMAP init by factor {init_scale:.6g} to match diffusion distances.")
    print(f"Target diffusion geodesic distances: {D.shape[0]} x {D.shape[1]}")

    save_input_plots(umap_init, labels, lineage_pt, dir_fig, suffix=scope_suffix)
    np.save(dir_embeddings / f"paul15_umap_init{scope_suffix}.npy", umap_init)

    optimizer_kind = normalize_finsler_optimizer(finsler_optimizer)
    symmetric_smacof_path = first_compatible_embedding_path(
        dir_embeddings,
        ["paul15_smacof_randers_alpha0_lineages.npz", "paul15_smacof_randers_alpha0.npz"],
        n_samples=D.shape[0],
        cell_ids=cell_ids,
    )
    init_embedding, init_description, init_kind = resolve_finsler_init(
        init_finsler_mds,
        umap_init=umap_init,
        embedding_sources={
            "smacof": symmetric_smacof_path
            or latest_compatible_embedding_path(
                dir_embeddings,
                "paul15_smacof_*.npz",
                n_samples=D.shape[0],
                cell_ids=cell_ids,
            ),
            "path_frozen": latest_compatible_embedding_path(
                dir_embeddings,
                "paul15_path_frozen_*.npz",
                n_samples=D.shape[0],
                cell_ids=cell_ids,
            ),
        },
        n_samples=D.shape[0],
        n_components=n_components,
        cell_ids=cell_ids,
    )
    method_key, method_title, optimizer_kwargs = optimizer_run_spec(
        optimizer_kind,
        metric=metric,
        init=init_embedding,
        smacof=smacof,
        path_frozen=path_frozen,
        seed=seed,
        n_components=n_components,
    )
    method_key = scoped_method_key(method_key, include_non_lineage_cells)

    print(f"{method_title} from {init_description}")
    embedding, stress = fit_finsler_mds(D, **optimizer_kwargs)
    full_geodesic_stress = geodesic_embedding_stress(
        embedding,
        D,
        metric=metric,
        n_neighbors=path_frozen["graph_neighbors"],
        on_unreachable="warn_skip",
    )
    print(f"  optimizer stress: {stress}")
    print(f"  final full embedding-geodesic stress: {full_geodesic_stress}")

    save_embedding_result(
        dir_embeddings / f"paul15_{method_key}.npz",
        embedding=embedding,
        stress=stress,
        full_geodesic_stress=full_geodesic_stress,
        init=init_embedding,
        cell_ids=cell_ids,
        metadata={
            "method": method_key,
            "title": method_title,
            "init_finsler_mds": init_kind,
            "init_description": init_description,
            "include_non_lineage_cells": include_non_lineage_cells,
            "metric": {"kind": "randers", "alpha": metric.alpha},
            "target_graph": target_graph,
            "smacof": smacof,
            "path_frozen": path_frozen,
            "seed": seed,
        },
    )
    save_embedding_plots(method_key, method_title, embedding, labels, lineage_pt, dir_fig)

    embeddings = {"umap_init": umap_init, method_key: embedding}
    comparison_items = [("umap_init", "UMAP init")]
    if init_kind != "umap":
        init_key = f"{init_kind}_init"
        embeddings[init_key] = init_embedding
        comparison_items.append((init_key, f"{display_init_kind(init_kind)} init"))
    comparison_items.append((method_key, display_optimizer_kind(optimizer_kind)))
    save_method_comparison(
        embeddings,
        labels,
        lineage_pt,
        dir_fig / f"comparaison{scope_suffix}.pdf",
        ordered=comparison_items,
    )

    stresses = {
        method_key: {
            "optimizer_stress": float(stress),
            "full_geodesic_stress": float(full_geodesic_stress),
            "init_finsler_mds": init_kind,
        }
    }
    save_summary(dir_raw / f"paul15_finsler_mds_summary{scope_suffix}.json", stresses)
    print(f"Saved Paul15 Finsler-MDS outputs in: {dir_res}")


def build_paul15_diffmap_inputs(
    *,
    seed,
    preprocessing,
    target_graph,
    umap,
    pseudotime,
    include_non_lineage_cells,
):
    print("Loading Scanpy Paul15 mouse hematopoiesis dataset")
    adata = sc.datasets.paul15()
    adata = restrict_to_lineage_union(
        adata,
        lineages=pseudotime["lineages"],
        include_non_lineage_cells=include_non_lineage_cells,
        cluster_key="paul15_clusters",
    )
    if preprocessing["use_float64"]:
        adata.X = adata.X.astype("float64")

    print("Preprocessing with recipe_zheng17, PCA, and diffusion map")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Use sc.pp.highly_variable_genes instead",
            category=FutureWarning,
            module="scanpy.preprocessing._recipes",
        )
        sc.pp.recipe_zheng17(adata)

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

    print("Computing PAGA on Paul15 clusters for UMAP initialization")
    sc.tl.paga(adata, groups="paul15_clusters")
    sc.pl.paga(adata, threshold=0.03, show=False)
    plt.close("all")

    print(f"Computing target kNN graph in {target_graph['use_rep']} (k={target_graph['neighbors']})")
    sc.pp.neighbors(
        adata,
        n_neighbors=target_graph["neighbors"],
        use_rep=target_graph["use_rep"],
        random_state=seed,
    )
    target_distances = adata.obsp["distances"].copy()
    target_distances, density_info = density_scaled_knn_distances(
        target_distances,
        gamma=target_graph["density_gamma"],
    )
    if target_graph["density_gamma"] != 0:
        print(
            "Applied local-density distance scaling: "
            f"gamma={target_graph['density_gamma']}, "
            f"median_sigma={density_info['median_sigma']:.6g}, "
            f"sigma_range=({density_info['min_sigma']:.6g}, {density_info['max_sigma']:.6g})"
        )

    print("Computing lineage pseudotimes for plotting and temporal asymmetry")
    compute_global_and_lineage_pseudotimes(
        adata,
        root_cluster=pseudotime["root_cluster"],
        n_dcs=pseudotime["n_dcs"],
        subset_neighbors=pseudotime["subset_neighbors"],
        lineages=pseudotime["lineages"],
        cluster_key="paul15_clusters",
    )
    lineage_keys = lineage_pseudotime_keys(pseudotime)
    ensure_combined_lineage_pseudotime(adata, lineage_keys=lineage_keys, key=COMBINED_LINEAGE_DPT_KEY)

    shortest_path_directed = False
    if target_graph["time_asymmetry"]["enabled"]:
        time_key = target_graph["time_asymmetry"]["pseudotime_key"]
        target_distances, asymmetry_info = temporally_asymmetric_knn_distances(
            target_distances,
            pseudotime=np.asarray(adata.obs[time_key], dtype=float),
            lambda_time=target_graph["time_asymmetry"]["lambda"],
            min_factor=target_graph["time_asymmetry"]["min_factor"],
        )
        shortest_path_directed = True
        print(
            "Applied temporal asymmetric edge weights: "
            f"lambda={asymmetry_info['lambda_time']:.6g}, "
            f"min_factor={asymmetry_info['min_factor']:.6g}, "
            f"finite_edges={asymmetry_info['finite_edges']}, "
            f"floored={asymmetry_info['floored_edges']} "
            f"({100 * asymmetry_info['floored_fraction']:.3f}%)"
        )

    n_components, component = connected_components(
        target_distances,
        directed=shortest_path_directed,
        connection="strong" if shortest_path_directed else "weak",
    )
    if n_components != 1:
        counts = np.bincount(component)
        raise ValueError(
            "The Paul15 target kNN graph is disconnected "
            f"({n_components} components, largest={counts.max()}). "
            "Increase target_graph['neighbors']."
        )
    dissimilarities = shortest_path(
        target_distances,
        directed=shortest_path_directed,
        return_predecessors=False,
    )
    if not np.all(np.isfinite(dissimilarities)):
        raise ValueError("Shortest-path diffusion distances contain non-finite values.")
    np.fill_diagonal(dissimilarities, 0.0)

    print(f"Computing PAGA-initialized UMAP for Finsler-MDS init (k={umap['neighbors']})")
    sc.pp.neighbors(
        adata,
        n_neighbors=umap["neighbors"],
        use_rep=umap["use_rep"],
        random_state=seed,
    )
    sc.tl.umap(
        adata,
        init_pos=umap["init_pos"],
        min_dist=umap["min_dist"],
        spread=umap["spread"],
        maxiter=umap["maxiter"],
        negative_sample_rate=umap["negative_sample_rate"],
        random_state=seed,
    )

    return {
        "dissimilarities": np.asarray(dissimilarities, dtype=float),
        "umap": np.asarray(adata.obsm["X_umap"], dtype=float),
        "labels": np.asarray(adata.obs["paul15_clusters"].astype(str), dtype=str),
        "cell_ids": np.asarray(adata.obs_names.astype(str), dtype=str),
        "include_non_lineage_cells": np.asarray(include_non_lineage_cells),
        "dpt_pseudotime_finite": np.asarray(adata.obs["dpt_pseudotime_finite"], dtype=float),
        COMBINED_LINEAGE_DPT_KEY: np.asarray(adata.obs[COMBINED_LINEAGE_DPT_KEY], dtype=float),
        **{key: np.asarray(adata.obs[key], dtype=float) for key in lineage_keys},
    }


def restrict_to_lineage_union(adata, *, lineages, include_non_lineage_cells, cluster_key):
    if include_non_lineage_cells:
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


def save_inputs(path, inputs):
    np.savez(path, **inputs)


def load_inputs(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_embedding_result(path, *, embedding, stress, full_geodesic_stress, init, cell_ids, metadata):
    np.savez(
        path,
        embedding=np.asarray(embedding, dtype=float),
        stress=np.asarray(stress, dtype=float),
        full_geodesic_stress=np.asarray(full_geodesic_stress, dtype=float),
        init=np.asarray(init, dtype=float),
        cell_ids=np.asarray(cell_ids, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def resolve_finsler_init(init_finsler_mds, *, umap_init, embedding_sources, n_samples, n_components, cell_ids):
    init_kind = normalize_finsler_init_kind(init_finsler_mds)
    if init_kind == "umap":
        init = adapt_embedding_dimension(umap_init, n_components)
        if init.shape[0] != n_samples:
            raise ValueError(f"UMAP init has {init.shape[0]} samples, expected {n_samples}.")
        return init, "UMAP init", init_kind

    source_path = embedding_sources.get(init_kind)
    if source_path is None or not source_path.exists():
        raise FileNotFoundError(
            f"Requested init_finsler_mds={init_kind!r}, but no saved embedding was found."
        )

    init = adapt_embedding_dimension(
        load_saved_embedding(source_path, cell_ids=cell_ids),
        n_components,
    )
    expected_shape = (n_samples, n_components)
    if init.shape != expected_shape:
        raise ValueError(f"Saved {init_kind} init has shape {init.shape}, expected {expected_shape}: {source_path}")
    return init, f"saved {init_kind} ({source_path})", init_kind


def adapt_embedding_dimension(embedding, n_components):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError("Saved/init embedding must be a 2D array.")
    if embedding.shape[1] == n_components:
        return embedding
    if embedding.shape[1] > n_components:
        return embedding[:, :n_components]

    init = np.zeros((embedding.shape[0], n_components), dtype=float)
    init[:, : embedding.shape[1]] = embedding
    return init


def optimizer_run_spec(optimizer_kind, *, metric, init, smacof, path_frozen, seed, n_components):
    alpha_tag = metric_alpha_tag(metric.alpha)
    if optimizer_kind == "smacof":
        return (
            f"smacof_randers_alpha{alpha_tag}",
            f"Randers SMACOF alpha={metric.alpha:g}",
            dict(
                optimizer="smacof_randers",
                metric=metric,
                init=init,
                n_components=n_components,
                n_init=1,
                n_jobs=1,
                print_time=True,
                **smacof,
            ),
        )
    if optimizer_kind == "path_frozen":
        return (
            f"path_frozen_randers_alpha{alpha_tag}",
            f"Randers path-frozen alpha={metric.alpha:g}",
            dict(
                optimizer="path_frozen",
                metric=metric,
                init=init,
                n_components=n_components,
                mask_random_state=seed,
                target_random_state=seed + 3,
                print_time=True,
                **path_frozen,
            ),
        )
    raise ValueError("finsler_optimizer must be one of {'smacof', 'path_frozen'}.")


def normalize_finsler_optimizer(finsler_optimizer):
    if not isinstance(finsler_optimizer, str):
        raise TypeError("finsler_optimizer must be one of {'smacof', 'path_frozen'}.")
    optimizer = finsler_optimizer.lower().replace("-", "_")
    if optimizer in {"smacof", "randers_smacof", "smacof_randers"}:
        return "smacof"
    if optimizer in {"path_frozen", "frozen_paths"}:
        return "path_frozen"
    raise ValueError("finsler_optimizer must be one of {'smacof', 'path_frozen'}.")


def normalize_finsler_init_kind(init_finsler_mds):
    if not isinstance(init_finsler_mds, str):
        raise TypeError("init_finsler_mds must be one of {'umap', 'smacof', 'path_frozen'}.")
    init_kind = init_finsler_mds.lower().replace("-", "_")
    if init_kind in {"umap", "umap_2d", "umap2d"}:
        return "umap"
    if init_kind in {"smacof", "randers_smacof", "smacof_randers"}:
        return "smacof"
    if init_kind in {"path_frozen", "frozen_paths"}:
        return "path_frozen"
    raise ValueError("init_finsler_mds must be one of {'umap', 'smacof', 'path_frozen'}.")


def latest_compatible_embedding_path(directory, pattern, *, n_samples, cell_ids):
    candidates = sorted(
        Path(directory).glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if saved_embedding_is_compatible(path, n_samples=n_samples, cell_ids=cell_ids):
            return path
    return None


def first_compatible_embedding_path(directory, patterns, *, n_samples, cell_ids):
    for pattern in patterns:
        path = latest_compatible_embedding_path(
            directory,
            pattern,
            n_samples=n_samples,
            cell_ids=cell_ids,
        )
        if path is not None:
            return path
    return None


def saved_embedding_is_compatible(path, *, n_samples, cell_ids):
    try:
        embedding = load_saved_embedding(path, cell_ids=cell_ids)
    except (ValueError, KeyError, FileNotFoundError, OSError):
        return False
    return embedding.shape[0] == n_samples


def load_saved_embedding(path, *, cell_ids=None):
    path = Path(path)
    if path.suffix == ".npy":
        embedding = np.asarray(np.load(path), dtype=float)
        if cell_ids is not None and embedding.shape[0] != len(cell_ids):
            raise ValueError(f"Saved embedding has {embedding.shape[0]} cells, expected {len(cell_ids)}: {path}")
        return embedding
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "embedding" not in data:
                raise KeyError(f"Saved embedding file has no 'embedding' array: {path}")
            embedding = np.asarray(data["embedding"], dtype=float)
            if cell_ids is None:
                return embedding
            if embedding.shape[0] == len(cell_ids) and "cell_ids" not in data:
                return embedding
            if "cell_ids" not in data:
                raise ValueError(f"Saved embedding has no cell_ids and incompatible size: {path}")
            saved_cell_ids = np.asarray(data["cell_ids"]).astype(str)
            return align_embedding_to_cell_ids(embedding, saved_cell_ids=saved_cell_ids, target_cell_ids=cell_ids)
    raise ValueError(f"Unsupported saved embedding extension: {path}")


def align_embedding_to_cell_ids(embedding, *, saved_cell_ids, target_cell_ids):
    target_cell_ids = np.asarray(target_cell_ids).astype(str)
    if np.array_equal(saved_cell_ids, target_cell_ids):
        return embedding

    positions = {cell_id: idx for idx, cell_id in enumerate(saved_cell_ids)}
    try:
        order = np.asarray([positions[cell_id] for cell_id in target_cell_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Saved embedding is missing target cell id {exc.args[0]!r}.") from exc
    return embedding[order]


def display_optimizer_kind(optimizer_kind):
    if optimizer_kind == "smacof":
        return "SMACOF"
    if optimizer_kind == "path_frozen":
        return "Path-frozen"
    return optimizer_kind


def display_init_kind(init_kind):
    if init_kind == "smacof":
        return "SMACOF"
    if init_kind == "path_frozen":
        return "Path-frozen"
    if init_kind == "umap":
        return "UMAP"
    return init_kind


def temporally_asymmetric_knn_distances(distances, *, pseudotime, lambda_time, min_factor):
    graph = distances.tocsr(copy=True)
    graph = graph.maximum(graph.T).tocoo(copy=True)
    pseudotime = np.asarray(pseudotime, dtype=float)
    lambda_time = float(lambda_time)
    min_factor = float(min_factor)
    if min_factor <= 0:
        raise ValueError("Temporal asymmetry min_factor must be positive.")

    data = np.asarray(graph.data, dtype=float).copy()
    finite = (
        (data > 0)
        & np.isfinite(pseudotime[graph.row])
        & np.isfinite(pseudotime[graph.col])
    )
    delta_t = pseudotime[graph.col[finite]] - pseudotime[graph.row[finite]]
    base = data[finite]
    floor = min_factor * base
    proposed = base - lambda_time * delta_t
    floored = proposed < floor
    data[finite] = np.maximum(floor, proposed)
    graph.data = data

    info = {
        "lambda_time": lambda_time,
        "min_factor": min_factor,
        "finite_edges": int(np.sum(finite)),
        "floored_edges": int(np.sum(floored)),
        "floored_fraction": float(np.mean(floored)) if len(floored) else 0.0,
    }
    return graph.tocsr(), info


def density_scaled_knn_distances(distances, *, gamma):
    graph = distances.tocsr(copy=True)
    gamma = float(gamma)
    sigmas = local_knn_scales(graph)
    finite = np.isfinite(sigmas) & (sigmas > 0)
    if not np.any(finite):
        raise ValueError("Cannot density-scale distances because all local scales are non-positive.")

    median_sigma = float(np.median(sigmas[finite]))
    sigmas = sigmas.copy()
    sigmas[~finite] = median_sigma
    info = {
        "median_sigma": median_sigma,
        "min_sigma": float(np.min(sigmas)),
        "max_sigma": float(np.max(sigmas)),
    }
    if gamma == 0:
        return graph, info

    coo = graph.tocoo(copy=True)
    local_scale = np.sqrt(sigmas[coo.row] * sigmas[coo.col])
    factors = np.divide(
        median_sigma,
        local_scale,
        out=np.ones_like(coo.data, dtype=float),
        where=local_scale > 0,
    )
    coo.data = coo.data * factors**gamma
    return coo.tocsr(), info


def local_knn_scales(graph):
    graph = graph.tocsr()
    sigmas = np.empty(graph.shape[0], dtype=float)
    for i in range(graph.shape[0]):
        row = graph.data[graph.indptr[i]:graph.indptr[i + 1]]
        row = row[row > 0]
        sigmas[i] = np.max(row) if len(row) else np.nan
    return sigmas


def cache_token(value):
    return str(value).replace("-", "m").replace(".", "p")


def metric_alpha_tag(alpha):
    alpha = float(alpha)
    if alpha.is_integer():
        return str(int(alpha))
    return cache_token(alpha)


def time_asymmetry_cache_tag(config):
    if not config["enabled"]:
        return "sym"
    return (
        f"tasym_l{cache_token(config['lambda'])}"
        f"_floor{cache_token(config['min_factor'])}"
    )


def cell_scope_cache_tag(include_non_lineage_cells):
    return "" if include_non_lineage_cells else "_lineages"


def scoped_method_key(method_key, include_non_lineage_cells):
    return method_key if include_non_lineage_cells else f"{method_key}_lineages"


def scale_embedding_to_dissimilarities(embedding, dissimilarities, *, random_state, n_pairs=50_000):
    embedding = np.asarray(embedding, dtype=float)
    D = np.asarray(dissimilarities, dtype=float)
    X = embedding - embedding.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(random_state)
    n = len(X)
    rows = rng.integers(0, n, size=n_pairs)
    cols = rng.integers(0, n, size=n_pairs)
    valid = rows != cols
    rows = rows[valid]
    cols = cols[valid]
    target = D[rows, cols]
    current = np.linalg.norm(X[rows] - X[cols], axis=1)
    valid = np.isfinite(target) & (target > 0) & (current > 0)
    if not np.any(valid):
        return X, 1.0
    ratios = target[valid] / current[valid]
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        return X, 1.0
    return X * scale, scale


def save_summary(path, stresses):
    path.write_text(json.dumps(stresses, indent=2, sort_keys=True), encoding="utf-8")


def save_input_plots(init, labels, pseudotime, dir_fig, *, suffix=""):
    plot_categorical_embedding(
        init,
        labels=labels,
        title="Paul15 UMAP init (PAGA initialized)",
        save_path=dir_fig / f"umap_init_clusters{suffix}.pdf",
        s=7,
    )
    plt.close("all")
    plot_continuous_embedding(
        init,
        pseudotime,
        title="Paul15 UMAP init: lineage pseudotime",
        save_path=dir_fig / f"umap_init_lineage_pseudotime{suffix}.pdf",
    )


def save_embedding_plots(method_key, method_title, embedding, labels, pseudotime, dir_fig):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] == 3:
        plot_3d_embedding_views(
            embedding,
            labels=labels,
            title=method_title,
            save_path=dir_fig / f"{method_key}_views.pdf",
            point_fraction=1.0,
            random_state=42,
            s=7,
        )
        plt.close("all")
        plot_3d_continuous_embedding_views(
            embedding,
            values=pseudotime,
            title=f"{method_title}: lineage pseudotime",
            save_path=dir_fig / f"{method_key}_lineage_pseudotime_views.pdf",
            point_fraction=1.0,
            random_state=42,
            s=7,
        )
        plt.close("all")
        return

    plot_categorical_embedding(
        embedding,
        labels=labels,
        title=method_title,
        save_path=dir_fig / f"{method_key}_clusters.pdf",
        s=7,
    )
    plt.close("all")
    plot_continuous_embedding(
        embedding,
        pseudotime,
        title=f"{method_title}: lineage pseudotime",
        save_path=dir_fig / f"{method_key}_lineage_pseudotime.pdf",
    )


def save_method_comparison(embeddings, labels, pseudotime, save_path, *, ordered):
    fig, axes = plt.subplots(2, len(ordered), figsize=(5 * len(ordered), 9))
    axes = np.asarray(axes).reshape(2, len(ordered))
    for col, (key, title) in enumerate(ordered):
        embedding = embeddings[key]
        scatter_categorical(axes[0, col], embedding, labels, title)
        scatter_continuous(axes[1, col], embedding, pseudotime, f"{title}: lineage DPT")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_3d_continuous_embedding_views(
    embedding,
    *,
    values,
    title,
    save_path,
    views=None,
    point_fraction=1.0,
    random_state=None,
    s=8,
    cmap="viridis",
):
    embedding = np.asarray(embedding, dtype=float)
    values = np.asarray(values, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] != 3:
        raise ValueError("embedding must have shape (n_samples, 3).")
    if values.shape != (len(embedding),):
        raise ValueError("values must have shape (n_samples,).")
    if views is None:
        views = [
            ("front", 20, -60),
            ("side", 20, 30),
            ("top", 90, -90),
            ("diagonal", 35, 135),
        ]

    rng = np.random.default_rng(random_state)
    n = len(embedding)
    if point_fraction >= 1:
        plot_idx = np.arange(n)
    else:
        size = max(1, int(np.ceil(point_fraction * n)))
        plot_idx = np.sort(rng.choice(n, size=size, replace=False))

    finite = np.isfinite(values)
    n_views = len(views)
    n_cols = min(2, n_views)
    n_rows = int(np.ceil(n_views / n_cols))
    fig = plt.figure(figsize=(6 * n_cols, 5.5 * n_rows))
    mappable = None
    for view_id, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(n_rows, n_cols, view_id + 1, projection="3d")
        view_idx = plot_idx
        finite_idx = view_idx[finite[view_idx]]
        missing_idx = view_idx[~finite[view_idx]]
        if len(missing_idx):
            ax.scatter(
                embedding[missing_idx, 0],
                embedding[missing_idx, 1],
                embedding[missing_idx, 2],
                c="lightgray",
                s=s,
                lw=0,
            )
        if len(finite_idx):
            mappable = ax.scatter(
                embedding[finite_idx, 0],
                embedding[finite_idx, 1],
                embedding[finite_idx, 2],
                c=values[finite_idx],
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                s=s,
                lw=0,
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name)
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        ax.set_zlabel("Embedding 3")

    if title is not None:
        fig.suptitle(title)
    if mappable is not None:
        fig.colorbar(mappable, ax=fig.axes, fraction=0.025, pad=0.02)
    fig.tight_layout(rect=(0, 0, 0.96, 0.96))
    fig.savefig(save_path)
    plt.close(fig)


def plot_continuous_embedding(embedding, values, *, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_continuous(ax, embedding, values, title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def scatter_continuous(ax, embedding, values, title):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if np.any(~finite):
        ax.scatter(embedding[~finite, 0], embedding[~finite, 1], c="lightgray", s=7, lw=0)
    scatter = ax.scatter(
        embedding[finite, 0],
        embedding[finite, 1],
        c=values[finite],
        cmap="viridis",
        s=7,
        lw=0,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def scatter_categorical(ax, embedding, labels, title):
    labels = np.asarray(labels).astype(str)
    categories, codes = np.unique(labels, return_inverse=True)
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=codes, cmap="tab20", s=7, lw=0)
    handles, _ = scatter.legend_elements(num=len(categories))
    ax.legend(
        handles,
        categories,
        title="clusters",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=7,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    main_paul15_finsler_mds()
