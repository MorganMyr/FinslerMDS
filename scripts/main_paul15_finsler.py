"""Fast Paul15 Finsler-MDS experiments from diffusion-map geodesics."""

from __future__ import annotations

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
from finsler_mds.utils import (
    plot_3d_continuous_embedding_views,
    plot_3d_embedding_views,
    plot_categorical_embedding,
    plot_continuous_embedding,
)
from finsler_mds.utils.dissimilarity_graphs import (
    density_scaled_knn_distances,
    normalize_asymmetry_type,
    temporally_asymmetric_knn_distances,
)
from finsler_mds.utils.embedding_io import (
    latest_compatible_embedding_path,
    load_npz_inputs as load_inputs,
    metric_alpha_tag,
    resolve_finsler_init,
    save_embedding_result,
    save_npz_inputs as save_inputs,
    save_summary,
    scale_embedding_to_dissimilarities,
)
from finsler_mds.utils.paul15 import (
    COMBINED_LINEAGE_DPT_KEY,
    cell_scope_cache_tag,
    compute_global_and_lineage_pseudotimes,
    ensure_combined_lineage_pseudotime,
    lineage_pseudotime_keys,
    paul15_inputs_cache_name,
    paul15_output_family,
    restrict_to_lineage_union,
    scoped_method_key,
)


def main_paul15_finsler_mds():
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15"
    dir_raw = dir_res / "raw"

    finsler_optimizer = "path_frozen"  # one of {"smacof", "path_frozen"}
    init_finsler_mds = "path_frozen"  # one of {"umap", "smacof", "path_frozen"}
    n_components = 2
    include_non_lineage_cells = True
    exclude_19lymph_when_all_cells = True
    alpha_embedding = 0

    preprocessing = {
        "n_pcs": 20,
        "initial_neighbors": 4,
        "trajectory_neighbors": 10,
        "use_float64": True,
    }
    target_graph = {
        "neighbors": 12,
        "use_rep": "X_diffmap",
        "asymmetry_type": None,  # one of {None, "pseudotime", "density"}
        "density_gamma": 1,
        "time_asymmetry": {
            "lambda": 0.13,
            "min_factor": 0.1,
            "pseudotime_key": COMBINED_LINEAGE_DPT_KEY,
        },
    }
    output_family = paul15_output_family(target_graph)
    dir_family = dir_res / output_family
    dir_fig = dir_family / "figures"
    dir_embeddings = dir_family / "embeddings"
    umap = {
        "neighbors": 20,
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
        "max_iter": 20,
        "pseudo_inv_solver": "gmres",
        "project_on_V": True,
        "check_monotony": False,
        "device": "auto",
    }
    path_frozen = {
        "graph_neighbors": 20,
        "outer_iter": 50,
        "inner_iter": 5,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 30},
        "n_landmark": 200,
        "n_local_pairs": 20,
        "local_pair_mode": "direct",
        "targets_per_landmark": 200,
        "global_target_sampling": "random",
        "local_global_reweighting": "count",
        "local_weight": 1,
        "device": "auto",
        "verbose": 1,
    }
    cache = {
        "use_cache": True,
        "inputs_path": dir_raw / paul15_inputs_cache_name(
            target_graph,
            include_non_lineage_cells=include_non_lineage_cells,
            exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
            seed=seed,
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
            exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
        )
        save_inputs(cache["inputs_path"], inputs)
        print(f"Saved Paul15 diffusion-map inputs: {cache['inputs_path']}")

    D = inputs["dissimilarities"]
    labels = inputs["labels"]
    cell_ids = inputs["cell_ids"]
    lineage_pt = inputs[COMBINED_LINEAGE_DPT_KEY]
    scope_suffix = cell_scope_cache_tag(
        include_non_lineage_cells,
        exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
    )
    umap_init, init_scale = scale_embedding_to_dissimilarities(inputs["umap"], D, random_state=seed)
    print(f"Rescaled UMAP init by factor {init_scale:.6g} to match diffusion distances.")
    print(f"Target diffusion geodesic distances: {D.shape[0]} x {D.shape[1]}")

    save_input_plots(umap_init, labels, lineage_pt, dir_fig, suffix=scope_suffix)
    np.save(dir_embeddings / f"umap{scope_suffix}.npy", umap_init)

    optimizer_kind = normalize_finsler_optimizer(finsler_optimizer)
    init_embedding, init_description, init_kind = resolve_finsler_init(
        init_finsler_mds,
        umap_init=umap_init,
        embedding_sources={
            "smacof": latest_compatible_embedding_path(
                dir_embeddings,
                "smacof*.npz",
                n_samples=D.shape[0],
                cell_ids=cell_ids,
            ),
            "path_frozen": latest_compatible_embedding_path(
                dir_embeddings,
                "pf*.npz",
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
        output_family=output_family,
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
        dir_embeddings / f"{method_key}.npz",
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
            "exclude_19lymph_when_all_cells": exclude_19lymph_when_all_cells,
            "metric": {"kind": "randers", "alpha": metric.alpha},
            "target_graph": target_graph,
            "smacof": smacof,
            "path_frozen": path_frozen,
            "seed": seed,
        },
    )
    save_embedding_plots(method_key, method_title, embedding, labels, lineage_pt, dir_fig)

    stresses = {
        method_key: {
            "optimizer_stress": float(stress),
            "full_geodesic_stress": float(full_geodesic_stress),
            "init_finsler_mds": init_kind,
        }
    }
    save_summary(dir_raw / f"summary_{output_family}{scope_suffix}.json", stresses)
    print(f"Saved Paul15 Finsler-MDS outputs in: {dir_family}")


def build_paul15_diffmap_inputs(
    *,
    seed,
    preprocessing,
    target_graph,
    umap,
    pseudotime,
    include_non_lineage_cells,
    exclude_19lymph_when_all_cells,
):
    asymmetry_type = normalize_asymmetry_type(target_graph.get("asymmetry_type"))
    print("Loading Scanpy Paul15 mouse hematopoiesis dataset")
    adata = sc.datasets.paul15()
    adata = restrict_to_lineage_union(
        adata,
        lineages=pseudotime["lineages"],
        include_non_lineage_cells=include_non_lineage_cells,
        exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
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
        mode="source" if asymmetry_type == "density" else "symmetric",
    )
    if target_graph["density_gamma"] != 0:
        mode_label = "source-asymmetric" if asymmetry_type == "density" else "symmetric"
        print(
            "Applied local-density distance scaling: "
            f"mode={mode_label}, gamma={target_graph['density_gamma']}, "
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

    shortest_path_directed = asymmetry_type in {"density", "pseudotime"}
    if asymmetry_type == "pseudotime":
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
    elif asymmetry_type == "density":
        print("Keeping source-density scaled kNN graph directed.")

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
        "exclude_19lymph_when_all_cells": np.asarray(exclude_19lymph_when_all_cells),
        "dpt_pseudotime_finite": np.asarray(adata.obs["dpt_pseudotime_finite"], dtype=float),
        COMBINED_LINEAGE_DPT_KEY: np.asarray(adata.obs[COMBINED_LINEAGE_DPT_KEY], dtype=float),
        **{key: np.asarray(adata.obs[key], dtype=float) for key in lineage_keys},
    }


def optimizer_run_spec(optimizer_kind, *, metric, output_family, init, smacof, path_frozen, seed, n_components):
    alpha_tag = metric_alpha_tag(metric.alpha)
    if optimizer_kind == "smacof":
        method_key = "smacof" if output_family == "symmetric" else f"smacof_a{alpha_tag}"
        return (
            method_key,
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
        method_key = "pf" if output_family == "symmetric" else f"pf_a{alpha_tag}"
        return (
            method_key,
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


def save_input_plots(init, labels, pseudotime, dir_fig, *, suffix=""):
    plot_categorical_embedding(
        init,
        labels=labels,
        title="Paul15 UMAP init (PAGA initialized)",
        save_path=dir_fig / f"umap{suffix}_clust.pdf",
        s=7,
    )
    plt.close("all")
    plot_continuous_embedding(
        init,
        pseudotime,
        title="Paul15 UMAP init: lineage pseudotime",
        save_path=dir_fig / f"umap{suffix}_pt.pdf",
    )


def save_embedding_plots(method_key, method_title, embedding, labels, pseudotime, dir_fig):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] == 3:
        plot_3d_embedding_views(
            embedding,
            labels=labels,
            title=method_title,
            save_path=dir_fig / f"{method_key}_clust.pdf",
            point_fraction=1.0,
            random_state=42,
            s=7,
        )
        plt.close("all")
        plot_3d_continuous_embedding_views(
            embedding,
            values=pseudotime,
            title=f"{method_title}: lineage pseudotime",
            save_path=dir_fig / f"{method_key}_pt.pdf",
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
        save_path=dir_fig / f"{method_key}_clust.pdf",
        s=7,
    )
    plt.close("all")
    plot_continuous_embedding(
        embedding,
        pseudotime,
        title=f"{method_title}: lineage pseudotime",
        save_path=dir_fig / f"{method_key}_pt.pdf",
    )


if __name__ == "__main__":
    main_paul15_finsler_mds()
