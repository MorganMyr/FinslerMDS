"""Compare Monocle 3 pseudotime on UMAP vs path-frozen geodesic MDS.

This script is intentionally split between Python and R:

* Python loads Paul15, builds symmetric diffusion-geodesic dissimilarities,
  computes the path-frozen 2D embedding, and makes comparison plots.
* R/Monocle 3 learns the principal graph and pseudotime, first on its own
  UMAP and then on our embedding injected as ``reducedDims(cds)$UMAP``.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as scipy_io
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds, geodesic_embedding_stress
from finsler_mds.utils import plot_continuous_embedding
from finsler_mds.utils.dissimilarity_graphs import density_scaled_knn_distances
from finsler_mds.utils.embedding_io import cache_token, scale_embedding_to_dissimilarities
from finsler_mds.utils.paul15 import cell_scope_cache_tag, restrict_to_lineage_union


def main_paul15_monocle():
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15" / "monocle3"
    dir_raw = dir_res / "raw"
    dir_fig = dir_res / "figures"
    bridge_script = script_dir / "monocle3_bridge" / "run_paul15_monocle3.R"

    include_non_lineage_cells = True
    exclude_19lymph_when_all_cells = True
    extra_excluded_clusters = ["11DC"]
    lineages = {
        "erythrocyte": ["10GMP", "7MEP", "8Mk", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
        "monocyte": ["10GMP", "9GMP", "14Mo", "15Mo"],
    }

    preprocessing = {
        "n_pcs": 20,
        "initial_neighbors": 4,
        "trajectory_neighbors": 10,
        "use_float64": True,
    }
    target_graph = {
        "neighbors": 12,
        "use_rep": "X_diffmap",
        "density_gamma": 0.5,
        "density_mode": "symmetric",
    }
    path_frozen = {
        "graph_neighbors": 20,
        "outer_iter": 1,
        "inner_iter": 2,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 30},
        "n_landmark": 80,
        "landmark_sampling": "random",
        "n_local_pairs": 50,
        "local_pair_mode": "direct",
        "targets_per_landmark": 120,
        "local_global_reweighting": "count",
        "local_weight": 50.0,
        "device": "auto",
        "verbose": 1,
    }
    root_cluster = "7MEP"
    continue_cached_path_frozen = False
    scope_suffix = cell_scope_cache_tag(
        include_non_lineage_cells,
        exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
    ) + excluded_clusters_cache_tag(extra_excluded_clusters)
    target_suffix = f"_k{target_graph['neighbors']}_dg{cache_token(target_graph['density_gamma'])}"
    path_suffix = "_pf_m1_i2_lw50_lm80_t120_ln50"
    input_dir = dir_res / f"monocle_input{scope_suffix}"

    for directory in (dir_raw, dir_fig, input_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inputs_path = dir_raw / (
        f"paul15_monocle_inputs_k{target_graph['neighbors']}"
        f"_dg{cache_token(target_graph['density_gamma'])}{scope_suffix}.npz"
    )
    standard_monocle_path = dir_raw / f"monocle_umap_pseudotime{scope_suffix}.csv"
    standard_graph_prefix = dir_raw / f"monocle_umap_principal_graph{scope_suffix}"
    mds_embedding_path = dir_raw / f"paul15_path_frozen_alpha0_embedding{target_suffix}{path_suffix}{scope_suffix}.npz"
    mds_embedding_csv = dir_raw / f"paul15_path_frozen_alpha0_embedding{target_suffix}{path_suffix}{scope_suffix}.csv"
    mds_monocle_path = dir_raw / f"monocle_mds_geo_pseudotime{target_suffix}{path_suffix}{scope_suffix}.csv"
    mds_graph_prefix = dir_raw / f"monocle_mds_geo_principal_graph{target_suffix}{path_suffix}{scope_suffix}"

    inputs = load_or_build_inputs(
        inputs_path,
        input_dir=input_dir,
        preprocessing=preprocessing,
        target_graph=target_graph,
        lineages=lineages,
        include_non_lineage_cells=include_non_lineage_cells,
        exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
        extra_excluded_clusters=extra_excluded_clusters,
        seed=seed,
    )
    D = inputs["dissimilarities"]
    cell_ids = inputs["cell_ids"].astype(str)
    labels = inputs["labels"].astype(str)

    run_monocle_if_needed(
        bridge_script,
        mode="standard",
        input_dir=input_dir,
        output_csv=standard_monocle_path,
        root_cluster=root_cluster,
        graph_prefix=standard_graph_prefix,
    )
    standard = load_monocle_result(standard_monocle_path, cell_ids)
    standard_graph = load_monocle_graph(standard_graph_prefix)
    umap = np.column_stack([standard["dim1"], standard["dim2"]])
    pt_umap = finite_rescaled(standard["pseudotime"])

    cached_mds_geo = None
    if mds_embedding_path.exists():
        with np.load(mds_embedding_path) as data:
            cached_mds_geo = np.asarray(data["embedding"], dtype=float)
        print(f"Loaded cached path-frozen embedding for warm start: {mds_embedding_path}")

    if cached_mds_geo is not None and not continue_cached_path_frozen:
        mds_geo = cached_mds_geo
    else:
        warm_start = cached_mds_geo if cached_mds_geo is not None else umap
        mds_geo = compute_path_frozen_embedding(
            D,
            warm_start,
            path_frozen=path_frozen,
            seed=seed,
            already_scaled=cached_mds_geo is not None,
        )
        full_stress = geodesic_embedding_stress(
            mds_geo,
            D,
            metric=RandersMetric(alpha=0.0),
            n_neighbors=path_frozen["graph_neighbors"],
            on_unreachable="warn_skip",
        )
        np.savez(
            mds_embedding_path,
            embedding=mds_geo,
            cell_ids=cell_ids,
            full_geodesic_stress=np.asarray(full_stress),
        )
        print(f"Saved path-frozen embedding: {mds_embedding_path}")
        print(f"Path-frozen final full geodesic stress: {full_stress}")

    mds_geo_monocle_scale = rescale_embedding_to_reference(mds_geo, umap)
    save_embedding_csv(mds_embedding_csv, cell_ids, mds_geo_monocle_scale)

    run_monocle_if_needed(
        bridge_script,
        mode="injected",
        input_dir=input_dir,
        output_csv=mds_monocle_path,
        root_cluster=root_cluster,
        embedding_csv=mds_embedding_csv,
        graph_prefix=mds_graph_prefix,
    )
    mds_monocle = load_monocle_result(mds_monocle_path, cell_ids)
    mds_graph = load_monocle_graph(mds_graph_prefix)
    mds_geo_for_plots = np.column_stack([mds_monocle["dim1"], mds_monocle["dim2"]])
    pt_mds = finite_rescaled(mds_monocle["pseudotime"])

    save_four_pseudotime_plots(
        umap,
        mds_geo_for_plots,
        pt_umap,
        pt_mds,
        dir_fig=dir_fig,
    )
    save_principal_graph_plots(
        umap,
        mds_geo_for_plots,
        pt_umap,
        pt_mds,
        standard_graph=standard_graph,
        mds_graph=mds_graph,
        dir_fig=dir_fig,
    )
    save_cluster_reference_plots(
        umap,
        mds_geo_for_plots,
        labels,
        standard_graph=standard_graph,
        mds_graph=mds_graph,
        dir_fig=dir_fig,
    )
    print(f"Saved Monocle 3 comparison figures in: {dir_fig}")


def load_or_build_inputs(
    path,
    *,
    input_dir,
    preprocessing,
    target_graph,
    lineages,
    include_non_lineage_cells,
    exclude_19lymph_when_all_cells,
    extra_excluded_clusters,
    seed,
):
    export_files = [
        input_dir / "expression_gene_by_cell.mtx",
        input_dir / "cell_metadata.csv",
        input_dir / "gene_metadata.csv",
    ]
    if path.exists() and all(file.exists() for file in export_files):
        print(f"Loading cached Paul15 Monocle inputs: {path}")
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}

    print("Loading Scanpy Paul15 mouse hematopoiesis dataset")
    adata_raw = sc.datasets.paul15()
    adata_raw.var_names_make_unique()
    print(f"Raw Paul15 shape: {adata_raw.n_obs} cells x {adata_raw.n_vars} genes")
    adata_raw = restrict_to_lineage_union(
        adata_raw,
        lineages=lineages,
        include_non_lineage_cells=include_non_lineage_cells,
        exclude_19lymph_when_all_cells=exclude_19lymph_when_all_cells,
        cluster_key="paul15_clusters",
    )
    adata_raw = exclude_clusters(adata_raw, extra_excluded_clusters, cluster_key="paul15_clusters")
    print(f"Selected Paul15 shape: {adata_raw.n_obs} cells x {adata_raw.n_vars} genes")
    export_monocle_input(adata_raw, input_dir)

    adata = adata_raw.copy()
    if preprocessing["use_float64"]:
        adata.X = adata.X.astype("float64")

    print("Computing symmetric diffusion-geodesic dissimilarities for path-frozen")
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
    sc.pp.neighbors(
        adata,
        n_neighbors=target_graph["neighbors"],
        use_rep=target_graph["use_rep"],
        random_state=seed,
    )
    target_distances = adata.obsp["distances"].maximum(adata.obsp["distances"].T).tocsr()
    target_distances, density_info = density_scaled_knn_distances(
        target_distances,
        gamma=target_graph["density_gamma"],
        mode=target_graph["density_mode"],
    )
    if target_graph["density_gamma"] != 0:
        print(
            "Applied local-density distance scaling: "
            f"mode={target_graph['density_mode']}, gamma={target_graph['density_gamma']}, "
            f"median_sigma={density_info['median_sigma']:.6g}, "
            f"sigma_range=({density_info['min_sigma']:.6g}, {density_info['max_sigma']:.6g})"
        )
    n_components, component = connected_components(target_distances, directed=False, connection="weak")
    if n_components != 1:
        counts = np.bincount(component)
        raise ValueError(
            "Paul15 target graph is disconnected "
            f"({n_components} components, largest={counts.max()}). "
            "Increase target_graph['neighbors']."
        )
    D = shortest_path(target_distances, directed=False, return_predecessors=False)
    if not np.all(np.isfinite(D)):
        raise ValueError("Paul15 diffusion shortest-path distances contain non-finite values.")
    np.fill_diagonal(D, 0.0)

    inputs = {
        "dissimilarities": np.asarray(D, dtype=float),
        "cell_ids": np.asarray(adata_raw.obs_names.astype(str), dtype=str),
        "labels": np.asarray(adata_raw.obs["paul15_clusters"].astype(str), dtype=str),
    }
    np.savez(path, **inputs)
    print(f"Saved Paul15 Monocle inputs: {path}")
    return inputs


def exclude_clusters(adata, clusters, *, cluster_key):
    clusters = [str(cluster) for cluster in clusters]
    if not clusters:
        return adata

    labels = np.asarray(adata.obs[cluster_key].astype(str))
    mask = ~np.isin(labels, clusters)
    removed = int(np.sum(~mask))
    print(
        "Excluding Paul15 clusters "
        f"{', '.join(clusters)}: removed {removed} / {adata.n_obs} cells"
    )
    return adata[mask].copy()


def excluded_clusters_cache_tag(clusters):
    clusters = [str(cluster).lower() for cluster in clusters]
    if not clusters:
        return ""
    safe = ["".join(char if char.isalnum() else "" for char in cluster) for cluster in clusters]
    return "_no" + "_no".join(safe)


def export_monocle_input(adata, input_dir):
    input_dir.mkdir(parents=True, exist_ok=True)
    X = adata.X
    if sparse.issparse(X):
        expression = X.T.tocoo()
    else:
        expression = sparse.coo_matrix(np.asarray(X).T)

    matrix_path = input_dir / "expression_gene_by_cell.mtx"
    cell_path = input_dir / "cell_metadata.csv"
    gene_path = input_dir / "gene_metadata.csv"

    scipy_io.mmwrite(matrix_path, expression)
    pd.DataFrame(
        {
            "cell_id": np.asarray(adata.obs_names.astype(str), dtype=str),
            "paul15_clusters": np.asarray(adata.obs["paul15_clusters"].astype(str), dtype=str),
        }
    ).to_csv(cell_path, index=False)
    pd.DataFrame(
        {
            "gene_id": np.asarray(adata.var_names.astype(str), dtype=str),
            "gene_short_name": np.asarray(adata.var_names.astype(str), dtype=str),
        }
    ).to_csv(gene_path, index=False)
    print(f"Exported Monocle 3 input files in: {input_dir}")


def compute_path_frozen_embedding(D, init_embedding, *, path_frozen, seed, already_scaled=False):
    metric = RandersMetric(alpha=0.0)
    if already_scaled:
        init = np.asarray(init_embedding, dtype=float)
        print("Continuing path-frozen from cached embedding.")
    else:
        init, scale = scale_embedding_to_dissimilarities(init_embedding, D, random_state=seed)
        print(f"Rescaled Monocle UMAP init by factor {scale:.6g}.")
    print("Running alpha=0 path-frozen geodesic MDS")
    embedding, stress = fit_finsler_mds(
        D,
        metric=metric,
        optimizer="path_frozen",
        init=init,
        n_components=2,
        mask_random_state=seed,
        target_random_state=seed + 3,
        print_time=True,
        **path_frozen,
    )
    print(f"Path-frozen optimizer stress: {stress}")
    return embedding


def run_monocle_if_needed(
    bridge_script,
    *,
    mode,
    input_dir,
    output_csv,
    root_cluster,
    embedding_csv=None,
    graph_prefix=None,
    monocle_graph=None,
):
    outputs = [output_csv]
    if graph_prefix is not None:
        outputs.extend([Path(f"{graph_prefix}_nodes.csv"), Path(f"{graph_prefix}_edges.csv")])

    if output_csv.exists() and all(path.exists() for path in outputs):
        dependencies = [bridge_script]
        dependencies.extend(
            input_dir / name
            for name in ("expression_gene_by_cell.mtx", "cell_metadata.csv", "gene_metadata.csv")
        )
        if embedding_csv is not None:
            dependencies.append(embedding_csv)
        newest_dependency = max(Path(path).stat().st_mtime for path in dependencies if Path(path).exists())
        if output_csv.stat().st_mtime >= newest_dependency:
            print(f"Loading cached Monocle 3 {mode} result: {output_csv}")
            return
        print(f"Cached Monocle 3 {mode} result is stale; recomputing: {output_csv}")
    elif output_csv.exists():
        print(f"Cached Monocle 3 {mode} result is missing graph export; recomputing: {output_csv}")

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript was not found. Install R + Monocle 3, then rerun this script. "
            f"Bridge script: {bridge_script}"
        )

    command = [
        rscript,
        str(bridge_script),
        mode,
        str(input_dir),
        str(output_csv),
        root_cluster,
    ]
    if embedding_csv is not None:
        command.append(str(embedding_csv))
    if graph_prefix is not None:
        command.append(str(graph_prefix))
    if monocle_graph is not None:
        if graph_prefix is None:
            command.append("NA")
        command.extend(monocle_graph_cli_args(monocle_graph))

    print("Running Monocle 3:", " ".join(command))
    subprocess.run(command, check=True)


def monocle_graph_cli_args(options):
    options = {} if options is None else dict(options)
    control = options.get("learn_graph_control", {}) or {}
    control_arg = ";".join(f"{key}={value}" for key, value in control.items())
    return [
        str(options.get("cluster_k", 20)),
        str(options.get("partition_qval", 0.05)),
        bool_cli(options.get("use_partition", True)),
        bool_cli(options.get("close_loop", True)),
        control_arg,
    ]


def bool_cli(value):
    return "TRUE" if bool(value) else "FALSE"


def load_monocle_result(path, cell_ids):
    table = pd.read_csv(path)
    required = {"cell_id", "dim1", "dim2", "pseudotime"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Monocle result {path} is missing columns: {sorted(missing)}")

    table = table.set_index("cell_id")
    missing_cells = [cell_id for cell_id in cell_ids if cell_id not in table.index]
    if missing_cells:
        raise ValueError(f"Monocle result {path} is missing cell {missing_cells[0]!r}.")
    table = table.loc[cell_ids]
    return {
        "dim1": table["dim1"].to_numpy(dtype=float),
        "dim2": table["dim2"].to_numpy(dtype=float),
        "pseudotime": table["pseudotime"].to_numpy(dtype=float),
    }


def load_monocle_graph(prefix):
    nodes_path = Path(f"{prefix}_nodes.csv")
    edges_path = Path(f"{prefix}_edges.csv")
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing Monocle principal graph export for prefix {prefix}")
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    required_nodes = {"node_id", "dim1", "dim2", "is_root"}
    required_edges = {"from", "to"}
    missing_nodes = required_nodes.difference(nodes.columns)
    missing_edges = required_edges.difference(edges.columns)
    if missing_nodes or missing_edges:
        raise ValueError(
            f"Invalid Monocle graph export {prefix}: "
            f"missing node columns={sorted(missing_nodes)}, edge columns={sorted(missing_edges)}"
        )
    return {"nodes": nodes, "edges": edges}


def save_embedding_csv(path, cell_ids, embedding):
    pd.DataFrame(
        {
            "cell_id": np.asarray(cell_ids, dtype=str),
            "dim1": np.asarray(embedding[:, 0], dtype=float),
            "dim2": np.asarray(embedding[:, 1], dtype=float),
        }
    ).to_csv(path, index=False)


def rescale_embedding_to_reference(embedding, reference):
    embedding = np.asarray(embedding, dtype=float)
    reference = np.asarray(reference, dtype=float)
    centered_embedding = embedding - embedding.mean(axis=0, keepdims=True)
    centered_reference = reference - reference.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered_reference) / np.linalg.norm(centered_embedding)
    return centered_embedding * scale + reference.mean(axis=0, keepdims=True)


def save_four_pseudotime_plots(umap, mds_geo, pt_umap, pt_mds, *, dir_fig):
    panels = [
        (umap, pt_umap, "UMAP coordinates, Monocle-UMAP pseudotime", "umap_with_umap_pseudotime.pdf"),
        (umap, pt_mds, "UMAP coordinates, Monocle-MDS pseudotime", "umap_with_mds_geo_pseudotime.pdf"),
        (mds_geo, pt_umap, "MDS-geodesic coordinates, Monocle-UMAP pseudotime", "mds_geo_with_umap_pseudotime.pdf"),
        (mds_geo, pt_mds, "MDS-geodesic coordinates, Monocle-MDS pseudotime", "mds_geo_with_mds_geo_pseudotime.pdf"),
    ]
    for embedding, values, title, filename in panels:
        plot_continuous_embedding(
            embedding,
            values,
            title=title,
            save_path=dir_fig / filename,
            s=8,
        )
        plt.close("all")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (embedding, values, title, _) in zip(axes.ravel(), panels):
        plot_continuous_embedding(
            embedding,
            values,
            title=title,
            fig_ax=(fig, ax),
            s=8,
        )
    fig.tight_layout()
    fig.savefig(dir_fig / "monocle_umap_vs_mds_geo_pseudotime_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def save_principal_graph_plots(umap, mds_geo, pt_umap, pt_mds, *, standard_graph, mds_graph, dir_fig):
    graph_panels = [
        (
            umap,
            pt_umap,
            standard_graph,
            "UMAP coordinates, Monocle-UMAP pseudotime + principal graph",
            "umap_with_umap_pseudotime_principal_graph.pdf",
        ),
        (
            mds_geo,
            pt_mds,
            mds_graph,
            "MDS-geodesic coordinates, Monocle-MDS pseudotime + principal graph",
            "mds_geo_with_mds_geo_pseudotime_principal_graph.pdf",
        ),
    ]
    for embedding, values, graph, title, filename in graph_panels:
        fig, ax = plot_continuous_embedding(embedding, values, title=title, s=8)
        overlay_principal_graph(ax, graph)
        fig.savefig(dir_fig / filename, bbox_inches="tight")
        plt.close(fig)


def overlay_principal_graph(ax, graph, *, edge_color="black", node_color="white", root_color="crimson"):
    nodes = graph["nodes"]
    edges = graph["edges"]
    coords = nodes.set_index("node_id")[["dim1", "dim2"]]

    for _, edge in edges.iterrows():
        source = edge["from"]
        target = edge["to"]
        if source not in coords.index or target not in coords.index:
            continue
        segment = coords.loc[[source, target]].to_numpy(dtype=float)
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=edge_color,
            lw=1.2,
            alpha=0.82,
            zorder=4,
        )

    ax.scatter(
        nodes["dim1"],
        nodes["dim2"],
        s=14,
        c=node_color,
        edgecolors=edge_color,
        linewidths=0.55,
        zorder=5,
    )
    root_nodes = nodes[np.asarray(nodes["is_root"], dtype=bool)]
    if len(root_nodes):
        ax.scatter(
            root_nodes["dim1"],
            root_nodes["dim2"],
            s=36,
            c=root_color,
            marker="*",
            edgecolors="black",
            linewidths=0.45,
            zorder=6,
            label="root graph nodes",
        )
        ax.legend(loc="best", fontsize=8)


def save_cluster_reference_plots(umap, mds_geo, labels, *, standard_graph=None, mds_graph=None, dir_fig):
    from finsler_mds.utils import plot_categorical_embedding

    display_labels = paul15_display_labels(labels)

    fig, ax = plot_categorical_embedding(
        umap,
        labels=display_labels,
        title="Paul15 Monocle UMAP clusters",
        save_path=dir_fig / "umap_clusters.pdf",
        s=8,
    )
    if standard_graph is not None:
        overlay_principal_graph(ax, standard_graph)
        fig.savefig(dir_fig / "umap_clusters.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    fig, ax = plot_categorical_embedding(
        mds_geo,
        labels=display_labels,
        title="Paul15 MDS-geodesic clusters",
        save_path=dir_fig / "mds_geo_clusters.pdf",
        s=8,
    )
    if mds_graph is not None:
        overlay_principal_graph(ax, mds_graph)
        fig.savefig(dir_fig / "mds_geo_clusters.pdf", bbox_inches="tight")
    plt.close(fig)


def paul15_display_labels(labels):
    """Map Scanpy Paul15 cluster labels to the coarser labels used by Monocle docs."""
    mapping = {
        "1Ery": "Erythrocyte",
        "2Ery": "Erythrocyte",
        "3Ery": "Erythrocyte",
        "4Ery": "Erythrocyte",
        "5Ery": "Erythrocyte",
        "6Ery": "Erythrocyte",
        "7MEP": "Multipotent progenitors",
        "8Mk": "Megakaryocytes",
        "9GMP": "GMP",
        "10GMP": "GMP",
        "11DC": "Dendritic cells",
        "12Baso": "Basophils",
        "13Baso": "Basophils",
        "14Mo": "Monocytes",
        "15Mo": "Monocytes",
        "16Neu": "Neutrophils",
        "17Neu": "Neutrophils",
        "18Eos": "Eosinophils",
        "19Lymph": "Lymphoid",
    }
    return np.asarray([mapping.get(str(label), str(label)) for label in labels], dtype=str)


def finite_rescaled(values):
    values = np.asarray(values, dtype=float)
    out = values.copy()
    out[~np.isfinite(out)] = np.nan
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    lo = float(np.nanmin(out[finite]))
    hi = float(np.nanmax(out[finite]))
    out[finite] = (out[finite] - lo) / (hi - lo) if hi > lo else 0.0
    return out


if __name__ == "__main__":
    main_paul15_monocle()
