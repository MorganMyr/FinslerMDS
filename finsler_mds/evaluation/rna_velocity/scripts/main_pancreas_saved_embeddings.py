"""Evaluate saved pancreas embeddings with projected scVelo velocities."""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import warnings

import numpy as np

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    cross_boundary_direction_correctness,
    in_cluster_velocity_coherence,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_scvelo_umap import (  # noqa: E402
    PANCREAS_TRANSITIONS,
    _csv_value,
    _neighbor_indices_from_sparse_distances,
    _slug,
)


def main_pancreas_saved_embeddings():
    project_root = Path(__file__).resolve().parents[4]
    embedding_dir = project_root / "scripts" / "res" / "pancreas" / "raw"
    out_dir = project_root / "scripts" / "res" / "pancreas" / "rna_velocity_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    velocity_mode = "dynamical"
    preprocessing = {
        "min_shared_counts": 20,
        "n_top_genes": 2000,
        "n_pcs": 30,
        "n_neighbors": 30,
    }
    umap = {
        "min_dist": 0.5,
        "spread": 1.0,
        "init_pos": "spectral",
    }
    velocity_graph = {
        "n_neighbors": 30,
        "n_jobs": -1,
    }
    dynamical = {
        "recover_dynamics_max_iter": 10,
        "recover_dynamics_n_jobs": -1,
    }
    n_eval_neighbors = 30

    embeddings = _saved_pancreas_embeddings(embedding_dir)
    if not embeddings:
        raise FileNotFoundError(f"No saved pancreas embeddings found in {embedding_dir}.")

    adata = _compute_scvelo_state(
        mode=velocity_mode,
        seed=seed,
        preprocessing=preprocessing,
        umap=umap,
        velocity_graph=velocity_graph,
        dynamical=dynamical,
    )
    labels = np.asarray(adata.obs["clusters"].to_numpy(), dtype=str)
    expression_neighbors = _neighbor_indices_from_sparse_distances(
        adata.obsp["distances"],
        n_neighbors=n_eval_neighbors,
    )

    rows = []
    for info in embeddings:
        embedding = _load_embedding(info["path"])
        if embedding.shape[0] != adata.n_obs:
            print(
                f"Skipping {info['path'].name}: {embedding.shape[0]} cells, "
                f"expected {adata.n_obs}."
            )
            continue
        if embedding.shape[1] not in {2, 3}:
            print(f"Skipping {info['path'].name}: embedding dim {embedding.shape[1]} is not 2 or 3.")
            continue

        print(f"Projecting true scVelo velocities on {info['short_name']}")
        velocity_embedding = _project_velocity_to_embedding(adata, embedding)
        cbdir = cross_boundary_direction_correctness(
            embedding,
            labels,
            PANCREAS_TRANSITIONS,
            velocity_vectors=velocity_embedding,
            neighbor_indices=expression_neighbors,
            n_neighbors=n_eval_neighbors,
        )
        icvcoh = in_cluster_velocity_coherence(
            embedding,
            labels,
            velocity_vectors=velocity_embedding,
            neighbor_indices=expression_neighbors,
            n_neighbors=n_eval_neighbors,
        )

        row = {
            "embedding": info["path"].name,
            "short_name": info["short_name"],
            "family": info["family"],
            "embedding_dim": embedding.shape[1],
            "n_cells": embedding.shape[0],
            "velocity_mode": velocity_mode,
            "velocity_source": "scvelo_velocity_graph_projected_to_embedding",
            "neighbor_space": "expression",
            "eval_neighbors": n_eval_neighbors,
            "cbdir": cbdir.score,
            "icvcoh": icvcoh.score,
        }
        row.update(info["metadata"])
        for edge, score in cbdir.transitions.items():
            source, target = edge
            prefix = _slug(f"cbdir_{source}_to_{target}")
            row[prefix] = score.score
            row[prefix + "_boundary_cells"] = score.n_boundary_cells
            row[prefix + "_neighbor_pairs"] = score.n_neighbor_pairs
        for cluster, score in icvcoh.clusters.items():
            prefix = _slug(f"icvcoh_{cluster}")
            row[prefix] = score.score
            row[prefix + "_cells"] = score.n_cells
            row[prefix + "_neighbor_pairs"] = score.n_neighbor_pairs
        rows.append(row)
        print(f"  CBDir={cbdir.score:.6f}, ICVCoh={icvcoh.score:.6f}")

    out_path = out_dir / "pancreas_saved_embeddings_true_velocity_metrics.csv"
    _write_csv(out_path, rows)
    print(f"Saved metrics: {out_path}")
    return rows


def _compute_scvelo_state(*, mode, seed, preprocessing, umap, velocity_graph, dynamical):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")

    print(f"Computing shared scVelo {mode} state for embedding evaluation")
    adata = scv.datasets.pancreas()
    print(f"  Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=preprocessing["min_shared_counts"],
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=preprocessing["n_top_genes"],
        flavor="seurat",
        subset=True,
    )
    sc.tl.pca(adata, n_comps=preprocessing["n_pcs"], random_state=seed)
    sc.pp.neighbors(
        adata,
        n_neighbors=preprocessing["n_neighbors"],
        n_pcs=preprocessing["n_pcs"],
        random_state=seed,
    )
    scv.pp.moments(
        adata,
        n_neighbors=preprocessing["n_neighbors"],
        n_pcs=preprocessing["n_pcs"],
    )
    if mode == "dynamical":
        scv.tl.recover_dynamics(
            adata,
            max_iter=dynamical["recover_dynamics_max_iter"],
            n_jobs=dynamical["recover_dynamics_n_jobs"],
            show_progress_bar=False,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion of an array with ndim > 0 to a scalar is deprecated.*",
            category=DeprecationWarning,
            module="scvelo.tools.optimization",
        )
        scv.tl.velocity(adata, mode=mode)
    sc.tl.umap(
        adata,
        min_dist=umap["min_dist"],
        spread=umap["spread"],
        init_pos=umap["init_pos"],
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks.*",
            category=DeprecationWarning,
        )
        scv.tl.velocity_graph(
            adata,
            n_neighbors=velocity_graph["n_neighbors"],
            n_jobs=velocity_graph["n_jobs"],
            show_progress_bar=False,
        )
    return adata


def _project_velocity_to_embedding(adata, embedding):
    import scvelo as scv

    adata.obsm["X_eval"] = np.asarray(embedding, dtype=float)
    if "velocity_eval" in adata.obsm:
        del adata.obsm["velocity_eval"]
    scv.tl.velocity_embedding(adata, basis="eval")
    return np.asarray(adata.obsm["velocity_eval"], dtype=float)


def _saved_pancreas_embeddings(embedding_dir):
    paths = []
    paths.extend(embedding_dir.glob("pancreas_umap*.npy"))
    paths.extend(embedding_dir.glob("pancreas_randers_smacof*.npz"))
    paths.extend(embedding_dir.glob("pancreas_path_frozen*.npz"))
    paths.extend(embedding_dir.glob("pancreas_soft_bf*.npz"))

    unique_paths = sorted(set(paths), key=lambda path: (_family_order(path.name), path.name))
    return [_embedding_info(path) for path in unique_paths]


def _embedding_info(path):
    family = _embedding_family(path.name)
    metadata = {}
    if path.suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as cache:
                if "metadata_json" in cache:
                    raw_metadata = json.loads(str(cache["metadata_json"].item()))
                    metadata = _flatten_metadata(raw_metadata)
                if "stress" in cache:
                    metadata["optimizer_stress"] = _scalar_or_nan(cache["stress"])
                if "full_geodesic_stress" in cache:
                    metadata["full_geodesic_stress"] = _scalar_or_nan(cache["full_geodesic_stress"])
        except Exception as exc:  # pragma: no cover - diagnostic path
            metadata = {"metadata_error": str(exc)}
    return {
        "path": path,
        "family": family,
        "short_name": _short_embedding_name(path.name),
        "metadata": metadata,
    }


def _load_embedding(path):
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as cache:
            return np.asarray(cache["embedding"], dtype=float)
    raise ValueError(f"Unsupported embedding file: {path}")


def _embedding_family(name):
    if name.startswith("pancreas_umap"):
        return "umap"
    if name.startswith("pancreas_randers_smacof"):
        return "randers_smacof"
    if name.startswith("pancreas_path_frozen"):
        return "path_frozen"
    if name.startswith("pancreas_soft_bf"):
        return "soft_bf"
    return "other"


def _family_order(name):
    order = {
        "umap": 0,
        "randers_smacof": 1,
        "path_frozen": 2,
        "soft_bf": 3,
        "other": 4,
    }
    return order.get(_embedding_family(name), 99)


def _short_embedding_name(name):
    if name == "pancreas_umap_dynamical_s42.npy":
        return "UMAP 2D"
    if name == "pancreas_umap_3d_dynamical_s42.npy":
        return "UMAP 3D"
    name = name.removeprefix("pancreas_").removesuffix(".npz").removesuffix(".npy")
    name = name.replace("cmatsumoto", "cMats")
    name = name.replace("randers_smacof", "Randers-SMACOF")
    name = name.replace("path_frozen", "path-frozen")
    name = name.replace("soft_bf", "soft-BF")
    name = re.sub(r"_seed\\d+$", "", name)
    return name


def _flatten_metadata(metadata):
    out = {}
    for key in ["optimizer", "init", "seed", "randers_alpha_embedding", "matsumoto_alpha_embedding"]:
        if key in metadata:
            out[f"metadata_{key}"] = metadata[key]
    if isinstance(metadata.get("geodesic_metric"), dict):
        metric = metadata["geodesic_metric"]
        if "kind" in metric:
            out["metadata_geodesic_metric_kind"] = metric["kind"]
        if "alpha" in metric:
            out["metadata_geodesic_metric_alpha"] = metric["alpha"]
    if isinstance(metadata.get("velocity"), dict) and "alpha" in metadata["velocity"]:
        out["metadata_velocity_alpha"] = metadata["velocity"]["alpha"]
        if "distance_formula" in metadata["velocity"]:
            out["metadata_velocity_distance_formula"] = metadata["velocity"]["distance_formula"]
    if isinstance(metadata.get("smacof"), dict) and "max_iter" in metadata["smacof"]:
        out["metadata_smacof_max_iter"] = metadata["smacof"]["max_iter"]
    return out


def _scalar_or_nan(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.nan
    return float(array.reshape(-1)[-1])


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(_csv_value(row.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main_pancreas_saved_embeddings()
