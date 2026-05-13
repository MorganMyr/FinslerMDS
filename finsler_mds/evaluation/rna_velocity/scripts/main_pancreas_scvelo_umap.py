"""Evaluate scVelo pancreas UMAP embeddings with true projected velocities."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import scipy.sparse

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    cross_boundary_direction_correctness,
    in_cluster_velocity_coherence,
)


PANCREAS_TRANSITIONS = [
    ("Ngn3 high EP", "Pre-endocrine"),
    ("Pre-endocrine", "Alpha"),
    ("Pre-endocrine", "Beta"),
    ("Pre-endocrine", "Delta"),
    ("Pre-endocrine", "Epsilon"),
]


def main_pancreas_scvelo_umap():
    project_root = Path(__file__).resolve().parents[4]
    out_dir = project_root / "scripts" / "res" / "pancreas" / "rna_velocity_evaluation"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    force_recompute = False
    modes = ["stochastic", "dynamical"]
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

    rows = []
    for mode in modes:
        print(f"Evaluating scVelo {mode} pancreas UMAP with true velocity_umap")
        cache_path = raw_dir / f"pancreas_scvelo_umap_{mode}_s{seed}.npz"
        cache = None if force_recompute else _load_cache(cache_path)
        if cache is None:
            cache = _compute_scvelo_umap_cache(
                mode=mode,
                seed=seed,
                preprocessing=preprocessing,
                umap=umap,
                velocity_graph=velocity_graph,
                dynamical=dynamical,
            )
            np.savez(cache_path, **cache)
            print(f"Saved true scVelo UMAP cache: {cache_path}")
        else:
            print(f"Loaded true scVelo UMAP cache: {cache_path}")

        mode_rows = _evaluate_cache(
            cache,
            mode=mode,
            n_eval_neighbors=n_eval_neighbors,
            cache_path=cache_path,
        )
        rows.extend(mode_rows)
        for row in mode_rows:
            print(
                f"  neighbor_space={row['neighbor_space']}: "
                f"CBDir={row['cbdir']:.6f}, ICVCoh={row['icvcoh']:.6f}"
            )

    out_path = out_dir / "pancreas_scvelo_umap_true_velocity_metrics.csv"
    _write_csv(out_path, rows)
    print(f"Saved metrics: {out_path}")
    return rows


def _compute_scvelo_umap_cache(*, mode, seed, preprocessing, umap, velocity_graph, dynamical):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")

    print("  Loading scVelo pancreas dataset")
    adata = scv.datasets.pancreas()
    print(f"  Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

    print("  Preprocessing with scVelo-style literature defaults")
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
    print(f"  Preprocessed pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

    if mode == "dynamical":
        print(
            "  Recovering scVelo dynamical model "
            f"(max_iter={dynamical['recover_dynamics_max_iter']})"
        )
        scv.tl.recover_dynamics(
            adata,
            max_iter=dynamical["recover_dynamics_max_iter"],
            n_jobs=dynamical["recover_dynamics_n_jobs"],
            show_progress_bar=False,
        )

    print(f"  Computing scVelo velocity mode={mode}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion of an array with ndim > 0 to a scalar is deprecated.*",
            category=DeprecationWarning,
            module="scvelo.tools.optimization",
        )
        scv.tl.velocity(adata, mode=mode)

    print("  Computing UMAP")
    sc.tl.umap(
        adata,
        min_dist=umap["min_dist"],
        spread=umap["spread"],
        init_pos=umap["init_pos"],
        random_state=seed,
    )

    print("  Computing velocity graph and projected velocity_umap")
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
    scv.tl.velocity_embedding(adata, basis="umap")

    labels = np.asarray(adata.obs["clusters"].to_numpy(), dtype=str)
    X_umap = np.asarray(adata.obsm["X_umap"], dtype=float)
    velocity_umap = np.asarray(adata.obsm["velocity_umap"], dtype=float)
    expression_neighbors = _neighbor_indices_from_sparse_distances(
        adata.obsp["distances"],
        n_neighbors=preprocessing["n_neighbors"],
    )

    metadata = {
        "mode": mode,
        "seed": seed,
        "preprocessing": preprocessing,
        "umap": umap,
        "velocity_graph": velocity_graph,
        "dynamical": dynamical if mode == "dynamical" else None,
        "velocity_source": "scvelo_velocity_umap",
    }
    return {
        "X_umap": X_umap,
        "velocity_umap": velocity_umap,
        "labels": labels,
        "expression_neighbors": expression_neighbors,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def _evaluate_cache(cache, *, mode, n_eval_neighbors, cache_path):
    embedding = np.asarray(cache["X_umap"], dtype=float)
    velocity_umap = np.asarray(cache["velocity_umap"], dtype=float)
    labels = np.asarray(cache["labels"], dtype=str)
    expression_neighbors = np.asarray(cache["expression_neighbors"], dtype=int)

    rows = []
    for neighbor_space, neighbor_indices in [
            ("expression", expression_neighbors),
            ("umap", None),
    ]:
        cbdir = cross_boundary_direction_correctness(
            embedding,
            labels,
            PANCREAS_TRANSITIONS,
            velocity_vectors=velocity_umap,
            n_neighbors=n_eval_neighbors,
            neighbor_indices=neighbor_indices,
        )
        icvcoh = in_cluster_velocity_coherence(
            embedding,
            labels,
            velocity_vectors=velocity_umap,
            n_neighbors=n_eval_neighbors,
            neighbor_indices=neighbor_indices,
        )
        row = {
            "mode": mode,
            "embedding": cache_path.name,
            "velocity_source": "scvelo_velocity_umap",
            "neighbor_space": neighbor_space,
            "n_cells": len(embedding),
            "embedding_dim": embedding.shape[1],
            "eval_neighbors": n_eval_neighbors,
            "cbdir": cbdir.score,
            "icvcoh": icvcoh.score,
        }
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
    return rows


def _neighbor_indices_from_sparse_distances(distances, *, n_neighbors):
    distances = scipy.sparse.csr_matrix(distances)
    n_samples = distances.shape[0]
    neighbors = np.full((n_samples, n_neighbors), -1, dtype=int)
    for row_idx in range(n_samples):
        start, end = distances.indptr[row_idx], distances.indptr[row_idx + 1]
        cols = distances.indices[start:end]
        data = distances.data[start:end]
        keep = cols != row_idx
        cols = cols[keep]
        data = data[keep]
        if len(cols) == 0:
            continue
        order = np.argsort(data, kind="stable")[:n_neighbors]
        cols = cols[order]
        neighbors[row_idx, :len(cols)] = cols
    return neighbors


def _load_cache(path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        required = {"X_umap", "velocity_umap", "labels", "expression_neighbors", "metadata_json"}
        if not required.issubset(cache.files):
            print(f"Cache is incomplete, recomputing: {path}")
            return None
        return {key: cache[key] for key in cache.files}


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(_csv_value(row.get(key, "")) for key in keys) + "\n")


def _csv_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.12g}"
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        text = '"' + text.replace('"', '""') + '"'
    return text


def _slug(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(">", "to")
    )


if __name__ == "__main__":
    main_pancreas_scvelo_umap()
