"""Create/evaluate pancreas gap-distance embeddings.

This script owns the trajectory-gap experiment. ``main_pancreas.py`` remains a
plain embedding generator for the full pancreas dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import fit_finsler_mds  # noqa: E402
from finsler_mds import utils  # noqa: E402
from finsler_mds.evaluation.rna_velocity import normalized_gap_distance  # noqa: E402
from finsler_mds.evaluation.rna_velocity.pancreas_gap import (  # noqa: E402
    PancreasGapSelection,
    gap_arrays_to_cache,
    normalize_pancreas_gap_config,
    pancreas_gap_prefix,
    select_pancreas_gap,
)
from finsler_mds.utils.pancreas import (  # noqa: E402
    PANCREAS_DATASET_SOURCE,
    compute_pancreas_velocity,
    compute_pancreas_velocity_graph,
    load_pancreas_dataset,
    preprocess_pancreas_for_velocity,
    setup_scvelo_settings,
)
from finsler_mds.utils.pancreas_files import (  # noqa: E402
    pancreas_raw_embedding_path,
    pancreas_velocity_cache_path,
)
from scripts.main_pancreas import (  # noqa: E402
    cache_token,
    compute_umap_from_neighbors,
    labels_to_cache,
    labels_to_numpy,
    load_saved_embedding,
    load_velocity_inputs_cache,
    main_pancreas,
    make_embedding_metric,
    normalize_embedding_dim,
    normalize_embedding_metric_kind,
    normalize_velocity_distance_formula,
    orient_pancreas_embedding_by_velocity,
    pancreas_cache_metadata,
    pancreas_file_prefix,
    pancreas_state_cache_matches,
    pancreas_state_cache_metadata,
    pancreas_state_cache_path,
    pancreas_umap_embedding_path,
    pancreas_umap_variant_tag,
    project_velocity_to_pca,
    save_pancreas_state_cache,
    velocity_distance_formula_tag,
)


SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
PANCREAS_DIR = SCRIPT_DIR / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
GAP_RAW_DIR = PANCREAS_DIR / "gap" / "raw"
OUTPUT_CSV = PANCREAS_DIR / "rna_velocity_evaluation" / "pancreas_gap_distance.csv"

PREPROCESSING = {
    "min_shared_counts": 20,
    "n_top_genes": 3000,
    "n_pcs": 50,
    "moments_n_neighbors": 30,
}
VELOCITY = {
    "mode": "dynamical",
    "distance_formula": "randers",
    "alpha": 0.35,
    "cos_clip": 0.4,
    "velocity_neighbors": 30,
    "kNN_euclid": 30,
    "kNN_finsler": 0,
    "average_velocity": True,
    "symmetrize_support": True,
    "graph_n_jobs": -1,
    "recover_dynamics_max_iter": 20,
    "recover_dynamics_n_jobs": -1,
}
UMAP = {
    "n_neighbors": 50,
    "min_dist": 0.5,
    "spread": 1.0,
    "maxiter": 1500,
    "negative_sample_rate": 10,
    "init_pos": "spectral",
}
GAP = {
    "enabled": True,
    "name": "preendocrine",
    "selection": "veloviz_latent_time",
    "n_before": 300,
    "n_after": 300,
    "removed_labels": ("Pre-endocrine",),
    "before_labels": ("Ngn3 high EP",),
    "after_labels": ("Alpha", "Beta", "Delta", "Epsilon"),
}
GRADIENT_DESCENT = {
    "max_iter": 300,
    "eps": 1e-8,
    "method": "L-BFGS-B",
    "optimizer_options": {"ftol": 1e-10, "maxls": 80, "maxcor": 30},
    "device": "auto",
    "gpu_block_size": 256,
    "verbose": 1,
}


def main():
    args = parse_args()
    gap = normalize_pancreas_gap_config(GAP)
    velocity = {**VELOCITY, "alpha": args.velocity_alpha, "cos_clip": args.velocity_cos_clip}
    embedding_dim = normalize_embedding_dim(args.embedding_dim)
    method = normalize_method(args.method)
    metric = make_embedding_metric({"kind": args.finsler_metric, "alpha": args.alpha_embedding})

    full_embedding, full_selection = ensure_embedding(
        "full",
        method=method,
        embedding_dim=embedding_dim,
        velocity=velocity,
        metric_kind=args.finsler_metric,
        alpha_embedding=args.alpha_embedding,
    )
    gap_embedding, gap_selection = ensure_embedding(
        "gap",
        method=method,
        embedding_dim=embedding_dim,
        velocity=velocity,
        metric_kind=args.finsler_metric,
        alpha_embedding=args.alpha_embedding,
    )
    rows = [
        evaluate_scope(
            full_embedding,
            full_selection.before_original_indices,
            full_selection.after_original_indices,
            scope="full",
            method=method,
            embedding_dim=embedding_dim,
            velocity=velocity,
            metric=metric,
            alpha_embedding=args.alpha_embedding,
            gap=gap,
        ),
        evaluate_scope(
            gap_embedding,
            gap_selection.before_indices,
            gap_selection.after_indices,
            scope="gap",
            method=method,
            embedding_dim=embedding_dim,
            velocity=velocity,
            metric=metric,
            alpha_embedding=args.alpha_embedding,
            gap=gap,
        ),
    ]
    for row in rows:
        print(
            f"{row['method']} {row['scope']}: normalized_gap={float(row['normalized_gap_distance']):.6f} "
            f"(raw={float(row['gap_distance']):.6f}, n={row['n_cells']})"
        )
    upsert_csv(OUTPUT_CSV, rows, key_columns=csv_key_columns())
    print(f"Saved gap-distance rows: {OUTPUT_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("umap", "gradient_descent", "gd"), default="umap")
    parser.add_argument("--embedding-dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--velocity-alpha", type=float, default=VELOCITY["alpha"])
    parser.add_argument("--velocity-cos-clip", type=float, default=VELOCITY["cos_clip"])
    parser.add_argument("--finsler-metric", default="randers")
    parser.add_argument("--alpha-embedding", type=float, default=0.5)
    return parser.parse_args()


def normalize_method(method):
    method = method.lower()
    if method == "gd":
        return "gradient_descent"
    if method not in {"umap", "gradient_descent"}:
        raise ValueError("method must be 'umap' or 'gradient_descent'.")
    return method


def ensure_embedding(scope, *, method, embedding_dim, velocity, metric_kind, alpha_embedding):
    if scope == "full":
        selection = full_gap_selection(velocity)
        if method == "umap":
            return ensure_full_umap(embedding_dim, velocity), selection
        return ensure_full_gradient_descent(embedding_dim, velocity, metric_kind, alpha_embedding), selection
    if scope == "gap":
        x_umap, dists, labels, _, _, selection = load_or_create_gap_inputs(velocity, embedding_dim)
        if method == "umap":
            return x_umap, selection
        return ensure_gap_gradient_descent(x_umap, dists, labels, embedding_dim, velocity, metric_kind, alpha_embedding), selection
    raise ValueError("scope must be 'full' or 'gap'.")


def full_gap_selection(velocity):
    adata = load_full_state_for_selection(velocity)
    labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    ordering = None
    gap = normalize_pancreas_gap_config(GAP)
    if gap["selection"] in {"latent_time", "veloviz_latent_time"}:
        ordering = load_or_compute_pancreas_gap_ordering(
            GAP_RAW_DIR / f"{pancreas_gap_prefix(gap)}_selection_s{SEED}.npz",
            preprocessing=PREPROCESSING,
            velocity=velocity,
            gap=gap,
            seed=SEED,
        )
    return select_pancreas_gap(labels, gap, cell_ids=np.asarray(adata.obs_names, dtype=str), ordering=ordering)


def ensure_full_umap(embedding_dim, velocity):
    cache_tag = f"{cache_token(velocity['mode'])}_{pancreas_umap_variant_tag(UMAP)}s{SEED}"
    path = pancreas_umap_embedding_path(RAW_DIR, cache_tag, n_components=embedding_dim, dataset_prefix="pancreas")
    if not Path(path).exists():
        main_pancreas({
            "finsler_optimizer": None,
            "init_finsler_mds": f"umap_{embedding_dim}D",
            "velocity": velocity,
            "embedding_dim": embedding_dim,
            "umap": UMAP,
        })
    return load_saved_embedding(path)


def ensure_full_gradient_descent(embedding_dim, velocity, metric_kind, alpha_embedding):
    output_path = gradient_descent_path(RAW_DIR, "pancreas", embedding_dim, velocity, metric_kind, alpha_embedding)
    if not output_path.exists():
        main_pancreas({
            "finsler_optimizer": "gradient_descent",
            "init_finsler_mds": "umap_2D",
            "embedding_dim": embedding_dim,
            "velocity": velocity,
            "umap": UMAP,
            "finsler_metric": metric_kind,
            "alpha_embedding": alpha_embedding,
            "gradient_descent": GRADIENT_DESCENT,
            "cluster_reweight_rho": 0,
            "frontier_pairs_weight": 1,
            "distance_reweighting": {"power": 0, "epsilon": 1e-6},
        })
    return load_saved_embedding(output_path)


def load_or_create_gap_inputs(velocity, embedding_dim):
    GAP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = gap_velocity_cache_path(velocity)
    metadata = velocity_cache_metadata(velocity, gap=GAP, embedding_dim=embedding_dim)
    if cache_path.exists():
        cached = load_gap_inputs_cache(cache_path, metadata)
        if cached is not None:
            return cached

    import scanpy as sc

    setup_scvelo_settings()
    state_path = pancreas_state_cache_path(
        GAP_RAW_DIR,
        preprocessing=PREPROCESSING,
        velocity=velocity,
        dataset_prefix=pancreas_gap_prefix(GAP),
        seed=SEED,
    )
    state_metadata = pancreas_state_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=PREPROCESSING["min_shared_counts"],
        n_top_genes=PREPROCESSING["n_top_genes"],
        n_pcs=PREPROCESSING["n_pcs"],
        moments_n_neighbors=PREPROCESSING["moments_n_neighbors"],
        velocity_mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        gap=normalize_pancreas_gap_config(GAP),
        seed=SEED,
    )
    adata = load_gap_state_cache(state_path, state_metadata)
    if adata is None:
        print(f"Loading pancreas dataset from {PANCREAS_DATASET_SOURCE} for gap experiment")
        adata_raw = load_pancreas_dataset()
        selection = full_gap_selection(velocity)
        print(
            f"Applying pancreas gap {selection.config['name']!r}: removed "
            f"{selection.removed_mask.sum()} cells; kept {selection.keep_mask.sum()} cells."
        )
        adata = adata_raw[selection.keep_mask].copy()
        labels = selection.labels
        cell_ids = selection.cell_ids
        original_indices = selection.original_indices
        preprocess_pancreas_for_velocity(adata, PREPROCESSING, seed=SEED)
        compute_pancreas_velocity(
            adata,
            mode=velocity["mode"],
            recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
            recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
        )
        save_pancreas_state_cache(
            state_path,
            adata,
            labels=labels,
            cell_ids=cell_ids,
            original_indices=original_indices,
            metadata=state_metadata,
        )
    else:
        selection = full_gap_selection(velocity)
        labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
        cell_ids = np.asarray(adata.obs_names, dtype=str)
        original_indices = np.asarray(adata.obs["finsler_mds_original_index"], dtype=int)

    sc.pp.neighbors(adata, n_neighbors=UMAP["n_neighbors"], n_pcs=PREPROCESSING["n_pcs"], random_state=SEED)
    x_umap = compute_umap_from_neighbors(adata, UMAP, n_components=embedding_dim, random_state=SEED)
    x_umap = orient_pancreas_embedding_by_velocity(adata, x_umap, velocity=velocity, label=f"gap {embedding_dim}D UMAP")
    x_pca = np.asarray(adata.obsm["X_pca"][:, :PREPROCESSING["n_pcs"]], dtype=float)
    velocity_pca = project_velocity_to_pca(adata, PREPROCESSING["n_pcs"])
    dists, _, _, _ = utils.compute_velocity_dist_matrix(
        x_pca,
        velocity_pca,
        kNN_euclid=velocity["kNN_euclid"],
        kNN_finsler=velocity["kNN_finsler"],
        alpha=velocity["alpha"],
        distance_formula=velocity["distance_formula"],
        cos_clip=velocity["cos_clip"],
        velocity_neighbors=velocity["velocity_neighbors"],
        average_velocity=velocity["average_velocity"],
        symmetrize_support=velocity["symmetrize_support"],
        n_jobs=velocity["graph_n_jobs"],
    )
    np.savez(
        cache_path,
        x_umap=x_umap,
        dists_velocity=dists,
        labels=labels_to_cache(labels),
        cell_ids=np.asarray(cell_ids, dtype=str),
        original_indices=np.asarray(original_indices, dtype=int),
        **gap_arrays_to_cache(selection),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Saved gap velocity cache: {cache_path}")
    return x_umap, dists, labels, cell_ids, original_indices, selection


def load_gap_state_cache(cache_path, expected_metadata):
    if not Path(cache_path).exists():
        return None
    import scanpy as sc

    adata = sc.read_h5ad(cache_path)
    if not pancreas_state_cache_matches(adata, expected_metadata):
        return None
    return adata


def load_gap_inputs_cache(cache_path, expected_metadata):
    cached = load_velocity_inputs_cache(cache_path, expected_metadata)
    if cached is None:
        return None
    x_umap, dists, labels, cell_ids, original_indices = cached
    with np.load(cache_path, allow_pickle=False) as cache:
        selection = cached_gap_selection(cache, labels, cell_ids, original_indices)
    return x_umap, dists, labels, cell_ids, original_indices, selection


def cached_gap_selection(cache, labels, cell_ids, original_indices):
    original_indices = np.asarray(original_indices, dtype=int)
    removed_original = np.asarray(cache["gap_removed_original_indices"], dtype=int)
    n_full = int(max(np.max(original_indices, initial=-1), np.max(removed_original, initial=-1)) + 1)
    keep_mask = np.zeros(n_full, dtype=bool)
    keep_mask[original_indices] = True
    removed_mask = np.zeros(n_full, dtype=bool)
    removed_mask[removed_original] = True
    return PancreasGapSelection(
        config=normalize_pancreas_gap_config(GAP),
        labels=np.asarray(labels, dtype=str),
        cell_ids=np.asarray(cell_ids, dtype=str),
        keep_mask=keep_mask,
        removed_mask=removed_mask,
        original_indices=original_indices,
        before_indices=np.asarray(cache["gap_before_indices"], dtype=int),
        after_indices=np.asarray(cache["gap_after_indices"], dtype=int),
        before_original_indices=np.asarray(cache["gap_before_original_indices"], dtype=int),
        after_original_indices=np.asarray(cache["gap_after_original_indices"], dtype=int),
        ordering=np.asarray(cache["gap_ordering_full"], dtype=float) if "gap_ordering_full" in cache else None,
    )


def ensure_gap_gradient_descent(x_umap, dists, labels, embedding_dim, velocity, metric_kind, alpha_embedding):
    output_path = gradient_descent_path(GAP_RAW_DIR, pancreas_gap_prefix(GAP), embedding_dim, velocity, metric_kind, alpha_embedding)
    if output_path.exists():
        return load_saved_embedding(output_path)
    metric = make_embedding_metric({"kind": metric_kind, "alpha": alpha_embedding})
    init = np.zeros((len(x_umap), embedding_dim), dtype=float)
    init[:, :x_umap.shape[1]] = x_umap[:, :min(x_umap.shape[1], embedding_dim)]
    embedding, stress, n_iter = fit_finsler_mds(
        dists,
        metric=metric,
        optimizer="gradient_descent",
        init=init,
        n_components=embedding_dim,
        return_n_iter=True,
        random_state=SEED,
        print_time=True,
        **GRADIENT_DESCENT,
    )
    np.savez(
        output_path,
        embedding=embedding,
        stress=np.asarray(stress),
        init_finsler_mds=np.asarray("gap_umap"),
        labels=labels_to_cache(labels),
        metadata_json=json.dumps(
            {
                "optimizer": "gradient_descent",
                "scope": "gap",
                "embedding_dim": embedding_dim,
                "seed": SEED,
                "velocity": velocity,
                "finsler_metric": normalize_embedding_metric_kind(metric_kind),
                "alpha_embedding": alpha_embedding,
                "n_iter": int(n_iter),
                "gradient_descent": GRADIENT_DESCENT,
            },
            sort_keys=True,
        ),
    )
    print(f"Saved gap gradient-descent embedding: {output_path}")
    return embedding


def load_full_state_for_selection(velocity):
    state_path = pancreas_state_cache_path(
        RAW_DIR,
        preprocessing=PREPROCESSING,
        velocity=velocity,
        dataset_prefix="pancreas",
        seed=SEED,
    )
    if not state_path.exists():
        main_pancreas({"finsler_optimizer": None, "velocity": velocity, "umap": UMAP})
    import scanpy as sc

    return sc.read_h5ad(state_path)


def load_or_compute_pancreas_gap_ordering(cache_path, *, preprocessing, velocity, gap, seed):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = pancreas_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        preprocessing=preprocessing,
        velocity_mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        gap=normalize_pancreas_gap_config(gap),
        seed=seed,
    )
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if "metadata_json" in cache and "latent_time" in cache:
                if json.loads(str(cache["metadata_json"].item())) == metadata:
                    print(f"Loaded pancreas gap latent-time selection cache: {cache_path}")
                    return np.asarray(cache["latent_time"], dtype=float)

    print("Computing full-pancreas latent time for VeloViz-like gap selection")
    adata = load_pancreas_dataset()
    preprocess_pancreas_for_velocity(adata, preprocessing, seed=seed)
    compute_pancreas_velocity(
        adata,
        mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
    )
    compute_pancreas_velocity_graph(adata, n_jobs=velocity["graph_n_jobs"], show_progress_bar=False)
    import scvelo as scv

    scv.tl.latent_time(adata)
    latent_time = np.asarray(adata.obs["latent_time"], dtype=float)
    np.savez(
        cache_path,
        latent_time=latent_time,
        labels=labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None),
        cell_ids=np.asarray(adata.obs_names, dtype=str),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Saved pancreas gap latent-time selection cache: {cache_path}")
    return latent_time


def gradient_descent_path(raw_dir, dataset_prefix, embedding_dim, velocity, metric_kind, alpha_embedding):
    file_prefix = pancreas_file_prefix(dataset_prefix)
    velocity_tag = velocity_distance_formula_tag(velocity["distance_formula"], alpha=velocity["alpha"])
    metric = make_embedding_metric({"kind": metric_kind, "alpha": alpha_embedding})
    metric_tag = {"randers": "r", "matsumoto": "mats", "convexified_matsumoto": "cmats"}[
        normalize_embedding_metric_kind(metric_kind)
    ] + cache_token(metric.alpha)
    name = f"{file_prefix}gd_{embedding_dim}d_{velocity_tag}_{metric_tag}_s{SEED}.npz"
    return pancreas_raw_embedding_path(raw_dir, name)


def gap_velocity_cache_path(velocity):
    tag = (
        f"{cache_token(velocity['mode'])}_"
        f"{velocity_distance_formula_tag(velocity['distance_formula'])}_"
        f"valpha{cache_token(velocity['alpha'])}_"
        f"cclip{cache_token(velocity['cos_clip'])}_"
        f"ke{velocity['kNN_euclid']}_kf{velocity['kNN_finsler']}_"
        f"{pancreas_umap_variant_tag(UMAP)}s{SEED}"
    )
    return pancreas_velocity_cache_path(GAP_RAW_DIR, f"{pancreas_gap_prefix(GAP)}_velocity_inputs_{tag}.npz")


def velocity_cache_metadata(velocity, *, gap, embedding_dim):
    return pancreas_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=PREPROCESSING["min_shared_counts"],
        n_top_genes=PREPROCESSING["n_top_genes"],
        n_pcs=PREPROCESSING["n_pcs"],
        moments_n_neighbors=PREPROCESSING["moments_n_neighbors"],
        velocity_mode=velocity["mode"],
        velocity_distance_formula=normalize_velocity_distance_formula(velocity["distance_formula"]),
        velocity_alpha=velocity["alpha"],
        velocity_cos_clip=velocity["cos_clip"],
        velocity_neighbors=velocity["velocity_neighbors"],
        velocity_kNN_euclid=velocity["kNN_euclid"],
        velocity_kNN_finsler=velocity["kNN_finsler"],
        average_velocity=velocity["average_velocity"],
        symmetrize_velocity_support=velocity["symmetrize_support"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        umap_n_neighbors=UMAP["n_neighbors"],
        umap_min_dist=UMAP["min_dist"],
        umap_spread=UMAP["spread"],
        umap_maxiter=UMAP["maxiter"],
        umap_negative_sample_rate=UMAP["negative_sample_rate"],
        umap_init_pos=UMAP["init_pos"],
        embedding_dim=embedding_dim,
        gap=normalize_pancreas_gap_config(gap),
        seed=SEED,
    )


def evaluate_scope(embedding, before, after, *, scope, method, embedding_dim, velocity, metric, alpha_embedding, gap):
    result = normalized_gap_distance(embedding, before, after)
    return {
        "method": method,
        "scope": scope,
        "n_cells": int(len(embedding)),
        "embedding_dim": int(embedding_dim),
        "velocity_distance_formula": normalize_velocity_distance_formula(velocity["distance_formula"]),
        "velocity_alpha": float(velocity["alpha"]),
        "velocity_cos_clip": float(velocity["cos_clip"]),
        "finsler_metric": normalize_embedding_metric_kind(type(metric).__name__.replace("Metric", "").lower()),
        "alpha_embedding": float(alpha_embedding),
        "gap_name": gap["name"],
        "gap_selection": gap["selection"],
        "n_before_config": int(gap["n_before"]),
        "n_after_config": int(gap["n_after"]),
        "n_before": int(result.n_before),
        "n_after": int(result.n_after),
        "gap_distance": float(result.distance),
        "normalized_gap_distance": float(result.normalized_distance),
        "before_representative": int(result.before_representative),
        "after_representative": int(result.after_representative),
        "max_pairwise_distance": float(result.max_pairwise_distance),
        "seed": SEED,
    }


def csv_key_columns():
    return [
        "method",
        "scope",
        "embedding_dim",
        "velocity_distance_formula",
        "velocity_alpha",
        "velocity_cos_clip",
        "finsler_metric",
        "alpha_embedding",
        "gap_name",
        "gap_selection",
        "n_before_config",
        "n_after_config",
        "seed",
    ]


def upsert_csv(path, rows, *, key_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists() and path.stat().st_size:
        with open(path, newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    by_key = {row_key(row, key_columns): row for row in existing}
    for row in rows:
        by_key[row_key(row, key_columns)] = {key: str(value) for key, value in row.items()}
    all_rows = list(by_key.values())
    fieldnames = sorted({key for row in all_rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def row_key(row, columns):
    return tuple(str(row.get(column, "")) for column in columns)


if __name__ == "__main__":
    main()
