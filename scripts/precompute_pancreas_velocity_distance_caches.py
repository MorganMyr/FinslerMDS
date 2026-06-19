"""Precompute pancreas velocity-distance caches from the heavy state cache.

The expensive preprocessing/RNA-velocity state is stored once as an h5ad cache.
Changing velocity alpha or cos_clip only rebuilds the directed dissimilarity
npz files consumed by main_pancreas.py and the campaign scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import utils  # noqa: E402
from finsler_mds.utils.pancreas import PANCREAS_DATASET_SOURCE  # noqa: E402
from scripts.main_pancreas import (  # noqa: E402
    cache_token,
    labels_to_cache,
    main_pancreas,
    normalize_pancreas_gap_config,
    normalize_velocity_distance_formula,
    pancreas_cache_metadata,
    pancreas_gap_prefix,
    pancreas_state_cache_metadata,
    pancreas_state_cache_path,
    pancreas_umap_variant_tag,
    project_velocity_to_pca,
    load_pancreas_state_cache,
)


SEED = 42
PREPROCESSING = {
    "min_shared_counts": 20,
    "n_top_genes": 3000,
    "n_pcs": 50,
    "moments_n_neighbors": 30,
}
VELOCITY = {
    "mode": "dynamical",
    "distance_formula": "randers",
    "velocity_neighbors": 30,
    "kNN_euclid": 30,
    "kNN_finsler": 0,
    "average_velocity": True,
    "symmetrize_support": True,
    "graph_n_jobs": -1,
    "recover_dynamics_max_iter": 20,
}
UMAP = {
    "n_neighbors": 50,
    "min_dist": 0.5,
    "spread": 1.0,
    "maxiter": 1500,
    "negative_sample_rate": 10,
    "init_pos": "spectral",
}
GAP = normalize_pancreas_gap_config({"enabled": False})

ALPHA_CLIP_GRID = [
    (0.0, 0.4),
    (0.1, 0.4),
    (0.25, 0.4),
    (0.5, 0.4),
    (0.75, 0.4),
    (1.0, 0.4),
    (1.25, 0.4),
    (1.5, 0.4),
    (2.0, 0.4),
    (3.0, 0.3),
]


def main():
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "res" / "pancreas" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    adata, labels, cell_ids, original_indices, gap_arrays = load_or_create_state(raw_dir)
    x_pca = np.asarray(adata.obsm["X_pca"][:, :PREPROCESSING["n_pcs"]], dtype=float)
    velocity_pca = project_velocity_to_pca(adata, PREPROCESSING["n_pcs"])
    x_umap = load_or_create_umap_2d(raw_dir)

    for alpha, cos_clip in ALPHA_CLIP_GRID:
        ensure_distance_cache(
            raw_dir,
            x_pca,
            velocity_pca,
            x_umap,
            labels,
            cell_ids,
            original_indices,
            gap_arrays,
            alpha,
            cos_clip,
        )


def load_or_create_state(raw_dir):
    state_metadata = pancreas_state_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=PREPROCESSING["min_shared_counts"],
        n_top_genes=PREPROCESSING["n_top_genes"],
        n_pcs=PREPROCESSING["n_pcs"],
        moments_n_neighbors=PREPROCESSING["moments_n_neighbors"],
        velocity_mode=VELOCITY["mode"],
        recover_dynamics_max_iter=VELOCITY["recover_dynamics_max_iter"],
        gap=GAP,
        seed=SEED,
    )
    state_path = pancreas_state_cache_path(
        raw_dir,
        preprocessing=PREPROCESSING,
        velocity=VELOCITY,
        dataset_prefix=pancreas_gap_prefix(GAP),
        seed=SEED,
    )
    state = load_pancreas_state_cache(state_path, state_metadata)
    if state is not None:
        return state

    print("State cache missing; running main_pancreas once to create it.")
    main_pancreas({
        "finsler_optimizer": None,
        "velocity": {"alpha": ALPHA_CLIP_GRID[0][0], "cos_clip": ALPHA_CLIP_GRID[0][1]},
    })
    state = load_pancreas_state_cache(state_path, state_metadata)
    if state is None:
        raise RuntimeError(f"Failed to create pancreas state cache: {state_path}")
    return state


def load_or_create_umap_2d(raw_dir):
    path = raw_dir / f"umap_{VELOCITY['mode']}_{pancreas_umap_variant_tag(UMAP)}s{SEED}.npy"
    if path.exists():
        return np.asarray(np.load(path), dtype=float)
    print("2D UMAP cache missing; running main_pancreas once to create it.")
    main_pancreas({"finsler_optimizer": None})
    if not path.exists():
        raise RuntimeError(f"Failed to create UMAP cache: {path}")
    return np.asarray(np.load(path), dtype=float)


def ensure_distance_cache(
        raw_dir,
        x_pca,
        velocity_pca,
        x_umap,
        labels,
        cell_ids,
        original_indices,
        gap_arrays,
        alpha,
        cos_clip,
):
    path = velocity_inputs_path(raw_dir, alpha=alpha, cos_clip=cos_clip)
    if path.exists():
        print(f"Exists: {path.name}")
        return

    print(f"Computing {path.name}")
    dists, _, _, _ = utils.compute_velocity_dist_matrix(
        x_pca,
        velocity_pca,
        kNN_euclid=VELOCITY["kNN_euclid"],
        kNN_finsler=VELOCITY["kNN_finsler"],
        alpha=alpha,
        distance_formula=VELOCITY["distance_formula"],
        cos_clip=cos_clip,
        velocity_neighbors=VELOCITY["velocity_neighbors"],
        average_velocity=VELOCITY["average_velocity"],
        symmetrize_support=VELOCITY["symmetrize_support"],
        n_jobs=VELOCITY["graph_n_jobs"],
    )
    metadata = velocity_cache_metadata(alpha, cos_clip)
    np.savez(
        path,
        x_umap=x_umap,
        dists_velocity=dists,
        labels=labels_to_cache(labels),
        cell_ids=np.asarray(cell_ids, dtype=str),
        original_indices=np.asarray(original_indices, dtype=int),
        gap_removed_original_indices=np.asarray(gap_arrays.get("gap_removed_original_indices", []), dtype=int),
        gap_before_indices=np.asarray(gap_arrays.get("gap_before_indices", []), dtype=int),
        gap_after_indices=np.asarray(gap_arrays.get("gap_after_indices", []), dtype=int),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Saved: {path.name}")


def velocity_inputs_path(raw_dir, *, alpha, cos_clip):
    tag = (
        f"{cache_token(VELOCITY['mode'])}_"
        f"{velocity_formula_tag()}_"
        f"valpha{cache_token(alpha)}_"
        f"cclip{cache_token(cos_clip)}_"
        f"ke{VELOCITY['kNN_euclid']}_kf{VELOCITY['kNN_finsler']}_"
        f"{pancreas_umap_variant_tag(UMAP)}s{SEED}"
    )
    return Path(raw_dir) / f"{pancreas_gap_prefix(GAP)}_velocity_inputs_{tag}.npz"


def velocity_cache_metadata(alpha, cos_clip):
    return pancreas_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=PREPROCESSING["min_shared_counts"],
        n_top_genes=PREPROCESSING["n_top_genes"],
        n_pcs=PREPROCESSING["n_pcs"],
        moments_n_neighbors=PREPROCESSING["moments_n_neighbors"],
        velocity_mode=VELOCITY["mode"],
        velocity_distance_formula=normalize_velocity_distance_formula(VELOCITY["distance_formula"]),
        velocity_alpha=alpha,
        velocity_cos_clip=cos_clip,
        velocity_neighbors=VELOCITY["velocity_neighbors"],
        velocity_kNN_euclid=VELOCITY["kNN_euclid"],
        velocity_kNN_finsler=VELOCITY["kNN_finsler"],
        average_velocity=VELOCITY["average_velocity"],
        symmetrize_velocity_support=VELOCITY["symmetrize_support"],
        recover_dynamics_max_iter=VELOCITY["recover_dynamics_max_iter"],
        umap_n_neighbors=UMAP["n_neighbors"],
        umap_min_dist=UMAP["min_dist"],
        umap_spread=UMAP["spread"],
        umap_maxiter=UMAP["maxiter"],
        umap_negative_sample_rate=UMAP["negative_sample_rate"],
        umap_init_pos=UMAP["init_pos"],
        gap=GAP,
        seed=SEED,
    )


def velocity_formula_tag():
    return {
        "exponential": "vexp",
        "randers": "vrand",
        "matsumoto": "vmats",
    }[normalize_velocity_distance_formula(VELOCITY["distance_formula"])]


if __name__ == "__main__":
    main()
