"""Evaluate asymmetry preservation for UMAP plus projected velocities."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

import numpy as np

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    project_velocity_graph_to_embedding,
    velocity_field_asymmetry_preservation_from_neighbors,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_scvelo_umap import (  # noqa: E402
    _csv_value,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_umap import (  # noqa: E402
    transition_graph_from_directed_distances,
)


def main_pancreas_umap_asymmetry():
    project_root = Path(__file__).resolve().parents[4]
    raw_dir = project_root / "scripts" / "res" / "pancreas" / "raw"
    eval_dir = project_root / "scripts" / "res" / "pancreas" / "rna_velocity_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    velocity_mode = "dynamical"
    tau = 0.02
    n_neighbors = 30
    transition_graph_neighbors = 30
    expression_neighbors = _load_expression_neighbors(
        eval_dir / "raw" / f"pancreas_scvelo_umap_{velocity_mode}_s{seed}.npz",
        n_neighbors=n_neighbors,
    )

    rows = []
    for embedding_path in _selected_umap_embeddings(raw_dir, velocity_mode=velocity_mode, seed=seed):
        cache_path = _velocity_cache_for_umap(raw_dir, velocity_mode=velocity_mode, seed=seed)
        embedding = np.asarray(np.load(embedding_path), dtype=float)
        with np.load(cache_path, allow_pickle=False) as cache:
            data_dissimilarities = np.asarray(cache["dists_velocity"], dtype=float)
            metadata = _load_metadata_from_cache(cache)

        velocity_alpha = float(metadata.get("velocity_alpha", 1.0))
        transition_graph = transition_graph_from_directed_distances(
            data_dissimilarities,
            n_neighbors=transition_graph_neighbors,
        )
        velocity_embedding = project_velocity_graph_to_embedding(
            embedding,
            transition_graph,
        )
        result = velocity_field_asymmetry_preservation_from_neighbors(
            data_dissimilarities,
            embedding,
            velocity_embedding,
            expression_neighbors,
            alpha=velocity_alpha,
            tau=tau,
            unique_pairs=True,
        )

        row = {
            "embedding": embedding_path.name,
            "embedding_dim": embedding.shape[1],
            "velocity_cache": cache_path.name,
            "velocity_projection_source": "directed_distance_cache_projected_to_umap",
            "velocity_alpha": velocity_alpha,
            "neighbor_space": "expression_pca",
            "n_neighbors": n_neighbors,
            "transition_graph_neighbors": transition_graph_neighbors,
            "tau": result.tau,
            "n_pairs": result.n_pairs,
            "n_strong_pairs": result.n_strong_pairs,
            "sign_accuracy": result.sign_accuracy,
            "weighted_sign_accuracy": result.weighted_sign_accuracy,
            "spearman": result.spearman,
            "pearson": result.pearson,
            "gamma": result.gamma,
            "normalized_mse": result.normalized_mse,
            "mean_abs_data_asymmetry": result.mean_abs_data_asymmetry,
            "mean_abs_embedding_asymmetry": result.mean_abs_embedding_asymmetry,
        }
        rows.append(row)
        print(
            f"{embedding_path.name}: sign={result.sign_accuracy:.3f}, "
            f"Spearman={result.spearman:.3f}, gamma={result.gamma:.3f}, "
            f"|A_data|={result.mean_abs_data_asymmetry:.3f}, "
            f"|A_umap+vel|={result.mean_abs_embedding_asymmetry:.3f}"
        )

    if not rows:
        raise FileNotFoundError(f"No saved UMAP embeddings found in {raw_dir}.")

    out_path = eval_dir / "pancreas_umap_asymmetry_preservation_metrics.csv"
    _write_csv(out_path, rows)
    print(f"Saved UMAP asymmetry metrics: {out_path}")
    return rows


def _selected_umap_embeddings(raw_dir, *, velocity_mode, seed):
    names = [
        f"pancreas_umap_{velocity_mode}_s{seed}.npy",
        f"pancreas_umap_3d_{velocity_mode}_s{seed}.npy",
    ]
    return [raw_dir / name for name in names if (raw_dir / name).exists()]


def _velocity_cache_for_umap(raw_dir, *, velocity_mode, seed):
    path = raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_s{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing velocity-input cache: {path}")
    return path


def _load_expression_neighbors(cache_path, *, n_neighbors):
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing expression/PCA neighbor cache: {cache_path}. "
            "Run main_pancreas_scvelo_umap.py first."
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        neighbors = np.asarray(cache["expression_neighbors"], dtype=int)
    return neighbors[:, :n_neighbors]


def _load_metadata_from_cache(cache):
    if "metadata_json" not in cache:
        return {}
    return json.loads(str(cache["metadata_json"].item()))


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(_csv_value(row.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main_pancreas_umap_asymmetry()
