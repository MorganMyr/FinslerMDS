"""Evaluate saved pancreas UMAP embeddings with RNA-velocity direction metrics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.sparse

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    cross_boundary_direction_correctness,
    in_cluster_velocity_coherence,
    project_velocity_graph_to_embedding,
)
from finsler_mds.utils.pancreas import PANCREAS_TRANSITIONS  # noqa: E402
from finsler_mds.utils.pancreas_files import pancreas_velocity_cache_dir, pancreas_velocity_cache_path  # noqa: E402


def main_pancreas_umap():
    project_root = Path(__file__).resolve().parents[4]
    raw_dir = _pancreas_raw_dir(project_root)
    out_dir = raw_dir.parent / "rna_velocity_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_eval_neighbors = 30
    n_transition_neighbors = 30
    rows = []
    print(
        "Evaluating projected velocities reconstructed from cached directed "
        "distances, not from the original scVelo velocity graph."
    )
    for embedding_path in sorted(set(raw_dir.glob("umap*.npy")) | set(raw_dir.glob("pancreas_umap*.npy"))):
        cache_path = _matching_velocity_cache(raw_dir, embedding_path)
        if cache_path is None:
            print(f"Skipping {embedding_path.name}: no matching velocity-input cache.")
            continue

        embedding = np.asarray(np.load(embedding_path), dtype=float)
        with np.load(cache_path) as cache:
            dists_velocity = np.asarray(cache["dists_velocity"], dtype=float)
            labels = np.asarray(cache["labels"], dtype=str)

        transition_graph = transition_graph_from_directed_distances(
            dists_velocity,
            n_neighbors=n_transition_neighbors,
        )
        velocity_embedding = project_velocity_graph_to_embedding(
            embedding,
            transition_graph,
        )
        cbdir = cross_boundary_direction_correctness(
            embedding,
            labels,
            PANCREAS_TRANSITIONS,
            velocity_vectors=velocity_embedding,
            n_neighbors=n_eval_neighbors,
        )
        icvcoh = in_cluster_velocity_coherence(
            embedding,
            labels,
            velocity_vectors=velocity_embedding,
            n_neighbors=n_eval_neighbors,
        )

        row = {
            "embedding": embedding_path.name,
            "velocity_cache": cache_path.name,
            "n_cells": len(embedding),
            "embedding_dim": embedding.shape[1],
            "transition_graph_neighbors": n_transition_neighbors,
            "eval_neighbors": n_eval_neighbors,
            "velocity_source": "directed_distance_cache",
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

        print(f"{embedding_path.name}")
        print(f"  cache: {cache_path.name}")
        print(f"  CBDir:  {cbdir.score:.6f}")
        print(f"  ICVCoh: {icvcoh.score:.6f}")
        for edge, score in cbdir.transitions.items():
            print(
                f"    {edge[0]} -> {edge[1]}: {score.score:.6f} "
                f"({score.n_boundary_cells} boundary cells, "
                f"{score.n_neighbor_pairs} neighbor pairs)"
            )

    if not rows:
        raise FileNotFoundError(f"No evaluable pancreas UMAP embeddings found in {raw_dir}.")

    out_path = out_dir / "pancreas_umap_directional_metrics.csv"
    _write_csv(out_path, rows)
    print(f"Saved metrics: {out_path}")
    return rows


def transition_graph_from_directed_distances(
        distances,
        *,
        n_neighbors=30,
        temperature=None,
        eps=1e-12,
):
    """Build local transition affinities from a directed distance matrix.

    This is a pragmatic fallback for cached runs that did not save the original
    scVelo transition graph. For each source cell, the closest finite directed
    targets are selected and converted to affinities with a row-wise exponential
    kernel.
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distances must be a square matrix.")
    n_samples = D.shape[0]
    n_neighbors = min(int(n_neighbors), n_samples - 1)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")

    rows = []
    cols = []
    data = []
    for source in range(n_samples):
        row = D[source].copy()
        row[source] = np.inf
        finite_targets = np.flatnonzero(np.isfinite(row))
        if len(finite_targets) == 0:
            continue
        if len(finite_targets) > n_neighbors:
            local_values = row[finite_targets]
            keep = np.argpartition(local_values, n_neighbors - 1)[:n_neighbors]
            targets = finite_targets[keep]
        else:
            targets = finite_targets
        target_distances = row[targets]
        order = np.argsort(target_distances)
        targets = targets[order]
        target_distances = target_distances[order]

        shifted = target_distances - target_distances[0]
        if temperature is None:
            local_temperature = float(np.median(target_distances[target_distances > eps]))
            if not np.isfinite(local_temperature) or local_temperature <= eps:
                local_temperature = float(np.mean(target_distances) + eps)
        else:
            local_temperature = float(temperature)
        weights = np.exp(-shifted / max(local_temperature, eps))
        rows.extend([source] * len(targets))
        cols.extend(targets.tolist())
        data.extend(weights.tolist())

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=D.shape)


def _pancreas_raw_dir(project_root):
    candidates = [
        project_root / "scripts" / "res" / "pancreas" / "raw",
        project_root / "res" / "pancreas" / "raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _matching_velocity_cache(raw_dir, embedding_path):
    name = embedding_path.name
    if not name.endswith(".npy"):
        return None
    if name.startswith("pancreas_umap_3d_"):
        tag = name[len("pancreas_umap_3d_"):-len(".npy")]
    elif name.startswith("pancreas_umap_"):
        tag = name[len("pancreas_umap_"):-len(".npy")]
    elif name.startswith("umap_3d_"):
        tag = name[len("umap_3d_"):-len(".npy")]
    elif name.startswith("umap_"):
        tag = name[len("umap_"):-len(".npy")]
    else:
        return None
    direct = pancreas_velocity_cache_path(raw_dir, f"pancreas_velocity_inputs_{tag}.npz")
    if direct.exists():
        return direct
    candidates = sorted(pancreas_velocity_cache_dir(raw_dir).glob("pancreas_velocity_inputs_*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    return None


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
    main_pancreas_umap()
