"""Evaluate pancreas trajectory-gap distances for saved embeddings."""

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
    DEFAULT_PANCREAS_GAP,
    normalized_gap_distance,
    pancreas_gap_group_indices,
)


def main_pancreas_gap_distance():
    project_root = Path(__file__).resolve().parents[4]
    raw_dir = project_root / "scripts" / "res" / "pancreas" / "raw"
    gap_dir = project_root / "scripts" / "res" / "pancreas" / "gap"
    out_dir = project_root / "scripts" / "res" / "pancreas" / "rna_velocity_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    reference_gap = _reference_gap_indices(gap_dir)
    for info in _gap_embedding_sources(raw_dir, gap_dir):
        embedding, labels, metadata, cached_before, cached_after = _load_embedding_and_labels(info["path"], info["kind"])
        if cached_before is not None and cached_after is not None:
            before, after = cached_before, cached_after
        elif info["scope"] == "full" and reference_gap is not None:
            before, after = reference_gap
        else:
            before, after = pancreas_gap_group_indices(labels, DEFAULT_PANCREAS_GAP)
        result = normalized_gap_distance(embedding, before, after)
        row = {
            "embedding": info["path"].name,
            "kind": info["kind"],
            "scope": info["scope"],
            "short_name": info["short_name"],
            "n_cells": embedding.shape[0],
            "embedding_dim": embedding.shape[1],
            "gap_name": DEFAULT_PANCREAS_GAP["name"],
            "removed_labels": ";".join(DEFAULT_PANCREAS_GAP["removed_labels"]),
            "before_labels": ";".join(DEFAULT_PANCREAS_GAP["before_labels"]),
            "after_labels": ";".join(DEFAULT_PANCREAS_GAP["after_labels"]),
            "n_before": result.n_before,
            "n_after": result.n_after,
            "gap_distance": result.distance,
            "normalized_gap_distance": result.normalized_distance,
            "before_representative": result.before_representative,
            "after_representative": result.after_representative,
            "max_pairwise_distance": result.max_pairwise_distance,
        }
        row.update(metadata)
        rows.append(row)
        print(
            f"{info['short_name']}: gap={result.normalized_distance:.6f} "
            f"(raw={result.distance:.6f}, n={embedding.shape[0]})"
        )

    out_path = out_dir / "pancreas_gap_distance.csv"
    _write_csv(out_path, rows)
    print(f"Saved gap-distance metrics: {out_path}")
    return rows


def _gap_embedding_sources(raw_dir, gap_dir):
    sources = []
    current_velocity = "dynamical_vrand_valpha2_cclip0p4_s42"
    current_smacof = "vrand_r0p8_iter100_seed42"
    gap_prefix = "gap_lt300"
    for path in sorted(raw_dir.glob(f"pancreas_velocity_inputs_{current_velocity}.npz")):
        sources.append({"path": path, "kind": "umap", "scope": "full", "short_name": "UMAP 2D full"})
    for path in sorted(gap_dir.glob(f"{gap_prefix}_velocity_inputs_{current_velocity}.npz")):
        sources.append({"path": path, "kind": "umap", "scope": "gap", "short_name": "UMAP 2D gap"})
    for path in sorted(raw_dir.glob(f"pancreas_randers_smacof_{current_smacof}.npz")):
        sources.append({"path": path, "kind": "smacof_randers", "scope": "full", "short_name": "SMACOF-Randers full"})
    for path in sorted(gap_dir.glob(f"{gap_prefix}_randers_smacof_{current_smacof}.npz")):
        sources.append({"path": path, "kind": "smacof_randers", "scope": "gap", "short_name": "SMACOF-Randers gap"})
    for path in sorted(gap_dir.glob("gap*_path_frozen_*.npz")):
        sources.append({"path": path, "kind": "path_frozen", "scope": "gap", "short_name": "path-frozen gap"})
    for path in sorted(gap_dir.glob("gap*_soft_bf_*.npz")):
        sources.append({"path": path, "kind": "soft_bf", "scope": "gap", "short_name": "soft-BF gap"})
    return sources


def _load_embedding_and_labels(path, kind):
    with np.load(path, allow_pickle=False) as cache:
        if kind == "umap":
            embedding = np.asarray(cache["x_umap"], dtype=float)
        else:
            embedding = np.asarray(cache["embedding"], dtype=float)
        labels = _labels_from_cache_or_fallback(cache, path, len(embedding))
        before = _cached_indices(cache, "gap_before_indices")
        after = _cached_indices(cache, "gap_after_indices")
        metadata = {}
        if "metadata_json" in cache:
            raw = json.loads(str(cache["metadata_json"].item()))
            metadata = _flatten_metadata(raw)
        if "stress" in cache:
            metadata["optimizer_stress"] = _last_scalar(cache["stress"])
        if "full_geodesic_stress" in cache:
            metadata["full_geodesic_stress"] = _last_scalar(cache["full_geodesic_stress"])
    return embedding, labels, metadata, before, after


def _reference_gap_indices(raw_dir):
    candidates = sorted(
        raw_dir.glob("gap*_velocity_inputs_*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        with np.load(path, allow_pickle=False) as cache:
            before = _cached_indices(cache, "gap_before_original_indices")
            after = _cached_indices(cache, "gap_after_original_indices")
            if before is not None and after is not None:
                return before, after
    return None


def _cached_indices(cache, key):
    if key not in cache:
        return None
    values = np.asarray(cache[key], dtype=int)
    return values if values.size else None


def _labels_from_cache_or_fallback(cache, path, n_samples):
    if "labels" in cache and cache["labels"].size == n_samples:
        return np.asarray(cache["labels"], dtype=str)
    scope = "gap" if path.name.startswith("gap") else "full"
    prefix = "gap*_velocity_inputs_" if scope == "gap" else "pancreas_velocity_inputs_"
    for label_cache_path in sorted(path.parent.glob(prefix + "*.npz")):
        with np.load(label_cache_path, allow_pickle=False) as label_cache:
            if "labels" in label_cache and label_cache["labels"].size == n_samples:
                return np.asarray(label_cache["labels"], dtype=str)
    raise KeyError(f"No labels found for {path}.")


def _flatten_metadata(metadata):
    out = {}
    for key in ("optimizer", "init", "seed", "randers_alpha_embedding"):
        if key in metadata:
            out[f"metadata_{key}"] = metadata[key]
    if isinstance(metadata.get("velocity"), dict):
        velocity = metadata["velocity"]
        for key in ("mode", "distance_formula", "alpha", "cos_clip"):
            if key in velocity:
                out[f"metadata_velocity_{key}"] = velocity[key]
    if isinstance(metadata.get("smacof"), dict) and "max_iter" in metadata["smacof"]:
        out["metadata_smacof_max_iter"] = metadata["smacof"]["max_iter"]
    return out


def _last_scalar(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.nan
    return float(array.reshape(-1)[-1])


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main_pancreas_gap_distance()
