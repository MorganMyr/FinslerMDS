"""Evaluate saved pancreas embeddings with geometry-induced directions."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import re
import sys

import numpy as np

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.metrics import (  # noqa: E402
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)
from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    cross_boundary_direction_correctness,
    finsler_induced_velocity_field,
    in_cluster_velocity_coherence,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_saved_embeddings import (  # noqa: E402
    _embedding_info,
    _load_embedding,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_scvelo_umap import (  # noqa: E402
    PANCREAS_TRANSITIONS,
    _csv_value,
    _slug,
)


def main_pancreas_geometry_embeddings():
    project_root = Path(__file__).resolve().parents[4]
    embedding_dir = project_root / "scripts" / "res" / "pancreas" / "raw"
    eval_dir = project_root / "scripts" / "res" / "pancreas" / "rna_velocity_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    velocity_mode = "dynamical"
    n_eval_neighbors = 30
    geometry = {
        "mode": "soft",
        "beta": 8.0,
        "n_neighbors": "optimizer_graph",
    }

    labels, expression_neighbors = _load_eval_context(
        eval_dir / "raw" / f"pancreas_scvelo_umap_{velocity_mode}_s{seed}.npz"
    )
    normal_scores = _load_normal_scores(
        eval_dir / "pancreas_saved_embeddings_true_velocity_metrics.csv"
    )

    rows = []
    for info in _saved_geometry_embeddings(embedding_dir):
        path = info["path"]
        embedding = _load_embedding(path)
        metadata = _load_metadata(path)
        if embedding.shape[0] != len(labels):
            print(f"Skipping {path.name}: {embedding.shape[0]} cells, expected {len(labels)}.")
            continue
        if embedding.shape[1] != 3:
            print(f"Skipping {path.name}: geometry field expects 3D Finsler embeddings.")
            continue

        metric = _metric_from_metadata_or_name(metadata, path.name)
        n_geometry_neighbors = _geometry_neighbors_from_metadata(
            metadata,
            family=info["family"],
            default=20,
        )
        print(
            f"Evaluating geometry field on {info['short_name']} "
            f"(alpha={metric.alpha:g}, k={n_geometry_neighbors}, beta={geometry['beta']:g})"
        )
        field = finsler_induced_velocity_field(
            embedding,
            metric,
            n_neighbors=n_geometry_neighbors,
            beta=geometry["beta"],
            mode=geometry["mode"],
        )
        cbdir = cross_boundary_direction_correctness(
            embedding,
            labels,
            PANCREAS_TRANSITIONS,
            velocity_vectors=field.vectors,
            neighbor_indices=expression_neighbors,
            n_neighbors=n_eval_neighbors,
        )
        icvcoh = in_cluster_velocity_coherence(
            embedding,
            labels,
            velocity_vectors=field.vectors,
            neighbor_indices=expression_neighbors,
            n_neighbors=n_eval_neighbors,
        )

        normal = normal_scores.get(path.name, {})
        finite_confidence = field.confidence[np.isfinite(field.confidence)]
        row = {
            "embedding": path.name,
            "short_name": info["short_name"],
            "family": info["family"],
            "embedding_dim": embedding.shape[1],
            "n_cells": embedding.shape[0],
            "metric": "convexified_matsumoto",
            "metric_alpha": metric.alpha,
            "geometry_mode": geometry["mode"],
            "geometry_beta": geometry["beta"],
            "geometry_n_neighbors": n_geometry_neighbors,
            "neighbor_space": "expression",
            "eval_neighbors": n_eval_neighbors,
            "geometry_velocity_source": "embedding_knn_softmax_phi_no_distance_kernel",
            "geometry_cbdir": cbdir.score,
            "geometry_icvcoh": icvcoh.score,
            "normal_cbdir": _float_or_nan(normal.get("cbdir", np.nan)),
            "normal_icvcoh": _float_or_nan(normal.get("icvcoh", np.nan)),
            "optimizer_stress": _stress_from_cache(path),
            "mean_geometry_confidence": (
                float(np.mean(finite_confidence)) if len(finite_confidence) else np.nan
            ),
            "median_geometry_confidence": (
                float(np.median(finite_confidence)) if len(finite_confidence) else np.nan
            ),
        }
        for edge, score in cbdir.transitions.items():
            source, target = edge
            prefix = _slug(f"geometry_cbdir_{source}_to_{target}")
            row[prefix] = score.score
            row[prefix + "_boundary_cells"] = score.n_boundary_cells
            row[prefix + "_neighbor_pairs"] = score.n_neighbor_pairs
        for cluster, score in icvcoh.clusters.items():
            prefix = _slug(f"geometry_icvcoh_{cluster}")
            row[prefix] = score.score
            row[prefix + "_cells"] = score.n_cells
            row[prefix + "_neighbor_pairs"] = score.n_neighbor_pairs
        rows.append(row)
        print(
            f"  Geometry-CBDir={cbdir.score:.6f}, Geometry-ICVCoh={icvcoh.score:.6f}; "
            f"normal CBDir={row['normal_cbdir']:.6f}, normal ICVCoh={row['normal_icvcoh']:.6f}"
        )

    out_path = eval_dir / "pancreas_geometry_velocity_metrics.csv"
    _write_csv(out_path, rows)
    print(f"Saved geometry metrics: {out_path}")
    return rows


def _saved_geometry_embeddings(embedding_dir):
    paths = []
    paths.extend(embedding_dir.glob("pf*.npz"))
    paths.extend(embedding_dir.glob("sbf*.npz"))
    paths.extend(embedding_dir.glob("pancreas_path_frozen*.npz"))
    paths.extend(embedding_dir.glob("pancreas_soft_bf*.npz"))
    return [_embedding_info(path) for path in sorted(set(paths), key=lambda item: item.name)]


def _load_eval_context(cache_path):
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing scVelo evaluation cache: {cache_path}. "
            "Run main_pancreas_scvelo_umap.py first."
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        labels = np.asarray(cache["labels"], dtype=str)
        expression_neighbors = np.asarray(cache["expression_neighbors"], dtype=int)
    return labels, expression_neighbors


def _load_normal_scores(path):
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[row.get("embedding", "")] = row
    return out


def _load_metadata(path):
    try:
        with np.load(path, allow_pickle=False) as cache:
            if "metadata_json" not in cache:
                return {}
            return json.loads(str(cache["metadata_json"].item()))
    except Exception:
        return {}


def _metric_from_metadata_or_name(metadata, filename):
    metric_metadata = metadata.get("geodesic_metric")
    if isinstance(metric_metadata, dict):
        kind = metric_metadata.get("kind", "convexified_matsumoto")
        alpha = float(metric_metadata.get("alpha", 0.0))
        return _metric_from_kind_alpha(kind, alpha)

    alpha = metadata.get("matsumoto_alpha_embedding")
    if alpha is not None:
        return ConvexifiedMatsumotoMetric(alpha=float(alpha))

    parsed = _parse_metric_tag_from_name(filename)
    if parsed is None:
        raise ValueError(f"Could not infer embedding metric from {filename}.")
    kind, alpha = parsed
    return _metric_from_kind_alpha(kind, alpha)


def _metric_from_kind_alpha(kind, alpha):
    kind = kind.lower()
    if kind in {"randers", "r"}:
        return RandersMetric(alpha=float(alpha))
    if kind in {"matsumoto", "mats", "m"}:
        return MatsumotoMetric(alpha=float(alpha))
    if kind in {"convexified_matsumoto", "convexifiedmatsumoto", "cmats", "cm"}:
        return ConvexifiedMatsumotoMetric(alpha=float(alpha))
    raise ValueError(f"Unsupported embedding metric kind {kind!r}.")


def _parse_metric_tag_from_name(filename):
    legacy = re.search(r"cmatsumoto_alpha([0-9p]+)", filename)
    if legacy is not None:
        return "convexified_matsumoto", float(legacy.group(1).replace("p", "."))
    match = re.search(r"_(cmats|mats|r)([0-9p]+)(?:_|\\.npz)", filename)
    if match is None:
        return None
    kind, alpha = match.groups()
    aliases = {
        "r": "randers",
        "mats": "matsumoto",
        "cmats": "convexified_matsumoto",
    }
    return aliases[kind], float(alpha.replace("p", "."))


def _geometry_neighbors_from_metadata(metadata, *, family, default):
    optimizer_key = {
        "path_frozen": "path_frozen",
        "soft_bf": "soft_bf",
    }.get(family)
    options = metadata.get(optimizer_key, {}) if optimizer_key is not None else {}
    return int(options.get("graph_neighbors", options.get("n_neighbors", default)))


def _stress_from_cache(path):
    try:
        with np.load(path, allow_pickle=False) as cache:
            if "full_geodesic_stress" in cache:
                return _scalar_or_nan(cache["full_geodesic_stress"])
            if "stress" in cache:
                return _scalar_or_nan(cache["stress"])
    except Exception:
        return np.nan
    return np.nan


def _scalar_or_nan(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.nan
    return float(array.reshape(-1)[-1])


def _float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(_csv_value(row.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main_pancreas_geometry_embeddings()
