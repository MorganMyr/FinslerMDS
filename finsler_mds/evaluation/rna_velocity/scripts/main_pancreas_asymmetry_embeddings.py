"""Evaluate local asymmetry preservation on saved pancreas embeddings."""

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

from finsler_mds.evaluation import asymmetry_preservation_from_neighbors  # noqa: E402
from finsler_mds.metrics import (  # noqa: E402
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_saved_embeddings import (  # noqa: E402
    _embedding_family,
    _load_embedding,
    _short_embedding_name,
)
from finsler_mds.evaluation.rna_velocity.scripts.main_pancreas_scvelo_umap import _csv_value  # noqa: E402


def main_pancreas_asymmetry_embeddings():
    project_root = Path(__file__).resolve().parents[4]
    raw_dir = project_root / "scripts" / "res" / "pancreas" / "raw"
    eval_dir = project_root / "scripts" / "res" / "pancreas" / "rna_velocity_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    tau = 0.02
    n_neighbors = 30
    seed = 42
    velocity_mode = "dynamical"
    context_cache_path = eval_dir / "raw" / f"pancreas_scvelo_umap_{velocity_mode}_s{seed}.npz"
    expression_neighbors = _load_expression_neighbors(context_cache_path, n_neighbors=n_neighbors)

    normal_scores = _load_normal_scores(eval_dir / "pancreas_saved_embeddings_true_velocity_metrics.csv")
    rows = []
    for path in _selected_embedding_paths(raw_dir):
        metadata = _load_metadata(path)
        metric, metric_name, metric_alpha = _metric_from_metadata(metadata, path.name)
        velocity_alpha = _velocity_alpha_from_metadata(metadata, default=2.0)
        velocity_formula = _velocity_formula_from_metadata(metadata, default="exponential")
        data_path = _velocity_cache_path(
            raw_dir,
            velocity_alpha,
            velocity_formula=velocity_formula,
            velocity_mode=velocity_mode,
            seed=seed,
        )
        if not data_path.exists():
            print(f"Skipping {path.name}: missing velocity cache {data_path.name}.")
            continue

        embedding = _load_embedding(path)
        data_dissimilarities = _load_data_dissimilarities(data_path)
        if embedding.shape[0] != data_dissimilarities.shape[0]:
            print(
                f"Skipping {path.name}: {embedding.shape[0]} embedding points, "
                f"{data_dissimilarities.shape[0]} dissimilarity points."
            )
            continue

        print(
            f"Evaluating asymmetry preservation on {_short_embedding_name(path.name)} "
            f"(data alpha={velocity_alpha:g}, {metric_name} alpha={metric_alpha:g})"
        )
        result = asymmetry_preservation_from_neighbors(
            data_dissimilarities,
            embedding,
            metric,
            expression_neighbors,
            tau=tau,
            unique_pairs=True,
        )
        normal = normal_scores.get(path.name, {})
        row = {
            "embedding": path.name,
            "short_name": _short_embedding_name(path.name),
            "family": _embedding_family(path.name),
            "metric": metric_name,
            "metric_alpha": metric_alpha,
            "velocity_alpha": velocity_alpha,
            "velocity_distance_formula": velocity_formula,
            "neighbor_space": "expression_pca",
            "n_neighbors": n_neighbors,
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
            "cbdir": _float_or_nan(normal.get("cbdir", np.nan)),
            "icvcoh": _float_or_nan(normal.get("icvcoh", np.nan)),
            "optimizer_stress": _stress_from_cache(path),
            "data_cache": data_path.name,
        }
        rows.append(row)
        print(
            f"  sign={result.sign_accuracy:.3f}, "
            f"Spearman={result.spearman:.3f}, gamma={result.gamma:.3f}, "
            f"|A_data|={result.mean_abs_data_asymmetry:.3f}, "
            f"|A_emb|={result.mean_abs_embedding_asymmetry:.3f}"
        )

    out_path = eval_dir / "pancreas_asymmetry_preservation_metrics.csv"
    _write_csv(out_path, rows)
    print(f"Saved asymmetry metrics: {out_path}")
    return rows


def _selected_embedding_paths(raw_dir):
    paths = []
    paths.extend(raw_dir.glob("smacof*.npz"))
    paths.extend(raw_dir.glob("pf*.npz"))
    paths.extend(raw_dir.glob("sbf*.npz"))
    paths.extend(raw_dir.glob("pancreas_randers_smacof*.npz"))
    paths.extend(raw_dir.glob("pancreas_path_frozen*.npz"))
    paths.extend(raw_dir.glob("pancreas_soft_bf*.npz"))
    return sorted(set(paths), key=lambda path: (_embedding_family(path.name), path.name))


def _load_expression_neighbors(cache_path, *, n_neighbors):
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing expression/PCA neighbor cache: {cache_path}. "
            "Run main_pancreas_scvelo_umap.py first."
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        neighbors = np.asarray(cache["expression_neighbors"], dtype=int)
    return neighbors[:, :n_neighbors]


def _load_data_dissimilarities(path):
    with np.load(path, allow_pickle=False) as cache:
        return np.asarray(cache["dists_velocity"], dtype=float)


def _load_metadata(path):
    try:
        with np.load(path, allow_pickle=False) as cache:
            if "metadata_json" in cache:
                return json.loads(str(cache["metadata_json"].item()))
    except Exception:
        pass
    return {}


def _metric_from_metadata(metadata, filename):
    family = _embedding_family(filename)
    if family == "randers_smacof":
        alpha = metadata.get("randers_alpha_embedding")
        if alpha is None:
            match = re.search(r"_alpha([0-9p]+)(?:_|\\.npz)", filename)
            if match is not None:
                alpha = float(match.group(1).replace("p", "."))
        if alpha is None:
            raise ValueError(f"Could not infer Randers alpha from {filename}.")
        return RandersMetric(alpha=float(alpha)), "randers", float(alpha)

    if family in {"path_frozen", "soft_bf"}:
        metric_metadata = metadata.get("geodesic_metric")
        if isinstance(metric_metadata, dict):
            metric = _metric_from_kind_alpha(
                metric_metadata.get("kind", "convexified_matsumoto"),
                float(metric_metadata.get("alpha", 0.0)),
            )
            return metric, _metric_name(metric), float(metric_metadata.get("alpha", 0.0))

        alpha = metadata.get("matsumoto_alpha_embedding")
        if alpha is None:
            parsed = _parse_metric_tag_from_name(filename)
            if parsed is not None:
                kind, alpha = parsed
                metric = _metric_from_kind_alpha(kind, alpha)
                return metric, _metric_name(metric), float(alpha)
        else:
            return ConvexifiedMatsumotoMetric(alpha=float(alpha)), "convexified_matsumoto", float(alpha)
        if alpha is None:
            raise ValueError(f"Could not infer Convexified Matsumoto alpha from {filename}.")

    raise ValueError(f"Unsupported embedding family for asymmetry metrics: {filename}")


def _metric_from_kind_alpha(kind, alpha):
    kind = kind.lower()
    if kind in {"randers", "r"}:
        return RandersMetric(alpha=float(alpha))
    if kind in {"matsumoto", "mats", "m"}:
        return MatsumotoMetric(alpha=float(alpha))
    if kind in {"convexified_matsumoto", "convexifiedmatsumoto", "cmats", "cm"}:
        return ConvexifiedMatsumotoMetric(alpha=float(alpha))
    raise ValueError(f"Unsupported embedding metric kind {kind!r}.")


def _metric_name(metric):
    if isinstance(metric, RandersMetric):
        return "randers"
    if isinstance(metric, ConvexifiedMatsumotoMetric):
        return "convexified_matsumoto"
    if isinstance(metric, MatsumotoMetric):
        return "matsumoto"
    return type(metric).__name__


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


def _velocity_alpha_from_metadata(metadata, *, default):
    velocity = metadata.get("velocity", {})
    if isinstance(velocity, dict) and "alpha" in velocity:
        return float(velocity["alpha"])
    return float(default)


def _velocity_formula_from_metadata(metadata, *, default):
    velocity = metadata.get("velocity", {})
    if isinstance(velocity, dict) and "distance_formula" in velocity:
        return str(velocity["distance_formula"])
    return default


def _velocity_cache_path(raw_dir, velocity_alpha, *, velocity_formula, velocity_mode, seed):
    formula_tag = {"exponential": "vexp", "randers": "vrand"}.get(str(velocity_formula), "vexp")
    token = _cache_token(velocity_alpha)
    legacy_token = str(float(velocity_alpha)).replace(".", "p")
    candidates = [
        raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_{formula_tag}_valpha{token}_cclip0p4_ke30_kf0_s{seed}.npz",
        raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_{formula_tag}_valpha{token}_s{seed}.npz",
        raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_valpha{token}_s{seed}.npz",
        raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_{formula_tag}_valpha{legacy_token}_s{seed}.npz",
        raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_valpha{legacy_token}_s{seed}.npz",
    ]
    if np.isclose(velocity_alpha, 2.0) and formula_tag == "vexp":
        candidates.append(raw_dir / f"pancreas_velocity_inputs_{velocity_mode}_s{seed}.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    globbed = sorted(
        raw_dir.glob(f"pancreas_velocity_inputs_{velocity_mode}_{formula_tag}_valpha{token}_*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return globbed[0] if globbed else candidates[0]


def _cache_token(value):
    if isinstance(value, str):
        return value.replace(".", "p")
    return f"{float(value):g}".replace(".", "p")


def _load_normal_scores(path):
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[row.get("embedding", "")] = row
    return out


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
    main_pancreas_asymmetry_embeddings()
