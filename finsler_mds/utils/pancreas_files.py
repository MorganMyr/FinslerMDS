"""Paths and minimal embedding I/O for pancreas experiments."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .embedding_io import cache_token, load_saved_embedding, metric_alpha_tag


_METHOD_TAGS = {
    "smacof": "smacof",
    "smacof_randers": "smacof",
    "gradient_descent": "gd",
    "gd": "gd",
    "finsler_umap": "fumap",
    "fumap": "fumap",
    "path_frozen": "pf",
    "pf": "pf",
}
_TAG_TO_DIRECTORY = {
    "smacof": "smacof",
    "gd": "gradient_descent",
    "fumap": "finsler_umap",
    "pf": "path_frozen",
    "umap": "umap",
    "isomap": "isomap",
}


def pancreas_method_tag(method):
    key = str(method).lower().replace("-", "_")
    try:
        return _METHOD_TAGS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown pancreas method: {method!r}.") from exc


def pancreas_result_stem(method, *, n_components, velocity_tag, metric_tag, dataset_prefix="pancreas"):
    prefix = pancreas_file_prefix(dataset_prefix)
    return f"{prefix}{pancreas_method_tag(method)}_{int(n_components)}d_{velocity_tag}_{metric_tag}"


def pancreas_reference_stem(
        method,
        *,
        n_components,
        n_neighbors,
        min_dist=None,
        dataset_prefix="pancreas",
):
    method = str(method).lower()
    if method not in {"umap", "isomap"}:
        raise ValueError("Reference method must be 'umap' or 'isomap'.")
    stem = (
        f"{pancreas_file_prefix(dataset_prefix)}{method}_{int(n_components)}d_"
        f"k{int(n_neighbors)}"
    )
    if method == "umap" and min_dist is not None:
        stem += f"_md{cache_token(min_dist)}"
    return stem


def pancreas_artifact_dir(base_dir, filename):
    name = Path(filename).name.lower()
    method = next(
        (directory for tag, directory in _TAG_TO_DIRECTORY.items() if name.startswith(f"{tag}_")),
        None,
    )
    dimension = re.search(r"(^|_)([23])d(_|$)", Path(filename).stem.lower())
    if method is None or dimension is None:
        return Path(base_dir)
    return Path(base_dir) / method / f"{dimension.group(2)}D"


def pancreas_embedding_path(raw_dir, stem):
    filename = f"{stem}.npz"
    path = pancreas_artifact_dir(raw_dir, filename) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pancreas_velocity_cache_path(raw_dir, filename: str):
    path = Path(raw_dir) / "velocity_inputs" / Path(filename).name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pancreas_figure_path(pancreas_dir, filename: str):
    path = pancreas_artifact_dir(Path(pancreas_dir) / "figure", filename) / Path(filename).name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_pancreas_embedding(path, embedding, cell_ids, *, objective=None, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "embedding": np.asarray(embedding, dtype=float),
        "cell_ids": np.asarray(cell_ids, dtype=str),
    }
    if objective is not None:
        arrays["objective"] = np.asarray(objective, dtype=float)
    if metadata is not None:
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(path, **arrays)
    print(f"Saved embedding: {path}")


def load_pancreas_embedding(
        path,
        *,
        cell_ids=None,
        expected_shape=None,
        expected_metadata=None,
):
    if Path(path).suffix != ".npz":
        raise ValueError("Pancreas embeddings must use the .npz format.")
    embedding = load_saved_embedding(path, cell_ids=cell_ids)
    if expected_metadata is not None:
        with np.load(path, allow_pickle=False) as cache:
            if "metadata_json" not in cache:
                raise ValueError(f"Saved embedding has no cache metadata: {path}")
            metadata = json.loads(str(cache["metadata_json"].item()))
        if metadata != expected_metadata:
            raise ValueError(f"Saved embedding has incompatible cache metadata: {path}")
    if expected_shape is not None and embedding.shape != tuple(expected_shape):
        raise ValueError(
            f"Saved embedding has shape {embedding.shape}, expected {expected_shape}: {path}"
        )
    return embedding


def resolve_pancreas_embedding_path(value: str, raw_dir):
    """Resolve an arbitrary initialization path; new embeddings are NPZ-only."""
    path = Path(str(value))
    if path.suffix not in {"", ".npz"}:
        raise ValueError("Pancreas initialization files must use the .npz format.")
    if path.suffix == "":
        path = path.with_suffix(".npz")

    raw_dir = Path(raw_dir)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, raw_dir / path]
    if not path.is_absolute():
        candidates.extend(raw_dir.glob(f"*/*/{path.name}"))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if existing:
        return max(existing, key=lambda candidate: candidate.stat().st_mtime)
    searched = ", ".join(map(str, candidates))
    raise FileNotFoundError(f"Embedding file not found. Searched: {searched}")


def latest_pancreas_embedding_path(raw_dir, method, embedding_dim, dataset_prefix="pancreas"):
    """Return the latest result, preferring the requested dimension over 2D."""
    tag = pancreas_method_tag(method)
    prefix = pancreas_file_prefix(dataset_prefix)
    raw_dir = Path(raw_dir)
    dimensions = [int(embedding_dim)] + ([2] if int(embedding_dim) == 3 else [])
    for dimension in dimensions:
        pattern = f"{prefix}{tag}_{dimension}d_*.npz"
        directory = raw_dir / _TAG_TO_DIRECTORY[tag] / f"{dimension}D"
        candidates = list(directory.glob(pattern)) + list(raw_dir.glob(pattern))
        if candidates:
            return max(set(candidates), key=lambda path: path.stat().st_mtime)
    return None


def pancreas_file_prefix(dataset_prefix):
    return "" if dataset_prefix == "pancreas" else f"{dataset_prefix}_"


def pancreas_velocity_inputs_path(raw_dir, *, velocity, dataset_prefix="pancreas", seed=42):
    formula = str(velocity["distance_formula"]).lower()
    formula_tag = {"randers": "vrand", "matsumoto": "vmats"}.get(formula)
    if formula_tag is None:
        raise ValueError("velocity distance formula must be 'randers' or 'matsumoto'.")
    name = (
        f"{pancreas_file_prefix(dataset_prefix)}velocity_inputs_{velocity['mode']}_"
        f"{formula_tag}_valpha{metric_alpha_tag(velocity['alpha'])}_"
        f"cclip{cache_token(velocity['cos_clip'])}_"
        f"ke{int(velocity['kNN_euclid'])}_kf{int(velocity['kNN_finsler'])}_s{int(seed)}.npz"
    )
    return pancreas_velocity_cache_path(raw_dir, name)


__all__ = [
    "latest_pancreas_embedding_path",
    "load_pancreas_embedding",
    "pancreas_artifact_dir",
    "pancreas_embedding_path",
    "pancreas_figure_path",
    "pancreas_file_prefix",
    "pancreas_method_tag",
    "pancreas_reference_stem",
    "pancreas_result_stem",
    "pancreas_velocity_cache_path",
    "pancreas_velocity_inputs_path",
    "resolve_pancreas_embedding_path",
    "save_pancreas_embedding",
]
