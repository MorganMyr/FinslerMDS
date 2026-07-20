"""Path helpers for pancreas embeddings and figures."""

from __future__ import annotations

from pathlib import Path
import re


PREFIX_TO_METHOD = (
    ("s_new_", "new_smacof"),
    ("s_old_", "old_smacof"),
    ("smacof_", "smacof"),
    ("sklearn_mds_", "sklearn_mds"),
    ("gd_", "gradient_descent"),
    ("fumap_", "finsler_umap"),
    ("pf_", "path_frozen"),
    ("sbf_", "soft_bf"),
    ("umap_", "umap"),
    ("isomap_", "isomap"),
)

FAMILY_TO_METHOD = {
    "gd": "gradient_descent",
    "gradient_descent": "gradient_descent",
    "fumap": "finsler_umap",
    "finsler_umap": "finsler_umap",
    "pf": "path_frozen",
    "path_frozen": "path_frozen",
    "sbf": "soft_bf",
    "soft_bf": "soft_bf",
    "smacof": "new_smacof",
    "s_new": "new_smacof",
    "s_old": "old_smacof",
    "umap": "umap",
    "isomap": "isomap",
}


def pancreas_artifact_method(filename: str) -> str | None:
    name = Path(filename).name
    if Path(name).stem in {"umap", "isomap"}:
        return Path(name).stem
    for prefix, method in PREFIX_TO_METHOD:
        if name.startswith(prefix):
            return method
    return None


def pancreas_artifact_dimension(filename: str) -> int | None:
    stem = Path(filename).stem.lower()
    if re.search(r"(^|_)3d(_|$)", stem):
        return 3
    if re.search(r"(^|_)2d(_|$)", stem):
        return 2
    if stem in {"umap", "isomap"} or stem.startswith(("umap_", "isomap_")):
        return 2
    return None


def pancreas_artifact_dir(base_dir, filename: str):
    method = pancreas_artifact_method(filename)
    dim = pancreas_artifact_dimension(filename)
    if method is None or dim is None:
        return Path(base_dir)
    return Path(base_dir) / method / f"{dim}D"


def pancreas_raw_embedding_path(raw_dir, filename: str):
    path = pancreas_artifact_dir(raw_dir, filename) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pancreas_velocity_cache_dir(raw_dir):
    path = Path(raw_dir) / "velocity_inputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pancreas_velocity_cache_path(raw_dir, filename: str):
    return pancreas_velocity_cache_dir(raw_dir) / Path(filename).name


def pancreas_figure_path(pancreas_dir, filename: str):
    path = pancreas_artifact_dir(Path(pancreas_dir) / "figure", filename) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_pancreas_embedding_path(value: str, raw_dir):
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        raw_dir = Path(raw_dir)
        candidates.extend([
            raw_dir / path,
            pancreas_raw_embedding_path(raw_dir, path.name),
        ])
        if path.suffix == "":
            for suffix in (".npz", ".npy"):
                candidates.extend([
                    raw_dir / path.with_suffix(suffix),
                    pancreas_raw_embedding_path(raw_dir, path.with_suffix(suffix).name),
                ])
    for candidate in candidates:
        if candidate.exists() and candidate.suffix in {".npz", ".npy"}:
            return candidate
    raw_dir = Path(raw_dir)
    for suffix in ([path.suffix] if path.suffix else [".npz", ".npy"]):
        matches = list(raw_dir.glob(f"*/*/{path.stem}{suffix}"))
        if matches:
            return max(matches, key=lambda item: item.stat().st_mtime)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Embedding file not found. Searched: {searched}")


def latest_pancreas_embedding_path(raw_dir, family: str, embedding_dim: int, dataset_prefix=""):
    raw_dir = Path(raw_dir)
    method = FAMILY_TO_METHOD.get(family, family)
    prefix = "" if dataset_prefix == "pancreas" else (f"{dataset_prefix}_" if dataset_prefix else "")
    dim_patterns = [[f"{prefix}{family}_{int(embedding_dim)}d_*.npz"]]
    if int(embedding_dim) == 3:
        dim_patterns.append([f"{prefix}{family}_2d_*.npz"])
    candidate_groups = []
    for patterns in dim_patterns:
        candidates = []
        for pattern in patterns:
            dim = 2 if "_2d_" in pattern else int(embedding_dim)
            candidates.extend((raw_dir / method / f"{dim}D").glob(pattern))
            candidates.extend(raw_dir.glob(pattern))
        candidate_groups.append(candidates)
    if family == "smacof":
        for short_family, short_method in (("s_new", "new_smacof"), ("s_old", "old_smacof")):
            patterns = [[f"{prefix}{short_family}_{int(embedding_dim)}d_*.npz"]]
            if int(embedding_dim) == 3:
                patterns.append([f"{prefix}{short_family}_2d_*.npz"])
            for pattern_group in patterns:
                candidates = []
                for pattern in pattern_group:
                    dim = 2 if "_2d_" in pattern else int(embedding_dim)
                    candidates.extend((raw_dir / short_method / f"{dim}D").glob(pattern))
                    candidates.extend(raw_dir.glob(pattern))
                candidate_groups.append(candidates)
    for candidates in candidate_groups:
        if candidates:
            return max(set(candidates), key=lambda path: path.stat().st_mtime)
    return None


__all__ = [
    "pancreas_artifact_dimension",
    "pancreas_artifact_dir",
    "pancreas_artifact_method",
    "pancreas_figure_path",
    "pancreas_raw_embedding_path",
    "pancreas_velocity_cache_dir",
    "pancreas_velocity_cache_path",
    "resolve_pancreas_embedding_path",
    "latest_pancreas_embedding_path",
]
