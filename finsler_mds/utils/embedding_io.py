"""Small helpers for cached embeddings and experiment arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_npz_inputs(path, inputs):
    np.savez(path, **inputs)


def load_npz_inputs(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_embedding_result(path, *, embedding, stress, full_geodesic_stress, init, cell_ids, metadata):
    np.savez(
        path,
        embedding=np.asarray(embedding, dtype=float),
        stress=np.asarray(stress, dtype=float),
        full_geodesic_stress=np.asarray(full_geodesic_stress, dtype=float),
        init=np.asarray(init, dtype=float),
        cell_ids=np.asarray(cell_ids, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def resolve_finsler_init(init_finsler_mds, *, umap_init, embedding_sources, n_samples, n_components, cell_ids):
    init_kind = normalize_finsler_init_kind(init_finsler_mds)
    if init_kind == "umap":
        init = adapt_embedding_dimension(umap_init, n_components)
        if init.shape[0] != n_samples:
            raise ValueError(f"UMAP init has {init.shape[0]} samples, expected {n_samples}.")
        return init, "UMAP init", init_kind

    source_path = embedding_sources.get(init_kind)
    if source_path is None or not source_path.exists():
        raise FileNotFoundError(
            f"Requested init_finsler_mds={init_kind!r}, but no saved embedding was found."
        )

    init = adapt_embedding_dimension(load_saved_embedding(source_path, cell_ids=cell_ids), n_components)
    expected_shape = (n_samples, n_components)
    if init.shape != expected_shape:
        raise ValueError(f"Saved {init_kind} init has shape {init.shape}, expected {expected_shape}: {source_path}")
    return init, f"saved {init_kind} ({source_path})", init_kind


def normalize_finsler_init_kind(init_finsler_mds):
    if not isinstance(init_finsler_mds, str):
        raise TypeError("init_finsler_mds must be one of {'umap', 'smacof', 'path_frozen'}.")
    init_kind = init_finsler_mds.lower().replace("-", "_")
    if init_kind in {"umap", "umap_2d", "umap2d"}:
        return "umap"
    if init_kind in {"smacof", "randers_smacof", "smacof_randers"}:
        return "smacof"
    if init_kind in {"path_frozen", "frozen_paths"}:
        return "path_frozen"
    raise ValueError("init_finsler_mds must be one of {'umap', 'smacof', 'path_frozen'}.")


def adapt_embedding_dimension(embedding, n_components):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError("Saved/init embedding must be a 2D array.")
    if embedding.shape[1] == n_components:
        return embedding
    if embedding.shape[1] > n_components:
        return embedding[:, :n_components]

    init = np.zeros((embedding.shape[0], n_components), dtype=float)
    init[:, : embedding.shape[1]] = embedding
    return init


def latest_compatible_embedding_path(directory, pattern, *, n_samples, cell_ids):
    candidates = sorted(Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return next(
        (path for path in candidates if saved_embedding_is_compatible(path, n_samples=n_samples, cell_ids=cell_ids)),
        None,
    )


def saved_embedding_is_compatible(path, *, n_samples, cell_ids):
    try:
        return load_saved_embedding(path, cell_ids=cell_ids).shape[0] == n_samples
    except (ValueError, KeyError, FileNotFoundError, OSError):
        return False


def load_saved_embedding(path, *, cell_ids=None):
    path = Path(path)
    if path.suffix == ".npy":
        embedding = np.asarray(np.load(path), dtype=float)
        if cell_ids is not None and embedding.shape[0] != len(cell_ids):
            raise ValueError(f"Saved embedding has {embedding.shape[0]} cells, expected {len(cell_ids)}: {path}")
        return embedding

    if path.suffix != ".npz":
        raise ValueError(f"Unsupported saved embedding extension: {path}")

    with np.load(path) as data:
        if "embedding" not in data:
            raise KeyError(f"Saved embedding file has no 'embedding' array: {path}")
        embedding = np.asarray(data["embedding"], dtype=float)
        if cell_ids is None:
            return embedding
        if embedding.shape[0] == len(cell_ids) and "cell_ids" not in data:
            return embedding
        if "cell_ids" not in data:
            raise ValueError(f"Saved embedding has no cell_ids and incompatible size: {path}")
        saved_cell_ids = np.asarray(data["cell_ids"]).astype(str)
        return align_embedding_to_cell_ids(embedding, saved_cell_ids=saved_cell_ids, target_cell_ids=cell_ids)


def align_embedding_to_cell_ids(embedding, *, saved_cell_ids, target_cell_ids):
    target_cell_ids = np.asarray(target_cell_ids).astype(str)
    if np.array_equal(saved_cell_ids, target_cell_ids):
        return embedding

    positions = {cell_id: idx for idx, cell_id in enumerate(saved_cell_ids)}
    try:
        order = np.asarray([positions[cell_id] for cell_id in target_cell_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Saved embedding is missing target cell id {exc.args[0]!r}.") from exc
    return embedding[order]


def scale_embedding_to_dissimilarities(embedding, dissimilarities, *, random_state, n_pairs=50_000):
    embedding = np.asarray(embedding, dtype=float)
    D = np.asarray(dissimilarities, dtype=float)
    X = embedding - embedding.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(random_state)
    rows = rng.integers(0, len(X), size=n_pairs)
    cols = rng.integers(0, len(X), size=n_pairs)
    valid = rows != cols
    rows, cols = rows[valid], cols[valid]
    target = D[rows, cols]
    current = np.linalg.norm(X[rows] - X[cols], axis=1)
    valid = np.isfinite(target) & (target > 0) & (current > 0)
    if not np.any(valid):
        return X, 1.0

    scale = float(np.median(target[valid] / current[valid]))
    return (X * scale, scale) if np.isfinite(scale) and scale > 0 else (X, 1.0)


def save_summary(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def cache_token(value):
    return str(value).replace("-", "m").replace(".", "p")


def metric_alpha_tag(alpha):
    alpha = float(alpha)
    return str(int(alpha)) if alpha.is_integer() else cache_token(alpha)


def validate_saved_outputs(paths):
    missing = [Path(path) for path in paths if not Path(path).exists() or Path(path).stat().st_size == 0]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"Output generation failed, missing or empty files:\n{formatted}")
