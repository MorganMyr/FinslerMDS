"""Shared helpers for the pancreas RNA-velocity dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse

from finsler_mds.metrics import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)

from .embedding_io import cache_token, metric_alpha_tag
from .graph import compute_velocity_dist_matrix
from .initialization import IsomapWithPreds
from .orientation import rotate_embedding_to_mean_velocity_down, rotation_to_down_axis
from .pancreas_files import (
    load_pancreas_embedding,
    pancreas_embedding_path,
    pancreas_reference_stem,
    pancreas_velocity_inputs_path,
    save_pancreas_embedding,
)


PANCREAS_DATASET_SOURCE = "cellrank.datasets.pancreas"
PANCREAS_CLUSTER_ALIASES = {
    "Fev+": "Pre-endocrine",
}
PANCREAS_TRANSITIONS = [
    ("Ngn3 high EP", "Pre-endocrine"),
    ("Pre-endocrine", "Alpha"),
    ("Pre-endocrine", "Beta"),
    ("Pre-endocrine", "Delta"),
    ("Pre-endocrine", "Epsilon"),
]
PANCREAS_CLUSTER_COLORS = {
    "Alpha": "#7b3294",
    "Beta": "#08306b",
    "Delta": "#28b7c9",
    "Ductal": "#c51b7d",
    "Epsilon": "#74c476",
    "Ngn3 high EP": "#f28e2b",
    "Ngn3 low EP": "#d62728",
    "Pre-endocrine": "#f1c40f",
}
PANCREAS_N_EVAL_NEIGHBORS = 30
PANCREAS_PREPROCESSING = {
    "min_shared_counts": 20,
    "n_top_genes": 3000,
    "n_pcs": 50,
    "moments_n_neighbors": 30,
}


def suppress_pancreas_noise_warnings():
    """Hide noisy third-party warnings that do not affect pancreas runs."""
    filters = [
        "ignore::DeprecationWarning",
    ]
    existing = os.environ.get("PYTHONWARNINGS")
    if existing:
        present = set(existing.split(","))
        filters = [item for item in filters if item not in present]
        if filters:
            os.environ["PYTHONWARNINGS"] = existing + "," + ",".join(filters)
    else:
        os.environ["PYTHONWARNINGS"] = ",".join(filters)
    warnings.filterwarnings(
        "ignore",
        message="This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*lib/python.*/multiprocessing/popen_fork.py.*use of fork\\(\\) may lead to deadlocks.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*lib/python.*/site-packages/joblib/.*use of fork\\(\\) may lead to deadlocks.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="'(oneOf|parseString|resetCache|enablePackrat)' deprecated - use '.*'",
    )
    warnings.filterwarnings(
        "ignore",
        message="In .*matplotlib.*: '(oneOf|parseString|resetCache|enablePackrat)' deprecated - use '.*'",
    )
    try:
        from pyparsing.exceptions import PyparsingDeprecationWarning
    except Exception:
        pass
    else:
        warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)


def load_pancreas_dataset(*, canonicalize_clusters=True):
    """Load the CellRank endocrine pancreas dataset used in recent examples."""
    suppress_pancreas_noise_warnings()
    try:
        import cellrank as cr
    except ImportError as exc:
        raise ImportError(
            "CellRank is required for the pancreas scripts. "
            "Install it with `pip install cellrank`."
        ) from exc

    adata = cr.datasets.pancreas()
    adata.uns["finsler_mds_dataset_source"] = PANCREAS_DATASET_SOURCE
    if canonicalize_clusters:
        canonicalize_pancreas_clusters(adata)
    return adata


def canonicalize_pancreas_clusters(adata, *, aliases=None):
    """Normalize CellRank labels to the names used by our pancreas metrics."""
    aliases = dict(PANCREAS_CLUSTER_ALIASES if aliases is None else aliases)
    for key in ("clusters", "clusters_coarse", "clusters_fine"):
        if key not in adata.obs:
            continue
        values = np.asarray(adata.obs[key].astype(str), dtype=object)
        values = np.asarray([aliases.get(value, value) for value in values], dtype=str)
        adata.obs[key] = values
    return adata


def setup_scvelo_settings():
    """Configure scVelo plotting/verbosity and return the module."""
    suppress_pancreas_noise_warnings()
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")
    return scv


def preprocess_pancreas_for_velocity(adata, preprocessing, *, seed):
    """Run the Scanpy/scVelo preprocessing used before velocity estimation."""
    import scanpy as sc

    scv = setup_scvelo_settings()
    n_neighbors = int(preprocessing["moments_n_neighbors"])
    n_pcs = int(preprocessing["n_pcs"])

    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=preprocessing["min_shared_counts"],
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=preprocessing["n_top_genes"],
        flavor="seurat",
        subset=True,
    )
    sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    return adata


def compute_pancreas_velocity(
        adata,
        *,
        mode,
        recover_dynamics_max_iter=20,
        recover_dynamics_n_jobs=-1,
        show_progress_bar=False,
):
    """Compute scVelo velocities on a preprocessed pancreas AnnData."""
    scv = setup_scvelo_settings()
    if mode == "dynamical":
        scv.tl.recover_dynamics(
            adata,
            max_iter=recover_dynamics_max_iter,
            n_jobs=recover_dynamics_n_jobs,
            show_progress_bar=show_progress_bar,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion of an array with ndim > 0 to a scalar is deprecated.*",
            category=DeprecationWarning,
            module="scvelo.tools.optimization",
        )
        scv.tl.velocity(adata, mode=mode)
    return adata


def compute_pancreas_velocity_graph(
        adata,
        *,
        n_neighbors=None,
        n_jobs=-1,
        show_progress_bar=False,
):
    scv = setup_scvelo_settings()
    kwargs = {"n_jobs": n_jobs, "show_progress_bar": show_progress_bar}
    if n_neighbors is not None:
        kwargs["n_neighbors"] = n_neighbors
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks.*",
            category=DeprecationWarning,
        )
        scv.tl.velocity_graph(adata, **kwargs)
    return adata


def neighbor_indices_from_sparse_distances(distances, *, n_neighbors):
    """Return sorted neighbor indices from a Scanpy/scVelo sparse distance graph."""
    distances = sparse.csr_matrix(distances)
    n_neighbors = int(n_neighbors)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    neighbors = np.full((distances.shape[0], n_neighbors), -1, dtype=int)
    for row_idx in range(distances.shape[0]):
        start, end = distances.indptr[row_idx], distances.indptr[row_idx + 1]
        cols = distances.indices[start:end]
        data = distances.data[start:end]
        keep = cols != row_idx
        cols = cols[keep]
        data = data[keep]
        if len(cols) == 0:
            continue
        order = np.argsort(data, kind="stable")[:n_neighbors]
        neighbors[row_idx, :len(order)] = cols[order]
    return neighbors


def project_velocity_to_pca(adata, n_pcs):
    """Project scVelo gene-space velocities onto Scanpy/scVelo PCA axes."""
    if "velocity" not in adata.layers:
        raise ValueError("adata.layers['velocity'] is missing. Run scVelo velocity first.")
    if "PCs" not in adata.varm:
        raise ValueError("adata.varm['PCs'] is missing. Run PCA before projecting velocity.")

    velocity = adata.layers["velocity"]
    if sparse.issparse(velocity):
        velocity = velocity.toarray()
    velocity = np.asarray(velocity, dtype=float)
    velocity = np.nan_to_num(velocity, copy=False)

    pcs = np.asarray(adata.varm["PCs"][:, :n_pcs], dtype=float)
    return np.asarray(velocity @ pcs, dtype=float)


def project_velocity_to_embedding(adata, embedding, *, basis="eval"):
    """Use scVelo's local projection to express velocities in an embedding."""
    scv = setup_scvelo_settings()
    key = f"velocity_{basis}"
    adata.obsm[f"X_{basis}"] = np.asarray(embedding, dtype=float)
    if key in adata.obsm:
        del adata.obsm[key]
    scv.tl.velocity_embedding(adata, basis=basis)
    return np.asarray(adata.obsm[key], dtype=float)


@dataclass
class PancreasInputs:
    """Arrays needed by pancreas embeddings, plus the reusable state cache."""

    dissimilarities: np.ndarray
    labels: np.ndarray
    cell_ids: np.ndarray
    original_indices: np.ndarray
    state_path: Path
    state_metadata: dict
    adata: object | None = None


def normalize_velocity_distance_formula(distance_formula):
    if not isinstance(distance_formula, str):
        raise TypeError("velocity distance formula must be 'randers' or 'matsumoto'.")
    aliases = {
        "randers": "randers",
        "mats": "matsumoto",
        "matsumoto": "matsumoto",
    }
    try:
        return aliases[distance_formula.lower()]
    except KeyError as exc:
        raise ValueError("velocity distance formula must be 'randers' or 'matsumoto'.") from exc


def velocity_distance_formula_tag(distance_formula, alpha=None):
    prefix = {
        "randers": "vrand",
        "matsumoto": "vmats",
    }[normalize_velocity_distance_formula(distance_formula)]
    return prefix if alpha is None else f"{prefix}{metric_alpha_tag(alpha)}"


def normalize_embedding_metric_kind(kind):
    if not isinstance(kind, str):
        raise TypeError("metric kind must be a string.")
    aliases = {
        "r": "randers",
        "randers": "randers",
        "m": "matsumoto",
        "mats": "matsumoto",
        "matsumoto": "matsumoto",
        "cm": "convexified_matsumoto",
        "cmats": "convexified_matsumoto",
        "convexified_matsumoto": "convexified_matsumoto",
        "convexifiedmatsumoto": "convexified_matsumoto",
    }
    try:
        return aliases[kind.lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(
            "metric kind must be 'randers', 'matsumoto', or 'convexified_matsumoto'."
        ) from exc


def make_embedding_metric(kind, alpha=0.0):
    metric_class = {
        "randers": RandersMetric,
        "matsumoto": MatsumotoMetric,
        "convexified_matsumoto": ConvexifiedMatsumotoMetric,
    }[normalize_embedding_metric_kind(kind)]
    return metric_class(alpha=float(alpha))


def embedding_metric_tag(metric):
    if isinstance(metric, RandersMetric):
        prefix = "r"
    elif isinstance(metric, ConvexifiedMatsumotoMetric):
        prefix = "cmats"
    elif isinstance(metric, MatsumotoMetric):
        prefix = "mats"
    else:
        raise TypeError(f"Unsupported embedding metric {type(metric).__name__}.")
    return f"{prefix}{metric_alpha_tag(metric.alpha)}"


def metric_display_name(metric):
    name = type(metric).__name__.removesuffix("Metric")
    if name.startswith("Convexified"):
        name = name.replace("Convexified", "Convexified ", 1)
    return f"{name} alpha={metric.alpha:g}"


def normalize_embedding_dim(embedding_dim):
    embedding_dim = int(embedding_dim)
    if embedding_dim not in {2, 3}:
        raise ValueError("embedding_dim must be 2 or 3.")
    return embedding_dim


def labels_to_numpy(labels):
    if labels is None:
        return None
    if hasattr(labels, "to_numpy"):
        labels = labels.to_numpy()
    return np.asarray(labels, dtype=str)


def labels_to_cache(labels):
    return np.asarray([], dtype=str) if labels is None else np.asarray(labels, dtype=str)


def pancreas_cache_metadata(**params):
    return {key: _json_safe(value) for key, value in params.items()}


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def pancreas_state_cache_path(raw_dir, *, preprocessing, velocity, dataset_prefix="pancreas", seed=42):
    prefix = "" if dataset_prefix == "pancreas" else f"{dataset_prefix}_"
    return Path(raw_dir) / (
        f"{prefix}pancreas_campaign_state_{cache_token(velocity['mode'])}_"
        f"hvg{preprocessing['n_top_genes']}_pca{preprocessing['n_pcs']}_s{int(seed)}.h5ad"
    )


def load_pancreas_state_cache(cache_path, expected_metadata):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    print(f"Loading cached pancreas velocity state: {cache_path}")
    import scanpy as sc

    adata = sc.read_h5ad(cache_path)
    if not pancreas_state_cache_matches(adata, expected_metadata):
        print("Cached pancreas velocity state has different parameters; recomputing.")
        return None
    _validate_pancreas_state(adata, cache_path)
    labels = labels_to_numpy(adata.obs.get("clusters"))
    cell_ids = np.asarray(adata.obs_names, dtype=str)
    original_indices = np.asarray(
        adata.obs.get("finsler_mds_original_index", np.arange(adata.n_obs)),
        dtype=int,
    )
    return adata, labels, cell_ids, original_indices


def pancreas_state_cache_matches(adata, expected_metadata):
    metadata_json = adata.uns.get("finsler_mds_state_metadata_json")
    if metadata_json is None:
        return False
    try:
        return json.loads(str(metadata_json)) == expected_metadata
    except json.JSONDecodeError:
        return False


def _validate_pancreas_state(adata, cache_path):
    required = {
        "obs['clusters']": "clusters" in adata.obs,
        "obsm['X_pca']": "X_pca" in adata.obsm,
        "layers['velocity']": "velocity" in adata.layers,
        "varm['PCs']": "PCs" in adata.varm,
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise ValueError(f"Invalid pancreas state cache {cache_path}: missing {', '.join(missing)}.")


def save_pancreas_state_cache(
    cache_path,
    adata,
    *,
    labels,
    cell_ids,
    original_indices,
    metadata,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    adata = adata.copy()
    adata.obs_names = np.asarray(cell_ids, dtype=str)
    if labels is not None:
        adata.obs["clusters"] = np.asarray(labels, dtype=str)
    adata.obs["finsler_mds_original_index"] = np.asarray(original_indices, dtype=int)
    adata.uns["finsler_mds_state_metadata_json"] = json.dumps(metadata, sort_keys=True)
    adata.write_h5ad(cache_path)
    print(f"Saved pancreas velocity state: {cache_path}")


def pancreas_state_metadata(preprocessing, velocity, seed, **extra):
    return pancreas_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=preprocessing["min_shared_counts"],
        n_top_genes=preprocessing["n_top_genes"],
        n_pcs=preprocessing["n_pcs"],
        moments_n_neighbors=preprocessing["moments_n_neighbors"],
        velocity_mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        seed=seed,
        **extra,
    )


def load_or_compute_pancreas_state(
    raw_dir,
    *,
    preprocessing,
    velocity,
    seed=42,
    dataset_prefix="pancreas",
    prepare_dataset=None,
    metadata_extra=None,
):
    metadata_extra = dict(metadata_extra or {})
    metadata = pancreas_state_metadata(preprocessing, velocity, seed, **metadata_extra)
    path = pancreas_state_cache_path(
        raw_dir,
        preprocessing=preprocessing,
        velocity=velocity,
        dataset_prefix=dataset_prefix,
        seed=seed,
    )
    cached = load_pancreas_state_cache(path, metadata)
    if cached is not None:
        return (*cached, path, metadata)

    print(f"Loading pancreas dataset from {PANCREAS_DATASET_SOURCE}")
    adata = load_pancreas_dataset()
    labels = labels_to_numpy(adata.obs.get("clusters"))
    cell_ids = np.asarray(adata.obs_names, dtype=str)
    original_indices = np.arange(adata.n_obs, dtype=int)
    if prepare_dataset is not None:
        adata, labels, cell_ids, original_indices = prepare_dataset(adata)
    preprocess_pancreas_for_velocity(adata, preprocessing, seed=seed)
    compute_pancreas_velocity(
        adata,
        mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
    )
    save_pancreas_state_cache(
        path,
        adata,
        labels=labels,
        cell_ids=cell_ids,
        original_indices=original_indices,
        metadata=metadata,
    )
    return adata, labels, cell_ids, original_indices, path, metadata


def pancreas_distance_cache_metadata(preprocessing, velocity, seed, **extra):
    return pancreas_cache_metadata(
        **pancreas_state_metadata(preprocessing, velocity, seed, **extra),
        velocity_distance_formula=normalize_velocity_distance_formula(velocity["distance_formula"]),
        velocity_alpha=velocity["alpha"],
        velocity_cos_clip=velocity["cos_clip"],
        velocity_neighbors=velocity["velocity_neighbors"],
        velocity_kNN_euclid=velocity["kNN_euclid"],
        velocity_kNN_finsler=velocity["kNN_finsler"],
        average_velocity=velocity["average_velocity"],
        symmetrize_velocity_support=velocity["symmetrize_support"],
    )


def load_pancreas_distance_cache(path, expected_metadata):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as cache:
        if "metadata_json" not in cache:
            return None
        if json.loads(str(cache["metadata_json"].item())) != expected_metadata:
            return None
        dissimilarities = np.asarray(cache["dists_velocity"], dtype=float)
        labels = np.asarray(cache["labels"], dtype=str)
        cell_ids = np.asarray(cache["cell_ids"], dtype=str)
        original_indices = np.asarray(cache["original_indices"], dtype=int)
    print(f"Loaded pancreas velocity dissimilarities: {path}")
    return dissimilarities, labels, cell_ids, original_indices


def save_pancreas_distance_cache(
    path,
    dissimilarities,
    *,
    labels,
    cell_ids,
    original_indices,
    metadata,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dists_velocity=np.asarray(dissimilarities, dtype=float),
        labels=labels_to_cache(labels),
        cell_ids=np.asarray(cell_ids, dtype=str),
        original_indices=np.asarray(original_indices, dtype=int),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Saved pancreas velocity dissimilarities: {path}")


def load_or_compute_pancreas_inputs(
    raw_dir,
    *,
    preprocessing,
    velocity,
    seed=42,
    dataset_prefix="pancreas",
    state=None,
    prepare_dataset=None,
    metadata_extra=None,
):
    velocity = dict(velocity)
    velocity["distance_formula"] = normalize_velocity_distance_formula(velocity["distance_formula"])
    metadata_extra = dict(metadata_extra or {})
    metadata = pancreas_distance_cache_metadata(
        preprocessing,
        velocity,
        seed,
        **metadata_extra,
    )
    distance_path = pancreas_velocity_inputs_path(
        raw_dir,
        velocity=velocity,
        dataset_prefix=dataset_prefix,
        seed=seed,
    )
    state_path = pancreas_state_cache_path(
        raw_dir,
        preprocessing=preprocessing,
        velocity=velocity,
        dataset_prefix=dataset_prefix,
        seed=seed,
    )
    state_metadata = pancreas_state_metadata(preprocessing, velocity, seed, **metadata_extra)
    adata = None
    if state is not None:
        adata, _, state_cell_ids, _, state_path, state_metadata = state
    cached = load_pancreas_distance_cache(distance_path, metadata)
    if cached is not None:
        dissimilarities, labels, cell_ids, original_indices = cached
        if state is not None and not np.array_equal(cell_ids, state_cell_ids):
            raise ValueError("Pancreas state and distance cache contain different cells.")
        return PancreasInputs(
            dissimilarities,
            labels,
            cell_ids,
            original_indices,
            state_path,
            state_metadata,
            adata,
        )

    if state is None:
        adata, labels, cell_ids, original_indices, state_path, state_metadata = (
            load_or_compute_pancreas_state(
                raw_dir,
                preprocessing=preprocessing,
                velocity=velocity,
                seed=seed,
                dataset_prefix=dataset_prefix,
                prepare_dataset=prepare_dataset,
                metadata_extra=metadata_extra,
            )
        )
    else:
        _, labels, cell_ids, original_indices, _, _ = state
    x_pca = np.asarray(adata.obsm["X_pca"][:, :preprocessing["n_pcs"]], dtype=float)
    velocity_pca = project_velocity_to_pca(adata, preprocessing["n_pcs"])
    print("Building directed velocity dissimilarities")
    dissimilarities, _, _, _ = compute_velocity_dist_matrix(
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
    save_pancreas_distance_cache(
        distance_path,
        dissimilarities,
        labels=labels,
        cell_ids=np.asarray(cell_ids, dtype=str),
        original_indices=np.asarray(original_indices, dtype=int),
        metadata=metadata,
    )
    return PancreasInputs(
        np.asarray(dissimilarities, dtype=float),
        labels,
        cell_ids,
        original_indices,
        state_path,
        state_metadata,
        adata,
    )


def load_inputs_state(
        inputs,
        *,
        preprocessing=None,
        velocity=None,
        seed=42,
        dataset_prefix="pancreas",
):
    if inputs.adata is not None:
        return inputs.adata
    cached = load_pancreas_state_cache(inputs.state_path, inputs.state_metadata)
    if cached is None:
        if preprocessing is None or velocity is None:
            raise FileNotFoundError(f"Pancreas state cache is unavailable: {inputs.state_path}")
        cached = load_or_compute_pancreas_state(
            inputs.state_path.parent,
            preprocessing=preprocessing,
            velocity=velocity,
            seed=seed,
            dataset_prefix=dataset_prefix,
        )
        adata, _, cell_ids, _, inputs.state_path, inputs.state_metadata = cached
        if not np.array_equal(cell_ids, inputs.cell_ids):
            raise ValueError("Rebuilt pancreas state does not match the distance-cache cells.")
        inputs.adata = adata
    else:
        inputs.adata = cached[0]
    return inputs.adata


def compute_pancreas_umap(adata, *, preprocessing, umap, n_components, seed):
    import scanpy as sc

    sc.pp.neighbors(
        adata,
        n_neighbors=umap["n_neighbors"],
        n_pcs=preprocessing["n_pcs"],
        random_state=seed,
    )
    sc.tl.umap(
        adata,
        n_components=n_components,
        min_dist=umap["min_dist"],
        spread=umap["spread"],
        maxiter=umap["maxiter"],
        negative_sample_rate=umap["negative_sample_rate"],
        init_pos=umap["init_pos"],
        random_state=seed,
    )
    return np.asarray(adata.obsm["X_umap"], dtype=float)


def orient_pancreas_embedding_by_velocity(adata, embedding, *, velocity, label):
    embedding = np.asarray(embedding, dtype=float)
    if "velocity_graph" not in adata.uns:
        compute_pancreas_velocity_graph(
            adata,
            n_neighbors=velocity["velocity_neighbors"],
            n_jobs=velocity["graph_n_jobs"],
        )
    velocity_embedding = project_velocity_to_embedding(adata, embedding, basis="orientation")
    oriented, _, mean_velocity = rotate_embedding_to_mean_velocity_down(
        embedding,
        velocity_embedding,
    )
    oriented_mean = np.nanmean(velocity_embedding, axis=0) @ rotation_to_down_axis(mean_velocity).T
    print(
        f"Oriented {label}: {np.array2string(mean_velocity, precision=3)} -> "
        f"{np.array2string(oriented_mean, precision=3)}"
    )
    return oriented


def ensure_pancreas_reference_embedding(
    inputs,
    raw_dir,
    *,
    method,
    n_components,
    preprocessing,
    velocity,
    options,
    seed=42,
    dataset_prefix="pancreas",
):
    method = str(method).lower()
    n_neighbors = int(options["n_neighbors"])
    stem = pancreas_reference_stem(
        method,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=options.get("min_dist") if method == "umap" else None,
        dataset_prefix=dataset_prefix,
    )
    reference_metadata = pancreas_cache_metadata(
        method=method,
        n_components=n_components,
        state=pancreas_state_metadata(preprocessing, velocity, seed),
        velocity_neighbors=velocity["velocity_neighbors"],
        options=options,
    )
    path = pancreas_embedding_path(raw_dir, stem)
    if path.exists():
        try:
            return (
                load_pancreas_embedding(
                    path,
                    cell_ids=inputs.cell_ids,
                    expected_shape=(len(inputs.cell_ids), n_components),
                    expected_metadata=reference_metadata,
                ),
                path,
            )
        except (KeyError, OSError, ValueError):
            print(f"Ignoring incompatible {method} cache: {path}")

    adata = load_inputs_state(
        inputs,
        preprocessing=preprocessing,
        velocity=velocity,
        seed=seed,
        dataset_prefix=dataset_prefix,
    )
    if method == "umap":
        embedding = compute_pancreas_umap(
            adata,
            preprocessing=preprocessing,
            umap=options,
            n_components=n_components,
            seed=seed,
        )
    elif method == "isomap":
        embedding = IsomapWithPreds(
            n_components=n_components,
            n_neighbors=n_neighbors,
        ).fit_transform(np.asarray(adata.obsm["X_pca"][:, :preprocessing["n_pcs"]], dtype=float))
    else:
        raise ValueError("Reference method must be 'umap' or 'isomap'.")

    embedding = orient_pancreas_embedding_by_velocity(
        adata,
        embedding,
        velocity=velocity,
        label=f"{n_components}D {method.upper()}",
    )
    save_pancreas_embedding(
        path,
        embedding,
        inputs.cell_ids,
        metadata=reference_metadata,
    )
    return np.asarray(embedding, dtype=float), path
