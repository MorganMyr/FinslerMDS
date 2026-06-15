"""Shared helpers for the pancreas RNA-velocity dataset."""

from __future__ import annotations

import os
import warnings

import numpy as np
from scipy import sparse


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
    "alpha": "#7b3294",
    "beta": "#08306b",
    "delta": "#28b7c9",
    "ductal": "#c51b7d",
    "epsilon": "#74c476",
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


def cluster_balanced_pair_weights(
        labels,
        dissimilarities=None,
        *,
        rho=0.5,
        return_info=False,
):
    """Return pair weights that soften cluster-size imbalance.

    Each cell receives weight ``(median_cluster_size / cluster_size) ** rho``.
    Pair weights are the product of source and target cell weights. With
    ``rho=1``, each directed cluster-to-cluster block has comparable total
    weight; ``rho=0`` gives uniform weights.
    """
    labels = np.asarray(labels, dtype=str)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if len(labels) == 0:
        raise ValueError("labels must not be empty.")
    rho = float(rho)
    if rho < 0:
        raise ValueError("rho must be non-negative.")

    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_ref = float(np.median(counts))
    cell_weights = (n_ref / counts[inverse].astype(float)) ** rho
    weights = np.outer(cell_weights, cell_weights)
    np.fill_diagonal(weights, 0.0)

    active = ~np.eye(len(labels), dtype=bool)
    if dissimilarities is not None:
        D = np.asarray(dissimilarities, dtype=float)
        if D.shape != weights.shape:
            raise ValueError("dissimilarities must have shape (n_cells, n_cells).")
        active &= np.isfinite(D)

    info = {
        "rho": rho,
        "n_ref": n_ref,
        "cluster_sizes": {str(label): int(count) for label, count in zip(unique, counts)},
        "cluster_cell_weights": {
            str(label): float((n_ref / count) ** rho)
            for label, count in zip(unique, counts)
        },
        "mean_active_weight": float(np.mean(weights[active])) if np.any(active) else np.nan,
        "min_active_weight": float(np.min(weights[active])) if np.any(active) else np.nan,
        "max_active_weight": float(np.max(weights[active])) if np.any(active) else np.nan,
    }
    if return_info:
        return weights, info
    return weights


def apply_distance_pair_reweight(
        weights,
        dissimilarities,
        *,
        power=0.0,
        epsilon=1e-6,
        return_info=False,
):
    """Downweight large dissimilarities with a median-scaled inverse power.

    The multiplicative factor is
    ``(d_ij / median(d) + epsilon) ** (-power)`` on finite off-diagonal pairs.
    """
    power = float(power)
    epsilon = float(epsilon)
    if power < 0:
        raise ValueError("distance reweighting power must be non-negative.")
    if epsilon < 0:
        raise ValueError("distance reweighting epsilon must be non-negative.")

    weights = np.asarray(weights, dtype=float).copy()
    D = np.asarray(dissimilarities, dtype=float)
    if D.shape != weights.shape:
        raise ValueError("dissimilarities must have the same shape as weights.")

    active = np.isfinite(D) & (weights != 0)
    np.fill_diagonal(active, False)
    positive = active & (D > 0)
    median_delta = float(np.median(D[positive])) if np.any(positive) else np.nan
    if power == 0 or not np.isfinite(median_delta) or median_delta <= 0:
        info = {
            "power": power,
            "epsilon": epsilon,
            "enabled": False,
            "median_delta": median_delta,
        }
        return (weights, info) if return_info else weights

    factor = np.ones_like(D, dtype=float)
    factor[active] = np.power(D[active] / median_delta + epsilon, -power)
    weights[active] *= factor[active]

    info = {
        "power": power,
        "epsilon": epsilon,
        "enabled": True,
        "median_delta": median_delta,
        "mean_factor": float(np.mean(factor[active])) if np.any(active) else np.nan,
        "min_factor": float(np.min(factor[active])) if np.any(active) else np.nan,
        "max_factor": float(np.max(factor[active])) if np.any(active) else np.nan,
    }
    return (weights, info) if return_info else weights


def normalize_pair_weights(
        weights,
        dissimilarities=None,
):
    """Normalize the final active pair-weight matrix to mean one."""
    weights = np.asarray(weights, dtype=float).copy()
    active = weights != 0
    np.fill_diagonal(active, False)
    if dissimilarities is not None:
        D = np.asarray(dissimilarities, dtype=float)
        if D.shape != weights.shape:
            raise ValueError("dissimilarities must have the same shape as weights.")
        active &= np.isfinite(D)

    normalizer = float(np.mean(weights[active])) if np.any(active) else 0.0
    if normalizer > 0:
        weights[active] /= normalizer
    return weights, normalizer


def apply_frontier_pair_weight(
        weights,
        boundary_plan,
        *,
        factor,
        dissimilarities=None,
        symmetric=True,
        return_info=False,
):
    """Multiply weights of CBDir frontier neighbor pairs.

    ``boundary_plan`` is the precomputed CBDir plan. Its pairs are directed
    source-cluster cell -> target-cluster neighbor pairs. With ``symmetric``,
    the reverse pairs receive the same multiplier.
    """
    factor = float(factor)
    if factor <= 0:
        raise ValueError("frontier pair weight factor must be positive.")
    weights = np.asarray(weights, dtype=float).copy()
    if factor == 1.0:
        info = {"factor": factor, "enabled": False, "n_directed_pairs": 0, "n_weighted_pairs": 0}
        return (weights, info) if return_info else weights

    pair_mask = np.zeros(weights.shape, dtype=bool)
    for transition in boundary_plan.transitions.values():
        for pos, source in enumerate(transition.cell_indices):
            start = transition.target_indptr[pos]
            end = transition.target_indptr[pos + 1]
            targets = transition.target_indices[start:end]
            pair_mask[int(source), targets] = True

    directed_pair_count = int(np.count_nonzero(pair_mask))
    if symmetric:
        pair_mask |= pair_mask.T
    if dissimilarities is not None:
        D = np.asarray(dissimilarities, dtype=float)
        if D.shape != weights.shape:
            raise ValueError("dissimilarities must have shape (n_cells, n_cells).")
        pair_mask &= np.isfinite(D)
    np.fill_diagonal(pair_mask, False)
    weights[pair_mask] *= factor

    info = {
        "factor": factor,
        "enabled": True,
        "symmetric": bool(symmetric),
        "n_directed_pairs": directed_pair_count,
        "n_weighted_pairs": int(np.count_nonzero(pair_mask)),
    }
    if return_info:
        return weights, info
    return weights


def setup_scvelo_settings():
    """Configure scVelo plotting/verbosity and return the module."""
    suppress_pancreas_noise_warnings()
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")
    return scv


def pancreas_preprocessing_neighbors(preprocessing):
    if "moments_n_neighbors" in preprocessing:
        return int(preprocessing["moments_n_neighbors"])
    return int(preprocessing["n_neighbors"])


def preprocess_pancreas_for_velocity(adata, preprocessing, *, seed):
    """Run the Scanpy/scVelo preprocessing used before velocity estimation."""
    import scanpy as sc

    scv = setup_scvelo_settings()
    n_neighbors = pancreas_preprocessing_neighbors(preprocessing)
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
