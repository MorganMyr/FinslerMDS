"""Utilities for transforming sparse dissimilarity graphs."""

from __future__ import annotations

import numpy as np


def temporally_asymmetric_knn_distances(distances, *, pseudotime, lambda_time, min_factor):
    graph = distances.tocsr(copy=True).maximum(distances.T).tocoo(copy=True)
    pseudotime = np.asarray(pseudotime, dtype=float)
    lambda_time = float(lambda_time)
    min_factor = float(min_factor)
    if min_factor <= 0:
        raise ValueError("Temporal asymmetry min_factor must be positive.")

    data = np.asarray(graph.data, dtype=float).copy()
    finite = (data > 0) & np.isfinite(pseudotime[graph.row]) & np.isfinite(pseudotime[graph.col])
    delta_t = pseudotime[graph.col[finite]] - pseudotime[graph.row[finite]]
    base = data[finite]
    proposed = base - lambda_time * delta_t
    floor = min_factor * base
    floored = proposed < floor
    data[finite] = np.maximum(floor, proposed)
    graph.data = data
    return graph.tocsr(), {
        "lambda_time": lambda_time,
        "min_factor": min_factor,
        "finite_edges": int(np.sum(finite)),
        "floored_edges": int(np.sum(floored)),
        "floored_fraction": float(np.mean(floored)) if len(floored) else 0.0,
    }


def density_scaled_knn_distances(distances, *, gamma, mode="symmetric"):
    mode = normalize_density_scaling_mode(mode)
    graph = distances.tocsr(copy=True)
    if mode == "source":
        graph = graph.maximum(graph.T).tocsr(copy=True)

    sigmas = local_knn_scales(graph)
    finite = np.isfinite(sigmas) & (sigmas > 0)
    if not np.any(finite):
        raise ValueError("Cannot density-scale distances because all local scales are non-positive.")

    median_sigma = float(np.median(sigmas[finite]))
    sigmas = np.where(finite, sigmas, median_sigma)
    info = {
        "median_sigma": median_sigma,
        "min_sigma": float(np.min(sigmas)),
        "max_sigma": float(np.max(sigmas)),
    }
    gamma = float(gamma)
    if gamma == 0:
        return graph, info

    coo = graph.tocoo(copy=True)
    local_scale = np.sqrt(sigmas[coo.row] * sigmas[coo.col]) if mode == "symmetric" else sigmas[coo.row]
    factors = np.divide(median_sigma, local_scale, out=np.ones_like(coo.data, dtype=float), where=local_scale > 0)
    coo.data = coo.data * factors**gamma
    return coo.tocsr(), info


def local_knn_scales(graph):
    graph = graph.tocsr()
    sigmas = np.empty(graph.shape[0], dtype=float)
    for i in range(graph.shape[0]):
        row = graph.data[graph.indptr[i]:graph.indptr[i + 1]]
        positive = row[row > 0]
        sigmas[i] = np.max(positive) if len(positive) else np.nan
    return sigmas


def normalize_density_scaling_mode(mode):
    return _normalize_alias(
        mode,
        {"symmetric": "symmetric", "sym": "symmetric", "source": "source", "source_asymmetric": "source", "density": "source"},
        "density scaling mode must be one of {'symmetric', 'source'}.",
    )


def normalize_asymmetry_type(asymmetry_type):
    if asymmetry_type is None:
        return None
    return _normalize_alias(
        asymmetry_type,
        {
            "none": None,
            "symmetric": None,
            "sym": None,
            "pseudotime": "pseudotime",
            "time": "pseudotime",
            "pt": "pseudotime",
            "density": "density",
            "dens": "density",
            "den": "density",
        },
        "target_graph['asymmetry_type'] must be one of {None, 'pseudotime', 'density'}.",
    )


def _normalize_alias(value, aliases, error):
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        if key in aliases:
            return aliases[key]
    raise ValueError(error)
