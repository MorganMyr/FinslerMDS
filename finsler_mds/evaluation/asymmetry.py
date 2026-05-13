"""Asymmetry-preservation metrics for directed dissimilarities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


__all__ = [
    "AsymmetryPreservationResult",
    "asymmetry_preservation_from_neighbors",
    "asymmetry_preservation_from_pairs",
    "asymmetry_score",
    "neighbor_pairs",
    "summarize_asymmetry_preservation",
]


@dataclass(frozen=True)
class AsymmetryPreservationResult:
    """Summary of how well an embedding preserves directed asymmetry."""

    n_pairs: int
    n_strong_pairs: int
    tau: float
    sign_accuracy: float
    weighted_sign_accuracy: float
    spearman: float
    pearson: float
    gamma: float
    normalized_mse: float
    mean_abs_data_asymmetry: float
    mean_abs_embedding_asymmetry: float
    sources: np.ndarray
    targets: np.ndarray
    data_asymmetry: np.ndarray
    embedding_asymmetry: np.ndarray
    strong_mask: np.ndarray


def asymmetry_score(forward, backward, *, eps=1e-12):
    """Return ``(backward - forward) / (forward + backward)``.

    Positive values mean that the forward direction is cheaper than the reverse
    direction. The score is scale-free and lies in ``[-1, 1]`` for non-negative
    finite distances.
    """
    forward = np.asarray(forward, dtype=float)
    backward = np.asarray(backward, dtype=float)
    denom = forward + backward
    return np.divide(
        backward - forward,
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=np.isfinite(denom) & (denom > eps),
    )


def asymmetry_preservation_from_neighbors(
        data_dissimilarities,
        embedding,
        metric,
        neighbor_indices,
        *,
        tau=0.02,
        unique_pairs=True,
        eps=1e-12,
):
    """Compare source and embedding asymmetries on a fixed neighbor set.

    ``neighbor_indices`` should usually come from a symmetric neighborhood in
    the original expression/PCA space, not from the embedding being evaluated.
    This prevents the embedding from selecting only the pairs whose asymmetry it
    already represents well.
    """
    sources, targets = neighbor_pairs(neighbor_indices, unique_pairs=unique_pairs)
    return asymmetry_preservation_from_pairs(
        data_dissimilarities,
        embedding,
        metric,
        sources,
        targets,
        tau=tau,
        eps=eps,
    )


def asymmetry_preservation_from_pairs(
        data_dissimilarities,
        embedding,
        metric,
        sources,
        targets,
        *,
        tau=0.02,
        eps=1e-12,
):
    """Compare source and embedding asymmetries on explicit oriented pairs."""
    D = np.asarray(data_dissimilarities, dtype=float)
    X = np.asarray(embedding, dtype=float)
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("data_dissimilarities must be a square matrix.")
    if X.ndim != 2 or X.shape[0] != D.shape[0]:
        raise ValueError("embedding must have one row per data point.")
    if sources.shape != targets.shape:
        raise ValueError("sources and targets must have the same shape.")
    if not hasattr(metric, "length"):
        raise TypeError("metric must expose a length(u) method.")

    data_forward = D[sources, targets]
    data_backward = D[targets, sources]
    data_asym = asymmetry_score(data_forward, data_backward, eps=eps)

    displacements = X[targets] - X[sources]
    emb_forward = metric.length(displacements)
    emb_backward = metric.length(-displacements)
    emb_asym = asymmetry_score(emb_forward, emb_backward, eps=eps)

    valid = np.isfinite(data_asym) & np.isfinite(emb_asym)
    data_asym = data_asym[valid]
    emb_asym = emb_asym[valid]
    sources = sources[valid]
    targets = targets[valid]

    tau = float(tau)
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    strong = np.abs(data_asym) > tau

    return _summarize_asymmetry_preservation(
        sources,
        targets,
        data_asym,
        emb_asym,
        strong,
        tau=tau,
    )


def neighbor_pairs(neighbor_indices, *, unique_pairs=True):
    """Return oriented pairs from a dense neighbor-index matrix."""
    neighbors = np.asarray(neighbor_indices, dtype=int)
    if neighbors.ndim != 2:
        raise ValueError("neighbor_indices must have shape (n_samples, n_neighbors).")

    rows = np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])
    cols = neighbors.reshape(-1)
    keep = (cols >= 0) & (cols != rows)
    rows = rows[keep]
    cols = cols[keep]

    if unique_pairs:
        lo = np.minimum(rows, cols)
        hi = np.maximum(rows, cols)
        pairs = np.stack([lo, hi], axis=1)
        pairs = np.unique(pairs, axis=0)
        rows = pairs[:, 0]
        cols = pairs[:, 1]
    return rows.astype(int, copy=False), cols.astype(int, copy=False)


def summarize_asymmetry_preservation(
        sources,
        targets,
        data_asymmetry,
        embedding_asymmetry,
        *,
        tau=0.02,
):
    """Summarize two precomputed asymmetry-score arrays.

    This is useful when the embedding-side asymmetry is not produced by a
    uniform Finsler metric, for example for an RNA-velocity field projected on a
    UMAP embedding.
    """
    sources = np.asarray(sources, dtype=int)
    targets = np.asarray(targets, dtype=int)
    data_asymmetry = np.asarray(data_asymmetry, dtype=float)
    embedding_asymmetry = np.asarray(embedding_asymmetry, dtype=float)
    if sources.shape != targets.shape:
        raise ValueError("sources and targets must have the same shape.")
    if data_asymmetry.shape != embedding_asymmetry.shape:
        raise ValueError("asymmetry arrays must have the same shape.")
    if sources.shape != data_asymmetry.shape:
        raise ValueError("pairs and asymmetry arrays must have the same shape.")

    valid = np.isfinite(data_asymmetry) & np.isfinite(embedding_asymmetry)
    data_asymmetry = data_asymmetry[valid]
    embedding_asymmetry = embedding_asymmetry[valid]
    sources = sources[valid]
    targets = targets[valid]
    tau = float(tau)
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    strong = np.abs(data_asymmetry) > tau
    return _summarize_asymmetry_preservation(
        sources,
        targets,
        data_asymmetry,
        embedding_asymmetry,
        strong,
        tau=tau,
    )


def _summarize_asymmetry_preservation(
        sources,
        targets,
        data_asym,
        emb_asym,
        strong,
        *,
        tau,
):
    n_pairs = int(len(data_asym))
    n_strong = int(np.count_nonzero(strong))
    if n_strong == 0:
        return AsymmetryPreservationResult(
            n_pairs=n_pairs,
            n_strong_pairs=0,
            tau=tau,
            sign_accuracy=np.nan,
            weighted_sign_accuracy=np.nan,
            spearman=np.nan,
            pearson=np.nan,
            gamma=np.nan,
            normalized_mse=np.nan,
            mean_abs_data_asymmetry=np.nan,
            mean_abs_embedding_asymmetry=np.nan,
            sources=sources,
            targets=targets,
            data_asymmetry=data_asym,
            embedding_asymmetry=emb_asym,
            strong_mask=strong,
        )

    x = data_asym[strong]
    y = emb_asym[strong]
    same_sign = np.sign(x) == np.sign(y)
    weights = np.abs(x)
    sign_accuracy = float(np.mean(same_sign))
    weighted_sign_accuracy = float(np.sum(weights * same_sign) / np.sum(weights))

    spearman = _safe_spearman(x, y)
    pearson = _safe_pearson(x, y)
    denom = float(x @ x)
    gamma = float((x @ y) / denom) if denom > 0 else np.nan
    mse_denom = denom / len(x)
    normalized_mse = (
        float(np.mean((y - x) ** 2) / mse_denom)
        if mse_denom > 0
        else np.nan
    )

    return AsymmetryPreservationResult(
        n_pairs=n_pairs,
        n_strong_pairs=n_strong,
        tau=tau,
        sign_accuracy=sign_accuracy,
        weighted_sign_accuracy=weighted_sign_accuracy,
        spearman=spearman,
        pearson=pearson,
        gamma=gamma,
        normalized_mse=normalized_mse,
        mean_abs_data_asymmetry=float(np.mean(np.abs(x))),
        mean_abs_embedding_asymmetry=float(np.mean(np.abs(y))),
        sources=sources,
        targets=targets,
        data_asymmetry=data_asym,
        embedding_asymmetry=emb_asym,
        strong_mask=strong,
    )


def _safe_spearman(x, y):
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def _safe_pearson(x, y):
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])
