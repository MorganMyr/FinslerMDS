"""Pancreas-specific trajectory-gap definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_PANCREAS_GAP = {
    "enabled": False,
    "name": "preendocrine",
    "selection": "labels",
    "removed_labels": ("Pre-endocrine",),
    "before_labels": ("Ngn3 high EP",),
    "after_labels": ("Alpha", "Beta", "Delta", "Epsilon"),
    "n_before": 300,
    "n_after": 300,
}


@dataclass(frozen=True)
class PancreasGapSelection:
    config: dict
    labels: np.ndarray
    cell_ids: np.ndarray
    keep_mask: np.ndarray
    removed_mask: np.ndarray
    original_indices: np.ndarray
    before_indices: np.ndarray
    after_indices: np.ndarray
    before_original_indices: np.ndarray
    after_original_indices: np.ndarray
    ordering: np.ndarray | None = None

    @property
    def enabled(self):
        return bool(self.config["enabled"])


def normalize_pancreas_gap_config(config):
    if config is None:
        config = {}
    merged = dict(DEFAULT_PANCREAS_GAP)
    merged.update(config)
    merged["enabled"] = bool(merged.get("enabled", False))
    for key in ("removed_labels", "before_labels", "after_labels"):
        merged[key] = tuple(str(label) for label in merged.get(key, ()))
    merged["name"] = str(merged.get("name") or "gap")
    merged["selection"] = str(merged.get("selection") or "labels")
    merged["n_before"] = int(merged.get("n_before", 300))
    merged["n_after"] = int(merged.get("n_after", 300))
    return merged


def pancreas_gap_prefix(config, *, base="pancreas"):
    config = normalize_pancreas_gap_config(config)
    if not config["enabled"]:
        return base
    suffix = ""
    if config["selection"] in {"latent_time", "veloviz_latent_time"}:
        suffix = f"_lt{config['n_before']}"
    return f"gap{suffix}"


def select_pancreas_gap(labels, config, *, cell_ids=None, ordering=None):
    """Build masks/indices for a pancreas gap experiment.

    The returned labels/cell_ids/original_indices correspond to the kept
    dataset. ``before_indices`` and ``after_indices`` are indices in that kept
    dataset, which is the coordinate system used by saved embeddings.
    """
    config = normalize_pancreas_gap_config(config)
    labels = np.asarray(labels, dtype=str)
    if cell_ids is None:
        cell_ids = np.arange(len(labels)).astype(str)
    else:
        cell_ids = np.asarray(cell_ids, dtype=str)
    if cell_ids.shape[0] != labels.shape[0]:
        raise ValueError("cell_ids and labels must have the same length.")

    removed_mask = np.isin(labels, config["removed_labels"]) if config["enabled"] else np.zeros(len(labels), dtype=bool)
    keep_mask = ~removed_mask
    kept_labels = labels[keep_mask]
    original_indices = np.flatnonzero(keep_mask)
    if config["enabled"] and config["selection"] in {"latent_time", "veloviz_latent_time"}:
        before_original, after_original = _latent_time_before_after(labels, removed_mask, ordering, config)
        kept_position = np.full(len(labels), -1, dtype=int)
        kept_position[original_indices] = np.arange(len(original_indices))
        before_indices = kept_position[before_original]
        after_indices = kept_position[after_original]
    else:
        before_indices, after_indices = pancreas_gap_group_indices(kept_labels, config)
        before_original = original_indices[before_indices]
        after_original = original_indices[after_indices]
    return PancreasGapSelection(
        config=config,
        labels=kept_labels,
        cell_ids=cell_ids[keep_mask],
        keep_mask=keep_mask,
        removed_mask=removed_mask,
        original_indices=original_indices,
        before_indices=before_indices,
        after_indices=after_indices,
        before_original_indices=before_original,
        after_original_indices=after_original,
        ordering=None if ordering is None else np.asarray(ordering, dtype=float),
    )


def pancreas_gap_group_indices(labels, config=None):
    config = normalize_pancreas_gap_config(config)
    labels = np.asarray(labels, dtype=str)
    before = np.flatnonzero(np.isin(labels, config["before_labels"]))
    after = np.flatnonzero(np.isin(labels, config["after_labels"]))
    return before, after


def gap_arrays_to_cache(selection):
    arrays = {
        "labels": selection.labels,
        "cell_ids": selection.cell_ids,
        "original_indices": selection.original_indices,
        "gap_removed_original_indices": np.flatnonzero(selection.removed_mask),
        "gap_before_indices": selection.before_indices,
        "gap_after_indices": selection.after_indices,
        "gap_before_original_indices": selection.before_original_indices,
        "gap_after_original_indices": selection.after_original_indices,
    }
    if selection.ordering is not None:
        arrays["gap_ordering_full"] = selection.ordering
        arrays["gap_ordering"] = selection.ordering[selection.keep_mask]
    return arrays


def _latent_time_before_after(labels, removed_mask, ordering, config):
    if ordering is None:
        raise ValueError("ordering is required for pancreas gap selection='veloviz_latent_time'.")
    ordering = np.asarray(ordering, dtype=float)
    if ordering.shape[0] != labels.shape[0]:
        raise ValueError("ordering and labels must have the same length.")
    finite = np.isfinite(ordering)
    removed_times = ordering[removed_mask & finite]
    if removed_times.size == 0:
        raise ValueError("No finite ordering values for removed gap cells.")
    lower = float(np.min(removed_times))
    upper = float(np.max(removed_times))

    before_candidates = np.flatnonzero((~removed_mask) & finite & (ordering < lower))
    after_candidates = np.flatnonzero((~removed_mask) & finite & (ordering > upper))
    n_before = config["n_before"]
    n_after = config["n_after"]
    if before_candidates.size < n_before or after_candidates.size < n_after:
        raise ValueError(
            "Not enough cells around the gap in ordering space: "
            f"{before_candidates.size} before candidates, {after_candidates.size} after candidates, "
            f"need {n_before}/{n_after}."
        )

    before = before_candidates[np.argsort(ordering[before_candidates])[-n_before:]]
    after = after_candidates[np.argsort(ordering[after_candidates])[:n_after]]
    return np.sort(before), np.sort(after)


def _safe_token(value):
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")
