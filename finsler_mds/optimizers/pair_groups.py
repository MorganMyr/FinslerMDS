"""Active-pair construction shared by geodesic optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state


@dataclass(frozen=True)
class ActivePairs:
    """Directed active pairs grouped by source."""

    sources: np.ndarray
    targets: list[np.ndarray]
    weights: list[np.ndarray]
    dissimilarities: list[np.ndarray]
    denom: float
    n_pairs: int


@dataclass(frozen=True)
class LocalGlobalPairs:
    """Local and global active-pair groups.

    Local pairs are the nearest targets in the original dissimilarity matrix.
    Global pairs are all targets from landmark source rows, or all allowed
    pairs when no sparse local/global selection is requested.
    """

    global_pairs: ActivePairs
    local_pairs: ActivePairs
    local_pair_mode: str
    local_global_reweighting: str
    local_weight: float
    global_factor: float
    local_factor: float

    @property
    def n_pairs(self):
        return self.global_pairs.n_pairs + self.local_pairs.n_pairs


def empty_active_pairs():
    return ActivePairs(
        sources=np.array([], dtype=int),
        targets=[],
        weights=[],
        dissimilarities=[],
        denom=0.0,
        n_pairs=0,
    )


def build_local_global_pairs(
        D,
        W,
        *,
        pair_mask=None,
        n_local_neighbors=None,
        local_pair_mode="direct",
        landmark_indices=None,
        n_global_landmarks=0,
        random_state=None,
        local_weight=1.0,
        local_global_reweighting="none",
):
    """Build disjoint local and global pair groups.

    ``local_global_reweighting`` controls the relative mass of local and global
    groups before ``local_weight`` is applied:

    - ``"none"``: keep the input weights unchanged.
    - ``"count"``: balance ``sum w_ij`` in the two groups.
    - ``"energy"``: balance ``sum w_ij D_ij^2`` in the two groups.
    """
    local_pair_mode = _validate_local_pair_mode(local_pair_mode)
    local_global_reweighting = _validate_reweighting(local_global_reweighting)
    local_weight = float(local_weight)
    if local_weight < 0:
        raise ValueError("local_weight must be non-negative.")

    allowed = (W != 0) & np.isfinite(D)
    np.fill_diagonal(allowed, False)

    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != D.shape:
            raise ValueError("pair_mask must have the same shape as dissimilarities.")
        allowed &= pair_mask

    use_sparse_builder = (
        n_local_neighbors is not None
        or landmark_indices is not None
        or n_global_landmarks > 0
    )

    local_mask = np.zeros_like(allowed, dtype=bool)
    global_mask = np.zeros_like(allowed, dtype=bool)

    if use_sparse_builder:
        if n_local_neighbors is not None and n_local_neighbors > 0:
            _add_local_pairs(local_mask, allowed, D, int(n_local_neighbors))

        landmarks = select_landmarks(
            D.shape[0],
            landmark_indices=landmark_indices,
            n_global_landmarks=n_global_landmarks,
            random_state=random_state,
        )
        if len(landmarks) > 0:
            global_mask[landmarks, :] = allowed[landmarks, :]
    else:
        global_mask = allowed.copy()

    # A local pair should contribute to the local group only. This makes the
    # local/global weights interpretable and prevents accidental double counts
    # when a landmark source also has local-neighbor constraints.
    global_mask &= ~local_mask

    global_pairs = active_pairs_from_mask(D, W, global_mask, allow_empty=True)
    local_pairs = active_pairs_from_mask(D, W, local_mask, allow_empty=True)
    if global_pairs.n_pairs == 0 and local_pairs.n_pairs == 0:
        raise ValueError("No active pair remains for geodesic optimization.")

    global_factor, local_factor = group_reweighting_factors(
        global_pairs,
        local_pairs,
        mode=local_global_reweighting,
        local_weight=local_weight,
    )
    global_pairs = scale_active_pairs(global_pairs, global_factor)
    local_pairs = scale_active_pairs(local_pairs, local_factor)

    return LocalGlobalPairs(
        global_pairs=global_pairs,
        local_pairs=local_pairs,
        local_pair_mode=local_pair_mode,
        local_global_reweighting=local_global_reweighting,
        local_weight=local_weight,
        global_factor=global_factor,
        local_factor=local_factor,
    )


def active_pairs_from_mask(D, W, active_mask, *, allow_empty=False):
    sources = np.flatnonzero(np.any(active_mask, axis=1))
    if len(sources) == 0:
        if allow_empty:
            return empty_active_pairs()
        raise ValueError("No active pair remains for geodesic optimization.")

    targets = []
    weights = []
    dissimilarities = []
    denom = 0.0
    n_pairs = 0
    for source in sources:
        source_targets = np.flatnonzero(active_mask[source])
        source_weights = W[source, source_targets].astype(float, copy=False)
        source_dissimilarities = D[source, source_targets].astype(float, copy=False)
        targets.append(source_targets)
        weights.append(source_weights)
        dissimilarities.append(source_dissimilarities)
        denom += float(np.sum(source_weights * source_dissimilarities**2))
        n_pairs += len(source_targets)

    return ActivePairs(
        sources=sources.astype(int, copy=False),
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        denom=denom,
        n_pairs=n_pairs,
    )


def select_landmarks(n_samples, *, landmark_indices, n_global_landmarks, random_state):
    if landmark_indices is not None:
        landmarks = np.asarray(landmark_indices, dtype=int)
        if landmarks.ndim != 1:
            raise ValueError("landmark_indices must be a 1D array-like.")
        if np.any((landmarks < 0) | (landmarks >= n_samples)):
            raise ValueError("landmark_indices contains an out-of-range index.")
        return np.unique(landmarks)

    if n_global_landmarks <= 0:
        return np.array([], dtype=int)

    rng = check_random_state(random_state)
    n_global_landmarks = min(int(n_global_landmarks), n_samples)
    return np.sort(rng.choice(n_samples, size=n_global_landmarks, replace=False))


def sample_active_pairs(
        active_pairs,
        *,
        max_targets_per_source,
        target_sampling,
        random_state,
):
    """Sample targets within each source row with unbiased weight correction."""
    if max_targets_per_source is None:
        return active_pairs

    max_targets_per_source = int(max_targets_per_source)
    if max_targets_per_source <= 0:
        raise ValueError("max_targets_per_source must be positive or None.")
    if target_sampling not in {"random", "farthest", "mixed"}:
        raise ValueError("target_sampling must be 'random', 'farthest', or 'mixed'.")

    rng = check_random_state(random_state)
    targets = []
    weights = []
    dissimilarities = []
    denom = 0.0
    n_pairs = 0

    for source_pos in range(len(active_pairs.sources)):
        source_targets = active_pairs.targets[source_pos]
        source_weights = active_pairs.weights[source_pos]
        source_dissimilarities = active_pairs.dissimilarities[source_pos]
        n_available = len(source_targets)

        if n_available > max_targets_per_source:
            chosen = _sample_target_indices(
                source_dissimilarities,
                max_targets_per_source,
                target_sampling,
                rng,
            )
            sampled_targets = source_targets[chosen]
            sampled_weights = source_weights[chosen].copy()
            sampled_dissimilarities = source_dissimilarities[chosen]
            sampled_weights *= n_available / max_targets_per_source
        else:
            sampled_targets = source_targets
            sampled_weights = source_weights
            sampled_dissimilarities = source_dissimilarities

        targets.append(sampled_targets)
        weights.append(sampled_weights)
        dissimilarities.append(sampled_dissimilarities)
        denom += float(np.sum(sampled_weights * sampled_dissimilarities**2))
        n_pairs += len(sampled_targets)

    return ActivePairs(
        sources=active_pairs.sources,
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        denom=denom,
        n_pairs=n_pairs,
    )


def merge_active_pairs(*groups):
    groups = [group for group in groups if group.n_pairs > 0]
    if not groups:
        return empty_active_pairs()

    all_sources = np.unique(np.concatenate([group.sources for group in groups]))
    targets = []
    weights = []
    dissimilarities = []
    denom = 0.0
    n_pairs = 0

    group_maps = []
    for group in groups:
        group_maps.append({int(source): pos for pos, source in enumerate(group.sources)})

    for source in all_sources:
        source_targets = []
        source_weights = []
        source_dissimilarities = []
        for group, source_map in zip(groups, group_maps):
            pos = source_map.get(int(source))
            if pos is None:
                continue
            source_targets.append(group.targets[pos])
            source_weights.append(group.weights[pos])
            source_dissimilarities.append(group.dissimilarities[pos])
        merged_targets = np.concatenate(source_targets).astype(int, copy=False)
        merged_weights = np.concatenate(source_weights).astype(float, copy=False)
        merged_dissimilarities = np.concatenate(source_dissimilarities).astype(float, copy=False)
        targets.append(merged_targets)
        weights.append(merged_weights)
        dissimilarities.append(merged_dissimilarities)
        denom += float(np.sum(merged_weights * merged_dissimilarities**2))
        n_pairs += len(merged_targets)

    return ActivePairs(
        sources=all_sources.astype(int, copy=False),
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        denom=denom,
        n_pairs=n_pairs,
    )


def scale_active_pairs(active_pairs, factor):
    factor = float(factor)
    if factor == 1.0 or active_pairs.n_pairs == 0:
        return active_pairs
    return ActivePairs(
        sources=active_pairs.sources,
        targets=active_pairs.targets,
        weights=[weights * factor for weights in active_pairs.weights],
        dissimilarities=active_pairs.dissimilarities,
        denom=active_pairs.denom * factor,
        n_pairs=active_pairs.n_pairs,
    )


def group_reweighting_factors(global_pairs, local_pairs, *, mode, local_weight):
    if global_pairs.n_pairs == 0:
        return 1.0, float(local_weight)
    if local_pairs.n_pairs == 0:
        return 1.0, 1.0
    if mode == "none":
        return 1.0, float(local_weight)

    global_mass = _group_mass(global_pairs, mode)
    local_mass = _group_mass(local_pairs, mode)
    if global_mass <= 0 or local_mass <= 0:
        return 1.0, float(local_weight)

    target_mass = max(global_mass, local_mass)
    return target_mass / global_mass, float(local_weight) * target_mass / local_mass


def _add_local_pairs(active, allowed, D, n_local_neighbors):
    for source in range(D.shape[0]):
        candidates = np.flatnonzero(allowed[source])
        if len(candidates) == 0:
            continue
        k = min(n_local_neighbors, len(candidates))
        distances = D[source, candidates]
        chosen = candidates[np.argpartition(distances, k - 1)[:k]]
        active[source, chosen] = True


def _sample_target_indices(dissimilarities, n_keep, target_sampling, rng):
    n_available = len(dissimilarities)
    if target_sampling == "random":
        chosen = rng.choice(n_available, size=n_keep, replace=False)
    elif target_sampling == "farthest":
        chosen = np.argpartition(dissimilarities, n_available - n_keep)[-n_keep:]
    else:
        n_far = n_keep // 2
        far = (
            np.argpartition(dissimilarities, n_available - n_far)[-n_far:]
            if n_far > 0
            else np.array([], dtype=int)
        )
        remaining_mask = np.ones(n_available, dtype=bool)
        remaining_mask[far] = False
        remaining = np.flatnonzero(remaining_mask)
        n_random = n_keep - len(far)
        random = rng.choice(remaining, size=n_random, replace=False)
        chosen = np.concatenate([far, random])
    return np.sort(chosen)


def _group_mass(active_pairs, mode):
    if mode == "count":
        return float(sum(np.sum(weights) for weights in active_pairs.weights))
    if mode == "energy":
        return float(active_pairs.denom)
    raise ValueError("mode must be 'none', 'count', or 'energy'.")


def _validate_local_pair_mode(local_pair_mode):
    if local_pair_mode not in {"direct", "geodesic"}:
        raise ValueError("local_pair_mode must be 'direct' or 'geodesic'.")
    return local_pair_mode


def _validate_reweighting(mode):
    if mode not in {"none", "count", "energy"}:
        raise ValueError("local_global_reweighting must be 'none', 'count', or 'energy'.")
    return mode


__all__ = [
    "ActivePairs",
    "LocalGlobalPairs",
    "active_pairs_from_mask",
    "build_local_global_pairs",
    "empty_active_pairs",
    "merge_active_pairs",
    "sample_active_pairs",
    "scale_active_pairs",
    "select_landmarks",
]
