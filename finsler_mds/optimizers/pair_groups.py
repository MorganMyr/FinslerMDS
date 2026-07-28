"""Selection and sampling of the pairs optimized by Path-Frozen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state


@dataclass(frozen=True)
class ActivePairs:
    """Directed pairs grouped by source."""

    sources: np.ndarray
    targets: list[np.ndarray]
    weights: list[np.ndarray]
    dissimilarities: list[np.ndarray]
    n_pairs: int


@dataclass(frozen=True)
class PairBatch:
    """Pairs and weight factors used during one outer iteration."""

    geodesic_pairs: ActivePairs
    direct_pairs: ActivePairs
    global_factor: float
    local_factor: float


class PairSampler:
    """Build local/global pairs and refresh stochastic choices when needed."""

    def __init__(
        self,
        D,
        W,
        *,
        n_local_pairs,
        n_landmark,
        random_landmark_fraction,
        targets_per_landmark,
        local_weight,
        random_state,
    ):
        self.D = D
        self.W = W
        self.n_landmark = max(0, min(int(n_landmark), len(D)))
        self.random_landmark_fraction = _validate_fraction(random_landmark_fraction)
        self.targets_per_landmark = targets_per_landmark
        self.local_weight = float(local_weight)
        if self.local_weight < 0:
            raise ValueError("local_weight must be non-negative.")

        # Separate generators reproduce the former default behavior while
        # exposing only one public random_state.
        self.landmark_rng = check_random_state(random_state)
        self.target_rng = check_random_state(random_state)
        self.allowed = _allowed_pair_mask(D, W)
        self.local_mask = _local_pair_mask(D, self.allowed, n_local_pairs)
        self.local_pairs = _pairs_from_mask(D, W, self.local_mask)
        self.use_sparse_pairs = n_local_pairs is not None or self.n_landmark > 0

        n_random = int(round(self.random_landmark_fraction * self.n_landmark))
        self.n_random_landmarks = min(max(n_random, 0), self.n_landmark)
        n_farthest = self.n_landmark - self.n_random_landmarks
        self.fixed_farthest = _farthest_point_landmarks(
            D,
            n_farthest,
            self.landmark_rng,
        )
        self.is_stochastic = (
            (self.n_random_landmarks > 0 and self.n_landmark > 0)
            or targets_per_landmark is not None
        )
        self._static_batch = None

    def sample(self):
        if self._static_batch is not None:
            return self._static_batch

        global_pairs = self._global_pairs()
        global_pairs, local_pairs, global_factor, local_factor = _balance_pairs(
            global_pairs,
            self.local_pairs,
            local_weight=self.local_weight,
        )
        geodesic_pairs = _sample_targets(
            global_pairs,
            max_targets_per_source=self.targets_per_landmark,
            rng=self.target_rng,
        )
        batch = PairBatch(
            geodesic_pairs=geodesic_pairs,
            direct_pairs=local_pairs,
            global_factor=global_factor,
            local_factor=local_factor,
        )
        if not self.is_stochastic:
            self._static_batch = batch
        return batch

    def _global_pairs(self):
        if not self.use_sparse_pairs:
            return _pairs_from_mask(self.D, self.W, self.allowed)
        landmarks = _select_landmarks(
            len(self.D),
            self.fixed_farthest,
            self.n_random_landmarks,
            self.landmark_rng,
        )
        return _pairs_from_landmarks(
            self.D,
            self.W,
            self.allowed,
            self.local_mask,
            landmarks,
        )


def empty_active_pairs():
    return ActivePairs(
        sources=np.array([], dtype=int),
        targets=[],
        weights=[],
        dissimilarities=[],
        n_pairs=0,
    )


def _allowed_pair_mask(D, W):
    allowed = (W != 0) & np.isfinite(D)
    np.fill_diagonal(allowed, False)
    return allowed


def _local_pair_mask(D, allowed, n_local_pairs):
    active = np.zeros_like(allowed, dtype=bool)
    if n_local_pairs is None or n_local_pairs <= 0:
        return active

    selection_distances = _symmetrized_distances(D)
    for source in range(len(D)):
        candidates = np.flatnonzero(allowed[source])
        if len(candidates) == 0:
            continue
        k = min(int(n_local_pairs), len(candidates))
        chosen = np.argpartition(selection_distances[source, candidates], k - 1)[:k]
        active[source, candidates[chosen]] = True
    return active


def _pairs_from_mask(D, W, mask):
    sources = np.flatnonzero(np.any(mask, axis=1))
    if len(sources) == 0:
        return empty_active_pairs()
    targets = [np.flatnonzero(mask[source]) for source in sources]
    weights = [
        W[source, target].astype(float, copy=False)
        for source, target in zip(sources, targets)
    ]
    dissimilarities = [
        D[source, target].astype(float, copy=False)
        for source, target in zip(sources, targets)
    ]
    return ActivePairs(
        sources=sources.astype(int, copy=False),
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        n_pairs=sum(map(len, targets)),
    )


def _pairs_from_landmarks(D, W, allowed, local_mask, landmarks):
    mask = np.zeros_like(allowed, dtype=bool)
    mask[landmarks] = allowed[landmarks] & ~local_mask[landmarks]
    return _pairs_from_mask(D, W, mask)


def _select_landmarks(n_samples, fixed_farthest, n_random, rng):
    if n_random <= 0:
        return fixed_farthest
    available = np.setdiff1d(
        np.arange(n_samples, dtype=int),
        fixed_farthest,
        assume_unique=False,
    )
    random = rng.choice(available, size=min(n_random, len(available)), replace=False)
    return np.sort(np.concatenate([fixed_farthest, random]).astype(int, copy=False))


def _farthest_point_landmarks(D, n_landmarks, rng):
    if n_landmarks <= 0:
        return np.array([], dtype=int)
    distances = _symmetrized_distances(D)
    selected = np.empty(n_landmarks, dtype=int)
    selected[0] = int(rng.randint(len(D)))
    min_dist = distances[selected[0]].copy()
    min_dist[selected[0]] = -np.inf
    for pos in range(1, n_landmarks):
        selected[pos] = int(np.argmax(min_dist))
        min_dist = np.minimum(min_dist, distances[selected[pos]])
        min_dist[selected[:pos + 1]] = -np.inf
    return np.sort(selected)


def _symmetrized_distances(D):
    reverse = D.T
    finite_both = np.isfinite(D) & np.isfinite(reverse)
    finite_forward = np.isfinite(D) & ~np.isfinite(reverse)
    finite_reverse = ~np.isfinite(D) & np.isfinite(reverse)

    selection = np.full_like(D, np.inf, dtype=float)
    selection[finite_both] = 0.5 * (D[finite_both] + reverse[finite_both])
    selection[finite_forward] = D[finite_forward]
    selection[finite_reverse] = reverse[finite_reverse]
    np.fill_diagonal(selection, np.inf)
    return selection


def _balance_pairs(global_pairs, local_pairs, *, local_weight):
    if global_pairs.n_pairs == 0 and local_pairs.n_pairs == 0:
        raise ValueError("No active pair remains for Path-Frozen optimization.")
    if global_pairs.n_pairs == 0:
        return global_pairs, _scale_pairs(local_pairs, local_weight), 1.0, local_weight
    if local_pairs.n_pairs == 0:
        return global_pairs, local_pairs, 1.0, 1.0

    global_mass = _weight_mass(global_pairs)
    local_mass = _weight_mass(local_pairs)
    if global_mass <= 0 or local_mass <= 0:
        return global_pairs, _scale_pairs(local_pairs, local_weight), 1.0, local_weight
    target_mass = max(global_mass, local_mass)
    global_factor = target_mass / global_mass
    local_factor = local_weight * target_mass / local_mass
    return (
        _scale_pairs(global_pairs, global_factor),
        _scale_pairs(local_pairs, local_factor),
        global_factor,
        local_factor,
    )


def _weight_mass(pairs):
    return float(sum(np.sum(weights) for weights in pairs.weights))


def _scale_pairs(pairs, factor):
    factor = float(factor)
    if factor == 1.0 or pairs.n_pairs == 0:
        return pairs
    return ActivePairs(
        sources=pairs.sources,
        targets=pairs.targets,
        weights=[weights * factor for weights in pairs.weights],
        dissimilarities=pairs.dissimilarities,
        n_pairs=pairs.n_pairs,
    )


def _sample_targets(pairs, *, max_targets_per_source, rng):
    if max_targets_per_source is None:
        return pairs
    limit = int(max_targets_per_source)
    if limit <= 0:
        raise ValueError("targets_per_landmark must be positive or None.")

    targets = []
    weights = []
    dissimilarities = []
    for source_targets, source_weights, source_dissimilarities in zip(
        pairs.targets,
        pairs.weights,
        pairs.dissimilarities,
    ):
        n_available = len(source_targets)
        if n_available > limit:
            chosen = np.sort(rng.choice(n_available, size=limit, replace=False))
            targets.append(source_targets[chosen])
            weights.append(source_weights[chosen] * (n_available / limit))
            dissimilarities.append(source_dissimilarities[chosen])
        else:
            targets.append(source_targets)
            weights.append(source_weights)
            dissimilarities.append(source_dissimilarities)
    return ActivePairs(
        sources=pairs.sources,
        targets=targets,
        weights=weights,
        dissimilarities=dissimilarities,
        n_pairs=sum(map(len, targets)),
    )


def _validate_fraction(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("random_landmark_fraction must be between 0 and 1.")
    return value


__all__ = ["ActivePairs", "PairBatch", "PairSampler", "empty_active_pairs"]
