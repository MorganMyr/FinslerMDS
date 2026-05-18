"""Soft Bellman-Ford optimizer for Finsler-MDS.

This is an all-sources, differentiable Bellman-Ford algorithm. Hard edge
minimum updates are replaced by softmin updates, and gradients are propagated
explicitly through the relaxation trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import scipy.optimize
from sklearn.utils import check_random_state

from finsler_mds.evaluation import geodesic_embedding_stress
from finsler_mds.optimizers.common import (
    initial_embedding,
    prepare_weights_and_mask,
    validate_metric,
)
from finsler_mds.optimizers.path_frozen import (
    _DirectPairsObjective,
    _add_raw_objective,
    _cupy_metric_length_and_grad,
    _GpuDirectPairsObjective,
    _gpu_metric_supported,
    _load_cupy,
    _geodesic_source_count,
    _resolve_log_frequency,
    _sampled_pair_count,
    _should_log_iteration,
)
from finsler_mds.optimizers.pair_groups import (
    build_local_global_pairs,
    empty_active_pairs,
    merge_active_pairs,
    sample_active_pairs,
)
from finsler_mds.utils.graph import softmin_with_probs, symmetric_knn_graph


@dataclass(frozen=True)
class SoftBellmanFordResult:
    embedding: np.ndarray
    stress: float
    n_iter: int
    n_graph_updates: int
    optimizer_results: list


@dataclass(frozen=True)
class _GraphSupport:
    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]


class _UnreachableTracker:
    def __init__(self):
        self.last_skipped = 0
        self.last_skipped_denom = 0.0
        self.max_skipped = 0
        self.max_skipped_denom = 0.0
        self.calls_with_skipped = 0

    def record(self, skipped, skipped_denom):
        skipped = int(skipped)
        skipped_denom = float(skipped_denom)
        self.last_skipped = skipped
        self.last_skipped_denom = skipped_denom
        if skipped > 0:
            self.calls_with_skipped += 1
        if skipped > self.max_skipped:
            self.max_skipped = skipped
            self.max_skipped_denom = skipped_denom


def _resolve_gpu_backend(device, metric, verbose):
    if device not in {"cpu", "auto", "gpu", "cuda"}:
        raise ValueError("device must be one of 'cpu', 'auto', 'gpu', or 'cuda'.")
    if device == "cpu":
        return None
    if not _gpu_metric_supported(metric):
        message = (
            "soft_bellman_ford GPU backend currently supports RandersMetric, "
            "MatsumotoMetric without forbidden_grad_norm, and "
            "ConvexifiedMatsumotoMetric, and ConvexifiedToblerMetric only."
        )
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise ValueError(message)

    cp, error = _load_cupy()
    if cp is None:
        message = f"soft_bellman_ford GPU backend unavailable: {error}"
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise RuntimeError(message) from error

    if verbose:
        device_id = cp.cuda.Device().id
        device_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
        if hasattr(device_name, "decode"):
            device_name = device_name.decode()
        print(f"soft_bellman_ford GPU backend enabled on CUDA device {device_id}: {device_name}")
    return cp


def _support_from_embedding(X, *, graph_neighbors, neighbors_algorithm, n_jobs):
    support = symmetric_knn_graph(
        X,
        n_neighbors=graph_neighbors,
        neighbors_algorithm=neighbors_algorithm,
        n_jobs=n_jobs,
    ).tocoo()
    return _GraphSupport(
        rows=support.row.astype(int, copy=False),
        cols=support.col.astype(int, copy=False),
        shape=support.shape,
    )


def _metric_edges(X, support, metric):
    edge_vectors = X[support.cols] - X[support.rows]
    edge_lengths = metric.length(edge_vectors)
    finite = np.isfinite(edge_lengths)
    return (
        support.rows[finite],
        support.cols[finite],
        edge_lengths[finite],
        edge_vectors[finite],
    )


def soft_bellman_ford_distances(
        n_samples,
        tails,
        heads,
        costs,
        *,
        beta,
        n_relaxations=None,
        sources=None,
        prob_dtype=np.float32,
):
    """Return soft source-to-all distances and a trace for backpropagation.

    The update is synchronous: after ``t`` passes, paths with at most ``t``
    edges have contributed. With a hard min, ``n_samples - 1`` passes recover
    Bellman-Ford distances on graphs without negative cycles.
    """
    if n_relaxations is None:
        n_relaxations = n_samples - 1
    if n_relaxations < 0:
        raise ValueError("n_relaxations must be non-negative.")

    if sources is None:
        sources = np.arange(n_samples, dtype=int)
    else:
        sources = np.asarray(sources, dtype=int)
    distances = np.full((len(sources), n_samples), np.inf, dtype=float)
    distances[np.arange(len(sources)), sources] = 0.0
    incoming_edges = [np.flatnonzero(heads == v) for v in range(n_samples)]

    trace = []
    for _ in range(n_relaxations):
        next_distances = np.empty_like(distances)
        keep_probs = np.zeros((len(sources), n_samples), dtype=prob_dtype)
        edge_probs = np.zeros((len(costs), len(sources)), dtype=prob_dtype)

        for target in range(n_samples):
            incoming = incoming_edges[target]
            if incoming.size == 0:
                next_distances[:, target] = distances[:, target]
                keep_probs[:, target] = 1.0
                continue

            candidates = np.empty((len(sources), incoming.size + 1), dtype=float)
            candidates[:, 0] = distances[:, target]
            candidates[:, 1:] = distances[:, tails[incoming]] + costs[incoming][None, :]
            values, probs = softmin_with_probs(
                candidates,
                beta=beta,
                axis=1,
                prob_dtype=prob_dtype,
            )

            # Keep exact zero self-distances. Otherwise the entropy term can
            # make zero cycles slightly negative when beta is finite.
            self_rows = sources == target
            values[self_rows] = 0.0
            probs[self_rows, :] = 0.0
            probs[self_rows, 0] = 1.0

            next_distances[:, target] = values
            keep_probs[:, target] = probs[:, 0]
            edge_probs[incoming, :] = probs[:, 1:].T

        distances = next_distances
        trace.append((keep_probs, edge_probs))

    return distances, trace


def _soft_bellman_ford_pullback(final_adjoint, trace, tails, heads):
    adjoint = np.asarray(final_adjoint, dtype=float)
    cost_adjoint = np.zeros(len(tails), dtype=float)

    for keep_probs, edge_probs in reversed(trace):
        keep_probs = keep_probs.astype(float, copy=False)
        edge_probs = edge_probs.astype(float, copy=False)
        previous = keep_probs * adjoint

        edge_to_head_adjoint = adjoint[:, heads].T
        edge_contrib = edge_probs * edge_to_head_adjoint
        cost_adjoint += edge_contrib.sum(axis=1)
        for edge_id, tail in enumerate(tails):
            previous[:, tail] += edge_contrib[edge_id]

        adjoint = previous

    return cost_adjoint


def _validate_on_unreachable(on_unreachable):
    if on_unreachable not in {"raise", "warn_skip"}:
        raise ValueError("on_unreachable must be 'raise' or 'warn_skip'.")


def _active_stress_and_adjoint(
        distances,
        active_pairs,
        *,
        normalized_stress,
        on_unreachable,
        unreachable_tracker=None,
):
    final_adjoint = np.zeros_like(distances, dtype=float)
    raw_stress = 0.0
    skipped = 0
    skipped_denom = 0.0
    for source_pos in range(len(active_pairs.sources)):
        targets = active_pairs.targets[source_pos]
        weights = active_pairs.weights[source_pos]
        dissimilarities = active_pairs.dissimilarities[source_pos]
        finite = np.isfinite(distances[source_pos, targets])
        if not np.all(finite):
            if on_unreachable == "raise":
                raise ValueError(
                    "The soft Bellman-Ford graph has unreachable active pairs. "
                    "Increase graph_neighbors, increase n_relaxations, reduce the "
                    "active-pair mask, or set on_unreachable='warn_skip'."
                )
            skipped += int(np.sum(~finite))
            skipped_denom += float(np.sum(weights[~finite] * dissimilarities[~finite] ** 2))
            targets = targets[finite]
            weights = weights[finite]
            dissimilarities = dissimilarities[finite]
            if len(targets) == 0:
                continue
        residual = distances[source_pos, targets] - dissimilarities
        raw_stress += float(np.sum(weights * residual**2))
        final_adjoint[source_pos, targets] = 2.0 * weights * residual

    if unreachable_tracker is not None:
        unreachable_tracker.record(skipped, skipped_denom)

    if not normalized_stress:
        return raw_stress, final_adjoint

    denom = active_pairs.denom - skipped_denom
    if denom <= 0:
        final_adjoint[:] = 0.0
        return np.inf, final_adjoint
    if raw_stress <= 0:
        final_adjoint[:] = 0.0
        return 0.0, final_adjoint

    stress = np.sqrt(raw_stress / denom)
    final_adjoint *= 1.0 / (2.0 * np.sqrt(raw_stress * denom))
    return float(stress), final_adjoint


def _soft_bf_stress_and_grad(
        X_flat,
        *,
        shape,
        support,
        active_pairs,
        metric,
        beta,
        n_relaxations,
        prob_dtype,
        normalized_stress,
        on_unreachable,
        unreachable_tracker,
):
    X = X_flat.reshape(shape)
    tails, heads, costs, edge_vectors = _metric_edges(X, support, metric)
    distances, trace = soft_bellman_ford_distances(
        X.shape[0],
        tails,
        heads,
        costs,
        beta=beta,
        n_relaxations=n_relaxations,
        sources=active_pairs.sources,
        prob_dtype=prob_dtype,
    )

    stress, final_adjoint = _active_stress_and_adjoint(
        distances,
        active_pairs,
        normalized_stress=normalized_stress,
        on_unreachable=on_unreachable,
        unreachable_tracker=unreachable_tracker,
    )

    edge_adjoint = _soft_bellman_ford_pullback(final_adjoint, trace, tails, heads)
    edge_grads = metric.grad_u(edge_vectors)
    used_edges = edge_adjoint != 0
    if not np.all(np.isfinite(edge_grads[used_edges])):
        raise ValueError(
            "The metric produced non-finite gradients on active graph edges. "
            "If using Matsumoto with forbidden directions, set a finite "
            "forbidden_grad_norm or use a convexified/clipped metric."
        )

    grad = np.zeros_like(X)
    scaled_edge_grads = edge_adjoint[used_edges, None] * edge_grads[used_edges]
    np.add.at(grad, tails[used_edges], -scaled_edge_grads)
    np.add.at(grad, heads[used_edges], scaled_edge_grads)
    return float(stress), grad.ravel()


def _cupy_softmin_with_probs(cp, candidates, *, beta, axis=-1, prob_dtype=None):
    finite = cp.isfinite(candidates)
    any_finite = cp.any(finite, axis=axis, keepdims=True)
    shifted_min = cp.min(cp.where(finite, candidates, cp.inf), axis=axis, keepdims=True)

    valid = finite & cp.broadcast_to(any_finite, candidates.shape)
    scores = cp.where(valid, cp.exp(-beta * (candidates - shifted_min)), 0.0)
    denom = cp.sum(scores, axis=axis, keepdims=True)
    probs = cp.where(denom > 0, scores / denom, 0.0)

    soft = cp.squeeze(shifted_min, axis=axis) - cp.log(cp.squeeze(denom, axis=axis)) / beta
    soft = cp.where(cp.squeeze(any_finite, axis=axis), soft, cp.inf)
    if prob_dtype is not None:
        probs = probs.astype(prob_dtype, copy=False)
    return soft, probs


def _cupy_soft_bellman_ford_distances(
        cp,
        n_samples,
        tails,
        heads,
        costs,
        sources,
        *,
        beta,
        n_relaxations,
        prob_dtype,
):
    n_sources = len(sources)
    distances = cp.full((n_sources, n_samples), cp.inf, dtype=cp.float64)
    distances[cp.arange(n_sources), sources] = 0.0

    source_offsets = cp.arange(n_sources, dtype=cp.int64) * n_samples
    flat_head_indices = (source_offsets[:, None] + heads[None, :]).ravel()
    self_rows = cp.arange(n_sources, dtype=cp.int32)
    self_head_mask = heads[None, :] == sources[:, None]

    trace = []
    for _ in range(n_relaxations):
        edge_candidates = distances[:, tails] + costs[None, :]

        shifted_min = distances.copy()
        cp.minimum.at(shifted_min.ravel(), flat_head_indices, edge_candidates.ravel())

        finite_min = cp.isfinite(shifted_min)
        keep_valid = cp.isfinite(distances) & finite_min
        keep_scores = cp.where(
            keep_valid,
            cp.exp(-beta * (distances - shifted_min)),
            0.0,
        )

        edge_shifted_min = shifted_min[:, heads]
        edge_valid = cp.isfinite(edge_candidates) & cp.isfinite(edge_shifted_min)
        edge_scores = cp.where(
            edge_valid,
            cp.exp(-beta * (edge_candidates - edge_shifted_min)),
            0.0,
        )

        denom = keep_scores.copy()
        cp.add.at(denom.ravel(), flat_head_indices, edge_scores.ravel())

        positive_denom = denom > 0
        safe_denom = cp.where(positive_denom, denom, 1.0)
        next_distances = cp.where(
            positive_denom,
            shifted_min - cp.log(safe_denom) / beta,
            cp.inf,
        )
        keep_probs = keep_scores / safe_denom
        edge_safe_denom = safe_denom[:, heads]
        edge_probs = edge_scores / edge_safe_denom

        # Keep exact zero self-distances, as in the CPU implementation.
        next_distances[self_rows, sources] = 0.0
        keep_probs[self_rows, sources] = 1.0
        edge_probs = cp.where(self_head_mask, 0.0, edge_probs)

        distances = next_distances
        trace.append((
            keep_probs.astype(prob_dtype, copy=False),
            edge_probs.astype(prob_dtype, copy=False),
        ))

    return distances, trace


def _cupy_soft_bellman_ford_pullback(cp, final_adjoint, trace, tails, heads):
    adjoint = final_adjoint
    cost_adjoint = cp.zeros(len(tails), dtype=cp.float64)
    n_sources = final_adjoint.shape[0]
    n_samples = final_adjoint.shape[1]
    source_offsets = cp.arange(n_sources, dtype=cp.int64) * n_samples
    flat_tail_indices = source_offsets[:, None] + tails[None, :]

    for keep_probs, edge_probs in reversed(trace):
        previous = keep_probs.astype(cp.float64, copy=False) * adjoint
        edge_contrib = edge_probs.astype(cp.float64, copy=False) * adjoint[:, heads]
        cost_adjoint += cp.sum(edge_contrib, axis=0)
        cp.add.at(previous.ravel(), flat_tail_indices.ravel(), edge_contrib.ravel())
        adjoint = previous

    return cost_adjoint


class _GpuSoftBellmanFordObjective:
    def __init__(
            self,
            cp,
            *,
            shape,
            support,
            active_pairs,
            metric,
            beta,
            n_relaxations,
            prob_dtype,
            normalized_stress,
            source_batch_size,
            max_trace_entries,
            on_unreachable,
            unreachable_tracker,
    ):
        self.cp = cp
        self.shape = shape
        self.metric = metric
        self.beta = beta
        self.n_relaxations = n_relaxations
        self.prob_dtype = prob_dtype
        self.normalized_stress = normalized_stress
        self.source_batch_size = source_batch_size
        self.max_trace_entries = max_trace_entries
        self.active_pairs = active_pairs
        self.on_unreachable = on_unreachable
        self.unreachable_tracker = unreachable_tracker

        n_sources = len(active_pairs.sources)
        batch_size = n_sources if source_batch_size is None else int(source_batch_size)
        if batch_size <= 0:
            raise ValueError("source_batch_size must be positive or None.")
        trace_entries = n_relaxations * min(batch_size, n_sources) * (shape[0] + len(support.rows))
        if max_trace_entries is not None and trace_entries > max_trace_entries:
            raise MemoryError(
                "soft_bellman_ford GPU trace would be too large for one source batch. "
                "Reduce source_batch_size or n_relaxations."
            )

        self.rows = cp.asarray(support.rows, dtype=cp.int32)
        self.cols = cp.asarray(support.cols, dtype=cp.int32)

    def _iter_source_batches(self):
        n_sources = len(self.active_pairs.sources)
        batch_size = n_sources if self.source_batch_size is None else int(self.source_batch_size)
        if batch_size <= 0:
            raise ValueError("source_batch_size must be positive or None.")
        for start in range(0, n_sources, batch_size):
            stop = min(start + batch_size, n_sources)
            yield start, stop

    def __call__(self, X_flat):
        cp = self.cp
        X = cp.asarray(X_flat.reshape(self.shape))
        grad = cp.zeros_like(X)

        edge_vectors = X[self.cols] - X[self.rows]
        edge_lengths, edge_grads = _cupy_metric_length_and_grad(cp, edge_vectors, self.metric)
        finite_edges = cp.isfinite(edge_lengths)
        if not cp.all(cp.isfinite(edge_grads[finite_edges])).item():
            raise ValueError("The metric produced non-finite gradients on active graph edges.")

        tails = self.rows[finite_edges]
        heads = self.cols[finite_edges]
        costs = edge_lengths[finite_edges]
        edge_grads = edge_grads[finite_edges]
        total_raw_stress = 0.0
        total_edge_adjoint = cp.zeros(len(costs), dtype=cp.float64)
        total_skipped = 0
        total_skipped_denom = 0.0

        for start, stop in self._iter_source_batches():
            batch_sources_np = self.active_pairs.sources[start:stop]
            batch_sources = cp.asarray(batch_sources_np, dtype=cp.int32)
            n_batch = len(batch_sources_np)
            trace_entries = self.n_relaxations * n_batch * (self.shape[0] + len(costs))
            if self.max_trace_entries is not None and trace_entries > self.max_trace_entries:
                raise MemoryError(
                    "soft_bellman_ford GPU trace would be too large for one source batch. "
                    "Reduce source_batch_size or n_relaxations."
                )

            distances, trace = _cupy_soft_bellman_ford_distances(
                cp,
                self.shape[0],
                tails,
                heads,
                costs,
                batch_sources,
                beta=self.beta,
                n_relaxations=self.n_relaxations,
                prob_dtype=self.prob_dtype,
            )

            final_adjoint = cp.zeros_like(distances)
            raw_stress = cp.asarray(0.0, dtype=cp.float64)
            for local_pos, source_pos in enumerate(range(start, stop)):
                targets = cp.asarray(self.active_pairs.targets[source_pos], dtype=cp.int32)
                weights = cp.asarray(self.active_pairs.weights[source_pos], dtype=cp.float64)
                dissimilarities = cp.asarray(self.active_pairs.dissimilarities[source_pos], dtype=cp.float64)
                finite = cp.isfinite(distances[local_pos, targets])
                if not cp.all(finite).item():
                    if self.on_unreachable == "raise":
                        raise ValueError(
                            "The soft Bellman-Ford graph has unreachable active pairs. "
                            "Increase graph_neighbors, increase n_relaxations, reduce the "
                            "active-pair mask, or set on_unreachable='warn_skip'."
                        )
                    unreachable = ~finite
                    total_skipped += int(cp.count_nonzero(unreachable).item())
                    total_skipped_denom += float(cp.asnumpy(
                        cp.sum(weights[unreachable] * dissimilarities[unreachable] ** 2)
                    ))
                    targets = targets[finite]
                    weights = weights[finite]
                    dissimilarities = dissimilarities[finite]
                    if len(targets) == 0:
                        continue
                residual = distances[local_pos, targets] - dissimilarities
                raw_stress += cp.sum(weights * residual**2)
                final_adjoint[local_pos, targets] = 2.0 * weights * residual

            total_raw_stress += float(cp.asnumpy(raw_stress))
            total_edge_adjoint += _cupy_soft_bellman_ford_pullback(
                cp,
                final_adjoint,
                trace,
                tails,
                heads,
            )

        if self.unreachable_tracker is not None:
            self.unreachable_tracker.record(total_skipped, total_skipped_denom)

        stress = total_raw_stress
        if self.normalized_stress:
            denom = self.active_pairs.denom - total_skipped_denom
            if denom <= 0:
                return np.inf, np.zeros(self.shape, dtype=float).ravel()
            if total_raw_stress <= 0:
                return 0.0, np.zeros(self.shape, dtype=float).ravel()
            stress = np.sqrt(total_raw_stress / denom)
            total_edge_adjoint *= 1.0 / (2.0 * np.sqrt(total_raw_stress * denom))

        used_edges = total_edge_adjoint != 0
        scaled_edge_grads = total_edge_adjoint[used_edges, None] * edge_grads[used_edges]
        cp.add.at(grad, tails[used_edges], -scaled_edge_grads)
        cp.add.at(grad, heads[used_edges], scaled_edge_grads)
        return float(stress), cp.asnumpy(grad).ravel()


def soft_bellman_ford(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    graph_neighbors=10,
    beta=10.0,
    n_relaxations=None,
    max_iter=100,
    n_graph_updates=1,
    verbose=0,
    eps=1e-6,
    random_state=None,
    normalized_stress=False,
    weight=None,
    pair_mask=None,
    n_local_neighbors=None,
    local_pair_mode="direct",
    landmark_indices=None,
    n_global_landmarks=0,
    mask_random_state=None,
    max_global_targets_per_source=None,
    global_target_sampling="random",
    target_random_state=None,
    local_weight=1.0,
    local_global_reweighting="none",
    device="cpu",
    source_batch_size=8,
    gpu_max_trace_entries=250_000_000,
    on_unreachable="warn_skip",
    method="L-BFGS-B",
    optimizer_options=None,
    log_frequency=None,
    neighbors_algorithm="auto",
    n_jobs=None,
    prob_dtype=np.float32,
    return_n_iter=False,
    return_result=False,
):
    """Optimize Finsler-MDS stress with soft Bellman-Ford geodesics.

    Sparse-pair options mirror ``path_frozen``. With the default
    ``local_pair_mode="direct"``, ``n_local_neighbors`` are handled as direct
    local Finsler constraints instead of launching soft Bellman-Ford from every
    point. Pass ``local_pair_mode="geodesic"`` to run soft Bellman-Ford from
    local sources too. ``device="auto"`` uses a CuPy backend when available.
    ``log_frequency`` controls progress-line frequency across graph updates,
    with the same adaptive default as ``path_frozen``.
    """
    metric = validate_metric(metric)
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if n_graph_updates < 1:
        raise ValueError("n_graph_updates must be at least 1.")
    _validate_on_unreachable(on_unreachable)

    D, W = prepare_weights_and_mask(dissimilarities, weight)
    if n_relaxations is None:
        n_relaxations = D.shape[0] - 1
    if n_relaxations < 0:
        raise ValueError("n_relaxations must be non-negative.")
    if mask_random_state is None:
        mask_random_state = random_state
    if target_random_state is None:
        target_random_state = random_state
    target_random_state = check_random_state(target_random_state)
    gpu_backend = _resolve_gpu_backend(device, metric, verbose)

    pair_groups = build_local_global_pairs(
        D,
        W,
        pair_mask=pair_mask,
        n_local_neighbors=n_local_neighbors,
        local_pair_mode=local_pair_mode,
        landmark_indices=landmark_indices,
        n_global_landmarks=n_global_landmarks,
        random_state=mask_random_state,
        local_weight=local_weight,
        local_global_reweighting=local_global_reweighting,
    )
    global_pairs = pair_groups.global_pairs
    local_pairs = pair_groups.local_pairs
    local_geodesic_pairs = local_pairs if local_pair_mode == "geodesic" else empty_active_pairs()
    direct_pairs = local_pairs if local_pair_mode == "direct" else empty_active_pairs()
    X = initial_embedding(D, n_components, init, random_state)
    shape = X.shape
    direct_objective = None
    if direct_pairs.n_pairs > 0:
        if gpu_backend is not None:
            direct_objective = _GpuDirectPairsObjective(
                gpu_backend,
                shape=shape,
                direct_pairs=direct_pairs,
                metric=metric,
            )
        else:
            direct_objective = _DirectPairsObjective(
                shape=shape,
                direct_pairs=direct_pairs,
                metric=metric,
            )

    options = {"maxiter": max_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    optimizer_results = []
    total_iter = 0
    stress = np.inf
    log_frequency = _resolve_log_frequency(log_frequency, n_graph_updates)

    if verbose:
        sampled_global_n_pairs = _sampled_pair_count(global_pairs, max_global_targets_per_source)
        n_geodesic_sources = _geodesic_source_count(
            global_pairs,
            local_geodesic_pairs,
            max_global_targets_per_source,
        )
        print(
            "soft_bellman_ford: "
            f"{sampled_global_n_pairs + local_pairs.n_pairs} pairs "
            f"({sampled_global_n_pairs} global, {local_pairs.n_pairs} local-{local_pair_mode}) "
            f"over {D.shape[0] * (D.shape[0] - 1)}; "
            f"{n_geodesic_sources} active sources"
        )
        if local_global_reweighting != "none" or local_weight != 1.0:
            print(
                "soft_bellman_ford pair weights: "
                f"reweighting={local_global_reweighting}, "
                f"global_factor={pair_groups.global_factor:.6g}, "
                f"local_factor={pair_groups.local_factor:.6g}"
            )
        if log_frequency != 1:
            print(f"soft_bellman_ford logging every {log_frequency} graph updates")

    for graph_update in range(n_graph_updates):
        unreachable_tracker = _UnreachableTracker()
        iteration_global_pairs = sample_active_pairs(
            global_pairs,
            max_targets_per_source=max_global_targets_per_source,
            target_sampling=global_target_sampling,
            random_state=target_random_state,
        )
        iteration_pairs = merge_active_pairs(iteration_global_pairs, local_geodesic_pairs)

        raw_objective = None
        if iteration_pairs.n_pairs > 0:
            support = _support_from_embedding(
                X,
                graph_neighbors=graph_neighbors,
                neighbors_algorithm=neighbors_algorithm,
                n_jobs=n_jobs,
            )

            if gpu_backend is not None:
                try:
                    raw_objective = _GpuSoftBellmanFordObjective(
                        gpu_backend,
                        shape=shape,
                        support=support,
                        active_pairs=iteration_pairs,
                        metric=metric,
                        beta=beta,
                        n_relaxations=n_relaxations,
                        prob_dtype=prob_dtype,
                        normalized_stress=False,
                        source_batch_size=source_batch_size,
                        max_trace_entries=gpu_max_trace_entries,
                        on_unreachable=on_unreachable,
                        unreachable_tracker=unreachable_tracker,
                    )
                except MemoryError:
                    if device in {"gpu", "cuda"}:
                        raise
                    warnings.warn(
                        "Falling back to CPU soft_bellman_ford because the GPU "
                        "trace would be too large.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    raw_objective = None

            if raw_objective is None:
                def raw_objective(x_flat):
                    return _soft_bf_stress_and_grad(
                        x_flat,
                        shape=shape,
                        support=support,
                        active_pairs=iteration_pairs,
                        metric=metric,
                        beta=beta,
                        n_relaxations=n_relaxations,
                        prob_dtype=prob_dtype,
                        normalized_stress=False,
                        on_unreachable=on_unreachable,
                        unreachable_tracker=unreachable_tracker,
                    )

        def denom():
            return iteration_pairs.denom + direct_pairs.denom - unreachable_tracker.last_skipped_denom

        def objective(x_flat):
            return _add_raw_objective(
                x_flat,
                raw_objective,
                shape=shape,
                direct_pairs=direct_pairs,
                direct_objective=direct_objective,
                metric=metric,
                denom=denom,
                normalized_stress=normalized_stress,
            )

        result = scipy.optimize.minimize(
            objective,
            X.ravel(),
            jac=True,
            method=method,
            options=options,
        )
        optimizer_results.append(result)
        X = result.x.reshape(shape)
        stress = float(result.fun)
        total_iter += int(getattr(result, "nit", max_iter))

        if on_unreachable == "warn_skip" and unreachable_tracker.max_skipped > 0:
            warnings.warn(
                "soft_bellman_ford skipped unreachable geodesic pairs in "
                f"graph update {graph_update}: up to "
                f"{unreachable_tracker.max_skipped}/{iteration_pairs.n_pairs} "
                "pairs per objective evaluation were ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

        if verbose and _should_log_iteration(graph_update, n_graph_updates, log_frequency):
            nit = getattr(result, "nit", "?")
            nfev = getattr(result, "nfev", "?")
            print(
                f"soft_bellman_ford graph update {graph_update}: stress {stress} "
                f"(nit={nit}, nfev={nfev})"
            )

    if verbose:
        full_stress = geodesic_embedding_stress(
            X,
            D,
            metric=metric,
            n_neighbors=graph_neighbors,
            weight=W,
            normalized_stress=normalized_stress,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            on_unreachable="inf",
        )
        print(f"soft_bellman_ford final full geodesic stress: {full_stress}")

    sbf_result = SoftBellmanFordResult(
        embedding=X,
        stress=float(stress),
        n_iter=total_iter,
        n_graph_updates=n_graph_updates,
        optimizer_results=optimizer_results,
    )

    if return_result:
        return sbf_result
    if return_n_iter:
        return sbf_result.embedding, sbf_result.stress, sbf_result.n_iter
    return sbf_result.embedding, sbf_result.stress


def optimize_soft_bellman_ford(*args, **kwargs):
    """Alias used by the higher-level API layer."""
    return soft_bellman_ford(*args, **kwargs)


# Backward-compatible aliases for code written before the rename.
RelaxedBellmanFordResult = SoftBellmanFordResult
relaxed_bellman_ford_distances = soft_bellman_ford_distances
relaxed_bellman_ford = soft_bellman_ford
optimize_relaxed_bellman_ford = optimize_soft_bellman_ford


__all__ = [
    "SoftBellmanFordResult",
    "soft_bellman_ford_distances",
    "soft_bellman_ford",
    "optimize_soft_bellman_ford",
    "RelaxedBellmanFordResult",
    "relaxed_bellman_ford_distances",
    "relaxed_bellman_ford",
    "optimize_relaxed_bellman_ford",
]
