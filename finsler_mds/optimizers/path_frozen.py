"""Path-Frozen geodesic optimizer for Finsler-MDS."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.optimize

from finsler_mds.evaluation.distance_embedding import compute_embedding_distances
from finsler_mds.optimizers.common import (
    initial_embedding,
    prepare_weights_and_mask,
    validate_metric,
)
from finsler_mds.optimizers.direct_stress import (
    build_direct_pairs_objective,
    build_direct_stress_objective,
)
from finsler_mds.optimizers.frozen_paths import build_frozen_path_objective
from finsler_mds.optimizers.metric_kernels import gpu_metric_supported, load_cupy
from finsler_mds.optimizers.pair_groups import PairSampler
from finsler_mds.utils.graph import symmetric_knn_graph


@dataclass(frozen=True)
class PathFrozenResult:
    """Detailed result returned when return_result is enabled."""

    embedding: np.ndarray
    stress: float
    n_iter: int
    n_path_updates: int
    history: list
    final_full_geodesic_stress: float | None = None
    final_normalized_full_geodesic_stress: float | None = None


def path_frozen(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    graph_neighbors=10,
    outer_iter=20,
    inner_iter=5,
    verbose=0,
    eps=1e-6,
    random_state=None,
    weight=None,
    n_local_pairs=None,
    n_landmark=0,
    random_landmark_fraction=1.0,
    targets_per_landmark=None,
    local_weight=1.0,
    direct_stress_weight=0.0,
    device="cpu",
    optimizer_options=None,
    outer_step_size=1.0,
    log_frequency=None,
    record_history=False,
    n_jobs=None,
    return_n_iter=False,
    return_result=False,
):
    """Optimize a frozen graph-geodesic stress.

    Local constraints use direct Finsler distances. Global constraints use
    shortest paths frozen at the beginning of each outer iteration. Automatic
    landmarks mix farthest-point landmarks, kept fixed, with random landmarks,
    resampled at every outer iteration. Local and global groups are balanced by
    total weight before local_weight is applied.

    targets_per_landmark optionally subsamples global targets at every outer
    iteration with an unbiased weight correction. direct_stress_weight adds an
    all-pairs direct Finsler-MDS stress. The inner solver is L-BFGS-B.

    When record_history is true, full geodesic stress is evaluated at the
    frequency selected by log_frequency. The final state is always included,
    including when early stopping occurs between two scheduled logs.
    """
    metric = validate_metric(metric)
    outer_iter = int(outer_iter)
    inner_iter = int(inner_iter)
    if outer_iter <= 0:
        raise ValueError("outer_iter must be positive.")
    if inner_iter <= 0:
        raise ValueError("inner_iter must be positive.")

    outer_step_size = float(outer_step_size)
    if not 0.0 <= outer_step_size <= 1.0:
        raise ValueError("outer_step_size must be between 0 and 1.")

    D, W = prepare_weights_and_mask(dissimilarities, weight)
    sampler = PairSampler(
        D,
        W,
        n_local_pairs=n_local_pairs,
        n_landmark=n_landmark,
        random_landmark_fraction=random_landmark_fraction,
        targets_per_landmark=targets_per_landmark,
        local_weight=local_weight,
        random_state=random_state,
    )
    first_batch = sampler.sample()

    X = initial_embedding(D, n_components, init, random_state)
    shape = X.shape
    gpu_backend = _resolve_gpu_backend(device, metric, verbose)
    direct_regularizer = build_direct_stress_objective(
        D=D,
        W=W,
        shape=shape,
        metric=metric,
        weight=direct_stress_weight,
        gpu_backend=gpu_backend,
    )

    options = {"maxiter": inner_iter, "gtol": eps}
    if verbose:
        options["disp"] = True
    if optimizer_options is not None:
        options.update(optimizer_options)

    log_frequency = _resolve_log_frequency(log_frequency, outer_iter)
    full_active_mask, full_denominator = _full_stress_mask_and_denominator(D, W)
    history = []
    old_stress = None
    total_inner_iter = 0
    final_full_stress = None
    final_normalized_full_stress = None
    optimization_start = perf_counter()
    logging_elapsed = 0.0

    if verbose:
        _print_configuration(
            first_batch,
            n_samples=len(D),
            n_landmark=n_landmark,
            random_landmark_fraction=random_landmark_fraction,
            targets_per_landmark=targets_per_landmark,
            local_weight=local_weight,
            direct_stress_weight=direct_stress_weight,
            log_frequency=log_frequency,
        )

    for outer_it in range(outer_iter):
        batch = first_batch if outer_it == 0 else sampler.sample()
        path_objective = build_frozen_path_objective(
            X,
            shape=shape,
            active_pairs=batch.geodesic_pairs,
            metric=metric,
            graph_neighbors=graph_neighbors,
            n_jobs=n_jobs,
            verbose=verbose,
            gpu_backend=gpu_backend,
            device=device,
        )
        local_objective = build_direct_pairs_objective(
            batch.direct_pairs,
            shape=shape,
            metric=metric,
            gpu_backend=gpu_backend,
        )
        objective = _sum_objectives(
            path_objective,
            local_objective,
            direct_regularizer,
        )

        X_start = X.copy()
        result = scipy.optimize.minimize(
            objective,
            X.ravel(),
            jac=True,
            method="L-BFGS-B",
            options=options,
        )
        X_optimized = result.x.reshape(shape)
        if outer_step_size == 1.0:
            X = X_optimized
            stress = float(result.fun)
        else:
            X = X_start + outer_step_size * (X_optimized - X_start)
            stress = float(objective(X.ravel())[0])
        total_inner_iter += int(getattr(result, "nit", inner_iter))

        converged = (
            not sampler.is_stochastic
            and old_stress is not None
            and old_stress != 0
            and np.abs(1 - stress / old_stress) < eps
        )
        is_final = converged or outer_it == outer_iter - 1
        should_log = _is_scheduled_log(outer_it, log_frequency) or is_final
        record_point = (record_history or verbose >= 2) and should_log
        evaluate_full = record_point or (verbose and is_final)

        nit = getattr(result, "nit", "?")
        nfev = getattr(result, "nfev", "?")
        elapsed = perf_counter() - optimization_start - logging_elapsed
        full_stress = None
        normalized_full_stress = None
        if evaluate_full:
            log_start = perf_counter()
            full_stress, normalized_full_stress = _full_geodesic_stress(
                X,
                D,
                W,
                metric=metric,
                active_mask=full_active_mask,
                denominator=full_denominator,
                graph_neighbors=graph_neighbors,
                n_jobs=n_jobs,
                warn_on_connect=verbose >= 1,
            )
            logging_elapsed += perf_counter() - log_start
            if is_final:
                final_full_stress = full_stress
                final_normalized_full_stress = normalized_full_stress
            if record_point:
                history.append(
                    {
                        "outer_iter": outer_it,
                        "elapsed": elapsed,
                        "masked_stress": stress,
                        "full_geodesic_stress": full_stress,
                        "normalized_full_geodesic_stress": normalized_full_stress,
                        "nit": nit,
                        "nfev": nfev,
                    }
                )

        if verbose >= 2 and should_log:
            print(
                f"path_frozen outer {outer_it}: masked stress {stress}, "
                f"full geodesic stress {full_stress}, "
                f"normalized {normalized_full_stress} "
                f"(elapsed={elapsed:.3f}s, nit={nit}, nfev={nfev})"
            )
        elif verbose and should_log:
            print(
                f"path_frozen outer {outer_it}: masked stress {stress} "
                f"(nit={nit}, nfev={nfev})"
            )

        old_stress = stress
        if is_final:
            break

    if verbose:
        print(f"path_frozen final full geodesic stress: {final_full_stress}")

    pf_result = PathFrozenResult(
        embedding=X,
        stress=float(stress),
        n_iter=total_inner_iter,
        n_path_updates=outer_it + 1,
        history=history,
        final_full_geodesic_stress=final_full_stress,
        final_normalized_full_geodesic_stress=final_normalized_full_stress,
    )
    if return_result:
        return pf_result
    if return_n_iter:
        return pf_result.embedding, pf_result.stress, pf_result.n_iter
    return pf_result.embedding, pf_result.stress


def _sum_objectives(*objectives):
    objectives = tuple(objective for objective in objectives if objective is not None)
    if not objectives:
        raise ValueError("Path-Frozen has no active objective.")

    def objective(X_flat):
        stress = 0.0
        grad = np.zeros_like(X_flat)
        for term in objectives:
            term_stress, term_grad = term(X_flat)
            stress += term_stress
            grad += term_grad
        return float(stress), grad

    return objective


def _full_stress_mask_and_denominator(D, W):
    active = (W != 0) & np.isfinite(D)
    np.fill_diagonal(active, False)
    denominator = float(np.sum(W[active] * D[active] ** 2))
    return active, denominator


def _full_geodesic_stress(
    X,
    D,
    W,
    *,
    metric,
    active_mask,
    denominator,
    graph_neighbors,
    n_jobs,
    warn_on_connect,
):
    support = symmetric_knn_graph(
        X,
        n_neighbors=graph_neighbors,
        neighbors_algorithm="auto",
        n_jobs=n_jobs,
        ensure_connected=True,
        warn_on_connect=warn_on_connect,
    )
    embedded = compute_embedding_distances(
        X,
        metric=metric,
        mode="geodesic",
        support_graph=support,
        neighbors_algorithm="auto",
        n_jobs=n_jobs,
    )
    if np.any(~np.isfinite(embedded[active_mask])):
        return np.inf, np.inf

    residual = embedded[active_mask] - D[active_mask]
    raw_stress = float(np.sum(W[active_mask] * residual**2))
    normalized = np.sqrt(raw_stress / denominator) if denominator > 0 else np.inf
    return raw_stress, float(normalized)


def _resolve_gpu_backend(device, metric, verbose):
    if device not in {"cpu", "auto", "gpu", "cuda"}:
        raise ValueError("device must be one of 'cpu', 'auto', 'gpu', or 'cuda'.")
    if device == "cpu":
        return None
    if not gpu_metric_supported(metric):
        message = (
            "Path-Frozen GPU support is limited to RandersMetric, "
            "MatsumotoMetric, and ConvexifiedMatsumotoMetric."
        )
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise ValueError(message)

    cp, error = load_cupy()
    if cp is None:
        message = f"Path-Frozen GPU backend unavailable: {error}"
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
        print(f"Path-Frozen GPU backend enabled on CUDA device {device_id}: {device_name}")
    return cp


def _default_log_frequency(max_iter):
    if max_iter < 30:
        return 1
    decade = 10 ** int(np.floor(np.log10(max_iter)))
    if max_iter < 3 * decade:
        return max(1, decade // 10)
    return max(1, decade // 2)


def _resolve_log_frequency(log_frequency, max_iter):
    if log_frequency is None:
        return _default_log_frequency(max_iter)
    log_frequency = int(log_frequency)
    if log_frequency < 0:
        raise ValueError("log_frequency must be non-negative or None.")
    return log_frequency


def _is_scheduled_log(iteration, log_frequency):
    return log_frequency > 0 and (
        iteration == 0 or iteration % log_frequency == 0
    )


def _print_configuration(
    batch,
    *,
    n_samples,
    n_landmark,
    random_landmark_fraction,
    targets_per_landmark,
    local_weight,
    direct_stress_weight,
    log_frequency,
):
    global_pairs = batch.geodesic_pairs.n_pairs
    local_pairs = batch.direct_pairs.n_pairs
    print(
        "path_frozen: "
        f"{global_pairs + local_pairs} pairs "
        f"({global_pairs} global-geodesic, {local_pairs} local-direct) over "
        f"{n_samples * (n_samples - 1)} off-diagonal pairs; "
        f"{len(batch.geodesic_pairs.sources)} geodesic sources"
    )
    if local_pairs:
        print(
            "path_frozen pair weights: "
            f"count balancing, global_factor={batch.global_factor:.6g}, "
            f"local_factor={batch.local_factor:.6g}, "
            f"local_weight={float(local_weight):.6g}"
        )
    if n_landmark:
        print(
            "path_frozen landmark sampling: "
            f"random_fraction={float(random_landmark_fraction):.6g}"
        )
    if targets_per_landmark is not None:
        print(
            "path_frozen global target sampling: "
            f"targets_per_landmark={int(targets_per_landmark)}"
        )
    if direct_stress_weight:
        print(
            "path_frozen direct MDS regularizer: "
            f"weight={float(direct_stress_weight):.6g}"
        )
    if log_frequency not in {0, 1}:
        print(f"path_frozen logging every {log_frequency} outer iterations")


__all__ = ["PathFrozenResult", "path_frozen"]
