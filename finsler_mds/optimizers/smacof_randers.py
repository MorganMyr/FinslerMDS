"""Randers-specific SMACOF optimizer.

This module contains the closed-form majorization update used by the original
Finsler-MDS code. It is specific to ``RandersMetric``; other Finsler metrics
should use generic gradient-based optimizers unless a dedicated update is
derived for them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from joblib import effective_n_jobs
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.parallel import Parallel, delayed

from finsler_mds.metrics import RandersMetric


@dataclass(frozen=True)
class SmacofRandersResult:
    embedding: np.ndarray
    stress: float
    n_iter: int


def _validate_randers_metric(metric):
    if metric is None:
        return RandersMetric(alpha=0.0)
    if not isinstance(metric, RandersMetric):
        raise TypeError(
            "smacof_randers requires a RandersMetric. "
            "Use a gradient-based optimizer for other Finsler metrics."
        )
    return metric


def _laplacian_from_weights(weight):
    V = -weight.copy()
    diag = np.arange(len(V))
    V[diag, diag] = 0
    V[diag, diag] += np.abs(V.sum(axis=1))
    return V


def _solve_randers_update(
    *,
    V,
    A,
    right_mat,
    total_right_mat,
    solver,
    project_on_V,
):
    left_mat = V
    left_mat_2 = A
    if project_on_V:
        left_mat = V.T @ left_mat
        left_mat_2 = V.T @ left_mat_2
        total_right_mat = V.T @ total_right_mat

    system = scipy.sparse.csr_array(np.kron(np.eye(*right_mat.shape), left_mat))
    system += scipy.sparse.csr_array(np.kron(right_mat.T, left_mat_2))
    rhs = total_right_mat.flatten(order="F")

    if solver == "pinv":
        X_flat = scipy.linalg.pinv(system.todense()) @ rhs
    elif solver == "cg":
        X_flat, info = scipy.sparse.linalg.cg(system, rhs)
        if info != 0:
            warnings.warn(f"cg did not fully converge, info={info}.", RuntimeWarning)
    elif solver == "gmres":
        X_flat, info = scipy.sparse.linalg.gmres(system, rhs)
        if info != 0:
            warnings.warn(f"gmres did not fully converge, info={info}.", RuntimeWarning)
    else:
        raise ValueError("pseudo_inv_solver must be one of {'gmres', 'cg', 'pinv'}.")

    return X_flat.reshape(total_right_mat.shape, order="F")


def _smacof_randers_single(
    dissimilarities,
    *,
    metric,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    normalized_stress=False,
    weight=None,
    pseudo_inv_solver="gmres",
    project_on_V=False,
    check_monotony=True,
):
    """Run one Randers-SMACOF initialization."""
    metric = _validate_randers_metric(metric)
    alpha = metric.alpha
    if not 0 <= alpha < 1:
        raise ValueError("Randers-SMACOF requires 0 <= metric.alpha < 1.")

    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)
    if weight is None:
        weight = np.ones_like(dissimilarities, dtype=float)
    else:
        weight = np.asarray(weight, dtype=float)
        if weight.shape != dissimilarities.shape:
            raise ValueError("weight must have the same shape as dissimilarities.")

    if init is None:
        X = random_state.uniform(size=n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        X = np.asarray(init, dtype=float).copy()
        n_components = X.shape[1]
        if X.shape[0] != n_samples:
            raise ValueError(
                f"init should have shape ({n_samples}, {n_components}), "
                f"got {X.shape}."
            )

    V = _laplacian_from_weights(weight)
    V_pinv = None if alpha > 0 else np.linalg.pinv(V)

    diag_one_end = np.zeros((n_components, n_components))
    diag_one_end[-1, -1] = 1
    diag_sum_weights = np.diag(weight.sum(axis=1))
    A = alpha * (diag_sum_weights - weight)
    mat_one_last_col = np.zeros((n_samples, n_components))
    mat_one_last_col[:, -1] = 1
    C = alpha * ((weight * dissimilarities - weight.T * dissimilarities.T) @ mat_one_last_col)

    old_X = X.copy()
    old_stress = None

    for it in range(max_iter):
        if alpha > 0:
            embedded_dissimilarities = metric.pairwise(X)
        else:
            embedded_dissimilarities = euclidean_distances(X)

        stress = (weight.ravel() * (embedded_dissimilarities.ravel() - dissimilarities.ravel()) ** 2).sum()
        if normalized_stress:
            denom = (weight.ravel() * dissimilarities.ravel() ** 2).sum()
            stress = np.sqrt(stress / denom) if denom > 0 else np.inf

        dis = embedded_dissimilarities.copy()
        dis[dis == 0] = 1e-5
        ratio = dissimilarities / dis
        B = -ratio * weight
        diag = np.arange(len(B))
        B[diag, diag] += -B.sum(axis=1)

        if alpha == 0:
            X = V_pinv @ B @ X
        else:
            total_right_mat = B @ X - C
            X = _solve_randers_update(
                V=V,
                A=A,
                right_mat=diag_one_end,
                total_right_mat=total_right_mat,
                solver=pseudo_inv_solver,
                project_on_V=project_on_V,
            )

        if verbose >= 2:
            print(f"it: {it}, stress {stress}")

        if old_stress is not None and check_monotony:
            if stress > old_stress + 0.1:
                X = old_X
                stress = old_stress
                it -= 1
                if verbose:
                    print(f"breaking at iteration {it} due to stress increase, stress {stress}")
                break
            if np.abs(1 - stress / old_stress) < eps:
                if verbose:
                    print(f"breaking at iteration {it} with stress {stress}")
                break

        old_stress = stress
        old_X = X.copy()

    return SmacofRandersResult(embedding=X, stress=float(stress), n_iter=it + 1)


def smacof_randers(
    dissimilarities,
    *,
    metric=None,
    n_components=2,
    init=None,
    n_init=8,
    n_jobs=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    return_n_iter=False,
    normalized_stress=False,
    weight=None,
    pseudo_inv_solver="gmres",
    project_on_V=False,
    check_monotony=True,
    return_result=False,
):
    """Compute a Randers-MDS embedding with the Randers-SMACOF update.

    Parameters are intentionally close to the legacy ``_mds_finsler.smacof``
    function, but the Randers strength now comes from ``metric.alpha``.
    """
    metric = _validate_randers_metric(metric)
    dissimilarities = check_array(dissimilarities)
    if dissimilarities.shape[0] != dissimilarities.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")
    random_state = check_random_state(random_state)

    if normalized_stress == "auto":
        normalized_stress = False
    if normalized_stress == "warn":
        warnings.warn(
            "normalized_stress='warn' is a legacy value; using False.",
            FutureWarning,
        )
        normalized_stress = False
    if normalized_stress not in (True, False):
        raise ValueError("normalized_stress must be True, False, 'auto', or 'warn'.")

    if hasattr(init, "__array__"):
        init = np.asarray(init, dtype=float).copy()
        if n_init != 1:
            warnings.warn(
                "Explicit initial positions passed: performing only one init "
                f"instead of {n_init}.",
                RuntimeWarning,
            )
            n_init = 1

    if effective_n_jobs(n_jobs) == 1:
        best = None
        for _ in range(n_init):
            result = _smacof_randers_single(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=random_state,
                normalized_stress=normalized_stress,
                weight=weight,
                pseudo_inv_solver=pseudo_inv_solver,
                project_on_V=project_on_V,
                check_monotony=check_monotony,
            )
            if best is None or result.stress < best.stress:
                best = result
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_randers_single)(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized_stress,
                weight=weight,
                pseudo_inv_solver=pseudo_inv_solver,
                project_on_V=project_on_V,
                check_monotony=check_monotony,
            )
            for seed in seeds
        )
        best = min(results, key=lambda result: result.stress)

    if return_result:
        return best
    if return_n_iter:
        return best.embedding, best.stress, best.n_iter
    return best.embedding, best.stress


def optimize_smacof_randers(*args, **kwargs):
    """Alias used by the higher-level API layer."""
    return smacof_randers(*args, **kwargs)


__all__ = [
    "SmacofRandersResult",
    "smacof_randers",
    "optimize_smacof_randers",
]
