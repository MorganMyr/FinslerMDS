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


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        return None, exc

    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return None, RuntimeError("CuPy did not find a CUDA device.")
        values = cp.arange(4, dtype=cp.float64)
        indices = cp.asarray([0, 2], dtype=cp.int32)
        cp.asnumpy(values[indices] + 1.0)
        matrix = cp.eye(2, dtype=cp.float64)
        cp.asnumpy(matrix @ matrix)
    except Exception as exc:
        return None, exc
    return cp, None


def _resolve_gpu_backend(device, alpha, verbose, *, allow_zero_alpha=False):
    if device not in {"cpu", "auto", "gpu", "cuda"}:
        raise ValueError("device must be one of 'cpu', 'auto', 'gpu', or 'cuda'.")
    if device == "cpu":
        return None
    if alpha < 0 or (alpha == 0 and not allow_zero_alpha):
        message = (
            "smacof_randers GPU backend requires metric.alpha > 0, "
            "except for alpha=0 with uniform weights and project_on_V=True."
        )
        if device == "auto":
            if verbose:
                print(message + " Falling back to CPU.")
            return None
        raise ValueError(message)

    cp, error = _load_cupy()
    if cp is None:
        message = f"smacof_randers GPU backend unavailable: {error}"
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
        print(f"smacof_randers GPU backend enabled on CUDA device {device_id}: {device_name}")
    return cp


def _laplacian_from_weights(weight):
    V = -weight.copy()
    diag = np.arange(len(V))
    V[diag, diag] = 0
    V[diag, diag] += np.abs(V.sum(axis=1))
    return V


def _check_symmetric_weight(weight):
    if not np.allclose(weight, weight.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The corrected Randers-SMACOF MM update requires symmetric weights.")


def _solve_randers_update(
    *,
    V,
    A,
    right_mat,
    total_right_mat,
    solver,
    project_on_V,
    uniform_offdiag_weight=None,
    uniform_right_scale=1.0,
):
    if not _is_diagonal(right_mat):
        return _solve_randers_update_kron(
            V=V,
            A=A,
            right_mat=right_mat,
            total_right_mat=total_right_mat,
            solver=solver,
            project_on_V=project_on_V,
        )

    if uniform_offdiag_weight is not None and project_on_V:
        return _solve_uniform_projected_randers_update(
            total_right_mat,
            right_diag=np.diag(right_mat),
            offdiag_weight=uniform_offdiag_weight,
            right_scale=uniform_right_scale,
        )

    left_mat = V
    left_mat_2 = A
    if project_on_V:
        left_mat = V.T @ left_mat
        left_mat_2 = V.T @ left_mat_2
        total_right_mat = V.T @ total_right_mat

    return _solve_block_diagonal_randers_update(
        left_mat=left_mat,
        left_mat_2=left_mat_2,
        right_diag=np.diag(right_mat),
        total_right_mat=total_right_mat,
        solver=solver,
    )


def _solve_uniform_projected_randers_update(
    total_right_mat,
    *,
    right_diag,
    offdiag_weight,
    right_scale=1.0,
):
    centered_rhs = total_right_mat - total_right_mat.mean(axis=0, keepdims=True)
    denominators = len(total_right_mat) * offdiag_weight * (1 + right_scale * right_diag)
    return centered_rhs / denominators[None, :]


def _solve_block_diagonal_randers_update(
    *,
    left_mat,
    left_mat_2,
    right_diag,
    total_right_mat,
    solver,
):
    X = np.empty_like(total_right_mat, dtype=float)
    for coefficient in np.unique(right_diag):
        cols = np.flatnonzero(right_diag == coefficient)
        system = left_mat + coefficient * left_mat_2
        if solver == "pinv":
            X[:, cols] = scipy.linalg.pinv(system) @ total_right_mat[:, cols]
        else:
            operator = scipy.sparse.linalg.aslinearoperator(system)
            for col in cols:
                X[:, col] = _solve_linear_system(
                    operator,
                    total_right_mat[:, col],
                    solver=solver,
                    label=f"column {col}",
                )
    return X


def _solve_linear_system(operator, rhs, *, solver, label):
    if solver == "cg":
        solution, info = scipy.sparse.linalg.cg(operator, rhs)
        if info != 0:
            warnings.warn(f"cg did not fully converge for {label}, info={info}.", RuntimeWarning)
    elif solver == "gmres":
        solution, info = scipy.sparse.linalg.gmres(operator, rhs)
        if info != 0:
            warnings.warn(f"gmres did not fully converge for {label}, info={info}.", RuntimeWarning)
    else:
        raise ValueError("pseudo_inv_solver must be one of {'gmres', 'cg', 'pinv'}.")
    return solution


def _is_diagonal(matrix):
    return np.allclose(matrix, np.diag(np.diag(matrix)))


def _solve_randers_update_kron(
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


def _uniform_offdiag_weight(weight):
    n_samples = weight.shape[0]
    if n_samples < 2:
        return None
    offdiag = weight[~np.eye(n_samples, dtype=bool)]
    if not np.all(np.isfinite(offdiag)):
        return None
    offdiag_weight = float(offdiag[0])
    if offdiag_weight <= 0:
        return None
    if not np.allclose(offdiag, offdiag_weight):
        return None
    return offdiag_weight


def _uniform_randers_C(dissimilarities, alpha, n_components, offdiag_weight):
    C = np.zeros((dissimilarities.shape[0], n_components), dtype=float)
    C[:, -1] = alpha * offdiag_weight * np.sum(
        dissimilarities - dissimilarities.T,
        axis=1,
    )
    return C


def _cupy_randers_pairwise(cp, X, alpha):
    euclidean = _cupy_euclidean_distances(cp, X)
    z = X[:, -1]
    return euclidean + alpha * (z[None, :] - z[:, None])


def _cupy_euclidean_distances(cp, X):
    squared_norms = cp.sum(X * X, axis=1)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (X @ X.T)
    return cp.sqrt(cp.maximum(squared_distances, 0.0))


def _cupy_stress(cp, embedded_dissimilarities, dissimilarities, weight, *, uniform_weight, normalized_stress, denom):
    residual = embedded_dissimilarities - dissimilarities
    if weight is None:
        raw_stress = uniform_weight * cp.sum(residual * residual)
    else:
        raw_stress = cp.sum(weight * residual * residual)
    stress = float(cp.asnumpy(raw_stress))
    if normalized_stress:
        stress = np.sqrt(stress / denom) if denom is not None and denom > 0 else np.inf
    return stress


def _cupy_uniform_projected_update(cp, total_right_mat, *, right_diag, offdiag_weight, right_scale=1.0):
    centered_rhs = total_right_mat - cp.mean(total_right_mat, axis=0, keepdims=True)
    denominators = len(total_right_mat) * offdiag_weight * (1 + right_scale * right_diag)
    return centered_rhs / denominators[None, :]


def _smacof_randers_single_gpu(
    dissimilarities,
    *,
    cp,
    metric,
    X,
    max_iter,
    verbose,
    eps,
    normalized_stress,
    weight,
    pseudo_inv_solver,
    project_on_V,
    check_monotony,
    corrected=True,
):
    alpha = metric.alpha
    n_samples, n_components = X.shape
    uniform_offdiag_weight = _uniform_offdiag_weight(weight)
    use_uniform_projected_update = project_on_V and uniform_offdiag_weight is not None

    right_mat = np.zeros((n_components, n_components))
    right_mat[-1, -1] = alpha * alpha if corrected else 1.0
    right_diag = np.diag(right_mat)
    D_major = 0.5 * (dissimilarities + dissimilarities.T) if corrected else dissimilarities

    if use_uniform_projected_update:
        V = None
        A = None
        C = _uniform_randers_C(
            dissimilarities,
            alpha,
            n_components,
            uniform_offdiag_weight,
        )
    else:
        V = _laplacian_from_weights(weight)
        A = V if corrected else alpha * V
        mat_one_last_col = np.zeros((n_samples, n_components))
        mat_one_last_col[:, -1] = 1
        C = alpha * ((weight * dissimilarities - weight.T * dissimilarities.T) @ mat_one_last_col)

    X_gpu = cp.asarray(X)
    D_gpu = cp.asarray(dissimilarities)
    D_major_gpu = cp.asarray(D_major)
    C_gpu = cp.asarray(C)
    right_diag_gpu = cp.asarray(right_diag)
    weight_gpu = None if uniform_offdiag_weight is not None else cp.asarray(weight)
    uniform_weight = 1.0 if uniform_offdiag_weight is None else uniform_offdiag_weight
    denom = None
    if normalized_stress:
        if weight_gpu is None:
            denom = uniform_weight * float(cp.asnumpy(cp.sum(D_gpu * D_gpu)))
        else:
            denom = float(cp.asnumpy(cp.sum(weight_gpu * D_gpu * D_gpu)))

    old_X_gpu = X_gpu.copy()
    old_stress = _cupy_stress(
        cp,
        _cupy_randers_pairwise(cp, X_gpu, alpha),
        D_gpu,
        weight_gpu,
        uniform_weight=uniform_weight,
        normalized_stress=normalized_stress,
        denom=denom,
    )
    stress = old_stress
    diag = cp.arange(n_samples)

    for it in range(max_iter):
        if corrected:
            dis = _cupy_euclidean_distances(cp, X_gpu)
        else:
            dis = _cupy_randers_pairwise(cp, X_gpu, alpha)
        dis = cp.where(dis == 0, 1e-5, dis)
        ratio = D_major_gpu / dis
        B = -uniform_weight * ratio if weight_gpu is None else -ratio * weight_gpu
        B[diag, diag] = 0.0
        B[diag, diag] = -cp.sum(B, axis=1)
        total_right_mat_gpu = B @ X_gpu - (0.5 if corrected else 1.0) * C_gpu

        if use_uniform_projected_update:
            X_gpu = _cupy_uniform_projected_update(
                cp,
                total_right_mat_gpu,
                right_diag=right_diag_gpu,
                offdiag_weight=uniform_offdiag_weight,
                right_scale=1.0 if corrected else alpha,
            )
        else:
            total_right_mat = cp.asnumpy(total_right_mat_gpu)
            X = _solve_randers_update(
                V=V,
                A=A,
                right_mat=right_mat,
                total_right_mat=total_right_mat,
                solver=pseudo_inv_solver,
                project_on_V=project_on_V,
                uniform_offdiag_weight=uniform_offdiag_weight,
                uniform_right_scale=1.0 if corrected else alpha,
            )
            X_gpu = cp.asarray(X)

        stress = _cupy_stress(
            cp,
            _cupy_randers_pairwise(cp, X_gpu, alpha),
            D_gpu,
            weight_gpu,
            uniform_weight=uniform_weight,
            normalized_stress=normalized_stress,
            denom=denom,
        )
        if verbose >= 2:
            print(f"it: {it}, stress {stress}")

        if check_monotony:
            tolerance = 1e-10 * max(1.0, abs(old_stress)) if corrected else 0.1
            if stress > old_stress + tolerance:
                X_gpu = old_X_gpu
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
        old_X_gpu = X_gpu.copy()

    return SmacofRandersResult(
        embedding=cp.asnumpy(X_gpu),
        stress=float(stress),
        n_iter=it + 1,
    )


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
    device="cpu",
    corrected=True,
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
    if corrected:
        _check_symmetric_weight(weight)

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

    uniform_offdiag_weight = _uniform_offdiag_weight(weight)
    use_uniform_euclidean_update = (
        alpha == 0
        and project_on_V
        and uniform_offdiag_weight is not None
    )

    gpu_backend = _resolve_gpu_backend(
        device,
        alpha,
        verbose,
        allow_zero_alpha=use_uniform_euclidean_update,
    )
    if gpu_backend is not None:
        return _smacof_randers_single_gpu(
            dissimilarities,
            cp=gpu_backend,
            metric=metric,
            X=X,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            normalized_stress=normalized_stress,
            weight=weight,
            pseudo_inv_solver=pseudo_inv_solver,
            project_on_V=project_on_V,
            check_monotony=check_monotony,
            corrected=corrected,
        )

    use_uniform_projected_update = (
        alpha > 0
        and project_on_V
        and uniform_offdiag_weight is not None
    )
    V = None if use_uniform_projected_update else _laplacian_from_weights(weight)
    V_pinv = None if alpha > 0 or use_uniform_euclidean_update else np.linalg.pinv(V)

    right_mat = np.zeros((n_components, n_components))
    right_mat[-1, -1] = alpha * alpha if corrected else 1.0
    D_major = 0.5 * (dissimilarities + dissimilarities.T) if corrected else dissimilarities
    if use_uniform_projected_update:
        A = None
        C = _uniform_randers_C(
            dissimilarities,
            alpha,
            n_components,
            uniform_offdiag_weight,
        )
    else:
        A = V if corrected else alpha * V
        mat_one_last_col = np.zeros((n_samples, n_components))
        mat_one_last_col[:, -1] = 1
        C = alpha * ((weight * dissimilarities - weight.T * dissimilarities.T) @ mat_one_last_col)

    old_X = X.copy()
    denom = None
    if normalized_stress:
        denom = float(np.sum(weight * dissimilarities * dissimilarities))
    if alpha > 0:
        embedded_dissimilarities = metric.pairwise(X)
    else:
        embedded_dissimilarities = euclidean_distances(X)
    old_stress = (weight.ravel() * (embedded_dissimilarities.ravel() - dissimilarities.ravel()) ** 2).sum()
    if normalized_stress:
        old_stress = np.sqrt(old_stress / denom) if denom > 0 else np.inf
    stress = old_stress

    for it in range(max_iter):
        dis = euclidean_distances(X) if corrected else metric.pairwise(X)
        dis[dis == 0] = 1e-5
        ratio = D_major / dis
        B = -ratio * weight
        diag = np.arange(len(B))
        B[diag, diag] = 0.0
        B[diag, diag] = -B.sum(axis=1)

        if alpha == 0:
            if use_uniform_euclidean_update:
                X = (B @ X) / (n_samples * uniform_offdiag_weight)
                X -= X.mean(axis=0, keepdims=True)
            else:
                X = V_pinv @ B @ X
        else:
            total_right_mat = B @ X - (0.5 if corrected else 1.0) * C
            X = _solve_randers_update(
                V=V,
                A=A,
                right_mat=right_mat,
                total_right_mat=total_right_mat,
                solver=pseudo_inv_solver,
                project_on_V=project_on_V,
                uniform_offdiag_weight=uniform_offdiag_weight,
                uniform_right_scale=1.0 if corrected else alpha,
            )

        if alpha > 0:
            embedded_dissimilarities = metric.pairwise(X)
        else:
            embedded_dissimilarities = euclidean_distances(X)
        stress = (weight.ravel() * (embedded_dissimilarities.ravel() - dissimilarities.ravel()) ** 2).sum()
        if normalized_stress:
            stress = np.sqrt(stress / denom) if denom > 0 else np.inf
        if verbose >= 2:
            print(f"it: {it}, stress {stress}")

        if check_monotony:
            tolerance = 1e-10 * max(1.0, abs(old_stress)) if corrected else 0.1
            if stress > old_stress + tolerance:
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
    device="cpu",
    version="corrected",
    return_result=False,
):
    """Compute a Randers-MDS embedding with the Randers-SMACOF update.

    The Randers strength comes from ``metric.alpha``.
    ``device="auto"`` uses a CuPy/CUDA backend for the dense SMACOF iteration
    when available and falls back to CPU otherwise.

    version : {"corrected", "legacy"}, default="corrected"
        Which SMACOF implementation to use. ``corrected`` uses the current
        revised Randers-SMACOF implementation. ``legacy`` uses the previous
        dense implementation from this module, including GPU support.
    """
    metric = _validate_randers_metric(metric)
    dissimilarities = check_array(dissimilarities)
    if dissimilarities.shape[0] != dissimilarities.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")
    random_state = check_random_state(random_state)

    if version not in {"corrected", "legacy"}:
        raise ValueError("version must be one of {'corrected', 'legacy'}.")
    use_legacy_update = version == "legacy"

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
                device=device,
                corrected=not use_legacy_update,
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
                device=device,
                corrected=not use_legacy_update,
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
