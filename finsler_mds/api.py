"""Public entry points for Finsler-MDS experiments."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from finsler_mds.metrics import RandersMetric
from finsler_mds.optimizers import (
    finsler_umap,
    gradient_descent,
    path_frozen,
    smacof_randers,
)


_OPTIMIZERS = {
    "gradient_descent": gradient_descent,
    "gd": gradient_descent,
    "finsler_umap": finsler_umap,
    "fumap": finsler_umap,
    "path_frozen": path_frozen,
    "frozen_paths": path_frozen,
    "smacof": smacof_randers,
    "smacof_randers": smacof_randers,
    "randers_smacof": smacof_randers,
}


def fit_finsler_mds(
    dissimilarities,
    *,
    metric=None,
    optimizer="smacof_randers",
    print_time=False,
    **optimizer_kwargs,
):
    """Fit a Finsler-MDS embedding using the selected optimizer.

    Parameters
    ----------
    dissimilarities : array-like of shape (n_samples, n_samples)
        Pairwise target dissimilarities.

    metric : object, default=None
        Metric object consumed by the optimizer. The default is
        ``RandersMetric(alpha=0)`` so the current default optimizer has a valid
        metric.

    optimizer : str or callable, default="smacof_randers"
        Optimizer name or function. Callable optimizers must accept
        ``(dissimilarities, metric=metric, **optimizer_kwargs)``.

    print_time : bool, default=False
        If True, print the elapsed time spent inside the selected optimizer.

    **optimizer_kwargs
        Passed directly to the selected optimizer.
    """
    if metric is None:
        metric = RandersMetric(alpha=0.0)

    if isinstance(optimizer, str):
        try:
            optimizer_fn = _OPTIMIZERS[optimizer]
        except KeyError as exc:
            known = ", ".join(sorted(_OPTIMIZERS))
            raise ValueError(f"Unknown optimizer {optimizer!r}. Known optimizers: {known}.") from exc
    elif isinstance(optimizer, Callable):
        optimizer_fn = optimizer
    else:
        raise TypeError("optimizer must be a string name or a callable.")

    start = perf_counter()
    result = optimizer_fn(dissimilarities, metric=metric, **optimizer_kwargs)
    elapsed = perf_counter() - start
    if print_time:
        optimizer_name = optimizer if isinstance(optimizer, str) else optimizer_fn.__name__
        print(f"Finsler-MDS ({optimizer_name}) finished in {elapsed:.3f} s")
    return result


__all__ = [
    "fit_finsler_mds",
]
