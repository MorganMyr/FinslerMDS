"""Vectorized metric kernels shared by optimizer backends."""

from __future__ import annotations

from finsler_mds.metrics import (
    ConvexifiedMatsumotoMetric,
    ConvexifiedToblerMetric,
    MatsumotoMetric,
    RandersMetric,
)


def gpu_metric_supported(metric):
    if isinstance(metric, MatsumotoMetric) and metric.forbidden_grad_norm is not None:
        return False
    return isinstance(
        metric,
        (
            RandersMetric,
            MatsumotoMetric,
            ConvexifiedMatsumotoMetric,
            ConvexifiedToblerMetric,
        ),
    )


def cupy_metric_length_and_grad(cp, edge_vectors, metric):
    r = cp.linalg.norm(edge_vectors, axis=1)
    z = edge_vectors[:, -1]
    nonzero = r > 1e-12
    safe_r = cp.where(nonzero, r, 1.0)
    s = cp.where(nonzero, z / safe_r, 0.0)

    if isinstance(metric, RandersMetric):
        length = r + metric.alpha * z
        grad = cp.where(nonzero[:, None], edge_vectors / safe_r[:, None], 0.0)
        grad[:, -1] += metric.alpha
        grad = cp.where(nonzero[:, None], grad, 0.0)
        return length, grad

    if isinstance(metric, MatsumotoMetric):
        denominator = 1 - metric.alpha * s
        allowed = denominator > 0
        phi = cp.where(allowed, 1.0 / denominator, cp.inf)
        dphi = cp.where(allowed, metric.alpha / denominator**2, cp.nan)
        if metric.max_phi is not None:
            clipped = phi >= metric.max_phi
            phi = cp.minimum(phi, metric.max_phi)
            dphi = cp.where(clipped, 0.0, dphi)
        if metric.forbidden_grad_norm is not None and cp.any(~allowed & nonzero).item():
            raise ValueError(
                "The GPU backend does not support MatsumotoMetric with "
                "forbidden_grad_norm. Use device='cpu' for this metric."
            )
    elif isinstance(metric, ConvexifiedMatsumotoMetric):
        if metric.alpha == 0:
            phi = cp.ones_like(s)
            dphi = cp.zeros_like(s)
        else:
            linear = s > 1 / (2 * metric.alpha)
            denominator = 1 - metric.alpha * s
            phi = cp.where(linear, 4 * metric.alpha * s, 1 / denominator)
            dphi = cp.where(linear, 4 * metric.alpha, metric.alpha / denominator**2)
    elif isinstance(metric, ConvexifiedToblerMetric):
        slope_denominator = cp.sqrt(cp.maximum(1 - s**2, 0.0))
        finite_slope = slope_denominator > 1e-12
        slope = cp.where(finite_slope, s / slope_denominator, cp.sign(s) * cp.inf)
        dslope = cp.where(finite_slope, 1.0 / slope_denominator**3, cp.inf)
        shifted = slope + metric.b
        base_phi = cp.exp(metric.a * cp.abs(shifted)) / metric.speed
        base_dphi = base_phi * metric.a * cp.sign(shifted) * dslope

        uphill = s > metric.s_uphill
        downhill = s < metric.s_downhill
        phi = cp.where(uphill, s / metric.z_max, base_phi)
        phi = cp.where(downhill, s / metric.z_min, phi)
        dphi = cp.where(uphill, 1.0 / metric.z_max, base_dphi)
        dphi = cp.where(downhill, 1.0 / metric.z_min, dphi)
    else:
        raise TypeError(f"Unsupported GPU metric {type(metric).__name__}.")

    length = r * phi
    coeff = phi - s * dphi
    direction = cp.where(nonzero[:, None], edge_vectors / safe_r[:, None], 0.0)
    grad = coeff[:, None] * direction
    grad[:, -1] += dphi
    grad = cp.where(nonzero[:, None], grad, 0.0)
    return length, grad


__all__ = [
    "cupy_metric_length_and_grad",
    "gpu_metric_supported",
]
