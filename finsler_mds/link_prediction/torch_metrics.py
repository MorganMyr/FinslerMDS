"""Differentiable PyTorch kernels for the project's Finsler metrics."""

from __future__ import annotations

import torch

from finsler_mds.metrics import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)


def torch_metric_length(displacements: torch.Tensor, metric) -> torch.Tensor:
    """Evaluate a supported metric without breaking PyTorch autograd."""
    if displacements.ndim < 1:
        raise ValueError("displacements must have at least one dimension.")

    radius = torch.linalg.vector_norm(displacements, dim=-1)
    z = displacements[..., -1]
    nonzero = radius > torch.finfo(displacements.dtype).eps
    safe_radius = torch.where(nonzero, radius, torch.ones_like(radius))
    s = torch.where(nonzero, z / safe_radius, torch.zeros_like(radius))

    if isinstance(metric, RandersMetric):
        return radius + metric.alpha * z

    if isinstance(metric, MatsumotoMetric):
        if metric.forbidden_grad_norm is not None:
            raise ValueError(
                "MatsumotoMetric(forbidden_grad_norm=...) uses a manual surrogate "
                "gradient that is not supported by PyTorch autograd."
            )
        denominator = 1 - metric.alpha * s
        allowed = denominator > 0
        safe_denominator = torch.where(allowed, denominator, torch.ones_like(denominator))
        phi = torch.where(
            allowed,
            safe_denominator.reciprocal(),
            torch.full_like(denominator, torch.inf),
        )
        if metric.max_phi is not None:
            phi = torch.clamp_max(phi, metric.max_phi)
        return radius * phi

    if isinstance(metric, ConvexifiedMatsumotoMetric):
        if metric.alpha == 0:
            return radius
        denominator = 1 - metric.alpha * s
        linear_branch = s > 1 / (2 * metric.alpha)
        safe_denominator = torch.where(
            linear_branch, torch.ones_like(denominator), denominator
        )
        rational = safe_denominator.reciprocal()
        linear = 4 * metric.alpha * s
        phi = torch.where(linear_branch, linear, rational)
        return radius * phi

    raise TypeError(
        "Link prediction currently supports RandersMetric, MatsumotoMetric, "
        f"and ConvexifiedMatsumotoMetric, got {type(metric).__name__}."
    )


__all__ = ["torch_metric_length"]
