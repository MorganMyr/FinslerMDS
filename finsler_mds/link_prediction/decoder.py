"""Distance-based decoder used by Finsler link prediction."""

from __future__ import annotations

import torch
from torch import nn

from .torch_metrics import torch_metric_length


class FermiDiracDecoder(nn.Module):
    r"""Decode directed distances with ``sigmoid((r - d**2) / t)``."""

    def __init__(self, metric, *, radius: float, temperature: float):
        super().__init__()
        if radius <= 0:
            raise ValueError("radius must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.metric = metric
        self.radius = float(radius)
        self.temperature = float(temperature)

    def distances(self, displacements: torch.Tensor) -> torch.Tensor:
        return torch_metric_length(displacements, self.metric)

    def logits(self, displacements: torch.Tensor) -> torch.Tensor:
        distances = self.distances(displacements)
        return (self.radius - distances.square()) / self.temperature

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(displacements))


__all__ = ["FermiDiracDecoder"]
