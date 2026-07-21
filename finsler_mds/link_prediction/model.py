"""Direct node-embedding model for Finsler link prediction."""

from __future__ import annotations

import math

import torch
from torch import nn

from .decoder import FermiDiracDecoder


class FinslerLinkPredictor(nn.Module):
    """One trainable coordinate per node, with no MLP or GNN encoder."""

    def __init__(
        self,
        num_nodes: int,
        dimension: int,
        metric,
        *,
        radius: float,
        temperature: float,
    ):
        super().__init__()
        if num_nodes <= 0 or dimension <= 0:
            raise ValueError("num_nodes and dimension must be positive.")
        self.num_nodes = int(num_nodes)
        self.dimension = int(dimension)
        self.embedding = nn.Embedding(num_nodes, dimension)
        self.decoder = FermiDiracDecoder(
            metric,
            radius=radius,
            temperature=temperature,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=1 / math.sqrt(self.dimension))
        self.center_embeddings_()

    def edge_displacements(self, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_examples, 2).")
        source = self.embedding(pairs[:, 0])
        target = self.embedding(pairs[:, 1])
        return target - source

    def logits(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.decoder.logits(self.edge_displacements(pairs))

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(pairs))

    @torch.no_grad()
    def center_embeddings_(self):
        self.embedding.weight.sub_(self.embedding.weight.mean(dim=0, keepdim=True))
        return self


__all__ = ["FinslerLinkPredictor"]
