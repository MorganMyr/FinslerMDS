"""Direct node-embedding model for Finsler link prediction."""

from __future__ import annotations

import math

import torch
from torch import nn

from .decoder import FermiDiracDecoder
from .initialization import DEFAULT_INITIALIZATION, INITIALIZATION_NAMES


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
        initialization: str = DEFAULT_INITIALIZATION,
        initial_embedding=None,
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
        self._initialize_embedding(initialization, initial_embedding, radius)

    @torch.no_grad()
    def _initialize_embedding(self, initialization, initial_embedding, radius):
        if initialization not in INITIALIZATION_NAMES:
            raise ValueError(f"Unknown embedding initialization {initialization!r}.")
        # Always consume the same Gaussian draw so later training randomness is
        # unchanged when only the initialization mode changes.
        nn.init.normal_(self.embedding.weight, std=1 / math.sqrt(self.dimension))
        if initialization == "spectral":
            if initial_embedding is None:
                raise ValueError("Spectral initialization requires coordinates.")
            self.embedding.weight.copy_(
                torch.as_tensor(
                    initial_embedding,
                    dtype=self.embedding.weight.dtype,
                    device=self.embedding.weight.device,
                )
            )
        self.center_embeddings_()
        if initialization == "normal":
            self.embedding.weight.mul_(math.sqrt(self.dimension))
        elif initialization == "radius":
            mean_pair_squared = (
                2 * self.embedding.weight.square().sum() / (self.num_nodes - 1)
            )
            self.embedding.weight.mul_(math.sqrt(radius / mean_pair_squared.item()))

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
