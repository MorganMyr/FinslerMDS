"""Data structures for directed link-prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _edge_array(edge_index, *, copy: bool = True):
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, n_edges), got {edges.shape}.")
    return np.array(edges, dtype=np.int64, copy=copy, order="C")


@dataclass(frozen=True)
class DirectedGraphData:
    """Canonical directed graph used by the benchmark.

    Edge order is intentionally preserved: the StellarGraph-style splitter inherits
    deterministic ordering from the graph loader before applying its seeded
    shuffle.
    """

    name: str
    num_nodes: int
    edge_index: np.ndarray
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must not be empty.")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        edges = _edge_array(self.edge_index)
        if edges.size:
            if edges.min() < 0 or edges.max() >= self.num_nodes:
                raise ValueError("edge_index contains a node outside [0, num_nodes).")
            linear = edges[0] * self.num_nodes + edges[1]
            if np.unique(linear).size != linear.size:
                raise ValueError("edge_index must not contain duplicate directed edges.")

        edges.setflags(write=False)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint including edge order and node count."""
        digest = sha256()
        digest.update(np.asarray([self.num_nodes], dtype="<i8").tobytes())
        digest.update(self.edge_index.astype("<i8", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def statistics(self) -> "DirectedGraphStatistics":
        src, dst = self.edge_index
        loops = src == dst
        non_loop_src = src[~loops]
        non_loop_dst = dst[~loops]
        directed_keys = non_loop_src * self.num_nodes + non_loop_dst
        reverse_keys = non_loop_dst * self.num_nodes + non_loop_src
        reciprocal_arcs = np.intersect1d(
            directed_keys, reverse_keys, assume_unique=True
        ).size
        unordered_keys = (
            np.minimum(non_loop_src, non_loop_dst) * self.num_nodes
            + np.maximum(non_loop_src, non_loop_dst)
        )
        return DirectedGraphStatistics(
            num_nodes=self.num_nodes,
            num_directed_edges=self.num_edges,
            num_self_loops=int(loops.sum()),
            num_non_loop_directed_edges=int(len(directed_keys)),
            num_reciprocal_pairs=int(reciprocal_arcs // 2),
            num_unordered_pairs=int(np.unique(unordered_keys).size),
        )

    def without_self_loops(self) -> "DirectedGraphData":
        keep = self.edge_index[0] != self.edge_index[1]
        if bool(np.all(keep)):
            return self
        metadata = dict(self.metadata)
        metadata["removed_self_loops"] = int((~keep).sum())
        return DirectedGraphData(
            name=self.name,
            num_nodes=self.num_nodes,
            edge_index=self.edge_index[:, keep],
            source=self.source,
            metadata=metadata,
        )


@dataclass(frozen=True)
class DirectedGraphStatistics:
    num_nodes: int
    num_directed_edges: int
    num_self_loops: int
    num_non_loop_directed_edges: int
    num_reciprocal_pairs: int
    num_unordered_pairs: int

    def as_dict(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in vars(self).items()
        }


__all__ = [
    "DirectedGraphData",
    "DirectedGraphStatistics",
]
