"""Initial coordinates for direct Finsler node embeddings."""

from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b

import numpy as np


INITIALIZATION_NAMES = ("current", "normal", "radius", "spectral")
DEFAULT_INITIALIZATION = "current"
_SPECTRAL_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_SPECTRAL_CACHE_SIZE = 4


def spectral_initialization(
    edge_index: np.ndarray,
    num_nodes: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Return cached Laplacian eigenvectors of the observed undirected graph."""
    from scipy import sparse
    from sklearn.manifold import spectral_embedding

    edges = np.ascontiguousarray(edge_index, dtype=np.int64)
    digest = blake2b(edges.view(np.uint8), digest_size=16).digest()
    key = (num_nodes, dimension, seed, digest)
    if key in _SPECTRAL_CACHE:
        _SPECTRAL_CACHE.move_to_end(key)
        return _SPECTRAL_CACHE[key]

    rows = np.concatenate((edges[0], edges[1]))
    columns = np.concatenate((edges[1], edges[0]))
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(num_nodes, num_nodes),
    )
    adjacency.data.fill(1)
    coordinates = spectral_embedding(
        adjacency,
        n_components=dimension,
        eigen_solver="arpack",
        random_state=seed,
        drop_first=True,
    ).astype(np.float32, copy=False)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    coordinates *= np.sqrt(2 / _mean_pair_squared_distance(coordinates))

    _SPECTRAL_CACHE[key] = coordinates
    _SPECTRAL_CACHE.move_to_end(key)
    while len(_SPECTRAL_CACHE) > _SPECTRAL_CACHE_SIZE:
        _SPECTRAL_CACHE.popitem(last=False)
    return coordinates


def _mean_pair_squared_distance(coordinates) -> float:
    """Mean squared Euclidean distance over all unordered node pairs."""
    return float(2 * np.square(coordinates).sum() / (len(coordinates) - 1))


__all__ = [
    "DEFAULT_INITIALIZATION",
    "INITIALIZATION_NAMES",
    "spectral_initialization",
]
