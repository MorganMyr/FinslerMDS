"""Directed link prediction with trainable Finsler node embeddings.

Dataset and split utilities stay importable without PyTorch. Training modules
are intentionally imported from their submodules so link prediction remains an
optional extension of the core Finsler-MDS package.
"""

from .data import DirectedGraphData, DirectedGraphStatistics
from .datasets import load_directed_dataset
from .split_cache import load_or_create_splits
from .splits import (
    EdgeExamples,
    LinkPredictionSplit,
    LinkTask,
    SPLIT_PROTOCOL,
    generate_splits,
    split_protocol_metadata,
)

__all__ = [
    "DirectedGraphData",
    "DirectedGraphStatistics",
    "EdgeExamples",
    "LinkPredictionSplit",
    "LinkTask",
    "SPLIT_PROTOCOL",
    "generate_splits",
    "load_directed_dataset",
    "load_or_create_splits",
    "split_protocol_metadata",
]
