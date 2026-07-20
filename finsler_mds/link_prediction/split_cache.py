"""Versioned persistence for link-prediction splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import DirectedGraphData
from .splits import (
    EdgeExamples,
    LinkPredictionSplit,
    LinkTask,
    SPLIT_PROTOCOL,
    generate_splits,
    split_protocol_metadata,
)


_CACHE_FORMAT = f"{SPLIT_PROTOCOL}_splits"


def load_or_create_splits(
    path,
    graph: DirectedGraphData,
    task: LinkTask | str,
    *,
    num_splits: int = 10,
    first_seed: int = 0,
) -> list[LinkPredictionSplit]:
    path = Path(path)
    if path.exists():
        return load_splits(
            path,
            graph,
            task=task,
            expected_num_splits=num_splits,
            expected_first_seed=first_seed,
        )
    splits = generate_splits(
        graph,
        task,
        num_splits=num_splits,
        first_seed=first_seed,
    )
    save_splits(path, graph, splits)
    return splits


def save_splits(path, graph: DirectedGraphData, splits: list[LinkPredictionSplit]):
    if not splits:
        raise ValueError("Cannot save an empty split list.")
    task = splits[0].task
    if any(split.task != task for split in splits):
        raise ValueError("All cached splits must use the same task.")
    metadata = {
        "format": _CACHE_FORMAT,
        "graph_name": graph.name,
        "graph_fingerprint": graph.fingerprint,
        "task": task.value,
        "seeds": [int(split.seed) for split in splits],
        **split_protocol_metadata(),
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for index, split in enumerate(splits):
        prefix = f"split_{index}"
        arrays[f"{prefix}_observed_edge_index"] = split.observed_edge_index
        for partition in ("train", "validation", "test"):
            examples = getattr(split, partition)
            arrays[f"{prefix}_{partition}_pairs"] = examples.pairs
            arrays[f"{prefix}_{partition}_labels"] = examples.labels

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(output, **arrays)


def load_splits(
    path,
    graph: DirectedGraphData,
    *,
    task: LinkTask | str,
    expected_num_splits: int | None = None,
    expected_first_seed: int | None = None,
) -> list[LinkPredictionSplit]:
    task = LinkTask(task)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format") != _CACHE_FORMAT:
            raise ValueError(f"Unsupported split cache format in {path}.")
        if metadata.get("graph_fingerprint") != graph.fingerprint:
            raise ValueError("Split cache does not match the supplied graph.")
        if metadata.get("task") != task.value:
            raise ValueError("Split cache was generated for a different task.")
        seeds = metadata["seeds"]
        if expected_num_splits is not None and len(seeds) != expected_num_splits:
            raise ValueError(
                f"Split cache contains {len(seeds)} splits; expected {expected_num_splits}."
            )
        if expected_first_seed is not None and seeds[0] != expected_first_seed:
            raise ValueError(
                f"Split cache starts at seed {seeds[0]}; expected {expected_first_seed}."
            )

        results = []
        for index, seed in enumerate(seeds):
            prefix = f"split_{index}"
            partitions = {
                partition: EdgeExamples(
                    archive[f"{prefix}_{partition}_pairs"],
                    archive[f"{prefix}_{partition}_labels"],
                )
                for partition in ("train", "validation", "test")
            }
            results.append(
                LinkPredictionSplit(
                    seed=int(seed),
                    task=task,
                    observed_edge_index=archive[f"{prefix}_observed_edge_index"],
                    train=partitions["train"],
                    validation=partitions["validation"],
                    test=partitions["test"],
                )
            )
    return results


__all__ = ["load_or_create_splits", "load_splits", "save_splits"]
