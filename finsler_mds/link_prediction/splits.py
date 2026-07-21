"""Binary link-prediction splits reconstructing MagNet's noisy protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np

from .data import DirectedGraphData


SPLIT_PROTOCOL = "magnet_random_noisy"
_TEST_FRACTION = 0.15
_VALIDATION_FRACTION = 0.05
_TRAIN_EXAMPLE_FRACTION = 0.99
_BALANCE_SEED = 1000


class LinkTask(str, Enum):
    EXISTENCE = "existence"
    DIRECTION = "direction"


def split_protocol_metadata() -> dict:
    """Serializable definition shared by caches and experiment records."""
    return {
        "protocol": SPLIT_PROTOCOL,
        "reference": "MagNet EdgeSplitter pipeline with randomized noisy labels",
        "test_fraction": _TEST_FRACTION,
        "validation_fraction_of_remainder": _VALIDATION_FRACTION,
        "training_positive_fraction_of_observed": _TRAIN_EXAMPLE_FRACTION,
        "existence_mix": "raw 25% edge / 75% negative, then class-balanced",
        "reciprocal_policy": "included with seeded random ambiguous targets",
        "observed_reciprocal_policy": "both directed arcs retained",
        "negative_sampling_reference": "full_graph",
        "partitions_disjoint": True,
    }


@dataclass(frozen=True)
class EdgeExamples:
    pairs: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        pairs = np.asarray(self.pairs, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=np.float32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must have shape (n_examples, 2), got {pairs.shape}.")
        if labels.ndim != 1 or len(labels) != len(pairs):
            raise ValueError("labels must have shape (n_examples,).")
        if len(labels) == 0 or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must be a non-empty binary array.")
        object.__setattr__(self, "pairs", np.array(pairs, copy=True, order="C"))
        object.__setattr__(self, "labels", np.array(labels, copy=True, order="C"))

    @property
    def num_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def num_negative(self) -> int:
        return int(len(self.labels) - self.num_positive)


@dataclass(frozen=True)
class LinkPredictionSplit:
    seed: int
    task: LinkTask
    observed_edge_index: np.ndarray
    train: EdgeExamples
    validation: EdgeExamples
    test: EdgeExamples

    def __post_init__(self):
        object.__setattr__(self, "task", LinkTask(self.task))
        edges = np.asarray(self.observed_edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(
                "observed_edge_index must have shape (2, n_edges), "
                f"got {edges.shape}."
            )
        object.__setattr__(self, "observed_edge_index", np.array(edges, copy=True))


def generate_splits(
    graph: DirectedGraphData,
    task: LinkTask | str,
    *,
    num_splits: int = 10,
    first_seed: int = 0,
    evaluation_reverse_negative_fraction: float | None = None,
) -> list[LinkPredictionSplit]:
    """Generate the noisy binary tasks used by MagNet's EdgeSplitter pipeline.

    The only deliberate correction is that sampled non-edges are checked against
    the complete original graph and cannot be reused by another partition.
    """
    task = LinkTask(task)
    if num_splits <= 0:
        raise ValueError("num_splits must be positive.")
    if evaluation_reverse_negative_fraction is not None and not (
        0 <= evaluation_reverse_negative_fraction <= 1
    ):
        raise ValueError("evaluation_reverse_negative_fraction must be in [0, 1].")
    if np.any(graph.edge_index[0] == graph.edge_index[1]):
        raise ValueError("Remove self-loops before generating splits.")

    directed_edges = set(map(tuple, graph.edge_index.T.tolist()))
    full_graph = _underlying_graph(graph)
    results = []
    for seed in range(first_seed, first_seed + num_splits):
        used_non_edges: set[tuple[int, int]] = set()

        test_rng = np.random.RandomState(seed)
        after_test, test_positive = _remove_edges(
            full_graph, _TEST_FRACTION, test_rng, keep_connected=True
        )
        test = _make_examples(
            test_positive,
            directed_edges,
            task,
            full_graph,
            test_rng,
            used_non_edges,
            evaluation_reverse_negative_fraction,
        )

        validation_rng = np.random.RandomState(seed)
        observed, validation_positive = _remove_edges(
            after_test, _VALIDATION_FRACTION, validation_rng, keep_connected=True
        )
        validation = _make_examples(
            validation_positive,
            directed_edges,
            task,
            full_graph,
            validation_rng,
            used_non_edges,
            evaluation_reverse_negative_fraction,
        )

        train_rng = np.random.RandomState(seed)
        _, train_positive = _remove_edges(
            observed, _TRAIN_EXAMPLE_FRACTION, train_rng, keep_connected=False
        )
        train = _make_examples(
            train_positive, directed_edges, task, full_graph, train_rng, used_non_edges
        )
        results.append(
            LinkPredictionSplit(
                seed=seed,
                task=task,
                observed_edge_index=_observed_edges(observed, directed_edges),
                train=train,
                validation=validation,
                test=test,
            )
        )
    return results


def _underlying_graph(graph: DirectedGraphData) -> nx.Graph:
    result = nx.Graph()
    result.add_nodes_from(range(graph.num_nodes))
    result.add_edges_from(graph.edge_index.T.tolist())
    return result


def _remove_edges(
    graph: nx.Graph,
    fraction: float,
    rng: np.random.RandomState,
    *,
    keep_connected: bool,
) -> tuple[nx.Graph, np.ndarray]:
    """Port StellarGraph ``EdgeSplitter`` positive sampling."""
    reduced = graph.copy()
    protected: set[tuple[int, int]] = set()
    if keep_connected:
        spanning_forest = list(nx.minimum_spanning_edges(graph, data=False))
        protected.update(spanning_forest)
        protected.update((v, u) for u, v in spanning_forest)

    target = int(graph.number_of_edges() * fraction)
    if target == 0:
        raise ValueError("Split fraction selects no edges; use a larger graph.")
    if target > graph.number_of_edges() - len(protected) // 2:
        raise ValueError("Not enough removable edges to preserve graph connectivity.")

    candidates = list(reduced.edges())
    rng.shuffle(candidates)
    removed = []
    for edge in candidates:
        if edge in protected:
            continue
        removed.append(edge)
        reduced.remove_edge(*edge)
        if len(removed) == target:
            break
    if len(removed) != target:
        raise RuntimeError("Positive-edge sampling did not reach its target.")
    return reduced, np.asarray(removed, dtype=np.int64)


def _make_examples(
    positive_pairs: np.ndarray,
    directed_edges: set[tuple[int, int]],
    task: LinkTask,
    full_graph: nx.Graph,
    rng: np.random.RandomState,
    used_non_edges: set[tuple[int, int]],
    reverse_negative_fraction: float | None = None,
) -> EdgeExamples:
    non_edges = np.empty((0, 2), dtype=np.int64)
    if task is LinkTask.EXISTENCE:
        sampled = _sample_global_non_edges(
            full_graph, len(positive_pairs), rng, excluded=used_non_edges
        )
        used_non_edges.update(sampled)
        used_non_edges.update((v, u) for u, v in sampled)
        non_edges = np.asarray(sampled, dtype=np.int64)
    return _label_noisy(
        positive_pairs,
        non_edges,
        directed_edges,
        task,
        rng,
        reverse_negative_fraction,
    )


def _label_noisy(
    positive_pairs: np.ndarray,
    non_edges: np.ndarray,
    directed_edges: set[tuple[int, int]],
    task: LinkTask,
    rng: np.random.RandomState,
    reverse_negative_fraction: float | None = None,
) -> EdgeExamples:
    """Alternate orientations and randomize the target of reciprocal pairs."""
    pairs = np.asarray(positive_pairs, dtype=np.int64).copy()
    labels = np.empty(len(pairs), dtype=np.float32)
    for index, (u_raw, v_raw) in enumerate(pairs):
        u, v = int(u_raw), int(v_raw)
        forward = (u, v) in directed_edges
        reverse = (v, u) in directed_edges
        if forward:
            source, target = u, v
        elif reverse:
            source, target = v, u
        else:
            raise ValueError(f"Positive pair ({u}, {v}) is absent from the graph.")
        if forward and reverse and rng.randint(2):
            source, target = target, source
        if index % 2:
            pairs[index] = (target, source)
            labels[index] = 0
        else:
            pairs[index] = (source, target)
            labels[index] = 1

    if task is LinkTask.DIRECTION:
        return EdgeExamples(pairs, labels)

    if reverse_negative_fraction is not None:
        positive = np.flatnonzero(labels == 1)
        reverse = np.flatnonzero(labels == 0)
        target = len(positive)
        while round(target * reverse_negative_fraction) > len(reverse):
            target -= 1
        num_reverse = round(target * reverse_negative_fraction)
        selected = np.concatenate(
            (
                rng.choice(positive, target, replace=False),
                rng.choice(reverse, num_reverse, replace=False),
                len(pairs)
                + rng.choice(len(non_edges), target - num_reverse, replace=False),
            )
        )
        all_labels = np.concatenate(
            (labels, np.zeros(len(non_edges), dtype=np.float32))
        )
        return EdgeExamples(
            np.vstack((pairs, non_edges))[selected], all_labels[selected]
        )

    pairs = np.vstack((pairs, non_edges))
    labels = np.concatenate((labels, np.zeros(len(non_edges), dtype=np.float32)))
    negative = np.flatnonzero(labels == 0)
    num_to_drop = len(negative) - int(labels.sum())
    if num_to_drop < 0:
        raise RuntimeError("MagNet existence candidates contain too few negatives.")
    keep = np.ones(len(labels), dtype=bool)
    if num_to_drop:
        rng = np.random.default_rng(_BALANCE_SEED)
        keep[rng.choice(negative, size=num_to_drop, replace=False)] = False
    return EdgeExamples(pairs[keep], labels[keep])


def _sample_global_non_edges(
    graph: nx.Graph,
    num_samples: int,
    rng: np.random.RandomState,
    *,
    excluded: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Port StellarGraph's global node-pair sampler with anti-leak exclusions."""
    edges = set(graph.edges())
    edges.update((v, u) for u, v in tuple(edges))
    sampled = set()
    start_nodes = list(graph.nodes())
    end_nodes = list(graph.nodes())
    output = []
    for _ in range(int(np.ceil(num_samples / len(start_nodes))) + 5):
        rng.shuffle(start_nodes)
        rng.shuffle(end_nodes)
        for u, v in zip(start_nodes, end_nodes, strict=True):
            if u == v or (u, v) in edges or (u, v) in excluded or (u, v) in sampled:
                continue
            output.append((u, v))
            sampled.update(((u, v), (v, u)))
            if len(output) == num_samples:
                return output
    raise RuntimeError(f"Could only sample {len(output)} of {num_samples} non-edges.")


def _observed_edges(
    observed: nx.Graph,
    directed_edges: set[tuple[int, int]],
) -> np.ndarray:
    """Restore every original orientation of each observed underlying pair."""
    result = []
    for u_raw, v_raw in observed.edges():
        u, v = int(u_raw), int(v_raw)
        if (u, v) in directed_edges:
            result.append((u, v))
        if (v, u) in directed_edges:
            result.append((v, u))
    return np.asarray(result, dtype=np.int64).T


__all__ = [
    "EdgeExamples",
    "LinkPredictionSplit",
    "LinkTask",
    "SPLIT_PROTOCOL",
    "generate_splits",
    "split_protocol_metadata",
]
