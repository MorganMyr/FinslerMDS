from __future__ import annotations

import networkx as nx
import numpy as np

from finsler_mds.link_prediction.data import DirectedGraphData
from finsler_mds.link_prediction.split_cache import load_splits, save_splits
from finsler_mds.link_prediction.splits import (
    LinkTask,
    _label_noisy,
    _observed_edges,
    generate_splits,
)


def _synthetic_graph():
    edges = []
    for node in range(60):
        edges.append((node, (node + 1) % 60))
        edges.append((node, (node + 3) % 60))
    return DirectedGraphData(
        name="synthetic",
        num_nodes=60,
        edge_index=np.asarray(edges, dtype=np.int64).T,
    )


def test_graph_statistics_distinguish_arcs_and_pairs():
    graph = DirectedGraphData(
        name="tiny",
        num_nodes=4,
        edge_index=np.asarray([(0, 1), (1, 0), (1, 2), (3, 3)]).T,
    )
    stats = graph.statistics()
    assert stats.num_directed_edges == 4
    assert stats.num_self_loops == 1
    assert stats.num_non_loop_directed_edges == 3
    assert stats.num_reciprocal_pairs == 1
    assert stats.num_unordered_pairs == 2
    assert graph.without_self_loops().num_edges == 3


def test_splits_are_deterministic_balanced_and_connected():
    graph = _synthetic_graph()
    first = generate_splits(graph, LinkTask.DIRECTION, num_splits=2)
    second = generate_splits(graph, LinkTask.DIRECTION, num_splits=2)

    for split_a, split_b in zip(first, second, strict=True):
        np.testing.assert_array_equal(split_a.train.pairs, split_b.train.pairs)
        np.testing.assert_array_equal(split_a.validation.pairs, split_b.validation.pairs)
        np.testing.assert_array_equal(split_a.test.pairs, split_b.test.pairs)
        assert abs(split_a.train.num_positive - split_a.train.num_negative) <= 1
        assert abs(split_a.validation.num_positive - split_a.validation.num_negative) <= 1
        assert abs(split_a.test.num_positive - split_a.test.num_negative) <= 1

        observed = nx.Graph()
        observed.add_nodes_from(range(graph.num_nodes))
        observed.add_edges_from(split_a.observed_edge_index.T.tolist())
        assert nx.is_connected(observed)


def test_existence_splits_are_balanced_after_magnet_downsampling():
    split = generate_splits(_synthetic_graph(), LinkTask.EXISTENCE, num_splits=1)[0]
    for examples in (split.train, split.validation, split.test):
        assert examples.num_positive == examples.num_negative


def test_existence_evaluation_negative_composition_is_configurable():
    graph = _synthetic_graph()
    directed_edges = set(map(tuple, graph.edge_index.T.tolist()))
    legacy = generate_splits(graph, LinkTask.EXISTENCE, num_splits=1)[0]

    for fraction in (0.0, 1.0):
        split = generate_splits(
            graph,
            LinkTask.EXISTENCE,
            num_splits=1,
            evaluation_reverse_negative_fraction=fraction,
        )[0]
        np.testing.assert_array_equal(split.train.pairs, legacy.train.pairs)
        np.testing.assert_array_equal(split.train.labels, legacy.train.labels)
        for examples in (split.validation, split.test):
            negatives = examples.pairs[examples.labels == 0]
            reversed_fraction = np.mean(
                [(int(v), int(u)) in directed_edges for u, v in negatives]
            )
            assert reversed_fraction == fraction


def test_noisy_direction_randomizes_reciprocal_targets_reproducibly():
    positive = np.asarray([(i, i + 100) for i in range(100)])
    directed_edges = set(map(tuple, positive.tolist()))
    directed_edges.update((v, u) for u, v in positive)
    direction = _label_noisy(
        positive,
        np.empty((0, 2), dtype=np.int64),
        directed_edges,
        LinkTask.DIRECTION,
        np.random.RandomState(0),
    )
    repeated = _label_noisy(
        positive,
        np.empty((0, 2), dtype=np.int64),
        directed_edges,
        LinkTask.DIRECTION,
        np.random.RandomState(0),
    )
    np.testing.assert_array_equal(direction.pairs, repeated.pairs)
    ascending_matches = (
        direction.pairs[:, 0] < direction.pairs[:, 1]
    ) == direction.labels
    assert 0.35 < ascending_matches.mean() < 0.65


def test_observed_graph_retains_both_reciprocal_arcs():
    observed = nx.Graph([(0, 1), (1, 2)])
    directed_edges = {(0, 1), (1, 0), (1, 2)}
    edge_index = _observed_edges(observed, directed_edges)
    assert edge_index.shape == (2, 3)
    assert set(map(tuple, edge_index.T.tolist())) == {(0, 1), (1, 0), (1, 2)}


def test_training_samples_99_percent_of_observed_pairs():
    split = generate_splits(_synthetic_graph(), LinkTask.DIRECTION, num_splits=1)[0]
    observed_pairs = {
        tuple(sorted(map(int, pair))) for pair in split.observed_edge_index.T
    }
    assert len(split.train.pairs) == int(0.99 * len(observed_pairs))


def test_partitions_share_no_unordered_pair():
    graph = _synthetic_graph()
    for task in LinkTask:
        split = generate_splits(graph, task, num_splits=1)[0]
        keys = []
        for examples in (split.train, split.validation, split.test):
            keys.append({tuple(sorted(map(int, pair))) for pair in examples.pairs})
        assert keys[0].isdisjoint(keys[1])
        assert keys[0].isdisjoint(keys[2])
        assert keys[1].isdisjoint(keys[2])


def test_observed_graph_excludes_validation_and_test_pairs():
    graph = _synthetic_graph()
    split = generate_splits(graph, LinkTask.DIRECTION, num_splits=1)[0]
    observed = {
        tuple(sorted(map(int, pair)))
        for pair in split.observed_edge_index.T
    }
    held_out = {
        tuple(sorted(map(int, pair)))
        for examples in (split.validation, split.test)
        for pair in examples.pairs
    }
    assert observed.isdisjoint(held_out)
    assert split.observed_edge_index.shape[0] == 2


def test_split_cache_round_trip(tmp_path):
    graph = _synthetic_graph()
    expected = generate_splits(graph, LinkTask.EXISTENCE, num_splits=2)
    path = tmp_path / "splits.npz"
    save_splits(path, graph, expected)
    actual = load_splits(path, graph, task=LinkTask.EXISTENCE)

    assert [split.seed for split in actual] == [0, 1]
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before.train.pairs, after.train.pairs)
        np.testing.assert_array_equal(before.train.labels, after.train.labels)
        np.testing.assert_array_equal(
            before.observed_edge_index,
            after.observed_edge_index,
        )
        np.testing.assert_array_equal(before.test.pairs, after.test.pairs)
        np.testing.assert_array_equal(before.test.labels, after.test.labels)
