from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric_signed_directed")

from finsler_mds.link_prediction.baselines import (  # noqa: E402
    BaselineTrainingConfig,
    MagNetHyperparameters,
)
from finsler_mds.link_prediction.baselines.magnet import (  # noqa: E402
    _in_out_degree,
    fit_magnet,
)
from finsler_mds.link_prediction.data import DirectedGraphData  # noqa: E402
from finsler_mds.link_prediction.splits import LinkTask, generate_splits  # noqa: E402


def test_degree_features_use_only_observed_arcs():
    edges = torch.tensor([[0, 0, 2], [1, 2, 1]])
    features = _in_out_degree(edges, 4, torch)
    torch.testing.assert_close(
        features,
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),
    )


def test_magnet_training_smoke_on_common_split():
    edges = []
    for node in range(24):
        edges.extend((node, (node + offset) % 24) for offset in (1, 3, 5))
    graph = DirectedGraphData(
        name="magnet_smoke",
        num_nodes=24,
        edge_index=np.asarray(edges, dtype=np.int64).T,
    )
    split = generate_splits(graph, LinkTask.DIRECTION, num_splits=1)[0]
    result = fit_magnet(
        graph,
        split,
        MagNetHyperparameters(hidden_channels=4, dropout=0.0),
        BaselineTrainingConfig(max_epochs=3, patience=2, device="cpu", seed=2),
    )
    assert 1 <= result.best_epoch <= 3
    assert np.isfinite(result.validation_auc)
    assert np.isfinite(result.test_auc)
