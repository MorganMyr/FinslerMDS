from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from finsler_mds.metrics import (  # noqa: E402
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)
from finsler_mds.link_prediction.decoder import FermiDiracDecoder  # noqa: E402
from finsler_mds.link_prediction.evaluation import link_scores  # noqa: E402
from finsler_mds.link_prediction.initialization import (  # noqa: E402
    spectral_initialization,
)
from finsler_mds.link_prediction.model import FinslerLinkPredictor  # noqa: E402
from finsler_mds.link_prediction.torch_metrics import torch_metric_length  # noqa: E402
from finsler_mds.link_prediction.data import DirectedGraphData  # noqa: E402
from finsler_mds.link_prediction.splits import (  # noqa: E402
    LinkTask,
    generate_splits,
)
from finsler_mds.link_prediction.training import (  # noqa: E402
    TrainingConfig,
    _negative_sampling_plan,
    _sample_negatives,
    fit_link_predictor,
)


@pytest.mark.parametrize(
    "metric",
    [
        RandersMetric(alpha=0.4),
        MatsumotoMetric(alpha=0.4),
        ConvexifiedMatsumotoMetric(alpha=0.8),
    ],
)
def test_torch_lengths_and_gradients_match_numpy(metric):
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(32, 5))
    tensor = torch.tensor(vectors, dtype=torch.float64, requires_grad=True)
    lengths = torch_metric_length(tensor, metric)

    np.testing.assert_allclose(lengths.detach().numpy(), metric.length(vectors), rtol=1e-10)
    lengths.sum().backward()
    np.testing.assert_allclose(tensor.grad.numpy(), metric.grad_u(vectors), rtol=1e-9, atol=1e-9)


def test_fermi_dirac_decoder_preserves_randers_directionality():
    decoder = FermiDiracDecoder(
        RandersMetric(alpha=0.5), radius=2.0, temperature=1.0
    )
    upward = torch.tensor([[0.0, 1.0]])
    downward = -upward
    assert decoder(upward).item() < decoder(downward).item()


def test_direction_score_is_the_logit_difference():
    class Model:
        def logits(self, pairs):
            return pairs[:, 0].float() - 2 * pairs[:, 1].float()

    model = Model()
    pairs = torch.tensor([[1, 3], [4, 2]])
    forward = model.logits(pairs)
    np.testing.assert_allclose(
        link_scores(model, pairs, LinkTask.EXISTENCE), forward.sigmoid()
    )
    np.testing.assert_allclose(
        link_scores(model, pairs, LinkTask.DIRECTION),
        forward - model.logits(pairs.flip(1)),
    )


def test_matsumoto_forbidden_surrogate_is_rejected():
    metric = MatsumotoMetric(alpha=1.2, forbidden_grad_norm=4.8)
    with pytest.raises(ValueError, match="surrogate"):
        torch_metric_length(torch.tensor([[0.0, 1.0]]), metric)


def test_random_initializations_share_one_draw_and_radius_only_rescales_it():
    def initialize(name):
        torch.manual_seed(7)
        model = FinslerLinkPredictor(
            30,
            5,
            RandersMetric(alpha=0.4),
            radius=3.0,
            temperature=1.0,
            initialization=name,
        )
        return model.embedding.weight.detach().clone(), torch.rand(4)

    current, random_after_current = initialize("current")
    normal, random_after_normal = initialize("normal")
    radius, random_after_radius = initialize("radius")

    torch.testing.assert_close(normal, current * np.sqrt(5))
    scale = (radius * current).sum() / current.square().sum()
    torch.testing.assert_close(radius, current * scale)
    mean_pair_squared = 2 * radius.square().sum() / (len(radius) - 1)
    torch.testing.assert_close(mean_pair_squared, torch.tensor(3.0))
    torch.testing.assert_close(random_after_current, random_after_normal)
    torch.testing.assert_close(random_after_current, random_after_radius)


def test_spectral_initialization_is_centered_and_uses_default_scale():
    edges = np.asarray(
        [(node, (node + offset) % 20) for node in range(20) for offset in (1, 3)],
        dtype=np.int64,
    ).T
    coordinates = spectral_initialization(edges, 20, 4, seed=3)

    assert coordinates.shape == (20, 4)
    np.testing.assert_allclose(coordinates.mean(axis=0), 0, atol=1e-6)
    mean_pair_squared = 2 * np.square(coordinates).sum() / 19
    np.testing.assert_allclose(mean_pair_squared, 2, rtol=1e-6)


@pytest.mark.parametrize("initialization", ["current", "spectral"])
def test_end_to_end_training_smoke(initialization):
    edges = []
    for node in range(20):
        edges.extend(
            (node, (node + offset) % 20)
            for offset in (1, 3, 5)
        )
    graph = DirectedGraphData(
        name="training_smoke",
        num_nodes=20,
        edge_index=np.asarray(edges, dtype=np.int64).T,
    )
    split = generate_splits(graph, LinkTask.DIRECTION, num_splits=1)[0]
    result = fit_link_predictor(
        graph.num_nodes,
        split,
        RandersMetric(alpha=0.3),
        dimension=4,
        radius=2.0,
        temperature=1.0,
        config=TrainingConfig(
            max_epochs=5,
            patience=2,
            seed=3,
            device="cpu",
            initialization=initialization,
        ),
    )
    assert result.embedding.shape == (20, 4)
    assert np.isfinite(result.validation_auc)
    assert np.isfinite(result.test_auc)


def test_negative_sampling_never_uses_held_out_pairs():
    edges = np.asarray(
        [(node, (node + offset) % 20) for node in range(20) for offset in (1, 3, 5)],
        dtype=np.int64,
    ).T
    graph = DirectedGraphData("negative_sampling", 20, edges)
    split = generate_splits(graph, LinkTask.EXISTENCE, num_splits=1)[0]
    device = torch.device("cpu")
    inverse, forbidden = _negative_sampling_plan(split, graph.num_nodes, device)
    sampled = _sample_negatives(
        10_000, graph.num_nodes, inverse, forbidden, device, 0.3
    )

    held_out = np.vstack((split.validation.pairs, split.test.pairs))
    held_out_keys = set(
        np.concatenate(
            (
                held_out[:, 0] * graph.num_nodes + held_out[:, 1],
                held_out[:, 1] * graph.num_nodes + held_out[:, 0],
            )
        ).tolist()
    )
    sampled_keys = set(
        (sampled[:, 0] * graph.num_nodes + sampled[:, 1]).tolist()
    )
    assert sampled_keys.isdisjoint(held_out_keys)
    inverse_keys = set(
        (inverse[:, 0] * graph.num_nodes + inverse[:, 1]).tolist()
    )
    assert set(
        (sampled[:3_000, 0] * graph.num_nodes + sampled[:3_000, 1]).tolist()
    ).issubset(inverse_keys)
    assert set(
        (sampled[3_000:, 0] * graph.num_nodes + sampled[3_000:, 1]).tolist()
    ).isdisjoint(inverse_keys)
