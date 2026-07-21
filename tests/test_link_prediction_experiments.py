from types import SimpleNamespace

from finsler_mds.link_prediction import experiments
from finsler_mds.link_prediction.splits import LinkTask


def _parameters(dimension):
    return {
        "dimension": dimension,
        "alpha": 0.5,
        "radius": 1.0,
        "temperature": 1.0,
        "learning_rate": 0.01,
        "positive_weight": 1.0,
        "reverse_negative_fraction": 0.5,
    }


def test_reranking_ignores_split_zero_and_selects_other_split_mean(
    monkeypatch,
):
    trials = [
        SimpleNamespace(number=0, value=0.99, params=_parameters(5)),
        SimpleNamespace(number=1, value=0.90, params=_parameters(10)),
        SimpleNamespace(number=2, value=0.80, params=_parameters(20)),
    ]
    splits = [
        SimpleNamespace(task=LinkTask.EXISTENCE, seed=seed) for seed in range(3)
    ]

    def fake_fit(graph, split, hyperparameters, *args, **kwargs):
        score = 0.5 if hyperparameters.dimension == 5 else 0.8
        return SimpleNamespace(validation_auc=score + 0.01 * split.seed)

    monkeypatch.setattr(experiments, "_fit_hyperparameters", fake_fit)
    result = experiments.rerank_hyperparameters(
        SimpleNamespace(num_nodes=3),
        splits,
        SimpleNamespace(trials=trials),
        metric_name="randers",
        top_candidates=2,
        num_reranking_splits=2,
    )

    assert result.split_indices == (1, 2)
    assert [candidate.trial_number for candidate in result.candidates] == [0, 1]
    assert result.selected_trial_number == 1
    assert result.hyperparameters.dimension == 10
