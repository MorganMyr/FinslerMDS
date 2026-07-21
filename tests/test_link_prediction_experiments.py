from types import SimpleNamespace

from finsler_mds.link_prediction import experiments
from finsler_mds.link_prediction.baselines import base
from finsler_mds.link_prediction.optimization import OptunaConfig
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


def test_each_split_is_tuned_and_tested_independently(monkeypatch):
    calls = []
    splits = [SimpleNamespace(task=LinkTask.EXISTENCE, seed=seed) for seed in range(3)]

    class SearchSpace:
        def sample(self, trial):
            return experiments.ModelHyperparameters(**_parameters(5 + trial.split_index))

    def fake_optimize(objective, config, *, study_name, seed_offset):
        split_index = int(study_name.removeprefix("split_"))
        trial = SimpleNamespace(split_index=split_index)
        objective(trial)
        return SimpleNamespace(
            best_params=_parameters(5 + split_index),
            best_trial=SimpleNamespace(number=10 + split_index),
            set_user_attr=lambda *args: None,
        )

    def fake_fit(graph, split, hyperparameters, *args, evaluate_test, **kwargs):
        calls.append((split.seed, hyperparameters.dimension, evaluate_test))
        return SimpleNamespace(
            best_epoch=7,
            validation_auc=0.8 + split.seed / 100,
            test_auc=0.7 + split.seed / 100 if evaluate_test else None,
        )

    monkeypatch.setattr(experiments, "optimize_study", fake_optimize)
    monkeypatch.setattr(experiments, "_fit_hyperparameters", fake_fit)
    summary = experiments.run_experiment(
        SimpleNamespace(name="toy", fingerprint="abc", num_nodes=3),
        splits,
        metric_name="randers",
        search_space=SearchSpace(),
        optuna_config=OptunaConfig(num_trials=1),
    )

    assert calls == [
        (0, 5, False), (0, 5, True),
        (1, 6, False), (1, 6, True),
        (2, 7, False), (2, 7, True),
    ]
    assert [run.hyperparameters.dimension for run in summary.runs] == [5, 6, 7]
    assert [run.best_trial_number for run in summary.runs] == [10, 11, 12]


def test_baseline_uses_the_same_independent_split_protocol(monkeypatch):
    calls = []
    splits = [SimpleNamespace(task=LinkTask.DIRECTION, seed=seed) for seed in range(2)]

    class Baseline:
        name = "toy"

        def suggest_hyperparameters(self, trial):
            return {"value": trial.split_index}

        def fit(self, graph, split, hyperparameters, config, *, evaluate_test=True):
            calls.append((split.seed, hyperparameters["value"], evaluate_test))
            return base.BaselineFitResult(1, 0.8, 0.7 if evaluate_test else None)

    def fake_optimize(objective, config, *, study_name, seed_offset):
        split_index = int(study_name.removeprefix("split_"))
        trial = SimpleNamespace(
            split_index=split_index,
            set_user_attr=lambda *args: None,
        )
        objective(trial)
        return SimpleNamespace(
            best_trial=SimpleNamespace(
                number=split_index,
                user_attrs={"hyperparameters": {"value": split_index}},
            ),
            set_user_attr=lambda *args: None,
        )

    monkeypatch.setattr(base, "optimize_study", fake_optimize)
    summary = base.run_baseline(
        Baseline(),
        SimpleNamespace(name="toy", fingerprint="abc"),
        splits,
        base.BaselineTrainingConfig(max_epochs=1, patience=1),
        optuna_config=OptunaConfig(num_trials=1),
    )

    assert calls == [(0, 0, False), (0, 0, True), (1, 1, False), (1, 1, True)]
    assert [run.hyperparameters for run in summary.runs] == [{"value": 0}, {"value": 1}]
