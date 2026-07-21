"""Per-split optimization and evaluation of Finsler link predictors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np

from finsler_mds.metrics import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)

from .data import DirectedGraphData
from .evaluation import score_name
from .optimization import HYPERPARAMETER_SELECTION_PROTOCOL, OptunaConfig, optimize_study
from .splits import LinkPredictionSplit
from .training import EMBEDDING_TRAINING_PROTOCOL, TrainingConfig, fit_link_predictor

METRIC_NAMES = ("randers", "matsumoto", "convexified_matsumoto")
DEFAULT_EMBEDDING_DIMENSION = 50
DEFAULT_EMBEDDING_DIMENSIONS = (5, 10, 20, 50)


@dataclass(frozen=True)
class ModelHyperparameters:
    dimension: int
    alpha: float
    radius: float
    temperature: float
    learning_rate: float
    positive_weight: float
    reverse_negative_fraction: float

    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if min(
            self.radius,
            self.temperature,
            self.learning_rate,
            self.positive_weight,
        ) <= 0:
            raise ValueError("Positive hyperparameters must be positive.")
        if not 0 <= self.reverse_negative_fraction <= 1:
            raise ValueError("reverse_negative_fraction must be in [0, 1].")


@dataclass(frozen=True)
class OptunaSearchSpace:
    dimensions: tuple[int, ...] = DEFAULT_EMBEDDING_DIMENSIONS
    alpha_min: float = 0.0
    alpha_max: float = 0.999
    radius_min: float = 0.25
    radius_max: float = 8.0
    temperature_min: float = 0.05
    temperature_max: float = 4.0
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 5e-1
    positive_weight_min: float = 0.25
    positive_weight_max: float = 4.0

    def __post_init__(self):
        dimensions = tuple(self.dimensions)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError("dimensions must contain positive integers.")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("dimensions must not contain duplicates.")
        object.__setattr__(self, "dimensions", dimensions)
        bounds = (
            ("alpha", self.alpha_min, self.alpha_max, True),
            ("radius", self.radius_min, self.radius_max, False),
            ("temperature", self.temperature_min, self.temperature_max, False),
            ("learning_rate", self.learning_rate_min, self.learning_rate_max, False),
            ("positive_weight", self.positive_weight_min, self.positive_weight_max, False),
        )
        for name, lower, upper, allow_zero in bounds:
            if upper <= lower or lower < 0 or (not allow_zero and lower == 0):
                raise ValueError(f"Invalid {name} interval [{lower}, {upper}].")

    def sample(self, trial) -> ModelHyperparameters:
        return ModelHyperparameters(
            dimension=trial.suggest_categorical("dimension", self.dimensions),
            alpha=trial.suggest_float("alpha", self.alpha_min, self.alpha_max),
            radius=trial.suggest_float("radius", self.radius_min, self.radius_max, log=True),
            temperature=trial.suggest_float(
                "temperature", self.temperature_min, self.temperature_max, log=True
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", self.learning_rate_min, self.learning_rate_max, log=True
            ),
            positive_weight=trial.suggest_float(
                "positive_weight", self.positive_weight_min, self.positive_weight_max, log=True
            ),
            reverse_negative_fraction=trial.suggest_float(
                "reverse_negative_fraction", 0.0, 1.0
            ),
        )


@dataclass(frozen=True)
class SplitRunResult:
    split_index: int
    split_seed: int
    hyperparameters: ModelHyperparameters
    best_trial_number: int | None
    best_epoch: int
    validation_roc_auc: float
    test_roc_auc: float


@dataclass(frozen=True)
class ExperimentSummary:
    dataset: str
    graph_fingerprint: str
    task: str
    metric: str
    training_config: TrainingConfig
    runs: tuple[SplitRunResult, ...]

    @property
    def mean_test_roc_auc(self) -> float:
        return float(np.mean([run.test_roc_auc for run in self.runs]))

    @property
    def std_test_roc_auc(self) -> float:
        return float(np.std([run.test_roc_auc for run in self.runs], ddof=0))

    def as_dict(self):
        selection = (
            HYPERPARAMETER_SELECTION_PROTOCOL
            if any(run.best_trial_number is not None for run in self.runs)
            else "fixed"
        )
        return {
            "dataset": self.dataset,
            "graph_fingerprint": self.graph_fingerprint,
            "task": self.task,
            "evaluation_score": score_name(self.task),
            "metric": self.metric,
            "hyperparameter_selection_protocol": selection,
            "embedding_training_protocol": EMBEDDING_TRAINING_PROTOCOL,
            "training_config": asdict(self.training_config),
            "mean_test_roc_auc": self.mean_test_roc_auc,
            "std_test_roc_auc": self.std_test_roc_auc,
            "runs": [asdict(run) for run in self.runs],
        }


def make_metric(name: str, alpha: float):
    metrics = {
        "randers": RandersMetric,
        "matsumoto": MatsumotoMetric,
        "convexified_matsumoto": ConvexifiedMatsumotoMetric,
    }
    try:
        return metrics[name.lower()](alpha=alpha)
    except KeyError as exc:
        raise ValueError(f"Unknown metric {name!r}; choose from {METRIC_NAMES}.") from exc


def run_experiment(
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    *,
    metric_name: str,
    training_config: TrainingConfig | None = None,
    search_space: OptunaSearchSpace | None = None,
    optuna_config: OptunaConfig | None = None,
    fixed_hyperparameters: ModelHyperparameters | None = None,
    progress: Callable[[SplitRunResult], None] | None = None,
) -> ExperimentSummary:
    """Select and test one independent model on every split."""
    _validate_splits(splits)
    if (fixed_hyperparameters is None) == (search_space is None):
        raise ValueError("Provide exactly one of search_space or fixed_hyperparameters.")
    training_config = training_config or TrainingConfig()
    optuna_config = optuna_config or OptunaConfig()
    runs = []

    for split_index, split in enumerate(splits):
        study = None
        best_trial_number = None
        if fixed_hyperparameters is None:
            def objective(trial):
                hyperparameters = search_space.sample(trial)
                try:
                    return _fit_hyperparameters(
                        graph,
                        split,
                        hyperparameters,
                        metric_name,
                        training_config,
                        evaluate_test=False,
                    ).validation_auc
                except FloatingPointError as exc:
                    import optuna

                    raise optuna.TrialPruned(str(exc)) from exc

            study = optimize_study(
                objective,
                optuna_config,
                study_name=f"split_{split_index}",
                seed_offset=split.seed,
            )
            hyperparameters = ModelHyperparameters(**study.best_params)
            best_trial_number = study.best_trial.number
        else:
            hyperparameters = fixed_hyperparameters

        result = _fit_hyperparameters(
            graph,
            split,
            hyperparameters,
            metric_name,
            training_config,
            evaluate_test=True,
        )
        if result.test_auc is None:
            raise RuntimeError("Final split evaluation did not produce a test AUC.")
        if study is not None:
            study.set_user_attr("validation_roc_auc", result.validation_auc)
            study.set_user_attr("test_roc_auc", result.test_auc)
            study.set_user_attr("best_epoch", result.best_epoch)
        run = SplitRunResult(
            split_index=split_index,
            split_seed=split.seed,
            hyperparameters=hyperparameters,
            best_trial_number=best_trial_number,
            best_epoch=result.best_epoch,
            validation_roc_auc=result.validation_auc,
            test_roc_auc=result.test_auc,
        )
        runs.append(run)
        if progress is not None:
            progress(run)

    return ExperimentSummary(
        dataset=graph.name,
        graph_fingerprint=graph.fingerprint,
        task=splits[0].task.value,
        metric=metric_name,
        training_config=training_config,
        runs=tuple(runs),
    )


def _fit_hyperparameters(
    graph,
    split,
    hyperparameters,
    metric_name,
    training_config,
    *,
    evaluate_test,
):
    return fit_link_predictor(
        graph.num_nodes,
        split,
        make_metric(metric_name, hyperparameters.alpha),
        dimension=hyperparameters.dimension,
        radius=hyperparameters.radius,
        temperature=hyperparameters.temperature,
        positive_weight=hyperparameters.positive_weight,
        reverse_negative_fraction=hyperparameters.reverse_negative_fraction,
        config=replace(
            training_config,
            learning_rate=hyperparameters.learning_rate,
            seed=training_config.seed + split.seed,
        ),
        evaluate_test=evaluate_test,
    )


def _validate_splits(splits):
    if not splits:
        raise ValueError("splits must not be empty.")
    if any(split.task != splits[0].task for split in splits):
        raise ValueError("All splits in an experiment must use the same task.")


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "ExperimentSummary",
    "HYPERPARAMETER_SELECTION_PROTOCOL",
    "METRIC_NAMES",
    "ModelHyperparameters",
    "OptunaSearchSpace",
    "SplitRunResult",
    "make_metric",
    "run_experiment",
]
