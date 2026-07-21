"""Reusable Optuna and multi-split experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import numpy as np

from finsler_mds.metrics import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
)

from .data import DirectedGraphData
from .evaluation import score_name
from .runs import save_json
from .splits import LinkPredictionSplit
from .training import (
    EMBEDDING_TRAINING_PROTOCOL,
    TrainingConfig,
    fit_link_predictor,
)

METRIC_NAMES = ("randers", "matsumoto", "convexified_matsumoto")
DEFAULT_EMBEDDING_DIMENSION = 50
DEFAULT_EMBEDDING_DIMENSIONS = (5, 10, 20, 50)
HYPERPARAMETER_SELECTION_PROTOCOL = "split0_topn_rerank_other_splits_v1"


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
            raise ValueError(
                "radius, temperature, learning_rate, and positive_weight "
                "must be positive."
            )
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
            (
                "positive_weight",
                self.positive_weight_min,
                self.positive_weight_max,
                False,
            ),
        )
        for name, lower, upper, allow_zero in bounds:
            if upper <= lower or lower < 0 or (not allow_zero and lower == 0):
                raise ValueError(f"Invalid {name} search interval [{lower}, {upper}].")

    def sample(self, trial) -> ModelHyperparameters:
        return ModelHyperparameters(
            dimension=trial.suggest_categorical("dimension", self.dimensions),
            alpha=trial.suggest_float("alpha", self.alpha_min, self.alpha_max),
            radius=trial.suggest_float(
                "radius", self.radius_min, self.radius_max, log=True
            ),
            temperature=trial.suggest_float(
                "temperature", self.temperature_min, self.temperature_max, log=True
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                self.learning_rate_min,
                self.learning_rate_max,
                log=True,
            ),
            positive_weight=trial.suggest_float(
                "positive_weight",
                self.positive_weight_min,
                self.positive_weight_max,
                log=True,
            ),
            reverse_negative_fraction=trial.suggest_float(
                "reverse_negative_fraction", 0.0, 1.0
            ),
        )


@dataclass(frozen=True)
class OptunaConfig:
    num_trials: int = 50
    seed: int = 0
    storage: str | None = None
    timeout: float | None = None

    def __post_init__(self):
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive.")


@dataclass(frozen=True)
class CandidateRerankingResult:
    trial_number: int
    hyperparameters: ModelHyperparameters
    tuning_validation_roc_auc: float
    reranking_validation_roc_auc: tuple[float, ...]

    @property
    def mean_reranking_validation_roc_auc(self) -> float:
        return float(np.mean(self.reranking_validation_roc_auc))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "hyperparameters": asdict(self.hyperparameters),
            "tuning_validation_roc_auc": self.tuning_validation_roc_auc,
            "reranking_validation_roc_auc": self.reranking_validation_roc_auc,
            "mean_reranking_validation_roc_auc": (
                self.mean_reranking_validation_roc_auc
            ),
        }


@dataclass(frozen=True)
class RerankingResult:
    split_indices: tuple[int, ...]
    candidates: tuple[CandidateRerankingResult, ...]
    selected_trial_number: int

    @property
    def hyperparameters(self) -> ModelHyperparameters:
        return next(
            candidate.hyperparameters
            for candidate in self.candidates
            if candidate.trial_number == self.selected_trial_number
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": HYPERPARAMETER_SELECTION_PROTOCOL,
            "tuning_split_index": 0,
            "reranking_split_indices": self.split_indices,
            "selection_score": "mean_validation_roc_auc_excluding_tuning_split",
            "selected_trial_number": self.selected_trial_number,
            "selected_hyperparameters": asdict(self.hyperparameters),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class SplitRunResult:
    split_index: int
    split_seed: int
    best_epoch: int
    validation_roc_auc: float
    test_roc_auc: float


@dataclass(frozen=True)
class ExperimentSummary:
    dataset: str
    graph_fingerprint: str
    task: str
    metric: str
    hyperparameters: ModelHyperparameters
    training_config: TrainingConfig
    runs: tuple[SplitRunResult, ...]

    @property
    def mean_test_roc_auc(self) -> float:
        return float(np.mean([run.test_roc_auc for run in self.runs]))

    @property
    def std_test_roc_auc(self) -> float:
        return float(np.std([run.test_roc_auc for run in self.runs], ddof=0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "graph_fingerprint": self.graph_fingerprint,
            "task": self.task,
            "evaluation_score": score_name(self.task),
            "metric": self.metric,
            "dimension": self.hyperparameters.dimension,
            "embedding_training_protocol": EMBEDDING_TRAINING_PROTOCOL,
            "hyperparameters": asdict(self.hyperparameters),
            "training_config": asdict(self.training_config),
            "mean_test_roc_auc": self.mean_test_roc_auc,
            "std_test_roc_auc": self.std_test_roc_auc,
            "runs": [asdict(run) for run in self.runs],
        }


def make_metric(name: str, alpha: float):
    name = name.lower()
    if name == "randers":
        return RandersMetric(alpha=alpha)
    if name == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    if name == "convexified_matsumoto":
        return ConvexifiedMatsumotoMetric(alpha=alpha)
    raise ValueError(f"Unknown metric {name!r}; choose from {METRIC_NAMES}.")


def tune_hyperparameters(
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    *,
    metric_name: str,
    training_config: TrainingConfig | None = None,
    search_space: OptunaSearchSpace | None = None,
    optuna_config: OptunaConfig | None = None,
):
    """Run an Optuna study on the validation set of split 0."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for tune_hyperparameters; install the "
            "link-prediction dependencies."
        ) from exc

    training_config = (
        TrainingConfig() if training_config is None else training_config
    )
    search_space = OptunaSearchSpace() if search_space is None else search_space
    optuna_config = OptunaConfig() if optuna_config is None else optuna_config
    _validate_splits(splits)
    tuning_split = splits[0]

    def objective(trial):
        params = search_space.sample(trial)
        try:
            result = _fit_hyperparameters(
                graph,
                tuning_split,
                params,
                metric_name,
                training_config,
                evaluate_test=False,
            )
        except FloatingPointError as exc:
            raise optuna.TrialPruned(str(exc)) from exc
        return result.validation_auc

    sampler = optuna.samplers.TPESampler(seed=optuna_config.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=optuna_config.storage,
    )
    study.optimize(
        objective,
        n_trials=optuna_config.num_trials,
        timeout=optuna_config.timeout,
    )
    return study


def rerank_hyperparameters(
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    study,
    *,
    metric_name: str,
    top_candidates: int = 8,
    num_reranking_splits: int = 3,
    training_config: TrainingConfig | None = None,
    progress: Callable[[int, int, CandidateRerankingResult], None] | None = None,
) -> RerankingResult:
    """Rerank the best split-0 trials on other validation partitions."""
    _validate_splits(splits)
    if top_candidates <= 0 or num_reranking_splits <= 0:
        raise ValueError("Reranking counts must be positive.")
    if num_reranking_splits >= len(splits):
        raise ValueError("Reranking requires enough splits after split 0.")
    trials = sorted(
        (trial for trial in study.trials if trial.value is not None),
        key=lambda trial: trial.value,
        reverse=True,
    )[:top_candidates]
    if not trials:
        raise RuntimeError("The Optuna study has no completed trial to rerank.")

    training_config = TrainingConfig() if training_config is None else training_config
    split_indices = tuple(range(1, num_reranking_splits + 1))
    candidates = []
    for trial in trials:
        hyperparameters = ModelHyperparameters(**trial.params)
        validation_auc = tuple(
            _fit_hyperparameters(
                graph,
                splits[index],
                hyperparameters,
                metric_name,
                training_config,
                evaluate_test=False,
            ).validation_auc
            for index in split_indices
        )
        candidate = CandidateRerankingResult(
            trial_number=trial.number,
            hyperparameters=hyperparameters,
            tuning_validation_roc_auc=float(trial.value),
            reranking_validation_roc_auc=validation_auc,
        )
        candidates.append(candidate)
        if progress is not None:
            progress(len(candidates), len(trials), candidate)
    selected = max(
        candidates,
        key=lambda candidate: candidate.mean_reranking_validation_roc_auc,
    )
    return RerankingResult(
        split_indices=split_indices,
        candidates=tuple(candidates),
        selected_trial_number=selected.trial_number,
    )


def evaluate_hyperparameters(
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    hyperparameters: ModelHyperparameters,
    *,
    metric_name: str,
    training_config: TrainingConfig | None = None,
) -> ExperimentSummary:
    _validate_splits(splits)
    training_config = TrainingConfig() if training_config is None else training_config
    training_config = replace(
        training_config,
        learning_rate=hyperparameters.learning_rate,
    )
    runs = []
    for split_index, split in enumerate(splits):
        result = _fit_hyperparameters(
            graph,
            split,
            hyperparameters,
            metric_name,
            training_config,
            evaluate_test=True,
        )
        if result.test_auc is None:
            raise RuntimeError("Final split evaluation did not produce test metrics.")
        runs.append(
            SplitRunResult(
                split_index=split_index,
                split_seed=split.seed,
                best_epoch=result.best_epoch,
                validation_roc_auc=result.validation_auc,
                test_roc_auc=result.test_auc,
            )
        )
    return ExperimentSummary(
        dataset=graph.name,
        graph_fingerprint=graph.fingerprint,
        task=splits[0].task.value,
        metric=metric_name,
        hyperparameters=hyperparameters,
        training_config=training_config,
        runs=tuple(runs),
    )


def _fit_hyperparameters(
    graph: DirectedGraphData,
    split: LinkPredictionSplit,
    hyperparameters: ModelHyperparameters,
    metric_name: str,
    training_config: TrainingConfig,
    *,
    evaluate_test: bool,
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


def _validate_splits(splits: list[LinkPredictionSplit]):
    if not splits:
        raise ValueError("splits must not be empty.")
    if any(split.task != splits[0].task for split in splits):
        raise ValueError("All splits in an experiment must use the same task.")


def save_experiment_summary(path, summary: ExperimentSummary):
    save_json(path, summary.as_dict())


__all__ = [
    "CandidateRerankingResult",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "ExperimentSummary",
    "HYPERPARAMETER_SELECTION_PROTOCOL",
    "METRIC_NAMES",
    "ModelHyperparameters",
    "OptunaConfig",
    "OptunaSearchSpace",
    "RerankingResult",
    "SplitRunResult",
    "evaluate_hyperparameters",
    "make_metric",
    "rerank_hyperparameters",
    "save_experiment_summary",
    "tune_hyperparameters",
]
