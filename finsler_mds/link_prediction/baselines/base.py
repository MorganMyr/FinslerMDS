"""Common per-split runner for external link-prediction baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from ..data import DirectedGraphData
from ..optimization import (
    HYPERPARAMETER_SELECTION_PROTOCOL,
    OptunaConfig,
    optimize_study,
)
from ..splits import LinkPredictionSplit


@dataclass(frozen=True)
class BaselineTrainingConfig:
    max_epochs: int = 3_000
    patience: int = 300
    evaluation_frequency: int = 1
    device: str = "auto"
    seed: int = 0

    def __post_init__(self):
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("max_epochs and patience must be positive.")
        if self.evaluation_frequency <= 0:
            raise ValueError("evaluation_frequency must be positive.")


@dataclass(frozen=True)
class BaselineFitResult:
    best_epoch: int
    validation_auc: float
    test_auc: float | None


class LinkPredictionBaseline(Protocol):
    name: str

    def suggest_hyperparameters(self, trial) -> dict[str, Any]: ...

    def fit(
        self,
        graph: DirectedGraphData,
        split: LinkPredictionSplit,
        hyperparameters: Mapping[str, Any],
        config: BaselineTrainingConfig,
        *,
        evaluate_test: bool = True,
    ) -> BaselineFitResult: ...


@dataclass(frozen=True)
class BaselineSplitResult:
    split_index: int
    split_seed: int
    hyperparameters: Mapping[str, Any]
    best_trial_number: int | None
    best_epoch: int
    validation_roc_auc: float
    test_roc_auc: float


@dataclass(frozen=True)
class BaselineSummary:
    baseline: str
    dataset: str
    graph_fingerprint: str
    task: str
    training_config: BaselineTrainingConfig
    runs: tuple[BaselineSplitResult, ...]

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
            "baseline": self.baseline,
            "dataset": self.dataset,
            "graph_fingerprint": self.graph_fingerprint,
            "task": self.task,
            "hyperparameter_selection_protocol": selection,
            "training_config": asdict(self.training_config),
            "mean_test_roc_auc": self.mean_test_roc_auc,
            "std_test_roc_auc": self.std_test_roc_auc,
            "runs": [asdict(run) for run in self.runs],
        }


def run_baseline(
    baseline: LinkPredictionBaseline,
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    training_config: BaselineTrainingConfig,
    *,
    optuna_config: OptunaConfig | None = None,
    fixed_hyperparameters: Mapping[str, Any] | None = None,
    progress: Callable[[BaselineSplitResult], None] | None = None,
) -> BaselineSummary:
    """Select and test one independent baseline model on every split."""
    _validate_splits(splits)
    if (optuna_config is None) == (fixed_hyperparameters is None):
        raise ValueError("Provide exactly one of optuna_config or fixed_hyperparameters.")
    runs = []

    for split_index, split in enumerate(splits):
        study = None
        best_trial_number = None
        if fixed_hyperparameters is None:
            def objective(trial):
                hyperparameters = baseline.suggest_hyperparameters(trial)
                trial.set_user_attr("hyperparameters", hyperparameters)
                return baseline.fit(
                    graph,
                    split,
                    hyperparameters,
                    replace(training_config, seed=training_config.seed + split.seed),
                    evaluate_test=False,
                ).validation_auc

            study = optimize_study(
                objective,
                optuna_config,
                study_name=f"split_{split_index}",
                seed_offset=split.seed,
            )
            hyperparameters = dict(study.best_trial.user_attrs["hyperparameters"])
            best_trial_number = study.best_trial.number
        else:
            hyperparameters = dict(fixed_hyperparameters)

        result = baseline.fit(
            graph,
            split,
            hyperparameters,
            replace(training_config, seed=training_config.seed + split.seed),
        )
        if result.test_auc is None:
            raise RuntimeError("Final baseline evaluation produced no test AUC.")
        if study is not None:
            study.set_user_attr("validation_roc_auc", result.validation_auc)
            study.set_user_attr("test_roc_auc", result.test_auc)
            study.set_user_attr("best_epoch", result.best_epoch)
        run = BaselineSplitResult(
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

    return BaselineSummary(
        baseline=baseline.name,
        dataset=graph.name,
        graph_fingerprint=graph.fingerprint,
        task=splits[0].task.value,
        training_config=training_config,
        runs=tuple(runs),
    )


def _validate_splits(splits):
    if not splits:
        raise ValueError("splits must not be empty.")
    if any(split.task != splits[0].task for split in splits):
        raise ValueError("All splits in an experiment must use the same task.")


__all__ = [
    "BaselineFitResult",
    "BaselineSummary",
    "BaselineTrainingConfig",
    "LinkPredictionBaseline",
    "run_baseline",
]
