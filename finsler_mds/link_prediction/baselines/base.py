"""Small common runner for external link-prediction baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Protocol

import numpy as np

from ..data import DirectedGraphData
from ..runs import save_json
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
    best_epoch: int
    validation_roc_auc: float
    test_roc_auc: float


@dataclass(frozen=True)
class BaselineSummary:
    baseline: str
    dataset: str
    graph_fingerprint: str
    task: str
    hyperparameters: Mapping[str, Any]
    training_config: BaselineTrainingConfig
    runs: tuple[BaselineSplitResult, ...]

    @property
    def mean_test_roc_auc(self) -> float:
        return float(np.mean([run.test_roc_auc for run in self.runs]))

    @property
    def std_test_roc_auc(self) -> float:
        return float(np.std([run.test_roc_auc for run in self.runs], ddof=0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "dataset": self.dataset,
            "graph_fingerprint": self.graph_fingerprint,
            "task": self.task,
            "hyperparameters": dict(self.hyperparameters),
            "training_config": asdict(self.training_config),
            "mean_test_roc_auc": self.mean_test_roc_auc,
            "std_test_roc_auc": self.std_test_roc_auc,
            "runs": [asdict(run) for run in self.runs],
        }


def tune_baseline(
    baseline: LinkPredictionBaseline,
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    training_config: BaselineTrainingConfig,
    *,
    num_trials: int,
    optuna_seed: int = 0,
    storage: str | None = None,
    timeout: float | None = None,
):
    """Tune one method on split 0 validation data only."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required to tune external baselines.") from exc
    _validate_splits(splits)
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    tuning_split = splits[0]

    def objective(trial):
        hyperparameters = baseline.suggest_hyperparameters(trial)
        trial.set_user_attr("hyperparameters", hyperparameters)
        result = baseline.fit(
            graph,
            tuning_split,
            hyperparameters,
            replace(training_config, seed=training_config.seed + tuning_split.seed),
            evaluate_test=False,
        )
        return result.validation_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
        storage=storage,
    )
    study.optimize(objective, n_trials=num_trials, timeout=timeout)
    return dict(study.best_trial.user_attrs["hyperparameters"]), study


def evaluate_baseline(
    baseline: LinkPredictionBaseline,
    graph: DirectedGraphData,
    splits: list[LinkPredictionSplit],
    hyperparameters: Mapping[str, Any],
    training_config: BaselineTrainingConfig,
) -> BaselineSummary:
    _validate_splits(splits)
    runs = []
    for split_index, split in enumerate(splits):
        result = baseline.fit(
            graph,
            split,
            hyperparameters,
            replace(training_config, seed=training_config.seed + split.seed),
        )
        if result.test_auc is None:
            raise RuntimeError("Final baseline evaluation produced no test AUC.")
        runs.append(
            BaselineSplitResult(
                split_index=split_index,
                split_seed=split.seed,
                best_epoch=result.best_epoch,
                validation_roc_auc=result.validation_auc,
                test_roc_auc=result.test_auc,
            )
        )
    return BaselineSummary(
        baseline=baseline.name,
        dataset=graph.name,
        graph_fingerprint=graph.fingerprint,
        task=splits[0].task.value,
        hyperparameters=dict(hyperparameters),
        training_config=training_config,
        runs=tuple(runs),
    )


def save_baseline_summary(path, summary: BaselineSummary) -> None:
    save_json(path, summary.as_dict())


def _validate_splits(splits: list[LinkPredictionSplit]) -> None:
    if not splits:
        raise ValueError("splits must not be empty.")
    if any(split.task != splits[0].task for split in splits):
        raise ValueError("All splits in an experiment must use the same task.")


__all__ = [
    "BaselineFitResult",
    "BaselineSummary",
    "BaselineTrainingConfig",
    "LinkPredictionBaseline",
    "evaluate_baseline",
    "save_baseline_summary",
    "tune_baseline",
]
