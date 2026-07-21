"""Shared Optuna configuration for per-split model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

HYPERPARAMETER_SELECTION_PROTOCOL = "independent_optuna_per_split_v1"


@dataclass(frozen=True)
class OptunaConfig:
    num_trials: int = 50
    seed: int = 0
    storage: str | None = None
    timeout: float | None = None

    def __post_init__(self):
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive.")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive when provided.")


def optimize_study(
    objective: Callable,
    config: OptunaConfig,
    *,
    study_name: str,
    seed_offset: int = 0,
):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for tuned experiments.") from exc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed + seed_offset),
        storage=config.storage,
        study_name=study_name,
    )
    study.optimize(
        objective,
        n_trials=config.num_trials,
        timeout=config.timeout,
    )
    return study


__all__ = [
    "HYPERPARAMETER_SELECTION_PROTOCOL",
    "OptunaConfig",
    "optimize_study",
]
