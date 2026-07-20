"""MagNet adapter using the public PyTorch Geometric implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score

from ..data import DirectedGraphData
from ..splits import EdgeExamples, LinkPredictionSplit
from ..training import resolve_device
from .base import BaselineFitResult, BaselineTrainingConfig


@dataclass(frozen=True)
class MagNetHyperparameters:
    q: float = 0.25
    hidden_channels: int = 16
    dropout: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    chebyshev_order: int = 1
    num_layers: int = 2
    activation: bool = True

    def __post_init__(self):
        if not 0 <= self.q <= 0.25:
            raise ValueError("MagNet q must lie in [0, 0.25].")
        if self.hidden_channels <= 0 or self.chebyshev_order <= 0:
            raise ValueError("Hidden channels and Chebyshev order must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters.")


class MagNetBaseline:
    """Binary MagNet classifier evaluated on existing project splits."""

    name = "magnet"
    search_space = {
        "q": [0.05, 0.1, 0.15, 0.2, 0.25],
        "hidden_channels": [16, 32, 64],
        "dropout": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8],
        "learning_rate": [1e-4, 1e-2],
        "weight_decay": [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        "chebyshev_order": 1,
        "num_layers": 2,
        "activation": [False, True],
    }

    def suggest_hyperparameters(self, trial) -> dict[str, Any]:
        # The structural grid follows the released MagNet link-prediction runs.
        params = MagNetHyperparameters(
            q=trial.suggest_categorical("q", self.search_space["q"]),
            hidden_channels=trial.suggest_categorical(
                "hidden_channels", self.search_space["hidden_channels"]
            ),
            dropout=trial.suggest_categorical(
                "dropout", self.search_space["dropout"]
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *self.search_space["learning_rate"], log=True
            ),
            weight_decay=trial.suggest_categorical(
                "weight_decay", self.search_space["weight_decay"]
            ),
            activation=trial.suggest_categorical(
                "activation", self.search_space["activation"]
            ),
        )
        return asdict(params)

    def fit(
        self,
        graph: DirectedGraphData,
        split: LinkPredictionSplit,
        hyperparameters: Mapping[str, Any],
        config: BaselineTrainingConfig,
        *,
        evaluate_test: bool = True,
    ) -> BaselineFitResult:
        return fit_magnet(
            graph,
            split,
            MagNetHyperparameters(**hyperparameters),
            config,
            evaluate_test=evaluate_test,
        )


def fit_magnet(
    graph: DirectedGraphData,
    split: LinkPredictionSplit,
    hyperparameters: MagNetHyperparameters,
    config: BaselineTrainingConfig,
    *,
    evaluate_test: bool = True,
) -> BaselineFitResult:
    """Train MagNet without using its incompatible built-in splitter."""
    try:
        import torch
        import torch.nn.functional as torch_f
        from torch_geometric_signed_directed.nn.directed import (
            MagNet_link_prediction,
        )
    except ImportError as exc:
        raise ImportError(
            "MagNet requires torch-geometric and "
            "torch-geometric-signed-directed."
        ) from exc

    _seed_everything(config.seed, torch)
    device = resolve_device(config.device)
    edge_index = torch.as_tensor(
        split.observed_edge_index,
        dtype=torch.long,
        device=device,
    )
    edge_weight = torch.ones(edge_index.shape[1], device=device)
    features = _in_out_degree(edge_index, graph.num_nodes, torch)
    model = MagNet_link_prediction(
        num_features=2,
        hidden=hyperparameters.hidden_channels,
        q=hyperparameters.q,
        K=hyperparameters.chebyshev_order,
        label_dim=2,
        activation=hyperparameters.activation,
        layer=hyperparameters.num_layers,
        dropout=hyperparameters.dropout,
        normalization="sym",
        cached=True,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    train_pairs, train_labels = _to_tensors(split.train, device, torch)

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    checks_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        log_prob = model(
            features,
            features,
            edge_index=edge_index,
            query_edges=train_pairs,
            edge_weight=edge_weight,
        )
        loss = torch_f.nll_loss(log_prob, train_labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("MagNet training loss became non-finite.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % config.evaluation_frequency != 0 and epoch != config.max_epochs:
            continue
        validation_auc = _evaluate(
            model,
            features,
            edge_index,
            edge_weight,
            split.validation,
            device,
            torch,
        )
        if validation_auc > best_auc or best_state is None:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
            if checks_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("MagNet training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    validation_auc = _evaluate(
        model,
        features,
        edge_index,
        edge_weight,
        split.validation,
        device,
        torch,
    )
    test_auc = (
        _evaluate(
            model,
            features,
            edge_index,
            edge_weight,
            split.test,
            device,
            torch,
        )
        if evaluate_test
        else None
    )
    return BaselineFitResult(
        best_epoch=best_epoch,
        validation_auc=validation_auc,
        test_auc=test_auc,
    )


def _in_out_degree(edge_index, num_nodes, torch):
    """Match MagNet's released two degree features without SciPy round-trips."""
    features = torch.zeros((num_nodes, 2), device=edge_index.device)
    ones = torch.ones(edge_index.shape[1], device=edge_index.device)
    features[:, 0].index_add_(0, edge_index[0], ones)
    features[:, 1].index_add_(0, edge_index[1], ones)
    return features


def _to_tensors(examples: EdgeExamples, device, torch):
    return (
        torch.as_tensor(examples.pairs, dtype=torch.long, device=device),
        torch.as_tensor(examples.labels, dtype=torch.long, device=device),
    )


def _evaluate(
    model,
    features,
    edge_index,
    edge_weight,
    examples: EdgeExamples,
    device,
    torch,
) -> float:
    pairs, labels = _to_tensors(examples, device, torch)
    model.eval()
    with torch.no_grad():
        log_prob = model(
            features,
            features,
            edge_index=edge_index,
            query_edges=pairs,
            edge_weight=edge_weight,
        )
    scores = log_prob[:, 1].exp().cpu().numpy()
    return float(roc_auc_score(labels.cpu().numpy(), scores))


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["MagNetBaseline", "MagNetHyperparameters", "fit_magnet"]
