"""Training loop for direct Finsler node embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as torch_f

from .evaluation import roc_auc
from .initialization import (
    DEFAULT_INITIALIZATION,
    INITIALIZATION_NAMES,
    spectral_initialization,
)
from .model import FinslerLinkPredictor
from .splits import LinkPredictionSplit


EMBEDDING_TRAINING_PROTOCOL = "weighted_mixed_negatives_heldout_blind_v2"


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-2
    max_epochs: int = 3_000
    patience: int = 300
    evaluation_frequency: int = 1
    batch_size: int | None = None
    evaluation_batch_size: int | None = None
    device: str = "auto"
    seed: int = 0
    initialization: str = DEFAULT_INITIALIZATION

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("max_epochs and patience must be positive.")
        if self.evaluation_frequency <= 0:
            raise ValueError("evaluation_frequency must be positive.")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided.")
        if self.evaluation_batch_size is not None and self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive when provided.")
        if self.initialization not in INITIALIZATION_NAMES:
            raise ValueError(
                f"initialization must be one of {', '.join(INITIALIZATION_NAMES)}."
            )


@dataclass(frozen=True)
class FitResult:
    embedding: np.ndarray
    best_epoch: int
    validation_auc: float
    test_auc: float | None


def fit_link_predictor(
    num_nodes: int,
    split: LinkPredictionSplit,
    metric,
    *,
    dimension: int,
    radius: float,
    temperature: float,
    positive_weight: float = 1.0,
    reverse_negative_fraction: float = 0.5,
    config: TrainingConfig | None = None,
    evaluate_test: bool = True,
) -> FitResult:
    """Reconstruct the observed graph and restore the best validation state."""
    config = TrainingConfig() if config is None else config
    if positive_weight <= 0:
        raise ValueError("positive_weight must be positive.")
    if not 0 <= reverse_negative_fraction <= 1:
        raise ValueError("reverse_negative_fraction must be in [0, 1].")
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    initial_embedding = (
        spectral_initialization(
            split.observed_edge_index,
            num_nodes,
            dimension,
            config.seed,
        )
        if config.initialization == "spectral"
        else None
    )
    model = FinslerLinkPredictor(
        num_nodes,
        dimension,
        metric,
        radius=radius,
        temperature=temperature,
        initialization=config.initialization,
        initial_embedding=initial_embedding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    positive_pairs = torch.as_tensor(
        split.observed_edge_index.T.copy(), dtype=torch.long, device=device
    )
    if not len(positive_pairs):
        raise ValueError("The observed graph must contain at least one edge.")
    inverse_negatives, forbidden_keys = _negative_sampling_plan(
        split, num_nodes, device
    )
    if len(forbidden_keys) >= num_nodes * (num_nodes - 1):
        raise ValueError("Negative sampling requires at least one graph non-edge.")
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    checks_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(positive_pairs), device=device)
        batch_size = config.batch_size or len(positive_pairs)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            positives = positive_pairs[indices]
            negatives = _sample_negatives(
                len(positives),
                num_nodes,
                inverse_negatives,
                forbidden_keys,
                device,
                reverse_negative_fraction,
            )
            positive_loss = torch_f.softplus(-model.logits(positives)).mean()
            negative_loss = torch_f.softplus(model.logits(negatives)).mean()
            loss = (
                positive_weight * positive_loss + negative_loss
            ) / (positive_weight + 1)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "The link-prediction loss became non-finite; check metric "
                    "parameters, initialization, and learning rate."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.center_embeddings_()

        if epoch % config.evaluation_frequency != 0 and epoch != config.max_epochs:
            continue
        validation_auc = roc_auc(
            model,
            split.validation,
            task=split.task,
            device=device,
            batch_size=config.evaluation_batch_size,
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
        raise RuntimeError("Training ended without a validation checkpoint.")
    model.load_state_dict(best_state)
    validation_auc = roc_auc(
        model,
        split.validation,
        task=split.task,
        device=device,
        batch_size=config.evaluation_batch_size,
    )
    test_auc = (
        roc_auc(
            model,
            split.test,
            task=split.task,
            device=device,
            batch_size=config.evaluation_batch_size,
        )
        if evaluate_test
        else None
    )
    embedding = model.embedding.weight.detach().cpu().numpy().copy()
    return FitResult(
        embedding=embedding,
        best_epoch=best_epoch,
        validation_auc=validation_auc,
        test_auc=test_auc,
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    return resolved


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_negatives(
    count: int,
    num_nodes: int,
    inverse_negatives: torch.Tensor,
    forbidden_keys: torch.Tensor,
    device: torch.device,
    reverse_fraction: float = 0.5,
) -> torch.Tensor:
    """Draw a configurable mixture of inverse arcs and unrelated non-edges."""
    num_inverse = round(count * reverse_fraction)
    inverse = inverse_negatives[
        torch.randint(len(inverse_negatives), (num_inverse,), device=device)
    ]
    random = _sample_random_non_edges(
        count - num_inverse, num_nodes, forbidden_keys, device
    )
    return torch.vstack((inverse, random))


def _sample_random_non_edges(
    count: int,
    num_nodes: int,
    forbidden_keys: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pairs = torch.randint(num_nodes, (count, 2), device=device)
    invalid = _invalid_pairs(pairs, num_nodes, forbidden_keys)
    while torch.any(invalid):
        pairs[invalid] = torch.randint(
            num_nodes, (int(invalid.sum().item()), 2), device=device
        )
        invalid = _invalid_pairs(pairs, num_nodes, forbidden_keys)
    return pairs


def _negative_sampling_plan(
    split: LinkPredictionSplit,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build inverse negatives and exclusions without reading held-out labels."""
    observed = split.observed_edge_index.T
    held_out = np.vstack((split.validation.pairs, split.test.pairs))
    held_out = np.vstack((held_out, held_out[:, ::-1]))
    observed_keys = observed[:, 0] * num_nodes + observed[:, 1]
    held_out_keys = held_out[:, 0] * num_nodes + held_out[:, 1]
    if np.intersect1d(observed_keys, held_out_keys).size:
        raise ValueError("Observed and held-out node pairs must be disjoint.")
    reverse_keys = observed[:, 1] * num_nodes + observed[:, 0]
    inverse = observed[~np.isin(reverse_keys, observed_keys)][:, ::-1].copy()
    if not len(inverse):
        raise ValueError("Inverse sampling requires a non-reciprocal observed edge.")
    keys = np.unique(np.concatenate((observed_keys, reverse_keys, held_out_keys)))
    return (
        torch.as_tensor(inverse, dtype=torch.long, device=device),
        torch.as_tensor(keys, dtype=torch.long, device=device),
    )


def _invalid_pairs(
    pairs: torch.Tensor,
    num_nodes: int,
    forbidden_keys: torch.Tensor,
) -> torch.Tensor:
    keys = pairs[:, 0] * num_nodes + pairs[:, 1]
    positions = torch.searchsorted(forbidden_keys, keys)
    in_graph = positions < len(forbidden_keys)
    matched = torch.zeros_like(in_graph)
    matched[in_graph] = forbidden_keys[positions[in_graph]] == keys[in_graph]
    return (pairs[:, 0] == pairs[:, 1]) | matched


__all__ = [
    "EMBEDDING_TRAINING_PROTOCOL",
    "FitResult",
    "TrainingConfig",
    "fit_link_predictor",
    "resolve_device",
]
