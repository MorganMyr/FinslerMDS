"""ROC-AUC evaluation for directed-link prediction."""

from __future__ import annotations

from sklearn.metrics import roc_auc_score
import torch

from .splits import EdgeExamples, LinkTask


SCORING_PROTOCOL = "direction_logit_delta_v1"


def score_name(task: LinkTask | str) -> str:
    return (
        "logit_ij_minus_logit_ji"
        if LinkTask(task) is LinkTask.DIRECTION
        else "p_ij"
    )


def link_scores(model, pairs: torch.Tensor, task: LinkTask | str) -> torch.Tensor:
    """Return logits that rank the requested directed-link task."""
    logits = model.logits(pairs)
    if LinkTask(task) is LinkTask.EXISTENCE:
        return logits.sigmoid()
    return logits - model.logits(pairs.flip(1))


@torch.no_grad()
def roc_auc(
    model,
    examples: EdgeExamples,
    *,
    task: LinkTask | str,
    device: torch.device | str,
    batch_size: int | None = None,
) -> float:
    model.eval()
    device = torch.device(device)
    pairs = torch.as_tensor(examples.pairs, dtype=torch.long, device=device)
    labels = torch.as_tensor(examples.labels, dtype=torch.float32, device=device)
    batch_size = len(pairs) if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")

    scores = []
    for start in range(0, len(pairs), batch_size):
        scores.append(link_scores(model, pairs[start : start + batch_size], task))
    return float(
        roc_auc_score(labels.cpu().numpy(), torch.cat(scores).cpu().numpy())
    )


__all__ = ["SCORING_PROTOCOL", "link_scores", "roc_auc", "score_name"]
