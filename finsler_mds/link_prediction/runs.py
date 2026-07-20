"""Minimal on-disk layout for link-prediction runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


def create_tagged_run_directory(output_root, *tags: str) -> Path:
    """Create a unique run directory described by short filename-safe tags."""
    if not tags or any(not tag or "/" in tag for tag in tags):
        raise ValueError("Run tags must be non-empty and must not contain '/'.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = Path(output_root) / "runs" / "_".join((timestamp, *tags))
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def create_run_directory(
    output_root,
    *,
    dataset: str,
    metric: str,
    dimensions: tuple[int, ...],
    alpha_max: float | None,
    num_trials: int | None,
    protocol: str,
) -> Path:
    """Create and return a uniquely timestamped directory for one invocation."""
    dimension_tag = "-".join(map(str, dimensions))
    parts = [dataset, metric, f"m{dimension_tag}"]
    if alpha_max is None:
        parts.append("fixed")
    else:
        parts.extend((f"amax{alpha_max:g}", f"n{num_trials}"))
    parts.append(protocol)
    return create_tagged_run_directory(output_root, *parts)


def save_json(path, content: Mapping[str, Any]) -> None:
    """Atomically save a JSON record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    temporary.replace(path)


__all__ = ["create_run_directory", "create_tagged_run_directory", "save_json"]
