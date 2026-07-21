"""On-disk records for link-prediction runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


def create_run_directory(output_root, *tags: str) -> Path:
    if not tags or any(not tag or "/" in tag for tag in tags):
        raise ValueError("Run tags must be non-empty and must not contain '/'.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_root) / "runs" / "_".join((timestamp, *tags))
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path, content: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    temporary.replace(path)


__all__ = ["create_run_directory", "save_json"]
