"""Reusable helpers for pancreas experiment scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PANCREAS_SEED = 42
PANCREAS_N_EVAL_NEIGHBORS = 30
PANCREAS_PREPROCESSING = {
    "min_shared_counts": 20,
    "n_top_genes": 3000,
    "n_pcs": 50,
    "moments_n_neighbors": 30,
}
PANCREAS_VELOCITY = {
    "mode": "dynamical",
    "distance_formula": "randers",
    "cos_clip": 0.4,
    "velocity_neighbors": 30,
    "kNN_euclid": 30,
    "kNN_finsler": 0,
    "average_velocity": True,
    "symmetrize_support": True,
    "graph_n_jobs": -1,
    "recover_dynamics_max_iter": 20,
    "recover_dynamics_n_jobs": -1,
}
PANCREAS_UMAP = {
    "n_neighbors": 50,
    "min_dist": 0.5,
    "spread": 1.0,
    "maxiter": 1500,
    "negative_sample_rate": 10,
    "init_pos": "spectral",
}
PANCREAS_ISOMAP = {"n_neighbors": 30}
PANCREAS_GD_OPTIONS = {
    "method": "L-BFGS-B",
    "eps": 1e-8,
    "optimizer_options": {"ftol": 1e-10, "maxls": 80, "maxcor": 30},
    "device": "auto",
    "gpu_block_size": 192,
    "verbose": 0,
    "return_n_iter": True,
    "print_time": True,
}


def load_embedding(path):
    """Load a saved embedding from ``.npy`` or ``.npz``."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    with np.load(path, allow_pickle=False) as cache:
        return np.asarray(cache["embedding"], dtype=float)


def latest_matching_embedding(raw_dir, pattern):
    """Return the most recently modified embedding matching ``pattern``."""
    candidates = list(Path(raw_dir).glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_iter_from_name(name):
    """Extract the first ``i<number>`` token from an embedding filename."""
    for part in Path(name).stem.split("_"):
        if part.startswith("i") and part[1:].isdigit():
            return int(part[1:])
    return -1


def write_csv(path, rows):
    """Write heterogeneous row dictionaries with the union of all keys."""
    rows = list(rows)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path, row):
    """Append one heterogeneous row while preserving existing CSV columns."""
    path = Path(path)
    old_rows = []
    fieldnames = list(row)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_rows = list(reader)
            for key in reader.fieldnames or []:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for old in old_rows:
            writer.writerow(old)
        writer.writerow(row)


def read_csv_rows(path):
    """Read CSV rows, returning an empty list when the file is absent."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_csv_by_seed(path, output_path, *, group_keys, metric_keys):
    """Write mean/std summaries over seeds for selected metrics."""
    rows = read_csv_rows(path)
    groups = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in group_keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group_rows in sorted(groups.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        out["n_seeds"] = len({row.get("seed", "") for row in group_rows})
        for metric in metric_keys:
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric, "") not in {"", "nan", "NaN", "None"}
            ]
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values))
        summary.append(out)
    write_csv(output_path, summary)
    return summary


def print_metric_row(row):
    """Print a compact one-line summary for pancreas metric rows."""
    pieces = [str(row.get("name", row.get("label", "<unnamed>")))]
    for key in ("cbdir", "icvcoh", "spearman_cos", "sign_correctness", "direct_weighted_stress"):
        if key in row and row[key] != "":
            try:
                pieces.append(f"{key}={float(row[key]):.4g}")
            except (TypeError, ValueError):
                pieces.append(f"{key}={row[key]}")
    print(", ".join(pieces))


def velocity_cos_clip(velocity_alpha, overrides=None, default=0.4):
    """Return the cos-clip convention used by old pancreas sweeps."""
    overrides = {} if overrides is None else dict(overrides)
    return float(overrides.get(float(velocity_alpha), default))
