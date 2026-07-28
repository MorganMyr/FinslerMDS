"""Evaluate a saved pancreas embedding with RNA-velocity metrics.

Examples:

    python scripts/evaluate_pancreas_embedding.py gd_2d_vmats1_r0p3.npz
    python scripts/evaluate_pancreas_embedding.py gd_2d_vmats1_r0p3.npz --direct-stress
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scripts"
    / "res"
    / "pancreas"
    / "rna_velocity_evaluation"
    / "pancreas_embedding_evaluation.csv"
)

from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    cross_boundary_direction_correctness,
    global_velocity_coherence,
    in_cluster_velocity_coherence,
    load_or_compute_boundary_neighbor_plan,
    velocity_alignment_preservation_from_neighbors,
)
from finsler_mds.utils.pancreas import (  # noqa: E402
    PANCREAS_DATASET_SOURCE,
    PANCREAS_N_EVAL_NEIGHBORS,
    PANCREAS_TRANSITIONS,
    compute_pancreas_velocity_graph,
    load_or_compute_pancreas_state,
    load_pancreas_distance_cache,
    make_embedding_metric,
    neighbor_indices_from_sparse_distances,
    pancreas_cache_metadata,
    pancreas_distance_cache_metadata,
    pancreas_state_metadata,
    project_velocity_to_pca,
)
from finsler_mds.utils.pancreas_files import (  # noqa: E402
    load_pancreas_embedding,
    pancreas_velocity_inputs_path,
    resolve_pancreas_embedding_path,
)
from scripts.main_pancreas import CONFIG as MAIN_PANCREAS_CONFIG  # noqa: E402


SEED = int(MAIN_PANCREAS_CONFIG["seed"])
N_EVAL_NEIGHBORS = PANCREAS_N_EVAL_NEIGHBORS
PREPROCESSING = MAIN_PANCREAS_CONFIG["preprocessing"]
VELOCITY = MAIN_PANCREAS_CONFIG["velocity"]


@dataclass(frozen=True)
class PancreasEvaluationContext:
    labels: np.ndarray
    cell_ids: np.ndarray
    expression_neighbors: np.ndarray
    x_pca: np.ndarray
    velocity_pca: np.ndarray
    velocity_transition: sparse.csr_matrix
    cbdir_plan: object
    n_eval_neighbors: int


def main() -> None:
    args = parse_args()
    pancreas_dir = Path(__file__).parent / "res" / "pancreas"
    raw_dir = pancreas_dir / "raw"
    context = load_pancreas_evaluation_context(
        raw_dir,
        pancreas_dir / "rna_velocity_evaluation",
    )
    embedding_path = resolve_pancreas_embedding_path(args.embedding, raw_dir)
    embedding = load_pancreas_embedding(embedding_path, cell_ids=context.cell_ids)

    metric = dissimilarities = None
    if args.direct_stress:
        parameters = embedding_parameters_from_name(embedding_path)
        if parameters is not None:
            velocity_formula, velocity_alpha, metric_kind, metric_alpha = parameters
            velocity = {
                **VELOCITY,
                "distance_formula": velocity_formula,
                "alpha": velocity_alpha,
            }
            dissimilarities, labels = load_velocity_dissimilarities(
                raw_dir,
                velocity=velocity,
                cell_ids=context.cell_ids,
            )
            if not np.array_equal(labels, context.labels):
                raise ValueError("Velocity-cache labels do not match the evaluation context.")
            metric = make_embedding_metric(metric_kind, metric_alpha)

    row = evaluate_embedding(
        name=embedding_path.stem,
        embedding=embedding,
        context=context,
        metric=metric,
        dissimilarities=dissimilarities,
    )
    row["embedding_path"] = str(embedding_path)
    print_row(row)
    if args.output is not None:
        append_csv_row(args.output, row)
        print(f"Saved evaluation row: {args.output}")


def load_pancreas_evaluation_context(
        raw_dir,
        eval_dir,
        *,
        n_eval_neighbors=N_EVAL_NEIGHBORS,
):
    """Load or build the lightweight state shared by pancreas evaluations."""
    eval_raw_dir = Path(eval_dir) / "raw"
    eval_raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = evaluation_context_metadata(n_eval_neighbors)
    path = evaluation_context_path(eval_raw_dir, n_eval_neighbors)
    cached = load_evaluation_context_cache(path, metadata)
    if cached is None:
        adata, labels, cell_ids, _, _, _ = load_or_compute_pancreas_state(
            raw_dir,
            preprocessing=PREPROCESSING,
            velocity=VELOCITY,
            seed=SEED,
        )
        if "velocity_graph" not in adata.uns:
            compute_pancreas_velocity_graph(
                adata,
                n_neighbors=VELOCITY["velocity_neighbors"],
                n_jobs=VELOCITY["graph_n_jobs"],
            )
        expression_neighbors = neighbor_indices_from_sparse_distances(
            adata.obsp["distances"],
            n_neighbors=n_eval_neighbors,
        )
        x_pca = np.asarray(adata.obsm["X_pca"][:, :PREPROCESSING["n_pcs"]], dtype=float)
        velocity_pca = project_velocity_to_pca(adata, PREPROCESSING["n_pcs"])
        velocity_transition = scvelo_velocity_transition_matrix(adata)
        cached = labels, cell_ids, expression_neighbors, x_pca, velocity_pca, velocity_transition
        save_evaluation_context_cache(path, *cached, metadata=metadata)

    labels, cell_ids, expression_neighbors, x_pca, velocity_pca, velocity_transition = cached
    cbdir_plan = load_or_compute_boundary_neighbor_plan(
        eval_raw_dir / boundary_plan_name(n_eval_neighbors),
        labels,
        cluster_edges=PANCREAS_TRANSITIONS,
        neighbor_indices=expression_neighbors,
        n_neighbors=n_eval_neighbors,
        on_mismatch="recompute",
    )
    return PancreasEvaluationContext(
        labels=labels,
        cell_ids=cell_ids,
        expression_neighbors=expression_neighbors,
        x_pca=x_pca,
        velocity_pca=velocity_pca,
        velocity_transition=velocity_transition,
        cbdir_plan=cbdir_plan,
        n_eval_neighbors=int(n_eval_neighbors),
    )


def evaluate_embedding(
        *,
        name,
        embedding,
        context,
        metric=None,
        dissimilarities=None,
):
    """Evaluate CBDir, velocity coherence/alignment, and optional direct stress."""
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[0] != len(context.labels):
        raise ValueError(
            f"Embedding has {len(embedding)} rows, expected {len(context.labels)}."
        )
    velocity_embedding = project_velocity_to_embedding_from_transition(
        context.velocity_transition,
        embedding,
    )
    cbdir = cross_boundary_direction_correctness(
        embedding,
        context.labels,
        PANCREAS_TRANSITIONS,
        velocity_vectors=velocity_embedding,
        neighbor_indices=context.expression_neighbors,
        boundary_plan=context.cbdir_plan,
        n_neighbors=context.n_eval_neighbors,
    )
    icvcoh = in_cluster_velocity_coherence(
        embedding,
        context.labels,
        velocity_vectors=velocity_embedding,
        neighbor_indices=context.expression_neighbors,
        n_neighbors=context.n_eval_neighbors,
    )
    alignment = velocity_alignment_preservation_from_neighbors(
        context.x_pca,
        context.velocity_pca,
        embedding,
        velocity_embedding,
        context.expression_neighbors,
    )
    row = {
        "name": name,
        "embedding_dim": int(embedding.shape[1]),
        "n_eval_neighbors": int(context.n_eval_neighbors),
        "cbdir": float(cbdir.score),
        "icvcoh": float(icvcoh.score),
        "gvcoh": float(global_velocity_coherence(embedding, velocity_vectors=velocity_embedding).score),
        "spearman_cos": float(alignment.spearman),
        "sign_correctness": float(alignment.sign_accuracy),
    }
    if dissimilarities is not None:
        if metric is None:
            raise ValueError("metric is required with dissimilarities.")
        active = np.isfinite(dissimilarities)
        np.fill_diagonal(active, False)
        residual = metric.pairwise(embedding) - dissimilarities
        row["direct_stress"] = float(np.sum(residual[active] ** 2))
    return row


def append_csv_row(path, row):
    """Append a row while preserving columns already present in the CSV."""
    path = Path(path)
    rows = []
    fieldnames = list(row)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames += [
                key for key in reader.fieldnames or () if key not in fieldnames
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)


def load_velocity_dissimilarities(raw_dir, *, velocity, cell_ids):
    path = pancreas_velocity_inputs_path(raw_dir, velocity=velocity, seed=SEED)
    cached = load_pancreas_distance_cache(
        path,
        pancreas_distance_cache_metadata(PREPROCESSING, velocity, SEED),
    )
    if cached is None:
        raise FileNotFoundError(
            f"Missing or incompatible velocity cache: {path}. "
            "Run main_pancreas.py with matching velocity parameters first."
        )
    dissimilarities, labels, cached_ids, _ = cached
    order = cell_id_order(cached_ids, cell_ids)
    return dissimilarities[np.ix_(order, order)], labels[order]


def cell_id_order(source_ids, target_ids):
    source_ids = np.asarray(source_ids, dtype=str)
    target_ids = np.asarray(target_ids, dtype=str)
    if np.array_equal(source_ids, target_ids):
        return np.arange(len(source_ids))
    positions = {cell_id: index for index, cell_id in enumerate(source_ids)}
    try:
        order = np.asarray([positions[cell_id] for cell_id in target_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Velocity cache is missing cell {exc.args[0]!r}.") from exc
    if len(order) != len(source_ids):
        raise ValueError("Velocity cache and evaluation context contain different cells.")
    return order


def embedding_parameters_from_name(path):
    """Read velocity/embedding metric kinds and alphas from a result filename."""
    match = re.search(
        r"(?:^|_)(vrand|vmats)([^_]+)_(cmats|mats|r)([^_]+)(?:_|$)",
        Path(path).stem.lower(),
    )
    if match is None:
        warnings.warn(
            "Direct stress skipped: the filename must contain '<vrand|vmats><alpha>_"
            "<r|mats|cmats><alpha>' to identify both metrics and alphas.",
            stacklevel=2,
        )
        return None
    velocity_tag, velocity_alpha, metric_tag, metric_alpha = match.groups()
    try:
        velocity_alpha = float(velocity_alpha.replace("m", "-").replace("p", "."))
        metric_alpha = float(metric_alpha.replace("m", "-").replace("p", "."))
    except ValueError:
        warnings.warn(
            "Direct stress skipped: metric alphas could not be read from the filename.",
            stacklevel=2,
        )
        return None
    return (
        {"vrand": "randers", "vmats": "matsumoto"}[velocity_tag],
        velocity_alpha,
        {"r": "randers", "mats": "matsumoto", "cmats": "convexified_matsumoto"}[
            metric_tag
        ],
        metric_alpha,
    )


def evaluation_context_metadata(n_neighbors):
    return pancreas_cache_metadata(
        state=pancreas_state_metadata(PREPROCESSING, VELOCITY, SEED),
        velocity_neighbors=VELOCITY["velocity_neighbors"],
        n_eval_neighbors=int(n_neighbors),
        transition_scale=10,
        transition_negative_cosines=True,
    )


def evaluation_context_path(eval_raw_dir, n_neighbors):
    return Path(eval_raw_dir) / (
        f"pancreas_eval_context_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"k{int(n_neighbors)}_s{SEED}.npz"
    )


def save_evaluation_context_cache(
        path,
        labels,
        cell_ids,
        expression_neighbors,
        x_pca,
        velocity_pca,
        velocity_transition,
        *,
        metadata,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transition = sparse.csr_matrix(velocity_transition)
    np.savez_compressed(
        path,
        labels=np.asarray(labels, dtype=str),
        cell_ids=np.asarray(cell_ids, dtype=str),
        expression_neighbors=np.asarray(expression_neighbors, dtype=np.int32),
        x_pca=np.asarray(x_pca, dtype=np.float32),
        velocity_pca=np.asarray(velocity_pca, dtype=np.float32),
        transition_data=np.asarray(transition.data, dtype=np.float32),
        transition_indices=np.asarray(transition.indices, dtype=np.int32),
        transition_indptr=np.asarray(transition.indptr, dtype=np.int32),
        transition_shape=np.asarray(transition.shape, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_evaluation_context_cache(path, expected_metadata):
    if not Path(path).exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        if "metadata_json" not in cache:
            return None
        if json.loads(str(cache["metadata_json"].item())) != expected_metadata:
            return None
        shape = tuple(np.asarray(cache["transition_shape"], dtype=int))
        transition = sparse.csr_matrix(
            (
                np.asarray(cache["transition_data"], dtype=float),
                np.asarray(cache["transition_indices"], dtype=int),
                np.asarray(cache["transition_indptr"], dtype=int),
            ),
            shape=shape,
        )
        return (
            np.asarray(cache["labels"], dtype=str),
            np.asarray(cache["cell_ids"], dtype=str),
            np.asarray(cache["expression_neighbors"], dtype=int),
            np.asarray(cache["x_pca"], dtype=float),
            np.asarray(cache["velocity_pca"], dtype=float),
            transition,
        )


def scvelo_velocity_transition_matrix(adata):
    from scvelo.tools.transition_matrix import transition_matrix

    transition = sparse.csr_matrix(
        transition_matrix(
            adata,
            vkey="velocity",
            scale=10,
            self_transitions=True,
            use_negative_cosines=True,
        )
    )
    transition.setdiag(0.0)
    transition.eliminate_zeros()
    return transition


def project_velocity_to_embedding_from_transition(transition, embedding):
    """Reproduce scVelo's local projection from a cached transition matrix."""
    transition = sparse.csr_matrix(transition)
    embedding = np.asarray(embedding, dtype=float)
    velocity_embedding = np.zeros_like(embedding)
    for source in range(transition.shape[0]):
        start, end = transition.indptr[source], transition.indptr[source + 1]
        targets = transition.indices[start:end]
        if len(targets) == 0:
            continue
        displacements = embedding[targets] - embedding[source]
        norms = np.linalg.norm(displacements, axis=1)
        nonzero = norms > 0
        displacements[nonzero] /= norms[nonzero, None]
        displacements[~nonzero] = 0.0
        probabilities = transition.data[start:end]
        velocity_embedding[source] = (
            probabilities @ displacements
            - probabilities.mean() * displacements.sum(axis=0)
        )
    return velocity_embedding


def boundary_plan_name(n_neighbors):
    return (
        f"cbdir_boundaries_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"k{int(n_neighbors)}_s{SEED}.npz"
    )


def print_row(row):
    parts = [
        str(row["name"]),
        f"CBDir={row['cbdir']:.3f}",
        f"ICVCoh={row['icvcoh']:.3f}",
        f"VAC={row['spearman_cos']:.3f}",
        f"VAS={row['sign_correctness']:.3f}",
        f"GVCoh={row['gvcoh']:.3f}",
    ]
    if "direct_stress" in row:
        parts.append(f"stress={row['direct_stress']:.6g}")
    print(", ".join(parts))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("embedding", help="Embedding name or NPZ path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output; use '' to disable.")
    parser.add_argument("--direct-stress", action="store_true")
    args = parser.parse_args()
    args.output = None if str(args.output) == "" else Path(args.output)
    return args


if __name__ == "__main__":
    main()
