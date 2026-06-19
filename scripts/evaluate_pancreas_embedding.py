"""Evaluate saved pancreas embeddings with RNA-velocity metrics.

This module is intentionally importable by future pancreas test/campaign scripts.
It centralizes the loading of the cached pancreas state, CBDir boundary plans,
projected velocities, and optional direct weighted stress.

Examples:

    python scripts/evaluate_pancreas_embedding.py gd_2d_vrand2_r0p05_s42.npz
    python scripts/evaluate_pancreas_embedding.py scripts/res/pancreas/raw/umap_dynamical_s42.npy --output ''
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

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

from finsler_mds import RandersMetric  # noqa: E402
from finsler_mds.evaluation.rna_velocity import (  # noqa: E402
    cross_boundary_direction_correctness,
    global_velocity_coherence,
    in_cluster_velocity_coherence,
    load_or_compute_boundary_neighbor_plan,
    velocity_alignment_preservation_from_neighbors,
)
from finsler_mds.utils.pancreas import (  # noqa: E402
    PANCREAS_DATASET_SOURCE,
    PANCREAS_TRANSITIONS,
    apply_distance_pair_reweight,
    apply_frontier_pair_weight,
    cluster_balanced_pair_weights,
    normalize_pair_weights,
    neighbor_indices_from_sparse_distances,
    project_velocity_to_pca,
)


SEED = 42
N_EVAL_NEIGHBORS = 30
PREPROCESSING = {
    "min_shared_counts": 20,
    "n_top_genes": 3000,
    "n_pcs": 50,
    "moments_n_neighbors": 30,
}
VELOCITY = {
    "mode": "dynamical",
    "distance_formula": "randers",
    "cos_clip": 0.4,
    "velocity_neighbors": 30,
    "kNN_euclid": 30,
    "kNN_finsler": 0,
}


@dataclass(frozen=True)
class PancreasEvaluationContext:
    adata: object | None
    labels: np.ndarray
    expression_neighbors: np.ndarray
    x_pca: np.ndarray
    velocity_pca: np.ndarray
    velocity_transition: sparse.csr_matrix
    cbdir_plan: object
    n_eval_neighbors: int


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    pancreas_dir = script_dir / "res" / "pancreas"
    raw_dir = pancreas_dir / "raw"
    eval_dir = pancreas_dir / "rna_velocity_evaluation"

    embedding_path = resolve_embedding_path(args.embedding, raw_dir)
    embedding = load_embedding(embedding_path)
    context = load_pancreas_evaluation_context(
        raw_dir,
        eval_dir,
        n_eval_neighbors=args.n_neighbors,
    )
    if len(embedding) != len(context.labels):
        raise ValueError(
            f"Embedding has {len(embedding)} rows, but pancreas state has "
            f"{len(context.labels)} cells."
        )

    metric = RandersMetric(alpha=args.alpha_embedding)
    dists = None
    weight = None
    if args.weighted_stress:
        dists, cache_labels = load_velocity_dissimilarities(
            raw_dir,
            velocity_alpha=args.velocity_alpha,
            distance_formula=args.velocity_formula,
            cos_clip=args.velocity_cos_clip,
            kNN_euclid=args.velocity_knn_euclid,
            kNN_finsler=args.velocity_knn_finsler,
        )
        if not np.array_equal(context.labels, cache_labels):
            raise ValueError("Velocity cache labels do not match the cached pancreas state labels.")
        weight = make_pair_weights(
            context.labels,
            dists,
            cluster_reweight_rho=args.cluster_reweight_rho,
            frontier_pairs_weight=args.frontier_pairs_weight,
            selected_frontiers=args.selected_frontiers,
            distance_reweighting={
                "power": args.distance_reweight_power,
                "epsilon": args.distance_reweight_epsilon,
            },
            eval_raw_dir=eval_dir / "raw",
            neighbor_indices=context.expression_neighbors,
            n_neighbors=args.n_neighbors,
        )

    row = evaluate_embedding(
        name=args.name or embedding_path.stem,
        kind=args.kind,
        embedding=embedding,
        context=context,
        metric=metric,
        dissimilarities=dists,
        weight=weight,
    )
    row["embedding_path"] = str(embedding_path)

    print_row(row)
    if args.output is not None:
        append_csv(args.output, row)
        print(f"Saved evaluation row: {args.output}")


def load_pancreas_evaluation_context(
        raw_dir: Path,
        eval_dir: Path,
        *,
        n_eval_neighbors=N_EVAL_NEIGHBORS,
        load_adata=False,
) -> PancreasEvaluationContext:
    """Load a lightweight pancreas evaluation context.

    The heavy AnnData is only loaded when the lightweight cache is missing, or
    when callers explicitly need ``context.adata``.
    """
    raw_dir = Path(raw_dir)
    eval_raw_dir = Path(eval_dir) / "raw"
    eval_raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = evaluation_context_path(eval_raw_dir, n_eval_neighbors)
    state_path = pancreas_state_path(raw_dir)
    adata = None
    if cache_path.exists():
        labels, expression_neighbors, x_pca, velocity_pca, velocity_transition = load_evaluation_context_cache(cache_path)
    else:
        if not state_path.exists():
            raise FileNotFoundError(
                "Pancreas evaluation state cache is missing. Run main_pancreas.py or a "
                f"pancreas campaign once to create it: {state_path}"
            )

        import scanpy as sc

        adata = sc.read_h5ad(state_path)
        labels = np.asarray(adata.obs["clusters"].astype(str), dtype=str)
        expression_neighbors = neighbor_indices_from_sparse_distances(
            adata.obsp["distances"],
            n_neighbors=n_eval_neighbors,
        )
        x_pca = np.asarray(adata.obsm["X_pca"][:, :PREPROCESSING["n_pcs"]], dtype=float)
        velocity_pca = project_velocity_to_pca(adata, PREPROCESSING["n_pcs"])
        velocity_transition = scvelo_velocity_transition_matrix(adata)
        save_evaluation_context_cache(
            cache_path,
            labels=labels,
            expression_neighbors=expression_neighbors,
            x_pca=x_pca,
            velocity_pca=velocity_pca,
            velocity_transition=velocity_transition,
        )

    if load_adata and adata is None:
        if not state_path.exists():
            raise FileNotFoundError(
                "Pancreas evaluation state cache is missing. Run main_pancreas.py or a "
                f"pancreas campaign once to create it: {state_path}"
            )
        import scanpy as sc
        adata = sc.read_h5ad(state_path)

    cbdir_plan = load_cbdir_plan(
        eval_raw_dir,
        labels,
        expression_neighbors,
        n_neighbors=n_eval_neighbors,
    )
    return PancreasEvaluationContext(
        adata=adata,
        labels=labels,
        expression_neighbors=expression_neighbors,
        x_pca=x_pca,
        velocity_pca=velocity_pca,
        velocity_transition=velocity_transition,
        cbdir_plan=cbdir_plan,
        n_eval_neighbors=int(n_eval_neighbors),
    )


def evaluate_embedding(
        *,
        name: str,
        kind: str,
        embedding: np.ndarray,
        context: PancreasEvaluationContext,
        metric=None,
        dissimilarities: np.ndarray | None = None,
        weight: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate one embedding with CBDir, ICVCoh, Spearman-cos, sign correctness."""
    embedding = np.asarray(embedding, dtype=float)
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
    gvcoh = global_velocity_coherence(
        embedding,
        velocity_vectors=velocity_embedding,
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
        "kind": kind,
        "embedding_dim": int(embedding.shape[1]),
        "n_eval_neighbors": int(context.n_eval_neighbors),
        "cbdir": float(cbdir.score),
        "icvcoh": float(icvcoh.score),
        "gvcoh": float(gvcoh.score),
        "spearman_cos": float(alignment.spearman),
        "sign_correctness": float(alignment.sign_accuracy),
    }
    if weight is not None or dissimilarities is not None:
        if dissimilarities is None or metric is None:
            raise ValueError("metric, dissimilarities, and weight must be passed together.")
        if weight is None:
            weight = np.ones_like(dissimilarities, dtype=float)
            np.fill_diagonal(weight, 0.0)
        residual = metric.pairwise(embedding) - dissimilarities
        row["direct_weighted_stress"] = float(np.sum(weight * residual * residual))
    return row


def evaluation_context_path(eval_raw_dir: Path, n_neighbors: int) -> Path:
    return Path(eval_raw_dir) / (
        f"pancreas_eval_context_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"hvg{PREPROCESSING['n_top_genes']}_pca{PREPROCESSING['n_pcs']}_"
        f"k{int(n_neighbors)}_s{SEED}.npz"
    )


def save_evaluation_context_cache(
        path,
        *,
        labels,
        expression_neighbors,
        x_pca,
        velocity_pca,
        velocity_transition,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T = sparse.csr_matrix(velocity_transition)
    np.savez_compressed(
        path,
        labels=np.asarray(labels, dtype=str),
        expression_neighbors=np.asarray(expression_neighbors, dtype=np.int32),
        x_pca=np.asarray(x_pca, dtype=np.float32),
        velocity_pca=np.asarray(velocity_pca, dtype=np.float32),
        transition_data=np.asarray(T.data, dtype=np.float32),
        transition_indices=np.asarray(T.indices, dtype=np.int32),
        transition_indptr=np.asarray(T.indptr, dtype=np.int32),
        transition_shape=np.asarray(T.shape, dtype=np.int32),
    )


def load_evaluation_context_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        labels = cached_string_array(cache, "labels")
        expression_neighbors = np.asarray(cache["expression_neighbors"], dtype=int)
        x_pca = np.asarray(cache["x_pca"], dtype=float)
        velocity_pca = np.asarray(cache["velocity_pca"], dtype=float)
        shape = tuple(np.asarray(cache["transition_shape"], dtype=int))
        transition = sparse.csr_matrix(
            (
                np.asarray(cache["transition_data"], dtype=float),
                np.asarray(cache["transition_indices"], dtype=int),
                np.asarray(cache["transition_indptr"], dtype=int),
            ),
            shape=shape,
        )
    return labels, expression_neighbors, x_pca, velocity_pca, transition


def scvelo_velocity_transition_matrix(adata):
    """Return the scVelo transition matrix used by velocity_embedding."""
    try:
        from scvelo.tools.transition_matrix import transition_matrix
    except ImportError as exc:
        raise ImportError("scVelo is required to create the pancreas evaluation cache.") from exc

    transition = transition_matrix(
        adata,
        vkey="velocity",
        scale=10,
        self_transitions=True,
        use_negative_cosines=True,
    )
    transition = sparse.csr_matrix(transition)
    transition.setdiag(0.0)
    transition.eliminate_zeros()
    return transition


def project_velocity_to_embedding_from_transition(transition, embedding):
    """Project fixed scVelo transitions into an embedding without loading gene layers."""
    T = sparse.csr_matrix(transition)
    embedding = np.asarray(embedding, dtype=float)
    velocity_embedding = np.zeros_like(embedding, dtype=float)
    for row_idx in range(T.shape[0]):
        start, end = T.indptr[row_idx], T.indptr[row_idx + 1]
        indices = T.indices[start:end]
        if len(indices) == 0:
            continue
        dX = embedding[indices] - embedding[row_idx]
        norms = np.linalg.norm(dX, axis=1)
        nonzero = norms > 0
        dX[nonzero] /= norms[nonzero, None]
        dX[~nonzero] = 0.0
        probs = T.data[start:end]
        velocity_embedding[row_idx] = probs @ dX - probs.mean() * dX.sum(axis=0)
    return velocity_embedding


def pancreas_state_path(raw_dir: Path) -> Path:
    return Path(raw_dir) / (
        f"pancreas_campaign_state_{VELOCITY['mode']}_"
        f"hvg{PREPROCESSING['n_top_genes']}_pca{PREPROCESSING['n_pcs']}_s{SEED}.h5ad"
    )


def load_cbdir_plan(eval_raw_dir, labels, expression_neighbors, *, n_neighbors=N_EVAL_NEIGHBORS):
    eval_raw_dir = Path(eval_raw_dir)
    eval_raw_dir.mkdir(parents=True, exist_ok=True)
    path = eval_raw_dir / boundary_plan_name(n_neighbors)
    return load_or_compute_boundary_neighbor_plan(
        path,
        labels,
        cluster_edges=PANCREAS_TRANSITIONS,
        neighbor_indices=expression_neighbors,
        n_neighbors=n_neighbors,
        on_mismatch="recompute",
    )


def make_pair_weights(
        labels,
        dissimilarities,
        *,
        cluster_reweight_rho=0.0,
        frontier_pairs_weight=1.0,
        selected_frontiers="all",
        distance_reweighting=None,
        eval_raw_dir=None,
        neighbor_indices=None,
        n_neighbors=N_EVAL_NEIGHBORS,
):
    """Build the direct-stress pair weights used by pancreas experiments."""
    rho = float(cluster_reweight_rho)
    frontier_pairs_weight = float(frontier_pairs_weight)
    distance_reweighting = dict(distance_reweighting or {})
    distance_power = float(distance_reweighting.get("power", 0.0))
    selected_frontiers = normalize_selected_frontiers(selected_frontiers)
    if rho <= 0 and frontier_pairs_weight == 1.0 and distance_power <= 0:
        return None

    if rho > 0:
        weights = cluster_balanced_pair_weights(
            labels,
            dissimilarities,
            rho=rho,
        )
    else:
        weights = np.ones_like(dissimilarities, dtype=float)
        np.fill_diagonal(weights, 0.0)

    if distance_power > 0:
        weights = apply_distance_pair_reweight(
            weights,
            dissimilarities,
            power=distance_power,
            epsilon=distance_reweighting.get("epsilon", 1e-6),
        )

    if frontier_pairs_weight != 1.0:
        if eval_raw_dir is None:
            raise ValueError("eval_raw_dir is required when frontier_pairs_weight != 1.")
        boundary_plan = load_frontier_plan_for_weights(
            eval_raw_dir,
            labels,
            selected_frontiers=selected_frontiers,
            neighbor_indices=neighbor_indices,
            n_neighbors=n_neighbors,
        )
        weights = apply_frontier_pair_weight(
            weights,
            boundary_plan,
            factor=frontier_pairs_weight,
            dissimilarities=dissimilarities,
            symmetric=True,
        )
    return normalize_pair_weights(weights, dissimilarities)[0]


def load_frontier_plan_for_weights(
        eval_raw_dir,
        labels,
        *,
        selected_frontiers,
        neighbor_indices=None,
        n_neighbors=N_EVAL_NEIGHBORS,
):
    """Load a cached CBDir-frontier or all-intercluster-neighbor plan."""
    selected_frontiers = normalize_selected_frontiers(selected_frontiers)
    eval_raw_dir = Path(eval_raw_dir)
    if selected_frontiers == "cbdir":
        path = eval_raw_dir / boundary_plan_name(n_neighbors)
        return load_or_compute_boundary_neighbor_plan(
            path,
            labels,
            cluster_edges=PANCREAS_TRANSITIONS,
            neighbor_indices=neighbor_indices,
            n_neighbors=n_neighbors,
            on_mismatch="recompute" if neighbor_indices is not None else "error",
        )

    labels = np.asarray(labels, dtype=str)
    cluster_edges = [
        (source, target)
        for source in np.unique(labels)
        for target in np.unique(labels)
        if source != target
    ]
    path = eval_raw_dir / all_frontier_plan_name(n_neighbors)
    return load_or_compute_boundary_neighbor_plan(
        path,
        labels,
        cluster_edges=cluster_edges,
        neighbor_indices=neighbor_indices,
        n_neighbors=n_neighbors,
        on_mismatch="recompute" if neighbor_indices is not None else "error",
    )


def load_velocity_dissimilarities(
        raw_dir,
        *,
        velocity_alpha,
        distance_formula="randers",
        cos_clip=0.4,
        kNN_euclid=30,
        kNN_finsler=0,
):
    """Load a main_pancreas-compatible velocity-input cache."""
    path = velocity_inputs_path(
        raw_dir,
        velocity_alpha=velocity_alpha,
        distance_formula=distance_formula,
        cos_clip=cos_clip,
        kNN_euclid=kNN_euclid,
        kNN_finsler=kNN_finsler,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Velocity dissimilarity cache not found: {path}. "
            "Run main_pancreas.py or precompute_pancreas_velocity_distance_caches.py "
            "with matching velocity parameters first."
        )
    with np.load(path, allow_pickle=False) as cache:
        dists = np.asarray(cache["dists_velocity"], dtype=float)
        labels = cached_string_array(cache, "labels", default_size=dists.shape[0])
    return dists, labels


def velocity_inputs_path(
        raw_dir,
        *,
        velocity_alpha,
        distance_formula="randers",
        cos_clip=0.4,
        kNN_euclid=30,
        kNN_finsler=0,
):
    return Path(raw_dir) / (
        f"pancreas_velocity_inputs_{VELOCITY['mode']}_"
        f"{velocity_formula_tag(distance_formula)}_"
        f"valpha{cache_token(velocity_alpha)}_"
        f"cclip{cache_token(cos_clip)}_"
        f"ke{int(kNN_euclid)}_kf{int(kNN_finsler)}_s{SEED}.npz"
    )


def load_embedding(path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as cache:
            for key in ("embedding", "X", "x", "umap", "isomap"):
                if key in cache:
                    return np.asarray(cache[key], dtype=float)
            numeric_keys = [
                key for key in cache.files
                if np.asarray(cache[key]).ndim == 2 and np.issubdtype(np.asarray(cache[key]).dtype, np.number)
            ]
            if len(numeric_keys) == 1:
                return np.asarray(cache[numeric_keys[0]], dtype=float)
    raise ValueError(f"Cannot load embedding from {path}; expected .npy or .npz with key 'embedding'.")


def resolve_embedding_path(value: str, raw_dir: Path) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.insert(0, raw_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Embedding file not found. Searched: {searched}")


def boundary_plan_name(n_neighbors: int) -> str:
    return (
        f"cbdir_boundaries_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"hvg{PREPROCESSING['n_top_genes']}_pca{PREPROCESSING['n_pcs']}_"
        f"k{int(n_neighbors)}_s{SEED}.npz"
    )


def all_frontier_plan_name(n_neighbors: int) -> str:
    return (
        f"all_frontiers_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"hvg{PREPROCESSING['n_top_genes']}_pca{PREPROCESSING['n_pcs']}_"
        f"k{int(n_neighbors)}_s{SEED}.npz"
    )


def cached_string_array(cache, key, *, default_size=0):
    if key not in cache:
        return np.asarray([""] * int(default_size), dtype=str)
    values = np.asarray(cache[key])
    if values.dtype.kind in {"S", "a"}:
        values = np.char.decode(values, "utf-8")
    return np.asarray(values, dtype=str)


def velocity_formula_tag(distance_formula: str) -> str:
    distance_formula = str(distance_formula).lower()
    if distance_formula == "randers":
        return "vrand"
    if distance_formula in {"mats", "matsumoto"}:
        return "vmats"
    if distance_formula == "exponential":
        return "vexp"
    raise ValueError("distance_formula must be 'randers', 'matsumoto', or 'exponential'.")


def cache_token(value):
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return str(value).replace("/", "-").replace(".", "p")


def normalize_selected_frontiers(selected_frontiers):
    selected_frontiers = str(selected_frontiers).lower()
    if selected_frontiers not in {"cbdir", "all"}:
        raise ValueError("selected_frontiers must be one of {'cbdir', 'all'}.")
    return selected_frontiers


def print_row(row: dict[str, object]) -> None:
    parts = [
        str(row.get("name", "")),
        f"CBDir={float(row['cbdir']):.6f}",
        f"ICVCoh={float(row['icvcoh']):.6f}",
        f"GVCoh={float(row['gvcoh']):.6f}",
        f"SpearmanCos={float(row['spearman_cos']):.6f}",
        f"Sign={float(row['sign_correctness']):.6f}",
    ]
    if "direct_weighted_stress" in row:
        parts.append(f"stress={float(row['direct_weighted_stress']):.6g}")
    print(", ".join(parts))


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, object]] = []
    fieldnames = list(row)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_rows = list(reader)
            for key in reader.fieldnames or []:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved pancreas embedding (.npy or .npz with key 'embedding')."
    )
    parser.add_argument(
        "embedding",
        help="Embedding filename. Relative names are first searched in scripts/res/pancreas/raw/.",
    )
    parser.add_argument("--name", default=None, help="Name stored in the output row.")
    parser.add_argument("--kind", default="saved", help="Kind stored in the output row.")
    parser.add_argument("--n-neighbors", type=int, default=N_EVAL_NEIGHBORS)
    parser.add_argument("--alpha-embedding", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV file to append to. Use --output '' to disable saving.",
    )
    parser.add_argument(
        "--weighted-stress",
        action="store_true",
        help="Also compute direct weighted stress with cached pancreas velocity dissimilarities.",
    )
    parser.add_argument("--velocity-alpha", type=float, default=0.0)
    parser.add_argument("--velocity-formula", choices=("randers", "matsumoto", "exponential"), default="randers")
    parser.add_argument("--velocity-cos-clip", type=float, default=VELOCITY["cos_clip"])
    parser.add_argument("--velocity-knn-euclid", type=int, default=VELOCITY["kNN_euclid"])
    parser.add_argument("--velocity-knn-finsler", type=int, default=VELOCITY["kNN_finsler"])
    parser.add_argument("--cluster-reweight-rho", type=float, default=0.0)
    parser.add_argument("--frontier-pairs-weight", type=float, default=1.0)
    parser.add_argument("--selected-frontiers", choices=("cbdir", "all"), default="all")
    parser.add_argument("--distance-reweight-power", type=float, default=0.0)
    parser.add_argument("--distance-reweight-epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if str(args.output) == "":
        args.output = None
    else:
        args.output = Path(args.output)
    if args.n_neighbors <= 0:
        parser.error("--n-neighbors must be positive.")
    return args


if __name__ == "__main__":
    main()
