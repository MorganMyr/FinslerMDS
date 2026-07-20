"""Gradient-descent alpha campaign on pancreas.

Prepared for manual execution.  It tests all combinations of 2D/3D,
Randers/Matsumoto velocity dissimilarities, Randers/Matsumoto embedding
metrics, and the alpha grids defined below.  It always starts from the saved
2D UMAP embedding and appends one row per run to a CSV.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import MatsumotoMetric, RandersMetric, fit_finsler_mds, utils  # noqa: E402
from finsler_mds.utils.pancreas import suppress_pancreas_noise_warnings  # noqa: E402
from finsler_mds.utils.pancreas_files import resolve_pancreas_embedding_path  # noqa: E402
from finsler_mds.utils.pancreas_campaign import PANCREAS_N_EVAL_NEIGHBORS, append_csv_row, read_csv_rows  # noqa: E402
from scripts.evaluate_pancreas_embedding import (  # noqa: E402
    evaluate_embedding,
    load_embedding,
    load_pancreas_evaluation_context,
    velocity_inputs_path,
)
from scripts.main_pancreas import cache_token, normalize_velocity_distance_formula, velocity_distance_formula_tag  # noqa: E402


SEED = 42
OVERWRITE = False

# Edit these grids before launching the campaign.
VELOCITY_ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
EMBEDDING_ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

EMBEDDING_DIMS = (2, 3)
VELOCITY_DISTANCE_FORMULAS = ("randers", "matsumoto")
EMBEDDING_METRICS = ("randers", "matsumoto")
COS_CLIP = 1.0

VELOCITY = {
    "velocity_neighbors": 30,
    "kNN_euclid": 30,
    "kNN_finsler": 0,
    "average_velocity": True,
    "symmetrize_support": True,
    "graph_n_jobs": -1,
}
GRADIENT_DESCENT = {
    "max_iter": 100,
    "eps": 1e-8,
    "method": "L-BFGS-B",
    "optimizer_options": {"ftol": 1e-10, "maxls": 80, "maxcor": 30},
    "device": "auto",
    "gpu_block_size": 256,
    "verbose": 0,
}

SCRIPT_DIR = Path(__file__).resolve().parent
PANCREAS_DIR = SCRIPT_DIR / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
EVAL_DIR = PANCREAS_DIR / "rna_velocity_evaluation"
CSV_PATH = EVAL_DIR / "pancreas_gd_alpha_campaign.csv"


def main() -> None:
    validate_grids()
    suppress_pancreas_noise_warnings()
    context = load_pancreas_evaluation_context(
        RAW_DIR,
        EVAL_DIR,
        n_eval_neighbors=PANCREAS_N_EVAL_NEIGHBORS,
    )
    umap_2d = load_umap_2d_init()
    done = set() if OVERWRITE else existing_keys(CSV_PATH)

    for velocity_formula in VELOCITY_DISTANCE_FORMULAS:
        velocity_formula = normalize_velocity_distance_formula(velocity_formula)
        for v_alpha in VELOCITY_ALPHA_GRID:
            dists = load_or_compute_dissimilarities(context, velocity_formula, v_alpha)
            for embedding_metric in EMBEDDING_METRICS:
                for emb_alpha in EMBEDDING_ALPHA_GRID:
                    metric = make_embedding_metric(embedding_metric, emb_alpha)
                    for dim in EMBEDDING_DIMS:
                        key = row_key(velocity_formula, v_alpha, embedding_metric, emb_alpha, dim)
                        if key in done:
                            continue
                        init = init_for_dim(umap_2d, dim)
                        print(
                            f"Running GD {dim}D {velocity_formula}->{embedding_metric} "
                            f"v={v_alpha:g}, a={emb_alpha:g}",
                            flush=True,
                        )
                        embedding, stress, n_iter = fit_finsler_mds(
                            dists,
                            metric=metric,
                            optimizer="gradient_descent",
                            init=init,
                            n_components=dim,
                            random_state=SEED,
                            return_n_iter=True,
                            print_time=True,
                            **GRADIENT_DESCENT,
                        )
                        row = evaluate_run(
                            context=context,
                            embedding=embedding,
                            dists=dists,
                            metric=metric,
                            velocity_formula=velocity_formula,
                            v_alpha=v_alpha,
                            embedding_metric=embedding_metric,
                            emb_alpha=emb_alpha,
                            stress=stress,
                            n_iter=n_iter,
                        )
                        append_csv_row(CSV_PATH, row)
                        done.add(key)
                        print(
                            f"{row['name']}: CBDir={row['cbdir']:.3f}, "
                            f"ICVCoh={row['icvcoh']:.3f}, VAC={row['spearman_cos']:.3f}, "
                            f"VAS={row['sign_correctness']:.3f}, GVCoh={row['gvcoh']:.3f}, "
                            f"stress={row['optimizer_stress']:.4g}",
                            flush=True,
                        )
    print(f"Saved CSV: {CSV_PATH}")


def validate_grids() -> None:
    if any(alpha < 0 or alpha > 1 for alpha in VELOCITY_ALPHA_GRID):
        raise ValueError("VELOCITY_ALPHA_GRID values must be in [0, 1].")
    if any(alpha < 0 or alpha >= 1 for alpha in EMBEDDING_ALPHA_GRID):
        raise ValueError("EMBEDDING_ALPHA_GRID values must be in [0, 1).")


def load_umap_2d_init() -> np.ndarray:
    path = resolve_pancreas_embedding_path("umap_dynamical_s42.npy", RAW_DIR)
    return load_embedding(path)


def init_for_dim(umap_2d: np.ndarray, dim: int) -> np.ndarray:
    if dim == 2:
        return umap_2d
    if dim == 3:
        return np.column_stack([umap_2d, np.zeros(len(umap_2d))])
    raise ValueError("EMBEDDING_DIMS must contain only 2 and/or 3.")


def load_or_compute_dissimilarities(context, distance_formula: str, v_alpha: float) -> np.ndarray:
    path = velocity_inputs_path(
        RAW_DIR,
        velocity_alpha=v_alpha,
        distance_formula=distance_formula,
        cos_clip=COS_CLIP,
        kNN_euclid=VELOCITY["kNN_euclid"],
        kNN_finsler=VELOCITY["kNN_finsler"],
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            return np.asarray(cache["dists_velocity"], dtype=float)

    print(f"Computing distance cache: {path.name}", flush=True)
    dists, _, _, _ = utils.compute_velocity_dist_matrix(
        context.x_pca,
        context.velocity_pca,
        kNN_euclid=VELOCITY["kNN_euclid"],
        kNN_finsler=VELOCITY["kNN_finsler"],
        alpha=v_alpha,
        distance_formula=distance_formula,
        cos_clip=COS_CLIP,
        velocity_neighbors=VELOCITY["velocity_neighbors"],
        average_velocity=VELOCITY["average_velocity"],
        symmetrize_support=VELOCITY["symmetrize_support"],
        n_jobs=VELOCITY["graph_n_jobs"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, dists_velocity=dists, labels=np.asarray(context.labels, dtype=str))
    return np.asarray(dists, dtype=float)


def evaluate_run(
        *,
        context,
        embedding,
        dists,
        metric,
        velocity_formula,
        v_alpha,
        embedding_metric,
        emb_alpha,
        stress,
        n_iter,
) -> dict[str, object]:
    dim = int(embedding.shape[1])
    velocity_tag = velocity_distance_formula_tag(velocity_formula, alpha=v_alpha)
    metric_tag = {"randers": "r", "matsumoto": "mats"}[embedding_metric]
    name = f"gd_{dim}d_{velocity_tag}_{metric_tag}{cache_token(emb_alpha)}_s{SEED}"
    row = evaluate_embedding(
        name=name,
        kind=f"gd_{embedding_metric}",
        embedding=embedding,
        context=context,
        metric=metric,
        dissimilarities=dists,
        weight=None,
    )
    row.update(
        {
            "seed": SEED,
            "method": f"gd_{embedding_metric}",
            "optimizer": "gradient_descent",
            "embedding_metric": embedding_metric,
            "embedding_dim": dim,
            "init": "umap_2d",
            "init_embedding_dim": 2,
            "velocity_distance_formula": velocity_formula,
            "v_alpha": float(v_alpha),
            "cos_clip": COS_CLIP,
            "emb_alpha": float(emb_alpha),
            "max_iter": GRADIENT_DESCENT["max_iter"],
            "optimizer_stress": float(stress),
            "optimizer_n_iter": int(n_iter),
            "direct_stress_normalized": normalized_direct_stress(row, dists),
        }
    )
    return row


def normalized_direct_stress(row, dists):
    active = ~np.eye(dists.shape[0], dtype=bool)
    denom = float(np.sum(dists[active] ** 2))
    return float(row["direct_weighted_stress"]) / denom if denom > 0 else np.nan


def make_embedding_metric(name: str, alpha: float):
    if name == "randers":
        return RandersMetric(alpha=alpha)
    if name == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    raise ValueError("EMBEDDING_METRICS must contain only 'randers' and/or 'matsumoto'.")


def existing_keys(path: Path) -> set[tuple[str, str, str, str, int]]:
    keys = set()
    for row in read_csv_rows(path):
        try:
            if int(float(row.get("seed", -1))) != SEED:
                continue
            if row.get("init", "") != "umap_2d":
                continue
            if int(float(row.get("max_iter", -1))) != GRADIENT_DESCENT["max_iter"]:
                continue
            keys.add(
                row_key(
                    row.get("velocity_distance_formula", "randers"),
                    float(row["v_alpha"]),
                    row["embedding_metric"],
                    float(row["emb_alpha"]),
                    int(float(row["embedding_dim"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def row_key(velocity_formula, v_alpha, embedding_metric, emb_alpha, dim):
    return (
        normalize_velocity_distance_formula(velocity_formula),
        cache_token(v_alpha),
        str(embedding_metric),
        cache_token(emb_alpha),
        int(dim),
    )


if __name__ == "__main__":
    main()
