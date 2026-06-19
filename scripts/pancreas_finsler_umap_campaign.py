"""Finsler-UMAP alpha campaign on pancreas.

Prepared for manual execution.  It keeps pair reweighting disabled, always uses
the saved 2D UMAP as initialization, evaluates each embedding with the standard
pancreas RNA-velocity metrics, and appends rows to a CSV.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import MatsumotoMetric, RandersMetric, fit_finsler_mds, utils  # noqa: E402
from finsler_mds.utils.pancreas import normalize_pair_weights, suppress_pancreas_noise_warnings  # noqa: E402
from finsler_mds.utils.pancreas_campaign import (  # noqa: E402
    PANCREAS_N_EVAL_NEIGHBORS,
    PANCREAS_UMAP,
    PANCREAS_VELOCITY,
    append_csv_row,
    read_csv_rows,
    summarize_csv_by_seed,
)
from scripts.evaluate_pancreas_embedding import (  # noqa: E402
    evaluate_embedding,
    load_pancreas_evaluation_context,
    make_pair_weights,
    velocity_inputs_path,
)
from scripts.main_pancreas import (  # noqa: E402
    cache_token,
    labels_to_cache,
    normalize_velocity_distance_formula,
    velocity_distance_formula_tag,
)


SEED = 42
EMBEDDING_DIM = 3
UMAP_INIT_DIM = 2
EMBEDDING_METRIC = "randers"  # one of {"randers", "matsumoto"}
VELOCITY_DISTANCE_FORMULA = "matsumoto"  # one of {"randers", "matsumoto", "exponential"}
MAX_ITER = 500
OVERWRITE = False

V_ALPHA_COS_CLIP = {
    0.25: 1.0,
    0.5: 1.0,
    0.75: 1.0,
    1.0: 0.99,
    1.5: 0.66,
    2.0: 0.49,
}
EMB_ALPHA_GRID = (0.6, 0.8, 0.9)

FINSLER_UMAP_OPTIONS = {
    "n_neighbors": 50,
    "symmetrize_support": True,
    "symmetrize_rho": False,
    "symmetrize_sigma": True,
    "min_dist": 0.5,
    "spread": 1.0,
    "max_iter": MAX_ITER,
    "learning_rate": 1.0,
    "batch_size": 512,
    "negative_sample_rate": 10,
    "negative_sample_weight": 1.0,
    "negative_metric": "euclidean",
    "backend": "numba",
    "gradient_clip": 4.0,
    "verbose": 0,
}

CLUSTER_REWEIGHT_RHO = 0.0
FRONTIER_PAIRS_WEIGHT = 1.0
SELECTED_FRONTIERS = "all"
DISTANCE_REWEIGHT_POWER = 0.0

SCRIPT_DIR = Path(__file__).resolve().parent
PANCREAS_DIR = SCRIPT_DIR / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
EVAL_DIR = PANCREAS_DIR / "rna_velocity_evaluation"
CSV_PATH = EVAL_DIR / "pancreas_finsler_umap_campaign.csv"
SUMMARY_PATH = EVAL_DIR / "pancreas_finsler_umap_campaign_summary.csv"


def main() -> None:
    suppress_pancreas_noise_warnings()
    context = load_pancreas_evaluation_context(
        RAW_DIR,
        EVAL_DIR,
        n_eval_neighbors=PANCREAS_N_EVAL_NEIGHBORS,
    )
    init = load_umap_2d_init()
    done = set() if OVERWRITE else existing_keys(CSV_PATH)

    for v_alpha, cos_clip in V_ALPHA_COS_CLIP.items():
        dists = load_or_compute_dissimilarities(context, v_alpha, cos_clip)
        weight = make_weights(context, dists)
        for emb_alpha in EMB_ALPHA_GRID:
            key = row_key(VELOCITY_DISTANCE_FORMULA, v_alpha, cos_clip, emb_alpha)
            if key in done:
                continue
            print(
                f"Running Finsler-UMAP {EMBEDDING_DIM}D {EMBEDDING_METRIC} "
                f"v={v_alpha:g}, clip={cos_clip:g}, a={emb_alpha:g}",
                flush=True,
            )
            embedding, loss, n_iter = run_finsler_umap(init, dists, emb_alpha)
            row = evaluate_run(
                context=context,
                embedding=embedding,
                dists=dists,
                weight=weight,
                v_alpha=v_alpha,
                cos_clip=cos_clip,
                emb_alpha=emb_alpha,
                loss=loss,
                n_iter=n_iter,
            )
            append_csv_row(CSV_PATH, row)
            done.add(key)
            print(
                f"{row['name']}: CBDir={row['cbdir']:.4f}, ICVCoh={row['icvcoh']:.4f}, "
                f"Orient={row['spearman_cos']:.4f}, Sign={row['sign_correctness']:.4f}, "
                f"loss={row['optimizer_stress']:.4g}",
                flush=True,
            )

    write_summary()
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


def load_umap_2d_init():
    path = RAW_DIR / "umap_dynamical_s42.npy"
    if not path.exists():
        raise FileNotFoundError(f"Saved 2D UMAP init not found: {path}")
    init = np.asarray(np.load(path), dtype=float)
    if EMBEDDING_DIM == 2:
        return init
    if EMBEDDING_DIM == 3:
        return np.column_stack([init, np.zeros(len(init))])
    raise ValueError("EMBEDDING_DIM must be 2 or 3.")


def load_or_compute_dissimilarities(context, v_alpha, cos_clip):
    distance_formula = normalize_velocity_distance_formula(VELOCITY_DISTANCE_FORMULA)
    path = velocity_inputs_path(
        RAW_DIR,
        velocity_alpha=v_alpha,
        distance_formula=distance_formula,
        cos_clip=cos_clip,
        kNN_euclid=PANCREAS_VELOCITY["kNN_euclid"],
        kNN_finsler=PANCREAS_VELOCITY["kNN_finsler"],
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            return np.asarray(cache["dists_velocity"], dtype=float)

    print(f"Computing distance cache: {path.name}", flush=True)
    dists, _, _, _ = utils.compute_velocity_dist_matrix(
        context.x_pca,
        context.velocity_pca,
        kNN_euclid=PANCREAS_VELOCITY["kNN_euclid"],
        kNN_finsler=PANCREAS_VELOCITY["kNN_finsler"],
        alpha=v_alpha,
        distance_formula=distance_formula,
        cos_clip=cos_clip,
        velocity_neighbors=PANCREAS_VELOCITY["velocity_neighbors"],
        average_velocity=PANCREAS_VELOCITY["average_velocity"],
        symmetrize_support=PANCREAS_VELOCITY["symmetrize_support"],
        n_jobs=PANCREAS_VELOCITY["graph_n_jobs"],
    )
    np.savez(path, dists_velocity=dists, labels=labels_to_cache(context.labels))
    return np.asarray(dists, dtype=float)


def make_weights(context, dists):
    weights = make_pair_weights(
        context.labels,
        dists,
        cluster_reweight_rho=CLUSTER_REWEIGHT_RHO,
        frontier_pairs_weight=FRONTIER_PAIRS_WEIGHT,
        selected_frontiers=SELECTED_FRONTIERS,
        distance_reweighting={"power": DISTANCE_REWEIGHT_POWER, "epsilon": 1e-6},
        eval_raw_dir=EVAL_DIR / "raw",
        neighbor_indices=context.expression_neighbors,
        n_neighbors=PANCREAS_N_EVAL_NEIGHBORS,
    )
    if weights is not None:
        return weights
    weights = np.ones_like(dists, dtype=float)
    np.fill_diagonal(weights, 0.0)
    return normalize_pair_weights(weights, dists)[0]


def run_finsler_umap(init, dists, emb_alpha):
    return fit_finsler_mds(
        dists,
        metric=make_embedding_metric(emb_alpha),
        optimizer="finsler_umap",
        init=init,
        n_components=EMBEDDING_DIM,
        random_state=SEED,
        return_n_iter=True,
        print_time=True,
        **FINSLER_UMAP_OPTIONS,
    )


def evaluate_run(*, context, embedding, dists, weight, v_alpha, cos_clip, emb_alpha, loss, n_iter):
    velocity_tag = velocity_distance_formula_tag(VELOCITY_DISTANCE_FORMULA, alpha=v_alpha)
    name = (
        f"fumap_{EMBEDDING_DIM}d_{velocity_tag}_"
        f"{metric_short_name()}{cache_token(emb_alpha)}_i{MAX_ITER}_s{SEED}"
    )
    row = evaluate_embedding(
        name=name,
        kind=f"fumap_{EMBEDDING_METRIC}",
        embedding=embedding,
        context=context,
        metric=make_embedding_metric(emb_alpha),
        dissimilarities=dists,
        weight=weight,
    )
    row.update(
        {
            "seed": SEED,
            "method": f"fumap_{EMBEDDING_METRIC}",
            "embedding_metric": EMBEDDING_METRIC,
            "embedding_dim": EMBEDDING_DIM,
            "init_embedding_dim": UMAP_INIT_DIM,
            "init": "umap_2d",
            "velocity_distance_formula": normalize_velocity_distance_formula(VELOCITY_DISTANCE_FORMULA),
            "v_alpha": v_alpha,
            "cos_clip": cos_clip,
            "emb_alpha": emb_alpha,
            "max_iter": MAX_ITER,
            "backend": FINSLER_UMAP_OPTIONS["backend"],
            "negative_metric": FINSLER_UMAP_OPTIONS["negative_metric"],
            "symmetrize_support": FINSLER_UMAP_OPTIONS["symmetrize_support"],
            "symmetrize_rho": FINSLER_UMAP_OPTIONS["symmetrize_rho"],
            "symmetrize_sigma": FINSLER_UMAP_OPTIONS["symmetrize_sigma"],
            "cluster_reweight_rho": CLUSTER_REWEIGHT_RHO,
            "frontier_pairs_weight": FRONTIER_PAIRS_WEIGHT,
            "selected_frontiers": SELECTED_FRONTIERS,
            "distance_reweight_power": DISTANCE_REWEIGHT_POWER,
            "optimizer_stress": float(loss),
            "optimizer_n_iter": int(n_iter),
            "direct_stress_normalized": normalized_direct_stress(row, dists, weight),
        }
    )
    return row


def normalized_direct_stress(row, dists, weight):
    active = weight != 0
    denom = float(np.sum(weight[active] * dists[active] ** 2))
    return float(row["direct_weighted_stress"]) / denom if denom > 0 else np.nan


def existing_keys(path):
    return {
        row_key(row_velocity_distance_formula(row), float(row["v_alpha"]), float(row["cos_clip"]), float(row["emb_alpha"]))
        for row in read_csv_rows(path)
        if int(float(row.get("seed", -1))) == SEED
        and int(float(row.get("embedding_dim", -1))) == EMBEDDING_DIM
        and row.get("embedding_metric", "") == EMBEDDING_METRIC
        and row_velocity_distance_formula(row) == normalize_velocity_distance_formula(VELOCITY_DISTANCE_FORMULA)
        and int(float(row.get("max_iter", -1))) == MAX_ITER
        and row.get("backend", "numpy") == FINSLER_UMAP_OPTIONS["backend"]
        and row.get("negative_metric", "finsler") == FINSLER_UMAP_OPTIONS["negative_metric"]
        and parse_bool(row.get("symmetrize_support", False)) == FINSLER_UMAP_OPTIONS["symmetrize_support"]
        and parse_bool(row.get("symmetrize_rho", True)) == FINSLER_UMAP_OPTIONS["symmetrize_rho"]
        and parse_bool(row.get("symmetrize_sigma", True)) == FINSLER_UMAP_OPTIONS["symmetrize_sigma"]
    }


def row_key(distance_formula, v_alpha, cos_clip, emb_alpha):
    return (
        normalize_velocity_distance_formula(distance_formula),
        cache_token(v_alpha),
        cache_token(cos_clip),
        cache_token(emb_alpha),
    )


def row_velocity_distance_formula(row):
    value = row.get("velocity_distance_formula") or "randers"
    return normalize_velocity_distance_formula(value)


def write_summary():
    summarize_csv_by_seed(
        CSV_PATH,
        SUMMARY_PATH,
        group_keys=[
            "method",
            "embedding_metric",
            "embedding_dim",
            "init_embedding_dim",
            "velocity_distance_formula",
            "v_alpha",
            "cos_clip",
            "emb_alpha",
            "max_iter",
            "backend",
            "negative_metric",
            "symmetrize_support",
            "symmetrize_rho",
            "symmetrize_sigma",
        ],
        metric_keys=[
            "cbdir",
            "icvcoh",
            "gvcoh",
            "spearman_cos",
            "sign_correctness",
            "direct_weighted_stress",
            "direct_stress_normalized",
            "optimizer_stress",
        ],
    )


def make_embedding_metric(alpha):
    if EMBEDDING_METRIC == "randers":
        return RandersMetric(alpha=alpha)
    if EMBEDDING_METRIC == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    raise ValueError("EMBEDDING_METRIC must be one of {'randers', 'matsumoto'}.")


def metric_short_name():
    return {"randers": "r", "matsumoto": "mats"}[EMBEDDING_METRIC]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
