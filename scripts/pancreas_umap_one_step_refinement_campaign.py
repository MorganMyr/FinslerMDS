"""One-step velocity-aware UMAP refinement campaign on pancreas.

The experiment keeps cluster/distance reweighting disabled and does not save
intermediate embeddings or figures.  It writes one CSV with every seed/run and
one CSV summarizing mean/std over UMAP seeds.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import MatsumotoMetric, RandersMetric, fit_finsler_mds, utils  # noqa: E402
from finsler_mds.utils.pancreas import (  # noqa: E402
    normalize_pair_weights,
    project_velocity_to_embedding,
    suppress_pancreas_noise_warnings,
)
from finsler_mds.utils.pancreas_campaign import (  # noqa: E402
    PANCREAS_GD_OPTIONS,
    PANCREAS_N_EVAL_NEIGHBORS,
    PANCREAS_PREPROCESSING,
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
    compute_umap_from_neighbors,
    labels_to_cache,
    normalize_velocity_distance_formula,
    velocity_distance_formula_tag,
)


SEEDS = (42,)
EMBEDDING_DIM = 3
UMAP_INIT_DIM = 3
ITERATIONS = (1, 2, 3, 10, 100)
INTERPOLATION_T = (0.25, 0.5, 0.75)
V_ALPHA_COS_CLIP = {
    0.5: 1.0,
    1.0: 0.9,
    2.0: 0.4,
    3.0: 0.3,
}
EMB_ALPHA_GRID = (0.2, 0.4, 0.6, 0.8, 0.9)
BASELINE_CASE = (0.0, 1.0, 0.0)  # v_alpha, cos_clip, emb_alpha

CLUSTER_REWEIGHT_RHO = 0.0
FRONTIER_PAIRS_WEIGHTS = (1.0,)
SELECTED_FRONTIERS = "all"
DISTANCE_REWEIGHT_POWER = 0.0
VELOCITY_DISTANCE_FORMULA = "randers"  # one of {"randers", "matsumoto"}
EMBEDDING_METRIC = "matsumoto"  # one of {"randers", "matsumoto"}
N_KNN_OVERLAP = 30
OVERWRITE = False

SCRIPT_DIR = Path(__file__).resolve().parent
PANCREAS_DIR = SCRIPT_DIR / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
EVAL_DIR = PANCREAS_DIR / "rna_velocity_evaluation"
OUT_DIR = EVAL_DIR / "umap_one_step_refinement"


def main() -> None:
    if EMBEDDING_DIM not in (2, 3):
        raise ValueError("EMBEDDING_DIM must be 2 or 3.")
    suppress_pancreas_noise_warnings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    context = load_pancreas_evaluation_context(
        RAW_DIR,
        EVAL_DIR,
        n_eval_neighbors=PANCREAS_N_EVAL_NEIGHBORS,
        load_adata=True,
    )
    for frontier_pairs_weight in FRONTIER_PAIRS_WEIGHTS:
        raw_csv, summary_csv = output_paths(frontier_pairs_weight)
        if OVERWRITE and raw_csv.exists():
            raw_csv.unlink()
        run_campaign(context, frontier_pairs_weight, raw_csv, summary_csv)


def run_campaign(context, frontier_pairs_weight, raw_csv, summary_csv) -> None:
    done = existing_keys(raw_csv)
    cases = alpha_cases()

    print(f"\n### afpw={frontier_pairs_weight:g} ###", flush=True)
    for seed in SEEDS:
        print(f"\n=== UMAP seed {seed} ===", flush=True)
        x_umap = load_or_compute_umap(context, seed)
        add_row_if_needed(
            raw_csv,
            done,
            row_key(seed, "umap", np.nan, np.nan, np.nan, 0, np.nan),
            lambda: evaluate_run(
                name=f"umap_s{seed}",
                method="umap",
                seed=seed,
                embedding=x_umap,
                x_umap=x_umap,
                context=context,
                v_alpha=np.nan,
                cos_clip=np.nan,
                emb_alpha=np.nan,
                max_iter=0,
                interp_t=np.nan,
                optimizer_stress=np.nan,
                optimizer_n_iter=0,
                dissimilarities=None,
                weight=None,
                frontier_pairs_weight=frontier_pairs_weight,
            ),
        )

        for v_alpha, cos_clip, emb_alpha in cases:
            dists = load_or_compute_dissimilarities(context, v_alpha, cos_clip)
            weight = make_weights(context, dists, frontier_pairs_weight)
            y_one = None
            for max_iter in ITERATIONS:
                method = gd_method_name()
                key = row_key(seed, method, v_alpha, cos_clip, emb_alpha, max_iter, np.nan)
                if key in done and not (max_iter == 1 and needs_interpolation(done, seed, v_alpha, cos_clip, emb_alpha)):
                    continue
                print(
                    f"seed={seed} v={v_alpha:g} clip={cos_clip:g} "
                    f"a={emb_alpha:g} iter={max_iter}",
                    flush=True,
                )
                embedding, stress, n_iter = run_gd(x_umap, dists, emb_alpha, max_iter, weight)
                if max_iter == 1:
                    y_one = embedding
                add_row_if_needed(
                    raw_csv,
                    done,
                    key,
                    lambda embedding=embedding, stress=stress, n_iter=n_iter, max_iter=max_iter: evaluate_run(
                        name=run_name("gd", seed, v_alpha, emb_alpha, max_iter),
                        method=method,
                        seed=seed,
                        embedding=embedding,
                        x_umap=x_umap,
                        context=context,
                        v_alpha=v_alpha,
                        cos_clip=cos_clip,
                        emb_alpha=emb_alpha,
                        max_iter=max_iter,
                        interp_t=np.nan,
                        optimizer_stress=stress,
                        optimizer_n_iter=n_iter,
                        dissimilarities=dists,
                        weight=weight,
                        frontier_pairs_weight=frontier_pairs_weight,
                    ),
                )

            if y_one is None and needs_interpolation(done, seed, v_alpha, cos_clip, emb_alpha):
                y_one, _, _ = run_gd(x_umap, dists, emb_alpha, 1, weight)
            if y_one is not None:
                for t in INTERPOLATION_T:
                    key = row_key(seed, "interp_umap_gd1", v_alpha, cos_clip, emb_alpha, 1, t)
                    add_row_if_needed(
                        raw_csv,
                        done,
                        key,
                        lambda t=t: evaluate_run(
                            name=run_name("interp", seed, v_alpha, emb_alpha, 1, t=t),
                            method="interp_umap_gd1",
                            seed=seed,
                            embedding=(1.0 - t) * x_umap + t * y_one,
                            x_umap=x_umap,
                            context=context,
                            v_alpha=v_alpha,
                            cos_clip=cos_clip,
                            emb_alpha=emb_alpha,
                            max_iter=1,
                            interp_t=t,
                            optimizer_stress=np.nan,
                            optimizer_n_iter=0,
                            dissimilarities=dists,
                            weight=weight,
                            frontier_pairs_weight=frontier_pairs_weight,
                        ),
                    )

    write_summary(raw_csv, summary_csv)
    print(f"\nSaved raw CSV: {raw_csv}")
    print(f"Saved summary CSV: {summary_csv}")


def output_paths(frontier_pairs_weight):
    init_tag = f"_iu{UMAP_INIT_DIM}" if EMBEDDING_DIM != UMAP_INIT_DIM else ""
    velocity_tag = velocity_distance_formula_tag(VELOCITY_DISTANCE_FORMULA)
    tag = f"_{EMBEDDING_DIM}d{init_tag}_{velocity_tag}_{metric_short_name()}"
    if float(frontier_pairs_weight) != 1.0:
        tag += f"_afpw{cache_token(frontier_pairs_weight)}"
    return (
        OUT_DIR / f"pancreas_umap_one_step_refinement{tag}_raw.csv",
        OUT_DIR / f"pancreas_umap_one_step_refinement{tag}_summary.csv",
    )


def alpha_cases():
    cases = [BASELINE_CASE]
    cases.extend(
        (v_alpha, cos_clip, emb_alpha)
        for v_alpha, cos_clip in V_ALPHA_COS_CLIP.items()
        for emb_alpha in EMB_ALPHA_GRID
    )
    return cases


def load_or_compute_umap(context, seed):
    if seed == 42:
        path = RAW_DIR / "umap_dynamical_s42.npy"
        if path.exists():
            return as_embedding_dim(np.load(path))

    import scanpy as sc

    adata = context.adata.copy()
    sc.pp.neighbors(
        adata,
        n_neighbors=PANCREAS_UMAP["n_neighbors"],
        n_pcs=PANCREAS_PREPROCESSING["n_pcs"],
        random_state=seed,
    )
    embedding = compute_umap_from_neighbors(
        adata,
        PANCREAS_UMAP,
        n_components=2,
        random_state=seed,
    )
    velocity_embedding = project_velocity_to_embedding(adata, embedding)
    oriented, _, _ = utils.rotate_embedding_to_mean_velocity_down(
        embedding,
        velocity_embedding,
    )
    return as_embedding_dim(oriented)


def as_embedding_dim(embedding):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] == EMBEDDING_DIM:
        return embedding
    if embedding.shape[1] == 2 and EMBEDDING_DIM == 3:
        return np.column_stack([embedding, np.zeros(len(embedding))])
    raise ValueError(f"Cannot use a {embedding.shape[1]}D UMAP init for EMBEDDING_DIM={EMBEDDING_DIM}.")


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
    np.savez(
        path,
        dists_velocity=dists,
        labels=labels_to_cache(context.labels),
    )
    return np.asarray(dists, dtype=float)


def make_weights(context, dists, frontier_pairs_weight):
    weights = make_pair_weights(
        context.labels,
        dists,
        cluster_reweight_rho=CLUSTER_REWEIGHT_RHO,
        frontier_pairs_weight=frontier_pairs_weight,
        selected_frontiers=SELECTED_FRONTIERS,
        distance_reweighting={
            "power": DISTANCE_REWEIGHT_POWER,
            "epsilon": 1e-6,
        },
        eval_raw_dir=EVAL_DIR / "raw",
        neighbor_indices=context.expression_neighbors,
        n_neighbors=PANCREAS_N_EVAL_NEIGHBORS,
    )
    if weights is not None:
        return weights
    weights = np.ones_like(dists, dtype=float)
    np.fill_diagonal(weights, 0.0)
    return normalize_pair_weights(weights, dists)[0]


def run_gd(init, dists, emb_alpha, max_iter, weight):
    options = dict(PANCREAS_GD_OPTIONS)
    optimizer_options = dict(options.pop("optimizer_options"))
    options.update(
        {
            "max_iter": int(max_iter),
            "optimizer_options": optimizer_options,
            "verbose": 0,
            "print_time": False,
            "return_n_iter": True,
            "init": init,
            "n_components": init.shape[1],
            "random_state": 42,
            "weight": weight,
        }
    )
    return fit_finsler_mds(
        dists,
        metric=make_embedding_metric(emb_alpha),
        optimizer="gradient_descent",
        **options,
    )


def evaluate_run(
        *,
        name,
        method,
        seed,
        embedding,
        x_umap,
        context,
        v_alpha,
        cos_clip,
        emb_alpha,
        max_iter,
        interp_t,
        optimizer_stress,
        optimizer_n_iter,
        dissimilarities,
        weight,
        frontier_pairs_weight,
):
    row = evaluate_embedding(
        name=name,
        kind=method,
        embedding=embedding,
        context=context,
        metric=make_embedding_metric(0.0 if np.isnan(emb_alpha) else emb_alpha),
        dissimilarities=dissimilarities,
        weight=weight,
    )
    row.update(
        {
            "seed": int(seed),
            "method": method,
            "velocity_distance_formula": normalize_velocity_distance_formula(VELOCITY_DISTANCE_FORMULA),
            "embedding_metric": EMBEDDING_METRIC.lower(),
            "embedding_dim": int(embedding.shape[1]),
            "init_embedding_dim": UMAP_INIT_DIM,
            "v_alpha": v_alpha,
            "cos_clip": cos_clip,
            "emb_alpha": emb_alpha,
            "max_iter": int(max_iter),
            "interp_t": interp_t,
            "cluster_reweight_rho": CLUSTER_REWEIGHT_RHO,
            "frontier_pairs_weight": frontier_pairs_weight,
            "selected_frontiers": SELECTED_FRONTIERS,
            "distance_reweight_power": DISTANCE_REWEIGHT_POWER,
            "optimizer_stress": optimizer_stress,
            "optimizer_n_iter": int(optimizer_n_iter),
            "direct_stress_normalized": normalized_direct_stress(row, dissimilarities, weight),
            "rms_to_umap": rms_distance(embedding, x_umap),
            "procrustes_to_umap": procrustes_distance(embedding, x_umap),
            f"knn_overlap_umap_k{N_KNN_OVERLAP}": knn_overlap(embedding, x_umap, k=N_KNN_OVERLAP),
        }
    )
    return row


def normalized_direct_stress(row, dissimilarities, weight):
    if dissimilarities is None or weight is None or "direct_weighted_stress" not in row:
        return np.nan
    active = weight != 0
    denom = float(np.sum(weight[active] * dissimilarities[active] ** 2))
    return float(row["direct_weighted_stress"]) / denom if denom > 0 else np.nan


def rms_distance(embedding, reference):
    return float(np.sqrt(np.mean(np.sum((embedding - reference) ** 2, axis=1))))


def procrustes_distance(embedding, reference):
    x = center_scale(embedding)
    y = center_scale(reference)
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    x_aligned = x @ (u @ vt)
    return float(np.sqrt(np.mean(np.sum((x_aligned - y) ** 2, axis=1))))


def center_scale(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
    return x / scale if scale > 0 else x


def knn_overlap(embedding, reference, *, k):
    from sklearn.neighbors import NearestNeighbors

    k = int(k)
    idx_a = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)[:, 1:]
    idx_b = NearestNeighbors(n_neighbors=k + 1).fit(reference).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a).intersection(b)) / k for a, b in zip(idx_a, idx_b)]))


def existing_keys(path):
    return {
        row_key(
            int(row["seed"]),
            row["method"],
            parse_float(row.get("v_alpha", np.nan)),
            parse_float(row.get("cos_clip", np.nan)),
            parse_float(row.get("emb_alpha", np.nan)),
            int(float(row.get("max_iter", 0))),
            parse_float(row.get("interp_t", np.nan)),
            velocity_distance_formula=row_velocity_distance_formula(row),
            embedding_metric=row.get("embedding_metric", EMBEDDING_METRIC),
            embedding_dim=int(float(row.get("embedding_dim", EMBEDDING_DIM))),
            init_embedding_dim=int(float(row.get("init_embedding_dim", UMAP_INIT_DIM))),
        )
        for row in read_csv_rows(path)
    }


def row_key(
        seed,
        method,
        v_alpha,
        cos_clip,
        emb_alpha,
        max_iter,
        interp_t,
        *,
        velocity_distance_formula=None,
        embedding_metric=None,
        embedding_dim=None,
        init_embedding_dim=None,
):
    return (
        int(seed),
        str(method),
        normalize_velocity_distance_formula(velocity_distance_formula or VELOCITY_DISTANCE_FORMULA),
        str(embedding_metric or EMBEDDING_METRIC).lower(),
        int(embedding_dim or EMBEDDING_DIM),
        int(init_embedding_dim or UMAP_INIT_DIM),
        token_or_nan(v_alpha),
        token_or_nan(cos_clip),
        token_or_nan(emb_alpha),
        int(max_iter),
        token_or_nan(interp_t),
    )


def add_row_if_needed(raw_csv, done, key, row_factory):
    if key in done and not OVERWRITE:
        return
    row = row_factory()
    append_csv_row(raw_csv, row)
    done.add(key)
    print(
        f"{row['name']}: CBDir={row['cbdir']:.4f}, ICVCoh={row['icvcoh']:.4f}, "
        f"GVCoh={row['gvcoh']:.4f}, Orient={row['spearman_cos']:.4f}, "
        f"Sign={row['sign_correctness']:.4f}",
        flush=True,
    )


def row_velocity_distance_formula(row):
    return normalize_velocity_distance_formula(row.get("velocity_distance_formula") or "randers")


def needs_interpolation(done, seed, v_alpha, cos_clip, emb_alpha):
    return any(
        row_key(seed, "interp_umap_gd1", v_alpha, cos_clip, emb_alpha, 1, t) not in done
        for t in INTERPOLATION_T
    )


def write_summary(raw_csv, summary_csv):
    metric_keys = [
        "cbdir",
        "icvcoh",
        "gvcoh",
        "spearman_cos",
        "sign_correctness",
        "direct_weighted_stress",
        "direct_stress_normalized",
        "rms_to_umap",
        "procrustes_to_umap",
        f"knn_overlap_umap_k{N_KNN_OVERLAP}",
    ]
    summarize_csv_by_seed(
        raw_csv,
        summary_csv,
        group_keys=[
            "method",
            "velocity_distance_formula",
            "embedding_metric",
            "embedding_dim",
            "init_embedding_dim",
            "v_alpha",
            "cos_clip",
            "emb_alpha",
            "max_iter",
            "interp_t",
            "frontier_pairs_weight",
            "selected_frontiers",
        ],
        metric_keys=metric_keys,
    )


def run_name(prefix, seed, v_alpha, emb_alpha, max_iter, *, t=np.nan):
    velocity_tag = velocity_distance_formula_tag(VELOCITY_DISTANCE_FORMULA, alpha=v_alpha)
    name = (
        f"{prefix}_{EMBEDDING_DIM}d_{velocity_tag}_{metric_short_name()}{cache_token(emb_alpha)}"
        f"_i{max_iter}_s{seed}"
    )
    if not np.isnan(t):
        name += f"_t{cache_token(t)}"
    return name


def make_embedding_metric(alpha):
    metric = EMBEDDING_METRIC.lower()
    if metric == "randers":
        return RandersMetric(alpha=alpha)
    if metric == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    raise ValueError("EMBEDDING_METRIC must be one of {'randers', 'matsumoto'}.")


def metric_short_name():
    return {"randers": "r", "matsumoto": "mats"}[EMBEDDING_METRIC.lower()]


def gd_method_name():
    return f"gd_{EMBEDDING_METRIC.lower()}"


def token_or_nan(value):
    value = parse_float(value)
    return "nan" if np.isnan(value) else cache_token(value)


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


if __name__ == "__main__":
    main()
