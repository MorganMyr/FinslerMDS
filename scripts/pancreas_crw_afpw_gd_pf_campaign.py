from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.evaluation.rna_velocity import cross_boundary_direction_correctness
from finsler_mds.utils.pancreas import PANCREAS_TRANSITIONS, project_velocity_to_embedding
from scripts.evaluate_pancreas_embedding import (
    N_EVAL_NEIGHBORS,
    append_csv,
    evaluate_embedding,
    load_embedding,
    load_pancreas_evaluation_context,
    load_velocity_dissimilarities,
    make_pair_weights,
)
from scripts.main_pancreas import (
    cache_token,
    embedding_metric_tag,
    finsler_embedding_dim_tag,
    make_embedding_metric,
    main_pancreas,
    pair_weight_output_tag,
    velocity_distance_formula_tag,
)

SEED = 42
DISTANCE_REWEIGHTING = {"power": 0.0, "epsilon": 1e-6}
SELECTED_FRONTIERS = "all"
CRW_VALUES = [0.0, 0.5, 1.0]
AFPW_VALUES = [1.0, 20.0, 100.0]
SKIP_BASE_WEIGHT = True
OVERWRITE_CSV = False

TESTS = [
    {"label": "v2_a0p6_3d_mats", "v_alpha": 2.0, "alpha": 0.6, "dim": 3, "metric": "matsumoto"},
]

GD_OPTIONS = {"max_iter": 100}
PF_STAGE1 = {"outer_iter": 25, "inner_iter": 30}
PF_FINISHER = {"outer_iter": 15, "inner_iter": 3}

PANCREAS_DIR = SCRIPT_DIR / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
FIG_DIR = PANCREAS_DIR
EVAL_DIR = PANCREAS_DIR / "rna_velocity_evaluation"
CSV_PATH = EVAL_DIR / "pancreas_v1_a0p8_2d_crw_afpw_gd_pf_drw0_campaign.csv"


def main():
    if OVERWRITE_CSV and CSV_PATH.exists():
        CSV_PATH.unlink()
    context = None
    for test in TESTS:
        for crw in CRW_VALUES:
            for afpw in AFPW_VALUES:
                if SKIP_BASE_WEIGHT and np.isclose(crw, 0.0) and np.isclose(afpw, 1.0):
                    continue
                print(f"\n=== {test['label']} crw={crw:g} afpw={afpw:g} ===", flush=True)
                gd_npz, gd_fig = expected_paths(test, crw, afpw, optimizer="gd")
                pf_npz, pf_fig = expected_paths(test, crw, afpw, optimizer="pf")

                run_main(test, crw, afpw, optimizer="gradient_descent", init="umap_2D", extra={"gradient_descent": GD_OPTIONS})
                require_file(gd_npz)
                if context is None:
                    context = load_pancreas_evaluation_context(RAW_DIR, EVAL_DIR, n_eval_neighbors=N_EVAL_NEIGHBORS)
                append_stage_row("gd", gd_npz, test, crw, afpw, context)

                run_main(test, crw, afpw, optimizer="path_frozen", init=gd_npz.name, extra={"path_frozen": PF_STAGE1})
                require_file(pf_npz)
                stage1_npz = copy_with_suffix(pf_npz, "stage1")
                stage1_fig = copy_with_suffix(pf_fig, "stage1") if pf_fig.exists() else None
                append_stage_row("pf_stage1", stage1_npz, test, crw, afpw, context, figure_path=stage1_fig)

                run_main(test, crw, afpw, optimizer="path_frozen", init=stage1_npz.name, extra={"path_frozen": PF_FINISHER})
                require_file(pf_npz)
                append_stage_row("pf_final", pf_npz, test, crw, afpw, context)
                plt.close("all")

    print(f"\nSaved campaign CSV: {CSV_PATH}")


def run_main(test, crw, afpw, *, optimizer, init, extra):
    overrides = {
        "finsler_optimizer": optimizer,
        "init_finsler_mds": init,
        "embedding_dim": test["dim"],
        "finsler_metric": test["metric"],
        "alpha_embedding": test["alpha"],
        "cluster_reweight_rho": crw,
        "frontier_pairs_weight": afpw,
        "selected_frontiers": SELECTED_FRONTIERS,
        "distance_reweighting": DISTANCE_REWEIGHTING,
        "velocity": {"alpha": test["v_alpha"]},
    }
    overrides.update(extra)
    main_pancreas(overrides)
    plt.close("all")


def expected_paths(test, crw, afpw, *, optimizer):
    metric = make_embedding_metric({"kind": test["metric"], "alpha": test["alpha"]})
    metric_tag = embedding_metric_tag(metric)
    velocity_tag = velocity_distance_formula_tag("randers", test["v_alpha"])
    dim_tag = finsler_embedding_dim_tag(test["dim"])
    weight_tag = pair_weight_output_tag(
        crw,
        afpw,
        selected_frontiers=SELECTED_FRONTIERS,
        distance_reweighting=DISTANCE_REWEIGHTING,
    )
    raw_path = RAW_DIR / f"{optimizer}_{dim_tag}{velocity_tag}_{metric_tag}{weight_tag}_s{SEED}.npz"
    fig_path = FIG_DIR / f"{optimizer}_{dim_tag}{velocity_tag}_{metric_tag}{weight_tag}.pdf"
    return raw_path, fig_path


def copy_with_suffix(path, suffix):
    path = Path(path)
    target = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    shutil.copy2(path, target)
    return target


def append_stage_row(stage, embedding_path, test, crw, afpw, context, figure_path=None):
    if stage_row_exists(stage, test, crw, afpw, embedding_path):
        print(f"{stage}: row already present, skipping append", flush=True)
        return

    embedding = load_embedding(embedding_path)
    dists, labels = load_velocity_dissimilarities(
        RAW_DIR,
        velocity_alpha=test["v_alpha"],
        distance_formula="randers",
        cos_clip=0.4,
        kNN_euclid=30,
        kNN_finsler=0,
    )
    if not np.array_equal(context.labels, labels):
        raise ValueError("Velocity cache labels do not match evaluation context labels.")
    weight = make_pair_weights(
        context.labels,
        dists,
        cluster_reweight_rho=crw,
        frontier_pairs_weight=afpw,
        selected_frontiers=SELECTED_FRONTIERS,
        distance_reweighting=DISTANCE_REWEIGHTING,
        eval_raw_dir=EVAL_DIR / "raw",
        neighbor_indices=context.expression_neighbors,
        n_neighbors=context.n_eval_neighbors,
    )
    metric = make_embedding_metric({"kind": test["metric"], "alpha": test["alpha"]})
    row = evaluate_embedding(
        name=embedding_path.stem,
        kind=stage,
        embedding=embedding,
        context=context,
        metric=metric,
        dissimilarities=dists,
        weight=weight,
    )
    row.update({
        "stage": stage,
        "test": test["label"],
        "optimizer": "gradient_descent" if stage == "gd" else "path_frozen",
        "v_alpha": test["v_alpha"],
        "alpha_embedding": test["alpha"],
        "embedding_metric": test["metric"],
        "cluster_reweight_rho": crw,
        "frontier_pairs_weight": afpw,
        "selected_frontiers": SELECTED_FRONTIERS,
        "distance_reweight_power": DISTANCE_REWEIGHTING["power"],
        "distance_reweight_epsilon": DISTANCE_REWEIGHTING["epsilon"],
        "embedding_path": str(embedding_path),
        "figure_path": str(figure_path or expected_paths(test, crw, afpw, optimizer="gd" if stage == "gd" else "pf")[1]),
        "optimizer_saved_stress": saved_stress(embedding_path),
    })
    row.update(cbdir_breakdown(embedding, context))
    append_csv(CSV_PATH, row)
    print(
        f"{stage}: CBDir={row['cbdir']:.4f}, ICVCoh={row['icvcoh']:.4f}, "
        f"Orient={row['spearman_cos']:.4f}, Sign={row['sign_correctness']:.4f}",
        flush=True,
    )


def stage_row_exists(stage, test, crw, afpw, embedding_path):
    if not CSV_PATH.exists():
        return False
    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("stage") == stage
                and row.get("test") == test["label"]
                and np.isclose(float(row.get("cluster_reweight_rho", "nan")), crw)
                and np.isclose(float(row.get("frontier_pairs_weight", "nan")), afpw)
                and np.isclose(float(row.get("distance_reweight_power", "nan")), DISTANCE_REWEIGHTING["power"])
                and Path(row.get("embedding_path", "")).name == Path(embedding_path).name
            ):
                return True
    return False


def cbdir_breakdown(embedding, context):
    velocity_embedding = project_velocity_to_embedding(context.adata, embedding)
    result = cross_boundary_direction_correctness(
        embedding,
        context.labels,
        PANCREAS_TRANSITIONS,
        velocity_vectors=velocity_embedding,
        neighbor_indices=context.expression_neighbors,
        boundary_plan=context.cbdir_plan,
        n_neighbors=context.n_eval_neighbors,
    )
    row = {}
    by_source = {}
    for edge, score in result.transitions.items():
        source, target = edge
        key = f"cbdir_{safe_name(source)}_to_{safe_name(target)}"
        row[key] = float(score.score)
        row[f"{key}_n_cells"] = int(score.n_boundary_cells)
        by_source.setdefault(source, []).append(float(score.score))
    for source, values in by_source.items():
        row[f"cbdir_source_{safe_name(source)}"] = float(np.nanmean(values))
    return row


def saved_stress(path):
    try:
        with np.load(path, allow_pickle=False) as cache:
            if "stress" in cache:
                return float(np.asarray(cache["stress"]).reshape(-1)[0])
    except Exception:
        pass
    return np.nan


def safe_name(value):
    return str(value).replace(" ", "_").replace("-", "_").replace("/", "_")


def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Expected output was not created: {path}")


if __name__ == "__main__":
    main()
