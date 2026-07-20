from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import fit_finsler_mds
from finsler_mds.utils.pancreas_files import resolve_pancreas_embedding_path
from scripts.evaluate_pancreas_embedding import (
    N_EVAL_NEIGHBORS,
    append_csv,
    evaluate_embedding,
    load_embedding,
    load_pancreas_evaluation_context,
    load_velocity_dissimilarities,
)
from scripts.main_pancreas import make_embedding_metric


SEED = 42
VELOCITY_ALPHA = 1.0
VELOCITY_COS_CLIP = 0.99
EMBEDDING_ALPHA = 0.8
TARGETS_PER_LANDMARK = 1000
RANDOM_LANDMARK_FRACTION = 0.9

PANCREAS_DIR = SCRIPT_DIR / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
EVAL_DIR = PANCREAS_DIR / "rna_velocity_evaluation"
OUT_DIR = PANCREAS_DIR / "path_frozen_reproduce_cbdir"
CSV_PATH = EVAL_DIR / "pancreas_pf_reproduce_cbdir_campaign.csv"

BASE_OPTIONS = {
    "graph_neighbors": 30,
    "eps": 1e-6,
    "method": "L-BFGS-B",
    "optimizer_options": {"ftol": 1e-9, "maxls": 50},
    "random_landmark_fraction": RANDOM_LANDMARK_FRACTION,
    "resample_random_landmarks": True,
    "n_local_pairs": 30,
    "local_pair_mode": "direct",
    "targets_per_landmark": TARGETS_PER_LANDMARK,
    "local_global_reweighting": "count",
    "direct_stress_weight": 0.0,
    "device": "auto",
    "verbose": 0,
    "record_history": False,
}


ROUNDS = {
    "mats_quick": [
        {"name": "q_current", "s1": (10, 30, 400), "s2": (8, 3, 400), "lw": 1, "step": 1.0},
        {"name": "q_old_r_like", "s1": (5, 50, 800), "s2": (10, 3, 400), "lw": 10, "step": 1.0},
        {"name": "q_old_m_like", "s1": (5, 50, 800), "s2": (10, 3, 800), "lw": 10, "step": 1.0},
        {"name": "q_more_outer", "s1": (15, 10, 400), "s2": (10, 3, 400), "lw": 1, "step": 1.0},
        {"name": "q_many_lm", "s1": (10, 20, 800), "s2": (10, 3, 800), "lw": 1, "step": 1.0},
        {"name": "q_soft_step", "s1": (20, 10, 400), "s2": (10, 3, 400), "lw": 1, "step": 0.3},
    ],
    "mats_medium": [
        {"name": "m_lw1_lm800", "s1": (20, 20, 800), "s2": (15, 3, 800), "lw": 1, "step": 1.0},
        {"name": "m_lw3_lm800", "s1": (20, 20, 800), "s2": (15, 3, 800), "lw": 3, "step": 1.0},
        {"name": "m_lw10_lm800", "s1": (20, 20, 800), "s2": (15, 3, 800), "lw": 10, "step": 1.0},
        {"name": "m_lm1200_lw3", "s1": (15, 20, 1200), "s2": (12, 3, 1200), "lw": 3, "step": 1.0},
        {"name": "m_step03_lw3", "s1": (30, 10, 800), "s2": (15, 3, 800), "lw": 3, "step": 0.3},
        {"name": "m_few_targets", "s1": (25, 10, 800), "s2": (15, 3, 800), "lw": 3, "step": 0.5, "targets": 500},
        {"name": "m_all_targets", "s1": (15, 20, 800), "s2": (10, 3, 800), "lw": 3, "step": 1.0, "targets": None},
    ],
    "mats_focus": [
        {"name": "f_step02", "s1": (20, 10, 400), "s2": (10, 3, 400), "lw": 1, "step": 0.2},
        {"name": "f_step05", "s1": (20, 10, 400), "s2": (10, 3, 400), "lw": 1, "step": 0.5},
        {"name": "f_more_s1", "s1": (30, 10, 400), "s2": (10, 3, 400), "lw": 1, "step": 0.3},
        {"name": "f_more_finish", "s1": (20, 10, 400), "s2": (20, 3, 400), "lw": 1, "step": 0.3},
        {"name": "f_lm600", "s1": (20, 10, 600), "s2": (10, 3, 600), "lw": 1, "step": 0.3},
        {"name": "f_lw3", "s1": (20, 10, 400), "s2": (10, 3, 400), "lw": 3, "step": 0.3},
        {"name": "f_inner5_long", "s1": (35, 5, 400), "s2": (10, 3, 400), "lw": 1, "step": 0.3},
    ],
    "randers_quick": [
        {"name": "r_current", "s1": (10, 30, 400), "s2": (8, 3, 400), "lw": 1, "step": 1.0},
        {"name": "r_old_like", "s1": (5, 50, 800), "s2": (10, 3, 400), "lw": 10, "step": 1.0},
        {"name": "r_old_more_lm", "s1": (5, 50, 800), "s2": (10, 3, 800), "lw": 10, "step": 1.0},
        {"name": "r_many_lm", "s1": (10, 20, 800), "s2": (10, 3, 800), "lw": 1, "step": 1.0},
        {"name": "r_step03", "s1": (20, 10, 400), "s2": (10, 3, 400), "lw": 1, "step": 0.3},
    ],
}


def main() -> None:
    args = parse_args()
    metric_kind = args.metric
    init_name = f"gd_2d_vrand1_{'mats0p8' if metric_kind == 'matsumoto' else 'r0p8'}_s42.npz"
    init = load_embedding(resolve_pancreas_embedding_path(init_name, RAW_DIR))
    D, labels = load_velocity_dissimilarities(
        RAW_DIR,
        velocity_alpha=VELOCITY_ALPHA,
        distance_formula="randers",
        cos_clip=VELOCITY_COS_CLIP,
        kNN_euclid=30,
        kNN_finsler=0,
    )
    context = load_pancreas_evaluation_context(RAW_DIR, EVAL_DIR, n_eval_neighbors=N_EVAL_NEIGHBORS)
    if not np.array_equal(context.labels, labels):
        raise ValueError("Velocity cache labels do not match evaluation context labels.")
    metric = make_embedding_metric({"kind": metric_kind, "alpha": EMBEDDING_ALPHA})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = ROUNDS[args.round]
    done = existing_keys() if not args.overwrite else set()
    for cfg in configs:
        key = (args.round, metric_kind, cfg["name"])
        if key in done:
            print(f"Skipping existing {key}", flush=True)
            continue
        row = run_config(cfg, args.round, metric_kind, metric, D, init, context)
        save_embedding(row, cfg, metric_kind)
        append_csv(CSV_PATH, row)
        print(
            f"{row['run_name']}: CBDir={row['cbdir']:.4f}, "
            f"ICVCoh={row['icvcoh']:.4f}, VAC={row['spearman_cos']:.4f}, "
            f"VAS={row['sign_correctness']:.4f}, GVCoh={row['gvcoh']:.4f}, "
            f"time={row['wall_time']:.1f}s",
            flush=True,
        )
    print(f"Saved CSV: {CSV_PATH}", flush=True)


def run_config(cfg, round_name, metric_kind, metric, D, init, context):
    targets = cfg.get("targets", TARGETS_PER_LANDMARK)
    s1_outer, s1_inner, s1_landmarks = cfg["s1"]
    s2_outer, s2_inner, s2_landmarks = cfg["s2"]
    common = {
        **BASE_OPTIONS,
        "local_weight": cfg["lw"],
        "outer_step_size": cfg["step"],
        "targets_per_landmark": targets,
        "random_state": SEED,
        "mask_random_state": SEED,
        "target_random_state": SEED + 1,
        "n_components": 2,
        "return_result": True,
    }
    start = perf_counter()
    stage1 = fit_finsler_mds(
        D,
        metric=metric,
        optimizer="path_frozen",
        init=init,
        outer_iter=s1_outer,
        inner_iter=s1_inner,
        n_landmark=s1_landmarks,
        **common,
    )
    stage2 = fit_finsler_mds(
        D,
        metric=metric,
        optimizer="path_frozen",
        init=stage1.embedding,
        outer_iter=s2_outer,
        inner_iter=s2_inner,
        n_landmark=s2_landmarks,
        **common,
    )
    wall_time = perf_counter() - start
    row = evaluate_embedding(
        name=f"pf_repro_{metric_kind}_{round_name}_{cfg['name']}",
        kind="path_frozen",
        embedding=stage2.embedding,
        context=context,
    )
    row.update({
        "round": round_name,
        "run_name": cfg["name"],
        "metric": metric_kind,
        "velocity_alpha": VELOCITY_ALPHA,
        "velocity_cos_clip": VELOCITY_COS_CLIP,
        "alpha_embedding": EMBEDDING_ALPHA,
        "stage1_outer_iter": s1_outer,
        "stage1_inner_iter": s1_inner,
        "stage1_n_landmark": s1_landmarks,
        "stage2_outer_iter": s2_outer,
        "stage2_inner_iter": s2_inner,
        "stage2_n_landmark": s2_landmarks,
        "targets_per_landmark": "all" if targets is None else targets,
        "local_weight": cfg["lw"],
        "outer_step_size": cfg["step"],
        "random_landmark_fraction": RANDOM_LANDMARK_FRACTION,
        "resample_random_landmarks": True,
        "stage1_masked_stress": float(stage1.stress),
        "stage2_masked_stress": float(stage2.stress),
        "stage1_n_iter": int(stage1.n_iter),
        "stage2_n_iter": int(stage2.n_iter),
        "wall_time": wall_time,
    })
    row["_embedding"] = stage2.embedding
    return row


def save_embedding(row, cfg, metric_kind):
    embedding = row.pop("_embedding")
    path = OUT_DIR / f"{row['name']}.npz"
    metadata = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))}
    np.savez_compressed(path, embedding=embedding, metadata_json=json.dumps(metadata, sort_keys=True))
    row["embedding_path"] = str(path)


def existing_keys():
    if not CSV_PATH.exists():
        return set()
    keys = set()
    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row.get("round"), row.get("metric"), row.get("run_name")))
    return keys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=sorted(ROUNDS), default="mats_quick")
    parser.add_argument("--metric", choices=["matsumoto", "randers"], default="matsumoto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
