"""Stronger Paul15 Monocle/MDS-geodesic tuning campaign.

This campaign deliberately moves farther from the Monocle UMAP initialization
than ``run_paul15_monocle_experiments.py``. Embeddings are still rescaled to the
UMAP coordinate scale before injection into Monocle, because Monocle's
``learn_graph`` is sensitive to absolute coordinate scale.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_paul15_monocle_experiments import run_variant, score_existing


def main():
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15" / "monocle3"
    dir_raw = dir_res / "raw"
    dir_exp = dir_res / "experiments"
    dir_exp.mkdir(parents=True, exist_ok=True)

    bridge_script = script_dir / "monocle3_bridge" / "run_paul15_monocle3.R"
    input_dir = dir_res / "monocle_input_no19lymph_no11dc"
    standard_monocle_path = dir_raw / "monocle_umap_pseudotime_no19lymph_no11dc.csv"
    standard_graph_prefix = dir_raw / "monocle_umap_principal_graph_no19lymph_no11dc"

    standard = pd.read_csv(standard_monocle_path)
    cell_ids = standard["cell_id"].astype(str).to_numpy()
    umap = standard[["dim1", "dim2"]].to_numpy(dtype=float)
    pt_umap = standard["pseudotime"].to_numpy(dtype=float)

    common = {
        "graph_neighbors": 20,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 30},
        "local_pair_mode": "direct",
        "global_target_sampling": "random",
        "local_global_reweighting": "count",
        "device": "auto",
        "verbose": 0,
    }
    variants = [
        # Around the best less-conservative result from the first campaign.
        v("dg0p5_g150_l50_m8", 0.5, common, max_iter=8, inner_iter=2, landmarks=150, targets=200, local_neighbors=50, local_weight=50),
        v("dg0p5_g150_l50_m12", 0.5, common, max_iter=12, inner_iter=2, landmarks=150, targets=200, local_neighbors=50, local_weight=50),
        v("dg0p5_g150_l30_m10", 0.5, common, max_iter=10, inner_iter=2, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        v("dg0p5_g150_l20_m10", 0.5, common, max_iter=10, inner_iter=2, landmarks=150, targets=200, local_neighbors=35, local_weight=20),
        # More global pressure, while keeping local constraints strong enough.
        v("dg0p5_g250_l50_m8", 0.5, common, max_iter=8, inner_iter=2, landmarks=250, targets=300, local_neighbors=50, local_weight=50),
        v("dg0p5_g250_l30_m10", 0.5, common, max_iter=10, inner_iter=2, landmarks=250, targets=300, local_neighbors=40, local_weight=30),
        v("dg0p5_g250_l20_m12", 0.5, common, max_iter=12, inner_iter=2, landmarks=250, targets=300, local_neighbors=35, local_weight=20),
        # Gamma sweep with medium global pressure.
        v("dg0p75_g150_l50_m8", 0.75, common, max_iter=8, inner_iter=2, landmarks=150, targets=200, local_neighbors=50, local_weight=50),
        v("dg0p75_g150_l30_m10", 0.75, common, max_iter=10, inner_iter=2, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        v("dg1_g150_l30_m10", 1.0, common, max_iter=10, inner_iter=2, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        v("dg1_g250_l20_m12", 1.0, common, max_iter=12, inner_iter=2, landmarks=250, targets=300, local_neighbors=35, local_weight=20),
        # Slightly longer inner loops, but not too many.
        v("dg0p5_g150_l30_m8_i4", 0.5, common, max_iter=8, inner_iter=4, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        v("dg0p75_g150_l30_m8_i4", 0.75, common, max_iter=8, inner_iter=4, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        # Denser embedding graph for path-frozen geodesics.
        v("dg0p5_g150_l30_m10_knn30", 0.5, {**common, "graph_neighbors": 30}, max_iter=10, inner_iter=2, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        v("dg0p75_g150_l30_m10_knn30", 0.75, {**common, "graph_neighbors": 30}, max_iter=10, inner_iter=2, landmarks=150, targets=200, local_neighbors=40, local_weight=30),
        # A more aggressive run to see when Monocle breaks.
        v("dg0p5_g300_l15_m16", 0.5, common, max_iter=16, inner_iter=2, landmarks=300, targets=350, local_neighbors=30, local_weight=15),
        v("dg0p75_g300_l15_m16", 0.75, common, max_iter=16, inner_iter=2, landmarks=300, targets=350, local_neighbors=30, local_weight=15),
    ]

    rows = [with_novelty(score_existing("umap_standard", standard_monocle_path, standard_graph_prefix, umap, pt_umap))]
    for variant in variants:
        row = run_variant(variant, dir_raw, dir_exp, input_dir, bridge_script, cell_ids, umap, pt_umap, seed)
        rows.append(with_novelty(row))
        summary = pd.DataFrame(rows)
        summary.to_csv(dir_exp / "summary.csv", index=False)
        print(
            summary.sort_values("novel_score", ascending=False)[
                [
                    "name",
                    "novel_score",
                    "score",
                    "pt_corr",
                    "nodes",
                    "roots",
                    "knn15",
                    "procrustes_rmse",
                    "jump99",
                    "density_gamma",
                    "outer_iter",
                    "inner_iter",
                    "local_weight",
                    "n_landmark",
                    "n_local_pairs",
                ]
            ].to_string(index=False)
        )


def v(name, gamma, common, *, max_iter, inner_iter, landmarks, targets, local_neighbors, local_weight):
    return {
        "name": name,
        "density_gamma": gamma,
        "path": {
            **common,
            "outer_iter": max_iter,
            "inner_iter": inner_iter,
            "n_landmark": landmarks,
            "targets_per_landmark": targets,
            "n_local_pairs": local_neighbors,
            "local_weight": float(local_weight),
        },
    }


def with_novelty(row):
    row = dict(row)
    pt = float(row["pt_corr"])
    nodes = float(row["nodes"])
    roots = float(row["roots"])
    knn = float(row["knn15"])
    proc = float(row["procrustes_rmse"])
    jump = float(row["jump99"])
    pt_scale = max(float(row["pt_q75"]), 1e-12)

    graph_penalty = 0.25 * abs(nodes - 45.0) / 45.0 + 0.2 * abs(roots - 1.0)
    jump_penalty = 0.08 * jump / pt_scale
    novelty_bonus = 3.0 * min(proc, 0.08) + 0.18 * max(0.0, 0.92 - knn)
    row["novel_score"] = pt + novelty_bonus - graph_penalty - jump_penalty
    return row


if __name__ == "__main__":
    main()
