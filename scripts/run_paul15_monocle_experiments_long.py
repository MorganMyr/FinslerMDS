"""Longer Paul15 Monocle/MDS-geodesic tuning runs.

These runs intentionally use more path-frozen outer iterations than the earlier
campaigns. They are kept in a separate output directory so they do not overwrite
the conservative/main figures or the shorter tuning campaigns.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_paul15_monocle_experiments import run_variant, score_existing
from run_paul15_monocle_experiments_strong import v, with_novelty


def main():
    seed = 42
    dir_res = SCRIPT_DIR / "res" / "paul15" / "monocle3"
    dir_raw = dir_res / "raw"
    dir_exp = dir_res / "experiments"
    dir_exp.mkdir(parents=True, exist_ok=True)

    bridge_script = SCRIPT_DIR / "monocle3_bridge" / "run_paul15_monocle3.R"
    input_dir = dir_res / "monocle_input_no19lymph_no11dc"
    standard_monocle_path = dir_raw / "monocle_umap_pseudotime_no19lymph_no11dc.csv"
    standard_graph_prefix = dir_raw / "monocle_umap_principal_graph_no19lymph_no11dc"

    standard = pd.read_csv(standard_monocle_path)
    cell_ids = standard["cell_id"].astype(str).to_numpy()
    umap = standard[["dim1", "dim2"]].to_numpy(dtype=float)
    pt_umap = standard["pseudotime"].to_numpy(dtype=float)

    common = {
        "graph_neighbors": 30,
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
        v("dg0p65_g150_l35_m30_knn30", 0.65, common, max_iter=30, inner_iter=2, landmarks=150, targets=200, local_neighbors=45, local_weight=35),
        v("dg0p65_g150_l35_m50_knn30", 0.65, common, max_iter=50, inner_iter=2, landmarks=150, targets=200, local_neighbors=45, local_weight=35),
        v("dg0p65_g150_l25_m50_knn30", 0.65, common, max_iter=50, inner_iter=2, landmarks=150, targets=200, local_neighbors=35, local_weight=25),
        v("dg0p75_g150_l25_m50_knn30", 0.75, common, max_iter=50, inner_iter=2, landmarks=150, targets=200, local_neighbors=35, local_weight=25),
        v("dg0p65_g200_l30_m40_knn30", 0.65, common, max_iter=40, inner_iter=2, landmarks=200, targets=250, local_neighbors=40, local_weight=30),
    ]

    rows = [
        with_novelty(
            score_existing(
                "umap_standard",
                standard_monocle_path,
                standard_graph_prefix,
                umap,
                pt_umap,
            )
        )
    ]
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


if __name__ == "__main__":
    main()
