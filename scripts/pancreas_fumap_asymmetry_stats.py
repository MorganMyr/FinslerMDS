"""Measure Finsler-UMAP fuzzy-weight asymmetry on pancreas caches."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.utils.umap import directed_fuzzy_graph_from_dense  # noqa: E402
from scripts.main_pancreas import cache_token  # noqa: E402


SEED = 42
N_NEIGHBORS = 50
KNN_EUCLID = 30
KNN_FINSLER = 0
EPS = 1e-12

V_ALPHA_COS_CLIP = {
    0.0: 1.0,
    0.25: 1.0,
    0.5: 1.0,
    1.0: 0.99,
}
CONFIGS_TO_COMPARE = (
    {
        "name": "Symmetrized except rho",
        "symmetrize_support": True,
        "symmetrize_rho": False,
        "symmetrize_sigma": True,
    },
    {
        "name": "Not symmetrized",
        "symmetrize_support": False,
        "symmetrize_rho": False,
        "symmetrize_sigma": False,
    },
)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "res" / "pancreas" / "raw"
OUT_PATH = SCRIPT_DIR / "res" / "pancreas" / "rna_velocity_evaluation" / "fumap_fuzzy_asymmetry_stats.csv"


def main() -> None:
    rows = []
    for v_alpha, cos_clip in V_ALPHA_COS_CLIP.items():
        dists = load_dissimilarities(v_alpha, cos_clip)
        for config in CONFIGS_TO_COMPARE:
            graph = directed_fuzzy_graph_from_dense(
                dists,
                N_NEIGHBORS,
                symmetrize_support=config["symmetrize_support"],
                symmetrize_rho=config["symmetrize_rho"],
                symmetrize_sigma=config["symmetrize_sigma"],
            )
            asym = relative_asymmetry(graph)
            row = summarize(asym)
            row.update(
                {
                    "config": config["name"],
                    "v_alpha": v_alpha,
                    "cos_clip": cos_clip,
                    "symmetrize_support": config["symmetrize_support"],
                    "symmetrize_rho": config["symmetrize_rho"],
                    "symmetrize_sigma": config["symmetrize_sigma"],
                    "n_pairs": len(asym),
                    "n_neighbors": N_NEIGHBORS,
                }
            )
            rows.append(row)
            print_row(row)
    write_csv(OUT_PATH, rows)
    print(f"Saved: {OUT_PATH}")


def load_dissimilarities(v_alpha, cos_clip):
    path = RAW_DIR / (
        "pancreas_velocity_inputs_dynamical_vrand_"
        f"valpha{cache_token(v_alpha)}_"
        f"cclip{cache_token(cos_clip)}_"
        f"ke{KNN_EUCLID}_kf{KNN_FINSLER}_s{SEED}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}")
    with np.load(path, allow_pickle=False) as cache:
        return np.asarray(cache["dists_velocity"], dtype=float)


def relative_asymmetry(graph):
    weights = {(int(i), int(j)): float(p) for i, j, p in zip(graph.row, graph.col, graph.probability)}
    asym = []
    for i, j in sorted({tuple(sorted(pair)) for pair in weights}):
        p_ij = weights.get((i, j), 0.0)
        p_ji = weights.get((j, i), 0.0)
        asym.append((p_ij - p_ji) / (p_ij + p_ji + EPS))
    return np.asarray(asym, dtype=float)


def summarize(asym):
    abs_asym = np.abs(asym)
    return {
        "mean": float(np.mean(asym)),
        "mean_abs": float(np.mean(abs_asym)),
        "median_abs": float(np.median(abs_asym)),
        "p90_abs": float(np.percentile(abs_asym, 90)),
        "p95_abs": float(np.percentile(abs_asym, 95)),
        "p99_abs": float(np.percentile(abs_asym, 99)),
        "frac_abs_gt_0p25": float(np.mean(abs_asym > 0.25)),
        "frac_abs_gt_0p5": float(np.mean(abs_asym > 0.5)),
        "frac_abs_gt_0p75": float(np.mean(abs_asym > 0.75)),
    }


def print_row(row):
    print(
        "{config}: v={v_alpha:g} clip={cos_clip:g} "
        "mean|A|={mean_abs:.3f}, med|A|={median_abs:.3f}, "
        "p90|A|={p90_abs:.3f}, p95|A|={p95_abs:.3f}, "
        "frac>|0.5|={frac_abs_gt_0p5:.3f}".format(**row)
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "config",
        "v_alpha",
        "cos_clip",
        "symmetrize_support",
        "symmetrize_rho",
        "symmetrize_sigma",
        "n_pairs",
        "n_neighbors",
        "mean",
        "mean_abs",
        "median_abs",
        "p90_abs",
        "p95_abs",
        "p99_abs",
        "frac_abs_gt_0p25",
        "frac_abs_gt_0p5",
        "frac_abs_gt_0p75",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
