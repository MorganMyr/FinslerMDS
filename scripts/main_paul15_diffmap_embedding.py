"""Paul15 MDS/path-frozen from diffusion-map geodesics and full-UMAP init.

The target dissimilarities are loaded from the cached Paul15 diffmap graph
without 19Lymph. The initialization is the PAGA-UMAP computed on all clusters,
with 19Lymph cells removed afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds, geodesic_embedding_stress
from finsler_mds.utils import plot_continuous_embedding
from finsler_mds.utils.embedding_io import scale_embedding_to_dissimilarities


SEED = 42
OPTIMIZER = "path_frozen"  # one of {"path_frozen", "smacof"}
N_COMPONENTS = 2
ALPHA_EMBEDDING = 0.0

REMOVED_INIT_CLUSTERS = ("19Lymph",)
INPUTS_CACHE = "inputs_k12_dg1_sym_no19lymph_s42.npz"
RUN_TAG = "no19_umapfullinit"

PATH_FROZEN_OPTIONS = {
    "graph_neighbors": 20,
    # Exploration stage.
    "outer_iter": 40,
    "inner_iter": 50,
    "outer_step_size": 0.3,
    "eps": 1e-6,
    "method": "L-BFGS-B",
    "optimizer_options": {"ftol": 1e-8, "maxls": 30},
    "n_landmark": 200,
    "random_landmark_fraction": 0.0,
    "targets_per_landmark": 600,
    "n_local_pairs": 20,
    "local_pair_mode": "direct",
    "local_global_reweighting": "count",
    "local_weight": 1,
    "device": "auto",
    "verbose": 1,
    "log_frequency": 5,
}

PATH_FROZEN_FINISHER = {
    "outer_iter": 20,
    "inner_iter": 5,
    "outer_step_size": 0.1,
}

SMACOF_OPTIONS = {
    "max_iter": 1000,
    "eps": 1e-6,
    "n_init": 1,
    "n_jobs": 1,
    "pseudo_inv_solver": "gmres",
    "project_on_V": True,
    "check_monotony": True,
    "device": "auto",
    "version": "corrected",
    "verbose": 1,
}


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    res_dir = script_dir / "res" / "paul15"
    raw_dir = res_dir / "raw"
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_npz(raw_dir / INPUTS_CACHE)
    D = np.asarray(inputs["dissimilarities"], dtype=float)
    cell_ids = inputs["cell_ids"].astype(str)
    pseudotime = load_full_paga_lineage_pseudotime(res_dir, cell_ids)
    umap_full_init = load_full_umap_init(res_dir, cell_ids)
    init, scale = scale_embedding_to_dissimilarities(umap_full_init, D, random_state=SEED)
    print(f"Loaded {len(cell_ids)} Paul15 cells from {INPUTS_CACHE}")
    print(f"Using full-PAGA-UMAP init filtered without {', '.join(REMOVED_INIT_CLUSTERS)}")
    print("Coloring with full-PAGA lineage pseudotime filtered to the same cells.")
    print(f"Rescaled init by factor {scale:.6g}")

    metric = RandersMetric(alpha=ALPHA_EMBEDDING)
    optimizer = normalize_optimizer(args.optimizer)
    if optimizer == "path_frozen":
        embedding, stress, stage_names, stage_stresses, exploration_embedding = run_path_frozen_stages(
            D,
            init=init,
            metric=metric,
        )
        full_geodesic_stress = geodesic_embedding_stress(
            embedding,
            D,
            metric=metric,
            n_neighbors=PATH_FROZEN_OPTIONS["graph_neighbors"],
            on_unreachable="warn_skip",
        )
        print(f"path-frozen optimizer stress: {stress}")
        print(f"path-frozen full geodesic stress: {full_geodesic_stress}")
    else:
        result = fit_finsler_mds(
            D,
            optimizer="smacof",
            metric=metric,
            n_components=N_COMPONENTS,
            init=init,
            random_state=SEED,
            return_result=True,
            print_time=True,
            **SMACOF_OPTIONS,
        )
        embedding = result.embedding
        stress = result.stress
        full_geodesic_stress = np.nan
        print(f"SMACOF stress: {stress}")
        print(f"SMACOF iterations: {result.n_iter}")

    stem = f"paul15_diffmap_{optimizer}_{RUN_TAG}"
    np.savez(
        raw_dir / f"{stem}.npz",
        embedding=np.asarray(embedding, dtype=float),
        init=init,
        cell_ids=cell_ids,
        pseudotime=pseudotime,
        stress=np.asarray(stress, dtype=float),
        full_geodesic_stress=np.asarray(full_geodesic_stress, dtype=float),
        optimizer=np.asarray(optimizer),
        inputs_cache=np.asarray(INPUTS_CACHE),
        alpha_embedding=np.asarray(ALPHA_EMBEDDING),
        stage_names=np.asarray(stage_names if optimizer == "path_frozen" else [], dtype=str),
        stage_stresses=np.asarray(stage_stresses if optimizer == "path_frozen" else [], dtype=float),
        exploration_embedding=np.asarray(
            exploration_embedding if optimizer == "path_frozen" else np.empty((0, N_COMPONENTS)),
            dtype=float,
        ),
    )
    plot_continuous_embedding(
        embedding,
        pseudotime,
        title=plot_title(optimizer),
        save_path=fig_dir / f"{stem}_paga_lineage_pseudotime.pdf",
        s=7,
    )
    plt.close("all")
    print(f"Saved embedding: {raw_dir / f'{stem}.npz'}")
    print(f"Saved figure: {fig_dir / f'{stem}_paga_lineage_pseudotime.pdf'}")


def run_path_frozen_stages(D, *, init, metric):
    stages = [("exploration", {})]
    if int(PATH_FROZEN_FINISHER.get("outer_iter", 0)) > 0:
        stages.append(("finisher", PATH_FROZEN_FINISHER))

    current = init
    stage_names = []
    stage_stresses = []
    exploration_embedding = None
    stress = np.nan
    for stage_name, overrides in stages:
        options = dict(PATH_FROZEN_OPTIONS)
        options.update(overrides)
        print(
            f"Running path-frozen {stage_name}: "
            f"outer_iter={options['outer_iter']}, "
            f"inner_iter={options['inner_iter']}, "
            f"outer_step_size={options['outer_step_size']}"
        )
        current, stress = fit_finsler_mds(
            D,
            optimizer="path_frozen",
            metric=metric,
            n_components=N_COMPONENTS,
            init=current,
            mask_random_state=SEED,
            target_random_state=SEED + 3,
            random_state=SEED,
            print_time=True,
            **options,
        )
        stage_names.append(stage_name)
        stage_stresses.append(float(stress))
        if stage_name == "exploration":
            exploration_embedding = np.asarray(current, dtype=float).copy()

    return current, float(stress), stage_names, stage_stresses, exploration_embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Paul15 MDS/path-frozen from cached diffmap dissimilarities, "
            "using the full-cluster PAGA UMAP filtered without 19Lymph as init."
        )
    )
    parser.add_argument(
        "--optimizer",
        default=OPTIMIZER,
        choices=("path_frozen", "smacof"),
        help="Optimizer to run. Default: %(default)s.",
    )
    return parser.parse_args()


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_full_umap_init(res_dir, target_cell_ids):
    baseline = load_npz(res_dir / "baseline" / "embeddings" / "paul15_embeddings_metadata.npz")
    full_ids = baseline["cell_ids"].astype(str)
    labels = baseline["labels"].astype(str)
    keep = ~np.isin(labels, REMOVED_INIT_CLUSTERS)
    kept_ids = full_ids[keep]
    if not np.array_equal(kept_ids, target_cell_ids):
        raise ValueError(
            "The filtered full-UMAP cells do not match the diffmap inputs. "
            "Check that both are exactly Paul15 without 19Lymph."
        )
    return np.asarray(baseline["umap"][keep], dtype=float)


def load_full_paga_lineage_pseudotime(res_dir, target_cell_ids):
    baseline = load_npz(res_dir / "baseline" / "embeddings" / "paul15_embeddings_metadata.npz")
    full_ids = baseline["cell_ids"].astype(str)
    positions = {cell_id: idx for idx, cell_id in enumerate(full_ids)}
    try:
        order = np.asarray([positions[cell_id] for cell_id in target_cell_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Full PAGA baseline is missing cell id {exc.args[0]!r}.") from exc
    return np.asarray(baseline["dpt_lineage_pseudotime"][order], dtype=float)


def normalize_optimizer(optimizer):
    value = str(optimizer).lower().replace("-", "_")
    if value in {"path_frozen", "pf"}:
        return "path_frozen"
    if value in {"smacof", "mds"}:
        return "smacof"
    raise ValueError("OPTIMIZER must be one of {'path_frozen', 'smacof'}.")


def plot_title(optimizer):
    if optimizer == "path_frozen":
        return "Paul15 geodesic MDS from diffmap distances: PAGA lineage pseudotime"
    return "Paul15 MDS from diffmap distances: PAGA lineage pseudotime"


if __name__ == "__main__":
    main()
