"""Run small Paul15 Monocle/MDS-geodesic tuning experiments.

This is an exploratory script, separate from ``main_paul15_monocle.py`` so the
main demo can stay readable. It tests path-frozen settings, rescales the final
embedding to the Monocle UMAP scale, runs the Monocle bridge, and writes a CSV
summary with graph/pseudotime diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, fit_finsler_mds
from finsler_mds.utils.embedding_io import cache_token, scale_embedding_to_dissimilarities
from main_paul15_monocle import (
    excluded_clusters_cache_tag,
    finite_rescaled,
    load_monocle_graph,
    load_monocle_result,
    load_or_build_inputs,
    overlay_principal_graph,
    run_monocle_if_needed,
    save_embedding_csv,
)
from finsler_mds.utils import plot_continuous_embedding


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
        {
            "name": "dg0p25_local50_iter3",
            "density_gamma": 0.25,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg0p5_local50_iter3",
            "density_gamma": 0.5,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg0p5_local75_iter3",
            "density_gamma": 0.5,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 60,
                "local_weight": 75.0,
            },
        },
        {
            "name": "dg0p5_local50_iter1",
            "density_gamma": 0.5,
            "path": {
                **common,
                "outer_iter": 1,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg0p5_local50_iter5",
            "density_gamma": 0.5,
            "path": {
                **common,
                "outer_iter": 5,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg0p75_local50_iter3",
            "density_gamma": 0.75,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg0p5_global150_iter5",
            "density_gamma": 0.5,
            "path": {
                **common,
                "outer_iter": 5,
                "inner_iter": 2,
                "n_landmark": 150,
                "targets_per_landmark": 200,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg1_local50_iter3",
            "density_gamma": 1.0,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg1_local100_iter3",
            "density_gamma": 1.0,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 50,
                "targets_per_landmark": 80,
                "n_local_pairs": 60,
                "local_weight": 100.0,
            },
        },
        {
            "name": "dg1_local20_iter10",
            "density_gamma": 1.0,
            "path": {
                **common,
                "outer_iter": 10,
                "inner_iter": 3,
                "n_landmark": 150,
                "targets_per_landmark": 200,
                "n_local_pairs": 40,
                "local_weight": 20.0,
            },
        },
        {
            "name": "dg0_local50_iter3",
            "density_gamma": 0.0,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
        {
            "name": "dg1p5_local50_iter3",
            "density_gamma": 1.5,
            "path": {
                **common,
                "outer_iter": 3,
                "inner_iter": 2,
                "n_landmark": 80,
                "targets_per_landmark": 120,
                "n_local_pairs": 50,
                "local_weight": 50.0,
            },
        },
    ]

    rows = [score_existing("umap_standard", standard_monocle_path, standard_graph_prefix, umap, pt_umap)]
    for variant in variants:
        rows.append(run_variant(variant, dir_raw, dir_exp, input_dir, bridge_script, cell_ids, umap, pt_umap, seed))
        pd.DataFrame(rows).to_csv(dir_exp / "summary.csv", index=False)
        print(pd.DataFrame(rows).sort_values("score", ascending=False).to_string(index=False))


def run_variant(variant, dir_raw, dir_exp, input_dir, bridge_script, cell_ids, umap, pt_umap, seed):
    name = variant["name"]
    density_gamma = variant["density_gamma"]
    print(f"\n=== {name} ===")

    inputs_path = dir_raw / f"paul15_monocle_inputs_k12_dg{cache_token(density_gamma)}_no19lymph_no11dc.npz"
    inputs = load_or_build_inputs(
        inputs_path,
        input_dir=input_dir,
        preprocessing={"n_pcs": 20, "initial_neighbors": 4, "trajectory_neighbors": 10, "use_float64": True},
        target_graph={"neighbors": 12, "use_rep": "X_diffmap", "density_gamma": density_gamma, "density_mode": "symmetric"},
        lineages={
            "erythrocyte": ["10GMP", "7MEP", "8Mk", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
            "monocyte": ["10GMP", "9GMP", "14Mo", "15Mo"],
        },
        include_non_lineage_cells=True,
        exclude_19lymph_when_all_cells=True,
        extra_excluded_clusters=["11DC"],
        seed=seed,
    )
    D = inputs["dissimilarities"]
    init, scale = scale_embedding_to_dissimilarities(umap, D, random_state=seed)
    print(f"init scale to dissimilarities: {scale:.6g}")

    emb_path = dir_exp / f"{name}_embedding_raw.npz"
    if emb_path.exists():
        embedding = np.load(emb_path)["embedding"]
        print(f"loaded {emb_path}")
    else:
        start = perf_counter()
        embedding, stress = fit_finsler_mds(
            D,
            metric=RandersMetric(alpha=0.0),
            optimizer="path_frozen",
            init=init,
            n_components=2,
            mask_random_state=seed,
            target_random_state=seed + 3,
            print_time=True,
            **variant["path"],
        )
        print(f"optimizer stress: {stress}")
        np.savez(emb_path, embedding=embedding, elapsed=np.asarray(perf_counter() - start))

    embedding_for_monocle = rescale_to_reference(embedding, umap)
    emb_csv = dir_exp / f"{name}_embedding_umap_scale.csv"
    save_embedding_csv(emb_csv, cell_ids, embedding_for_monocle)

    pt_path = dir_exp / f"{name}_pseudotime.csv"
    graph_prefix = dir_exp / f"{name}_principal_graph"
    run_monocle_if_needed(
        bridge_script,
        mode="injected",
        input_dir=input_dir,
        output_csv=pt_path,
        root_cluster="7MEP",
        embedding_csv=emb_csv,
        graph_prefix=graph_prefix,
    )

    row = score_existing(name, pt_path, graph_prefix, umap, pt_umap)
    row.update({
        "density_gamma": density_gamma,
        "outer_iter": variant["path"]["outer_iter"],
        "inner_iter": variant["path"]["inner_iter"],
        "local_weight": variant["path"]["local_weight"],
        "n_landmark": variant["path"]["n_landmark"],
        "n_local_pairs": variant["path"]["n_local_pairs"],
    })
    save_variant_plot(name, pt_path, graph_prefix, dir_exp)
    return row


def rescale_to_reference(embedding, reference):
    X = np.asarray(embedding, dtype=float)
    R = np.asarray(reference, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    Rc = R - R.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(Rc) / np.linalg.norm(Xc)
    return Xc * scale + R.mean(axis=0, keepdims=True)


def score_existing(name, pseudotime_path, graph_prefix, umap, pt_umap):
    tab = pd.read_csv(pseudotime_path).set_index("cell_id")
    X = tab[["dim1", "dim2"]].to_numpy(dtype=float)
    pt = tab["pseudotime"].to_numpy(dtype=float)
    graph = load_monocle_graph(graph_prefix)
    nodes = graph["nodes"]
    edges = graph["edges"]

    nn_u = NearestNeighbors(n_neighbors=16).fit(umap).kneighbors(umap, return_distance=False)[:, 1:]
    nn_x = NearestNeighbors(n_neighbors=16).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    knn15 = float(np.mean([len(set(a).intersection(b)) / 15 for a, b in zip(nn_u, nn_x)]))

    local_jump99, local_jumpmax = local_jump_stats(X, pt)
    procrustes = procrustes_rmse(X, umap)
    pt_corr = float(np.corrcoef(pt_umap, pt)[0, 1])

    nodes_penalty = abs(len(nodes) - 45) / 45
    roots_penalty = abs(int(nodes["is_root"].sum()) - 1)
    jump_penalty = local_jump99 / max(np.quantile(pt, 0.99), 1e-12)
    score = pt_corr + 0.4 * knn15 - 0.35 * nodes_penalty - 0.2 * roots_penalty - 0.1 * jump_penalty

    return {
        "name": name,
        "score": score,
        "pt_corr": pt_corr,
        "pt_q25": float(np.quantile(pt, 0.25)),
        "pt_med": float(np.quantile(pt, 0.5)),
        "pt_q75": float(np.quantile(pt, 0.75)),
        "pt_max": float(np.max(pt)),
        "nodes": len(nodes),
        "edges": len(edges),
        "roots": int(nodes["is_root"].sum()),
        "knn15": knn15,
        "procrustes_rmse": procrustes,
        "jump99": local_jump99,
        "jumpmax": local_jumpmax,
    }


def local_jump_stats(X, pt):
    nn = NearestNeighbors(n_neighbors=11).fit(X)
    distances, indices = nn.kneighbors(X)
    rows = np.repeat(np.arange(len(X)), 10)
    cols = indices[:, 1:].reshape(-1)
    jumps = np.abs(pt[rows] - pt[cols])
    return float(np.quantile(jumps, 0.99)), float(np.max(jumps))


def procrustes_rmse(X, reference):
    Xc = X - X.mean(axis=0, keepdims=True)
    Rc = reference - reference.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(Xc, Rc)
    aligned = Xc @ rotation * (np.linalg.norm(Rc) / np.linalg.norm(Xc))
    rmse = np.sqrt(np.mean(np.sum((aligned - Rc) ** 2, axis=1)))
    radius = np.sqrt(np.mean(np.sum(Rc ** 2, axis=1)))
    return float(rmse / radius)


def save_variant_plot(name, pseudotime_path, graph_prefix, dir_exp):
    tab = pd.read_csv(pseudotime_path)
    X = tab[["dim1", "dim2"]].to_numpy(dtype=float)
    pt = finite_rescaled(tab["pseudotime"].to_numpy(dtype=float))
    graph = load_monocle_graph(graph_prefix)
    fig, ax = plot_continuous_embedding(X, pt, title=f"{name}: Monocle pseudotime + principal graph", s=8)
    overlay_principal_graph(ax, graph)
    fig.savefig(dir_exp / f"{name}_pseudotime_graph.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
