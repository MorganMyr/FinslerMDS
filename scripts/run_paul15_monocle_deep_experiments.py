"""Long Paul15 path-frozen experiments followed by Monocle graph tuning.

The first stage intentionally pushes path-frozen much farther than the default
demo settings, looking for embeddings that remain coherent by cluster geometry
while moving away from the Monocle UMAP layout. The second stage keeps those
embeddings fixed and varies Monocle principal-graph settings.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finsler_mds import RandersMetric, fit_finsler_mds
from finsler_mds.utils import plot_categorical_embedding
from finsler_mds.utils.embedding_io import cache_token, scale_embedding_to_dissimilarities
from main_paul15_monocle import (
    excluded_clusters_cache_tag,
    load_or_build_inputs,
    overlay_principal_graph,
    paul15_display_labels,
    run_monocle_if_needed,
    save_embedding_csv,
)
from run_paul15_monocle_experiments import (
    rescale_to_reference,
    save_variant_plot,
    score_existing,
)
from run_paul15_monocle_experiments_strong import with_novelty


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

    run_monocle_if_needed(
        bridge_script,
        mode="standard",
        input_dir=input_dir,
        output_csv=standard_monocle_path,
        root_cluster="7MEP",
        graph_prefix=standard_graph_prefix,
    )

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
    embedding_variants = [
        variant(
            "deep_balanced_g0p65_m100_i5",
            density_gamma=0.65,
            path={
                **common,
                "outer_iter": 100,
                "inner_iter": 5,
                "n_landmark": 150,
                "targets_per_landmark": 220,
                "n_local_pairs": 40,
                "local_weight": 25.0,
            },
        ),
        variant(
            "deep_global_g0p65_m100_i5",
            density_gamma=0.65,
            path={
                **common,
                "outer_iter": 100,
                "inner_iter": 5,
                "n_landmark": 180,
                "targets_per_landmark": 260,
                "n_local_pairs": 25,
                "local_weight": 12.0,
            },
        ),
        variant(
            "deep_density_g0p75_m80_i5",
            density_gamma=0.75,
            path={
                **common,
                "outer_iter": 80,
                "inner_iter": 5,
                "n_landmark": 150,
                "targets_per_landmark": 220,
                "n_local_pairs": 40,
                "local_weight": 25.0,
            },
        ),
    ]

    graph_variants = [
        ("graph_default", {}),
        ("graph_no_partition", {"use_partition": False}),
        (
            "graph_smooth_no_partition",
            {
                "use_partition": False,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 55,
                    "minimal_branch_len": 12,
                    "nn.k": 30,
                    "geodesic_distance_ratio": 0.45,
                },
            },
        ),
        (
            "graph_detailed_no_partition",
            {
                "use_partition": False,
                "close_loop": True,
                "learn_graph_control": {
                    "ncenter": 85,
                    "minimal_branch_len": 5,
                    "nn.k": 35,
                    "geodesic_distance_ratio": 0.35,
                },
            },
        ),
        (
            "graph_pruned_partition",
            {
                "cluster_k": 30,
                "partition_qval": 0.2,
                "use_partition": True,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 60,
                    "minimal_branch_len": 15,
                    "nn.k": 30,
                },
            },
        ),
        (
            "graph_tuned_n100_b3",
            {
                "use_partition": False,
                "close_loop": True,
                "learn_graph_control": {
                    "ncenter": 100,
                    "minimal_branch_len": 3,
                    "nn.k": 40,
                    "geodesic_distance_ratio": 0.35,
                },
            },
        ),
        (
            "graph_tuned_partition_n85_b5",
            {
                "cluster_k": 30,
                "partition_qval": 0.2,
                "use_partition": True,
                "close_loop": True,
                "learn_graph_control": {
                    "ncenter": 85,
                    "minimal_branch_len": 5,
                    "nn.k": 35,
                    "geodesic_distance_ratio": 0.35,
                },
            },
        ),
    ]

    rows = []
    for emb_variant in embedding_variants:
        embedding, labels = run_embedding_variant(
            emb_variant,
            dir_raw,
            dir_exp,
            input_dir,
            cell_ids,
            umap,
            seed,
        )
        embedding_for_monocle = rescale_to_reference(embedding, umap)
        emb_csv = dir_exp / f"{emb_variant['name']}_embedding_umap_scale.csv"
        save_embedding_csv(emb_csv, cell_ids, embedding_for_monocle)
        save_cluster_plot(emb_variant["name"], embedding_for_monocle, labels, dir_exp)

        for graph_name, graph_options in graph_variants:
            row = run_graph_variant(
                emb_variant["name"],
                graph_name,
                graph_options,
                emb_csv,
                dir_exp,
                input_dir,
                bridge_script,
                umap,
                pt_umap,
            )
            row.update(
                {
                    "embedding_variant": emb_variant["name"],
                    "graph_variant": graph_name,
                    "density_gamma": emb_variant["density_gamma"],
                    **emb_variant["path_summary"],
                }
            )
            rows.append(with_novelty(row))
            summary = pd.DataFrame(rows)
            summary.to_csv(dir_exp / "deep_summary.csv", index=False)
            print(
                summary.sort_values("novel_score", ascending=False)[
                    [
                        "embedding_variant",
                        "graph_variant",
                        "novel_score",
                        "score",
                        "pt_corr",
                        "nodes",
                        "roots",
                        "knn15",
                        "procrustes_rmse",
                        "jump99",
                    ]
                ].to_string(index=False)
            )


def variant(name, *, density_gamma, path):
    path_summary = {
        "outer_iter": path["outer_iter"],
        "inner_iter": path["inner_iter"],
        "local_weight": path["local_weight"],
        "n_landmark": path["n_landmark"],
        "targets_per_landmark": path["targets_per_landmark"],
        "n_local_pairs": path["n_local_pairs"],
    }
    return {
        "name": name,
        "density_gamma": density_gamma,
        "path": path,
        "path_summary": path_summary,
    }


def run_embedding_variant(variant, dir_raw, dir_exp, input_dir, cell_ids, umap, seed):
    name = variant["name"]
    density_gamma = variant["density_gamma"]
    print(f"\n=== embedding {name} ===")

    scope_suffix = "_no19lymph" + excluded_clusters_cache_tag(["11DC"])
    inputs_path = dir_raw / f"paul15_monocle_inputs_k12_dg{cache_token(density_gamma)}{scope_suffix}.npz"
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
    labels = inputs["labels"].astype(str)
    D = inputs["dissimilarities"]
    init, scale = scale_embedding_to_dissimilarities(umap, D, random_state=seed)
    print(f"init scale to dissimilarities: {scale:.6g}")

    emb_path = dir_exp / f"{name}_embedding_raw.npz"
    if emb_path.exists():
        with np.load(emb_path) as data:
            embedding = np.asarray(data["embedding"], dtype=float)
        print(f"loaded cached embedding: {emb_path}")
        return embedding, labels

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
    elapsed = perf_counter() - start
    print(f"optimizer stress: {stress}")
    np.savez(
        emb_path,
        embedding=embedding,
        elapsed=np.asarray(elapsed),
        stress=np.asarray(stress),
    )
    print(f"saved embedding: {emb_path}")
    return embedding, labels


def save_cluster_plot(name, embedding, labels, dir_exp):
    display_labels = paul15_display_labels(labels)
    fig, _ = plot_categorical_embedding(
        embedding,
        labels=display_labels,
        title=f"{name}: clusters",
        s=8,
    )
    fig.savefig(dir_exp / f"{name}_clusters.pdf", bbox_inches="tight")
    fig.savefig(dir_exp / f"{name}_clusters.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def run_graph_variant(
    embedding_name,
    graph_name,
    graph_options,
    emb_csv,
    dir_exp,
    input_dir,
    bridge_script,
    umap,
    pt_umap,
):
    name = f"{embedding_name}_{graph_name}"
    print(f"\n=== graph {name} ===")
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
        monocle_graph=graph_options,
    )
    save_variant_plot(name, pt_path, graph_prefix, dir_exp)
    return score_existing(name, pt_path, graph_prefix, umap, pt_umap)


if __name__ == "__main__":
    main()
