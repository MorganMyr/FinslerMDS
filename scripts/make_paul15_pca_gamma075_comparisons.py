"""Comparison plots for the gamma=0.75 PCA geodesic MDS Paul15 embedding."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main_paul15_monocle import (  # noqa: E402
    finite_rescaled,
    load_monocle_graph,
    overlay_principal_graph,
    run_monocle_if_needed,
)


def main():
    dir_res = SCRIPT_DIR / "res" / "paul15" / "monocle3"
    dir_raw = dir_res / "raw"
    dir_exp = dir_res / "experiments"
    dir_pca = dir_res / "experiments_PCA" / "gamma075"
    input_dir = dir_res / "monocle_input_no19lymph_no11dc"
    bridge_script = SCRIPT_DIR / "monocle3_bridge" / "run_paul15_monocle3.R"

    umap_csv = dir_raw / "monocle_umap_pseudotime_no19lymph_no11dc.csv"
    umap_graph_prefix = dir_raw / "monocle_umap_principal_graph_no19lymph_no11dc"

    pca_base_name = "g075_from_pca_aggr_lw1_m50_i3"
    pca_name = f"{pca_base_name}_default"
    pca_csv = dir_pca / f"{pca_name}_pseudotime.csv"
    pca_graph_prefix = dir_pca / f"{pca_name}_principal_graph"

    diff_name = "best_deep_refined_ncenter70_nnk35_minbranch5"
    diff_embedding_csv = dir_exp / "best_deep_refined_embedding_umap_scale.csv"
    diff_csv = dir_exp / f"{diff_name}_pseudotime.csv"
    diff_graph_prefix = dir_exp / f"{diff_name}_principal_graph"

    run_monocle_if_needed(
        bridge_script,
        mode="standard",
        input_dir=input_dir,
        output_csv=umap_csv,
        root_cluster="7MEP",
        graph_prefix=umap_graph_prefix,
    )
    run_monocle_if_needed(
        bridge_script,
        mode="injected",
        input_dir=input_dir,
        output_csv=diff_csv,
        root_cluster="7MEP",
        embedding_csv=diff_embedding_csv,
        graph_prefix=diff_graph_prefix,
        monocle_graph={
            "use_partition": False,
            "close_loop": True,
            "learn_graph_control": {
                "ncenter": 70,
                "nn.k": 35,
                "minimal_branch_len": 5,
            },
        },
    )

    umap = load_result(umap_csv)
    pca = load_result(pca_csv)
    diff = load_result(diff_csv)

    umap_graph = load_monocle_graph(umap_graph_prefix)
    pca_graph = load_monocle_graph(pca_graph_prefix)
    diff_graph = load_monocle_graph(diff_graph_prefix)

    make_cross_pseudotime_grid(
        left=umap,
        right=pca,
        left_graph=umap_graph,
        right_graph=pca_graph,
        left_name="UMAP",
        right_name="geodesic MDS",
        left_pt_name="UMAP",
        right_pt_name="geodesic MDS",
        output_path=dir_pca / "umap_vs_geodesic_mds_pca_gamma075_pseudotime.pdf",
    )

    make_cross_pseudotime_grid(
        left=pca,
        right=diff,
        left_graph=pca_graph,
        right_graph=diff_graph,
        left_name="PCA-init geodesic MDS",
        right_name="diffmap-init geodesic MDS",
        left_pt_name="PCA-init MDS",
        right_pt_name="diffmap-init MDS",
        output_path=dir_pca / "geodesic_mds_pca_gamma075_vs_diffmap_pseudotime.pdf",
    )


def load_result(path):
    table = pd.read_csv(path)
    table["cell_id"] = table["cell_id"].astype(str)
    table = table.set_index("cell_id")
    return {
        "path": path,
        "cell_ids": table.index.to_numpy(dtype=str),
        "embedding": table[["dim1", "dim2"]].to_numpy(dtype=float),
        "pseudotime": finite_rescaled(table["pseudotime"].to_numpy(dtype=float)),
    }


def align_values(source, target):
    values = pd.Series(source["pseudotime"], index=source["cell_ids"])
    return values.loc[target["cell_ids"]].to_numpy(dtype=float)


def make_cross_pseudotime_grid(
    *,
    left,
    right,
    left_graph,
    right_graph,
    left_name,
    right_name,
    left_pt_name,
    right_pt_name,
    output_path,
):
    panels = [
        (
            left,
            left["pseudotime"],
            left_graph,
            f"{left_name}\n{left_pt_name} pseudotime",
        ),
        (
            left,
            align_values(right, left),
            None,
            f"{left_name}\n{right_pt_name} pseudotime",
        ),
        (
            right,
            align_values(left, right),
            None,
            f"{right_name}\n{left_pt_name} pseudotime",
        ),
        (
            right,
            right["pseudotime"],
            right_graph,
            f"{right_name}\n{right_pt_name} pseudotime",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (result, values, graph, title) in zip(axes.ravel(), panels):
        plot_panel(ax, result["embedding"], values, title=title, graph=graph)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved comparison: {output_path}")


def plot_panel(ax, embedding, values, *, title, graph):
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        cmap="viridis",
        s=8,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    if graph is not None:
        overlay_principal_graph(ax, graph)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    main()
