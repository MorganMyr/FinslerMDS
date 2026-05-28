"""Aggressive PCA-based path-frozen Paul15 experiments + Monocle graph tuning."""

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

from finsler_mds import RandersMetric, fit_finsler_mds  # noqa: E402
from finsler_mds.utils import plot_categorical_embedding, plot_continuous_embedding  # noqa: E402
from finsler_mds.utils.embedding_io import cache_token, scale_embedding_to_dissimilarities  # noqa: E402
from main_paul15_monocle import (  # noqa: E402
    finite_rescaled,
    load_monocle_graph,
    overlay_principal_graph,
    paul15_display_labels,
    run_monocle_if_needed,
    save_embedding_csv,
)
from run_paul15_monocle_experiments import (  # noqa: E402
    rescale_to_reference,
    score_existing,
)
from run_paul15_monocle_pca_mds_experiments import export_monocle_pca  # noqa: E402
from run_paul15_monocle_pca_refinement_search import load_or_build_dissimilarities  # noqa: E402


def main():
    seed = 42
    dir_res = SCRIPT_DIR / "res" / "paul15" / "monocle3"
    dir_raw = dir_res / "raw"
    dir_out = dir_res / "experiments_PCA"
    dir_search = dir_out / "search_candidates"
    dir_aggressive = dir_out / "aggressive"
    input_dir = dir_res / "monocle_input_no19lymph_no11dc"
    bridge_script = SCRIPT_DIR / "monocle3_bridge" / "run_paul15_monocle3.R"
    pca_csv = dir_out / "monocle_pca_50_no19lymph_no11dc.csv"

    dir_aggressive.mkdir(parents=True, exist_ok=True)
    export_monocle_pca(input_dir, pca_csv, n_components=50)

    pca = pd.read_csv(pca_csv)
    cell_ids = pca["cell_id"].astype(str).to_numpy()
    pca_values = pca.drop(columns=["cell_id"]).to_numpy(dtype=float)
    metadata = pd.read_csv(input_dir / "cell_metadata.csv").set_index("cell_id")
    labels = metadata.loc[cell_ids, "paul15_clusters"].astype(str).to_numpy()

    standard_path = dir_raw / "monocle_umap_pseudotime_no19lymph_no11dc.csv"
    standard_graph_prefix = dir_raw / "monocle_umap_principal_graph_no19lymph_no11dc"
    run_monocle_if_needed(
        bridge_script,
        mode="standard",
        input_dir=input_dir,
        output_csv=standard_path,
        root_cluster="7MEP",
        graph_prefix=standard_graph_prefix,
    )
    standard = pd.read_csv(standard_path)
    standard = standard.set_index("cell_id").loc[cell_ids]
    umap = standard[["dim1", "dim2"]].to_numpy(dtype=float)
    pt_umap = standard["pseudotime"].to_numpy(dtype=float)

    embedding_specs = [
        embedding_spec(
            "aggr_from_g1p25_lw1_to_g1p25_lw0p5_m50_i3",
            init=dir_search / "cont2_k50_g1p25_lw1_embedding_raw.npz",
            k=50,
            gamma=1.25,
            max_iter=50,
            inner_iter=3,
            local_weight=0.5,
            n_global_landmarks=180,
            max_global_targets_per_source=260,
            n_local_neighbors=30,
            graph_neighbors=35,
        ),
        embedding_spec(
            "aggr_from_g1p25_lw1_to_g1p5_lw0p5_m45_i3",
            init=dir_search / "cont2_k50_g1p25_lw1_embedding_raw.npz",
            k=60,
            gamma=1.5,
            max_iter=45,
            inner_iter=3,
            local_weight=0.5,
            n_global_landmarks=180,
            max_global_targets_per_source=260,
            n_local_neighbors=30,
            graph_neighbors=35,
        ),
        embedding_spec(
            "aggr_from_g1p0_lw0p5_to_g1p25_lw0p25_m60_i3",
            init=dir_search / "cont2_k50_g1p0_lw0p5_embedding_raw.npz",
            k=50,
            gamma=1.25,
            max_iter=60,
            inner_iter=3,
            local_weight=0.25,
            n_global_landmarks=220,
            max_global_targets_per_source=320,
            n_local_neighbors=20,
            graph_neighbors=35,
        ),
        embedding_spec(
            "aggr_from_g1p25_lw1_to_k80_g1p25_lw0p5_m40_i3",
            init=dir_search / "cont2_k50_g1p25_lw1_embedding_raw.npz",
            k=80,
            gamma=1.25,
            max_iter=40,
            inner_iter=3,
            local_weight=0.5,
            n_global_landmarks=180,
            max_global_targets_per_source=260,
            n_local_neighbors=30,
            graph_neighbors=35,
        ),
    ]

    embedding_rows = []
    for spec in embedding_specs:
        row = run_embedding_spec(
            spec,
            pca_values,
            cell_ids,
            labels,
            umap,
            dir_aggressive,
            seed=seed,
        )
        embedding_rows.append(row)
        pd.DataFrame(embedding_rows).to_csv(dir_aggressive / "aggressive_embedding_summary.csv", index=False)

    make_cluster_contact_sheet(dir_aggressive, embedding_rows)

    selected = select_graph_tuning_candidates(embedding_rows)
    graph_rows = []
    for row in selected:
        graph_rows.extend(
            run_graph_tuning(
                row,
                cell_ids,
                umap,
                pt_umap,
                input_dir,
                bridge_script,
                dir_aggressive,
            )
        )
        pd.DataFrame(graph_rows).to_csv(dir_aggressive / "aggressive_graph_tuning_summary.csv", index=False)

    make_graph_contact_sheet(dir_aggressive, graph_rows)
    print(f"Saved aggressive PCA experiments in: {dir_aggressive}")


def embedding_spec(
    name,
    *,
    init,
    k,
    gamma,
    max_iter,
    inner_iter,
    local_weight,
    n_global_landmarks,
    max_global_targets_per_source,
    n_local_neighbors,
    graph_neighbors,
):
    return {
        "name": name,
        "init": Path(init),
        "k": int(k),
        "gamma": float(gamma),
        "max_iter": int(max_iter),
        "inner_iter": int(inner_iter),
        "local_weight": float(local_weight),
        "n_global_landmarks": int(n_global_landmarks),
        "max_global_targets_per_source": int(max_global_targets_per_source),
        "n_local_neighbors": int(n_local_neighbors),
        "graph_neighbors": int(graph_neighbors),
    }


def run_embedding_spec(spec, pca_values, cell_ids, labels, umap, dir_out, *, seed):
    name = spec["name"]
    out_npz = dir_out / f"{name}_embedding_raw.npz"
    out_csv = dir_out / f"{name}_embedding_umap_scale.csv"
    out_pdf = dir_out / f"{name}_clusters.pdf"
    out_png = dir_out / f"{name}_clusters.png"

    if not spec["init"].exists():
        raise FileNotFoundError(f"Missing init embedding: {spec['init']}")

    if out_npz.exists():
        with np.load(out_npz) as data:
            embedding = np.asarray(data["embedding"], dtype=float)
            stress = float(data["stress"])
            elapsed = float(data["elapsed"])
        print(f"Loaded cached aggressive embedding: {name}")
    else:
        D = load_or_build_dissimilarities(
            pca_values,
            cell_ids,
            labels,
            dir_out,
            k=spec["k"],
            gamma=spec["gamma"],
        )
        with np.load(spec["init"]) as data:
            base_init = np.asarray(data["embedding"], dtype=float)
        init, scale = scale_embedding_to_dissimilarities(base_init, D, random_state=seed)
        print(f"{name}: rescaled init by {scale:.6g}")
        start = perf_counter()
        embedding, stress = fit_finsler_mds(
            D,
            metric=RandersMetric(alpha=0.0),
            optimizer="path_frozen",
            init=init,
            n_components=2,
            graph_neighbors=spec["graph_neighbors"],
            max_iter=spec["max_iter"],
            inner_iter=spec["inner_iter"],
            eps=1e-6,
            method="L-BFGS-B",
            optimizer_options={"ftol": 1e-8, "maxls": 30},
            n_global_landmarks=spec["n_global_landmarks"],
            n_local_neighbors=spec["n_local_neighbors"],
            local_pair_mode="direct",
            max_global_targets_per_source=spec["max_global_targets_per_source"],
            global_target_sampling="random",
            local_global_reweighting="count",
            local_weight=spec["local_weight"],
            device="auto",
            verbose=1,
            mask_random_state=seed,
            target_random_state=seed + 3,
            print_time=True,
        )
        elapsed = perf_counter() - start
        np.savez(
            out_npz,
            embedding=embedding,
            stress=np.asarray(stress),
            elapsed=np.asarray(elapsed),
            cell_ids=cell_ids,
            init=str(spec["init"]),
            **{key: np.asarray(value) for key, value in spec.items() if key != "init"},
        )
        print(f"Saved aggressive embedding: {out_npz}")

    embedding_for_monocle = rescale_to_reference(embedding, umap)
    save_embedding_csv(out_csv, cell_ids, embedding_for_monocle)
    save_cluster_plot(embedding_for_monocle, labels, out_pdf, out_png, title=name)

    return {
        "name": name,
        "k": spec["k"],
        "gamma": spec["gamma"],
        "max_iter": spec["max_iter"],
        "inner_iter": spec["inner_iter"],
        "local_weight": spec["local_weight"],
        "n_global_landmarks": spec["n_global_landmarks"],
        "max_global_targets_per_source": spec["max_global_targets_per_source"],
        "n_local_neighbors": spec["n_local_neighbors"],
        "graph_neighbors": spec["graph_neighbors"],
        "stress": stress,
        "elapsed": elapsed,
        "npz": str(out_npz),
        "csv": str(out_csv),
        "pdf": str(out_pdf),
        "png": str(out_png),
    }


def save_cluster_plot(embedding, labels, out_pdf, out_png, *, title):
    display_labels = paul15_display_labels(labels)
    fig, _ = plot_categorical_embedding(embedding, labels=display_labels, title=title, s=8)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)


def select_graph_tuning_candidates(rows):
    by_name = {row["name"]: row for row in rows}
    preferred = [
        "aggr_from_g1p25_lw1_to_g1p25_lw0p5_m50_i3",
        "aggr_from_g1p0_lw0p5_to_g1p25_lw0p25_m60_i3",
    ]
    selected = [by_name[name] for name in preferred if name in by_name]
    if selected:
        return selected
    return sorted(rows, key=lambda row: row["stress"])[:2]


def graph_variants():
    return [
        (
            "default",
            {},
        ),
        (
            "n70_nnk35_b5_gr0p35",
            {
                "use_partition": False,
                "close_loop": True,
                "learn_graph_control": {
                    "ncenter": 70,
                    "nn.k": 35,
                    "minimal_branch_len": 5,
                    "geodesic_distance_ratio": 0.35,
                },
            },
        ),
        (
            "n85_nnk35_b5_gr0p35",
            {
                "use_partition": False,
                "close_loop": True,
                "learn_graph_control": {
                    "ncenter": 85,
                    "nn.k": 35,
                    "minimal_branch_len": 5,
                    "geodesic_distance_ratio": 0.35,
                },
            },
        ),
        (
            "n110_nnk40_b3_gr0p35",
            {
                "use_partition": False,
                "close_loop": True,
                "learn_graph_control": {
                    "ncenter": 110,
                    "nn.k": 40,
                    "minimal_branch_len": 3,
                    "geodesic_distance_ratio": 0.35,
                },
            },
        ),
        (
            "n70_nnk28_b8_gr0p5",
            {
                "use_partition": False,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 70,
                    "nn.k": 28,
                    "minimal_branch_len": 8,
                    "geodesic_distance_ratio": 0.5,
                },
            },
        ),
        (
            "n55_nnk25_b10_gr0p6",
            {
                "use_partition": False,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 55,
                    "nn.k": 25,
                    "minimal_branch_len": 10,
                    "geodesic_distance_ratio": 0.6,
                },
            },
        ),
        (
            "n70_nnk25_b10_gr0p6",
            {
                "use_partition": False,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 70,
                    "nn.k": 25,
                    "minimal_branch_len": 10,
                    "geodesic_distance_ratio": 0.6,
                },
            },
        ),
        (
            "n85_nnk28_b8_gr0p5",
            {
                "use_partition": False,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 85,
                    "nn.k": 28,
                    "minimal_branch_len": 8,
                    "geodesic_distance_ratio": 0.5,
                },
            },
        ),
        (
            "n70_nnk32_b8_gr0p45",
            {
                "use_partition": False,
                "close_loop": False,
                "learn_graph_control": {
                    "ncenter": 70,
                    "nn.k": 32,
                    "minimal_branch_len": 8,
                    "geodesic_distance_ratio": 0.45,
                },
            },
        ),
    ]


def run_graph_tuning(embedding_row, cell_ids, umap, pt_umap, input_dir, bridge_script, dir_out):
    rows = []
    name = embedding_row["name"]
    for graph_name, graph_options in graph_variants():
        full_name = f"{name}_{graph_name}"
        output_csv = dir_out / f"{full_name}_pseudotime.csv"
        graph_prefix = dir_out / f"{full_name}_principal_graph"
        run_monocle_if_needed(
            bridge_script,
            mode="injected",
            input_dir=input_dir,
            output_csv=output_csv,
            root_cluster="7MEP",
            embedding_csv=Path(embedding_row["csv"]),
            graph_prefix=graph_prefix,
            monocle_graph=graph_options,
        )
        save_pseudotime_graph_plot(full_name, output_csv, graph_prefix, dir_out)
        row = score_existing(full_name, output_csv, graph_prefix, umap, pt_umap)
        row.update(
            {
                "embedding": name,
                "graph_variant": graph_name,
                "embedding_stress": embedding_row["stress"],
                "embedding_csv": embedding_row["csv"],
            }
        )
        rows.append(row)
        print(
            pd.DataFrame(rows)[
                ["name", "score", "pt_corr", "nodes", "roots", "jump99", "knn15"]
            ].to_string(index=False)
        )
    return rows


def save_pseudotime_graph_plot(name, output_csv, graph_prefix, dir_out):
    table = pd.read_csv(output_csv)
    embedding = table[["dim1", "dim2"]].to_numpy(dtype=float)
    pseudotime = finite_rescaled(table["pseudotime"].to_numpy(dtype=float))
    graph = load_monocle_graph(graph_prefix)
    fig, ax = plot_continuous_embedding(
        embedding,
        pseudotime,
        title=f"{name}: Monocle pseudotime + principal graph",
        s=8,
    )
    overlay_principal_graph(ax, graph)
    fig.savefig(dir_out / f"{name}_pseudotime_graph.pdf", bbox_inches="tight")
    fig.savefig(dir_out / f"{name}_pseudotime_graph.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def make_cluster_contact_sheet(dir_out, rows):
    make_contact_sheet(
        dir_out,
        rows,
        image_key="png",
        title_fn=lambda row: f"{row['name']}\nstress={row['stress']:.3g}",
        output_name="aggressive_cluster_contact_sheet.png",
    )


def make_graph_contact_sheet(dir_out, rows):
    for row in rows:
        row["png"] = str(dir_out / f"{row['name']}_pseudotime_graph.png")
    make_contact_sheet(
        dir_out,
        rows,
        image_key="png",
        title_fn=lambda row: f"{row['graph_variant']}\nscore={row['score']:.3g}, nodes={row['nodes']}",
        output_name="aggressive_graph_contact_sheet.png",
    )


def make_contact_sheet(dir_out, rows, *, image_key, title_fn, output_name):
    if not rows:
        return
    images = []
    titles = []
    for row in rows:
        path = Path(row[image_key])
        if not path.exists():
            continue
        images.append(plt.imread(path))
        titles.append(title_fn(row))
    if not images:
        return
    cols = min(3, len(images))
    rows_n = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.4, rows_n * 4.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(images):]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(dir_out / output_name, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
