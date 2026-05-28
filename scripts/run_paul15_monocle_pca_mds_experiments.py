"""Path-frozen MDS-geodesic on the PCA geometry used by Monocle UMAP."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finsler_mds import RandersMetric, fit_finsler_mds, geodesic_embedding_stress  # noqa: E402
from finsler_mds.utils import plot_categorical_embedding  # noqa: E402
from finsler_mds.utils.embedding_io import scale_embedding_to_dissimilarities  # noqa: E402
from main_paul15_monocle import paul15_display_labels  # noqa: E402


def main():
    seed = 42
    dir_res = SCRIPT_DIR / "res" / "paul15" / "monocle3"
    dir_raw = dir_res / "raw"
    dir_exp = dir_res / "experiments"
    dir_out = dir_res / "experiments_PCA"
    input_dir = dir_res / "monocle_input_no19lymph_no11dc"
    dir_out.mkdir(parents=True, exist_ok=True)

    pca_csv = dir_out / "monocle_pca_50_no19lymph_no11dc.csv"
    dissimilarities_path = dir_out / "monocle_pca_knn15_cosine_dissimilarities.npz"
    export_monocle_pca(input_dir, pca_csv, n_components=50)

    if dissimilarities_path.exists():
        with np.load(dissimilarities_path) as data:
            D = np.asarray(data["dissimilarities"], dtype=float)
            cell_ids = data["cell_ids"].astype(str)
            labels = data["labels"].astype(str)
        print(f"Loaded PCA-kNN dissimilarities: {dissimilarities_path}")
    else:
        D, cell_ids, labels = build_pca_knn_dissimilarities(pca_csv, input_dir, n_neighbors=15)
        np.savez(
            dissimilarities_path,
            dissimilarities=D,
            cell_ids=cell_ids,
            labels=labels,
            n_neighbors=np.asarray(15),
            metric=np.asarray("cosine"),
            source=np.asarray("Monocle preprocess_cds PCA, num_dim=50"),
        )
        print(f"Saved PCA-kNN dissimilarities: {dissimilarities_path}")

    common_path_frozen = {
        "graph_neighbors": 30,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 30},
        "local_pair_mode": "direct",
        "global_target_sampling": "random",
        "local_global_reweighting": "count",
        "device": "auto",
        "verbose": 1,
        # Same family as the selected best_deep run.
        "n_global_landmarks": 150,
        "max_global_targets_per_source": 220,
        "n_local_neighbors": 40,
        "local_weight": 25.0,
    }

    experiments = [
        {
            "name": "pca_pf_from_best_deep_m20_i5",
            "init": load_best_deep_init(dir_exp, cell_ids),
            "max_iter": 20,
            "inner_iter": 5,
        },
        {
            "name": "pca_pf_from_monocle_umap_m50_i5",
            "init": load_monocle_umap_init(dir_raw, cell_ids),
            "max_iter": 50,
            "inner_iter": 5,
        },
    ]

    rows = []
    for experiment in experiments:
        rows.append(
            run_path_frozen_experiment(
                experiment,
                D,
                cell_ids,
                labels,
                dir_out,
                path_frozen={
                    **common_path_frozen,
                    "max_iter": experiment["max_iter"],
                    "inner_iter": experiment["inner_iter"],
                },
                seed=seed,
            )
        )
        pd.DataFrame(rows).to_csv(dir_out / "pca_path_frozen_summary.csv", index=False)

    print(f"Saved PCA experiments in: {dir_out}")


def export_monocle_pca(input_dir, output_csv, *, n_components):
    if output_csv.exists():
        print(f"Loaded cached Monocle PCA export: {output_csv}")
        return

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript was not found; cannot export the exact Monocle PCA.")

    helper = output_csv.parent / "_export_monocle_pca.R"
    helper.write_text(
        r'''
suppressPackageStartupMessages({
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Missing R package 'Matrix'.")
  }
  if (!requireNamespace("monocle3", quietly = TRUE)) {
    stop("Missing R package 'monocle3'.")
  }
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Missing R package 'SingleCellExperiment'.")
  }
  if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
    stop("Missing R package 'SummarizedExperiment'.")
  }
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[[1]]
output_csv <- args[[2]]
num_dim <- as.integer(args[[3]])

expr <- Matrix::readMM(file.path(input_dir, "expression_gene_by_cell.mtx"))
cell_metadata <- read.csv(file.path(input_dir, "cell_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
gene_metadata <- read.csv(file.path(input_dir, "gene_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
if (!("gene_short_name" %in% colnames(gene_metadata))) {
  gene_metadata$gene_short_name <- gene_metadata$gene_id
}
rownames(cell_metadata) <- cell_metadata$cell_id
rownames(gene_metadata) <- gene_metadata$gene_id
rownames(expr) <- gene_metadata$gene_id
colnames(expr) <- cell_metadata$cell_id

cds <- monocle3::new_cell_data_set(
  expression_data = expr,
  cell_metadata = cell_metadata,
  gene_metadata = gene_metadata
)
set.seed(42)
cds <- monocle3::preprocess_cds(cds, num_dim = num_dim)
pca <- SingleCellExperiment::reducedDims(cds)$PCA
if (is.null(pca)) {
  stop("Monocle did not expose a PCA reduced dimension.")
}
out <- data.frame(cell_id = colnames(cds), pca, stringsAsFactors = FALSE, check.names = FALSE)
colnames(out) <- c("cell_id", paste0("PC", seq_len(ncol(pca))))
write.csv(out, output_csv, row.names = FALSE)
message("Saved Monocle PCA: ", output_csv)
''',
        encoding="utf-8",
    )

    command = [rscript, str(helper), str(input_dir), str(output_csv), str(n_components)]
    print("Exporting Monocle PCA:", " ".join(command))
    subprocess.run(command, check=True)
    try:
        helper.unlink()
    except FileNotFoundError:
        pass


def build_pca_knn_dissimilarities(pca_csv, input_dir, *, n_neighbors):
    pca = pd.read_csv(pca_csv)
    cell_ids = pca["cell_id"].astype(str).to_numpy()
    X = pca.drop(columns=["cell_id"]).to_numpy(dtype=float)

    metadata = pd.read_csv(Path(input_dir) / "cell_metadata.csv").set_index("cell_id")
    labels = metadata.loc[cell_ids, "paul15_clusters"].astype(str).to_numpy()

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine").fit(X)
    distances, indices = nn.kneighbors(X)
    rows = np.repeat(np.arange(len(X)), n_neighbors)
    cols = indices[:, 1:].reshape(-1)
    vals = distances[:, 1:].reshape(-1)
    graph = csr_matrix((vals, (rows, cols)), shape=(len(X), len(X)))
    graph = graph.maximum(graph.T).tocsr()

    n_components, component = connected_components(graph, directed=False, connection="weak")
    if n_components != 1:
        counts = np.bincount(component)
        raise ValueError(
            "Monocle-PCA kNN graph is disconnected "
            f"({n_components} components, largest={counts.max()})."
        )

    D = shortest_path(graph, directed=False, return_predecessors=False)
    if not np.all(np.isfinite(D)):
        raise ValueError("PCA shortest-path distances contain non-finite values.")
    np.fill_diagonal(D, 0.0)
    return np.asarray(D, dtype=float), cell_ids, labels


def load_best_deep_init(dir_exp, cell_ids):
    path = dir_exp / "best_deep_refined_embedding_raw.npz"
    with np.load(path) as data:
        init = np.asarray(data["embedding"], dtype=float)
    if init.shape[0] != len(cell_ids):
        raise ValueError(f"Best-deep init has {init.shape[0]} cells, expected {len(cell_ids)}.")
    return init


def load_monocle_umap_init(dir_raw, cell_ids):
    path = dir_raw / "monocle_umap_pseudotime_no19lymph_no11dc.csv"
    table = pd.read_csv(path).set_index("cell_id")
    missing = [cell_id for cell_id in cell_ids if cell_id not in table.index]
    if missing:
        raise ValueError(f"Monocle UMAP init is missing {len(missing)} cells, e.g. {missing[:5]}.")
    return table.loc[cell_ids, ["dim1", "dim2"]].to_numpy(dtype=float)


def run_path_frozen_experiment(experiment, D, cell_ids, labels, dir_out, *, path_frozen, seed):
    name = experiment["name"]
    out_npz = dir_out / f"{name}_embedding_raw.npz"
    out_pdf = dir_out / f"{name}_clusters.pdf"

    if out_npz.exists():
        with np.load(out_npz) as data:
            embedding = np.asarray(data["embedding"], dtype=float)
            stress = float(data["stress"])
            elapsed = float(data["elapsed"])
        print(f"Loaded cached embedding: {out_npz}")
    else:
        init, scale = scale_embedding_to_dissimilarities(experiment["init"], D, random_state=seed)
        print(f"{name}: rescaled init by {scale:.6g}")
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
            **path_frozen,
        )
        elapsed = perf_counter() - start
        full_stress = geodesic_embedding_stress(
            embedding,
            D,
            metric=RandersMetric(alpha=0.0),
            n_neighbors=path_frozen["graph_neighbors"],
            on_unreachable="warn_skip",
        )
        np.savez(
            out_npz,
            embedding=embedding,
            stress=np.asarray(stress),
            full_geodesic_stress=np.asarray(full_stress),
            elapsed=np.asarray(elapsed),
            cell_ids=cell_ids,
            init_kind=np.asarray(name),
            pca_graph_neighbors=np.asarray(15),
            pca_graph_metric=np.asarray("cosine"),
            path_frozen=np.asarray(str(path_frozen)),
        )
        print(f"Saved embedding: {out_npz}")

    save_cluster_plot(embedding, labels, out_pdf, title=name)
    return {
        "name": name,
        "npz": str(out_npz),
        "pdf": str(out_pdf),
        "stress": stress,
        "elapsed": elapsed,
        "max_iter": path_frozen["max_iter"],
        "inner_iter": path_frozen["inner_iter"],
        "graph_neighbors": path_frozen["graph_neighbors"],
        "n_global_landmarks": path_frozen["n_global_landmarks"],
        "max_global_targets_per_source": path_frozen["max_global_targets_per_source"],
        "n_local_neighbors": path_frozen["n_local_neighbors"],
        "local_weight": path_frozen["local_weight"],
    }


def save_cluster_plot(embedding, labels, out_pdf, *, title):
    display_labels = paul15_display_labels(labels)
    fig, _ = plot_categorical_embedding(
        embedding,
        labels=display_labels,
        title=title,
        s=8,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cluster plot: {out_pdf}")


if __name__ == "__main__":
    main()
