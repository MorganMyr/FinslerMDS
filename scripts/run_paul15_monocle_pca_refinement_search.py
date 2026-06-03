"""Small search for cleaner PCA-based path-frozen Paul15 embeddings."""

from __future__ import annotations

from pathlib import Path
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

from finsler_mds import RandersMetric, fit_finsler_mds  # noqa: E402
from finsler_mds.utils import plot_categorical_embedding  # noqa: E402
from finsler_mds.utils.dissimilarity_graphs import density_scaled_knn_distances  # noqa: E402
from finsler_mds.utils.embedding_io import cache_token, scale_embedding_to_dissimilarities  # noqa: E402
from main_paul15_monocle import paul15_display_labels  # noqa: E402
from run_paul15_monocle_pca_mds_experiments import export_monocle_pca  # noqa: E402


def main():
    seed = 42
    dir_res = SCRIPT_DIR / "res" / "paul15" / "monocle3"
    dir_out = dir_res / "experiments_PCA"
    dir_candidates = dir_out / "search_candidates"
    input_dir = dir_res / "monocle_input_no19lymph_no11dc"
    pca_csv = dir_out / "monocle_pca_50_no19lymph_no11dc.csv"
    init_path = resolve_init_path(dir_candidates, dir_out)

    dir_candidates.mkdir(parents=True, exist_ok=True)
    export_monocle_pca(input_dir, pca_csv, n_components=50)

    pca = pd.read_csv(pca_csv)
    cell_ids = pca["cell_id"].astype(str).to_numpy()
    pca_values = pca.drop(columns=["cell_id"]).to_numpy(dtype=float)
    metadata = pd.read_csv(Path(input_dir) / "cell_metadata.csv").set_index("cell_id")
    labels = metadata.loc[cell_ids, "paul15_clusters"].astype(str).to_numpy()

    with np.load(init_path) as data:
        base_init = np.asarray(data["embedding"], dtype=float)
    if base_init.shape[0] != len(cell_ids):
        raise ValueError(f"Init has {base_init.shape[0]} cells, expected {len(cell_ids)}.")

    if init_path.name == "cont_k50_g1p0_lw2_embedding_raw.npz":
        variants = [
            variant("cont2_k35_g1p0_lw1", k=35, gamma=1.0, local_weight=1.0),
            variant("cont2_k50_g1p0_lw1", k=50, gamma=1.0, local_weight=1.0),
            variant("cont2_k50_g1p25_lw1", k=50, gamma=1.25, local_weight=1.0),
            variant("cont2_k50_g1p0_lw0p5", k=50, gamma=1.0, local_weight=0.5),
        ]
    elif init_path.parent == dir_candidates:
        variants = [
            variant("cont_k50_g0p75_lw2", k=50, gamma=0.75, local_weight=2.0),
            variant("cont_k50_g1p0_lw2", k=50, gamma=1.0, local_weight=2.0),
            variant("cont_k70_g0p75_lw2", k=70, gamma=0.75, local_weight=2.0),
            variant("cont_k70_g1p0_lw2", k=70, gamma=1.0, local_weight=2.0),
            variant("cont_k100_g0p75_lw2", k=100, gamma=0.75, local_weight=2.0),
            variant("cont_k50_g0p75_lw1", k=50, gamma=0.75, local_weight=1.0),
        ]
    else:
        variants = [
        # Larger PCA neighborhoods should smooth the source geometry relative to
        # the noisy k=15 baseline.  Lower local_weight lets global landmarks
        # clean up some of the local PCA roughness.
        variant("k25_g0p0_lw10", k=25, gamma=0.0, local_weight=10.0),
        variant("k25_g0p5_lw10", k=25, gamma=0.5, local_weight=10.0),
        variant("k25_g0p75_lw10", k=25, gamma=0.75, local_weight=10.0),
        variant("k35_g0p5_lw10", k=35, gamma=0.5, local_weight=10.0),
        variant("k35_g0p75_lw10", k=35, gamma=0.75, local_weight=10.0),
        variant("k35_g0p75_lw5", k=35, gamma=0.75, local_weight=5.0),
        variant("k50_g0p5_lw5", k=50, gamma=0.5, local_weight=5.0),
        variant("k50_g0p75_lw5", k=50, gamma=0.75, local_weight=5.0),
        variant("k70_g0p5_lw5", k=70, gamma=0.5, local_weight=5.0),
        ]

    rows = []
    for spec in variants:
        D = load_or_build_dissimilarities(
            pca_values,
            cell_ids,
            labels,
            dir_candidates,
            k=spec["k"],
            gamma=spec["gamma"],
        )
        rows.append(
            run_variant(
                spec,
                D,
                base_init,
                cell_ids,
                labels,
                dir_candidates,
                seed=seed,
            )
        )
        pd.DataFrame(rows).to_csv(dir_candidates / "pca_refinement_search_summary.csv", index=False)

    make_contact_sheet(dir_candidates, rows)
    print(f"Saved PCA refinement candidates in: {dir_candidates}")


def variant(name, *, k, gamma, local_weight):
    return {"name": name, "k": int(k), "gamma": float(gamma), "local_weight": float(local_weight)}


def resolve_init_path(dir_candidates, dir_out):
    for name in (
        "cont_k50_g1p0_lw2_embedding_raw.npz",
        "k50_g0p75_lw5_embedding_raw.npz",
    ):
        continuation = dir_candidates / name
        if continuation.exists():
            print(f"Continuing search from candidate: {continuation}")
            return continuation
    baseline = dir_out / "pca_pf_from_monocle_umap_m50_i5_embedding_raw.npz"
    print(f"Starting search from PCA baseline: {baseline}")
    return baseline


def load_or_build_dissimilarities(pca_values, cell_ids, labels, dir_candidates, *, k, gamma):
    path = dir_candidates / f"pca_knn{k}_g{cache_token(gamma)}_cosine_dissimilarities.npz"
    if path.exists():
        with np.load(path) as data:
            return np.asarray(data["dissimilarities"], dtype=float)

    graph = pca_knn_graph(pca_values, n_neighbors=k)
    graph, density_info = density_scaled_knn_distances(graph, gamma=gamma, mode="symmetric")
    n_components, component = connected_components(graph, directed=False, connection="weak")
    if n_components != 1:
        counts = np.bincount(component)
        raise ValueError(
            f"PCA kNN graph k={k}, gamma={gamma} disconnected "
            f"({n_components} components, largest={counts.max()})."
        )
    D = shortest_path(graph, directed=False, return_predecessors=False)
    if not np.all(np.isfinite(D)):
        raise ValueError(f"PCA shortest paths k={k}, gamma={gamma} contain non-finite values.")
    np.fill_diagonal(D, 0.0)
    np.savez(
        path,
        dissimilarities=np.asarray(D, dtype=float),
        cell_ids=cell_ids,
        labels=labels,
        k=np.asarray(k),
        gamma=np.asarray(gamma),
        metric=np.asarray("cosine"),
        median_sigma=np.asarray(density_info["median_sigma"]),
        min_sigma=np.asarray(density_info["min_sigma"]),
        max_sigma=np.asarray(density_info["max_sigma"]),
    )
    print(f"Saved dissimilarities: {path.name}")
    return np.asarray(D, dtype=float)


def pca_knn_graph(pca_values, *, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine").fit(pca_values)
    distances, indices = nn.kneighbors(pca_values)
    rows = np.repeat(np.arange(len(pca_values)), n_neighbors)
    cols = indices[:, 1:].reshape(-1)
    vals = distances[:, 1:].reshape(-1)
    graph = csr_matrix((vals, (rows, cols)), shape=(len(pca_values), len(pca_values)))
    return graph.maximum(graph.T).tocsr()


def run_variant(spec, D, base_init, cell_ids, labels, dir_candidates, *, seed):
    name = spec["name"]
    out_npz = dir_candidates / f"{name}_embedding_raw.npz"
    out_pdf = dir_candidates / f"{name}_clusters.pdf"
    out_png = dir_candidates / f"{name}_clusters.png"

    if out_npz.exists():
        with np.load(out_npz) as data:
            embedding = np.asarray(data["embedding"], dtype=float)
            stress = float(data["stress"])
            elapsed = float(data["elapsed"])
        print(f"Loaded cached candidate: {name}")
    else:
        init, scale = scale_embedding_to_dissimilarities(base_init, D, random_state=seed)
        print(f"{name}: rescaled PCA embedding init by {scale:.6g}")
        start = perf_counter()
        embedding, stress = fit_finsler_mds(
            D,
            metric=RandersMetric(alpha=0.0),
            optimizer="path_frozen",
            init=init,
            n_components=2,
            graph_neighbors=30,
            max_iter=20,
            inner_iter=2,
            eps=1e-6,
            method="L-BFGS-B",
            optimizer_options={"ftol": 1e-8, "maxls": 30},
            n_global_landmarks=150,
            n_local_neighbors=40,
            local_pair_mode="direct",
            max_global_targets_per_source=220,
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
            source_init=np.asarray("pca_pf_from_monocle_umap_m50_i5"),
            k=np.asarray(spec["k"]),
            gamma=np.asarray(spec["gamma"]),
            local_weight=np.asarray(spec["local_weight"]),
            max_iter=np.asarray(20),
            inner_iter=np.asarray(2),
        )
        print(f"Saved candidate: {out_npz}")

    save_cluster_plot(embedding, labels, out_pdf, out_png, title=name)
    return {
        "name": name,
        "k": spec["k"],
        "gamma": spec["gamma"],
        "local_weight": spec["local_weight"],
        "stress": stress,
        "elapsed": elapsed,
        "npz": str(out_npz),
        "pdf": str(out_pdf),
        "png": str(out_png),
    }


def save_cluster_plot(embedding, labels, out_pdf, out_png, *, title):
    display_labels = paul15_display_labels(labels)
    fig, _ = plot_categorical_embedding(embedding, labels=display_labels, title=title, s=8)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)


def make_contact_sheet(dir_candidates, rows):
    if not rows:
        return
    images = []
    titles = []
    for row in rows:
        path = Path(row["png"])
        if not path.exists():
            continue
        images.append(plt.imread(path))
        titles.append(f"{row['name']}\nstress={row['stress']:.3g}")

    n = len(images)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 4.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(images):]:
        ax.set_axis_off()
    out = dir_candidates / "pca_refinement_contact_sheet.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved contact sheet: {out}")


if __name__ == "__main__":
    main()
