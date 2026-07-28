"""Compare full and gapped pancreas embeddings with the gap-distance metric."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import fit_finsler_mds  # noqa: E402
from finsler_mds.evaluation.rna_velocity import normalized_gap_distance  # noqa: E402
from finsler_mds.evaluation.rna_velocity.pancreas_gap import (  # noqa: E402
    normalize_pancreas_gap_config,
    pancreas_gap_prefix,
    select_pancreas_gap,
)
from finsler_mds.utils.pancreas import (  # noqa: E402
    PANCREAS_CLUSTER_COLORS,
    compute_pancreas_umap,
    compute_pancreas_velocity_graph,
    embedding_metric_tag,
    load_or_compute_pancreas_inputs,
    load_or_compute_pancreas_state,
    make_embedding_metric,
    normalize_embedding_dim,
    orient_pancreas_embedding_by_velocity,
    pancreas_cache_metadata,
    velocity_distance_formula_tag,
)
from finsler_mds.utils.plotting import set_axes_equal  # noqa: E402
from scripts.main_pancreas import CONFIG as PANCREAS_CONFIG  # noqa: E402


# Choose among "umap", "finsler_umap", and "gradient_descent".
METHOD = "umap"
GAP = {
    "enabled": True,
    "name": "preendocrine",
    "selection": "veloviz_latent_time",
    "n_before": 300,
    "n_after": 300,
    "removed_labels": ("Pre-endocrine",),
    "before_labels": ("Ngn3 high EP",),
    "after_labels": ("Alpha", "Beta", "Delta", "Epsilon"),
}

PANCREAS_DIR = Path(__file__).parent / "res" / "pancreas"
RAW_DIR = PANCREAS_DIR / "raw"
GAP_RAW_DIR = PANCREAS_DIR / "gap" / "raw"


def main():
    method = normalize_method(METHOD)
    gap = normalize_pancreas_gap_config(GAP)
    seed = int(PANCREAS_CONFIG["seed"])
    n_components = normalize_embedding_dim(PANCREAS_CONFIG["embedding_dim"])
    preprocessing = PANCREAS_CONFIG["preprocessing"]
    velocity = PANCREAS_CONFIG["velocity"]

    full_state = load_or_compute_pancreas_state(
        RAW_DIR,
        preprocessing=preprocessing,
        velocity=velocity,
        seed=seed,
    )
    selection = make_gap_selection(full_state, gap, velocity)
    gap_prefix = pancreas_gap_prefix(gap)
    gap_metadata = {
        "gap": gap,
        "selection_hash": hashlib.sha256("\0".join(selection.cell_ids).encode()).hexdigest(),
    }
    gap_state = load_or_compute_pancreas_state(
        GAP_RAW_DIR,
        preprocessing=preprocessing,
        velocity=velocity,
        seed=seed,
        dataset_prefix=gap_prefix,
        prepare_dataset=gap_dataset_preparer(full_state[2], selection),
        metadata_extra=gap_metadata,
    )

    metric = make_embedding_metric(
        PANCREAS_CONFIG["finsler_metric"],
        PANCREAS_CONFIG["alpha_embedding"],
    )
    full_embedding = compute_embedding(
        full_state,
        RAW_DIR,
        method=method,
        metric=metric,
        n_components=n_components,
        dataset_prefix="pancreas",
        metadata_extra=None,
    )
    gap_embedding = compute_embedding(
        gap_state,
        GAP_RAW_DIR,
        method=method,
        metric=metric,
        n_components=n_components,
        dataset_prefix=gap_prefix,
        metadata_extra=gap_metadata,
    )

    full_result = normalized_gap_distance(
        full_embedding,
        selection.before_original_indices,
        selection.after_original_indices,
    )
    gap_result = normalized_gap_distance(
        gap_embedding,
        selection.before_indices,
        selection.after_indices,
    )
    print_gap_result("Full data", full_result)
    print_gap_result("Gapped data", gap_result)
    print(
        "Normalized gap increase: "
        f"{gap_result.normalized_distance - full_result.normalized_distance:.6f}"
    )

    figure_path = PANCREAS_DIR / "gap" / gap_figure_name(
        method,
        n_components,
        velocity,
        metric,
    )
    save_comparison_figure(
        full_embedding,
        gap_embedding,
        full_labels=full_state[1],
        gap_labels=gap_state[1],
        full_result=full_result,
        gap_result=gap_result,
        path=figure_path,
    )
    print(f"Saved comparison figure: {figure_path}")


def normalize_method(method):
    aliases = {"gd": "gradient_descent", "finsler_mds": "gradient_descent"}
    method = str(method).lower().replace("-", "_")
    method = aliases.get(method, method)
    if method not in {"umap", "finsler_umap", "gradient_descent"}:
        raise ValueError("METHOD must be 'umap', 'finsler_umap', or 'gradient_descent'.")
    return method


def gap_figure_name(method, n_components, velocity, metric):
    method_tag = {"gradient_descent": "gd", "finsler_umap": "fumap"}.get(method, method)
    parts = ["gap_distance", method_tag, f"{n_components}d"]
    if method != "umap":
        parts.extend(
            (
                velocity_distance_formula_tag(
                    velocity["distance_formula"],
                    velocity["alpha"],
                ),
                embedding_metric_tag(metric),
            )
        )
    return "_".join(parts) + ".pdf"


def make_gap_selection(full_state, gap, velocity):
    adata, labels, cell_ids, _, _, state_metadata = full_state
    ordering = None
    if gap["selection"] in {"latent_time", "veloviz_latent_time"}:
        ordering = load_or_compute_gap_ordering(
            adata,
            GAP_RAW_DIR / f"{pancreas_gap_prefix(gap)}_ordering.npz",
            metadata=pancreas_cache_metadata(
                state=state_metadata,
                gap=gap,
                velocity_neighbors=velocity["velocity_neighbors"],
            ),
            velocity=velocity,
        )
    return select_pancreas_gap(labels, gap, cell_ids=cell_ids, ordering=ordering)


def gap_dataset_preparer(full_cell_ids, selection):
    def prepare(adata):
        if not np.array_equal(np.asarray(adata.obs_names, dtype=str), full_cell_ids):
            raise ValueError("Fresh pancreas data do not match the cached full dataset.")
        return (
            adata[selection.keep_mask].copy(),
            selection.labels,
            selection.cell_ids,
            selection.original_indices,
        )

    return prepare


def load_or_compute_gap_ordering(adata, path, *, metadata, velocity):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if (
                "metadata_json" in cache
                and json.loads(str(cache["metadata_json"].item())) == metadata
            ):
                return np.asarray(cache["latent_time"], dtype=float)

    if "velocity_graph" not in adata.uns:
        compute_pancreas_velocity_graph(
            adata,
            n_neighbors=velocity["velocity_neighbors"],
            n_jobs=velocity["graph_n_jobs"],
        )
    import scvelo as scv

    scv.tl.latent_time(adata)
    ordering = np.asarray(adata.obs["latent_time"], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, latent_time=ordering, metadata_json=json.dumps(metadata, sort_keys=True))
    return ordering


def compute_embedding(
        state,
        raw_dir,
        *,
        method,
        metric,
        n_components,
        dataset_prefix,
        metadata_extra,
):
    adata = state[0]
    seed = int(PANCREAS_CONFIG["seed"])
    preprocessing = PANCREAS_CONFIG["preprocessing"]
    velocity = PANCREAS_CONFIG["velocity"]
    umap = compute_pancreas_umap(
        adata,
        preprocessing=preprocessing,
        umap=PANCREAS_CONFIG["umap"],
        n_components=n_components,
        seed=seed,
    )
    umap = orient_pancreas_embedding_by_velocity(
        adata,
        umap,
        velocity=velocity,
        label=f"{dataset_prefix} UMAP",
    )
    if method == "umap":
        return umap

    inputs = load_or_compute_pancreas_inputs(
        raw_dir,
        preprocessing=preprocessing,
        velocity=velocity,
        seed=seed,
        dataset_prefix=dataset_prefix,
        state=state,
        metadata_extra=metadata_extra,
    )
    embedding, objective = fit_finsler_mds(
        inputs.dissimilarities,
        metric=metric,
        optimizer=method,
        init=umap,
        n_components=n_components,
        random_state=seed,
        print_time=True,
        **PANCREAS_CONFIG[method],
    )
    print(f"{dataset_prefix} {method} objective: {objective}")
    return orient_pancreas_embedding_by_velocity(
        adata,
        embedding,
        velocity=velocity,
        label=f"{dataset_prefix} {method}",
    )


def print_gap_result(label, result):
    print(
        f"{label}: normalized_gap={result.normalized_distance:.6f}, "
        f"raw_gap={result.distance:.6f}, n_before={result.n_before}, "
        f"n_after={result.n_after}"
    )


def save_comparison_figure(
        full_embedding,
        gap_embedding,
        *,
        full_labels,
        gap_labels,
        full_result,
        gap_result,
        path,
):
    n_components = full_embedding.shape[1]
    if gap_embedding.shape[1] != n_components or n_components not in {2, 3}:
        raise ValueError("Full and gapped embeddings must both be 2D or both be 3D.")
    fig = plt.figure(figsize=(13, 5.5))
    axes = [
        fig.add_subplot(1, 2, index, projection="3d" if n_components == 3 else None)
        for index in (1, 2)
    ]
    panels = (
        (axes[0], full_embedding, full_labels, full_result, "Full data"),
        (axes[1], gap_embedding, gap_labels, gap_result, "Gapped data"),
    )
    for ax, embedding, labels, result, title in panels:
        colors = [PANCREAS_CLUSTER_COLORS.get(str(label), "#999999") for label in labels]
        before, after = result.before_representative, result.after_representative
        representative = embedding[[before, after]]
        if n_components == 2:
            ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=8, alpha=0.8)
            ax.plot(*representative.T, color="black", linewidth=1.5)
            ax.scatter(
                *representative.T,
                s=55,
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
                zorder=3,
            )
            ax.set_aspect("equal", adjustable="datalim")
        else:
            ax.scatter(*embedding.T, c=colors, s=8, alpha=0.8)
            ax.plot(*representative.T, color="black", linewidth=1.5)
            ax.scatter(
                *representative.T,
                s=55,
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
            )
            ax.set_zticks([])
            set_axes_equal(ax)
        ax.set_title(f"{title}\nnormalized gap = {result.normalized_distance:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])

    labels = sorted(set(map(str, full_labels)) | set(map(str, gap_labels)))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=PANCREAS_CLUSTER_COLORS.get(label, "#999999"),
            label=label,
        )
        for label in labels
    ]
    fig.legend(handles=handles, loc="center right", frameon=False)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
