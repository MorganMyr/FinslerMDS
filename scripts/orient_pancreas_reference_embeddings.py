"""Orient pancreas UMAP/Isomap inits so mean projected velocity points down."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.utils import rotate_embedding_to_mean_velocity_down  # noqa: E402
from finsler_mds.utils.pancreas import project_velocity_to_embedding  # noqa: E402
from finsler_mds.utils.pancreas_campaign import (  # noqa: E402
    PANCREAS_ISOMAP,
    PANCREAS_SEED,
    PANCREAS_UMAP,
)
from scripts.evaluate_pancreas_embedding import load_pancreas_evaluation_context  # noqa: E402
from scripts.main_pancreas import (  # noqa: E402
    cache_token,
    compute_isomap_from_pca,
    pancreas_isomap_embedding_path,
    pancreas_umap_embedding_path,
    pancreas_umap_variant_tag,
    plot_pancreas_finsler_embedding,
)


def main():
    script_dir = Path(__file__).resolve().parent
    pancreas_dir = script_dir / "res" / "pancreas"
    raw_dir = pancreas_dir / "raw"
    eval_dir = pancreas_dir / "rna_velocity_evaluation"

    context = load_pancreas_evaluation_context(raw_dir, eval_dir, load_adata=True)
    labels = context.labels

    # Ensure Isomap files exist before orientation.
    ensure_isomap(raw_dir, pancreas_dir, context.x_pca, labels, 2)
    ensure_isomap(raw_dir, pancreas_dir, context.x_pca, labels, 3)

    specs = [
        ("UMAP 2D", umap_embedding_path(raw_dir, 2), context.adata, labels, pancreas_dir / "umap.pdf"),
        ("Isomap 2D", isomap_embedding_path(raw_dir, 2), context.adata, labels, pancreas_dir / "isomap.pdf"),
        ("UMAP 3D", umap_embedding_path(raw_dir, 3), context.adata, labels, pancreas_dir / "umap_3d.pdf"),
        ("Isomap 3D", isomap_embedding_path(raw_dir, 3), context.adata, labels, pancreas_dir / "isomap_3d.pdf"),
    ]
    for name, path, adata, current_labels, fig_path in specs:
        orient_embedding_file(name, path, adata, current_labels, fig_path)


def orient_embedding_file(name, path, adata, labels, fig_path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    embedding = np.asarray(np.load(path), dtype=float)
    velocity_embedding = project_velocity_to_embedding(adata, embedding)
    oriented, rotation, mean_velocity = rotate_embedding_to_mean_velocity_down(
        embedding,
        velocity_embedding,
    )
    oriented_mean = np.nanmean(velocity_embedding @ rotation.T, axis=0)
    np.save(path, oriented)
    plot_pancreas_finsler_embedding(
        oriented,
        labels=labels,
        title=f"Pancreas {name} velocity-oriented",
        save_path=fig_path,
        random_state=PANCREAS_SEED,
    )
    print(
        f"{name}: {path.name} oriented; mean velocity "
        f"{np.array2string(mean_velocity, precision=4)} -> "
        f"{np.array2string(oriented_mean, precision=4)}"
    )
    print(f"Saved figure: {fig_path}")


def ensure_isomap(raw_dir, pancreas_dir, x_pca, labels, embedding_dim):
    path = isomap_embedding_path(raw_dir, embedding_dim)
    if Path(path).exists():
        return
    embedding = compute_isomap_from_pca(
        x_pca,
        PANCREAS_ISOMAP,
        n_components=embedding_dim,
    )
    np.save(path, embedding)
    plot_pancreas_finsler_embedding(
        embedding,
        labels=labels,
        title=f"Pancreas Isomap {embedding_dim}D",
        save_path=pancreas_dir / ("isomap.pdf" if embedding_dim == 2 else "isomap_3d.pdf"),
        random_state=PANCREAS_SEED,
    )


def umap_embedding_path(raw_dir, embedding_dim):
    return Path(
        pancreas_umap_embedding_path(
            raw_dir,
            f"{cache_token('dynamical')}_{pancreas_umap_variant_tag(PANCREAS_UMAP)}s{PANCREAS_SEED}",
            n_components=embedding_dim,
            dataset_prefix="pancreas",
        )
    )


def isomap_embedding_path(raw_dir, embedding_dim):
    return Path(
        pancreas_isomap_embedding_path(
            raw_dir,
            f"k{PANCREAS_ISOMAP['n_neighbors']}_s{PANCREAS_SEED}",
            n_components=embedding_dim,
            dataset_prefix="pancreas",
        )
    )


if __name__ == "__main__":
    main()
