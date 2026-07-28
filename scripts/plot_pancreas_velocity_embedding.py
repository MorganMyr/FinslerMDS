"""Plot a pancreas embedding with a sampled projected velocity field.

Examples:

    python scripts/plot_pancreas_velocity_embedding.py gd_2d_vrand2_r0p05.npz
    python scripts/plot_pancreas_velocity_embedding.py umap_2d_k50_md0p5.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.utils import plot_3d_embedding_views, plot_categorical_embedding  # noqa: E402
from finsler_mds.utils.pancreas import PANCREAS_CLUSTER_COLORS  # noqa: E402
from finsler_mds.utils.pancreas_files import (  # noqa: E402
    load_pancreas_embedding,
    pancreas_artifact_dir,
    resolve_pancreas_embedding_path,
)
from scripts.evaluate_pancreas_embedding import (  # noqa: E402
    N_EVAL_NEIGHBORS,
    SEED,
    load_pancreas_evaluation_context,
    project_velocity_to_embedding_from_transition,
)


# Visual settings.
POINT_FRACTION = 0.07
VECTOR_LENGTH_FRACTION = 0.025
VELOCITY_SCALE = 1.0
VELOCITY_NORM_POWER = 0.4
N_NEIGHBORS = N_EVAL_NEIGHBORS
TITLE = None
OUTPUT_NAME = None

POINT_SIZE = 8
POINT_WHITENING = 0.4
VECTOR_COLOR = "black"
VECTOR_ALPHA = 0.72
VECTOR_WIDTH_2D = 0.004
VECTOR_LINEWIDTH_3D = 0.85
ARROW_LENGTH_RATIO_3D = 0.25


def main() -> None:
    embedding_name = parse_args().embedding
    script_dir = Path(__file__).resolve().parent
    pancreas_dir = script_dir / "res" / "pancreas"
    raw_dir = pancreas_dir / "raw"

    embedding_path = resolve_pancreas_embedding_path(embedding_name, raw_dir)
    output_dir = pancreas_artifact_dir(
        pancreas_dir / "vector_plot",
        embedding_path.name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_pancreas_evaluation_context(
        raw_dir,
        pancreas_dir / "rna_velocity_evaluation",
        n_eval_neighbors=N_NEIGHBORS,
    )
    embedding = load_pancreas_embedding(
        embedding_path,
        cell_ids=context.cell_ids,
    )

    rng = np.random.default_rng(SEED)
    vector_idx = sample_indices(len(embedding), POINT_FRACTION, rng)
    velocity_embedding = project_velocity_to_embedding_from_transition(
        context.velocity_transition,
        embedding,
    )[vector_idx]
    velocity_embedding = temper_velocity_norms(
        velocity_embedding,
        power=VELOCITY_NORM_POWER,
    )
    velocity_embedding *= auto_velocity_scale(
        embedding,
        velocity_embedding,
        target_fraction=VECTOR_LENGTH_FRACTION,
        multiplier=VELOCITY_SCALE,
    )

    save_path = output_dir / f"{OUTPUT_NAME or embedding_path.stem}.pdf"
    plot_embedding_with_vectors(
        embedding,
        labels=context.labels,
        vector_idx=vector_idx,
        velocity_embedding=velocity_embedding,
        title=TITLE or embedding_path.stem,
        save_path=save_path,
        seed=SEED,
    )
    print(f"Saved vector plot: {save_path}")


def auto_velocity_scale(embedding, velocity_embedding, *, target_fraction, multiplier):
    norms = np.linalg.norm(velocity_embedding, axis=1)
    nonzero = norms[norms > 0]
    if len(nonzero) == 0:
        return 1.0
    span = np.ptp(embedding, axis=0)
    target = float(target_fraction) * max(float(np.linalg.norm(span)), 1e-12)
    return float(multiplier) * target / float(np.median(nonzero))


def temper_velocity_norms(velocity_embedding, *, power):
    """Compress vector length variation while preserving directions."""
    power = float(power)
    if not (0 <= power <= 1):
        raise ValueError("VELOCITY_NORM_POWER must be between 0 and 1.")
    velocity_embedding = np.asarray(velocity_embedding, dtype=float).copy()
    norms = np.linalg.norm(velocity_embedding, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    if not np.any(nonzero):
        return velocity_embedding
    velocity_embedding[nonzero] /= norms[nonzero]
    velocity_embedding[nonzero] *= norms[nonzero] ** power
    return velocity_embedding


def plot_embedding_with_vectors(
        embedding,
        *,
        labels,
        vector_idx,
        velocity_embedding,
        title,
        save_path,
        seed,
):
    embedding = np.asarray(embedding, dtype=float)
    cmap = whiten_palette(PANCREAS_CLUSTER_COLORS, POINT_WHITENING)
    if embedding.shape[1] == 2:
        fig, ax = plot_categorical_embedding(
            embedding,
            labels=labels,
            title=title,
            cmap=cmap,
            s=POINT_SIZE,
        )
        ax.quiver(
            embedding[vector_idx, 0],
            embedding[vector_idx, 1],
            velocity_embedding[:, 0],
            velocity_embedding[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=VECTOR_COLOR,
            width=VECTOR_WIDTH_2D,
            alpha=VECTOR_ALPHA,
            zorder=4,
        )
    elif embedding.shape[1] == 3:
        fig, axes = plot_3d_embedding_views(
            embedding,
            labels=labels,
            title=title,
            save_path=None,
            point_fraction=1.0,
            random_state=seed,
            cmap=cmap,
            s=POINT_SIZE,
        )
        pts = embedding[vector_idx]
        vec = velocity_embedding
        for ax in axes:
            ax.quiver(
                pts[:, 0], pts[:, 1], pts[:, 2],
                vec[:, 0], vec[:, 1], vec[:, 2],
                color=VECTOR_COLOR,
                linewidth=VECTOR_LINEWIDTH_3D,
                arrow_length_ratio=ARROW_LENGTH_RATIO_3D,
                alpha=VECTOR_ALPHA,
                normalize=False,
            )
    else:
        raise ValueError(f"Expected a 2D or 3D embedding, got shape {embedding.shape}.")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def sample_indices(n_points, point_fraction, rng):
    point_fraction = float(point_fraction)
    if not (0 < point_fraction <= 1):
        raise ValueError("POINT_FRACTION must be in (0, 1].")
    n_sample = max(1, int(round(point_fraction * n_points)))
    return np.sort(rng.choice(n_points, size=n_sample, replace=False))


def whiten_palette(cmap, amount):
    amount = float(amount)
    if not (0 <= amount <= 1):
        raise ValueError("POINT_WHITENING must be between 0 and 1.")
    return {
        key: to_hex((1 - amount) * np.asarray(to_rgb(color)) + amount)
        for key, color in cmap.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "embedding",
        help="Embedding filename or NPZ path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
