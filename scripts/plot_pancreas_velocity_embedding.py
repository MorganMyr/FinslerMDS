"""Plot a pancreas embedding with a sampled projected velocity field.

Examples:

    python scripts/plot_pancreas_velocity_embedding.py gd_2d_vrand2_r0p05_s42.npz
    python scripts/plot_pancreas_velocity_embedding.py umap_dynamical_s42.npy --point-fraction 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
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
from finsler_mds.utils.pancreas import (  # noqa: E402
    PANCREAS_CLUSTER_COLORS,
    project_velocity_to_embedding,
)
from scripts.evaluate_pancreas_embedding import (  # noqa: E402
    N_EVAL_NEIGHBORS,
    load_embedding,
    load_pancreas_evaluation_context,
    resolve_embedding_path,
)


# Defaults exposed here so quick visual tuning does not require touching argparse.
POINT_FRACTION = 0.07
VECTOR_LENGTH_FRACTION = 0.025
VELOCITY_SCALE = 1.0
VELOCITY_NORM_POWER = 0.4
N_NEIGHBORS = N_EVAL_NEIGHBORS
SEED = 42
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
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    pancreas_dir = script_dir / "res" / "pancreas"
    raw_dir = pancreas_dir / "raw"
    output_dir = pancreas_dir / "vector_plot"
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = resolve_embedding_path(args.embedding, raw_dir)
    embedding = load_embedding(embedding_path)
    context = load_pancreas_evaluation_context(
        raw_dir,
        pancreas_dir / "rna_velocity_evaluation",
        n_eval_neighbors=args.n_neighbors,
        load_adata=True,
    )
    if len(embedding) != len(context.labels):
        raise ValueError(
            f"Embedding has {len(embedding)} rows, but pancreas state has {len(context.labels)} cells."
        )

    rng = np.random.default_rng(args.seed)
    vector_idx = sample_indices(len(embedding), args.point_fraction, rng)
    velocity_embedding = project_velocity_to_embedding(
        context.adata,
        embedding,
        basis="vector_plot",
    )[vector_idx]
    velocity_embedding = temper_velocity_norms(
        velocity_embedding,
        power=args.velocity_norm_power,
    )
    velocity_embedding *= auto_velocity_scale(
        embedding,
        velocity_embedding,
        target_fraction=args.vector_length_fraction,
        multiplier=args.velocity_scale,
    )

    save_path = output_dir / f"{args.output_name or figure_stem(embedding_path)}.pdf"
    plot_embedding_with_vectors(
        embedding,
        labels=context.labels,
        vector_idx=vector_idx,
        velocity_embedding=velocity_embedding,
        title=args.title or embedding_path.stem,
        save_path=save_path,
        seed=args.seed,
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
        raise ValueError("--velocity-norm-power must be between 0 and 1.")
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
            xlabel="Finsler 1",
            ylabel="Finsler 2",
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
        raise ValueError("--point-fraction must be in (0, 1].")
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


def figure_stem(path):
    return re.sub(r"_s\d+$", "", Path(path).stem)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a saved pancreas embedding with sampled projected velocity vectors."
    )
    parser.add_argument(
        "embedding",
        help="Embedding filename. Relative names are first searched in scripts/res/pancreas/raw/.",
    )
    parser.add_argument("--point-fraction", type=float, default=POINT_FRACTION)
    parser.add_argument("--vector-length-fraction", type=float, default=VECTOR_LENGTH_FRACTION)
    parser.add_argument("--velocity-scale", type=float, default=VELOCITY_SCALE)
    parser.add_argument(
        "--velocity-norm-power",
        type=float,
        default=VELOCITY_NORM_POWER,
        help="Vector norm compression in [0, 1]: 0 gives equal arrow lengths, 1 keeps projected norms.",
    )
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    main()
