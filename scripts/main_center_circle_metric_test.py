from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.api import fit_finsler_mds
from finsler_mds.metrics import MatsumotoMetric, RandersMetric


SEED = 42
N_CIRCLE = 60
RADIUS = 1.0
CENTER_PAIR_WEIGHT = 20.0

SOURCE_ALPHA = 0.3
EMBEDDING_ALPHA = 0.3
GD_MAX_ITER = 100

RESULT_DIR = Path(__file__).resolve().parent / "res" / "center_circle_metric_test"
FIG_DIR = RESULT_DIR / "figures"


def main():
    np.random.seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, point_values = make_center_circle()
    init = X.copy()
    weights = center_pair_weights(len(X), CENTER_PAIR_WEIGHT)

    for source_name in ("randers", "matsumoto"):
        source_metric = make_metric(source_name, SOURCE_ALPHA)
        dissimilarities = source_metric.pairwise(X)

        for embedding_name in ("randers", "matsumoto"):
            embedding_metric = make_metric(embedding_name, EMBEDDING_ALPHA)
            tag = f"{metric_tag(source_name, SOURCE_ALPHA)}_{metric_tag(embedding_name, EMBEDDING_ALPHA)}"
            print(f"Running {source_name} -> {embedding_name} ({tag})")
            embedding, stress, n_iter = fit_finsler_mds(
                dissimilarities,
                metric=embedding_metric,
                optimizer="gradient_descent",
                n_components=2,
                init=init,
                weight=weights,
                max_iter=GD_MAX_ITER,
                eps=1e-8,
                method="L-BFGS-B",
                optimizer_options={"ftol": 1e-10, "maxls": 80, "maxcor": 30},
                device="auto",
                gpu_block_size=128,
                random_state=SEED,
                verbose=0,
                return_n_iter=True,
                print_time=True,
            )
            plot_embedding(
                embedding,
                point_values,
                title=f"{source_name.title()} -> {embedding_name.title()}",
                subtitle=f"stress={stress:.4g}, nit={n_iter}, center weight={CENTER_PAIR_WEIGHT:g}",
                path=FIG_DIR / f"center_circle_{tag}.pdf",
            )


def make_center_circle():
    theta = np.linspace(0.0, 2.0 * np.pi, N_CIRCLE, endpoint=False)
    circle = RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])
    X = np.vstack([np.zeros((1, 2)), circle])
    point_values = np.concatenate([[np.nan], theta])
    return X, point_values


def center_pair_weights(n_points, factor):
    weights = np.ones((n_points, n_points), dtype=float)
    np.fill_diagonal(weights, 0.0)
    weights[0, 1:] *= factor
    weights[1:, 0] *= factor
    return weights


def make_metric(name, alpha):
    if name == "randers":
        return RandersMetric(alpha=alpha)
    if name == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    raise ValueError("Metric name must be 'randers' or 'matsumoto'.")


def metric_tag(name, alpha):
    prefix = {"randers": "r", "matsumoto": "mats"}[name]
    return f"{prefix}{cache_token(alpha)}"


def cache_token(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def plot_embedding(embedding, point_values, *, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    circle_mask = np.isfinite(point_values)
    scatter = ax.scatter(
        embedding[circle_mask, 0],
        embedding[circle_mask, 1],
        c=point_values[circle_mask],
        cmap="hsv",
        s=34,
        edgecolors="none",
    )
    ax.scatter(
        embedding[0, 0],
        embedding[0, 1],
        c="black",
        s=90,
        marker="*",
        label="center",
        zorder=3,
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", frameon=False)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("circle angle")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
