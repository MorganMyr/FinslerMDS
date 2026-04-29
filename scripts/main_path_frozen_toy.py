import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from sklearn.decomposition import KernelPCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import MatsumotoMetric, RandersMetric, utils
from finsler_mds.api import fit_finsler_mds


def main_path_frozen_toy():
    # Hyperparameters
    n = 40
    proj_dim = 2
    target_alpha = 0.4
    randers_alpha = 0.4
    matsumoto_alpha = 0.4
    path_frozen_alpha = 0.4
    path_frozen_neighbors = 4
    max_iter_gd = 1500
    max_iter_path_frozen = 10
    inner_iter = 5
    max_iter_datasp = 1500
    seed = 42
    dir_res = "res/path_frozen_toy"

    np.random.seed(seed)
    os.makedirs(dir_res, exist_ok=True)

    x = np.linspace(-3, 3, n)
    y = two_bump_curve(x)
    X = np.stack([x, y], axis=1)
    color = (x - x.min()) / (x.max() - x.min())

    target_metric = MatsumotoMetric(alpha=target_alpha)
    support = chain_support(n)
    dists_target, _ = utils.compute_metric_dist_matrix(
        X,
        target_metric,
        support_graph=support,
        directed=True,
    )
    weights = np.ones_like(dists_target)

    fig_curve, ax_curve = plt.subplots(figsize=(7, 3))
    utils.set_window_title(fig_curve, f"Toy two-bump curve: target Matsumoto alpha={target_alpha}")
    ax_curve.scatter(X[:, 0], X[:, 1], c=color, cmap="viridis", s=25)
    ax_curve.plot(X[:, 0], X[:, 1], color="black", alpha=0.25)
    ax_curve.set_aspect("equal", adjustable="box")
    ax_curve.set_title(f"Input curve, target Matsumoto alpha={target_alpha}")
    fig_curve.savefig(os.path.join(dir_res, "input_curve.pdf"))

    fig_dists, ax_dists = plt.subplots(figsize=(5, 5))
    utils.set_window_title(fig_dists, "Toy two-bump: target geodesic distances")
    ax_dists.imshow(dists_target)
    ax_dists.set_title("Target geodesic distances")
    fig_dists.savefig(os.path.join(dir_res, "target_distances.pdf"))

    init = umap_warm_start(dists_target, proj_dim, seed)
    plot_embedding(init, color, "Toy two-bump: UMAP warm start", dir_res, "warm_start_umap.pdf")

    print("SMACOF Randers")
    proj_randers_direct, stress_randers_direct = fit_finsler_mds(
        dists_target,
        metric=RandersMetric(alpha=randers_alpha),
        optimizer="smacof_randers",
        init=init,
        n_components=proj_dim,
        n_init=1,
        n_jobs=1,
        weight=weights,
        max_iter=max_iter_gd,
        eps=1e-6,
        pseudo_inv_solver="gmres",
        project_on_V=True,
        check_monotony=False,
    )
    plot_embedding(
        proj_randers_direct,
        color,
        f"Toy two-bump: SMACOF Randers alpha={randers_alpha}",
        dir_res,
        "smacof_randers.pdf",
    )

    print("Matsumoto gradient descent")
    proj_matsumoto_direct, stress_matsumoto_direct = fit_finsler_mds(
        dists_target,
        metric=MatsumotoMetric(alpha=matsumoto_alpha),
        optimizer="gradient_descent",
        init=init,
        n_components=proj_dim,
        weight=weights,
        max_iter=max_iter_gd,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
    )
    plot_embedding(
        proj_matsumoto_direct,
        color,
        f"Toy two-bump: direct Matsumoto GD alpha={matsumoto_alpha}",
        dir_res,
        "direct_matsumoto_gd.pdf",
    )

    print("DataSP with Matsumoto")
    proj_path_frozen, stress_path_frozen = fit_finsler_mds(
        dists_target,
        metric=MatsumotoMetric(alpha=path_frozen_alpha),
        optimizer="dataSP",
        init=init,
        n_components=proj_dim,
        weight=weights,
        beta=50,
        n_neighbors=4,
        max_iter=max_iter_datasp,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
    )
    plot_embedding(
        proj_path_frozen,
        color,
        f"Toy two-bump: DataSP Matsumoto alpha={path_frozen_alpha}",
        dir_res,
        "dataSP_matsumoto.pdf",
    )

    print("Stress summary")
    print("  direct Randers:", stress_randers_direct)
    print("  direct Matsumoto:", stress_matsumoto_direct)
    print("  DataSP Matsumoto:", stress_path_frozen)

    plt.show()


def two_bump_curve(x):
    return (
        0.85 * np.exp(-((x + 1.25) / 0.65) ** 2)
        + 0.85 * np.exp(-((x - 1.25) / 0.65) ** 2)
        - 0.25 * np.exp(-(x / 0.7) ** 2)
    )


def chain_support(n):
    rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    data = np.ones_like(rows, dtype=float)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def umap_warm_start(dists, n_components, random_state):
    dists_sym = (dists + dists.T) / 2
    try:
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            n_neighbors=10,
        )
        return reducer.fit_transform(dists_sym)
    except ImportError:
        kernel_pca = KernelPCA(
            n_components=n_components,
            kernel="precomputed",
            eigen_solver="auto",
        ).set_output(transform="default")
        gram = -0.5 * dists_sym ** 2
        return kernel_pca.fit_transform(gram)


def plot_embedding(embedding, color, title, dir_res, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    utils.set_window_title(fig, title)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    fig.savefig(os.path.join(dir_res, filename))
    return fig, ax


if __name__ == "__main__":
    main_path_frozen_toy()
