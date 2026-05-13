import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import MatsumotoMetric, RandersMetric, utils
from finsler_mds.api import fit_finsler_mds


def main_donut():
    # Hyperparameters
    n = 250
    proj_dim = 3
    radius = 4.0
    radial_noise = 0.35
    init_tilt_theta = np.pi / 6
    randers_alpha_manifold = 0.45
    randers_alpha_embedding = 0.45
    matsumoto_alpha_embedding = 0.99
    k = 10
    max_iter_gd = 1000
    max_iter_path_frozen = 30
    inner_iter_path_frozen = 10
    seed = 7
    dir_res = "res/donut"

    rng = np.random.default_rng(seed)
    os.makedirs(dir_res, exist_ok=True)

    X, theta = sample_noisy_circle(
        n,
        radius=radius,
        radial_noise=radial_noise,
        rng=rng,
    )
    color = (theta % (2 * np.pi)) / (2 * np.pi)
    randers_field = clockwise_tangent_field(theta, randers_alpha_manifold)

    fig_input, ax_input = plot_2d_points(
        X,
        color,
        title=f"Donut input: clockwise Randers alpha={randers_alpha_manifold}",
        quiver_field=randers_field,
    )
    utils.set_window_title(fig_input, "Donut input")
    fig_input.savefig(os.path.join(dir_res, "donut_input.pdf"))

    print("Target Randers geodesic distances")
    dists_f, _ = utils.compute_dist_matrix(
        X,
        n_neighbors=k,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=randers_field,
    )
    weights = np.ones_like(dists_f)

    fig_dists, ax_dists = plt.subplots(figsize=(5, 5))
    utils.set_window_title(fig_dists, "Donut target distances")
    ax_dists.imshow(dists_f)
    ax_dists.set_title("Target Randers geodesic distances")
    fig_dists.savefig(os.path.join(dir_res, "donut_target_distances.pdf"))

    init = tilted_planar_init(X, init_tilt_theta)
    init_field = tilted_planar_vectors(randers_field, init_tilt_theta)
    plot_tilted_donut(
        init,
        color,
        title=f"Donut: tilted initialization before optimization theta={init_tilt_theta:.2f}",
        dir_res=dir_res,
        filename="donut_tilted_before_optimization.pdf",
        quiver_field=init_field,
    )

    print("Matsumoto direct gradient descent")
    proj_matsumoto_gd, stress_matsumoto_gd = fit_finsler_mds(
        dists_f,
        metric=MatsumotoMetric(alpha=matsumoto_alpha_embedding),
        optimizer="gradient_descent",
        init=init,
        n_components=proj_dim,
        weight=weights,
        max_iter=max_iter_gd,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
        device="auto",
        print_time=True,
    )
    plot_embedding(
        proj_matsumoto_gd,
        color,
        f"Donut: direct Matsumoto GD alpha={matsumoto_alpha_embedding}",
        dir_res,
        "donut_matsumoto_gd.pdf",
    )

    print("Randers path-frozen")
    proj_randers_pf, stress_randers_pf = fit_finsler_mds(
        dists_f,
        metric=RandersMetric(alpha=randers_alpha_embedding),
        optimizer="path_frozen",
        init=init,
        n_components=proj_dim,
        weight=weights,
        graph_neighbors=k,
        max_iter=max_iter_path_frozen,
        inner_iter=inner_iter_path_frozen,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
        device="auto",
        print_time=True,
    )
    plot_embedding(
        proj_randers_pf,
        color,
        f"Donut: Randers path-frozen alpha={randers_alpha_embedding}",
        dir_res,
        "donut_randers_path_frozen.pdf",
    )

    print("Matsumoto path-frozen")
    proj_matsumoto_pf, stress_matsumoto_pf = fit_finsler_mds(
        dists_f,
        metric=MatsumotoMetric(alpha=matsumoto_alpha_embedding),
        optimizer="path_frozen",
        init=init,
        n_components=proj_dim,
        weight=weights,
        graph_neighbors=k,
        max_iter=max_iter_path_frozen,
        inner_iter=inner_iter_path_frozen,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
        print_time=True,
    )
    plot_embedding(
        proj_matsumoto_pf,
        color,
        f"Donut: Matsumoto path-frozen alpha={matsumoto_alpha_embedding}",
        dir_res,
        "donut_matsumoto_path_frozen.pdf",
    )

    print("Stress summary")
    print("  direct Matsumoto GD:", stress_matsumoto_gd)
    print("  Randers path-frozen:", stress_randers_pf)
    print("  Matsumoto path-frozen:", stress_matsumoto_pf)

    plt.show()


def sample_noisy_circle(n, *, radius, radial_noise, rng):
    theta = rng.uniform(0, 2 * np.pi, size=n)
    radii = radius + rng.normal(scale=radial_noise, size=n)
    X = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    order = np.argsort(theta)
    return X[order], theta[order]


def clockwise_tangent_field(theta, alpha):
    tangent = np.column_stack([np.sin(theta), -np.cos(theta)])
    return alpha * tangent


def tilted_planar_init(X, theta):
    init = np.column_stack([X, np.zeros(len(X))])
    return rotate_x(init, theta)


def tilted_planar_vectors(vectors, theta):
    vectors_3d = np.column_stack([vectors, np.zeros(len(vectors))])
    return rotate_x(vectors_3d, theta)


def rotate_x(points, theta):
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    return points @ rotation.T


def plot_2d_points(X, color, *, title, quiver_field=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], c=color, cmap="hsv", s=25, lw=0)
    if quiver_field is not None:
        step = max(1, len(X) // 80)
        ax.quiver(
            X[::step, 0],
            X[::step, 1],
            quiver_field[::step, 0],
            quiver_field[::step, 1],
            color="black",
            alpha=0.6,
            angles="xy",
            scale_units="xy",
            scale=1.5,
            width=0.003,
        )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def plot_embedding(embedding, color, title, dir_res, filename):
    fig, ax = plt.subplots(figsize=(7, 7))
    utils.set_window_title(fig, title)
    if embedding.shape[1] == 3:
        ax.remove()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=color,
            cmap="hsv",
            s=25,
            lw=0,
        )
        ax.set_box_aspect([1, 1, 1])
        utils.set_axes_equal(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="hsv", s=25, lw=0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    ax.set_title(title)
    fig.savefig(os.path.join(dir_res, filename))
    return fig, ax


def plot_tilted_donut(embedding, color, *, title, dir_res, filename, quiver_field=None):
    fig = plt.figure(figsize=(7, 7))
    utils.set_window_title(fig, title)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=color,
        cmap="hsv",
        s=25,
        lw=0,
    )
    if quiver_field is not None:
        step = max(1, len(embedding) // 80)
        ax.quiver(
            embedding[::step, 0],
            embedding[::step, 1],
            embedding[::step, 2],
            quiver_field[::step, 0],
            quiver_field[::step, 1],
            quiver_field[::step, 2],
            color="black",
            length=0.7,
            normalize=True,
            alpha=0.6,
        )
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    utils.set_axes_equal(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.savefig(os.path.join(dir_res, filename))
    return fig, ax


if __name__ == "__main__":
    main_donut()
