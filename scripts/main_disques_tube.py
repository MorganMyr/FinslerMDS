import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, utils
from finsler_mds.api import fit_finsler_mds


def main_disques_tube():
    # Hyperparameters
    n_disk = 80
    n_tube = 25
    disk_radius = 1.5
    tube_radius = 0.45
    center_gap = 2.0
    disk_normal_noise = 0.04
    tube_radial_noise = 0.18
    randers_alpha_tube = 0.45
    randers_alpha_disk = 0.25
    randers_alpha_smacof_embedding = 0.45
    randers_alpha_geodesic_embedding = 0.45
    k = 7
    proj_dim = 3
    max_iter_smacof = 800
    max_iter_path_frozen = 40
    inner_iter_path_frozen = 10
    n_path_plot = 5
    max_iter_datasp = 25
    n_graph_updates_datasp = 10
    beta_datasp = 10.0
    seed = 41
    dir_res = "res/disques_tube"

    rng = np.random.default_rng(seed)
    os.makedirs(dir_res, exist_ok=True)

    X, labels, randers_field = sample_disks_tube(
        n_disk=n_disk,
        n_tube=n_tube,
        disk_radius=disk_radius,
        tube_radius=tube_radius,
        center_gap=center_gap,
        disk_normal_noise=disk_normal_noise,
        tube_radial_noise=tube_radial_noise,
        randers_alpha_tube=randers_alpha_tube,
        randers_alpha_disk=randers_alpha_disk,
        rng=rng,
    )
    order = display_order(X, labels)
    X = X[order]
    labels = labels[order]
    randers_field = randers_field[order]
    color = labels_to_color(labels, X)
    init = X.copy()

    plot_embedding(
        X,
        color,
        f"Original disks+tube: tube alpha={randers_alpha_tube}, disk alpha={randers_alpha_disk}",
        dir_res,
        "disques_tube_original.pdf",
        quiver_field=-randers_field,
    )

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
    utils.set_window_title(fig_dists, "Disks+tube target distances")
    ax_dists.imshow(dists_f)
    ax_dists.set_title("Target Randers geodesic distances")
    fig_dists.savefig(os.path.join(dir_res, "disques_tube_target_distances.pdf"))

    print("Randers SMACOF")
    proj_smacof, stress_smacof = fit_finsler_mds(
        dists_f,
        metric=RandersMetric(alpha=randers_alpha_smacof_embedding),
        optimizer="smacof_randers",
        init=init,
        n_components=proj_dim,
        n_init=1,
        n_jobs=1,
        weight=weights,
        max_iter=max_iter_smacof,
        eps=1e-6,
        pseudo_inv_solver="gmres",
        project_on_V=True,
        check_monotony=False,
        print_time=True,
    )
    plot_embedding(
        proj_smacof,
        color,
        f"Randers SMACOF alpha={randers_alpha_smacof_embedding}",
        dir_res,
        "disques_tube_randers_smacof.pdf",
    )

    print("Randers path-frozen")
    proj_path_frozen, stress_path_frozen = fit_finsler_mds(
        dists_f,
        metric=RandersMetric(alpha=randers_alpha_geodesic_embedding),
        optimizer="path_frozen",
        init=init,
        n_components=proj_dim,
        weight=weights,
        graph_neighbors=k,
        outer_iter=max_iter_path_frozen,
        inner_iter=inner_iter_path_frozen,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
        print_time=True,
        n_local_pairs=8,
        n_landmark=30,
        mask_random_state=seed,
        device="auto",
    )
    fig_path_frozen, ax_path_frozen = plot_embedding(
        proj_path_frozen,
        color,
        f"Randers path-frozen alpha={randers_alpha_geodesic_embedding}",
        dir_res,
        "disques_tube_randers_path_frozen.pdf",
    )
    utils.add_random_geodesic_paths(
        ax_path_frozen,
        proj_path_frozen,
        n_paths=n_path_plot,
        metric=RandersMetric(alpha=randers_alpha_geodesic_embedding),
        n_neighbors=k,
        random_state=seed,
        linewidth=3,
        alpha=0.9,
        arrow_size=1,
    )
    fig_path_frozen.savefig(os.path.join(dir_res, "disques_tube_randers_path_frozen.pdf"))

    """
    print("Randers DataSP")
    proj_datasp, stress_datasp = fit_finsler_mds(
        dists_f,
        metric=RandersMetric(alpha=randers_alpha_geodesic_embedding),
        optimizer="datasp",
        init=init,
        n_components=proj_dim,
        weight=weights,
        graph_neighbors=k,
        beta=beta_datasp,
        max_iter=max_iter_datasp,
        n_graph_updates=n_graph_updates_datasp,
        eps=1e-6,
        method="L-BFGS-B",
        optimizer_options={"ftol": 1e-9, "maxls": 50},
        print_time=True,
    )
    plot_embedding(
        proj_datasp,
        color,
        f"Randers DataSP alpha={randers_alpha_geodesic_embedding}, beta={beta_datasp}",
        dir_res,
        "disques_tube_randers_datasp.pdf",
    )
    """

    print("Stress summary")
    print("  Randers SMACOF:", stress_smacof)
    print("  Randers path-frozen:", stress_path_frozen)
    #print("  Randers DataSP:", stress_datasp)

    plt.show()


def sample_disks_tube(
        *,
        n_disk,
        n_tube,
        disk_radius,
        tube_radius,
        center_gap,
        disk_normal_noise,
        tube_radial_noise,
        randers_alpha_tube,
        randers_alpha_disk,
        rng,
):
    horizontal_center = np.array([0.0, 0.0, 0.0])
    vertical_center = np.array([center_gap, 0.0, center_gap])

    horizontal_disk = sample_oriented_disk(
        n_disk,
        disk_radius,
        center=horizontal_center,
        tangent_1=np.array([1.0, 0.0, 0.0]),
        tangent_2=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        normal_noise=disk_normal_noise,
        rng=rng,
    )
    vertical_disk = sample_oriented_disk(
        n_disk,
        disk_radius,
        center=vertical_center,
        tangent_1=np.array([0.0, 1.0, 0.0]),
        tangent_2=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        normal_noise=disk_normal_noise,
        rng=rng,
    )
    tube, tube_tangents = sample_curved_tube(
        n_tube,
        tube_radius,
        start=horizontal_center,
        end=vertical_center,
        radial_noise=tube_radial_noise,
        rng=rng,
    )

    X = np.vstack([horizontal_disk, vertical_disk, tube])
    labels = np.concatenate([
        np.zeros(n_disk, dtype=int),
        np.ones(n_disk, dtype=int),
        2 * np.ones(n_tube, dtype=int),
    ])

    randers_field = np.vstack([
        -inward_disk_field(horizontal_disk, horizontal_center, randers_alpha_disk),
        -inward_disk_field(vertical_disk, vertical_center, randers_alpha_disk),
        randers_alpha_tube * tube_tangents,
    ])
    return X, labels, randers_field


def sample_oriented_disk(n, radius, *, center, tangent_1, tangent_2, normal, normal_noise, rng):
    angle = rng.uniform(0, 2 * np.pi, size=n)
    radii = radius * np.sqrt(rng.uniform(0, 1, size=n))
    disk_points = (
        center
        + radii[:, None] * np.cos(angle)[:, None] * tangent_1
        + radii[:, None] * np.sin(angle)[:, None] * tangent_2
        + rng.normal(scale=normal_noise, size=n)[:, None] * normal
    )
    return disk_points


def inward_disk_field(points, center, alpha):
    directions = center - points
    norms = np.linalg.norm(directions, axis=1)
    return alpha * np.divide(
        directions,
        norms[:, None],
        out=np.zeros_like(directions),
        where=norms[:, None] > 1e-12,
    )


def sample_curved_tube(n, radius, *, start, end, radial_noise, rng):
    t = rng.uniform(0, 1, size=n)
    t.sort()
    theta = (np.pi / 2) * t
    arc_radius = end[0] - start[0]
    if not np.isclose(arc_radius, end[2] - start[2]):
        raise ValueError("The curved tube expects equal x and z center gaps.")

    # Quarter circle centered at (end.x, 0, start.z). It leaves the horizontal
    # disk vertically and reaches the vertical disk horizontally.
    centerline = np.column_stack([
        end[0] - arc_radius * np.cos(theta),
        np.zeros(n),
        start[2] + arc_radius * np.sin(theta),
    ])
    tangents = np.column_stack([
        arc_radius * np.sin(theta),
        np.zeros(n),
        arc_radius * np.cos(theta),
    ])
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    normal_1 = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    normal_2 = np.cross(tangents, normal_1)
    normal_2 /= np.linalg.norm(normal_2, axis=1)[:, None]

    angle = rng.uniform(0, 2 * np.pi, size=n)
    radii = np.minimum(np.abs(rng.normal(scale=radial_noise, size=n)), radius)
    noise = (
        radii[:, None] * np.cos(angle)[:, None] * normal_1
        + radii[:, None] * np.sin(angle)[:, None] * normal_2
    )
    return centerline + noise, tangents


def labels_to_color(labels, X):
    color = np.empty_like(labels, dtype=float)
    radial = local_disk_radius(X, labels)
    disk = labels != 2
    radial_norm = np.zeros_like(radial)
    if np.any(disk):
        radial_norm[disk] = (
            (radial[disk] - radial[disk].min())
            / (radial[disk].max() - radial[disk].min())
        )
    color[labels == 0] = 0.08 + 0.12 * radial_norm[labels == 0]
    color[labels == 1] = 0.55 + 0.12 * radial_norm[labels == 1]
    color[labels == 2] = 0.95
    return color


def display_order(X, labels):
    radial = local_disk_radius(X, labels)
    vertical = np.flatnonzero(labels == 1)
    tube = np.flatnonzero(labels == 2)
    horizontal = np.flatnonzero(labels == 0)
    return np.concatenate([
        vertical[np.argsort(-radial[vertical])],
        tube[np.argsort(-X[tube, 2])],
        horizontal[np.argsort(radial[horizontal])],
    ])


def local_disk_radius(X, labels):
    radial = np.zeros(len(X), dtype=float)
    horizontal = labels == 0
    vertical = labels == 1
    radial[horizontal] = np.linalg.norm(X[horizontal, :2], axis=1)
    radial[vertical] = np.linalg.norm(X[vertical][:, [1, 2]], axis=1)
    return radial


def plot_embedding(embedding, color, title, dir_res, filename, quiver_field=None, point_fraction=1.0, random_state=None):
    plot_idx = utils.sample_plot_indices(
        len(embedding),
        point_fraction=point_fraction,
        random_state=random_state,
    )
    fig = plt.figure(figsize=(8, 7))
    utils.set_window_title(fig, title)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        embedding[plot_idx, 0],
        embedding[plot_idx, 1],
        embedding[plot_idx, 2],
        c=color[plot_idx],
        cmap="viridis",
        s=22,
        lw=0,
    )
    if quiver_field is not None:
        step = max(1, len(embedding) // 80)
        nonzero = np.linalg.norm(quiver_field, axis=1) > 0
        ids = np.flatnonzero(nonzero)[::step]
        ax.quiver(
            embedding[ids, 0],
            embedding[ids, 1],
            embedding[ids, 2],
            quiver_field[ids, 0],
            quiver_field[ids, 1],
            quiver_field[ids, 2],
            color="black",
            length=0.5,
            normalize=True,
            alpha=0.75,
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
    main_disques_tube()
