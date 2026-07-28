
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
    utils,
)
from finsler_mds.api import fit_finsler_mds


def main_swiss_roll_full():

    # Hyperparameters
    n = 1000                            # 3000
    noise_level = 0.                    # 0.
    randers_field_dir = 'length'        # 'length'
    randers_w_alpha_manifold = 0.35     # On the input manifold
    randers_w_alpha_embedding = 0.5     # in the flat randers embedding
    init_strat = 'isomap'               # 'isomap' | 'rand'
    k = 10                              # 10
    proj_dim = 3                        # 3
    dir_res = 'res/swiss_roll_full'
    plot_point_fraction = 0.1          # fraction of points shown in 3D plots
    seed = 42

    max_iter_smacof = 100
    smacof_device = "auto"

    gradient_descent_metric = "randers"  # "randers" | "matsumoto" | "convexified_matsumoto"
    gradient_descent_alpha = randers_w_alpha_embedding
    gradient_descent_options = {
        "max_iter": 300,
        "device": "auto",
        "verbose": 1,
    }
    path_frozen_options = {
        "graph_neighbors": k,
        "outer_iter": 5,
        "inner_iter": 2000,
        "n_landmark": 50,
        "random_landmark_fraction": 1.0,
        "targets_per_landmark": int(0.2 * n),
        "n_local_pairs": 8,
        "local_weight": 1.0,
        "direct_stress_weight": 0.0,
        "outer_step_size": 1.0,
        "device": "auto",
        "verbose": 1,
    }

    np.random.seed(seed)  # Will reseed for each application invoking random number generation
    dir_res_raw = os.path.join(dir_res, 'raw')
    os.makedirs(dir_res_raw, exist_ok=True)

    ################### Swiss roll generation #################

    x_ = np.random.rand(n, 2)
    x = np.zeros((n, 2))
    x[:, 0] = x_[:, 0] * 3 * np.pi + 1.5 * np.pi
    x[:, 1] = x_[:, 1] * 20

    X_noiseless = np.zeros((n, 3))
    X_noiseless[:, 0] = x[:, 0] * np.cos(x[:, 0])
    X_noiseless[:, 1] = x[:, 1]
    X_noiseless[:, 2] = x[:, 0] * np.sin(x[:, 0])

    noise = noise_level * np.random.randn(n, 3)

    X = X_noiseless + noise

    ################### Plotting the Swiss roll #################

    # Plotting the Swiss roll
    fig, _ = utils.plot_points(
        X,
        X_noiseless=X_noiseless,
        shape_type='swiss_roll',
        point_fraction=plot_point_fraction,
        random_state=seed,
    )
    utils.set_window_title(fig, "Swiss roll full: input")

    fig.savefig(os.path.join(dir_res, 'swiss_roll_full_input_euclidean.pdf'))

    knn = utils.nearest_neighbors(X, k)  # For visualisation purposes only

    ################### Euclidean embeddings (Isomap/SMACOF) #################

    print('Isomap')

    isomap = utils.IsomapWithPreds(n_components=proj_dim, n_neighbors=k)
    proj = isomap.fit_transform(X)
    fig_isomap, _ = utils.plot_proj_points(
        proj,
        X=X,
        knn=knn,
        X_noiseless=X_noiseless,
        shape_type='swiss_roll',
        edge_alpha=1.,
        point_fraction=plot_point_fraction,
        random_state=seed,
    )
    utils.set_window_title(fig_isomap, "Swiss roll full: Isomap initialization")

    fig_isomap.savefig(os.path.join(dir_res, 'swiss_roll_full_isomap_euclidean.pdf'))



    ################### Randers field generation #################

    # Tangent field along the length of the swiss roll:
    # We calculate the tangent vectors at different points
    if randers_field_dir == 'length':
        tangent_x = np.cos(x[:, 0]) - x[:, 0] * np.sin(x[:, 0])
        tangent_z = np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 0])
        tangent_y = np.zeros(n)
    else:
        raise ValueError('Unknown Randers field direction')

    print('Randers field manifold weight: ', randers_w_alpha_manifold)

    randers_field = np.stack([tangent_x, tangent_y, tangent_z], axis=1)
    randers_field = randers_field / np.linalg.norm(randers_field, axis=1)[:, None]
    randers_field = randers_w_alpha_manifold * randers_field

    # Plotting the Swiss roll with Randers field
    fig_randers, _ = utils.plot_points(
        X,
        X_noiseless=X_noiseless,
        shape_type='swiss_roll',
        quiver_field=randers_field,
        step_quiver=1,
        point_fraction=plot_point_fraction,
        random_state=seed,
    )
    utils.set_window_title(fig_randers, f"Swiss roll full: Randers field alpha={randers_w_alpha_manifold}")
    fig_randers.savefig(os.path.join(dir_res, 'swiss_roll_full_randers_field.pdf'))

    # ################### Finsler embeddings #################

    # Compute Randers geodesic distances once, then compare optimizers.
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
        randers_field=randers_field
    )

    fig_dists, ax_dists = plt.subplots()
    utils.set_window_title(fig_dists, f"Swiss roll full: distances alpha={randers_w_alpha_manifold}")
    ax_dists.imshow(dists_f)

    np.random.seed(seed + 2)
    if init_strat == 'isomap':
        init = proj
    elif init_strat == 'rand':
        init = np.random.rand(n, proj_dim)
    else:
        raise ValueError('Unknown initialisation strategy')

    metric_classes = {
        "randers": RandersMetric,
        "matsumoto": MatsumotoMetric,
        "convexified_matsumoto": ConvexifiedMatsumotoMetric,
    }
    try:
        gradient_metric = metric_classes[gradient_descent_metric](
            alpha=gradient_descent_alpha
        )
    except KeyError as exc:
        raise ValueError(f"Unknown gradient-descent metric: {gradient_descent_metric}") from exc

    methods = [
        (
            'randers_smacof',
            'Randers SMACOF',
            dict(
                metric=RandersMetric(alpha=randers_w_alpha_embedding),
                optimizer="smacof_randers",
                init=init,
                n_components=proj_dim,
                n_init=1,
                n_jobs=1,
                max_iter=max_iter_smacof,
                pseudo_inv_solver="gmres",
                project_on_V=True,
                check_monotony=False,
                device=smacof_device,
                print_time=True,
            ),
        ),
        (
            f'{gradient_descent_metric}_gradient_descent',
            f'{gradient_descent_metric.replace("_", " ").title()} gradient descent',
            dict(
                metric=gradient_metric,
                optimizer="gradient_descent",
                init=init,
                n_components=proj_dim,
                random_state=seed,
                print_time=True,
                **gradient_descent_options,
            ),
        ),
        (
            'randers_path_frozen',
            'Randers path-frozen',
            dict(
                metric=RandersMetric(alpha=randers_w_alpha_embedding),
                optimizer="path_frozen",
                init=init,
                n_components=proj_dim,
                random_state=seed,
                print_time=True,
                **path_frozen_options,
            ),
        ),
    ]

    for method_key, method_title, kwargs in methods:
        print(method_title)
        proj_method, stress_method = fit_finsler_mds(dists_f, **kwargs)
        np.savez(
            os.path.join(dir_res_raw, 'swiss_roll_full_' + method_key + '.npz'),
            embedding=proj_method,
            stress=np.asarray(stress_method),
        )

        fig_method, _ = utils.plot_proj_points(
            proj_method,
            X=X,
            knn=knn,
            X_noiseless=X_noiseless,
            shape_type='swiss_roll',
            edge_alpha=1.,
            point_fraction=plot_point_fraction,
            random_state=seed,
        )
        utils.set_window_title(
            fig_method,
            f"Swiss roll full: {method_title}, manifold alpha={randers_w_alpha_manifold}"
        )
        fig_method.savefig(os.path.join(dir_res, 'swiss_roll_full_' + method_key + '.pdf'))

        print(f"  {method_title} optimizer stress: {stress_method}")

    plt.show()


if __name__ == '__main__':
    main_swiss_roll_full()
