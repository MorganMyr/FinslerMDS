
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import RandersMetric, geodesic_embedding_stress, utils
from finsler_mds.api import fit_finsler_mds


def main_swiss_roll_full():

    # Hyperparameters
    n = 1000                            # 3000
    noise_level = 0.                    # 0.
    randers_field_dir = 'length'        # 'length'
    randers_w_alpha_manifold = 0.35  # On the input manifold
    randers_w_alpha_embedding = 0.5     # 0.5 # in the flat randers embedding
    init_strat = 'isomap'               # 'isomap' | 'isomap', 'rand'
    k = 10                              # 10
    proj_dim = 3                        # 3
    dir_res = 'res/swiss_roll_full'
    folder_res_raw = 'raw/'             # 'raw/'
    plot_point_fraction = 0.1          # fraction of points shown in 3D plots
    seed = 42

    max_iter_smacof = 100
    smacof_device = "auto"

    max_iter_path_frozen = 5          # 10
    inner_iter_path_frozen = 2000         # 5
    n_global_landmarks_path_frozen = 50      # 350
    n_local_neighbors_path_frozen = 8      # 8
    local_pair_mode_path_frozen = "geodesic"  # "direct" | "geodesic"
    max_global_targets_per_source_path_frozen = int(0.2 * n)
    global_target_sampling_path_frozen = "random"
    local_global_reweighting_path_frozen = "count"  # "none" | "count" | "energy"
    local_weight_path_frozen = 1.0
    path_frozen_device = "auto"         # "auto" uses CuPy/CUDA when available, CPU otherwise

    beta_soft_bf = 50.0
    n_relaxations_soft_bf = 50
    max_iter_soft_bf = 500
    n_graph_updates_soft_bf = 6
    n_global_landmarks_soft_bf = 100
    n_local_neighbors_soft_bf = 10
    local_pair_mode_soft_bf = "direct"  # "direct" | "geodesic"
    max_global_targets_per_source_soft_bf = int(0.4 * n)
    global_target_sampling_soft_bf = "random"
    local_global_reweighting_soft_bf = "count"  # "none" | "count" | "energy"
    local_weight_soft_bf = 1.0
    soft_bf_device = "auto"
    soft_bf_source_batch_size = 64

    np.random.seed(seed)  # Will reseed for each application invoking random number generation
    dir_res_raw = os.path.join(dir_res, folder_res_raw)
    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    if not os.path.exists(dir_res_raw):
        os.makedirs(dir_res_raw)

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
    fig, ax = utils.plot_points(
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
    dists, preds = isomap.dist_matrix_, isomap.preds_

    fig_isomap, ax_isomap = utils.plot_proj_points(
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
    fig_randers, ax_randers = utils.plot_points(
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
    dists_f, preds_f = utils.compute_dist_matrix(
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
    im = ax_dists.imshow(dists_f)

    np.random.seed(seed + 2)
    if init_strat == 'isomap':
        init = proj
    elif init_strat == 'rand':
        init = np.random.rand(n, proj_dim)
    else:
        raise ValueError('Unknown initialisation strategy')

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
            'randers_path_frozen',
            'Randers path-frozen',
            dict(
                metric=RandersMetric(alpha=randers_w_alpha_embedding),
                optimizer="path_frozen",
                init=init,
                n_components=proj_dim,
                graph_neighbors=k,
                max_iter=max_iter_path_frozen,
                inner_iter=inner_iter_path_frozen,
                eps=1e-6,
                method="L-BFGS-B",
                optimizer_options={"ftol": 1e-9, "maxls": 50},
                n_global_landmarks=n_global_landmarks_path_frozen,
                n_local_neighbors=n_local_neighbors_path_frozen,
                local_pair_mode=local_pair_mode_path_frozen,
                mask_random_state=seed,
                max_global_targets_per_source=max_global_targets_per_source_path_frozen,
                global_target_sampling=global_target_sampling_path_frozen,
                target_random_state=seed + 3,
                local_global_reweighting=local_global_reweighting_path_frozen,
                local_weight=local_weight_path_frozen,
                device=path_frozen_device,
                verbose=1,
                print_time=True,
            ),
        ),
        (
            'randers_soft_bf',
            'Randers soft-BF',
            dict(
                metric=RandersMetric(alpha=randers_w_alpha_embedding),
                optimizer="soft_bellman_ford",
                init=init,
                n_components=proj_dim,
                graph_neighbors=k,
                beta=beta_soft_bf,
                n_relaxations=n_relaxations_soft_bf,
                max_iter=max_iter_soft_bf,
                n_graph_updates=n_graph_updates_soft_bf,
                eps=1e-6,
                method="L-BFGS-B",
                optimizer_options={"ftol": 1e-9, "maxls": 50},
                n_global_landmarks=n_global_landmarks_soft_bf,
                n_local_neighbors=n_local_neighbors_soft_bf,
                local_pair_mode=local_pair_mode_soft_bf,
                mask_random_state=seed,
                max_global_targets_per_source=max_global_targets_per_source_soft_bf,
                global_target_sampling=global_target_sampling_soft_bf,
                target_random_state=seed + 3,
                local_global_reweighting=local_global_reweighting_soft_bf,
                local_weight=local_weight_soft_bf,
                device=soft_bf_device,
                source_batch_size=soft_bf_source_batch_size,
                verbose=1,
                print_time=True,
            ),
        ),
    ]

    fig_methods_all = plt.figure(figsize=(10, 10))
    utils.set_window_title(fig_methods_all, "Swiss roll full: all Randers Finsler-MDS methods")
    ax_methods_all = fig_methods_all.add_subplot(111, projection='3d')

    for method_key, method_title, kwargs in methods:
        print(method_title)
        proj_method, stress_method = fit_finsler_mds(dists_f, **kwargs)
        full_geodesic_stress = None
        if kwargs["optimizer"] == "smacof_randers":
            full_geodesic_stress = geodesic_embedding_stress(
                proj_method,
                dists_f,
                metric=kwargs["metric"],
                n_neighbors=k,
                on_unreachable="inf",
            )
            print(f"  {method_title} final full geodesic stress: {full_geodesic_stress}")

        cache_payload = dict(
            embedding=proj_method,
            stress=np.asarray(stress_method),
        )
        if full_geodesic_stress is not None:
            cache_payload["full_geodesic_stress"] = np.asarray(full_geodesic_stress)
        np.savez(
            os.path.join(dir_res_raw, 'swiss_roll_full_' + method_key + '.npz'),
            **cache_payload,
        )

        fig_method, ax_method = utils.plot_proj_points(
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
            f"Swiss roll full: {method_title} alpha={randers_w_alpha_manifold}"
        )
        fig_method.savefig(os.path.join(dir_res, 'swiss_roll_full_' + method_key + '.pdf'))

        fig_methods_all, ax_methods_all = utils.plot_proj_points(
            proj_method,
            X=X,
            knn=knn,
            X_noiseless=X_noiseless,
            shape_type='swiss_roll',
            edge_alpha=1.,
            fig_ax=(fig_methods_all, ax_methods_all),
            point_fraction=plot_point_fraction,
            random_state=seed,
        )
        print(f"  {method_title} optimizer stress: {stress_method}")

    fig_methods_all.savefig(os.path.join(dir_res, 'swiss_roll_full_randers_methods_all.pdf'))

    plt.show()


if __name__ == '__main__':

    # import matplotlib
    # matplotlib.use('TkAgg')

    main_swiss_roll_full()

    plt.show()
