import json
import os
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import (
    ConvexifiedMatsumotoMetric,
    MatsumotoMetric,
    RandersMetric,
    fit_finsler_mds,
    geodesic_embedding_stress,
)
from finsler_mds import utils
from finsler_mds.utils import plot_3d_embedding_views, plot_categorical_embedding


def main_pancreas():
    # Hyperparameters
    seed = 42
    dir_res = "res/pancreas"
    dir_res_raw = os.path.join(dir_res, "raw")

    preprocessing = {
        "min_shared_counts": 20,
        "n_top_genes": 3000,
        "n_pcs": 50,
        "moments_n_neighbors": 30,
    }
    velocity = {
        "mode": "dynamical",
        "distance_formula": "randers",  # one of {"exponential", "randers"}
        "alpha": 2, # should be positive; randers needs alpha * cos_clip < 1
        "cos_clip": 0.4,
        "neighbors": 30, # to average the velocity locally (if "average" is True)
        "graph_neighbors": 30,  # to create the kNN graph for calculating dissimilarities through Dijkstra
        "average": True,
        "symmetrize_support": True, # for the kNN graph
        "graph_n_jobs": -1,
        "recover_dynamics_max_iter": 20,
        "recover_dynamics_n_jobs": -1,
    }
    umap = {
        "n_neighbors": 50,
        "min_dist": 0.5,
        "spread": 1.0,
        "maxiter": 1500,
        "negative_sample_rate": 10,
        "init_pos": "spectral",
    }
    isomap = {
        "n_neighbors": 30,
    }

    finsler_optimizer = "path_frozen"  # one of {"smacof_randers", "path_frozen", "soft_bf"}
    init_finsler_mds = "path_frozen"  # one of {"umap_2D", "umap_3D", "isomap_2D", "isomap_3D", "smacof", "path_frozen", "soft_bf"}

    # SMACOF is always run with a Randers embedding metric.
    randers_alpha_embedding = 0.8
    geodesic_metric = {
        # path-frozen and soft-BF can use one of:
        # {"randers", "matsumoto", "convexified_matsumoto"}.
        "kind": "randers",
        "alpha": 0.5,
    }
    smacof = {
        "max_iter": 100,
        "device": "auto",
        "check_monotony": True,
    }
    path_frozen = {
        "graph_neighbors": 20,
        "outer_iter": 20,
        "inner_iter": 5,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-9, "maxls": 50},
        "n_landmark": 400,
        "n_local_pairs": 20,
        "local_pair_mode": "direct",
        "targets_per_landmark": 400,
        "global_target_sampling": "random",
        "local_global_reweighting": "count",
        "local_weight": 0.1,
        "device": "auto",
        "verbose": 1,
    }
    soft_bf = {
        "graph_neighbors": 20,
        "beta": 80.0,
        "n_relaxations": 45,
        "max_iter": 50,
        "n_graph_updates": 10,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-9, "maxls": 50},
        "n_global_landmarks": 250,
        "n_local_neighbors": 25,
        "local_pair_mode": "direct",
        "max_global_targets_per_source": 200,
        "global_target_sampling": "mixed",
        "local_global_reweighting": "count",
        "local_weight": 3,
        "device": "auto",
        "source_batch_size": 32,
        "on_unreachable": "warn_skip",
        "verbose": 1,
    }

    os.makedirs(dir_res, exist_ok=True)
    os.makedirs(dir_res_raw, exist_ok=True)
    np.random.seed(seed)

    velocity_cache_metadata = pancreas_cache_metadata(
        min_shared_counts=preprocessing["min_shared_counts"],
        n_top_genes=preprocessing["n_top_genes"],
        n_pcs=preprocessing["n_pcs"],
        moments_n_neighbors=preprocessing["moments_n_neighbors"],
        velocity_mode=velocity["mode"],
        velocity_distance_formula=normalize_velocity_distance_formula(velocity["distance_formula"]),
        velocity_alpha=velocity["alpha"],
        velocity_cos_clip=velocity["cos_clip"],
        velocity_neighbors=velocity["neighbors"],
        velocity_graph_neighbors=velocity["graph_neighbors"],
        average_velocity=velocity["average"],
        symmetrize_velocity_support=velocity["symmetrize_support"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        umap_n_neighbors=umap["n_neighbors"],
        umap_min_dist=umap["min_dist"],
        umap_spread=umap["spread"],
        umap_maxiter=umap["maxiter"],
        umap_negative_sample_rate=umap["negative_sample_rate"],
        umap_init_pos=umap["init_pos"],
        seed=seed,
    )
    umap_cache_tag = f"{cache_token(velocity['mode'])}_s{seed}"
    isomap_cache_tag = f"k{isomap['n_neighbors']}_s{seed}"
    velocity_formula_tag = velocity_distance_formula_tag(velocity["distance_formula"])
    velocity_cache_tag = (
        f"{cache_token(velocity['mode'])}_"
        f"{velocity_formula_tag}_"
        f"valpha{cache_token(velocity['alpha'])}_"
        f"cclip{cache_token(velocity['cos_clip'])}_s{seed}"
    )
    velocity_cache_path = os.path.join(
        dir_res_raw,
        f"pancreas_velocity_inputs_{velocity_cache_tag}.npz",
    )
    legacy_velocity_cache_paths = []
    if normalize_velocity_distance_formula(velocity["distance_formula"]) == "exponential":
        legacy_velocity_cache_paths.append(
            os.path.join(
                dir_res_raw,
                f"pancreas_velocity_inputs_{cache_token(velocity['mode'])}_"
                f"valpha{cache_token(velocity['alpha'])}_s{seed}.npz",
            )
        )
    # UMAP alone is not enough to rerun SMACOF: the optimizer also needs the
    # directed velocity dissimilarities, so those live in velocity_cache_path.
    umap_2d_embedding_path = pancreas_umap_embedding_path(dir_res_raw, umap_cache_tag, n_components=2)
    umap_3d_embedding_path = pancreas_umap_embedding_path(dir_res_raw, umap_cache_tag, n_components=3)
    isomap_2d_embedding_path = pancreas_isomap_embedding_path(dir_res_raw, isomap_cache_tag, n_components=2)
    isomap_3d_embedding_path = pancreas_isomap_embedding_path(dir_res_raw, isomap_cache_tag, n_components=3)
    geodesic_metric_obj = make_embedding_metric(geodesic_metric)
    geodesic_metric_tag = embedding_metric_tag(geodesic_metric_obj)
    path_frozen_cache_path = os.path.join(
        dir_res_raw,
        "pancreas_path_frozen_"
        f"{velocity_formula_tag}_{geodesic_metric_tag}_seed{seed}.npz",
    )
    soft_bf_cache_path = os.path.join(
        dir_res_raw,
        "pancreas_soft_bf_"
        f"{velocity_formula_tag}_{geodesic_metric_tag}_seed{seed}.npz",
    )

    requested_umap_dim = requested_embedding_init_dimension(init_finsler_mds, "umap")
    requested_isomap_dim = requested_embedding_init_dimension(init_finsler_mds, "isomap")

    adata = None
    cached_inputs = load_velocity_inputs_cache(
        velocity_cache_path,
        velocity_cache_metadata,
        fallback_paths=legacy_velocity_cache_paths,
    )
    if cached_inputs is not None:
        x_umap, dists_velocity, labels = cached_inputs
        if not os.path.exists(umap_2d_embedding_path):
            np.save(umap_2d_embedding_path, x_umap)
        print(
            "Loaded directed velocity distances: "
            f"{dists_velocity.shape[0]} x {dists_velocity.shape[1]}, "
            f"finite={np.isfinite(dists_velocity).mean():.3f}"
        )
    else:
        # Keep these imports local: the rest of the project should remain usable
        # even when scvelo/scanpy are not installed.
        import scanpy as sc
        import scvelo as scv

        scv.settings.verbosity = 3
        scv.settings.set_figure_params("scvelo")

        print("Loading scVelo pancreas dataset")
        adata = scv.datasets.pancreas()
        print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

        print("Preprocessing")
        scv.pp.filter_and_normalize(
            adata,
            min_shared_counts=preprocessing["min_shared_counts"],
        )
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=preprocessing["n_top_genes"],
            flavor="seurat",
            subset=True,
        )
        sc.tl.pca(adata, n_comps=preprocessing["n_pcs"], random_state=seed)
        sc.pp.neighbors(
            adata,
            n_neighbors=preprocessing["moments_n_neighbors"],
            n_pcs=preprocessing["n_pcs"],
            random_state=seed,
        )
        scv.pp.moments(
            adata,
            n_pcs=preprocessing["n_pcs"],
            n_neighbors=preprocessing["moments_n_neighbors"],
        )
        print(f"Preprocessed pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

        print("Computing RNA velocity")
        if velocity["mode"] == "dynamical":
            print(
                "Recovering scVelo dynamical model "
                f"(max_iter={velocity['recover_dynamics_max_iter']}, "
                f"n_jobs={velocity['recover_dynamics_n_jobs']})"
            )
            scv.tl.recover_dynamics(
                adata,
                max_iter=velocity["recover_dynamics_max_iter"],
                n_jobs=velocity["recover_dynamics_n_jobs"],
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Conversion of an array with ndim > 0 to a scalar is deprecated.*",
                category=DeprecationWarning,
                module="scvelo.tools.optimization",
            )
            scv.tl.velocity(adata, mode=velocity["mode"])

        labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
        if os.path.exists(umap_2d_embedding_path):
            x_umap = np.asarray(np.load(umap_2d_embedding_path), dtype=float)
            print(f"Loaded 2D UMAP embedding: {umap_2d_embedding_path}")
        else:
            print(f"Computing UMAP neighbors (n_neighbors={umap['n_neighbors']})")
            sc.pp.neighbors(
                adata,
                n_neighbors=umap["n_neighbors"],
                n_pcs=preprocessing["n_pcs"],
                random_state=seed,
            )
            x_umap = compute_umap_from_neighbors(
                adata,
                umap,
                n_components=2,
                random_state=seed,
            )

        print("Building directed velocity dissimilarities")
        X_pca = adata.obsm["X_pca"][:, :preprocessing["n_pcs"]]
        velocity_pca = project_velocity_to_pca(adata, preprocessing["n_pcs"])
        dists_velocity, preds_velocity, velocity_graph, velocity_pca_smooth = utils.compute_velocity_dist_matrix(
            X_pca,
            velocity_pca,
            n_neighbors=velocity["graph_neighbors"],
            alpha=velocity["alpha"],
            distance_formula=velocity["distance_formula"],
            cos_clip=velocity["cos_clip"],
            velocity_neighbors=velocity["neighbors"],
            average_velocity=velocity["average"],
            symmetrize_support=velocity["symmetrize_support"],
            n_jobs=velocity["graph_n_jobs"],
        )
        adata.uns["finsler_mds_velocity_graph"] = velocity_graph
        adata.uns["finsler_mds_velocity_predecessors"] = preds_velocity
        adata.obsm["velocity_pca_smoothed"] = velocity_pca_smooth
        print(
            "Directed velocity distances: "
            f"{dists_velocity.shape[0]} x {dists_velocity.shape[1]}, "
            f"finite={np.isfinite(dists_velocity).mean():.3f}"
        )
        np.save(umap_2d_embedding_path, x_umap)
        np.savez(
            velocity_cache_path,
            x_umap=x_umap,
            dists_velocity=dists_velocity,
            labels=labels_to_cache(labels),
            metadata_json=json.dumps(velocity_cache_metadata, sort_keys=True),
        )
        print(f"Saved 2D UMAP embedding: {umap_2d_embedding_path}")
        print(f"Saved pancreas velocity cache: {velocity_cache_path}")
        if requested_umap_dim == 3 and not os.path.exists(umap_3d_embedding_path):
            x_umap_3d = compute_umap_from_neighbors(
                adata,
                umap,
                n_components=3,
                random_state=seed,
            )
            np.save(umap_3d_embedding_path, x_umap_3d)
            plot_3d_umap(
                x_umap_3d,
                labels=labels,
                save_path=os.path.join(dir_res, "pancreas_umap_3d_views.pdf"),
                random_state=seed,
            )
            adata.obsm["X_umap"] = x_umap
            print(f"Saved 3D UMAP embedding: {umap_3d_embedding_path}")
        if requested_isomap_dim == 3 and not os.path.exists(isomap_3d_embedding_path):
            x_isomap_3d = compute_isomap_from_pca(
                X_pca,
                isomap,
                n_components=3,
            )
            np.save(isomap_3d_embedding_path, x_isomap_3d)
            plot_3d_embedding_views(
                x_isomap_3d,
                labels=labels,
                title="scVelo pancreas Isomap 3D",
                save_path=os.path.join(dir_res, "pancreas_isomap_3d_views.pdf"),
                point_fraction=1.0,
                random_state=seed,
            )
            print(f"Saved 3D Isomap embedding: {isomap_3d_embedding_path}")
        elif requested_isomap_dim == 2 and not os.path.exists(isomap_2d_embedding_path):
            x_isomap = compute_isomap_from_pca(
                X_pca,
                isomap,
                n_components=2,
            )
            np.save(isomap_2d_embedding_path, x_isomap)
            plot_categorical_embedding(
                x_isomap,
                labels=labels,
                title="scVelo pancreas Isomap",
                xlabel="Isomap 1",
                ylabel="Isomap 2",
                save_path=os.path.join(dir_res, "pancreas_isomap.pdf"),
            )
            print(f"Saved 2D Isomap embedding: {isomap_2d_embedding_path}")

    if requested_umap_dim == 3 and not os.path.exists(umap_3d_embedding_path):
        x_umap_3d, labels_umap_3d = compute_and_save_pancreas_umap_embedding(
            umap_3d_embedding_path,
            preprocessing=preprocessing,
            umap=umap,
            n_components=3,
            random_state=seed,
        )
        plot_3d_umap(
            x_umap_3d,
            labels=labels_umap_3d,
            save_path=os.path.join(dir_res, "pancreas_umap_3d_views.pdf"),
            random_state=seed,
        )
        print(f"Saved 3D UMAP embedding: {umap_3d_embedding_path}")
    elif requested_umap_dim == 2 and not os.path.exists(umap_2d_embedding_path):
        x_umap, labels = compute_and_save_pancreas_umap_embedding(
            umap_2d_embedding_path,
            preprocessing=preprocessing,
            umap=umap,
            n_components=2,
            random_state=seed,
        )
        print(f"Saved 2D UMAP embedding: {umap_2d_embedding_path}")

    if requested_isomap_dim == 3 and not os.path.exists(isomap_3d_embedding_path):
        x_isomap_3d, labels_isomap_3d = compute_and_save_pancreas_isomap_embedding(
            isomap_3d_embedding_path,
            preprocessing=preprocessing,
            isomap=isomap,
            n_components=3,
            random_state=seed,
        )
        plot_3d_embedding_views(
            x_isomap_3d,
            labels=labels_isomap_3d,
            title="scVelo pancreas Isomap 3D",
            save_path=os.path.join(dir_res, "pancreas_isomap_3d_views.pdf"),
            point_fraction=1.0,
            random_state=seed,
        )
        print(f"Saved 3D Isomap embedding: {isomap_3d_embedding_path}")
    elif requested_isomap_dim == 2 and not os.path.exists(isomap_2d_embedding_path):
        x_isomap, labels_isomap = compute_and_save_pancreas_isomap_embedding(
            isomap_2d_embedding_path,
            preprocessing=preprocessing,
            isomap=isomap,
            n_components=2,
            random_state=seed,
        )
        plot_categorical_embedding(
            x_isomap,
            labels=labels_isomap,
            title="scVelo pancreas Isomap",
            xlabel="Isomap 1",
            ylabel="Isomap 2",
            save_path=os.path.join(dir_res, "pancreas_isomap.pdf"),
        )
        print(f"Saved 2D Isomap embedding: {isomap_2d_embedding_path}")

    plot_categorical_embedding(
        x_umap,
        labels=labels,
        title="scVelo pancreas UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=os.path.join(dir_res, "pancreas_umap.pdf"),
    )

    if finsler_optimizer is not None:
        init_3d, init_description = resolve_finsler_init(
            init_finsler_mds,
            embedding_sources={
                "umap": umap_2d_embedding_path,
                "umap_2d": umap_2d_embedding_path,
                "umap_3d": umap_3d_embedding_path,
                "isomap": isomap_2d_embedding_path,
                "isomap_2d": isomap_2d_embedding_path,
                "isomap_3d": isomap_3d_embedding_path,
                "smacof": latest_embedding_path(dir_res_raw, "pancreas_randers_smacof_*.npz"),
                "path_frozen": latest_embedding_path(dir_res_raw, "pancreas_path_frozen_*.npz"),
                "soft_bf": latest_embedding_path(dir_res_raw, "pancreas_soft_bf_*.npz"),
            },
            n_samples=len(x_umap),
            n_components=3,
        )
        if adata is not None and init_3d is not None:
            adata.obsm["X_finsler_init"] = init_3d

        optimizer_kind = normalize_finsler_optimizer(finsler_optimizer)
        if optimizer_kind == "smacof_randers":
            print(
                f"Running Randers SMACOF alpha={randers_alpha_embedding} "
                f"from {init_description}"
            )
            embedding_metric = RandersMetric(alpha=randers_alpha_embedding)
            embedding_metric_tag_value = embedding_metric_tag(embedding_metric)
            proj_finsler, stress_finsler = fit_finsler_mds(
                dists_velocity,
                metric=embedding_metric,
                optimizer="smacof_randers",
                init=init_3d,
                n_components=3,
                n_init=1,
                n_jobs=1,
                max_iter=smacof["max_iter"],
                pseudo_inv_solver="gmres",
                project_on_V=True,
                check_monotony=smacof["check_monotony"],
                device=smacof["device"],
                print_time=True,
                verbose=1,
            )
            full_geodesic_stress = geodesic_embedding_stress(
                proj_finsler,
                dists_velocity,
                metric=embedding_metric,
                n_neighbors=path_frozen["graph_neighbors"],
                on_unreachable="inf",
            )
            print(f"smacof_randers final full geodesic stress: {full_geodesic_stress}")
            output_cache_path = os.path.join(
                dir_res_raw,
                "pancreas_randers_smacof_"
                f"{velocity_formula_tag}_{embedding_metric_tag_value}_"
                f"iter{smacof['max_iter']}_seed{seed}.npz",
            )
            cache_payload = dict(
                embedding=proj_finsler,
                stress=np.asarray(stress_finsler),
                full_geodesic_stress=np.asarray(full_geodesic_stress),
                init_finsler_mds=np.asarray(str(init_finsler_mds)),
                metadata_json=json.dumps(
                    {
                        "optimizer": optimizer_kind,
                        "init": str(init_finsler_mds),
                        "seed": seed,
                        "velocity": velocity,
                        "randers_alpha_embedding": randers_alpha_embedding,
                        "smacof": smacof,
                        "path_frozen": path_frozen,
                        "soft_bf": soft_bf,
                    },
                    sort_keys=True,
                ),
            )
            np.savez(output_cache_path, **cache_payload)
            print(f"Saved {optimizer_kind} embedding: {output_cache_path}")
            if adata is not None:
                adata.obsm[f"X_finsler_randers_alpha_{cache_token(randers_alpha_embedding)}"] = proj_finsler
            plot_3d_embedding_views(
                proj_finsler,
                labels=labels,
                title=(
                    f"Pancreas Randers SMACOF "
                    f"({velocity_formula_tag}, alpha={randers_alpha_embedding})"
                ),
                save_path=os.path.join(
                    dir_res,
                    f"smacof_randers_{embedding_metric_tag_value}.pdf",
                ),
                point_fraction=1.0,
                random_state=seed,
            )
            print(f"{optimizer_kind} optimizer stress: {stress_finsler}")
            plt.close("all")
            return adata, dists_velocity
        elif optimizer_kind == "path_frozen":
            print(f"Running path-frozen {metric_display_name(geodesic_metric_obj)} from {init_description}")
            proj_finsler, stress_finsler = fit_finsler_mds(
                dists_velocity,
                metric=geodesic_metric_obj,
                optimizer="path_frozen",
                init=init_3d,
                n_components=3,
                **path_frozen,
                mask_random_state=seed,
                target_random_state=seed + 3,
                print_time=True,
            )
            output_cache_path = path_frozen_cache_path
            full_geodesic_stress = None
            plot_title = (
                f"Pancreas path-frozen "
                f"({velocity_formula_tag}, {metric_display_name(geodesic_metric_obj)})"
            )
            plot_path = os.path.join(
                dir_res,
                f"path_frozen_{geodesic_metric_tag}.pdf",
            )
            adata_key = "X_finsler_path_frozen"
        elif optimizer_kind == "soft_bf":
            print(f"Running soft-BF {metric_display_name(geodesic_metric_obj)} from {init_description}")
            proj_finsler, stress_finsler = fit_finsler_mds(
                dists_velocity,
                metric=geodesic_metric_obj,
                optimizer="soft_bellman_ford",
                init=init_3d,
                n_components=3,
                **soft_bf,
                mask_random_state=seed,
                target_random_state=seed + 3,
                print_time=True,
            )
            output_cache_path = soft_bf_cache_path
            full_geodesic_stress = None
            plot_title = (
                f"Pancreas soft-BF "
                f"({velocity_formula_tag}, {metric_display_name(geodesic_metric_obj)})"
            )
            plot_path = os.path.join(
                dir_res,
                f"soft_bf_{geodesic_metric_tag}.pdf",
            )
            adata_key = "X_finsler_soft_bf"
        else:
            raise ValueError(
                "finsler_optimizer must be one of {'smacof_randers', 'path_frozen', 'soft_bf', None}."
            )

        cache_payload = dict(
            embedding=proj_finsler,
            stress=np.asarray(stress_finsler),
            init_finsler_mds=np.asarray(str(init_finsler_mds)),
            metadata_json=json.dumps(
                {
                    "optimizer": optimizer_kind,
                    "init": str(init_finsler_mds),
                    "seed": seed,
                    "velocity": velocity,
                    "randers_alpha_embedding": randers_alpha_embedding,
                    "geodesic_metric": metric_metadata(geodesic_metric_obj),
                    "smacof": smacof,
                    "path_frozen": path_frozen,
                    "soft_bf": soft_bf,
                },
                sort_keys=True,
            ),
        )
        if full_geodesic_stress is not None:
            cache_payload["full_geodesic_stress"] = np.asarray(full_geodesic_stress)
        np.savez(output_cache_path, **cache_payload)
        print(f"Saved {optimizer_kind} embedding: {output_cache_path}")
        if adata is not None:
            adata.obsm[adata_key] = proj_finsler
        plot_3d_embedding_views(
            proj_finsler,
            labels=labels,
            title=plot_title,
            save_path=plot_path,
            point_fraction=1.0,
            random_state=seed,
        )
        print(f"{optimizer_kind} optimizer stress: {stress_finsler}")

    plt.show()
    return adata, dists_velocity


def project_velocity_to_pca(adata, n_pcs):
    """Project scVelo gene-space velocities onto Scanpy/scVelo PCA axes."""
    if "velocity" not in adata.layers:
        raise ValueError("adata.layers['velocity'] is missing. Run scv.tl.velocity first.")
    if "PCs" not in adata.varm:
        raise ValueError("adata.varm['PCs'] is missing. Run PCA/moments before projecting velocity.")

    velocity = adata.layers["velocity"]
    if sparse.issparse(velocity):
        velocity = velocity.toarray()
    velocity = np.asarray(velocity, dtype=float)
    velocity = np.nan_to_num(velocity, copy=False)

    pcs = np.asarray(adata.varm["PCs"][:, :n_pcs], dtype=float)
    velocity_pca = velocity @ pcs
    return np.asarray(velocity_pca, dtype=float)


def normalize_velocity_distance_formula(distance_formula):
    if not isinstance(distance_formula, str):
        raise TypeError("velocity['distance_formula'] must be 'exponential' or 'randers'.")
    formula = distance_formula.lower()
    aliases = {
        "exp": "exponential",
        "exponential": "exponential",
        "randers": "randers",
        "local_randers": "randers",
        "linear_randers": "randers",
    }
    if formula not in aliases:
        raise ValueError("velocity['distance_formula'] must be 'exponential' or 'randers'.")
    return aliases[formula]


def velocity_distance_formula_tag(distance_formula):
    formula = normalize_velocity_distance_formula(distance_formula)
    if formula == "exponential":
        return "vexp"
    if formula == "randers":
        return "vrand"
    raise RuntimeError(f"Unhandled velocity distance formula {formula!r}.")


def make_embedding_metric(config):
    if isinstance(config, str):
        config = {"kind": config}
    if not isinstance(config, dict):
        raise TypeError("geodesic_metric must be a dict or metric kind string.")
    kind = normalize_embedding_metric_kind(config.get("kind", "convexified_matsumoto"))
    alpha = float(config.get("alpha", 0.0))
    if kind == "randers":
        return RandersMetric(alpha=alpha)
    if kind == "matsumoto":
        return MatsumotoMetric(
            alpha=alpha,
            max_phi=config.get("max_phi", None),
            forbidden_grad_norm=config.get("forbidden_grad_norm", None),
        )
    if kind == "convexified_matsumoto":
        return ConvexifiedMatsumotoMetric(alpha=alpha)
    raise RuntimeError(f"Unhandled embedding metric kind {kind!r}.")


def normalize_embedding_metric_kind(kind):
    if not isinstance(kind, str):
        raise TypeError("metric kind must be a string.")
    normalized = kind.lower()
    aliases = {
        "r": "randers",
        "randers": "randers",
        "m": "matsumoto",
        "mats": "matsumoto",
        "matsumoto": "matsumoto",
        "cm": "convexified_matsumoto",
        "cmats": "convexified_matsumoto",
        "convexified_matsumoto": "convexified_matsumoto",
        "convexifiedmatsumoto": "convexified_matsumoto",
    }
    if normalized not in aliases:
        raise ValueError(
            "metric kind must be one of {'randers', 'matsumoto', 'convexified_matsumoto'}."
        )
    return aliases[normalized]


def embedding_metric_tag(metric):
    if isinstance(metric, RandersMetric):
        return f"r{cache_token(metric.alpha)}"
    if isinstance(metric, ConvexifiedMatsumotoMetric):
        return f"cmats{cache_token(metric.alpha)}"
    if isinstance(metric, MatsumotoMetric):
        return f"mats{cache_token(metric.alpha)}"
    raise TypeError(f"Unsupported embedding metric {type(metric).__name__}.")


def metric_display_name(metric):
    if isinstance(metric, RandersMetric):
        return f"Randers alpha={metric.alpha:g}"
    if isinstance(metric, ConvexifiedMatsumotoMetric):
        return f"Convexified Matsumoto alpha={metric.alpha:g}"
    if isinstance(metric, MatsumotoMetric):
        return f"Matsumoto alpha={metric.alpha:g}"
    return type(metric).__name__


def metric_metadata(metric):
    if isinstance(metric, RandersMetric):
        return {"kind": "randers", "alpha": metric.alpha}
    if isinstance(metric, ConvexifiedMatsumotoMetric):
        return {"kind": "convexified_matsumoto", "alpha": metric.alpha}
    if isinstance(metric, MatsumotoMetric):
        return {
            "kind": "matsumoto",
            "alpha": metric.alpha,
            "max_phi": metric.max_phi,
            "forbidden_grad_norm": metric.forbidden_grad_norm,
        }
    raise TypeError(f"Unsupported embedding metric {type(metric).__name__}.")


def pancreas_umap_embedding_path(cache_dir, cache_tag, *, n_components):
    if int(n_components) == 2:
        return os.path.join(cache_dir, f"pancreas_umap_{cache_tag}.npy")
    return os.path.join(cache_dir, f"pancreas_umap_{int(n_components)}d_{cache_tag}.npy")


def pancreas_isomap_embedding_path(cache_dir, cache_tag, *, n_components):
    if int(n_components) == 2:
        return os.path.join(cache_dir, f"pancreas_isomap_{cache_tag}.npy")
    return os.path.join(cache_dir, f"pancreas_isomap_{int(n_components)}d_{cache_tag}.npy")


def compute_and_save_pancreas_umap_embedding(
        output_path,
        *,
        preprocessing,
        umap,
        n_components,
        random_state,
):
    # Keep imports local so non-pancreas scripts do not require scanpy/scvelo.
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")

    print(f"Loading scVelo pancreas dataset for {n_components}D UMAP")
    adata = scv.datasets.pancreas()
    print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

    print("Preprocessing for UMAP")
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=preprocessing["min_shared_counts"],
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=preprocessing["n_top_genes"],
        flavor="seurat",
        subset=True,
    )
    sc.tl.pca(adata, n_comps=preprocessing["n_pcs"], random_state=random_state)
    sc.pp.neighbors(
        adata,
        n_neighbors=umap["n_neighbors"],
        n_pcs=preprocessing["n_pcs"],
        random_state=random_state,
    )
    embedding = compute_umap_from_neighbors(
        adata,
        umap,
        n_components=n_components,
        random_state=random_state,
    )
    labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    np.save(output_path, embedding)
    return embedding, labels


def compute_and_save_pancreas_isomap_embedding(
        output_path,
        *,
        preprocessing,
        isomap,
        n_components,
        random_state,
):
    # Keep imports local so non-pancreas scripts do not require scanpy/scvelo.
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 3
    scv.settings.set_figure_params("scvelo")

    print(f"Loading scVelo pancreas dataset for {n_components}D Isomap")
    adata = scv.datasets.pancreas()
    print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

    print("Preprocessing for Isomap")
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=preprocessing["min_shared_counts"],
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=preprocessing["n_top_genes"],
        flavor="seurat",
        subset=True,
    )
    sc.tl.pca(adata, n_comps=preprocessing["n_pcs"], random_state=random_state)
    X_pca = np.asarray(adata.obsm["X_pca"][:, :preprocessing["n_pcs"]], dtype=float)
    embedding = compute_isomap_from_pca(
        X_pca,
        isomap,
        n_components=n_components,
    )
    embedding = np.asarray(embedding, dtype=float)
    labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    np.save(output_path, embedding)
    return embedding, labels


def compute_isomap_from_pca(X_pca, isomap, *, n_components):
    print(
        f"Computing {n_components}D Isomap "
        f"(n_neighbors={isomap['n_neighbors']})"
    )
    embedding = utils.IsomapWithPreds(
        n_components=n_components,
        n_neighbors=isomap["n_neighbors"],
    ).fit_transform(np.asarray(X_pca, dtype=float))
    return np.asarray(embedding, dtype=float)


def compute_umap_from_neighbors(adata, umap, *, n_components, random_state):
    import scanpy as sc

    print(
        f"Computing {n_components}D UMAP "
        f"(maxiter={umap['maxiter']}, negative_sample_rate={umap['negative_sample_rate']})"
    )
    sc.tl.umap(
        adata,
        n_components=n_components,
        min_dist=umap["min_dist"],
        spread=umap["spread"],
        maxiter=umap["maxiter"],
        negative_sample_rate=umap["negative_sample_rate"],
        init_pos=umap["init_pos"],
        random_state=random_state,
    )
    return np.asarray(adata.obsm["X_umap"], dtype=float)


def plot_3d_umap(embedding, *, labels, save_path, random_state):
    plot_3d_embedding_views(
        embedding,
        labels=labels,
        title="scVelo pancreas UMAP 3D",
        save_path=save_path,
        point_fraction=1.0,
        random_state=random_state,
    )
    print(f"Saved 3D UMAP views: {save_path}")


def requested_embedding_init_dimension(init_finsler_mds, method):
    if init_finsler_mds is None:
        return None
    init_kind = normalize_finsler_init_kind(init_finsler_mds)
    if init_kind == f"{method}_2d":
        return 2
    if init_kind == f"{method}_3d":
        return 3
    return None


def resolve_finsler_init(init_finsler_mds, *, embedding_sources, n_samples, n_components=3):
    """Load the initial embedding used by the selected Finsler-MDS optimizer."""
    if init_finsler_mds is None:
        return None, "random initialization"

    init_kind = normalize_finsler_init_kind(init_finsler_mds)

    source_path = embedding_sources.get(init_kind)
    if source_path is None or not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Requested init_finsler_mds={init_kind!r}, but no saved embedding was found."
        )

    embedding = load_saved_embedding(source_path)
    if init_kind in {"umap_2d", "umap_3d", "isomap_2d", "isomap_3d"}:
        if embedding.shape[0] != n_samples:
            raise ValueError(
                f"Saved {init_kind} init has {embedding.shape[0]} samples, expected {n_samples}: {source_path}"
            )
        init = low_dim_to_finsler_init(embedding, n_components)
        init_label = "UMAP" if init_kind.startswith("umap") else "Isomap"
        return init, f"saved {embedding.shape[1]}D {init_label} ({source_path})"

    expected_shape = (n_samples, n_components)
    if embedding.shape != expected_shape:
        raise ValueError(
            f"Saved {init_kind} init has shape {embedding.shape}, expected {expected_shape}: {source_path}"
        )
    return embedding, f"saved {init_kind} ({source_path})"


def normalize_finsler_init_kind(init_finsler_mds):
    if not isinstance(init_finsler_mds, str):
        raise TypeError(
            "init_finsler_mds must be one of "
            "{'umap', 'umap_2D', 'umap_3D', 'isomap', 'isomap_2D', 'isomap_3D', "
            "'smacof', 'path_frozen', 'soft_bf', None}."
        )

    init_kind = init_finsler_mds.lower()
    if init_kind in {"umap", "umap_2d", "umap2d"}:
        return "umap_2d"
    if init_kind in {"umap_3d", "umap3d"}:
        return "umap_3d"
    if init_kind in {"isomap", "isomap_2d", "isomap2d"}:
        return "isomap_2d"
    if init_kind in {"isomap_3d", "isomap3d"}:
        return "isomap_3d"
    if init_kind in {"soft_bellman_ford", "sbf"}:
        return "soft_bf"
    if init_kind in {"smacof", "path_frozen", "soft_bf"}:
        return init_kind
    raise ValueError(
        "init_finsler_mds must be one of "
        "{'umap', 'umap_2D', 'umap_3D', 'isomap', 'isomap_2D', 'isomap_3D', "
        "'smacof', 'path_frozen', 'soft_bf', None}."
    )


def low_dim_to_finsler_init(embedding, n_components):
    embedding = np.asarray(embedding, dtype=float)
    init = np.zeros((len(embedding), n_components), dtype=float)
    n_copy = min(embedding.shape[1], n_components)
    init[:, :n_copy] = embedding[:, :n_copy]
    return init


def load_saved_embedding(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix == ".npz":
        with np.load(path) as cache:
            if "embedding" not in cache:
                raise KeyError(f"Saved embedding file has no 'embedding' array: {path}")
            return np.asarray(cache["embedding"], dtype=float)
    raise ValueError(f"Unsupported embedding file extension for {path}")


def latest_embedding_path(cache_dir, pattern):
    candidates = list(Path(cache_dir).glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_finsler_optimizer(finsler_optimizer):
    if not isinstance(finsler_optimizer, str):
        raise TypeError(
            "finsler_optimizer must be one of {'smacof_randers', 'path_frozen', 'soft_bf', None}."
        )
    optimizer = finsler_optimizer.lower()
    if optimizer in {"smacof", "randers_smacof", "smacof_randers"}:
        return "smacof_randers"
    if optimizer in {"path_frozen", "frozen_paths"}:
        return "path_frozen"
    if optimizer in {"soft_bf", "soft_bellman_ford", "sbf"}:
        return "soft_bf"
    raise ValueError(
        "finsler_optimizer must be one of {'smacof_randers', 'path_frozen', 'soft_bf', None}."
    )


def load_velocity_inputs_cache(cache_path, expected_metadata, *, fallback_paths=()):
    candidate_paths = [cache_path]
    candidate_paths.extend(path for path in fallback_paths if path not in candidate_paths)
    for candidate_path in candidate_paths:
        cached = _load_one_velocity_inputs_cache(candidate_path, expected_metadata)
        if cached is not None:
            return cached
    return None


def _load_one_velocity_inputs_cache(cache_path, expected_metadata):
    if not os.path.exists(cache_path):
        return None

    print(f"Loading cached pancreas UMAP and velocity distances: {cache_path}")
    with np.load(cache_path) as cache:
        if "metadata_json" not in cache:
            print("Cached pancreas inputs have no metadata; recomputing.")
            return None

        cached_metadata = json.loads(str(cache["metadata_json"].item()))
        if not velocity_cache_metadata_matches(cached_metadata, expected_metadata):
            print("Cached pancreas inputs were produced with different parameters; recomputing.")
            return None

        x_umap = cache["x_umap"]
        dists_velocity = cache["dists_velocity"]
        labels = cached_labels(cache)

    return x_umap, dists_velocity, labels


def velocity_cache_metadata_matches(cached_metadata, expected_metadata):
    if cached_metadata == expected_metadata:
        return True
    # Older caches used the exponential formula before the formula was explicit
    # in metadata. Keep them usable to avoid recomputing scVelo unnecessarily.
    if (
            expected_metadata.get("velocity_distance_formula") != "exponential"
            or "velocity_cos_clip" in expected_metadata
    ):
        return False
    legacy_expected = dict(expected_metadata)
    legacy_expected.pop("velocity_distance_formula", None)
    return cached_metadata == legacy_expected


def pancreas_cache_metadata(**params):
    """Return JSON-serializable metadata for cache validation."""
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in params.items()
    }


def cache_token(value):
    return str(value).replace(os.sep, "-").replace(".", "p")


def labels_to_numpy(labels):
    if labels is None:
        return None
    if hasattr(labels, "to_numpy"):
        labels = labels.to_numpy()
    return np.asarray(labels, dtype=str)


def labels_to_cache(labels):
    if labels is None:
        return np.asarray([], dtype=str)
    return np.asarray(labels, dtype=str)


def cached_labels(cache):
    if "labels" not in cache:
        return None
    labels = cache["labels"]
    if labels.size == 0:
        return None
    return labels


if __name__ == "__main__":
    main_pancreas()
