import json
import os
from pathlib import Path
import sys

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
from finsler_mds.evaluation.rna_velocity.pancreas_gap import (
    gap_arrays_to_cache,
    normalize_pancreas_gap_config,
    pancreas_gap_prefix,
    select_pancreas_gap,
)
from finsler_mds.evaluation.rna_velocity import load_or_compute_boundary_neighbor_plan
from finsler_mds.utils import plot_3d_embedding_views, plot_categorical_embedding
from finsler_mds.utils.pancreas import (
    PANCREAS_CLUSTER_COLORS,
    PANCREAS_DATASET_SOURCE,
    PANCREAS_TRANSITIONS,
    apply_distance_pair_reweight,
    apply_frontier_pair_weight,
    cluster_balanced_pair_weights,
    normalize_pair_weights,
    compute_pancreas_velocity,
    compute_pancreas_velocity_graph,
    load_pancreas_dataset,
    preprocess_pancreas_for_velocity,
    setup_scvelo_settings,
)

def main_pancreas(config_overrides=None):
    # Hyperparameters
    seed = 42
    script_dir = Path(__file__).resolve().parent
    dir_res_base = script_dir / "res" / "pancreas"

    preprocessing = {
        "min_shared_counts": 20,
        "n_top_genes": 3000,
        "n_pcs": 50,
        "moments_n_neighbors": 30,
    }
    velocity = {
        "mode": "dynamical",
        "distance_formula": "randers",  # one of {"exponential", "randers", "matsumoto"}
        "alpha": 1, # should be positive; randers needs alpha*cos_clip<1
        "cos_clip": 0.99,
        "velocity_neighbors": 30, # to average the velocity locally (if "average_velocity" is True)
        "kNN_euclid": 30,  # Euclidean kNN support before adding Finsler-nearest outgoing edges.
        "kNN_finsler": 0,  # Additional outgoing neighbors selected by local asymmetric distance.
        "average_velocity": True,
        "symmetrize_support": True, # for the kNN graph
        "graph_n_jobs": -1,
        "recover_dynamics_max_iter": 20,
        "recover_dynamics_n_jobs": -1,
    }
    gap = {
        "enabled": False,
        "name": "preendocrine",
        "selection": "veloviz_latent_time",
        "n_before": 300,
        "n_after": 300,
        "removed_labels": ("Pre-endocrine",),
        "before_labels": ("Ngn3 high EP",),
        "after_labels": ("Alpha", "Beta", "Delta", "Epsilon"),
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
    #MDS 2D:
    #gd_2d_vrand0_r0_s42.npz
    finsler_optimizer = "path_frozen"  # one of {"smacof", "gradient_descent", "finsler_umap", "path_frozen", "soft_bf"}
    init_finsler_mds = "path_frozen"  # one of {"umap_2D", "umap_3D", "isomap_2D", "isomap_3D", "smacof", "gradient_descent", "finsler_umap", "path_frozen", "soft_bf"}
    embedding_dim = 3  # 2 or 3; 2D inits can seed 3D, but 3D inits cannot seed 2D.
    cluster_reweight_rho = 0
    frontier_pairs_weight = 1
    selected_frontiers = "all"  # one of {"cbdir", "all"}
    distance_reweighting = {"power": 0, "epsilon": 1e-6}

    # SMACOF always uses Randers regardless of finsler_metric
    finsler_metric = "randers"  # one of {"randers", "matsumoto", "convexified_matsumoto"}
    alpha_embedding = 0.8
    smacof = {
        "max_iter": 1000,
        "eps": 1e-6,
        "device": "auto",
        "check_monotony": True,
        "project_on_V": True,
        "version": "corrected",
    }
    gradient_descent = {
        "max_iter": 100,
        "eps": 1e-8,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-10, "maxls": 80, "maxcor": 30},
        "device": "auto",
        "gpu_block_size": 256,
        "verbose": 1,
    }
    finsler_umap = {
        "n_neighbors": 30,
        "symmetrize_support": True,
        "symmetrize_rho": False,
        "symmetrize_sigma": True,
        "min_dist": 0.5,
        "spread": 1.0,
        "max_iter": 1500,
        "learning_rate": 1,
        "batch_size": 512,
        "negative_sample_rate": 10,
        "negative_sample_weight": 1.0,
        "negative_metric": "euclidean",
        "gradient_clip": 4.0,
        "verbose": 1,
        "log_frequency": 500,
        "backend": "numba",
    }
    path_frozen = {
        "graph_neighbors": 30,
        "outer_iter": 40,
        "inner_iter": 5,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-9, "maxls": 50},
        "n_landmark": 600,
        "random_landmark_fraction": 1.0, # 1.0=random, 0.0=farthest
        "n_local_pairs": 30,
        "local_pair_mode": "direct",
        "targets_per_landmark": 1000,
        "local_global_reweighting": "count",
        "local_weight": 10,
        "direct_stress_weight": 0.0,
        "direct_stress_mode": "hinge",
        "direct_stress_margin": 0.5,
        "outer_step_size": 0.05,
        "device": "auto",
        "verbose": 1,
    }
    soft_bf = {
        "graph_neighbors": 20,
        "beta": 80.0,
        "n_relaxations": 45,
        "max_iter": 50,
        "n_graph_updates": 5,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-9, "maxls": 50},
        "n_global_landmarks": 250,
        "n_local_neighbors": 25,
        "local_pair_mode": "direct",
        "max_global_targets_per_source": 200,
        "local_global_reweighting": "energy",
        "local_weight": 1,
        "device": "auto",
        "source_batch_size": 32,
        "on_unreachable": "warn_skip",
        "verbose": 1,
    }

    if config_overrides:
        overrides = dict(config_overrides)
        alpha_embedding = overrides.pop("alpha_embedding", alpha_embedding)
        finsler_metric = overrides.pop("finsler_metric", finsler_metric)
        finsler_optimizer = overrides.pop("finsler_optimizer", finsler_optimizer)
        init_finsler_mds = overrides.pop("init_finsler_mds", init_finsler_mds)
        embedding_dim = overrides.pop("embedding_dim", embedding_dim)
        cluster_reweight_rho = overrides.pop("cluster_reweight_rho", cluster_reweight_rho)
        frontier_pairs_weight = overrides.pop("frontier_pairs_weight", frontier_pairs_weight)
        selected_frontiers = overrides.pop("selected_frontiers", selected_frontiers)
        if "distance_reweighting" in overrides:
            distance_reweighting = {**distance_reweighting, **overrides.pop("distance_reweighting")}
        _deep_update_dicts(
            overrides,
            preprocessing=preprocessing,
            velocity=velocity,
            gap=gap,
            umap=umap,
            isomap=isomap,
            smacof=smacof,
            gradient_descent=gradient_descent,
            finsler_umap=finsler_umap,
            path_frozen=path_frozen,
            soft_bf=soft_bf,
        )

    np.random.seed(seed)
    embedding_dim = normalize_embedding_dim(embedding_dim)

    gap = normalize_pancreas_gap_config(gap)
    if gap["enabled"]:
        dir_res = str(dir_res_base / "gap")
        dir_res_raw = dir_res
    else:
        dir_res = str(dir_res_base)
        dir_res_raw = str(dir_res_base / "raw")
    os.makedirs(dir_res, exist_ok=True)
    os.makedirs(dir_res_raw, exist_ok=True)
    dataset_prefix = pancreas_gap_prefix(gap)
    file_prefix = pancreas_file_prefix(dataset_prefix)
    state_cache_metadata = pancreas_state_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=preprocessing["min_shared_counts"],
        n_top_genes=preprocessing["n_top_genes"],
        n_pcs=preprocessing["n_pcs"],
        moments_n_neighbors=preprocessing["moments_n_neighbors"],
        velocity_mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        gap=gap,
        seed=seed,
    )
    velocity_cache_metadata = pancreas_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        min_shared_counts=preprocessing["min_shared_counts"],
        n_top_genes=preprocessing["n_top_genes"],
        n_pcs=preprocessing["n_pcs"],
        moments_n_neighbors=preprocessing["moments_n_neighbors"],
        velocity_mode=velocity["mode"],
        velocity_distance_formula=normalize_velocity_distance_formula(velocity["distance_formula"]),
        velocity_alpha=velocity["alpha"],
        velocity_cos_clip=velocity["cos_clip"],
        velocity_neighbors=velocity["velocity_neighbors"],
        velocity_kNN_euclid=velocity["kNN_euclid"],
        velocity_kNN_finsler=velocity["kNN_finsler"],
        average_velocity=velocity["average_velocity"],
        symmetrize_velocity_support=velocity["symmetrize_support"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        umap_n_neighbors=umap["n_neighbors"],
        umap_min_dist=umap["min_dist"],
        umap_spread=umap["spread"],
        umap_maxiter=umap["maxiter"],
        umap_negative_sample_rate=umap["negative_sample_rate"],
        umap_init_pos=umap["init_pos"],
        gap=gap,
        seed=seed,
    )
    umap_variant_tag = pancreas_umap_variant_tag(umap)
    umap_cache_tag = f"{cache_token(velocity['mode'])}_{umap_variant_tag}s{seed}"
    isomap_cache_tag = f"k{isomap['n_neighbors']}_s{seed}"
    velocity_cache_formula_tag = velocity_distance_formula_tag(velocity["distance_formula"])
    velocity_formula_tag = velocity_distance_formula_tag(
        velocity["distance_formula"], alpha=velocity["alpha"]
    )
    velocity_cache_tag = (
        f"{cache_token(velocity['mode'])}_"
        f"{velocity_cache_formula_tag}_"
        f"valpha{cache_token(velocity['alpha'])}_"
        f"cclip{cache_token(velocity['cos_clip'])}_"
        f"ke{velocity['kNN_euclid']}_kf{velocity['kNN_finsler']}_"
        f"{umap_variant_tag}s{seed}"
    )
    velocity_cache_path = os.path.join(
        dir_res_raw,
        f"{dataset_prefix}_velocity_inputs_{velocity_cache_tag}.npz",
    )
    state_cache_path = pancreas_state_cache_path(
        dir_res_raw,
        preprocessing=preprocessing,
        velocity=velocity,
        dataset_prefix=dataset_prefix,
        seed=seed,
    )
    gap_selection_cache_path = os.path.join(
        dir_res_raw,
        f"{dataset_prefix}_selection_s{seed}.npz",
    )
    legacy_velocity_cache_paths = []
    if not gap["enabled"]:
        legacy_velocity_cache_paths.append(
            os.path.join(
                dir_res_raw,
                f"pancreas_velocity_inputs_{cache_token(velocity['mode'])}_"
                f"{velocity_cache_formula_tag}_"
                f"valpha{cache_token(velocity['alpha'])}_"
                f"cclip{cache_token(velocity['cos_clip'])}_s{seed}.npz",
            )
        )
    if not gap["enabled"] and normalize_velocity_distance_formula(velocity["distance_formula"]) == "exponential":
        legacy_velocity_cache_paths.append(
            os.path.join(
                dir_res_raw,
                f"pancreas_velocity_inputs_{cache_token(velocity['mode'])}_"
                f"valpha{cache_token(velocity['alpha'])}_s{seed}.npz",
            )
        )
    # UMAP alone is not enough to rerun SMACOF: the optimizer also needs the
    # directed velocity dissimilarities, so those live in velocity_cache_path.
    umap_2d_embedding_path = pancreas_umap_embedding_path(
        dir_res_raw, umap_cache_tag, n_components=2, dataset_prefix=dataset_prefix
    )
    umap_3d_embedding_path = pancreas_umap_embedding_path(
        dir_res_raw, umap_cache_tag, n_components=3, dataset_prefix=dataset_prefix
    )
    isomap_2d_embedding_path = pancreas_isomap_embedding_path(
        dir_res_raw, isomap_cache_tag, n_components=2, dataset_prefix=dataset_prefix
    )
    isomap_3d_embedding_path = pancreas_isomap_embedding_path(
        dir_res_raw, isomap_cache_tag, n_components=3, dataset_prefix=dataset_prefix
    )
    embedding_metric_obj = make_embedding_metric({"kind": finsler_metric, "alpha": alpha_embedding})
    embedding_metric_tag_value = embedding_metric_tag(embedding_metric_obj)
    smacof_metric_obj = RandersMetric(alpha=alpha_embedding)
    smacof_metric_tag_value = embedding_metric_tag(smacof_metric_obj)
    embedding_dim_tag = finsler_embedding_dim_tag(embedding_dim)
    selected_frontiers = normalize_selected_frontiers(selected_frontiers)
    pair_weight_tag = pair_weight_output_tag(
        cluster_reweight_rho,
        frontier_pairs_weight,
        distance_reweighting=distance_reweighting,
        selected_frontiers=selected_frontiers,
    )
    path_frozen_cache_path = os.path.join(
        dir_res_raw,
        f"{file_prefix}pf_{embedding_dim_tag}"
        f"{velocity_formula_tag}_{embedding_metric_tag_value}{pair_weight_tag}_s{seed}.npz",
    )
    soft_bf_cache_path = os.path.join(
        dir_res_raw,
        f"{file_prefix}sbf_{embedding_dim_tag}"
        f"{velocity_formula_tag}_{embedding_metric_tag_value}{pair_weight_tag}_s{seed}.npz",
    )
    gradient_descent_cache_path = os.path.join(
        dir_res_raw,
        f"{file_prefix}gd_{embedding_dim_tag}"
        f"{velocity_formula_tag}_{embedding_metric_tag_value}{pair_weight_tag}_s{seed}.npz",
    )
    finsler_umap_cache_path = os.path.join(
        dir_res_raw,
        f"{file_prefix}fumap_{embedding_dim_tag}"
        f"{velocity_formula_tag}_{embedding_metric_tag_value}{pair_weight_tag}_s{seed}.npz",
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
        x_umap, dists_velocity, labels, cell_ids, original_indices, gap_arrays = cached_inputs
        if not os.path.exists(umap_2d_embedding_path):
            np.save(umap_2d_embedding_path, x_umap)
        print(
            "Loaded directed velocity distances: "
            f"{dists_velocity.shape[0]} x {dists_velocity.shape[1]}, "
            f"finite={np.isfinite(dists_velocity).mean():.3f}"
        )
    else:
        # Keep these imports local: the rest of the project should remain usable
        # even when single-cell dependencies are not installed.
        import scanpy as sc

        setup_scvelo_settings()
        state_inputs = load_pancreas_state_cache(state_cache_path, state_cache_metadata)
        if state_inputs is not None:
            adata, labels, cell_ids, original_indices, gap_arrays = state_inputs
        else:
            print(f"Loading pancreas dataset from {PANCREAS_DATASET_SOURCE}")
            adata = load_pancreas_dataset()
            print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")
            full_labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
            cell_ids_full = np.asarray(adata.obs_names, dtype=str)
            gap_ordering = None
            if gap["enabled"] and gap["selection"] in {"latent_time", "veloviz_latent_time"}:
                gap_ordering = load_or_compute_pancreas_gap_ordering(
                    adata,
                    gap_selection_cache_path,
                    preprocessing=preprocessing,
                    velocity=velocity,
                    gap=gap,
                    seed=seed,
                )
            gap_selection = select_pancreas_gap(full_labels, gap, cell_ids=cell_ids_full, ordering=gap_ordering)
            if gap_selection.enabled:
                print(
                    f"Applying pancreas gap {gap_selection.config['name']!r}: "
                    f"removed {gap_selection.removed_mask.sum()} cells; "
                    f"kept {gap_selection.keep_mask.sum()} cells "
                    f"({len(gap_selection.before_indices)} before, {len(gap_selection.after_indices)} after)"
                )
                adata = adata[gap_selection.keep_mask].copy()
            labels = gap_selection.labels
            cell_ids = gap_selection.cell_ids
            original_indices = gap_selection.original_indices
            gap_arrays = gap_arrays_to_cache(gap_selection)

            print("Preprocessing")
            preprocess_pancreas_for_velocity(adata, preprocessing, seed=seed)
            print(f"Preprocessed pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")

            print("Computing RNA velocity")
            if velocity["mode"] == "dynamical":
                print(
                    "Recovering scVelo dynamical model "
                    f"(max_iter={velocity['recover_dynamics_max_iter']}, "
                    f"n_jobs={velocity['recover_dynamics_n_jobs']})"
                )
            compute_pancreas_velocity(
                adata,
                mode=velocity["mode"],
                recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
                recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
            )
            save_pancreas_state_cache(
                state_cache_path,
                adata,
                labels=labels,
                cell_ids=cell_ids,
                original_indices=original_indices,
                gap_arrays=gap_arrays,
                metadata=state_cache_metadata,
            )

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
            kNN_euclid=velocity["kNN_euclid"],
            kNN_finsler=velocity["kNN_finsler"],
            alpha=velocity["alpha"],
            distance_formula=velocity["distance_formula"],
            cos_clip=velocity["cos_clip"],
            velocity_neighbors=velocity["velocity_neighbors"],
            average_velocity=velocity["average_velocity"],
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
        x_umap = orient_pancreas_embedding_by_velocity(
            adata,
            x_umap,
            velocity=velocity,
            label="2D UMAP",
        )
        np.save(umap_2d_embedding_path, x_umap)
        np.savez(
            velocity_cache_path,
            x_umap=x_umap,
            dists_velocity=dists_velocity,
            **gap_arrays,
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
            x_umap_3d = orient_pancreas_embedding_by_velocity(
                adata,
                x_umap_3d,
                velocity=velocity,
                label="3D UMAP",
            )
            np.save(umap_3d_embedding_path, x_umap_3d)
            plot_3d_umap(
                x_umap_3d,
                labels=labels,
                save_path=os.path.join(dir_res, f"{file_prefix}umap_3d.pdf"),
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
            x_isomap_3d = orient_pancreas_embedding_by_velocity(
                adata,
                x_isomap_3d,
                velocity=velocity,
                label="3D Isomap",
            )
            np.save(isomap_3d_embedding_path, x_isomap_3d)
            plot_3d_embedding_views(
                x_isomap_3d,
                labels=labels,
                title="scVelo pancreas Isomap 3D",
                save_path=os.path.join(dir_res, f"{file_prefix}isomap_3d.pdf"),
                point_fraction=1.0,
                random_state=seed,
                cmap=PANCREAS_CLUSTER_COLORS,
            )
            print(f"Saved 3D Isomap embedding: {isomap_3d_embedding_path}")
        elif requested_isomap_dim == 2 and not os.path.exists(isomap_2d_embedding_path):
            x_isomap = compute_isomap_from_pca(
                X_pca,
                isomap,
                n_components=2,
            )
            x_isomap = orient_pancreas_embedding_by_velocity(
                adata,
                x_isomap,
                velocity=velocity,
                label="2D Isomap",
            )
            np.save(isomap_2d_embedding_path, x_isomap)
            plot_categorical_embedding(
                x_isomap,
                labels=labels,
                title="scVelo pancreas Isomap",
                xlabel="Isomap 1",
                ylabel="Isomap 2",
                save_path=os.path.join(dir_res, f"{file_prefix}isomap.pdf"),
                cmap=PANCREAS_CLUSTER_COLORS,
            )
            print(f"Saved 2D Isomap embedding: {isomap_2d_embedding_path}")

    if requested_umap_dim == 3 and not os.path.exists(umap_3d_embedding_path):
        x_umap_3d, labels_umap_3d = compute_and_save_pancreas_umap_embedding(
            umap_3d_embedding_path,
            preprocessing=preprocessing,
            velocity=velocity,
            umap=umap,
            gap=gap,
            n_components=3,
            random_state=seed,
        )
        plot_3d_umap(
            x_umap_3d,
            labels=labels_umap_3d,
            save_path=os.path.join(dir_res, f"{file_prefix}umap_3d.pdf"),
            random_state=seed,
        )
        print(f"Saved 3D UMAP embedding: {umap_3d_embedding_path}")
    elif requested_umap_dim == 2 and not os.path.exists(umap_2d_embedding_path):
        x_umap, labels = compute_and_save_pancreas_umap_embedding(
            umap_2d_embedding_path,
            preprocessing=preprocessing,
            velocity=velocity,
            umap=umap,
            gap=gap,
            n_components=2,
            random_state=seed,
        )
        print(f"Saved 2D UMAP embedding: {umap_2d_embedding_path}")

    if requested_isomap_dim == 3 and not os.path.exists(isomap_3d_embedding_path):
        x_isomap_3d, labels_isomap_3d = compute_and_save_pancreas_isomap_embedding(
            isomap_3d_embedding_path,
            preprocessing=preprocessing,
            velocity=velocity,
            isomap=isomap,
            gap=gap,
            n_components=3,
            random_state=seed,
        )
        plot_3d_embedding_views(
            x_isomap_3d,
            labels=labels_isomap_3d,
            title="scVelo pancreas Isomap 3D",
            save_path=os.path.join(dir_res, f"{file_prefix}isomap_3d.pdf"),
            point_fraction=1.0,
            random_state=seed,
            cmap=PANCREAS_CLUSTER_COLORS,
        )
        print(f"Saved 3D Isomap embedding: {isomap_3d_embedding_path}")
    elif requested_isomap_dim == 2 and not os.path.exists(isomap_2d_embedding_path):
        x_isomap, labels_isomap = compute_and_save_pancreas_isomap_embedding(
            isomap_2d_embedding_path,
            preprocessing=preprocessing,
            velocity=velocity,
            isomap=isomap,
            gap=gap,
            n_components=2,
            random_state=seed,
        )
        plot_categorical_embedding(
            x_isomap,
            labels=labels_isomap,
            title="scVelo pancreas Isomap",
            xlabel="Isomap 1",
            ylabel="Isomap 2",
            save_path=os.path.join(dir_res, f"{file_prefix}isomap.pdf"),
            cmap=PANCREAS_CLUSTER_COLORS,
        )
        print(f"Saved 2D Isomap embedding: {isomap_2d_embedding_path}")

    plot_categorical_embedding(
        x_umap,
        labels=labels,
        title="scVelo pancreas UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=os.path.join(dir_res, f"{file_prefix}umap.pdf"),
        cmap=PANCREAS_CLUSTER_COLORS,
    )

    if finsler_optimizer is not None:
        frontier_plan = None
        if float(frontier_pairs_weight) != 1.0:
            frontier_plan = load_pancreas_frontier_plan(
                labels,
                selected_frontiers=selected_frontiers,
                adata=adata,
                preprocessing=preprocessing,
                seed=seed,
                eval_raw_dir=dir_res_base / "rna_velocity_evaluation" / "raw",
            )
        pair_weight, pair_weight_info = make_pancreas_pair_weight(
            labels,
            dists_velocity,
            cluster_reweight_rho,
            frontier_pairs_weight=frontier_pairs_weight,
            selected_frontiers=selected_frontiers,
            distance_reweighting=distance_reweighting,
            boundary_plan=frontier_plan,
        )
        optimizer_kind = normalize_finsler_optimizer(finsler_optimizer)
        init_finsler, init_description = resolve_finsler_init(
            init_finsler_mds,
            embedding_sources={
                "umap": umap_2d_embedding_path,
                "umap_2d": umap_2d_embedding_path,
                "umap_3d": umap_3d_embedding_path,
                "isomap": isomap_2d_embedding_path,
                "isomap_2d": isomap_2d_embedding_path,
                "isomap_3d": isomap_3d_embedding_path,
                "smacof": latest_finsler_embedding_path(
                    dir_res_raw, dataset_prefix, "smacof", embedding_dim
                ),
                "gradient_descent": latest_finsler_embedding_path(
                    dir_res_raw, dataset_prefix, "gd", embedding_dim
                ),
                "finsler_umap": latest_finsler_embedding_path(
                    dir_res_raw, dataset_prefix, "fumap", embedding_dim
                ),
                "path_frozen": latest_finsler_embedding_path(
                    dir_res_raw, dataset_prefix, "pf", embedding_dim
                ),
                "soft_bf": latest_finsler_embedding_path(
                    dir_res_raw, dataset_prefix, "sbf", embedding_dim
                ),
            },
            raw_dir=dir_res_raw,
            n_samples=len(x_umap),
            n_components=embedding_dim,
            allow_internal_spectral=optimizer_kind == "finsler_umap",
        )
        if adata is not None and isinstance(init_finsler, np.ndarray):
            adata.obsm["X_finsler_init"] = init_finsler

        if optimizer_kind == "smacof":
            print(
                f"Running {embedding_dim}D Randers SMACOF alpha={alpha_embedding} "
                f"from {init_description}"
            )
            proj_finsler, stress_finsler, smacof_n_iter = fit_finsler_mds(
                dists_velocity,
                metric=smacof_metric_obj,
                optimizer="smacof",
                init=init_finsler,
                n_components=embedding_dim,
                n_init=1,
                n_jobs=1,
                max_iter=smacof["max_iter"],
                eps=smacof["eps"],
                pseudo_inv_solver="gmres",
                project_on_V=smacof["project_on_V"],
                check_monotony=smacof["check_monotony"],
                device=smacof["device"],
                weight=pair_weight,
                version=smacof["version"],
                return_n_iter=True,
                print_time=True,
                verbose=1,
            )
            smacof_tag = smacof_output_tag(
                embedding_dim=embedding_dim,
                metric_tag=smacof_metric_tag_value,
                init_finsler_mds=init_finsler_mds,
                n_iter=smacof_n_iter,
                smacof_version=smacof["version"],
            )
            full_geodesic_stress = geodesic_embedding_stress(
                proj_finsler,
                dists_velocity,
                metric=smacof_metric_obj,
                n_neighbors=path_frozen["graph_neighbors"],
                weight=pair_weight,
                on_unreachable="inf",
            )
            print(f"smacof final full geodesic stress: {full_geodesic_stress}")
            output_cache_path = os.path.join(
                dir_res_raw,
                f"{file_prefix}{smacof_tag}_{velocity_formula_tag}{pair_weight_tag}_s{seed}.npz",
            )
            cache_payload = dict(
                embedding=proj_finsler,
                stress=np.asarray(stress_finsler),
                full_geodesic_stress=np.asarray(full_geodesic_stress),
                init_finsler_mds=np.asarray(str(init_finsler_mds)),
                **gap_arrays,
                metadata_json=json.dumps(
                    {
                        "optimizer": optimizer_kind,
                        "init": str(init_finsler_mds),
                        "embedding_dim": embedding_dim,
                        "seed": seed,
                        "gap": gap,
                        "velocity": velocity,
                        "finsler_metric": "randers",
                        "alpha_embedding": alpha_embedding,
                        "randers_alpha_embedding": alpha_embedding,
                        "cluster_reweight_rho": cluster_reweight_rho,
                        "frontier_pairs_weight": frontier_pairs_weight,
                        "distance_reweighting": distance_reweighting,
                        "selected_frontiers": selected_frontiers,
                        "pair_weighting": pair_weight_info,
                        "n_iter": int(smacof_n_iter),
                        "smacof": smacof,
                        "gradient_descent": gradient_descent,
                        "finsler_umap": finsler_umap,
                        "path_frozen": path_frozen,
                        "soft_bf": soft_bf,
                    },
                    sort_keys=True,
                ),
            )
            np.savez(output_cache_path, **cache_payload)
            print(f"Saved {optimizer_kind} embedding: {output_cache_path}")
            if adata is not None:
                adata.obsm[f"X_finsler_randers_alpha_{cache_token(alpha_embedding)}"] = proj_finsler
            plot_pancreas_finsler_embedding(
                proj_finsler,
                labels=labels,
                title=(
                    f"Pancreas {embedding_dim}D Randers SMACOF "
                    f"({velocity_formula_tag}, alpha={alpha_embedding})"
                ),
                save_path=os.path.join(
                    dir_res,
                    f"{file_prefix}{smacof_tag}_{velocity_formula_tag}{pair_weight_tag}.pdf",
                ),
                random_state=seed,
            )
            print(f"{optimizer_kind} optimizer stress: {stress_finsler}")
            plt.close("all")
            return adata, dists_velocity
        elif optimizer_kind in {"gradient_descent", "finsler_umap", "path_frozen", "soft_bf"}:
            specs = {
                "gradient_descent": ("gradient_descent", "gradient descent", "gd", gradient_descent_cache_path, gradient_descent, True),
                "finsler_umap": ("finsler_umap", "Finsler-UMAP", "fumap", finsler_umap_cache_path, finsler_umap, True),
                "path_frozen": ("path_frozen", "path-frozen", "pf", path_frozen_cache_path, path_frozen, False),
                "soft_bf": ("soft_bellman_ford", "soft-BF", "sbf", soft_bf_cache_path, soft_bf, False),
            }
            optimizer_name, display_name, file_tag, output_cache_path, options, has_n_iter = specs[optimizer_kind]
            print(f"Running {embedding_dim}D {display_name} {metric_display_name(embedding_metric_obj)} from {init_description}")
            optimizer_kwargs = dict(options)
            if has_n_iter:
                optimizer_kwargs.update(return_n_iter=True, random_state=seed)
            else:
                optimizer_kwargs.update(mask_random_state=seed, target_random_state=seed + 3)
            optimizer_result = fit_finsler_mds(
                dists_velocity,
                metric=embedding_metric_obj,
                optimizer=optimizer_name,
                init=init_finsler,
                n_components=embedding_dim,
                weight=pair_weight,
                print_time=True,
                **optimizer_kwargs,
            )
            if has_n_iter:
                proj_finsler, stress_finsler, n_iter = optimizer_result
                n_iter = int(n_iter)
            else:
                proj_finsler, stress_finsler = optimizer_result
                n_iter = ""
            full_geodesic_stress = None
            plot_title = (
                f"Pancreas {embedding_dim}D {display_name} "
                f"({velocity_formula_tag}, {metric_display_name(embedding_metric_obj)})"
            )
            plot_path = os.path.join(
                dir_res,
                f"{file_prefix}{file_tag}_{embedding_dim_tag}{velocity_formula_tag}_{embedding_metric_tag_value}{pair_weight_tag}.pdf",
            )
            adata_key = f"X_finsler_{optimizer_kind}"
        else:
            raise ValueError(
                "finsler_optimizer must be one of {'smacof', 'gradient_descent', 'finsler_umap', 'path_frozen', 'soft_bf', None}."
            )

        cache_payload = dict(
            embedding=proj_finsler,
            stress=np.asarray(stress_finsler),
            init_finsler_mds=np.asarray(str(init_finsler_mds)),
            **gap_arrays,
            metadata_json=json.dumps(
                {
                    "optimizer": optimizer_kind,
                    "init": str(init_finsler_mds),
                    "embedding_dim": embedding_dim,
                    "seed": seed,
                    "gap": gap,
                    "velocity": velocity,
                    "finsler_metric": normalize_embedding_metric_kind(finsler_metric),
                    "alpha_embedding": alpha_embedding,
                    "randers_alpha_embedding": alpha_embedding if isinstance(embedding_metric_obj, RandersMetric) else None,
                    "cluster_reweight_rho": cluster_reweight_rho,
                    "frontier_pairs_weight": frontier_pairs_weight,
                    "distance_reweighting": distance_reweighting,
                    "selected_frontiers": selected_frontiers,
                    "pair_weighting": pair_weight_info,
                    "geodesic_metric": metric_metadata(embedding_metric_obj),
                    "n_iter": n_iter,
                    "smacof": smacof,
                    "gradient_descent": gradient_descent,
                    "finsler_umap": finsler_umap,
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
        plot_pancreas_finsler_embedding(
            proj_finsler,
            labels=labels,
            title=plot_title,
            save_path=plot_path,
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


def load_or_compute_pancreas_gap_ordering(
        adata_raw,
        cache_path,
        *,
        preprocessing,
        velocity,
        gap,
        seed,
):
    metadata = pancreas_cache_metadata(
        dataset_source=PANCREAS_DATASET_SOURCE,
        preprocessing=preprocessing,
        velocity_mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        gap=gap,
        seed=seed,
    )
    if os.path.exists(cache_path):
        with np.load(cache_path) as cache:
            if "metadata_json" in cache:
                cached_metadata = json.loads(str(cache["metadata_json"].item()))
                if cached_metadata == metadata and "latent_time" in cache:
                    print(f"Loaded pancreas gap latent-time selection cache: {cache_path}")
                    return np.asarray(cache["latent_time"], dtype=float)

    print("Computing full-pancreas latent time for VeloViz-like gap selection")
    adata = adata_raw.copy()
    preprocess_pancreas_for_velocity(adata, preprocessing, seed=seed)
    compute_pancreas_velocity(
        adata,
        mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
    )
    compute_pancreas_velocity_graph(
        adata,
        n_jobs=velocity["graph_n_jobs"],
        show_progress_bar=False,
    )
    import scvelo as scv

    scv.tl.latent_time(adata)
    latent_time = np.asarray(adata.obs["latent_time"], dtype=float)
    np.savez(
        cache_path,
        latent_time=latent_time,
        labels=labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None),
        cell_ids=np.asarray(adata.obs_names, dtype=str),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    print(f"Saved pancreas gap latent-time selection cache: {cache_path}")
    return latent_time


def normalize_velocity_distance_formula(distance_formula):
    if not isinstance(distance_formula, str):
        raise TypeError("velocity['distance_formula'] must be 'exponential', 'randers', or 'matsumoto'.")
    formula = distance_formula.lower()
    aliases = {
        "exp": "exponential",
        "exponential": "exponential",
        "randers": "randers",
        "local_randers": "randers",
        "linear_randers": "randers",
        "mats": "matsumoto",
        "matsumoto": "matsumoto",
        "local_matsumoto": "matsumoto",
    }
    if formula not in aliases:
        raise ValueError("velocity['distance_formula'] must be 'exponential', 'randers', or 'matsumoto'.")
    return aliases[formula]


def velocity_distance_formula_tag(distance_formula, alpha=None):
    formula = normalize_velocity_distance_formula(distance_formula)
    if formula == "exponential":
        prefix = "vexp"
    elif formula == "randers":
        prefix = "vrand"
    elif formula == "matsumoto":
        prefix = "vmats"
    else:
        raise RuntimeError(f"Unhandled velocity distance formula {formula!r}.")
    if alpha is None:
        return prefix
    return f"{prefix}{cache_token(alpha)}"


def make_embedding_metric(config):
    if isinstance(config, str):
        config = {"kind": config}
    if not isinstance(config, dict):
        raise TypeError("finsler_metric must be a dict or metric kind string.")
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


def pancreas_umap_embedding_path(cache_dir, cache_tag, *, n_components, dataset_prefix="pancreas"):
    prefix = pancreas_file_prefix(dataset_prefix)
    if int(n_components) == 2:
        return os.path.join(cache_dir, f"{prefix}umap_{cache_tag}.npy")
    return os.path.join(cache_dir, f"{prefix}umap_{int(n_components)}d_{cache_tag}.npy")


def pancreas_isomap_embedding_path(cache_dir, cache_tag, *, n_components, dataset_prefix="pancreas"):
    prefix = pancreas_file_prefix(dataset_prefix)
    if int(n_components) == 2:
        return os.path.join(cache_dir, f"{prefix}isomap_{cache_tag}.npy")
    return os.path.join(cache_dir, f"{prefix}isomap_{int(n_components)}d_{cache_tag}.npy")


def pancreas_umap_variant_tag(umap):
    """Return an empty tag for the historical default UMAP config."""
    n_neighbors = int(umap["n_neighbors"])
    min_dist = float(umap["min_dist"])
    if n_neighbors == 50 and np.isclose(min_dist, 0.5):
        return ""
    return f"u{n_neighbors}_md{cache_token(min_dist)}_"


def normalize_embedding_dim(embedding_dim):
    embedding_dim = int(embedding_dim)
    if embedding_dim not in {2, 3}:
        raise ValueError("embedding_dim must be 2 or 3.")
    return embedding_dim


def finsler_embedding_dim_tag(embedding_dim):
    return f"{int(embedding_dim)}d_"


def pancreas_file_prefix(dataset_prefix):
    return "" if dataset_prefix == "pancreas" else f"{dataset_prefix}_"


def plot_pancreas_finsler_embedding(embedding, *, labels, title, save_path, random_state):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] == 2:
        plot_categorical_embedding(
            embedding,
            labels=labels,
            title=title,
            xlabel="Finsler 1",
            ylabel="Finsler 2",
            save_path=save_path,
            cmap=PANCREAS_CLUSTER_COLORS,
        )
    elif embedding.shape[1] == 3:
        plot_3d_embedding_views(
            embedding,
            labels=labels,
            title=title,
            save_path=save_path,
            point_fraction=1.0,
            random_state=random_state,
            cmap=PANCREAS_CLUSTER_COLORS,
        )
    else:
        raise ValueError(f"Can only plot 2D or 3D embeddings, got shape {embedding.shape}.")


def smacof_output_tag(*, embedding_dim, metric_tag, init_finsler_mds, n_iter, smacof_version="corrected"):
    init_tag = finsler_init_output_tag(init_finsler_mds)
    version_tag = {"corrected": "new", "legacy": "old"}.get(
        smacof_version,
        smacof_version.replace(" ", "_"),
    )
    return f"s_{version_tag}_{int(embedding_dim)}d_{metric_tag}_{init_tag}_i{int(n_iter)}"


def finsler_init_output_tag(init_finsler_mds):
    try:
        return normalize_finsler_init_kind(init_finsler_mds).replace("_2d", "").replace("_3d", "")
    except ValueError:
        return "custom"


def _deep_update_dicts(overrides, **targets):
    for name, value in overrides.items():
        if name not in targets:
            raise KeyError(f"Unknown main_pancreas override {name!r}.")
        if not isinstance(value, dict):
            raise TypeError(f"Override {name!r} must be a dict.")
        targets[name].update(value)


def compute_and_save_pancreas_umap_embedding(
        output_path,
        *,
        preprocessing,
        velocity,
        umap,
        gap,
        n_components,
        random_state,
):
    # Keep imports local so non-pancreas scripts do not require scanpy/scvelo.
    import scanpy as sc

    setup_scvelo_settings()

    print(f"Loading pancreas dataset from {PANCREAS_DATASET_SOURCE} for {n_components}D UMAP")
    adata = load_pancreas_dataset()
    print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")
    labels_full = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    gap_selection = select_pancreas_gap(labels_full, gap, cell_ids=np.asarray(adata.obs_names, dtype=str))
    if gap_selection.enabled:
        print(
            f"Applying pancreas gap {gap_selection.config['name']!r}: "
            f"removed {gap_selection.removed_mask.sum()} cells."
        )
        adata = adata[gap_selection.keep_mask].copy()

    print("Preprocessing for UMAP")
    preprocess_pancreas_for_velocity(adata, preprocessing, seed=random_state)
    compute_pancreas_velocity(
        adata,
        mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
    )
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
    embedding = orient_pancreas_embedding_by_velocity(
        adata,
        embedding,
        velocity=velocity,
        label=f"{n_components}D UMAP",
    )
    labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    np.save(output_path, embedding)
    return embedding, labels


def compute_and_save_pancreas_isomap_embedding(
        output_path,
        *,
        preprocessing,
        velocity,
        isomap,
        gap,
        n_components,
        random_state,
):
    # Keep imports local so non-pancreas scripts do not require scanpy/scvelo.
    setup_scvelo_settings()

    print(f"Loading pancreas dataset from {PANCREAS_DATASET_SOURCE} for {n_components}D Isomap")
    adata = load_pancreas_dataset()
    print(f"Raw pancreas shape: {adata.n_obs} cells x {adata.n_vars} genes")
    labels_full = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    gap_selection = select_pancreas_gap(labels_full, gap, cell_ids=np.asarray(adata.obs_names, dtype=str))
    if gap_selection.enabled:
        print(
            f"Applying pancreas gap {gap_selection.config['name']!r}: "
            f"removed {gap_selection.removed_mask.sum()} cells."
        )
        adata = adata[gap_selection.keep_mask].copy()

    print("Preprocessing for Isomap")
    preprocess_pancreas_for_velocity(adata, preprocessing, seed=random_state)
    compute_pancreas_velocity(
        adata,
        mode=velocity["mode"],
        recover_dynamics_max_iter=velocity["recover_dynamics_max_iter"],
        recover_dynamics_n_jobs=velocity["recover_dynamics_n_jobs"],
    )
    X_pca = np.asarray(adata.obsm["X_pca"][:, :preprocessing["n_pcs"]], dtype=float)
    embedding = compute_isomap_from_pca(
        X_pca,
        isomap,
        n_components=n_components,
    )
    embedding = np.asarray(embedding, dtype=float)
    embedding = orient_pancreas_embedding_by_velocity(
        adata,
        embedding,
        velocity=velocity,
        label=f"{n_components}D Isomap",
    )
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


def orient_pancreas_umap_3d(embedding):
    """Deprecated compatibility wrapper; use velocity-based orientation."""
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] != 3:
        raise ValueError("orient_pancreas_umap_3d expects a 3D embedding.")
    return embedding


def orient_pancreas_embedding_by_velocity(adata, embedding, *, velocity, label):
    """Rotate an embedding so its mean projected velocity points downward."""
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[1] not in {2, 3}:
        raise ValueError("Only 2D and 3D pancreas embeddings can be velocity-oriented.")

    if "velocity_graph" not in adata.uns:
        print("Computing scVelo velocity graph for embedding orientation")
        compute_pancreas_velocity_graph(
            adata,
            n_neighbors=velocity["velocity_neighbors"],
            n_jobs=velocity["graph_n_jobs"],
            show_progress_bar=False,
        )

    scv = setup_scvelo_settings()
    adata.obsm["X_orientation"] = embedding
    if "velocity_orientation" in adata.obsm:
        del adata.obsm["velocity_orientation"]
    scv.tl.velocity_embedding(adata, basis="orientation")
    velocity_embedding = np.asarray(adata.obsm["velocity_orientation"], dtype=float)
    oriented, _, mean_velocity = utils.rotate_embedding_to_mean_velocity_down(
        embedding,
        velocity_embedding,
    )
    oriented_mean = np.nanmean(velocity_embedding, axis=0) @ utils.rotation_to_down_axis(mean_velocity).T
    print(
        f"Oriented {label}: mean projected velocity "
        f"{np.array2string(mean_velocity, precision=3)} -> "
        f"{np.array2string(oriented_mean, precision=3)}"
    )
    return oriented


def plot_3d_umap(embedding, *, labels, save_path, random_state):
    plot_3d_embedding_views(
        embedding,
        labels=labels,
        title="scVelo pancreas UMAP 3D",
        save_path=save_path,
        point_fraction=1.0,
        random_state=random_state,
        cmap=PANCREAS_CLUSTER_COLORS,
    )
    print(f"Saved 3D UMAP figure: {save_path}")


def requested_embedding_init_dimension(init_finsler_mds, method):
    if init_finsler_mds is None:
        return None
    try:
        init_kind = normalize_finsler_init_kind(init_finsler_mds)
    except ValueError:
        return None
    if init_kind == f"{method}_2d":
        return 2
    if init_kind == f"{method}_3d":
        return 3
    return None


def resolve_finsler_init(
        init_finsler_mds,
        *,
        embedding_sources,
        raw_dir,
        n_samples,
        n_components=3,
        allow_internal_spectral=False,
):
    """Load the initial embedding used by the selected Finsler-MDS optimizer."""
    if init_finsler_mds is None:
        if allow_internal_spectral:
            return "spectral", "internal spectral initialization"
        return None, "random initialization"

    if isinstance(init_finsler_mds, str) and init_finsler_mds.lower() in {"spectral", "spec"}:
        if allow_internal_spectral:
            return "spectral", "internal spectral initialization"
        raise ValueError("init_finsler_mds='spectral' is only supported for finsler_optimizer='finsler_umap'.")

    try:
        init_kind = normalize_finsler_init_kind(init_finsler_mds)
        source_path = embedding_sources.get(init_kind)
        if source_path is None or not os.path.exists(source_path):
            raise FileNotFoundError(
                f"Requested init_finsler_mds={init_kind!r}, but no saved embedding was found."
            )
    except ValueError:
        init_kind = Path(str(init_finsler_mds)).stem
        source_path = resolve_raw_embedding_path(init_finsler_mds, raw_dir)

    embedding = load_saved_embedding(source_path)
    if embedding.shape[0] != n_samples:
        raise ValueError(
            f"Saved {init_kind} init has {embedding.shape[0]} samples, expected {n_samples}: {source_path}"
        )
    if embedding.shape[1] > n_components:
        raise ValueError(
            f"Saved {init_kind} init is {embedding.shape[1]}D but the requested "
            f"embedding_dim is {n_components}. A 3D embedding cannot initialize a 2D run: {source_path}"
        )

    if init_kind in {"umap_2d", "umap_3d", "isomap_2d", "isomap_3d"}:
        init = low_dim_to_finsler_init(embedding, n_components)
        init_label = "UMAP" if init_kind.startswith("umap") else "Isomap"
        return init, f"saved {embedding.shape[1]}D {init_label} ({source_path})"

    if embedding.shape[1] < n_components:
        init = low_dim_to_finsler_init(embedding, n_components)
        return init, f"saved {embedding.shape[1]}D {init_kind} promoted to {n_components}D ({source_path})"
    return embedding, f"saved {init_kind} ({source_path})"


def resolve_raw_embedding_path(init_finsler_mds, raw_dir):
    path = Path(str(init_finsler_mds))
    candidates = [path] if path.is_absolute() else [Path(raw_dir) / path]
    if path.suffix == "":
        candidates.extend(candidate.with_suffix(suffix) for candidate in candidates[:] for suffix in (".npz", ".npy"))
    for candidate in candidates:
        if candidate.exists() and candidate.suffix in {".npz", ".npy"}:
            return candidate
    raise FileNotFoundError(
        f"init_finsler_mds={init_finsler_mds!r} is not a known init keyword and "
        f"was not found as a .npz/.npy file in {raw_dir}."
    )


def normalize_finsler_init_kind(init_finsler_mds):
    if not isinstance(init_finsler_mds, str):
        raise TypeError(
            "init_finsler_mds must be one of "
            "{'umap', 'umap_2D', 'umap_3D', 'isomap', 'isomap_2D', 'isomap_3D', "
            "'smacof', 'gradient_descent', 'finsler_umap', 'path_frozen', 'soft_bf', None}."
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
    if init_kind in {"gradient_descent", "gd"}:
        return "gradient_descent"
    if init_kind in {"finsler_umap", "fumap"}:
        return "finsler_umap"
    if init_kind in {"smacof", "path_frozen", "soft_bf"}:
        return init_kind
    raise ValueError(
        "init_finsler_mds must be one of "
        "{'umap', 'umap_2D', 'umap_3D', 'isomap', 'isomap_2D', 'isomap_3D', "
        "'smacof', 'gradient_descent', 'finsler_umap', 'path_frozen', 'soft_bf', None}."
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


def latest_finsler_embedding_path(cache_dir, dataset_prefix, family, embedding_dim):
    cache_dir = Path(cache_dir)
    prefix = pancreas_file_prefix(dataset_prefix)
    if int(embedding_dim) == 2:
        patterns = [f"{prefix}{family}_2d_*.npz"]
    else:
        patterns = [
            f"{prefix}{family}_3d_*.npz",
            f"{prefix}{family}_2d_*.npz",
        ]
    candidates = []
    for pattern in patterns:
        candidates.extend(cache_dir.glob(pattern))
    if family == "smacof":
        legacy_prefix = f"{dataset_prefix}_" if dataset_prefix else ""
        short_patterns = ["s_new", "s_old"]
        for short_family in short_patterns:
            if int(embedding_dim) == 2:
                candidates.extend(cache_dir.glob(f"{prefix}{short_family}_2d_*.npz"))
            else:
                candidates.extend(cache_dir.glob(f"{prefix}{short_family}_3d_*.npz"))
                candidates.extend(cache_dir.glob(f"{prefix}{short_family}_2d_*.npz"))
        legacy_family = "randers_smacof"
        if int(embedding_dim) == 2:
            candidates.extend(cache_dir.glob(f"{legacy_prefix}{legacy_family}_2d_*.npz"))
        else:
            candidates.extend(cache_dir.glob(f"{legacy_prefix}{legacy_family}_*.npz"))
            candidates.extend(cache_dir.glob(f"{legacy_prefix}{legacy_family}_2d_*.npz"))
    elif family in {"pf", "sbf"}:
        legacy_family = {"pf": "path_frozen", "sbf": "soft_bf"}[family]
        legacy_prefix = f"{dataset_prefix}_" if dataset_prefix else ""
        if int(embedding_dim) == 2:
            candidates.extend(cache_dir.glob(f"{legacy_prefix}{legacy_family}_2d_*.npz"))
        else:
            candidates.extend(cache_dir.glob(f"{legacy_prefix}{legacy_family}_*.npz"))
            candidates.extend(cache_dir.glob(f"{legacy_prefix}{legacy_family}_2d_*.npz"))
    if not candidates:
        return None
    return max(set(candidates), key=lambda path: path.stat().st_mtime)


def normalize_finsler_optimizer(finsler_optimizer):
    if not isinstance(finsler_optimizer, str):
        raise TypeError(
            "finsler_optimizer must be one of {'smacof', 'gradient_descent', 'finsler_umap', 'path_frozen', 'soft_bf', None}."
        )
    optimizer = finsler_optimizer.lower()
    if optimizer in {"smacof", "randers_smacof", "smacof_randers"}:
        return "smacof"
    if optimizer in {"gradient_descent", "gd"}:
        return "gradient_descent"
    if optimizer in {"finsler_umap", "fumap"}:
        return "finsler_umap"
    if optimizer in {"path_frozen", "frozen_paths"}:
        return "path_frozen"
    if optimizer in {"soft_bf", "soft_bellman_ford", "sbf"}:
        return "soft_bf"
    raise ValueError(
        "finsler_optimizer must be one of {'smacof', 'gradient_descent', 'finsler_umap', 'path_frozen', 'soft_bf', None}."
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
        cell_ids = cached_string_array(cache, "cell_ids", default_size=len(x_umap))
        original_indices = cached_int_array(cache, "original_indices", default_size=len(x_umap))
        gap_arrays = {
            "labels": labels_to_cache(labels),
            "cell_ids": cell_ids,
            "original_indices": original_indices,
            "gap_removed_original_indices": cached_int_array(cache, "gap_removed_original_indices"),
            "gap_before_indices": cached_int_array(cache, "gap_before_indices"),
            "gap_after_indices": cached_int_array(cache, "gap_after_indices"),
        }

    return x_umap, dists_velocity, labels, cell_ids, original_indices, gap_arrays


def pancreas_state_cache_path(raw_dir, *, preprocessing, velocity, dataset_prefix="pancreas", seed=42):
    prefix = "" if dataset_prefix == "pancreas" else f"{dataset_prefix}_"
    return os.path.join(
        raw_dir,
        f"{prefix}pancreas_campaign_state_{cache_token(velocity['mode'])}_"
        f"hvg{preprocessing['n_top_genes']}_pca{preprocessing['n_pcs']}_s{seed}.h5ad",
    )


def pancreas_state_cache_metadata(**params):
    """Metadata for the heavy preprocessed/velocity AnnData cache.

    This deliberately excludes distance-only parameters such as velocity alpha,
    cos_clip, kNN support, and UMAP settings. Changing those should only rebuild
    the directed dissimilarity cache.
    """
    return pancreas_cache_metadata(**params)


def load_pancreas_state_cache(cache_path, expected_metadata):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    print(f"Loading cached pancreas velocity state: {cache_path}")

    import scanpy as sc

    adata = sc.read_h5ad(cache_path)
    if not pancreas_state_cache_matches(adata, expected_metadata):
        print("Cached pancreas velocity state was produced with different parameters; recomputing.")
        return None
    validate_pancreas_state_cache(adata, cache_path)

    labels = labels_to_numpy(adata.obs["clusters"] if "clusters" in adata.obs else None)
    cell_ids = np.asarray(adata.obs_names, dtype=str)
    if "finsler_mds_original_index" in adata.obs:
        original_indices = np.asarray(adata.obs["finsler_mds_original_index"], dtype=int)
    else:
        original_indices = np.arange(adata.n_obs, dtype=int)
    gap_arrays = {
        "labels": labels_to_cache(labels),
        "cell_ids": cell_ids,
        "original_indices": original_indices,
        "gap_removed_original_indices": np.asarray(
            adata.uns.get("finsler_mds_gap_removed_original_indices", []),
            dtype=int,
        ),
        "gap_before_indices": np.asarray(
            adata.uns.get("finsler_mds_gap_before_indices", []),
            dtype=int,
        ),
        "gap_after_indices": np.asarray(
            adata.uns.get("finsler_mds_gap_after_indices", []),
            dtype=int,
        ),
    }
    return adata, labels, cell_ids, original_indices, gap_arrays


def pancreas_state_cache_matches(adata, expected_metadata):
    metadata_json = adata.uns.get("finsler_mds_state_metadata_json")
    if metadata_json is None:
        print("Cached pancreas velocity state has no metadata; accepting legacy state cache.")
        return True
    try:
        cached_metadata = json.loads(str(metadata_json))
    except json.JSONDecodeError:
        return False
    return cached_metadata == expected_metadata


def validate_pancreas_state_cache(adata, cache_path):
    missing = []
    if "clusters" not in adata.obs:
        missing.append("obs['clusters']")
    if "X_pca" not in adata.obsm:
        missing.append("obsm['X_pca']")
    if "velocity" not in adata.layers:
        missing.append("layers['velocity']")
    if "PCs" not in adata.varm:
        missing.append("varm['PCs']")
    if missing:
        raise ValueError(f"Invalid pancreas state cache {cache_path}: missing {', '.join(missing)}.")


def save_pancreas_state_cache(
        cache_path,
        adata,
        *,
        labels,
        cell_ids,
        original_indices,
        gap_arrays,
        metadata,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    adata = adata.copy()
    adata.obs_names = np.asarray(cell_ids, dtype=str)
    if labels is not None:
        adata.obs["clusters"] = np.asarray(labels, dtype=str)
    adata.obs["finsler_mds_original_index"] = np.asarray(original_indices, dtype=int)
    adata.uns["finsler_mds_state_metadata_json"] = json.dumps(metadata, sort_keys=True)
    for key, uns_key in [
            ("gap_removed_original_indices", "finsler_mds_gap_removed_original_indices"),
            ("gap_before_indices", "finsler_mds_gap_before_indices"),
            ("gap_after_indices", "finsler_mds_gap_after_indices"),
    ]:
        adata.uns[uns_key] = np.asarray(gap_arrays.get(key, []), dtype=int)
    adata.write_h5ad(cache_path)
    print(f"Saved pancreas velocity state cache: {cache_path}")


def velocity_cache_metadata_matches(cached_metadata, expected_metadata):
    if cached_metadata == expected_metadata:
        return True
    if not expected_metadata.get("gap", {}).get("enabled", False) and "gap" not in cached_metadata:
        legacy_expected = dict(expected_metadata)
        legacy_expected.pop("gap", None)
        if cached_metadata == legacy_expected:
            return True
    # Caches created before the Finsler-neighbor refactor used
    # ``velocity_graph_neighbors`` for the Euclidean support and had no
    # additional Finsler-neighbor edges.
    if (
            expected_metadata.get("velocity_kNN_finsler", 0) == 0
            and "velocity_graph_neighbors" in cached_metadata
            and "velocity_kNN_euclid" not in cached_metadata
    ):
        legacy_expected = dict(expected_metadata)
        legacy_expected["velocity_graph_neighbors"] = legacy_expected.pop("velocity_kNN_euclid")
        legacy_expected.pop("velocity_kNN_finsler", None)
        if not legacy_expected.get("gap", {}).get("enabled", False):
            legacy_expected.pop("gap", None)
        if cached_metadata == legacy_expected:
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
        key: json_safe_metadata_value(value)
        for key, value in params.items()
    }


def json_safe_metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_metadata_value(item) for item in value]
    return value


def cache_token(value):
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return str(value).replace(os.sep, "-").replace(".", "p")


def cluster_reweight_output_tag(rho):
    rho = float(rho)
    if rho <= 0:
        return ""
    return f"_crw{cache_token(rho)}"


def normalize_selected_frontiers(selected_frontiers):
    selected_frontiers = str(selected_frontiers).lower()
    if selected_frontiers not in {"cbdir", "all"}:
        raise ValueError("selected_frontiers must be one of {'cbdir', 'all'}.")
    return selected_frontiers


def frontier_pairs_weight_output_tag(weight, *, selected_frontiers="cbdir"):
    weight = float(weight)
    if weight == 1.0:
        return ""
    selected_frontiers = normalize_selected_frontiers(selected_frontiers)
    if selected_frontiers == "all":
        return f"_afpw{cache_token(weight)}"
    return f"_fpw{cache_token(weight)}"


def distance_reweight_output_tag(distance_reweighting):
    if not distance_reweighting:
        return ""
    power = float(distance_reweighting.get("power", 0.0))
    if power <= 0:
        return ""
    epsilon = float(distance_reweighting.get("epsilon", 1e-6))
    tag = f"_drw{cache_token(power)}"
    if epsilon != 1e-6:
        tag += f"e{cache_token(epsilon)}"
    return tag


def pair_weight_output_tag(
        cluster_reweight_rho,
        frontier_pairs_weight,
        *,
        distance_reweighting=None,
        selected_frontiers="cbdir",
):
    return (
        cluster_reweight_output_tag(cluster_reweight_rho)
        + distance_reweight_output_tag(distance_reweighting)
        + frontier_pairs_weight_output_tag(frontier_pairs_weight, selected_frontiers=selected_frontiers)
    )


def make_pancreas_pair_weight(
        labels,
        dissimilarities,
        cluster_reweight_rho,
        *,
        frontier_pairs_weight=1.0,
        selected_frontiers="cbdir",
        distance_reweighting=None,
        boundary_plan=None,
):
    rho = float(cluster_reweight_rho)
    frontier_pairs_weight = float(frontier_pairs_weight)
    distance_reweighting = dict(distance_reweighting or {})
    distance_power = float(distance_reweighting.get("power", 0.0))
    selected_frontiers = normalize_selected_frontiers(selected_frontiers)
    if rho <= 0 and frontier_pairs_weight == 1.0 and distance_power <= 0:
        return None, {
            "cluster_reweighting": {"rho": rho, "enabled": False},
            "distance_reweighting": {"power": distance_power, "enabled": False},
            "frontier_pair_weighting": {
                "factor": frontier_pairs_weight,
                "selected_frontiers": selected_frontiers,
                "enabled": False,
            },
        }
    if labels is None and (rho > 0 or frontier_pairs_weight != 1.0):
        raise ValueError(
            "pair weighting requires cluster labels in the velocity cache. "
            "Recompute the pancreas cache if it was created before labels were saved."
        )

    if rho > 0:
        weights, cluster_info = cluster_balanced_pair_weights(
            labels,
            dissimilarities,
            rho=rho,
            return_info=True,
        )
        cluster_info = dict(cluster_info)
        cluster_info["enabled"] = True
        print(
            "Cluster pair reweighting: "
            f"rho={rho:g}, n_ref={cluster_info['n_ref']:.6g}, "
            f"mean={cluster_info['mean_active_weight']:.6g}, "
            f"range=[{cluster_info['min_active_weight']:.6g}, {cluster_info['max_active_weight']:.6g}]"
        )
    else:
        weights = np.ones_like(dissimilarities, dtype=float)
        np.fill_diagonal(weights, 0.0)
        cluster_info = {"rho": rho, "enabled": False}

    if distance_power > 0:
        weights, distance_info = apply_distance_pair_reweight(
            weights,
            dissimilarities,
            power=distance_power,
            epsilon=distance_reweighting.get("epsilon", 1e-6),
            return_info=True,
        )
        print(
            "Distance pair reweighting: "
            f"power={distance_info['power']:g}, epsilon={distance_info['epsilon']:.6g}, "
            f"median_delta={distance_info['median_delta']:.6g}, "
            f"factor_range=[{distance_info['min_factor']:.6g}, {distance_info['max_factor']:.6g}]"
        )
    else:
        distance_info = {"power": distance_power, "epsilon": float(distance_reweighting.get("epsilon", 1e-6)), "enabled": False}

    if frontier_pairs_weight != 1.0:
        if boundary_plan is None:
            raise ValueError("frontier_pairs_weight != 1 requires a CBDir boundary plan.")
        weights, frontier_info = apply_frontier_pair_weight(
            weights,
            boundary_plan,
            factor=frontier_pairs_weight,
            dissimilarities=dissimilarities,
            symmetric=True,
            return_info=True,
        )
        print(
            "Frontier pair weighting: "
            f"selected={selected_frontiers}, factor={frontier_pairs_weight:g}, "
            f"directed_pairs={frontier_info['n_directed_pairs']}, "
            f"weighted_pairs={frontier_info['n_weighted_pairs']}"
        )
    else:
        frontier_info = {"factor": frontier_pairs_weight, "selected_frontiers": selected_frontiers, "enabled": False}

    weights = normalize_pair_weights(weights, dissimilarities)[0]

    return weights, {
        "cluster_reweighting": cluster_info,
        "distance_reweighting": distance_info,
        "frontier_pair_weighting": {**frontier_info, "selected_frontiers": selected_frontiers},
    }


def load_pancreas_frontier_plan(labels, *, selected_frontiers, adata, preprocessing, seed, eval_raw_dir):
    selected_frontiers = normalize_selected_frontiers(selected_frontiers)
    if selected_frontiers == "cbdir":
        return load_pancreas_cbdir_boundary_plan(
            labels,
            adata=adata,
            preprocessing=preprocessing,
            seed=seed,
            eval_raw_dir=eval_raw_dir,
        )
    return load_pancreas_all_frontier_plan(
        labels,
        adata=adata,
        preprocessing=preprocessing,
        seed=seed,
        eval_raw_dir=eval_raw_dir,
    )


def load_pancreas_cbdir_boundary_plan(labels, *, adata, preprocessing, seed, eval_raw_dir):
    eval_raw_dir = Path(eval_raw_dir)
    eval_raw_dir.mkdir(parents=True, exist_ok=True)
    n_neighbors = 30
    path = eval_raw_dir / (
        f"cbdir_boundaries_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"hvg{preprocessing['n_top_genes']}_pca{preprocessing['n_pcs']}_"
        f"k{n_neighbors}_s{seed}.npz"
    )
    neighbor_indices = None
    if adata is not None and "distances" in adata.obsp:
        neighbor_indices = neighbor_indices_from_sparse_distances(
            adata.obsp["distances"],
            n_neighbors=n_neighbors,
        )
    if neighbor_indices is None and not path.exists():
        raise FileNotFoundError(
            f"CBDir frontier-pair cache not found and cannot be rebuilt without AnnData neighbors: {path}"
        )
    return load_or_compute_boundary_neighbor_plan(
        path,
        labels,
        cluster_edges=PANCREAS_TRANSITIONS,
        neighbor_indices=neighbor_indices,
        n_neighbors=n_neighbors,
        on_mismatch="recompute" if neighbor_indices is not None else "error",
    )


def load_pancreas_all_frontier_plan(labels, *, adata, preprocessing, seed, eval_raw_dir):
    eval_raw_dir = Path(eval_raw_dir)
    eval_raw_dir.mkdir(parents=True, exist_ok=True)
    n_neighbors = 30
    path = eval_raw_dir / (
        f"all_frontiers_{PANCREAS_DATASET_SOURCE.replace('.', '_')}_"
        f"hvg{preprocessing['n_top_genes']}_pca{preprocessing['n_pcs']}_"
        f"k{n_neighbors}_s{seed}.npz"
    )
    neighbor_indices = None
    if adata is not None and "distances" in adata.obsp:
        neighbor_indices = neighbor_indices_from_sparse_distances(
            adata.obsp["distances"],
            n_neighbors=n_neighbors,
        )
    if neighbor_indices is None and not path.exists():
        raise FileNotFoundError(
            f"All-frontier pair cache not found and cannot be rebuilt without AnnData neighbors: {path}"
        )
    labels_array = np.asarray(labels, dtype=str)
    cluster_edges = [
        (source, target)
        for source in np.unique(labels_array)
        for target in np.unique(labels_array)
        if source != target
    ]
    return load_or_compute_boundary_neighbor_plan(
        path,
        labels,
        cluster_edges=cluster_edges,
        neighbor_indices=neighbor_indices,
        n_neighbors=n_neighbors,
        on_mismatch="recompute" if neighbor_indices is not None else "error",
    )


def neighbor_indices_from_sparse_distances(distances, *, n_neighbors):
    distances = sparse.csr_matrix(distances)
    neighbors = np.full((distances.shape[0], n_neighbors), -1, dtype=int)
    for row_idx in range(distances.shape[0]):
        start, end = distances.indptr[row_idx], distances.indptr[row_idx + 1]
        cols = distances.indices[start:end]
        data = distances.data[start:end]
        keep = cols != row_idx
        cols = cols[keep]
        data = data[keep]
        if len(cols) == 0:
            continue
        order = np.argsort(data, kind="stable")[:n_neighbors]
        neighbors[row_idx, :len(order)] = cols[order]
    return neighbors


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


def cached_string_array(cache, key, *, default_size=None):
    if key in cache:
        return np.asarray(cache[key], dtype=str)
    if default_size is None:
        return np.asarray([], dtype=str)
    return np.arange(default_size).astype(str)


def cached_int_array(cache, key, *, default_size=None):
    if key in cache:
        return np.asarray(cache[key], dtype=int)
    if default_size is None:
        return np.asarray([], dtype=int)
    return np.arange(default_size, dtype=int)


if __name__ == "__main__":
    main_pancreas()
