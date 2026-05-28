"""Visualize Sea initializations used by ``main_sea``."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from main_sea import classical_mds_initialization, plot_embedding
from sea_datasets import current_map_distances, make_sea_dataset, normalize_sea_dataset_name
from finsler_mds import utils


def main_sea_inits():
    seed = 0
    dataset_name = normalize_sea_dataset_name("sea2")
    alpha_current = 0.8
    n_samples = 2000
    n_components = 3
    n_neighbors = 10

    dir_res = SCRIPT_DIR / "res" / dataset_name
    dir_fig = dir_res / "figures"
    dir_fig.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dataset = make_sea_dataset(
        dataset_name,
        n_samples=n_samples,
        alpha_current=alpha_current,
        rng=rng,
        graph_neighbors=n_neighbors,
    )
    print(f"Dataset: {dataset.key} ({dataset.title})")

    print("Computing Isomap initialization")
    isomap = utils.IsomapWithPreds(
        n_components=n_components,
        n_neighbors=n_neighbors,
    ).fit_transform(dataset.X)
    plot_embedding(
        isomap,
        dataset.X,
        title=f"{dataset.title}, Isomap init ({n_components}D)",
        save_path=dir_fig / "init_isomap.pdf",
    )
    print(f"Saved: {dir_fig / 'init_isomap.pdf'}")

    print("Computing target distances for target-MDS initialization")
    target_distances, _ = current_map_distances(
        dataset,
        n_neighbors=n_neighbors,
        path_method="auto",
    )
    print("Computing target-MDS initialization")
    target_mds = classical_mds_initialization(
        target_distances,
        n_components=n_components,
    )
    plot_embedding(
        target_mds,
        dataset.X,
        title=f"{dataset.title}, target-MDS init ({n_components}D)",
        save_path=dir_fig / "init_target_mds.pdf",
    )
    print(f"Saved: {dir_fig / 'init_target_mds.pdf'}")

    print("Coordinate spreads:")
    print(f"  Isomap std:     {np.std(isomap, axis=0)}")
    print(f"  target-MDS std: {np.std(target_mds, axis=0)}")


if __name__ == "__main__":
    main_sea_inits()
