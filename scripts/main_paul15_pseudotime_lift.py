"""Lift saved 2D Paul15 embeddings into 3D using pseudotime as altitude."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from finsler_mds.utils import set_axes_equal
from finsler_mds.utils.embedding_io import load_npz_inputs as load_inputs, load_saved_embedding
from finsler_mds.utils.paul15 import COMBINED_LINEAGE_DPT_KEY


def main_paul15_pseudotime_lift():
    script_dir = Path(__file__).resolve().parent
    dir_res = script_dir / "res" / "paul15"
    dir_raw = dir_res / "raw"
    dir_fig = dir_res / "symmetric" / "figures" / "pseudotime_lift"
    dir_embeddings = dir_res / "symmetric" / "embeddings"

    pseudotime_key = COMBINED_LINEAGE_DPT_KEY
    fallback_pseudotime_key = "dpt_pseudotime_finite"
    z_at_t0 = 0.0
    z_at_t1 = -0.2

    dir_fig.mkdir(parents=True, exist_ok=True)

    inputs_path = latest_symmetric_inputs(dir_raw)
    inputs = load_inputs(inputs_path)
    cell_ids = inputs["cell_ids"]
    pseudotime, fallback_count = pseudotime_with_fallback(
        inputs,
        key=pseudotime_key,
        fallback_key=fallback_pseudotime_key,
    )
    print(f"Loaded Paul15 inputs: {inputs_path}")
    print(
        f"Using {pseudotime_key!r} for altitude "
        f"with {fallback_count} values filled from {fallback_pseudotime_key!r}."
    )

    methods = [
        (
            "umap",
            "UMAP init",
            dir_embeddings / "umap.npy",
        ),
        (
            "smacof",
            "SMACOF Randers alpha=0",
            dir_embeddings / "smacof.npz",
        ),
        (
            "pf",
            "Path-frozen Randers alpha=0",
            dir_embeddings / "pf.npz",
        ),
    ]

    lifted_embeddings = []
    for key, title, path in methods:
        if not path.exists():
            print(f"Skipping {key}: missing embedding {path}")
            continue
        embedding_2d = load_saved_embedding(path, cell_ids=cell_ids)
        if embedding_2d.shape[1] != 2:
            raise ValueError(f"Expected a 2D embedding for {key}, got shape {embedding_2d.shape}: {path}")

        lifted = lift_by_pseudotime(
            embedding_2d,
            pseudotime,
            z_at_t0=z_at_t0,
            z_at_t1=z_at_t1,
        )
        lifted_embeddings.append((key, title, lifted))
        save_lifted_views(
            lifted,
            pseudotime,
            title=f"{title}: pseudotime altitude",
            save_path=dir_fig / f"{key}_pt_altitude.pdf",
        )
        print(f"Saved {key} pseudotime-altitude views.")

    print(f"Saved lifted Paul15 visualizations in: {dir_fig}")


def latest_symmetric_inputs(dir_raw):
    candidates = sorted(
        path
        for path in dir_raw.glob("inputs_*_sym_s42.npz")
        if "_lin_" not in path.name
    )
    if not candidates:
        candidates = sorted(dir_raw.glob("inputs_k35_dg0_sym_s42.npz"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find cached symmetric Paul15 inputs. "
            "Run scripts/main_paul15_finsler.py once first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def pseudotime_with_fallback(inputs, *, key, fallback_key):
    if key not in inputs:
        raise KeyError(f"Paul15 inputs have no pseudotime key {key!r}.")

    pseudotime = np.asarray(inputs[key], dtype=float).copy()
    missing = ~np.isfinite(pseudotime)
    fallback_count = 0
    if np.any(missing) and fallback_key is not None:
        if fallback_key not in inputs:
            raise KeyError(f"Paul15 inputs have no fallback pseudotime key {fallback_key!r}.")
        fallback = np.asarray(inputs[fallback_key], dtype=float)
        fill = missing & np.isfinite(fallback)
        pseudotime[fill] = fallback[fill]
        fallback_count = int(np.sum(fill))

    finite = np.isfinite(pseudotime)
    if not np.any(finite):
        raise ValueError("No finite pseudotime values are available for altitude.")
    pseudotime[~finite] = 0.0
    return np.clip(pseudotime, 0.0, 1.0), fallback_count


def lift_by_pseudotime(embedding_2d, pseudotime, *, z_at_t0, z_at_t1):
    embedding_2d = np.asarray(embedding_2d, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float)
    z = z_at_t0 + (z_at_t1 - z_at_t0) * pseudotime
    return np.column_stack([embedding_2d, z])


def save_lifted_views(embedding, pseudotime, *, title, save_path):
    views = [
        ("oblique", 24, -58),
        ("side", 8, 0),
        ("profile", 0, 90),
        ("top", 90, -90),
    ]
    fig = plt.figure(figsize=(11, 9))
    mappable = None
    for view_id, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_id + 1, projection="3d")
        mappable = scatter_pseudotime_3d(ax, embedding, pseudotime)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name)
        format_3d_axis(ax, axis_labels=True)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.04, top=0.92, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.92, 0.2, 0.018, 0.6])
    fig.colorbar(mappable, cax=cax, label="pseudotime")
    fig.savefig(save_path)
    plt.close(fig)


def save_lifted_comparison(lifted_embeddings, pseudotime, *, save_path):
    views = [("oblique", 24, -58), ("side", 8, 0)]
    fig = plt.figure(figsize=(5.0 * len(lifted_embeddings), 8.2))
    mappable = None
    for row, (_, elev, azim) in enumerate(views):
        for col, (_key, title, embedding) in enumerate(lifted_embeddings):
            ax = fig.add_subplot(len(views), len(lifted_embeddings), row * len(lifted_embeddings) + col + 1, projection="3d")
            mappable = scatter_pseudotime_3d(ax, embedding, pseudotime)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title} ({views[row][0]})")
            format_3d_axis(ax, axis_labels=False)
    fig.subplots_adjust(left=0.02, right=0.9, bottom=0.03, top=0.96, wspace=0.0, hspace=0.18)
    cax = fig.add_axes([0.92, 0.22, 0.018, 0.56])
    fig.colorbar(mappable, cax=cax, label="pseudotime")
    fig.savefig(save_path)
    plt.close(fig)


def scatter_pseudotime_3d(ax, embedding, pseudotime):
    return ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=pseudotime,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=6,
        lw=0,
    )


def format_3d_axis(ax, *, axis_labels):
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    if axis_labels:
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        ax.set_zlabel("z = -0.2 t")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("z")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


if __name__ == "__main__":
    main_paul15_pseudotime_lift()
