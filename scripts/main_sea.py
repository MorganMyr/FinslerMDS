"""Generate Sea dataset and fit a direct or geodesic Finsler-MDS embedding."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import fit_finsler_mds, utils  # noqa: E402
from finsler_mds.utils.sea import (  # noqa: E402
    load_or_compute_sea_inputs,
    make_metric,
    metric_tag,
    normalize_metric_name,
    normalize_optimizer,
    normalized_x,
)


CONFIG = {
    "seed": 0,
    "n_samples": 2000,
    "n_neighbors": 10,
    "data_metric": "randers",  # "randers", "matsumoto", "convexified_matsumoto"
    "alpha_current": 0.8,
    "optimizer": "path_frozen",  # "smacof", "gradient_descent", "path_frozen"
    "embedding_metric": "matsumoto",  # same choices as data_metric
    "alpha_embedding": 0.2,
    "init": "isomap",  # "isomap", "latest", or an NPZ path
    "result_suffix": "",
    "sea": {"length": 10.0, "width": 10.0, "current_frequency": 2.0},
    "smacof": {
        "max_iter": 100,
        "project_on_V": True,
        "check_monotony": False,
        "device": "auto",  # "cpu", "auto", "gpu", or "cuda"
        "verbose": 1,
    },
    "gradient_descent": {
        "max_iter": 350,
        "eps": 1e-6,
        "device": "auto",  # "cpu", "auto", "gpu", or "cuda"
        "verbose": 0,
    },
    "path_frozen": {
        "graph_neighbors": 12,
        "outer_iter": 20,
        "inner_iter": 50,
        "n_landmark": 50,
        "random_landmark_fraction": 0.5,
        "n_local_pairs": 12,
        "targets_per_landmark": 100,
        "local_weight": 1.0,
        "direct_stress_weight": 0.1,
        "outer_step_size": 1,
        "device": "auto",  # "cpu", "auto", "gpu", or "cuda"
        "verbose": 1,
    },
}


def main_sea(config_overrides=None):
    config = merged_config(config_overrides)
    optimizer = normalize_optimizer(config["optimizer"])
    embedding_metric_name = normalize_metric_name(config["embedding_metric"])
    if optimizer == "smacof" and embedding_metric_name != "randers":
        raise ValueError("SMACOF is specific to a Randers embedding metric.")

    sea_dir = Path(__file__).parent / "res" / "sea"
    raw_dir = sea_dir / "raw"
    embedding_dir = sea_dir / "embeddings"
    figure_dir = sea_dir / "figures"
    for directory in (raw_dir, embedding_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sea = config["sea"]
    inputs = load_or_compute_sea_inputs(
        raw_dir,
        n_samples=config["n_samples"],
        seed=config["seed"],
        n_neighbors=config["n_neighbors"],
        data_metric=config["data_metric"],
        alpha_current=config["alpha_current"],
        sea_length=sea["length"],
        sea_width=sea["width"],
        current_frequency=sea["current_frequency"],
    )
    init, init_description = resolve_init(
        config["init"],
        inputs.X,
        embedding_dir=embedding_dir,
        optimizer=optimizer,
        data_metric=config["data_metric"],
        alpha_current=config["alpha_current"],
        embedding_metric=embedding_metric_name,
        alpha_embedding=config["alpha_embedding"],
        n_neighbors=config["n_neighbors"],
    )
    metric = make_metric(embedding_metric_name, config["alpha_embedding"])
    optimizer_name, options = optimizer_spec(optimizer, config)
    print(
        f"Running {optimizer} from {init_description}: "
        f"data={metric_tag(config['data_metric'], config['alpha_current'])}, "
        f"embedding={metric_tag(embedding_metric_name, config['alpha_embedding'])}"
    )
    embedding, objective = fit_finsler_mds(
        inputs.dissimilarities,
        optimizer=optimizer_name,
        metric=metric,
        init=init,
        n_components=3,
        random_state=config["seed"],
        print_time=True,
        **options,
    )

    stem = result_stem(config, optimizer, embedding_metric_name)
    output_path = embedding_dir / f"{stem}.npz"
    np.savez(
        output_path,
        embedding=embedding,
        objective=np.asarray(objective, dtype=float),
        input_cache=np.asarray(inputs.path.name),
        optimizer=np.asarray(optimizer),
        embedding_metric=np.asarray(embedding_metric_name),
        alpha_embedding=np.asarray(config["alpha_embedding"], dtype=float),
        init=np.asarray(str(config["init"])),
    )
    plot_current_map(
        inputs.X,
        inputs.current_field,
        path=figure_dir
        / f"sea_{metric_tag(config['data_metric'], config['alpha_current'])}_current.pdf",
    )
    plot_embedding(
        embedding,
        inputs.X,
        title=stem,
        path=figure_dir / f"{stem}.pdf",
    )
    print(f"Objective: {objective}")
    print(f"Saved embedding: {output_path}")
    return embedding, objective


def resolve_init(
        value,
        X,
        *,
        embedding_dir,
        optimizer,
        data_metric,
        alpha_current,
        embedding_metric,
        alpha_embedding,
        n_neighbors,
):
    normalized = str(value).lower().replace("-", "_")
    if normalized == "isomap":
        init = utils.IsomapWithPreds(
            n_components=3, n_neighbors=n_neighbors
        ).fit_transform(X)
        return init, "Isomap"
    if normalized in {"latest", "latest_same", "saved"}:
        prefix = result_prefix(
            optimizer,
            data_metric,
            alpha_current,
            embedding_metric,
            alpha_embedding,
        )
        candidates = sorted(
            embedding_dir.glob(f"{prefix}*.npz"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No saved Sea embedding matching {prefix}*.npz.")
        path = candidates[0]
    else:
        path = Path(str(value))
        if path.suffix == "":
            path = path.with_suffix(".npz")
        if not path.is_absolute() and not path.exists():
            path = embedding_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        init = np.asarray(saved["embedding"], dtype=float)
    if init.shape[0] != len(X):
        raise ValueError(f"Saved init has {len(init)} points; expected {len(X)}: {path}")
    return promote_embedding(init, 3), str(path)


def optimizer_spec(optimizer, config):
    if optimizer == "smacof":
        return "smacof_randers", {"n_init": 1, "n_jobs": 1, **config["smacof"]}
    return optimizer, dict(config[optimizer])


def result_prefix(
        optimizer,
        data_metric,
        alpha_current,
        embedding_metric,
        alpha_embedding,
):
    optimizer_tag = {"smacof": "smacof", "gradient_descent": "gd", "path_frozen": "pf"}[
        normalize_optimizer(optimizer)
    ]
    return (
        f"sea_{metric_tag(data_metric, alpha_current)}_{optimizer_tag}_"
        f"{metric_tag(embedding_metric, alpha_embedding)}"
    )


def result_stem(config, optimizer, embedding_metric):
    stem = result_prefix(
        optimizer,
        config["data_metric"],
        config["alpha_current"],
        embedding_metric,
        config["alpha_embedding"],
    )
    suffix = str(config.get("result_suffix", "")).strip().strip("_")
    return stem if not suffix else f"{stem}_{suffix}"


def promote_embedding(embedding, n_components):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] > n_components:
        raise ValueError(f"Initialization cannot be promoted from shape {embedding.shape} to {n_components}D.")
    if embedding.shape[1] == n_components:
        return embedding
    return np.column_stack((embedding, np.zeros((len(embedding), n_components - embedding.shape[1]))))


def plot_current_map(X, current_field, *, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = normalized_x(X)
    ax.scatter(X[:, 0], X[:, 1], c=colors, cmap="jet", s=25, lw=0)
    ax.quiver(
        X[:, 0], X[:, 1], current_field[:, 0], current_field[:, 1],
        color="black", angles="xy", scale_units="xy", scale=1.4,
        width=0.004, alpha=0.75,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_embedding(embedding, X, *, title, path):
    if embedding.shape[1] != 3:
        raise ValueError("Sea embeddings must be 3D.")
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=plt.get_cmap("jet")(normalized_x(X)),
        s=14,
        lw=0,
    )
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    utils.set_axes_equal(ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def merged_config(overrides):
    config = deepcopy(CONFIG)
    if overrides:
        deep_update(config, overrides)
    return config


def deep_update(target, updates):
    for key, value in updates.items():
        if key not in target:
            raise KeyError(f"Unknown main_sea option: {key!r}.")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Override {key!r} must be a dictionary.")
            deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    main_sea()
