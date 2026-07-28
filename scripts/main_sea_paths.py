"""Compare Sea graph paths on a saved embedding.

Example:
    python scripts/main_sea_paths.py sea_rand0p8_pf_mats0p7.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds import utils  # noqa: E402
from finsler_mds.utils.sea import (  # noqa: E402
    load_sea_inputs,
    make_metric,
    metric_tag,
    normalize_optimizer,
    normalized_x,
)


def main_sea_paths():
    args = parse_args()
    sea_dir = Path(__file__).parent / "res" / "sea"
    path = resolve_embedding_path(args.embedding, sea_dir / "embeddings")
    payload = load_payload(path)
    embedding = np.asarray(payload["embedding"], dtype=float)
    inputs = load_reference_inputs(payload, sea_dir / "raw")
    if len(embedding) != len(inputs["X"]):
        raise ValueError(
            f"Embedding has {len(embedding)} points, but its Sea inputs have "
            f"{len(inputs['X'])}."
        )

    source, target = select_endpoints(
        inputs["X"],
        direction=args.direction,
        source=args.source,
        target=args.target,
    )
    reference_path = utils.path_from_predecessors(
        inputs["predecessors"], source, target
    )
    if reference_path is None:
        raise ValueError("The target is unreachable in the saved Sea graph.")

    optimizer = normalize_optimizer(scalar(payload, "optimizer"))
    metric_name = scalar(payload, "embedding_metric", fallback="metric")
    alpha_embedding = float(scalar(payload, "alpha_embedding"))
    if optimizer == "path_frozen":
        method_path = utils.geodesic_path_indices(
            embedding,
            source,
            target,
            make_metric(metric_name, alpha_embedding),
            n_neighbors=args.embedding_neighbors,
            directed=True,
        )
        if method_path is None:
            raise ValueError("The target is unreachable in the embedding k-NN graph.")
        method_label = "embedding graph shortest path"
    else:
        method_path = [source, target]
        method_label = "direct chord (MDS has no path model)"

    figure_dir = sea_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = endpoint_suffix(args.direction, args.source, args.target)
    current_path = figure_dir / f"{path.stem}_paths_{suffix}_current.pdf"
    embedding_path = figure_dir / f"{path.stem}_paths_{suffix}.pdf"
    show_method_path = not args.hide_method_path
    plot_current_paths(
        inputs["X"],
        inputs["current_field"],
        reference_path,
        method_path,
        method_label=method_label,
        show_method_path=show_method_path,
        title=f"Sea paths — {inputs['data_label']}",
        path=current_path,
    )
    plot_embedding_paths(
        embedding,
        inputs["X"],
        reference_path,
        method_path,
        method_label=method_label,
        show_method_path=show_method_path,
        title=f"{path.stem}: source-target paths",
        path=embedding_path,
    )
    print(f"Source={source}, target={target}")
    print(f"Input graph path: {len(reference_path)} vertices")
    if show_method_path:
        print(f"{method_label}: {len(method_path)} vertices")
    print(f"Saved: {current_path}")
    print(f"Saved: {embedding_path}")


def load_reference_inputs(payload, raw_dir):
    if "input_cache" in payload:
        cache_value = Path(str(np.asarray(payload["input_cache"]).item()))
        cache_path = cache_value if cache_value.is_absolute() else Path(raw_dir) / cache_value.name
        inputs = load_sea_inputs(cache_path)
        metadata = inputs.metadata
        return {
            "X": inputs.X,
            "randers_field": inputs.randers_field,
            "current_field": inputs.current_field,
            "predecessors": inputs.predecessors,
            "data_label": metric_tag(metadata["data_metric"], metadata["alpha_current"]),
        }

    required = {"X", "randers_field", "current_field", "target_predecessors"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Embedding is missing its Sea input reference: {', '.join(missing)}.")
    alpha = float(scalar(payload, "alpha_current"))
    return {
        "X": np.asarray(payload["X"], dtype=float),
        "randers_field": np.asarray(payload["randers_field"], dtype=float),
        "current_field": np.asarray(payload["current_field"], dtype=float),
        "predecessors": np.asarray(payload["target_predecessors"], dtype=int),
        "data_label": metric_tag("randers", alpha),
    }


def resolve_embedding_path(value, directory):
    path = Path(value)
    if path.suffix == "":
        path = path.with_suffix(".npz")
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(directory) / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path)


def load_payload(path):
    with np.load(path, allow_pickle=False) as data:
        if "embedding" not in data:
            raise KeyError(f"Saved file has no 'embedding' array: {path}")
        return {key: np.asarray(data[key]) for key in data.files}


def scalar(payload, key, fallback=None):
    if key not in payload and fallback is not None:
        key = fallback
    if key not in payload:
        raise KeyError(f"Saved embedding has no {key!r} value.")
    return np.asarray(payload[key]).item()


def select_endpoints(X, *, direction, source, target):
    if (source is None) != (target is None):
        raise ValueError("--source and --target must be provided together.")
    if source is not None:
        source, target = int(source), int(target)
        if not (0 <= source < len(X) and 0 <= target < len(X)) or source == target:
            raise ValueError("Source and target must be distinct valid point indices.")
        return source, target

    x = normalized_x(X)
    y = (X[:, 1] - X[:, 1].min()) / max(float(np.ptp(X[:, 1])), 1e-12)
    top_left = int(np.argmin(x + 1.0 - y))
    bottom_right = int(np.argmin(1.0 - x + y))
    if normalize_direction(direction) == "top_left_to_bottom_right":
        return top_left, bottom_right
    return bottom_right, top_left


def normalize_direction(value):
    key = str(value).lower().replace("-", "_")
    aliases = {
        "top_left_to_bottom_right": "top_left_to_bottom_right",
        "tl_br": "top_left_to_bottom_right",
        "forward": "top_left_to_bottom_right",
        "bottom_right_to_top_left": "bottom_right_to_top_left",
        "br_tl": "bottom_right_to_top_left",
        "reverse": "bottom_right_to_top_left",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("direction must be 'tl_br' or 'br_tl'.") from exc


def endpoint_suffix(direction, source, target):
    if source is not None:
        return f"{int(source)}_{int(target)}"
    return "tl_br" if normalize_direction(direction) == "top_left_to_bottom_right" else "br_tl"


def plot_current_paths(
        X,
        current_field,
        reference_path,
        method_path,
        *,
        method_label,
        show_method_path,
        title,
        path,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], c=normalized_x(X), cmap="jet", s=18, lw=0, alpha=0.8)
    ax.quiver(
        X[:, 0], X[:, 1], current_field[:, 0], current_field[:, 1],
        color="0.25", angles="xy", scale_units="xy", scale=1.4,
        width=0.003, alpha=0.6,
    )
    add_paths(
        ax, X, reference_path, method_path,
        method_label=method_label, show_method_path=show_method_path,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_embedding_paths(
        embedding,
        X,
        reference_path,
        method_path,
        *,
        method_label,
        show_method_path,
        title,
        path,
):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] not in {2, 3}:
        raise ValueError("Sea path plots require a 2D or 3D embedding.")
    if embedding.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            embedding[:, 0], embedding[:, 1], c=normalized_x(X), cmap="jet",
            s=18, lw=0, alpha=0.75,
        )
        add_paths(
            ax, embedding, reference_path, method_path,
            method_label=method_label, show_method_path=show_method_path,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.legend(frameon=False)
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=plt.get_cmap("jet")(normalized_x(X)), s=10, lw=0, alpha=0.7,
        )
        add_paths(
            ax, embedding, reference_path, method_path,
            method_label=method_label,
            show_method_path=show_method_path,
        )
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect((1, 1, 1))
        utils.set_axes_equal(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def add_paths(
        ax,
        points,
        reference_path,
        method_path,
        *,
        method_label,
        show_method_path,
        reference_label="input graph shortest path",
):
    utils.add_index_path(
        ax,
        points,
        reference_path,
        color="crimson",
        linewidth=2.8,
        label=reference_label,
        show_arrow=True,
        zorder=20,
    )
    if show_method_path:
        utils.add_index_path(
            ax,
            points,
            method_path,
            color="deepskyblue",
            linewidth=2.5,
            linestyle="--",
            label=method_label,
            show_endpoints=False,
            show_arrow=True,
            zorder=25,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("embedding", help="Sea NPZ filename or path.")
    parser.add_argument("--direction", default="tl_br", help="tl_br or br_tl.")
    parser.add_argument("--source", type=int)
    parser.add_argument("--target", type=int)
    parser.add_argument("--embedding-neighbors", type=int, default=12)
    parser.add_argument("--hide-method-path", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main_sea_paths()
