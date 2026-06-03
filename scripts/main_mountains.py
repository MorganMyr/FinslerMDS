"""Three-mountain synthetic Finsler-MDS experiment."""

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

import matplotlib.pyplot as plt
import numpy as np

from finsler_mds import ConvexifiedMatsumotoMetric, MatsumotoMetric, RandersMetric, fit_finsler_mds, utils
from finsler_mds.utils.embedding_io import adapt_embedding_dimension


def main_mountains():
    seed = 42
    dir_res = SCRIPT_DIR / "res" / "mountains"
    dir_fig = dir_res / "figures"
    dir_embeddings = dir_res / "embeddings"

    optimizer = "path_frozen"  # one of {"smacof", "gd", "path_frozen"}
    metric_name = "randers"  # one of {"randers", "matsumoto", "c_matsumoto"}; SMACOF always uses Randers
    init_source = "terrain"  # one of {"terrain", "latest_same"}
    n_components = 3

    alpha_target = 1
    alpha_embedding = 0.9
    n_neighbors = 10

    grid = {
        "nx": 60,
        "ny": 40,
        "xlim": (-15.0, 15.0),
        "ylim": (-10.0, 10.0),
        "xy_noise": 0.08,
    }
    mountains = {
        "left": {"center": (-10.0, 0.0), "height": 4.0, "sigma": (2.2, 2.2)},
        "middle": {"center": (0.0, 0.0), "height": 4.0, "sigma": (0.8, 6.0)},
        "right": {"center": (10.0, 0.0), "height": 4.0, "sigma": (2.2, 2.2)},
    }
    smacof = {
        "max_iter": 20,
        "pseudo_inv_solver": "gmres",
        "project_on_V": True,
        "check_monotony": False,
    }
    gd = {
        "max_iter": 300,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-9, "maxls": 40},
        "verbose": 0,
    }
    path_frozen = {
        "graph_neighbors": 12,
        "outer_iter": 20,
        "inner_iter": 10,
        "eps": 1e-6,
        "method": "L-BFGS-B",
        "optimizer_options": {"ftol": 1e-8, "maxls": 40},
        "n_landmark": 150,
        "landmark_sampling": "random",
        "n_local_pairs": 12,
        "local_pair_mode": "direct",
        "targets_per_landmark": 300,
        "local_global_reweighting": "count",
        "local_weight": 1.0,
        "device": "auto",
        "verbose": 1,
    }

    for directory in (dir_fig, dir_embeddings):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    X = make_mountain_surface(grid, mountains, rng)
    target_metric = ConvexifiedMatsumotoMetric(alpha=alpha_target)
    effective_metric_name = "randers" if normalize_optimizer(optimizer) == "smacof" else normalize_metric_name(metric_name)
    run_key = mountains_run_key(optimizer, effective_metric_name, alpha_target, alpha_embedding)
    embedding_path = dir_embeddings / f"{run_key}.npz"
    source, target = mountain_summit_pair(X, mountains)

    print("Building Convexified Matsumoto geodesic target distances on the mountain surface")
    target_distances, target_predecessors = utils.compute_metric_dist_matrix(
        X,
        target_metric,
        n_neighbors=n_neighbors,
        directed=True,
    )
    if not np.all(np.isfinite(target_distances)):
        raise ValueError("Target kNN graph is disconnected. Increase n_neighbors.")
    np.fill_diagonal(target_distances, 0.0)

    surface_path = shortest_path_indices(
        X,
        shortest_path_metric("c_matsumoto", alpha_target),
        source=source,
        target=target,
        n_neighbors=n_neighbors,
    )
    print(f"Surface summit path: source={source}, target={target}, vertices={len(surface_path)}")

    plot_surface_views(
        X,
        title=f"Three-mountain target surface, Convexified Matsumoto alpha={alpha_target:g}",
        save_path=dir_fig / f"a{alpha_tag(alpha_target)}_surface.pdf",
        paths=[("surface shortest path", surface_path, "crimson")],
        source=source,
        target=target,
    )

    init = make_initialization(
        X,
        target_distances,
        init_source=init_source,
        dir_embeddings=dir_embeddings,
        optimizer=optimizer,
        alpha_target=alpha_target,
        n_components=n_components,
    )
    embedding_metric = make_metric(effective_metric_name, alpha_embedding)
    fit_kwargs = optimizer_kwargs(
        optimizer,
        metric=embedding_metric,
        init=init,
        n_components=n_components,
        smacof=smacof,
        gd=gd,
        path_frozen=path_frozen,
        seed=seed,
    )
    print(
        f"Running {display_optimizer_name(optimizer)} "
            f"with target Convexified Matsumoto alpha={alpha_target:g}, embedding alpha={alpha_embedding:g}"
    )
    embedding, stress = fit_finsler_mds(target_distances, print_time=True, **fit_kwargs)
    np.savez(
        embedding_path,
        embedding=embedding,
        stress=np.asarray(stress, dtype=float),
        X=X,
        init=init,
        target_distances=target_distances,
        target_predecessors=target_predecessors,
        source=np.asarray(source, dtype=int),
        target=np.asarray(target, dtype=int),
        surface_path=surface_path,
        optimizer=np.asarray(normalize_optimizer(optimizer)),
            target_metric=np.asarray("convexified_matsumoto"),
        embedding_metric=np.asarray(effective_metric_name),
        alpha_target=np.asarray(alpha_target, dtype=float),
        alpha_embedding=np.asarray(alpha_embedding, dtype=float),
    )
    print(f"Saved embedding: {embedding_path}")

    embedding_paths = [("surface path", surface_path, "crimson")]
    if normalize_optimizer(optimizer) in {"smacof", "gd"}:
        embedding_paths.append(("embedding chord", np.asarray([source, target], dtype=int), "deepskyblue"))
    else:
        embedding_path_indices = shortest_path_indices(
            embedding,
            shortest_path_metric(effective_metric_name, alpha_embedding),
            source=source,
            target=target,
            n_neighbors=path_frozen["graph_neighbors"],
        )
        embedding_paths.append(("embedding geodesic path", embedding_path_indices, "deepskyblue"))
        print(f"Embedding summit path vertices: {len(embedding_path_indices)}")

    plot_surface_views(
        embedding,
        colors=surface_colors(X),
        title=(
            f"Mountains {display_optimizer_name(optimizer)}, {display_metric_name(effective_metric_name)}, "
            f"target CMats alpha={alpha_target:g}, emb alpha={alpha_embedding:g}"
        ),
        save_path=dir_fig / f"{run_key}.pdf",
        paths=embedding_paths,
        source=source,
        target=target,
    )
    print(f"  stress: {stress}")
    print(f"Saved figures in: {dir_fig}")


def make_mountain_surface(grid, mountains, rng):
    xs = np.linspace(*grid["xlim"], int(grid["nx"]))
    ys = np.linspace(*grid["ylim"], int(grid["ny"]))
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    xy += rng.normal(scale=float(grid["xy_noise"]), size=xy.shape)
    z = sum(gaussian_bump(xy, **params) for params in mountains.values())
    return np.column_stack([xy, z])


def gaussian_bump(xy, *, center, height, sigma):
    center = np.asarray(center, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    scaled = (xy - center) / sigma
    return float(height) * np.exp(-0.5 * np.sum(scaled**2, axis=1))


def mountain_summit_pair(X, mountains):
    left = nearest_xy_index(X, mountains["left"]["center"])
    right = nearest_xy_index(X, mountains["right"]["center"])
    if left == right:
        raise ValueError("Left and right mountain summits collapsed to the same point.")
    return left, right


def nearest_xy_index(points, xy):
    points = np.asarray(points, dtype=float)
    xy = np.asarray(xy, dtype=float)
    return int(np.argmin(np.linalg.norm(points[:, :2] - xy, axis=1)))


def shortest_path_metric(metric_name, alpha):
    metric_name = normalize_metric_name(metric_name)
    if metric_name == "c_matsumoto":
        return MatsumotoMetric(alpha=alpha)
    return make_metric(metric_name, alpha)


def shortest_path_indices(points, metric, *, source, target, n_neighbors):
    path = utils.geodesic_path_indices(
        points,
        source,
        target,
        metric,
        n_neighbors=n_neighbors,
        directed=True,
    )
    if len(path) == 0:
        raise ValueError(f"No shortest path found from {source} to {target}.")
    return np.asarray(path, dtype=int)


def make_metric(metric_name, alpha):
    metric_name = normalize_metric_name(metric_name)
    if metric_name == "randers":
        return RandersMetric(alpha=alpha)
    if metric_name == "matsumoto":
        return MatsumotoMetric(alpha=alpha)
    if metric_name == "c_matsumoto":
        return ConvexifiedMatsumotoMetric(alpha=alpha)
    raise ValueError(f"Unknown metric {metric_name!r}.")


def make_initialization(X, target_distances, *, init_source, dir_embeddings, optimizer, alpha_target, n_components):
    source = normalize_init_source(init_source)
    if source == "terrain":
        return adapt_embedding_dimension(X - X.mean(axis=0, keepdims=True), n_components)

    path = latest_embedding_path(dir_embeddings, optimizer=optimizer, alpha_target=alpha_target)
    with np.load(path) as data:
        init = np.asarray(data["embedding"], dtype=float)
    print(f"Loaded initialization from: {path}")
    return adapt_embedding_dimension(init, n_components)


def optimizer_kwargs(optimizer, *, metric, init, n_components, smacof, gd, path_frozen, seed):
    optimizer = normalize_optimizer(optimizer)
    if optimizer == "smacof":
        return {
            "optimizer": "smacof_randers",
            "metric": RandersMetric(alpha=metric.alpha),
            "init": init,
            "n_components": n_components,
            "n_init": 1,
            "n_jobs": 1,
            **smacof,
        }
    if optimizer == "gd":
        return {"optimizer": "gradient_descent", "metric": metric, "init": init, "n_components": n_components, "random_state": seed, **gd}
    if optimizer == "path_frozen":
        return {"optimizer": "path_frozen", "metric": metric, "init": init, "n_components": n_components, "random_state": seed, **path_frozen}
    raise ValueError("optimizer must be one of {'smacof', 'gd', 'path_frozen'}.")


def plot_surface_views(points, *, title, save_path, colors=None, paths=(), source=None, target=None):
    points = np.asarray(points, dtype=float)
    colors = surface_colors(points) if colors is None else colors
    views = [("front", 20, -60), ("side", 20, 30), ("top", 90, -90), ("diagonal", 35, 135)]
    fig = plt.figure(figsize=(11, 9))
    for view_id, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_id + 1, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=16, lw=0)
        add_paths_to_axis(ax, points, paths)
        add_endpoints_to_axis(ax, points, source=source, target=target)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name)
        ax.set_box_aspect([1, 1, 0.45])
        utils.set_axes_equal(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path)
    plt.close(fig)


def add_paths_to_axis(ax, points, paths):
    for label, indices, color in paths:
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            continue
        ax.plot(
            points[indices, 0],
            points[indices, 1],
            points[indices, 2],
            color=color,
            lw=3.0,
            alpha=0.95,
            label=label,
            zorder=1_000_000,
        )


def add_endpoints_to_axis(ax, points, *, source, target):
    if source is not None:
        ax.scatter(points[source, 0], points[source, 1], points[source, 2], c="black", s=85, marker="o", zorder=1_000_001)
    if target is not None:
        ax.scatter(points[target, 0], points[target, 1], points[target, 2], c="white", edgecolor="black", s=85, marker="o", zorder=1_000_001)


def surface_colors(points):
    z = np.asarray(points)[:, 2]
    values = (z - z.min()) / max(np.ptp(z), 1e-12)
    return plt.cm.terrain(values)


def latest_embedding_path(dir_embeddings, *, optimizer, alpha_target):
    prefix = f"a{alpha_tag(alpha_target)}_{optimizer_abbrev(normalize_optimizer(optimizer))}_"
    candidates = sorted(dir_embeddings.glob(f"{prefix}*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No saved embedding matching {prefix}*.npz in {dir_embeddings}.")
    return candidates[0]


def mountains_run_key(optimizer, metric_name, alpha_target, alpha_embedding):
    optimizer = normalize_optimizer(optimizer)
    if optimizer == "smacof":
        return f"a{alpha_tag(alpha_target)}_smacof_a{alpha_tag(alpha_embedding)}"
    metric = metric_abbrev(normalize_metric_name(metric_name))
    return f"a{alpha_tag(alpha_target)}_{optimizer_abbrev(optimizer)}_{metric}_a{alpha_tag(alpha_embedding)}"


def normalize_optimizer(optimizer):
    aliases = {
        "smacof": "smacof",
        "randers_smacof": "smacof",
        "smacof_randers": "smacof",
        "gd": "gd",
        "gradient_descent": "gd",
        "path_frozen": "path_frozen",
        "pf": "path_frozen",
    }
    key = str(optimizer).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError("optimizer must be one of {'smacof', 'gd', 'path_frozen'}.")
    return aliases[key]


def normalize_init_source(init_source):
    aliases = {
        "terrain": "terrain",
        "original": "terrain",
        "latest": "latest_same",
        "latest_same": "latest_same",
        "saved": "latest_same",
    }
    key = str(init_source).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError("init_source must be one of {'terrain', 'latest_same'}.")
    return aliases[key]


def normalize_metric_name(metric_name):
    aliases = {
        "randers": "randers",
        "r": "randers",
        "matsumoto": "matsumoto",
        "mats": "matsumoto",
        "c_matsumoto": "c_matsumoto",
        "cmatsumoto": "c_matsumoto",
        "convexified_matsumoto": "c_matsumoto",
        "convexifiedmatsumoto": "c_matsumoto",
        "cmats": "c_matsumoto",
        "convmats": "c_matsumoto",
    }
    key = str(metric_name).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError("metric_name must be one of {'randers', 'matsumoto', 'c_matsumoto'}.")
    return aliases[key]


def optimizer_abbrev(optimizer):
    return {"smacof": "smacof", "gd": "gd", "path_frozen": "pf"}[optimizer]


def metric_abbrev(metric_name):
    return {"randers": "r", "matsumoto": "mats", "c_matsumoto": "cmats"}[metric_name]


def display_optimizer_name(optimizer):
    return {"smacof": "SMACOF", "gd": "GD", "path_frozen": "Path-frozen"}[normalize_optimizer(optimizer)]


def display_metric_name(metric_name):
    return {
        "randers": "Randers",
        "matsumoto": "Matsumoto",
        "c_matsumoto": "Convexified Matsumoto",
    }[normalize_metric_name(metric_name)]


def alpha_tag(alpha):
    alpha = float(alpha)
    return str(int(alpha)) if alpha.is_integer() else f"{alpha:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main_mountains()
