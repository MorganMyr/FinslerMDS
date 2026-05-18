"""Synthetic current-map datasets used by ``main_sea`` scripts."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as patches
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components

from finsler_mds import utils
from finsler_mds.utils.graph import symmetric_knn_graph


@dataclass(frozen=True)
class SeaDataset:
    key: str
    title: str
    X: np.ndarray
    randers_field: np.ndarray
    current_field: np.ndarray
    bounds: tuple[float, float, float, float]
    obstacles: tuple[dict, ...] = ()
    support_graph: scipy.sparse.csr_matrix | None = None


def make_sea_dataset(
    name,
    *,
    n_samples,
    alpha_current,
    rng,
    graph_neighbors=10,
    sea_length=10.0,
    sea_width=10.0,
    current_frequency=2.0,
):
    name = normalize_sea_dataset_name(name)
    if name == "sea1":
        return make_vortical_sea(
            n_samples,
            sea_length=sea_length,
            sea_width=sea_width,
            frequency=current_frequency,
            alpha_current=alpha_current,
            rng=rng,
        )
    if name == "sea2":
        return make_archipelago_sea(
            n_samples,
            alpha_current=alpha_current,
            rng=rng,
            graph_neighbors=graph_neighbors,
        )
    raise ValueError(f"Unknown sea dataset {name!r}.")


def normalize_sea_dataset_name(name):
    aliases = {
        "sea": "sea1",
        "sea1": "sea1",
        "vortical": "sea1",
        "old": "sea1",
        "sea2": "sea2",
        "archipelago": "sea2",
        "islands": "sea2",
    }
    key = str(name).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError("dataset_name must be one of {'sea1', 'sea2'}.")
    return aliases[key]


def make_vortical_sea(n_samples, *, sea_length, sea_width, frequency, alpha_current, rng):
    X = rng.random((n_samples, 2))
    X[:, 0] *= sea_length
    X[:, 1] *= sea_width

    field = np.empty_like(X)
    field[:, 0] = np.sin(frequency * X[:, 0]) + np.cos(frequency * X[:, 1])
    field[:, 1] = np.cos(frequency * X[:, 0]) - np.sin(frequency * X[:, 1])
    max_norm = np.linalg.norm(field, axis=1).max()
    if max_norm > 0:
        field = field / max_norm

    randers_field = alpha_current * field
    current_field = -randers_field
    return SeaDataset(
        key="sea1",
        title="Sea1 vortical current map",
        X=X,
        randers_field=randers_field,
        current_field=current_field,
        bounds=(0.0, sea_length, 0.0, sea_width),
    )


def make_archipelago_sea(n_samples, *, alpha_current, rng, graph_neighbors):
    bounds = (0.0, 12.0, 0.0, 6.0)
    obstacles = (
        ellipse_obstacle(4.25, 3.0, 0.72, 2.35, 4.0),
        ellipse_obstacle(6.9, 1.45, 0.88, 0.42, -18.0),
        ellipse_obstacle(7.8, 4.55, 0.94, 0.46, 22.0),
        ellipse_obstacle(9.45, 3.05, 0.58, 0.86, -10.0),
    )
    X = sample_water_points_stratified(n_samples, bounds=bounds, obstacles=obstacles, rng=rng)
    current_vector = archipelago_current_vector(X, obstacles, bounds=bounds)
    current_field = alpha_current * current_vector
    randers_field = -current_field
    support_graph = obstacle_aware_knn_graph(
        X,
        obstacles=obstacles,
        n_neighbors=graph_neighbors,
        max_neighbors=max(8 * graph_neighbors, graph_neighbors + 80),
    )
    return SeaDataset(
        key="sea2",
        title="Sea2 archipelago current map",
        X=X,
        randers_field=randers_field,
        current_field=current_field,
        bounds=bounds,
        obstacles=obstacles,
        support_graph=support_graph,
    )


def current_map_distances(dataset, *, n_neighbors, path_method="auto"):
    return utils.compute_dist_matrix(
        dataset.X,
        n_neighbors=n_neighbors,
        path_method=path_method,
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        randers_field=dataset.randers_field,
        nn_riemannian_precomputed=dataset.support_graph,
    )


def ellipse_obstacle(cx, cy, rx, ry, angle_deg):
    return {
        "kind": "ellipse",
        "center": np.asarray([cx, cy], dtype=float),
        "radii": np.asarray([rx, ry], dtype=float),
        "angle_deg": float(angle_deg),
        "angle": np.deg2rad(angle_deg),
    }


def sample_water_points(n_samples, *, bounds, obstacles, rng):
    xmin, xmax, ymin, ymax = bounds
    points = []
    batch = max(2048, 2 * n_samples)
    while sum(len(chunk) for chunk in points) < n_samples:
        candidates = np.column_stack(
            [
                rng.uniform(xmin, xmax, size=batch),
                rng.uniform(ymin, ymax, size=batch),
            ]
        )
        water = ~inside_any_obstacle(candidates, obstacles)
        if np.any(water):
            points.append(candidates[water])
    return np.vstack(points)[:n_samples]


def sample_water_points_stratified(n_samples, *, bounds, obstacles, rng):
    """Return an almost uniform jittered grid outside the obstacles."""
    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    obstacle_area = sum(np.pi * np.prod(obstacle["radii"]) for obstacle in obstacles)
    water_fraction = np.clip(1.0 - obstacle_area / (width * height), 0.55, 0.98)
    n_cells = int(np.ceil(1.25 * n_samples / water_fraction))

    for _ in range(8):
        nx = int(np.ceil(np.sqrt(n_cells * width / height)))
        ny = int(np.ceil(n_cells / nx))
        dx = width / nx
        dy = height / ny
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        candidates = np.column_stack(
            [
                xmin + (ix.ravel() + rng.random(ix.size)) * dx,
                ymin + (iy.ravel() + rng.random(iy.size)) * dy,
            ]
        )
        water = candidates[~inside_any_obstacle(candidates, obstacles)]
        if len(water) >= n_samples:
            return spatially_balanced_subset(water, n_samples, rng=rng)
        n_cells = int(np.ceil(1.35 * n_cells))

    return sample_water_points(n_samples, bounds=bounds, obstacles=obstacles, rng=rng)


def spatially_balanced_subset(points, n_samples, *, rng):
    points = np.asarray(points, dtype=float)
    if len(points) == n_samples:
        return points
    selected = np.empty(n_samples, dtype=int)
    selected[0] = int(rng.integers(len(points)))
    min_sqdist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    min_sqdist[selected[0]] = -1.0
    for k in range(1, n_samples):
        selected[k] = int(np.argmax(min_sqdist))
        distances = np.sum((points - points[selected[k]]) ** 2, axis=1)
        min_sqdist = np.minimum(min_sqdist, distances)
        min_sqdist[selected[: k + 1]] = -1.0
    return points[selected]


def inside_any_obstacle(points, obstacles, *, margin=0.0):
    points = np.asarray(points, dtype=float)
    inside = np.zeros(points.shape[0], dtype=bool)
    for obstacle in obstacles:
        inside |= ellipse_level(points, obstacle) <= (1.0 + margin) ** 2
    return inside


def ellipse_level(points, obstacle):
    local = rotate_points(points - obstacle["center"], -obstacle["angle"])
    scaled = local / obstacle["radii"]
    return np.sum(scaled**2, axis=1)


def rotate_points(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T


def archipelago_current_vector(X, obstacles, *, bounds):
    base = np.asarray([1.0, -0.08])
    base = base / np.linalg.norm(base)
    field = np.tile(base, (len(X), 1))

    x = X[:, 0]
    y = X[:, 1]
    lateral = 0.18 * np.sin(0.65 * x + 1.15 * y) + 0.12 * np.sin(1.4 * y)
    diagonal_down = gaussian_bump(X, center=(5.5, 4.7), sigma=(1.8, 0.9))
    diagonal_up = gaussian_bump(X, center=(8.6, 1.25), sigma=(1.7, 0.75))
    field[:, 1] += lateral - 0.32 * diagonal_down + 0.26 * diagonal_up
    field[:, 0] += 0.10 * np.sin(0.8 * y - 0.25 * x)

    for i, x in enumerate(X):
        best = nearest_obstacle_geometry(x, obstacles)
        if best is None:
            continue
        distance_proxy, normal = best
        # Smoothly project the flow onto the obstacle tangent near islands.
        weight = np.exp(-(distance_proxy / 0.7) ** 2)
        tangential = field[i] - weight * np.dot(field[i], normal) * normal
        tangent = np.asarray([-normal[1], normal[0]])
        if np.dot(tangent, base) < 0:
            tangent = -tangent
        field[i] = tangential + 0.16 * weight * tangent + 0.1 * (1.0 - weight) * base

    norms = np.linalg.norm(field, axis=1)
    directions = np.divide(field, norms[:, None], out=np.tile(base, (len(X), 1)), where=norms[:, None] > 1e-12)

    speed = 0.78 + 0.12 * normalize01(X[:, 0])
    speed -= 0.26 * gaussian_bump(X, center=(5.55, 3.0), sigma=(0.8, 1.2))
    speed -= 0.22 * gaussian_bump(X, center=(8.6, 3.25), sigma=(0.75, 1.05))
    speed += 0.10 * gaussian_bump(X, center=(10.8, 1.2), sigma=(1.0, 0.8))
    speed = np.clip(speed, 0.38, 0.98)
    return speed[:, None] * directions


def gaussian_bump(X, *, center, sigma):
    center = np.asarray(center, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    scaled = (np.asarray(X, dtype=float) - center) / sigma
    return np.exp(-0.5 * np.sum(scaled**2, axis=1))


def normalize01(values):
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def nearest_obstacle_geometry(x, obstacles):
    best = None
    for obstacle in obstacles:
        local = rotate_points((x - obstacle["center"])[None, :], -obstacle["angle"])[0]
        scaled = local / obstacle["radii"]
        level = float(np.sum(scaled**2))
        distance_proxy = max(np.sqrt(level) - 1.0, 0.0)

        grad_local = 2.0 * local / (obstacle["radii"] ** 2)
        grad = rotate_points(grad_local[None, :], obstacle["angle"])[0]
        norm = np.linalg.norm(grad)
        if norm <= 1e-12:
            continue
        normal = grad / norm
        if best is None or distance_proxy < best[0]:
            best = (distance_proxy, normal)
    return best


def obstacle_aware_knn_graph(X, *, obstacles, n_neighbors, max_neighbors):
    candidate = symmetric_knn_graph(X, n_neighbors=max(n_neighbors, max_neighbors))
    support = remove_obstacle_crossing_edges(X, candidate, obstacles)
    n_components, _ = connected_components(support)
    if n_components != 1:
        support = connect_obstacle_graph_components(X, support, obstacles)
        n_components, _ = connected_components(support)
        if n_components != 1:
            raise RuntimeError(
                "Could not build a connected obstacle-aware graph "
                f"({n_components} components). Reduce obstacle sizes or increase max_neighbors."
            )
    return support


def remove_obstacle_crossing_edges(X, graph, obstacles):
    coo = graph.tocoo()
    keep = np.ones(coo.nnz, dtype=bool)
    rows = coo.row
    cols = coo.col
    p = X[rows]
    q = X[cols]
    for obstacle in obstacles:
        for t in np.linspace(0.05, 0.95, 15):
            points = (1.0 - t) * p + t * q
            keep &= ellipse_level(points, obstacle) > 1.0
            if not np.any(keep):
                break
    return scipy.sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=graph.shape,
    ).tocsr()


def connect_obstacle_graph_components(X, graph, obstacles):
    graph = graph.tolil(copy=True)
    while True:
        n_components, labels = connected_components(graph.tocsr())
        if n_components <= 1:
            return graph.tocsr()
        base_label = largest_component_label(labels)
        base_indices = np.flatnonzero(labels == base_label)
        best = None
        for label in range(n_components):
            if label == base_label:
                continue
            other_indices = np.flatnonzero(labels == label)
            candidate = nearest_visible_pair(X, base_indices, other_indices, obstacles)
            if candidate is not None and (best is None or candidate[2] < best[2]):
                best = candidate
        if best is None:
            return graph.tocsr()
        i, j, distance = best
        graph[i, j] = distance
        graph[j, i] = distance


def largest_component_label(labels):
    counts = np.bincount(labels)
    return int(np.argmax(counts))


def nearest_visible_pair(X, first, second, obstacles):
    diff = X[first][:, None, :] - X[second][None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    flat_order = np.argsort(distances, axis=None)
    n_second = len(second)
    for flat_id in flat_order:
        a = flat_id // n_second
        b = flat_id % n_second
        i = int(first[a])
        j = int(second[b])
        if not segment_crosses_obstacles(X[i], X[j], obstacles):
            return i, j, float(distances[a, b])
    return None


def segment_crosses_obstacles(p, q, obstacles, *, n_checks=15):
    ts = np.linspace(0.05, 0.95, n_checks)
    segment = (1.0 - ts[:, None]) * p + ts[:, None] * q
    return bool(np.any(inside_any_obstacle(segment, obstacles, margin=0.0)))


def obstacle_array(obstacles):
    rows = []
    for obstacle in obstacles:
        rows.append(
            [
                obstacle["center"][0],
                obstacle["center"][1],
                obstacle["radii"][0],
                obstacle["radii"][1],
                obstacle["angle_deg"],
            ]
        )
    return np.asarray(rows, dtype=float)


def obstacles_from_array(values):
    values = np.asarray(values, dtype=float)
    return tuple(ellipse_obstacle(*row) for row in values.reshape((-1, 5)))


def add_obstacles_to_axis(ax, obstacles, *, facecolor="white", edgecolor="0.1", alpha=0.95, zorder=5):
    for obstacle in obstacles:
        patch = patches.Ellipse(
            xy=obstacle["center"],
            width=2 * obstacle["radii"][0],
            height=2 * obstacle["radii"][1],
            angle=obstacle["angle_deg"],
            facecolor=facecolor,
            edgecolor=edgecolor,
            lw=1.3,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(patch)
