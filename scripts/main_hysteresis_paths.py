from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.metrics import ConvexifiedToblerMetric, MatsumotoMetric
from finsler_mds.utils.graph import metric_graph_from_support, symmetric_knn_graph
from finsler_mds import utils


SEED = 42
DATA_METRIC = "matsumoto"  # one of {"matsumoto", "convexified_tobler"}
MATSUMOTO_ALPHA = 0.9
TOBLER_A = 3.5
TOBLER_B = 0.05

HEIGHT = 6.0
END_X = 2.0
BRANCH_SEPARATION = 2.4
RAMP_AMPLITUDE = 0.95
RAMP_TURNS = 2.0
RIBBON_HALF_WIDTH = 0.55
RIBBON_WIDTH_POINTS = 7
STAIR_SAMPLES_PER_UNIT = 6
RAMP_SAMPLES = 110
PORTAL_X_HALF_WIDTH = 0.5
PORTAL_X_POINTS = 5
PORTAL_Y_POINTS = 25
NOISE_SCALE = 0.035
KNN_NEIGHBORS = 10

RES_DIR = Path(__file__).resolve().parent / "res" / "hysteresis_paths"
FIG_DIR = RES_DIR / "figures"
RAW_DIR = RES_DIR / "raw"

GROUP_NAMES = ("portal", "staircase", "ramp")
GROUP_COLORS = ("#555555", "#377eb8", "#e41a1c")


def make_metric():
    if DATA_METRIC == "matsumoto":
        return MatsumotoMetric(alpha=MATSUMOTO_ALPHA)
    if DATA_METRIC == "convexified_tobler":
        return ConvexifiedToblerMetric(a=TOBLER_A, b=TOBLER_B)
    raise ValueError("DATA_METRIC must be 'matsumoto' or 'convexified_tobler'.")


class GraphBuilder:
    def __init__(self):
        self.points = []
        self.groups = []
        self.index_by_key = {}
        self.edges = set()

    def add_point(self, point, group):
        point = np.asarray(point, dtype=float)
        key = tuple(np.round(point, 10))
        index = self.index_by_key.get(key)
        if index is None:
            index = len(self.points)
            self.index_by_key[key] = index
            self.points.append(point)
            self.groups.append(group)
        return index

    def add_edge(self, i, j):
        if i != j:
            self.edges.add((min(i, j), max(i, j)))

    def arrays(self):
        X = np.vstack(self.points)
        groups = np.asarray(self.groups, dtype=int)
        rows = []
        cols = []
        for i, j in self.edges:
            rows.extend([i, j])
            cols.extend([j, i])
        support = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(X), len(X)))
        return X, groups, support


def add_polyline_ribbon(builder, centerline, group, width=RIBBON_HALF_WIDTH):
    centerline = np.asarray(centerline, dtype=float)
    offsets = np.linspace(-width, width, RIBBON_WIDTH_POINTS)
    previous_row = None

    for i, center in enumerate(centerline):
        if i == 0:
            tangent = centerline[1] - center
        elif i == len(centerline) - 1:
            tangent = center - centerline[i - 1]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        horizontal_tangent = tangent[:2]
        norm = np.linalg.norm(horizontal_tangent)
        if norm < 1e-12:
            normal = np.array([0.0, 1.0, 0.0])
        else:
            normal = np.array([-horizontal_tangent[1], horizontal_tangent[0], 0.0]) / norm

        row = [builder.add_point(center + offset * normal, group) for offset in offsets]
        for left, right in zip(row[:-1], row[1:]):
            builder.add_edge(left, right)
        if previous_row is not None:
            for current, previous in zip(row, previous_row):
                builder.add_edge(current, previous)
            for current, previous in zip(row[1:], previous_row[:-1]):
                builder.add_edge(current, previous)
            for current, previous in zip(row[:-1], previous_row[1:]):
                builder.add_edge(current, previous)
        previous_row = row


def add_endpoint_rectangle(builder, x_center, z):
    xs = np.linspace(x_center - PORTAL_X_HALF_WIDTH, x_center + PORTAL_X_HALF_WIDTH, PORTAL_X_POINTS)
    ys = np.linspace(-BRANCH_SEPARATION, BRANCH_SEPARATION, PORTAL_Y_POINTS)
    grid = np.empty((len(xs), len(ys)), dtype=int)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            grid[ix, iy] = builder.add_point((x, y, z), 0)
            if ix > 0:
                builder.add_edge(grid[ix - 1, iy], grid[ix, iy])
            if iy > 0:
                builder.add_edge(grid[ix, iy - 1], grid[ix, iy])
            if ix > 0 and iy > 0:
                builder.add_edge(grid[ix - 1, iy - 1], grid[ix, iy])
                builder.add_edge(grid[ix - 1, iy], grid[ix, iy - 1])


def sampled_segment(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = np.linalg.norm(b - a)
    n = max(2, int(round(STAIR_SAMPLES_PER_UNIT * length)) + 1)
    t = np.linspace(0.0, 1.0, n)
    return a[None, :] + t[:, None] * (b - a)[None, :]


def concatenate_segments(*segments):
    parts = []
    for segment in segments:
        if parts:
            segment = segment[1:]
        parts.append(segment)
    return np.vstack(parts)


def staircase_centerline():
    y = -BRANCH_SEPARATION
    mid_z = HEIGHT / 2.0
    return concatenate_segments(
        sampled_segment((0.0, y, 0.0), (0.0, y, mid_z)),
        sampled_segment((0.0, y, mid_z), (END_X, y, mid_z)),
        sampled_segment((END_X, y, mid_z), (END_X, y, HEIGHT)),
    )


def ramp_centerline():
    t = np.linspace(0.0, 1.0, RAMP_SAMPLES)
    angle = 2 * np.pi * RAMP_TURNS * t
    x = END_X * t
    y = BRANCH_SEPARATION + RAMP_AMPLITUDE * np.sin(angle)
    z = HEIGHT * t
    return np.column_stack([x, y, z])


def make_dataset():
    builder = GraphBuilder()
    add_endpoint_rectangle(builder, 0.0, 0.0)
    add_endpoint_rectangle(builder, END_X, HEIGHT)
    add_polyline_ribbon(builder, staircase_centerline(), 1)
    add_polyline_ribbon(builder, ramp_centerline(), 2)
    X, groups, _ = builder.arrays()
    X = add_noise(X)
    support = symmetric_knn_graph(
        X,
        n_neighbors=KNN_NEIGHBORS,
        ensure_connected=False,
    )
    n_components, _ = connected_components(support, directed=False)
    if n_components != 1:
        raise RuntimeError(f"The kNN support graph has {n_components} connected components.")
    return X, groups, support


def add_noise(X):
    rng = np.random.default_rng(SEED)
    X_noisy = X + rng.normal(scale=NOISE_SCALE, size=X.shape)
    center_start = np.argmin(np.linalg.norm(X - np.array([0.0, 0.0, 0.0]), axis=1))
    center_end = np.argmin(np.linalg.norm(X - np.array([END_X, 0.0, HEIGHT]), axis=1))
    X_noisy[center_start] = X[center_start]
    X_noisy[center_end] = X[center_end]
    return X_noisy


def shortest_path_indices(predecessors, start, end):
    if predecessors[end] < 0 and start != end:
        return np.array([], dtype=int)
    path = [end]
    node = end
    while node != start:
        node = predecessors[node]
        if node < 0:
            return np.array([], dtype=int)
        path.append(node)
    return np.asarray(path[::-1], dtype=int)


def branch_counts(path, groups):
    counts = Counter(GROUP_NAMES[group] for group in groups[path])
    return ", ".join(f"{name}={counts.get(name, 0)}" for name in GROUP_NAMES)


def compute_distances_and_paths(X, groups, support, metric):
    graph = metric_graph_from_support(X, support, metric)
    start = int(np.argmin(np.linalg.norm(X - np.array([0.0, 0.0, 0.0]), axis=1)))
    end = int(np.argmin(np.linalg.norm(X - np.array([END_X, 0.0, HEIGHT]), axis=1)))

    dist_up, pred_up = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    dist_down, pred_down = dijkstra(graph, directed=True, indices=end, return_predecessors=True)
    path_up = shortest_path_indices(pred_up, start, end)
    path_down = shortest_path_indices(pred_down, end, start)
    full_distances = dijkstra(graph, directed=True, return_predecessors=False)

    print(
        f"start->end distance={dist_up[end]:.6g}; "
        f"path counts: {branch_counts(path_up, groups)}"
    )
    print(
        f"end->start distance={dist_down[start]:.6g}; "
        f"path counts: {branch_counts(path_down, groups)}"
    )
    return full_distances, path_up, path_down


def group_colors(groups):
    return [GROUP_COLORS[group] for group in groups]


def add_legend(ax, *, include_paths=False):
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=6,
            label=name,
        )
        for name, color in zip(GROUP_NAMES, GROUP_COLORS)
    ]
    if include_paths:
        handles.extend(
            [
                plt.Line2D([0], [0], color="#111111", linewidth=4.5, label="start -> end"),
                plt.Line2D([0], [0], color="#ffd400", linewidth=4.5, label="end -> start"),
            ]
        )
    ax.legend(handles=handles, loc="best", frameon=True)


def save_3d_views(X, groups, path, title, path_up=None, path_down=None):
    views = [
        ("oblique", 24, -62),
        ("along x", 0, 0),
        ("along y", 0, 90),
        ("along z", 90, -90),
    ]
    fig = plt.figure(figsize=(11.0, 9.0))
    for index, (view_name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=group_colors(groups), s=5, alpha=0.38, linewidths=0)
        if path_up is not None and len(path_up) > 0:
            P = X[path_up]
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#111111", linewidth=5.0, zorder=10)
        if path_down is not None and len(path_down) > 0:
            P = X[path_down]
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color="#ffd400", linewidth=5.0, zorder=11)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title} - {view_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        utils.set_axes_equal(ax)
        if index == 1:
            add_legend(ax, include_paths=path_up is not None or path_down is not None)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    X, groups, support = make_dataset()
    metric = make_metric()
    print(
        f"Hysteresis dataset: n={len(X)}, kNN={KNN_NEIGHBORS}, "
        f"noise={NOISE_SCALE}, metric={DATA_METRIC}"
    )
    D, path_up, path_down = compute_distances_and_paths(X, groups, support, metric)

    np.savez(
        RAW_DIR / f"hysteresis_{DATA_METRIC}.npz",
        X=X,
        groups=groups,
        support=support,
        dissimilarities=D,
        path_up=path_up,
        path_down=path_down,
        metric=DATA_METRIC,
    )
    save_3d_views(X, groups, FIG_DIR / f"hysteresis_{DATA_METRIC}_dataset.pdf", "Dataset")
    save_3d_views(
        X,
        groups,
        FIG_DIR / f"hysteresis_{DATA_METRIC}_optimal_paths.pdf",
        "Optimal directed paths",
        path_up=path_up,
        path_down=path_down,
    )
    print(f"Saved figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
