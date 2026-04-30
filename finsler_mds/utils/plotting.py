import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .graph import metric_graph_from_support, symmetric_knn_graph


def set_window_title(fig, title):
    """Set a Matplotlib figure window title when the backend supports it."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def sample_plot_indices(n_points, point_fraction=1.0, random_state=None):
    if point_fraction is None or point_fraction >= 1:
        return np.arange(n_points)
    if point_fraction <= 0:
        raise ValueError("point_fraction must be in (0, 1].")
    n_keep = max(1, int(np.ceil(point_fraction * n_points)))
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_points, size=n_keep, replace=False))


def plot_points(
    X,
    X_noiseless=None,
    shape_type=None,
    quiver_field=None,
    step_quiver=None,
    point_fraction=1.0,
    random_state=None,
):
    assert X.shape[1] == 3

    X_ctr = X - np.mean(X, axis=0)
    X_color = X_noiseless if X_noiseless is not None else X
    plot_idx = sample_plot_indices(len(X), point_fraction=point_fraction, random_state=random_state)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scale_dot_plt = 200
    if shape_type is None:
        ax.scatter(X_ctr[plot_idx, 0], X_ctr[plot_idx, 1], X_ctr[plot_idx, 2], s=scale_dot_plt, lw=0, alpha=1)
    elif shape_type in ['swiss_roll']:
        factor = 200
        ax.scatter(
            X_ctr[plot_idx, 0],
            X_ctr[plot_idx, 1],
            X_ctr[plot_idx, 2],
            c=plt.cm.jet((X_color[plot_idx, 0] ** 2 + X_color[plot_idx, 2] ** 2) / factor),
            s=scale_dot_plt,
            lw=0,
            alpha=1,
        )
    else:
        raise ValueError('shape_type not implemented.')

    if quiver_field is not None:
        step = 1 if step_quiver is None else step_quiver
        ax.quiver(
            X_ctr[:, 0][::step],
            X_ctr[:, 1][::step],
            X_ctr[:, 2][::step],
            quiver_field[:, 0][::step],
            quiver_field[:, 1][::step],
            quiver_field[:, 2][::step],
            color='k',
            length=2,
            normalize=True,
        )

    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.axis("off")

    return fig, ax


def plot_proj_points(
    proj,
    X=None,
    knn=None,
    X_noiseless=None,
    shape_type=None,
    edge_alpha=1.,
    extrema=None,
    fig_ax=None,
    point_fraction=1.0,
    random_state=None,
):
    assert (proj.shape[1] == 2 or proj.shape[1] == 3)

    X_color = X_noiseless if X_noiseless is not None else X
    plot_idx = sample_plot_indices(len(proj), point_fraction=point_fraction, random_state=random_state)

    if fig_ax is None:
        fig = plt.figure(figsize=(10, 10))
        if proj.shape[1] == 2:
            ax = fig.add_subplot(111)
        elif proj.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax

    proj_scatter = (
        (proj[plot_idx, 0], proj[plot_idx, 1])
        if proj.shape[1] == 2
        else (proj[plot_idx, 0], proj[plot_idx, 1], proj[plot_idx, 2])
    )

    if shape_type is None:
        ax.scatter(*proj_scatter, s=200, lw=0, alpha=1)
    elif shape_type in ['swiss_roll']:
        factor = 200
        c = plt.cm.jet((X_color[plot_idx, 0] ** 2 + X_color[plot_idx, 2] ** 2) / factor) if X_color is not None else 'blue'
        ax.scatter(*proj_scatter, c=c, s=200, lw=0, alpha=1)
    elif shape_type in ['river', 'sea']:
        c = plt.cm.jet(X_color[plot_idx, 0] / (X_color[:, 0].max() - X_color[:, 0].min())) if X_color is not None else 'blue'
        ax.scatter(*proj_scatter, c=c, s=100, lw=0, alpha=1, zorder=-1)
    else:
        raise ValueError('shape_type not implemented.')

    if extrema is not None and shape_type in ['sea'] and proj.shape[1] == 3:
        maxima = extrema['maxima']['values']
        minima = extrema['minima']['values']

        ax.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color='black', s=200, alpha=1, marker='o', zorder=1e6)
        ax.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color='gray', s=200, alpha=1, marker='o', zorder=1e6)

        s_points = 100

        fig_base = plt.figure()
        ax_base = fig_base.add_subplot(111, projection='3d')
        ax_base.scatter(*proj_scatter, c=c, s=s_points, lw=0, alpha=1, zorder=-1)
        ax_base.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color='black', marker='o', s=100, alpha=0)
        ax_base.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color='gray', marker='o', s=100, alpha=0)
        ax_base.azim = -110
        ax_base.elev = 40
        fig_base.savefig('res/' + shape_type + '_base_plot.pdf', format='pdf')

        fig_extrema = plt.figure()
        ax_extrema = fig_extrema.add_subplot(111, projection='3d')
        ax_extrema.scatter(*proj_scatter, c=c, s=s_points, lw=0, alpha=0, zorder=-1)
        ax_extrema.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], color='black', marker='o', s=100, alpha=1)
        ax_extrema.scatter(minima[:, 0], minima[:, 1], minima[:, 2], color='gray', marker='o', s=100, alpha=1)
        ax_extrema.azim = -110
        ax_extrema.elev = 40
        ax_extrema.set_axis_off()
        fig_extrema.patch.set_alpha(0)
        fig_extrema.savefig('res/' + shape_type + '_extrema_overlay.pdf', format='pdf', transparent=True)

    if knn is not None:
        for i in plot_idx:
            neighbors = knn[i]
            for j in range(len(neighbors)):
                if proj.shape[1] == 2:
                    ax.plot(
                        proj[[i, neighbors.astype('int')[j]], 0],
                        proj[[i, neighbors.astype('int')[j]], 1],
                        color='black',
                        alpha=edge_alpha,
                        zorder=-2,
                    )
                elif proj.shape[1] == 3:
                    ax.plot(
                        proj[[i, neighbors.astype('int')[j]], 0],
                        proj[[i, neighbors.astype('int')[j]], 1],
                        proj[[i, neighbors.astype('int')[j]], 2],
                        color='black',
                        alpha=edge_alpha,
                        zorder=-2,
                    )

    ax.set_aspect('equal', adjustable='box')
    return fig, ax


def plot_categorical_embedding(
    embedding,
    labels=None,
    title=None,
    xlabel="Embedding 1",
    ylabel="Embedding 2",
    save_path=None,
    fig_ax=None,
    s=8,
    cmap="tab20",
):
    """Plot a 2D embedding, optionally colored by categorical labels."""
    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2).")

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig, ax = fig_ax

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=s, lw=0)
    else:
        if hasattr(labels, "astype") and hasattr(labels.astype("category"), "cat"):
            values = labels.astype("category")
            codes = values.cat.codes.to_numpy()
            categories = list(values.cat.categories)
        else:
            categories, codes = np.unique(np.asarray(labels), return_inverse=True)
            categories = list(categories)

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=codes,
            cmap=cmap,
            s=s,
            lw=0,
        )
        handles, _ = scatter.legend_elements()
        ax.legend(
            handles,
            categories,
            title="labels",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=8,
        )

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def nearest_point_index(points, coord):
    """Return the index of the point closest to ``coord``."""
    points = np.asarray(points)
    coord = np.asarray(coord)
    if points.ndim != 2:
        raise ValueError("points must have shape (n_samples, n_features).")
    if coord.shape != (points.shape[1],):
        raise ValueError(f"coord must have shape ({points.shape[1]},).")
    return int(np.argmin(np.linalg.norm(points - coord, axis=1)))


def build_geodesic_plot_graph(
    embedding,
    metric,
    n_neighbors=5,
    support_graph=None,
    neighbors_algorithm="auto",
    n_jobs=None,
):
    """Build the directed metric-weighted kNN graph used for path plots."""
    embedding = np.asarray(embedding)
    if support_graph is None:
        support_graph = symmetric_knn_graph(
            embedding,
            n_neighbors=n_neighbors,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
        )
    return metric_graph_from_support(embedding, support_graph, metric)


def geodesic_path_indices(
    embedding,
    source,
    target,
    metric,
    n_neighbors=5,
    graph=None,
    directed=True,
):
    """Return the vertex indices of the shortest path between two points."""
    embedding = np.asarray(embedding)
    if graph is None:
        graph = build_geodesic_plot_graph(embedding, metric, n_neighbors=n_neighbors)

    _, predecessors = dijkstra(
        graph,
        directed=directed,
        indices=int(source),
        return_predecessors=True,
    )
    return _path_from_predecessors(predecessors, int(source), int(target))


def add_geodesic_path(
    ax,
    embedding,
    source,
    target,
    metric,
    n_neighbors=5,
    graph=None,
    directed=True,
    color="crimson",
    linewidth=3,
    alpha=1.0,
    endpoint_color="black",
    endpoint_size=60,
    source_marker="o",
    target_marker="X",
    show_arrow=False,
    arrow_size=0.5,
    zorder=10,
):
    """Add one graph-geodesic path to an existing 2D or 3D embedding plot."""
    embedding = np.asarray(embedding)
    path = geodesic_path_indices(
        embedding,
        source,
        target,
        metric,
        n_neighbors=n_neighbors,
        graph=graph,
        directed=directed,
    )
    if path is None:
        return None
    _plot_path(ax, embedding, path, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    _plot_path_endpoints(
        ax,
        embedding,
        path,
        color=endpoint_color,
        size=endpoint_size,
        source_marker=source_marker,
        target_marker=target_marker,
        zorder=zorder + 1,
    )
    if show_arrow:
        _plot_path_arrow(
            ax,
            embedding,
            path,
            color=endpoint_color,
            arrow_size=arrow_size,
            alpha=alpha,
            zorder=zorder + 2,
        )
    return path


def add_geodesic_path_by_coords(
    ax,
    embedding,
    source_coord,
    target_coord,
    metric,
    n_neighbors=5,
    graph=None,
    directed=True,
    **plot_kwargs,
):
    """Add a path between the points nearest to two coordinates."""
    source = nearest_point_index(embedding, source_coord)
    target = nearest_point_index(embedding, target_coord)
    return add_geodesic_path(
        ax,
        embedding,
        source,
        target,
        metric,
        n_neighbors=n_neighbors,
        graph=graph,
        directed=directed,
        **plot_kwargs,
    )


def add_random_geodesic_paths(
    ax,
    embedding,
    n_paths,
    metric,
    n_neighbors=5,
    random_state=None,
    graph=None,
    directed=True,
    colors=None,
    linewidth=3,
    alpha=0.9,
    endpoint_darken=0.45,
    endpoint_size=45,
    source_marker="o",
    target_marker="X",
    show_arrows=False,
    arrow_size=0.5,
    zorder=10,
):
    """Add random graph-geodesic paths to an existing embedding plot."""
    embedding = np.asarray(embedding)
    rng = np.random.default_rng(random_state)
    if graph is None:
        graph = build_geodesic_plot_graph(embedding, metric, n_neighbors=n_neighbors)

    pairs = _random_distinct_pairs(len(embedding), n_paths, rng)
    colors = _path_colors(n_paths, colors)
    sources = np.unique(pairs[:, 0])
    _, predecessors = dijkstra(
        graph,
        directed=directed,
        indices=sources,
        return_predecessors=True,
    )
    source_to_row = {source: row for row, source in enumerate(sources)}

    paths = []
    for path_id, (source, target) in enumerate(pairs):
        pred_row = predecessors[source_to_row[source]]
        path = _path_from_predecessors(pred_row, int(source), int(target))
        if path is None:
            continue
        path_color = colors[path_id]
        endpoint_color = _darken_color(path_color, endpoint_darken)
        _plot_path(ax, embedding, path, color=path_color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        _plot_path_endpoints(
            ax,
            embedding,
            path,
            color=endpoint_color,
            size=endpoint_size,
            source_marker=source_marker,
            target_marker=target_marker,
            zorder=zorder + 1,
        )
        if show_arrows:
            _plot_path_arrow(
                ax,
                embedding,
                path,
                color=endpoint_color,
                arrow_size=arrow_size,
                alpha=alpha,
                zorder=zorder + 2,
            )
        paths.append(path)
    return paths


def _path_colors(n_paths, colors):
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(n_paths)]
    elif len(colors) == 0:
        raise ValueError("colors must be None or contain at least one color.")
    else:
        colors = [colors[i % len(colors)] for i in range(n_paths)]
    return colors


def _darken_color(color, factor):
    rgb = np.asarray(to_rgb(color))
    return tuple(np.clip(factor * rgb, 0, 1))


def _random_distinct_pairs(n_points, n_pairs, rng):
    if n_points < 2:
        raise ValueError("At least two points are needed to draw paths.")
    pairs = rng.integers(0, n_points, size=(n_pairs, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while np.any(same):
        pairs[same, 1] = rng.integers(0, n_points, size=np.sum(same))
        same = pairs[:, 0] == pairs[:, 1]
    return pairs


def _path_from_predecessors(predecessors, source, target):
    if source == target:
        return [source]

    path = [target]
    current = target
    while current != source:
        previous = int(predecessors[current])
        if previous < 0:
            return None
        path.append(previous)
        current = previous
    path.reverse()
    return path


def _plot_path(ax, embedding, path, *, color, linewidth, alpha, zorder):
    path_points = embedding[path]
    if embedding.shape[1] == 2:
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    elif embedding.shape[1] == 3:
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        raise ValueError("embedding must be 2D or 3D.")


def _plot_path_endpoints(
    ax,
    embedding,
    path,
    *,
    color,
    size,
    source_marker,
    target_marker,
    zorder,
):
    source = embedding[path[0]]
    target = embedding[path[-1]]
    if embedding.shape[1] == 2:
        ax.scatter(source[0], source[1], color=color, s=size, marker=source_marker, zorder=zorder)
        ax.scatter(target[0], target[1], color=color, s=size, marker=target_marker, zorder=zorder)
    elif embedding.shape[1] == 3:
        ax.scatter(
            source[0],
            source[1],
            source[2],
            color=color,
            s=size,
            marker=source_marker,
            zorder=zorder,
        )
        ax.scatter(
            target[0],
            target[1],
            target[2],
            color=color,
            s=size,
            marker=target_marker,
            zorder=zorder,
        )


def _plot_path_arrow(ax, embedding, path, *, color, arrow_size, alpha, zorder):
    if len(path) < 2:
        return

    path_points = embedding[path]
    segment_id = _first_nonzero_segment(path_points)
    if segment_id is None:
        return

    start = path_points[segment_id]
    end = path_points[segment_id + 1]
    direction = end - start
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        return

    base = start
    arrow = arrow_size * direction / length
    if embedding.shape[1] == 2:
        ax.annotate(
            "",
            xy=base + arrow,
            xytext=base,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.8,
                "alpha": alpha,
                "mutation_scale": 14,
            },
            zorder=zorder,
        )
    elif embedding.shape[1] == 3:
        ax.quiver(
            base[0],
            base[1],
            base[2],
            arrow[0],
            arrow[1],
            arrow[2],
            color=color,
            arrow_length_ratio=0.45,
            linewidth=1.8,
            alpha=alpha,
            zorder=zorder,
        )


def _first_nonzero_segment(path_points):
    segments = path_points[1:] - path_points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    nonzero = np.flatnonzero(lengths > 1e-12)
    if len(nonzero) == 0:
        return None
    return int(nonzero[0])


def get_extrema(X, radius=1):
    assert X.shape[1] == 3

    tree = cKDTree(X)
    maxima = {'indices': [], 'values': []}
    minima = {'indices': [], 'values': []}
    for i, point in enumerate(X):
        neighbors_idx = tree.query_ball_point(point, radius)

        is_max = all(X[i, 2] >= X[j, 2] for j in neighbors_idx if i != j)
        is_min = all(X[i, 2] <= X[j, 2] for j in neighbors_idx if i != j)

        if is_max:
            maxima['values'].append(point)
            maxima['indices'].append(i)
        if is_min:
            minima['values'].append(point)
            minima['indices'].append(i)

    maxima['values'] = np.array(maxima['values'])
    maxima['indices'] = np.array(maxima['indices'])
    minima['values'] = np.array(minima['values'])
    minima['indices'] = np.array(minima['indices'])

    extrema = {'minima': minima, 'maxima': maxima}

    return extrema


def plot_randers_w_arrow(data, ax, shape_type=None, location='top_left'):
    # Plot the randers w arrow (exaggerated size)

    assert data.shape[1] == 2, "Data should be 2D"

    x_min_proj, x_max_proj = data[:, 0].min(), data[:, 0].max()
    y_min_proj, y_max_proj = data[:, 1].min(), data[:, 1].max()

    if shape_type in [None, 'swiss_roll']:
        offset_x_arrow = x_min_proj + 0.01 * (x_max_proj - x_min_proj)
        offset_y_arrow = y_max_proj - 0.01 * (y_max_proj - y_min_proj)
        arrow_len = 0.2 * (y_max_proj - y_min_proj)
        fontsize = 50
        alpha_head = 0.05
    elif shape_type in ['binary_tree']:
        arrow_len = 0.2 * (y_max_proj - y_min_proj)

        if location == 'top_left':
            eps_x = 0.2
            eps_y = -0.05
        elif location == 'top_right':
            eps_x = 1 - 0.2
            eps_y = -0.05
        elif location == 'bottom_left':
            eps_x = 0.2
            eps_y = -(1 - 0.23 - arrow_len / (y_max_proj - y_min_proj))
        elif location == 'bottom_right':
            eps_x = 1 - 0.01
            eps_y = -(1 - 0.01)
        else:
            raise ValueError("Unknown arrow location.")
        offset_x_arrow = x_min_proj + eps_x * (x_max_proj - x_min_proj)
        offset_y_arrow = y_max_proj + eps_y * (y_max_proj - y_min_proj)

        fontsize = 30
        alpha_head = 0.08
    else:
        raise ValueError("shape_type not implemented.")

    ax.arrow(
        offset_x_arrow,
        offset_y_arrow - arrow_len - alpha_head * (y_max_proj - y_min_proj),
        0,
        arrow_len,
        head_width=0.1 * arrow_len,
        head_length=0.1 * arrow_len,
        fc="black",
        ec="black",
        lw=4,
        length_includes_head=True,
    )
    ax.text(offset_x_arrow, offset_y_arrow, r'$\omega$', ha='center', va='center', fontsize=fontsize)


__all__ = [
    "set_window_title",
    "set_axes_equal",
    "sample_plot_indices",
    "plot_points",
    "plot_proj_points",
    "plot_categorical_embedding",
    "nearest_point_index",
    "build_geodesic_plot_graph",
    "geodesic_path_indices",
    "add_geodesic_path",
    "add_geodesic_path_by_coords",
    "add_random_geodesic_paths",
    "get_extrema",
    "plot_randers_w_arrow",
]
