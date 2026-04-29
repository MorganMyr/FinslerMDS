import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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


def plot_points(X, X_noiseless=None, shape_type=None, quiver_field=None, step_quiver=None):
    assert X.shape[1] == 3

    X_ctr = X - np.mean(X, axis=0)
    X_color = X_noiseless if X_noiseless is not None else X

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scale_dot_plt = 200
    if shape_type is None:
        ax.scatter(X_ctr[:, 0], X_ctr[:, 1], X_ctr[:, 2], s=scale_dot_plt, lw=0, alpha=1)
    elif shape_type in ['swiss_roll']:
        factor = 200
        ax.scatter(
            X_ctr[:, 0],
            X_ctr[:, 1],
            X_ctr[:, 2],
            c=plt.cm.jet((X_color[:, 0] ** 2 + X_color[:, 2] ** 2) / factor),
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


def plot_proj_points(proj, X=None, knn=None, X_noiseless=None, shape_type=None, edge_alpha=1., extrema=None, fig_ax=None):
    assert (proj.shape[1] == 2 or proj.shape[1] == 3)

    X_color = X_noiseless if X_noiseless is not None else X

    if fig_ax is None:
        fig = plt.figure(figsize=(10, 10))
        if proj.shape[1] == 2:
            ax = fig.add_subplot(111)
        elif proj.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax

    proj_scatter = (proj[:, 0], proj[:, 1]) if proj.shape[1] == 2 else (proj[:, 0], proj[:, 1], proj[:, 2])

    if shape_type is None:
        ax.scatter(*proj_scatter, s=200, lw=0, alpha=1)
    elif shape_type in ['swiss_roll']:
        factor = 200
        c = plt.cm.jet((X_color[:, 0] ** 2 + X_color[:, 2] ** 2) / factor) if X_color is not None else 'blue'
        ax.scatter(*proj_scatter, c=c, s=200, lw=0, alpha=1)
    elif shape_type in ['river', 'sea']:
        c = plt.cm.jet(X_color[:, 0] / (X_color[:, 0].max() - X_color[:, 0].min())) if X_color is not None else 'blue'
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
        for i in range(len(X)):
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
    "plot_points",
    "plot_proj_points",
    "get_extrema",
    "plot_randers_w_arrow",
]
