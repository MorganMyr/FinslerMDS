import warnings

import numpy as np


def wormhole_mask(dists, border, X, randers_drift_upper_bound=0., small_dists_threshold=1e-6):
    """
    :param dists:
        Pairwise distances between points. Should be symmetric if randers_field is 0.

    :param border:
        Boolean array indicating which points are on the boundary.

    :param X:
        Data points.

    :param randers_drift_upper_bound:
        Upper bound on the Randers drift component w. If 0, then we are in the Euclidean case as in the original wormhole paper NeurIPS 2024.
        If not 0, then we are in the Finsler case with a Randers drift component w.
        The constraint is ||w||_2 < rander_drift_upper_bound < 1 everywhere on the full manifold.

    :param small_dists_threshold:
        Small distances threshold. If the distance between two points is smaller than this threshold, we set the mask to 1.

    :return:
        mask_euclidean:
            Boolean array indicating wormhole guaranteed pairs.

        mask_criterion_sum_dists_boundary:
            Boolean array indicating wormhole guaranteed pairs, without the small_dists_threshold.

        mask_small_dists:
            Boolean array indicating the pairs within the small_dists_threshold.
    """

    warnings.warn(
        "Running a poor implementation of the wormhole mask. Currently quadratic but can be made much faster.")

    if randers_drift_upper_bound == 0:
        assert np.allclose(dists, dists.T)

    dists_boundary = dists[:, border]

    mask_criterion_sum_dists_boundary = np.ones(dists.shape, dtype=bool)

    print('')
    border_ids = np.nonzero(border)[0]
    for b1 in range(len(border_ids)):
        for b2 in range(len(border_ids)):
            print('\r', b1, b2, '[total:', len(border_ids), ']', end='')
            meshgrid_dists_boundary_b1_b2 = np.meshgrid(dists_boundary[:, b1], dists_boundary[:, b2], indexing='ij')
            sum_min_dists_boundary_b1_b2 = meshgrid_dists_boundary_b1_b2[0] + meshgrid_dists_boundary_b1_b2[1]
            euclidean_dist_b1_b2 = np.linalg.norm(X[border_ids[b1], :] - X[border_ids[b2], :])
            randers_bound_dist_b1_b2 = (1 - randers_drift_upper_bound) * euclidean_dist_b1_b2
            mask_criterion_sum_dists_boundary_b1_b2 = sum_min_dists_boundary_b1_b2 + randers_bound_dist_b1_b2 > dists
            mask_criterion_sum_dists_boundary = np.logical_and(
                mask_criterion_sum_dists_boundary,
                mask_criterion_sum_dists_boundary_b1_b2,
            )
    print('')

    mask_small_dists = dists < small_dists_threshold

    mask_euclidean = np.logical_or(mask_criterion_sum_dists_boundary, mask_small_dists)

    return mask_euclidean, mask_criterion_sum_dists_boundary, mask_small_dists


__all__ = [
    "wormhole_mask",
]
