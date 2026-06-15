"""RNA-velocity evaluation utilities for low-dimensional embeddings."""

from .directional_metrics import (
    BoundaryNeighborPlan,
    BoundaryTransitionNeighbors,
    CBDirResult,
    ClusterCoherenceScore,
    ClusterTransitionScore,
    ICVCohResult,
    build_boundary_neighbor_plan,
    cross_boundary_direction_correctness,
    in_cluster_velocity_coherence,
    load_boundary_neighbor_plan,
    load_or_compute_boundary_neighbor_plan,
    project_velocity_graph_to_embedding,
    save_boundary_neighbor_plan,
)
from .geometry_velocity import (
    GeometryVelocityResult,
    finsler_induced_velocity_field,
)
from .gap_distance import (
    GapDistanceResult,
    normalized_gap_distance,
)
from .pancreas_gap import (
    DEFAULT_PANCREAS_GAP,
    PancreasGapSelection,
    gap_arrays_to_cache,
    normalize_pancreas_gap_config,
    pancreas_gap_group_indices,
    pancreas_gap_prefix,
    select_pancreas_gap,
)
from .asymmetry import (
    VelocityAlignmentPreservationResult,
    velocity_alignment_cosines_from_pairs,
    velocity_alignment_preservation_from_neighbors,
    velocity_alignment_preservation_from_pairs,
)

__all__ = [
    "CBDirResult",
    "BoundaryNeighborPlan",
    "BoundaryTransitionNeighbors",
    "build_boundary_neighbor_plan",
    "ClusterCoherenceScore",
    "ClusterTransitionScore",
    "DEFAULT_PANCREAS_GAP",
    "GapDistanceResult",
    "GeometryVelocityResult",
    "ICVCohResult",
    "PancreasGapSelection",
    "cross_boundary_direction_correctness",
    "finsler_induced_velocity_field",
    "gap_arrays_to_cache",
    "in_cluster_velocity_coherence",
    "load_boundary_neighbor_plan",
    "load_or_compute_boundary_neighbor_plan",
    "normalize_pancreas_gap_config",
    "normalized_gap_distance",
    "pancreas_gap_group_indices",
    "pancreas_gap_prefix",
    "project_velocity_graph_to_embedding",
    "save_boundary_neighbor_plan",
    "select_pancreas_gap",
    "VelocityAlignmentPreservationResult",
    "velocity_alignment_cosines_from_pairs",
    "velocity_alignment_preservation_from_neighbors",
    "velocity_alignment_preservation_from_pairs",
]
