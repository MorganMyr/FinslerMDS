"""Generic evaluation utilities for Finsler-MDS embeddings."""

from .geodesic_stress import (
    evaluate_geodesic_stress,
    geodesic_embedding_stress,
)
from .asymmetry import (
    AsymmetryPreservationResult,
    asymmetry_preservation_from_neighbors,
    asymmetry_preservation_from_pairs,
    asymmetry_score,
    neighbor_pairs,
    summarize_asymmetry_preservation,
)
from .distance_embedding import (
    DistanceEmbeddingEvaluation,
    StretchSummary,
    compute_embedding_distances,
    evaluate_distance_embedding,
    evaluate_precomputed_embedding_distances,
    stress_from_distance_matrices,
    stretch_summary,
)

__all__ = [
    "AsymmetryPreservationResult",
    "DistanceEmbeddingEvaluation",
    "StretchSummary",
    "asymmetry_preservation_from_neighbors",
    "asymmetry_preservation_from_pairs",
    "asymmetry_score",
    "compute_embedding_distances",
    "evaluate_distance_embedding",
    "evaluate_geodesic_stress",
    "evaluate_precomputed_embedding_distances",
    "geodesic_embedding_stress",
    "neighbor_pairs",
    "stress_from_distance_matrices",
    "summarize_asymmetry_preservation",
    "stretch_summary",
]
