"""Geometry utilities for QGS-Flow: local quadric fitting, k-NN, scene decomposition."""

from .quadric_fit import fit_local_quadrics
from .knn_radius import hybrid_radius_knn, gather_neighbors, default_r_max
from .decomposition import decompose_scene

__all__ = [
    "fit_local_quadrics",
    "hybrid_radius_knn",
    "gather_neighbors",
    "default_r_max",
    "decompose_scene",
]
