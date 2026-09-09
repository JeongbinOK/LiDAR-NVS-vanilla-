"""Compatibility exports for grid builder helpers.

Implementations are grouped by geometry, seed configuration, seed generation,
and token aggregation in the neighboring semantic modules.
"""
from .grid_geometry import UtoniaGridMapper, _cell_centers, xyz_to_theta_phi_r
from .grid_seed_config import GridSeedConfig
from .grid_seeds import (
    GridSeedData,
    _build_seed_data,
    _medoid_seed_data,
    _r_quantile_seed_bank,
    _r_quantile_seed_data,
    _token_position_seed_data,
    _viewpoint_seed_bank,
    counts_to_variable_k,
)
from .token_aggregation import (
    RawTokenMembership,
    _aggregate_points_to_cells,
    aggregate_points_to_cells_with_membership,
    aggregate_points_to_cells_with_seeds,
    split_by_offset,
)

__all__ = [
    "GridSeedConfig",
    "GridSeedData",
    "RawTokenMembership",
    "UtoniaGridMapper",
    "aggregate_points_to_cells_with_membership",
    "aggregate_points_to_cells_with_seeds",
    "counts_to_variable_k",
    "split_by_offset",
    "xyz_to_theta_phi_r",
]
