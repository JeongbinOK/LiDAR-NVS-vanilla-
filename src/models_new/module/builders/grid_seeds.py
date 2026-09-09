"""Deterministic occupied-grid seed generation algorithms."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .grid_geometry import _cell_centers


@dataclass(frozen=True)
class GridSeedData:
    """Own-frame seed geometry aligned with occupied token rows.

    Legacy count routing stores one padded seed set with shape
    ``(N, K_max, 3)`` and its deterministic ``anchor_k``. Learned Gumbel
    routing stores every K-specific candidate set with shape
    ``(N, K_max, K_max, 3)`` and ``anchor_k`` is ``None`` because K is
    predicted only after temporal feature fusion. ``learned_gumbel`` fills a
    candidate row with ``k`` range-quantile seeds. ``learned_gumbel_viewpt``
    instead reserves slot zero for one observed medoid (the Common Gaussian)
    and fills slots ``1..k-1`` with range-quantile Additional seeds.
    """

    seed_sensor: torch.Tensor
    delta_sensor: torch.Tensor
    anchor_k: torch.Tensor | None


def counts_to_variable_k(raw_count, points_per_gaussian: int, k_max: int):
    """Map positive own-frame point counts to ``ceil(count / ppg)`` in [1, Kmax]."""
    points_per_gaussian = int(points_per_gaussian)
    k_max = int(k_max)
    if points_per_gaussian <= 0:
        raise ValueError("points_per_gaussian must be positive")
    if k_max <= 0:
        raise ValueError("K_max must be positive")
    count = raw_count.to(dtype=torch.long).clamp_min(1)
    k = torch.div(
        count + points_per_gaussian - 1,
        points_per_gaussian,
        rounding_mode="floor",
    )
    return k.clamp_(min=1, max=k_max)


def _r_quantile_seed_data(points, cell_idx, counts, grid_coord, occupied,
                          metric_origin, mapper, points_per_gaussian, k_max,
                          fixed_k=False):
    """Select deterministic range-quantile points for each occupied cell.

    ``fixed_k`` gives every token all ``k_max`` slots instead of deriving the
    count from its raw point total. A token with fewer raw points than slots
    then repeats observed points, exactly as the learned-count candidate bank
    does; the per-slot parameter blocks separate them afterwards.
    """
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, 3))
    occupied_count = counts.long()[occupied_index]
    if fixed_k:
        anchor_k = torch.full_like(occupied_count, int(k_max))
    else:
        anchor_k = counts_to_variable_k(
            occupied_count, points_per_gaussian, k_max
        )
    if num_cells == 0:
        return GridSeedData(seeds, delta, anchor_k)

    point_range = points.norm(dim=-1)
    order = torch.arange(points.shape[0], device=points.device)
    # Stable least-to-most-significant sorting gives (cell, r, x, y, z).
    for value in (points[:, 2], points[:, 1], points[:, 0], point_range, cell_idx):
        order = order[torch.argsort(value[order], stable=True)]
    sorted_points = points[order]

    count_long = counts.long()
    starts = torch.cumsum(count_long, dim=0) - count_long
    slot = torch.arange(k_max, device=points.device).view(1, -1)
    valid = slot < anchor_k.view(-1, 1)
    rank = torch.div(
        (2 * slot + 1) * occupied_count.view(-1, 1),
        2 * anchor_k.view(-1, 1),
        rounding_mode="floor",
    )
    sorted_row = starts[occupied_index].view(-1, 1) + rank
    seeds[valid] = sorted_points[sorted_row[valid]]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    delta[valid] = seeds[valid] - cell_center[:, None, :].expand_as(seeds)[valid]
    return GridSeedData(seeds, delta, anchor_k)


def _r_quantile_seed_bank(points, cell_idx, counts, grid_coord, occupied,
                          metric_origin, mapper, k_max):
    """Precompute the range-quantile seed set for every candidate K.

    The learned count router runs after cross-frame feature fusion, while seed
    geometry must be transformed into ref/box-local frames by the temporal
    aggregator.  Therefore all candidates are built up front. Candidate
    ``k - 1`` follows the same deterministic rule as the legacy K path:

        rank(s, K) = floor(((2s + 1) * N_raw) / (2K)),  s = 0 .. K - 1.

    If ``N_raw < K``, quantiles may intentionally select the same observed raw
    point more than once; the K-specific Gaussian head can separate those slots
    through its learned position offsets.
    """
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, k_max, 3))
    if num_cells == 0:
        return GridSeedData(seeds, delta, None)

    point_range = points.norm(dim=-1)
    order = torch.arange(points.shape[0], device=points.device)
    # Stable least-to-most-significant sorting gives (cell, r, x, y, z).
    for value in (points[:, 2], points[:, 1], points[:, 0], point_range, cell_idx):
        order = order[torch.argsort(value[order], stable=True)]
    sorted_points = points[order]

    count_long = counts.long()
    occupied_count = count_long[occupied_index]
    starts = torch.cumsum(count_long, dim=0) - count_long
    cell_start = starts[occupied_index].view(-1, 1)
    for k in range(1, k_max + 1):
        slot = torch.arange(k, device=points.device).view(1, -1)
        rank = torch.div(
            (2 * slot + 1) * occupied_count.view(-1, 1),
            2 * k,
            rounding_mode="floor",
        )
        seeds[:, k - 1, :k] = sorted_points[cell_start + rank]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    for k in range(1, k_max + 1):
        delta[:, k - 1, :k] = (
            seeds[:, k - 1, :k] - cell_center[:, None, :]
        )
    return GridSeedData(seeds, delta, None)


def _viewpoint_seed_bank(points, cell_idx, counts, grid_coord, occupied,
                         metric_origin, mapper, k_max):
    """Build Common-medoid + Additional-range-quantile candidates.

    Candidate row ``K_total - 1`` always starts with the same observed medoid.
    Its remaining ``K_total - 1`` slots use deterministic range quantiles. If a
    cell contains fewer raw points than Additional slots, the integer quantile
    ranks intentionally repeat observed points; independent predictor outputs
    and learned offsets can subsequently separate the Gaussians.
    """
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, k_max, 3))
    if num_cells == 0:
        return GridSeedData(seeds, delta, None)

    count_long = counts.long()
    occupied_count = count_long[occupied_index]
    starts = torch.cumsum(count_long, dim=0) - count_long

    # Observed centroid-medoid, with the same deterministic xyz tie-break as
    # ``_medoid_seed_data``.
    point_sum = points.new_zeros((counts.numel(), 3))
    point_sum.index_add_(0, cell_idx, points)
    point_mean = point_sum / count_long.clamp_min(1).to(points.dtype).unsqueeze(-1)
    distance2 = ((points - point_mean[cell_idx]) ** 2).sum(dim=-1)
    distance_key = torch.round(distance2 * 1.0e6)
    medoid_order = torch.arange(points.shape[0], device=points.device)
    for value in (
        points[:, 2], points[:, 1], points[:, 0], distance_key, cell_idx,
    ):
        medoid_order = medoid_order[
            torch.argsort(value[medoid_order], stable=True)
        ]
    common_seed = points[medoid_order[starts[occupied_index]]]
    seeds[:, :, 0] = common_seed[:, None, :]

    # Additional slots use own-frame range quantiles. Stable sorting yields the
    # deterministic key (cell, range, x, y, z).
    point_range = points.norm(dim=-1)
    range_order = torch.arange(points.shape[0], device=points.device)
    for value in (
        points[:, 2], points[:, 1], points[:, 0], point_range, cell_idx,
    ):
        range_order = range_order[
            torch.argsort(value[range_order], stable=True)
        ]
    sorted_points = points[range_order]
    cell_start = starts[occupied_index].view(-1, 1)
    for total_k in range(2, k_max + 1):
        additional_k = total_k - 1
        slot = torch.arange(additional_k, device=points.device).view(1, -1)
        rank = torch.div(
            (2 * slot + 1) * occupied_count.view(-1, 1),
            2 * additional_k,
            rounding_mode="floor",
        )
        seeds[:, total_k - 1, 1:total_k] = sorted_points[cell_start + rank]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    for total_k in range(1, k_max + 1):
        delta[:, total_k - 1, :total_k] = (
            seeds[:, total_k - 1, :total_k] - cell_center[:, None, :]
        )
    return GridSeedData(seeds, delta, None)


def _medoid_seed_data(points, cell_idx, counts, grid_coord, occupied,
                      metric_origin, mapper, k_max, slots=1):
    """Observed centroid-medoid seeds for each occupied token.

    The seed is the raw point nearest its token's Cartesian raw-point mean.
    This is the same deterministic, observed-point medoid convention used by
    the spherical query path.  It avoids an O(n^2) exact pairwise-medoid cost
    while retaining a surface-supported seed even for sparse cells.

    ``slots`` seeds are emitted, all at that one medoid. Above one they are
    separated only by the independent parameter block each slot owns.
    """
    slots = int(slots)
    if not 1 <= slots <= k_max:
        raise ValueError("medoid seed slots must be in [1, K_max]")
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, 3))
    anchor_k = torch.full(
        (num_cells,), slots, dtype=torch.long, device=points.device
    )
    if num_cells == 0:
        return GridSeedData(seeds, delta, anchor_k)

    count_long = counts.long()
    point_sum = points.new_zeros((counts.numel(), 3))
    point_sum.index_add_(0, cell_idx, points)
    point_mean = point_sum / count_long.clamp_min(1).to(points.dtype).unsqueeze(-1)
    distance2 = ((points - point_mean[cell_idx]) ** 2).sum(dim=-1)
    distance_key = torch.round(distance2 * 1.0e6)

    # Stable least-to-most-significant sorting gives (cell, distance, x, y, z).
    order = torch.arange(points.shape[0], device=points.device)
    for value in (points[:, 2], points[:, 1], points[:, 0], distance_key, cell_idx):
        order = order[torch.argsort(value[order], stable=True)]
    starts = torch.cumsum(count_long, dim=0) - count_long
    medoid = points[order[starts[occupied_index]]]
    seeds[:, :slots] = medoid[:, None, :]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    delta[:, :slots] = seeds[:, :slots] - cell_center[:, None, :]
    return GridSeedData(seeds, delta, anchor_k)


def _token_position_seed_data(token_position, grid_coord, occupied,
                              metric_origin, mapper, k_max, slots=2):
    """Identical token-coordinate seeds per occupied token.

    Their slot identity is the output order of the joint head: each slot owns a
    distinct parameter (including offset) block, so they start at the same
    coordinate but learn independent displacement predictions.
    """
    slots = int(slots)
    if slots < 2:
        raise ValueError("p2g.grid_query.exp=2 needs at least two slots")
    if k_max < slots:
        raise ValueError(f"p2g.grid_query.exp=2 requires K_max >= {slots}")
    num_cells = token_position.shape[0]
    seeds = token_position.new_zeros((num_cells, k_max, 3))
    delta = token_position.new_zeros((num_cells, k_max, 3))
    anchor_k = torch.full(
        (num_cells,), slots, dtype=torch.long, device=token_position.device
    )
    seeds[:, :slots] = token_position[:, None, :]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, token_position.dtype
    )
    delta[:, :slots] = seeds[:, :slots] - cell_center[:, None, :]
    return GridSeedData(seeds, delta, anchor_k)


def _build_seed_data(config, points, cell_idx, counts, grid_coord, occupied,
                     metric_origin, mapper, token_position):
    """Dispatch one occupied frame to the seed rule its config selects."""
    k_max = config.k_max
    if config.learned:
        if config.seed_mode != "range_quantile":
            raise ValueError(
                f"Unsupported learned grid seed_mode={config.seed_mode!r}"
            )
        bank = (
            _viewpoint_seed_bank
            if config.count_mode == "learned_gumbel_viewpt"
            else _r_quantile_seed_bank
        )
        return bank(
            points, cell_idx, counts, grid_coord, occupied, metric_origin,
            mapper, k_max=k_max,
        )
    # A fixed count fills every slot; the grid anchor head instead derives a
    # per-token K from that token's own-frame raw point count.
    slots = k_max if config.fixed_count else None
    if config.exp is None:
        return _r_quantile_seed_data(
            points, cell_idx, counts, grid_coord, occupied, metric_origin,
            mapper, points_per_gaussian=config.points_per_gaussian,
            k_max=k_max, fixed_k=config.fixed_count,
        )
    if config.exp == 1:
        return _medoid_seed_data(
            points, cell_idx, counts, grid_coord, occupied, metric_origin,
            mapper, k_max=k_max, slots=1 if slots is None else slots,
        )
    return _token_position_seed_data(
        token_position, grid_coord, occupied, metric_origin, mapper,
        k_max=k_max, slots=2 if slots is None else slots,
    )

__all__ = ["GridSeedData", "counts_to_variable_k"]
