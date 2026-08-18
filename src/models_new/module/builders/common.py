"""Shared pieces for anchor/primitive feature builders.

Geometry, grid mapping, and scatter helpers shared by the config-selected
builders. The per-cell intensity statistics are
``[int_mean, int_var, theta, phi, r]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


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


@dataclass(frozen=True)
class RawTokenMembership:
    """Own-frame raw points and their compact occupied-token row indices."""

    points_sensor: torch.Tensor
    token_index: torch.Tensor


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


def split_by_offset(tensor, offset):
    splits = []
    start = 0
    for end in offset:
        end = int(end)
        splits.append(tensor[start:end])
        start = end
    return splits


def xyz_to_theta_phi_r(xyz):
    """(...,3) sensor-frame xyz -> (...,3) [theta, phi, r] about the sensor origin."""
    r = xyz.norm(dim=-1).clamp_min(1e-6)
    phi = torch.atan2(xyz[..., 1], xyz[..., 0])              # [-pi, pi]
    theta = torch.asin((xyz[..., 2] / r).clamp(-1.0, 1.0))   # [-pi/2, pi/2]
    return torch.stack([theta, phi, r], dim=-1)


class UtoniaGridMapper:
    """Maps metric (sensor-frame) points to the Utonia bottleneck feature grid,
    i.e. the same integer cell index space as features['grid_coord'].

    feature_grid_size (metric) = input_grid_size / coord_scale * stride_factor.
    With the pretrained Utonia (input GridSample 0.01 on coords scaled by 0.2,
    stride (2,2,2,2) -> 16x) this is 0.01/0.2*16 = 0.8 m.
    """

    def __init__(self, coord_scale: float, input_grid_size: float, stride_factor: int):
        self.coord_scale = coord_scale
        self.input_grid_size = input_grid_size
        self.stride_factor = stride_factor
        self.feature_grid_size = input_grid_size / coord_scale * stride_factor

    @staticmethod
    def _input_min_grid(input_coord, input_grid_coord, input_grid_size):
        raw_grid = torch.floor(input_coord / input_grid_size).to(dtype=torch.long)
        cand = raw_grid - input_grid_coord.to(device=raw_grid.device, dtype=torch.long)
        return cand.median(dim=0).values

    def metric_origin(self, input_coord, input_grid_coord):
        """Metric position of the grid's min corner for one frame (recovered from
        the dataloader's per-frame coord.min subtraction)."""
        min_grid = self._input_min_grid(input_coord, input_grid_coord, self.input_grid_size)
        return min_grid.to(dtype=input_coord.dtype) * (self.input_grid_size / self.coord_scale)

    def to_feature_grid(self, points, metric_origin):
        """metric points (M,3) -> float bottleneck-grid coords (M,3)."""
        metric_origin = metric_origin.to(device=points.device, dtype=points.dtype)
        return (points - metric_origin.unsqueeze(0)) / self.feature_grid_size


def _cell_centers(grid_coord, occupied, metric_origin, mapper, dtype):
    """Geometric centers for the compact occupied-token row order."""
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    return (
        (grid_coord[occupied_index].to(dtype) + 0.5) * mapper.feature_grid_size
        + metric_origin.to(device=grid_coord.device, dtype=dtype).unsqueeze(0)
    )


def _r_quantile_seed_data(points, cell_idx, counts, grid_coord, occupied,
                          metric_origin, mapper, points_per_gaussian, k_max):
    """Select deterministic range-quantile points for each occupied cell."""
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, 3))
    occupied_count = counts.long()[occupied_index]
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
                      metric_origin, mapper, k_max):
    """One observed centroid-medoid seed per occupied token.

    The seed is the raw point nearest its token's Cartesian raw-point mean.
    This is the same deterministic, observed-point medoid convention used by
    the spherical query path.  It avoids an O(n^2) exact pairwise-medoid cost
    while retaining a surface-supported seed even for sparse cells.
    """
    occupied_index = occupied.nonzero(as_tuple=True)[0]
    num_cells = occupied_index.numel()
    seeds = points.new_zeros((num_cells, k_max, 3))
    delta = points.new_zeros((num_cells, k_max, 3))
    anchor_k = torch.ones(num_cells, dtype=torch.long, device=points.device)
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
    seeds[:, 0] = points[order[starts[occupied_index]]]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, points.dtype
    )
    delta[:, 0] = seeds[:, 0] - cell_center
    return GridSeedData(seeds, delta, anchor_k)


def _token_position_seed_data(token_position, grid_coord, occupied,
                              metric_origin, mapper, k_max):
    """Two identical token-coordinate seeds per occupied token.

    Their slot identity is the output order of the K=2 joint head: slot 0 and
    slot 1 own distinct parameter (including offset) blocks, so they start at
    the same coordinate but learn independent displacement predictions.
    """
    if k_max < 2:
        raise ValueError("p2g.grid_query.exp=2 requires K_max >= 2")
    num_cells = token_position.shape[0]
    seeds = token_position.new_zeros((num_cells, k_max, 3))
    delta = token_position.new_zeros((num_cells, k_max, 3))
    anchor_k = torch.full((num_cells,), 2, dtype=torch.long, device=token_position.device)
    seeds[:, :2] = token_position[:, None, :]

    cell_center = _cell_centers(
        grid_coord, occupied, metric_origin, mapper, token_position.dtype
    )
    delta[:, :2] = seeds[:, :2] - cell_center[:, None, :]
    return GridSeedData(seeds, delta, anchor_k)


def _aggregate_points_to_cells(points_xyz, intensity, grid_coord, voxel_feats,
                               voxel_coord, metric_origin, mapper, seed_config=None,
                               return_membership=False):
    """Scatter raw points into the Utonia bottleneck cells they fall in, and pair
    each occupied cell with its Utonia feature (exact cell match, no trilinear).

    Returns (positions[M,3] metric, utonia_feat[M,C_stage], intensity5d[M,5],
    occ[V] bool, raw_count[M] long) for the Utonia cells that received >=1 point.
    positions = Utonia cell position; the intensity (theta,phi,r) come from the
    per-cell MEAN point position. ``occ`` (over all V grid_coord rows) lets callers
    recover the token grid_coords (grid_coord[occ]) for aligning a separate
    intensity encoder. ``raw_count`` is the own-frame point count and deliberately
    excludes temporally matched observations.
    """
    device = voxel_feats.device
    V = grid_coord.shape[0]
    D = voxel_feats.shape[1]
    if V == 0 or points_xyz.shape[0] == 0:
        result = (
            points_xyz.new_zeros((0, 3)), voxel_feats.new_zeros((0, D)),
            points_xyz.new_zeros((0, 5)),
            torch.zeros((V,), dtype=torch.bool, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )
        if return_membership:
            empty_membership = RawTokenMembership(
                points_sensor=points_xyz.new_zeros((0, 3)),
                token_index=torch.zeros((0,), dtype=torch.long, device=device),
            )
            result = result + (empty_membership,)
        if seed_config is not None:
            count_mode, _points_per_gaussian, k_max, _exp, _seed_mode = seed_config
            k_max = int(k_max)
            if count_mode in ("learned_gumbel", "learned_gumbel_viewpt"):
                empty_seed = points_xyz.new_zeros((0, k_max, k_max, 3))
                empty_k = None
            else:
                empty_seed = points_xyz.new_zeros((0, k_max, 3))
                empty_k = torch.zeros((0,), dtype=torch.long, device=device)
            result = result + (GridSeedData(empty_seed, empty_seed.clone(), empty_k),)
        return result

    grid_coord = grid_coord.to(device=device, dtype=torch.long)
    pcell = torch.floor(mapper.to_feature_grid(points_xyz, metric_origin)).long()   # (N,3)

    grid_max = grid_coord.max(dim=0).values
    dims = (grid_max + 1).clamp_min(1)
    stride_x = dims[1] * dims[2]
    stride_y = dims[2]

    ukeys = grid_coord[:, 0] * stride_x + grid_coord[:, 1] * stride_y + grid_coord[:, 2]
    sorted_keys, sorted_order = torch.sort(ukeys)

    in_bounds = ((pcell >= 0) & (pcell <= grid_max.view(1, 3))).all(dim=-1)
    pcell_c = pcell.clamp_min(0)
    pkeys = pcell_c[:, 0] * stride_x + pcell_c[:, 1] * stride_y + pcell_c[:, 2]
    pos = torch.searchsorted(sorted_keys, pkeys)
    pos_clamped = pos.clamp(max=sorted_keys.numel() - 1)
    found = (pos < sorted_keys.numel()) & (sorted_keys[pos_clamped] == pkeys) & in_bounds
    puidx = sorted_order[pos_clamped]   # (N,) utonia-cell index per point

    cell_idx = puidx[found]
    inten = intensity[found].to(dtype=points_xyz.dtype)
    pts = points_xyz[found]

    counts = torch.zeros(V, device=device, dtype=points_xyz.dtype)
    counts.scatter_add_(0, cell_idx, torch.ones_like(inten))
    sum_i = torch.zeros(V, device=device, dtype=points_xyz.dtype)
    sum_i.scatter_add_(0, cell_idx, inten)
    sum_i2 = torch.zeros(V, device=device, dtype=points_xyz.dtype)
    sum_i2.scatter_add_(0, cell_idx, inten * inten)
    sum_xyz = torch.zeros((V, 3), device=device, dtype=points_xyz.dtype)
    sum_xyz.index_add_(0, cell_idx, pts)

    occ = counts > 0
    cnt = counts.clamp_min(1.0)
    mean_i = sum_i / cnt
    var_i = (sum_i2 / cnt - mean_i * mean_i).clamp_min(0.0)
    # option B: per-cell MEAN position -> spherical (matches the spherical builder,
    # and avoids phi/atan2 wraparound from averaging per-point angles across the
    # +/-pi azimuth seam).
    mean_xyz = sum_xyz / cnt.unsqueeze(-1)
    mean_tpr = xyz_to_theta_phi_r(mean_xyz)

    if voxel_coord is not None:
        util_pos = voxel_coord.to(device=device, dtype=points_xyz.dtype) / mapper.coord_scale
    else:
        util_pos = (grid_coord.to(points_xyz.dtype) + 0.5) * mapper.feature_grid_size \
            + metric_origin.to(device=device, dtype=points_xyz.dtype).unsqueeze(0)

    int5 = torch.cat([mean_i.unsqueeze(-1), var_i.unsqueeze(-1), mean_tpr], dim=-1)
    result = util_pos[occ], voxel_feats[occ], int5[occ], occ, counts[occ].long()
    if return_membership:
        # ``cell_idx`` addresses the original Utonia grid rows.  Downstream
        # spherical code consumes only occupied tokens, so remap every retained
        # raw point to that compact row order (the same order as result[0:3]).
        full_to_compact = torch.full(
            (V,), -1, dtype=torch.long, device=device
        )
        full_to_compact[occ] = torch.cumsum(occ.to(torch.long), dim=0)[occ] - 1
        result = result + (RawTokenMembership(
            points_sensor=pts,
            token_index=full_to_compact[cell_idx],
        ),)
    if seed_config is not None:
        count_mode, points_per_gaussian, k_max, exp, seed_mode = seed_config
        if count_mode == "learned_gumbel":
            if seed_mode != "range_quantile":
                raise ValueError(
                    f"Unsupported learned grid seed_mode={seed_mode!r}"
                )
            seed_data = _r_quantile_seed_bank(
                pts, cell_idx, counts, grid_coord, occ, metric_origin, mapper,
                k_max=int(k_max),
            )
        elif count_mode == "learned_gumbel_viewpt":
            if seed_mode != "range_quantile":
                raise ValueError(
                    f"Unsupported viewpoint grid seed_mode={seed_mode!r}"
                )
            seed_data = _viewpoint_seed_bank(
                pts, cell_idx, counts, grid_coord, occ, metric_origin, mapper,
                k_max=int(k_max),
            )
        elif exp is None:
            seed_data = _r_quantile_seed_data(
                pts, cell_idx, counts, grid_coord, occ, metric_origin, mapper,
                points_per_gaussian=int(points_per_gaussian), k_max=int(k_max),
            )
        elif exp == 1:
            seed_data = _medoid_seed_data(
                pts, cell_idx, counts, grid_coord, occ, metric_origin, mapper,
                k_max=int(k_max),
            )
        elif exp == 2:
            seed_data = _token_position_seed_data(
                util_pos[occ], grid_coord, occ, metric_origin, mapper,
                k_max=int(k_max),
            )
        else:
            raise ValueError(f"Unsupported grid seed experiment exp={exp!r}")
        result = result + (seed_data,)
    return result


def aggregate_points_to_cells_with_membership(
    points_xyz, intensity, grid_coord, voxel_feats, voxel_coord, metric_origin,
    mapper,
):
    """Aggregate raw points and preserve raw-to-occupied-token membership."""
    return _aggregate_points_to_cells(
        points_xyz, intensity, grid_coord, voxel_feats,
        voxel_coord, metric_origin, mapper,
        return_membership=True,
    )


def aggregate_points_to_cells_with_seeds(
    points_xyz, intensity, grid_coord, voxel_feats, voxel_coord, metric_origin,
    mapper, points_per_gaussian, k_max, exp=None, count_mode="legacy",
    seed_mode=None, return_membership=False,
):
    """Aggregate raw points and construct padded grid-mode seed tensors.

    In ``count_mode="legacy"``, ``exp=None`` retains raw-count-driven
    range-quantile slots, ``exp=1`` emits one raw-point medoid per token, and
    ``exp=2`` emits two identical token-coordinate seeds. In
    learned count modes, legacy ``exp`` and ``points_per_gaussian`` are ignored.
    ``learned_gumbel`` builds the full range-quantile candidate bank;
    ``learned_gumbel_viewpt`` builds Common-medoid plus Additional-range-quantile
    candidates.
    """
    count_mode = str(count_mode).lower()
    if count_mode not in (
        "legacy", "learned_gumbel", "learned_gumbel_viewpt",
    ):
        raise ValueError(
            "grid count_mode must be 'legacy', 'learned_gumbel', or "
            "'learned_gumbel_viewpt'"
        )
    if count_mode in ("learned_gumbel", "learned_gumbel_viewpt"):
        seed_mode = "range_quantile" if seed_mode is None else str(seed_mode).lower()
    return _aggregate_points_to_cells(
        points_xyz, intensity, grid_coord, voxel_feats,
        voxel_coord, metric_origin, mapper,
        seed_config=(count_mode, points_per_gaussian, k_max, exp, seed_mode),
        return_membership=bool(return_membership),
    )
