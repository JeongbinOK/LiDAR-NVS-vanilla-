"""Shared pieces for anchor/primitive feature builders.

Geometry, grid mapping, and scatter helpers shared by the config-selected
builders. The per-cell intensity statistics are
``[int_mean, int_var, theta, phi, r]``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GridSeedData:
    """Own-frame variable-K seed geometry aligned with occupied token rows."""

    seed_sensor: torch.Tensor
    delta_sensor: torch.Tensor
    anchor_k: torch.Tensor


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


def encode_ray_meta(tpr, r_far):
    """(...,3)=[theta, phi, r] -> (...,4) encoded ray meta for K/V embeddings.

    Follows the same convention as ``IntensityMLPEncoder``: theta is scaled by
    pi/2, phi is mapped onto the unit circle
    (sin, cos) to remove the +-pi azimuth wraparound a bare Linear would see,
    and r is log1p-compressed/linearized before being normalized by log1p(r_far).
    r is clamp_min(0.0)'d before log1p.

        [theta/(pi/2), sin(phi), cos(phi), log1p(r)/log1p(r_far)]

    tpr   : (..., 3) [theta, phi, r], own-sensor-origin ray angles/range.
    r_far : python float/scalar, the normalizing max range (e.g. cfg.r_far).
    return: (..., 4).
    """
    theta, phi, r = tpr.unbind(dim=-1)
    theta_n = theta / (math.pi / 2.0)
    log_r_n = torch.log1p(r.clamp_min(0.0)) / math.log1p(float(r_far))
    return torch.stack([theta_n, torch.sin(phi), torch.cos(phi), log_r_n], dim=-1)


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

    cell_center = (
        (grid_coord[occupied_index].to(points.dtype) + 0.5)
        * mapper.feature_grid_size
        + metric_origin.to(device=points.device, dtype=points.dtype).unsqueeze(0)
    )
    delta[valid] = seeds[valid] - cell_center[:, None, :].expand_as(seeds)[valid]
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
            k_max = int(seed_config[1])
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
        seed_data = _r_quantile_seed_data(
            pts, cell_idx, counts, grid_coord, occ, metric_origin, mapper,
            points_per_gaussian=int(seed_config[0]), k_max=int(seed_config[1]),
        )
        result = result + (seed_data,)
    return result


def aggregate_points_to_cells(points_xyz, intensity, grid_coord, voxel_feats,
                              voxel_coord, metric_origin, mapper):
    """Aggregate raw points without constructing grid-mode Gaussian seeds."""
    return _aggregate_points_to_cells(
        points_xyz, intensity, grid_coord, voxel_feats,
        voxel_coord, metric_origin, mapper,
    )


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
    mapper, points_per_gaussian, k_max,
):
    """Aggregate raw points and construct padded grid-mode seed tensors."""
    return _aggregate_points_to_cells(
        points_xyz, intensity, grid_coord, voxel_feats,
        voxel_coord, metric_origin, mapper,
        seed_config=(points_per_gaussian, k_max),
    )
