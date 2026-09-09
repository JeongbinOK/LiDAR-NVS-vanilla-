"""Raw-point aggregation into occupied Utonia token cells."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .grid_geometry import xyz_to_theta_phi_r
from .grid_seeds import GridSeedData, _build_seed_data


@dataclass(frozen=True)
class RawTokenMembership:
    """Own-frame raw points and their compact occupied-token row indices."""

    points_sensor: torch.Tensor
    token_index: torch.Tensor


def split_by_offset(tensor, offset):
    splits = []
    start = 0
    for end in offset:
        end = int(end)
        splits.append(tensor[start:end])
        start = end
    return splits


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
            k_max = seed_config.k_max
            if seed_config.learned:
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
        result = result + (_build_seed_data(
            seed_config, pts, cell_idx, counts, grid_coord, occ,
            metric_origin, mapper, util_pos[occ],
        ),)
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
    mapper, seed_config,
):
    """Aggregate raw points and construct the grid-mode seed tensors.

    See :class:`GridSeedConfig` for what each mode places where.
    """
    return _aggregate_points_to_cells(
        points_xyz, intensity, grid_coord, voxel_feats,
        voxel_coord, metric_origin, mapper,
        seed_config=seed_config,
    )

__all__ = [
    "RawTokenMembership",
    "aggregate_points_to_cells_with_membership",
    "aggregate_points_to_cells_with_seeds",
    "split_by_offset",
]
