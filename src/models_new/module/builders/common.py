"""Shared pieces for anchor/primitive feature builders.

Geometry, grid mapping, and scatter helpers shared by the config-selected
builders. The per-cell intensity statistics are
``[int_mean, int_var, theta, phi, r]``.
"""
from __future__ import annotations

import math

import torch


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


def aggregate_points_to_cells(points_xyz, intensity, grid_coord, voxel_feats,
                              voxel_coord, metric_origin, mapper):
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
        return (points_xyz.new_zeros((0, 3)), voxel_feats.new_zeros((0, D)),
                points_xyz.new_zeros((0, 5)),
                torch.zeros((V,), dtype=torch.bool, device=device),
                torch.zeros((0,), dtype=torch.long, device=device))

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
    return util_pos[occ], voxel_feats[occ], int5[occ], occ, counts[occ].long()
