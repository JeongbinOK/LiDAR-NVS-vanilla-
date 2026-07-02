"""Shared pieces for anchor/primitive feature builders.

Both builders return, per frame, the same triple
    (positions[Mi,3], utonia_feat[Mi,576], intensity_feat[Mi,64])
so the Point2Gaus tail (agg_mlp -> time_agg -> gs_predictor -> assemble) is
mode-agnostic. The intensity 5D is [int_mean, int_var, theta, phi, r].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


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


def xyz_to_sph(xyz):
    return xyz_to_theta_phi_r(xyz)


class IntensityEncoder(nn.Module):
    """[mean_i, var_i, theta, phi, r] (5D) -> out_dim.

    Less naive than a bare Linear, because this is the ONLY reflectance path to the
    intensity-SH head (Utonia gets no intensity input):
    - Inputs are conditioned before the first layer (-> 6D encoded):
        mean_i, var_i        : already in [0,1], passed through
        theta -> theta/(pi/2): ~[-1,1]
        phi   -> (sin, cos)  : maps the line onto a circle, removing the +-pi azimuth
                               wraparound (a discontinuity a bare Linear would see)
        r     -> log1p(r)/log1p(r_far): compresses 0.2..70 m and *linearizes* the
                               power-law range falloff (log turns 1/r^2 into a line),
                               so the range dependence is easy to learn. Monotonic =>
                               no range information is lost.
    - A 2-layer MLP (+SiLU) can form nonlinear input cross-terms (e.g. range/angle
      compensation of intensity) that a single linear map provably cannot.
    - Output LayerNorm keeps the stream ~unit-scale for the fusion concat.
    """

    def __init__(self, in_dim: int = 5, out_dim: int = 64, hidden: int | None = None,
                 r_far: float = 70.0):
        super().__init__()
        assert in_dim == 5, "IntensityEncoder expects 5D [mean_i, var_i, theta, phi, r]"
        h = int(hidden) if hidden else out_dim
        self._log_r_far = math.log1p(float(r_far))
        self.net = nn.Sequential(
            nn.Linear(6, h),          # encoded input width = 6
            nn.SiLU(),
            nn.Linear(h, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def _encode(self, x5):
        mean_i, var_i, theta, phi, r = x5.unbind(dim=-1)
        theta_n = theta / (math.pi / 2.0)
        log_r_n = torch.log1p(r.clamp_min(0.0)) / self._log_r_far
        return torch.stack(
            [mean_i, var_i, theta_n, torch.sin(phi), torch.cos(phi), log_r_n], dim=-1
        )

    def forward(self, x5):
        if x5.shape[0] == 0:
            return x5.new_zeros((0, self.norm.normalized_shape[0]))
        return self.norm(self.net(self._encode(x5)))


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


def utonia_cond_trilinear(anchor_points, grid_coord, voxel_feats, metric_origin, mapper):
    """Trilinear (8-neighbour) interpolation of Utonia bottleneck features onto
    arbitrary metric anchor positions. Returns (feat[M_kept, D], keep_mask[M]).
    (Moved verbatim from Point2Gaus.utonia_cond; uses mapper.to_feature_grid.)"""
    M = anchor_points.shape[0]
    D = voxel_feats.shape[1]
    device = anchor_points.device
    if M == 0 or voxel_feats.shape[0] == 0:
        return voxel_feats.new_zeros((0, D)), torch.zeros((M,), dtype=torch.bool, device=device)

    grid_coord = grid_coord.to(device=device, dtype=torch.long)
    anchor_grid_f = mapper.to_feature_grid(anchor_points, metric_origin)
    base_grid = torch.floor(anchor_grid_f).long()
    frac = (anchor_grid_f - base_grid.to(anchor_points.dtype)).clamp(0.0, 1.0)

    offsets = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=device, dtype=torch.long,
    )
    neighbor_grids = base_grid[:, None, :] + offsets[None, :, :]

    grid_max = grid_coord.max(dim=0).values
    in_bounds = ((neighbor_grids >= 0) & (neighbor_grids <= grid_max.view(1, 1, 3))).all(dim=-1)
    safe_neighbor_grids = neighbor_grids.clamp(min=0)

    dims = (grid_max + 1).clamp_min(1)
    stride_x = dims[1] * dims[2]
    stride_y = dims[2]

    voxel_keys = grid_coord[:, 0] * stride_x + grid_coord[:, 1] * stride_y + grid_coord[:, 2]
    sorted_keys, sorted_order = torch.sort(voxel_keys)
    neighbor_keys = (
        safe_neighbor_grids[:, :, 0] * stride_x
        + safe_neighbor_grids[:, :, 1] * stride_y
        + safe_neighbor_grids[:, :, 2]
    )

    flat_keys = neighbor_keys.reshape(-1)
    pos = torch.searchsorted(sorted_keys, flat_keys)
    pos_clamped = pos.clamp(max=sorted_keys.numel() - 1)
    found = (pos < sorted_keys.numel()) & (sorted_keys[pos_clamped] == flat_keys)
    found = found & in_bounds.reshape(-1)
    neighbor_vi = sorted_order[pos_clamped].reshape(M, 8)
    valid_mask = found.reshape(M, 8)

    offsets_f = offsets.to(dtype=anchor_points.dtype)
    wx = torch.where(offsets_f[:, 0].view(1, 8) > 0, frac[:, 0:1], 1.0 - frac[:, 0:1])
    wy = torch.where(offsets_f[:, 1].view(1, 8) > 0, frac[:, 1:2], 1.0 - frac[:, 1:2])
    wz = torch.where(offsets_f[:, 2].view(1, 8) > 0, frac[:, 2:3], 1.0 - frac[:, 2:3])
    weights = wx * wy * wz
    weights = weights * valid_mask.to(dtype=weights.dtype)
    weight_sum = weights.sum(dim=-1)
    keep_mask = weight_sum > 1e-8

    safe_vi = neighbor_vi.clamp(min=0)
    out_all = (weights[:, :, None] * voxel_feats[safe_vi]).sum(dim=1)
    out_all = out_all / weight_sum.clamp_min(1e-8).unsqueeze(-1)
    return out_all[keep_mask], keep_mask


def aggregate_points_to_cells(points_xyz, intensity, grid_coord, voxel_feats,
                              voxel_coord, metric_origin, mapper):
    """Scatter raw points into the Utonia bottleneck cells they fall in, and pair
    each occupied cell with its Utonia feature (exact cell match, no trilinear).

    Returns (positions[M,3] metric, utonia_feat[M,576], intensity5d[M,5]) for the
    Utonia cells that received >=1 point. positions = Utonia cell position; the
    intensity (theta,phi,r) come from the per-cell MEAN point position.
    """
    device = voxel_feats.device
    V = grid_coord.shape[0]
    D = voxel_feats.shape[1]
    if V == 0 or points_xyz.shape[0] == 0:
        return (points_xyz.new_zeros((0, 3)), voxel_feats.new_zeros((0, D)),
                points_xyz.new_zeros((0, 5)))

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
    return util_pos[occ], voxel_feats[occ], int5[occ]
