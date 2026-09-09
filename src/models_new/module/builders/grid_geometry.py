"""Metric geometry and Utonia grid mapping."""
from __future__ import annotations

import torch


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

__all__ = ["UtoniaGridMapper", "xyz_to_theta_phi_r"]

