"""Spherical-anchor builder (original pipeline, modularised).

Spherical voxelizer -> per-frame anchors + (mean_i, var_i); Utonia features are
trilinearly interpolated onto the anchor positions; intensity 5D is
[mean_i, var_i, theta, phi, r] from the anchor position.
Returns the mode-agnostic triple consumed by Point2Gaus.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...utils.coord import Voxelizer
from .common import (
    IntensityEncoder,
    split_by_offset,
    utonia_cond_trilinear,
    xyz_to_theta_phi_r,
)


class SphericalAnchorBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.voxelizer = Voxelizer(cfg=cfg, max_frames=4)
        self.intensity_encoder = IntensityEncoder(cfg.int_proj.in_dim, cfg.int_proj.out_dim)

    def forward(self, lidar_points, offset, pose, features, ptv3_input, mapper):
        device = features["feat"].device

        vox = self.voxelizer(lidar_points, offset, pose, mode="sphere")
        anchor_list = vox["anchor_points"]
        mean_i_list = vox["mean_i"]
        var_i_list = vox["var_i"]

        grid_coord_list = split_by_offset(features["grid_coord"], features["offset"])
        feat_list = split_by_offset(features["feat"], features["offset"])

        in_coord = ptv3_input["coord"].to(device)
        in_gc = ptv3_input["grid_coord"].to(device)
        in_off = ptv3_input["offset"].to(device)
        in_coord_list = split_by_offset(in_coord, in_off)
        in_gc_list = split_by_offset(in_gc, in_off)
        origins = [mapper.metric_origin(c, g) for c, g in zip(in_coord_list, in_gc_list)]

        pos_list, ufeat_list, ifeat_list = [], [], []
        for i in range(len(anchor_list)):
            anchor = anchor_list[i].to(device)
            ufeat, keep = utonia_cond_trilinear(
                anchor, grid_coord_list[i], feat_list[i], origins[i], mapper
            )
            valid_anchor = anchor[keep]
            mean_i = mean_i_list[i].to(device)[keep]
            var_i = var_i_list[i].to(device)[keep]
            theta_phi_r = xyz_to_theta_phi_r(valid_anchor)
            int5 = torch.cat([mean_i.unsqueeze(-1), var_i.unsqueeze(-1), theta_phi_r], dim=-1)
            ifeat = self.intensity_encoder(int5)

            pos_list.append(valid_anchor)
            ufeat_list.append(ufeat)
            ifeat_list.append(ifeat)
        return pos_list, ufeat_list, ifeat_list
