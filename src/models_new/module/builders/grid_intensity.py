"""Grid-intensity builder (no spherical anchor).

For each frame the raw points are scattered into the Utonia bottleneck cells they
fall in (0.8 m, same grid as features['grid_coord']); per cell we aggregate
[mean_i, var_i, theta_mean, phi_mean, r_mean]. Each occupied cell is one
primitive: position = Utonia cell position, utonia_feat = that cell's feature,
intensity = IntensityEncoder([mean_i, var_i, theta, phi, r]).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import IntensityEncoder, aggregate_points_to_cells, split_by_offset


class GridIntensityBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.intensity_encoder = IntensityEncoder(
            cfg.int_proj.in_dim, cfg.int_proj.out_dim, r_far=float(getattr(cfg, "r_far", 70.0))
        )

    def forward(self, lidar_points, offset, pose, features, ptv3_input, mapper):
        device = features["feat"].device

        grid_coord_list = split_by_offset(features["grid_coord"], features["offset"])
        feat_list = split_by_offset(features["feat"], features["offset"])
        if "coord" in features:
            coord_list = split_by_offset(features["coord"], features["offset"])
        else:
            coord_list = [None] * len(feat_list)

        in_coord = ptv3_input["coord"].to(device)
        in_gc = ptv3_input["grid_coord"].to(device)
        in_off = ptv3_input["offset"].to(device)
        in_coord_list = split_by_offset(in_coord, in_off)
        in_gc_list = split_by_offset(in_gc, in_off)
        origins = [mapper.metric_origin(c, g) for c, g in zip(in_coord_list, in_gc_list)]

        pts_list = split_by_offset(lidar_points.to(device), offset)

        pos_list, ufeat_list, ifeat_list = [], [], []
        for i in range(len(feat_list)):
            pts = pts_list[i]
            xyz = pts[:, :3]
            inten = pts[:, 3]
            upos, ufeat, int5 = aggregate_points_to_cells(
                xyz, inten, grid_coord_list[i], feat_list[i],
                coord_list[i], origins[i], mapper,
            )
            ifeat = self.intensity_encoder(int5)
            pos_list.append(upos)
            ufeat_list.append(ufeat)
            ifeat_list.append(ifeat)
        return pos_list, ufeat_list, ifeat_list
