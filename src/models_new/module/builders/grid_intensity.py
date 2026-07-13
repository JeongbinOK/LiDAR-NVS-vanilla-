"""Grid-intensity builder (no spherical anchor).

For each frame the raw points are scattered into the Utonia bottleneck cells they
fall in (0.4/0.8 m, same grid as features['grid_coord']); each occupied cell is
one primitive: position = Utonia cell position, utonia_feat = that cell's feature.

Intensity per primitive comes from one of three encoders (config
``p2g.intensity_encoder.type``; absent -> legacy ``intensity_sparse.enable`` switch):
- "ptv3": a mini ``PointTransformerV3`` mirroring the Utonia backbone
  (``IntensityPTv3Encoder``), run once on the full batch. Its stage output aligns
  row-for-row with the Utonia token grid (same GridPooling), so intensity is read
  off with the same ``occ`` mask as the Utonia feature -- no gather. out_dim = last
  channel (no projection head).
- "sparse": a SubMConv3d encoder mirroring the Utonia grid hierarchy
  (``IntensitySparseEncoder``); reflectance gets a multi-scale conv receptive field
  and is gathered onto the token grid by grid_coord. The per-input-voxel intensity
  is ``ptv3_input['strength']`` -- the intensity of the SAME point GridSample kept
  for Utonia (dataloader ``keep_strength=True``), aligned to the encoder coord.
- "mlp" (fallback): ``IntensityMLPEncoder`` on the aggregated
  [mean_i, var_i, theta, phi, r] 5D.

All modes also return per-frame ``occ_gc`` (occupied token grid coordinates) and
``raw_count`` lists.  ``raw_count`` is the number of own-frame raw points in each
token and is used by grid mode to choose its variable Gaussian count.
"""
from __future__ import annotations

import torch.nn as nn

from .common import aggregate_points_to_cells, split_by_offset
from .intensity_encoder import (
    build_intensity_encoder,
    encode_intensity_features,
)


class GridIntensityBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        (
            self.intensity_mode,
            self.intensity_encoder,
            self.intensity_out_dim,
        ) = build_intensity_encoder(cfg)

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

        pos_list, ufeat_list, int5_list, occ_list, gc_list = [], [], [], [], []
        raw_count_list = []
        for i in range(len(feat_list)):
            pts = pts_list[i]
            xyz = pts[:, :3]
            inten = pts[:, 3]
            upos, ufeat, int5, occ, raw_count = aggregate_points_to_cells(
                xyz, inten, grid_coord_list[i], feat_list[i],
                coord_list[i], origins[i], mapper,
            )
            occ_gc = grid_coord_list[i].to(device)[occ]
            pos_list.append(upos)
            ufeat_list.append(ufeat)
            int5_list.append(int5)
            occ_list.append(occ)
            gc_list.append(occ_gc)
            raw_count_list.append(raw_count)

        ifeat_list = encode_intensity_features(
            self.intensity_mode,
            self.intensity_encoder,
            ptv3_input=ptv3_input,
            input_coord_list=in_coord_list,
            input_grid_coord_list=in_gc_list,
            token_grid_coord_list=grid_coord_list,
            occupied_masks=occ_list,
            occupied_grid_coord_list=gc_list,
            cell_statistics_list=int5_list,
        )
        return pos_list, ufeat_list, ifeat_list, gc_list, raw_count_list
