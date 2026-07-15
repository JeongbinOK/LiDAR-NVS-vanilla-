"""Shared occupied Cartesian-token builder for spherical and grid heads.

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
``raw_count`` lists. Grid mode additionally returns padded own-frame range-quantile
seeds and their offsets from geometric cell centers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .common import (
    GridSeedData,
    aggregate_points_to_cells,
    aggregate_points_to_cells_with_seeds,
    split_by_offset,
)
from .intensity_encoder import (
    build_intensity_encoder,
    encode_intensity_features,
)


@dataclass(frozen=True)
class OccupiedTokenBatch:
    """Per-frame occupied-token tensors shared by both downstream anchor modes."""

    positions: list[torch.Tensor]
    utonia_features: list[torch.Tensor]
    intensity_features: list[torch.Tensor]
    grid_coords: list[torch.Tensor]
    raw_counts: list[torch.Tensor]
    grid_seeds: Optional[list[GridSeedData]]


class OccupiedGridTokenBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.anchor_mode = str(getattr(cfg, "anchor_mode", "spherical")).lower()
        self.grid_seed_config = None
        if self.anchor_mode == "grid":
            grid_query = getattr(cfg, "grid_query", None)
            if grid_query is None:
                raise ValueError("p2g.grid_query config block is required for anchor_mode='grid'")
            k_max = getattr(grid_query, "K_max", None)
            points_per_gaussian = getattr(grid_query, "points_per_gaussian", None)
            if k_max is None or points_per_gaussian is None:
                raise ValueError(
                    "p2g.grid_query.K_max and points_per_gaussian are required for grid seeds"
                )
            k_max = int(k_max)
            points_per_gaussian = int(points_per_gaussian)
            if k_max <= 0:
                raise ValueError("p2g.grid_query.K_max must be positive")
            if points_per_gaussian <= 0:
                raise ValueError("p2g.grid_query.points_per_gaussian must be positive")
            self.grid_seed_config = (points_per_gaussian, k_max)
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
        seed_data_list = [] if self.grid_seed_config is not None else None
        for i in range(len(feat_list)):
            pts = pts_list[i]
            xyz = pts[:, :3]
            inten = pts[:, 3]
            if self.grid_seed_config is None:
                upos, ufeat, int5, occ, raw_count = aggregate_points_to_cells(
                    xyz, inten, grid_coord_list[i], feat_list[i],
                    coord_list[i], origins[i], mapper,
                )
            else:
                points_per_gaussian, k_max = self.grid_seed_config
                result = aggregate_points_to_cells_with_seeds(
                    xyz, inten, grid_coord_list[i], feat_list[i],
                    coord_list[i], origins[i], mapper,
                    points_per_gaussian, k_max,
                )
                upos, ufeat, int5, occ, raw_count, seed_data = result
                seed_data_list.append(seed_data)
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
        return OccupiedTokenBatch(
            positions=pos_list,
            utonia_features=ufeat_list,
            intensity_features=ifeat_list,
            grid_coords=gc_list,
            raw_counts=raw_count_list,
            grid_seeds=seed_data_list,
        )
