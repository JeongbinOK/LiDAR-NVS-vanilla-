"""Shared occupied Cartesian-token builder for spherical and grid heads.

For each frame the raw points are scattered into the Utonia bottleneck cells they
fall in (0.4/0.8 m, same grid as features['grid_coord']); each occupied cell is
one primitive: position = Utonia cell position, utonia_feat = that cell's feature.

With ``p2g.utonia_lora.enable=true``, aligned intensity is already part of the
Utonia XYZI input and this builder returns the occupied Utonia tokens directly.
No separate intensity encoder is constructed.  In the legacy path, intensity per
primitive comes from one of three encoders (config
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
``raw_count`` lists. Grid mode additionally returns either the legacy padded
own-frame seed set or a K-specific learned-count seed bank. Viewpoint routing
uses one medoid Common seed plus range-quantile Additional seeds. Every delta is
measured from the geometric cell center.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .common import (
    GridSeedData,
    RawTokenMembership,
    aggregate_points_to_cells_with_membership,
    aggregate_points_to_cells_with_seeds,
    split_by_offset,
)
from .intensity_encoder import (
    build_intensity_encoder,
    encode_intensity_features,
)
from ..grid_router_pseudo_gt import (
    GridRouterPseudoGT,
    grid_inverse_range_pseudo_gt,
)


@dataclass(frozen=True)
class OccupiedTokenBatch:
    """Per-frame occupied-token tensors shared by both downstream anchor modes."""

    positions: list[torch.Tensor]
    utonia_features: list[torch.Tensor]
    intensity_features: Optional[list[torch.Tensor]]
    grid_coords: list[torch.Tensor]
    raw_counts: list[torch.Tensor]
    grid_seeds: Optional[list[GridSeedData]]
    raw_memberships: Optional[list[RawTokenMembership]]
    router_guides: Optional[list[GridRouterPseudoGT]]


class OccupiedGridTokenBuilder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.anchor_mode = str(getattr(cfg, "anchor_mode", "spherical")).lower()
        self.grid_seed_config = None
        self.router_guide_kwargs = None
        if self.anchor_mode == "grid":
            grid_query = getattr(cfg, "grid_query", None)
            if grid_query is None:
                raise ValueError("p2g.grid_query config block is required for anchor_mode='grid'")
            count_mode = str(
                getattr(grid_query, "count_mode", "legacy")
            ).lower()
            if count_mode == "legacy":
                k_max = getattr(grid_query, "K_max", None)
                points_per_gaussian = getattr(
                    grid_query, "points_per_gaussian", None
                )
                if k_max is None or points_per_gaussian is None:
                    raise ValueError(
                        "p2g.grid_query.K_max and points_per_gaussian are "
                        "required for legacy grid seeds"
                    )
                k_max = int(k_max)
                points_per_gaussian = int(points_per_gaussian)
                if k_max <= 0:
                    raise ValueError("p2g.grid_query.K_max must be positive")
                if points_per_gaussian <= 0:
                    raise ValueError(
                        "p2g.grid_query.points_per_gaussian must be positive"
                    )
                exp = getattr(grid_query, "exp", None)
                if exp is not None:
                    exp = int(exp)
                    if exp not in (1, 2):
                        raise ValueError(
                            "p2g.grid_query.exp must be null, 1, or 2"
                        )
                    if exp > k_max:
                        raise ValueError(
                            f"p2g.grid_query.exp={exp} requires K_max >= {exp}"
                        )
                self.grid_seed_config = (
                    "legacy", points_per_gaussian, k_max, exp, None,
                )
            elif count_mode in (
                "learned_gumbel", "learned_decoupled_st",
                "learned_gumbel_viewpt",
            ):
                learned_count = getattr(grid_query, "learned_count", None)
                if learned_count is None:
                    raise ValueError(
                        "p2g.grid_query.learned_count is required for "
                        f"count_mode={count_mode!r}"
                    )
                k_max = getattr(learned_count, "K_max", None)
                if k_max is None or int(k_max) <= 0:
                    raise ValueError(
                        "p2g.grid_query.learned_count.K_max must be positive"
                    )
                seed_mode = str(
                    getattr(learned_count, "seed_mode", "range_quantile")
                ).lower()
                if seed_mode != "range_quantile":
                    raise ValueError(
                        "p2g.grid_query.learned_count.seed_mode currently "
                        "supports only 'range_quantile'"
                    )
                # Legacy points_per_gaussian/exp are deliberately not read in
                # this branch: K is predicted after temporal feature fusion.
                self.grid_seed_config = (
                    count_mode, None, int(k_max), None, seed_mode,
                )
                pseudo_gt = getattr(learned_count, "pseudo_gt", None)
                if pseudo_gt is not None and bool(
                    getattr(pseudo_gt, "enable", False)
                ):
                    if count_mode != "learned_gumbel":
                        raise ValueError(
                            "grid pseudo-GT currently supports count_mode="
                            "'learned_gumbel' only"
                        )
                    thresholds_m = tuple(
                        float(value)
                        for value in getattr(pseudo_gt, "thresholds_m", ())
                    )
                    if len(thresholds_m) != int(k_max) - 1:
                        raise ValueError(
                            "learned_count.pseudo_gt.thresholds_m must contain "
                            "K_max - 1 values"
                        )
                    self.router_guide_kwargs = {
                        "thresholds_m": thresholds_m,
                        "min_raw_points": int(
                            getattr(pseudo_gt, "min_raw_points", 3)
                        ),
                        "loo_denominator_min": float(
                            getattr(pseudo_gt, "loo_denominator_min", 1.0e-3)
                        ),
                        "min_valid_loo_count": int(
                            getattr(pseudo_gt, "min_valid_loo_count", 3)
                        ),
                        "min_valid_loo_fraction": float(
                            getattr(pseudo_gt, "min_valid_loo_fraction", 0.75)
                        ),
                        "residual_quantile": float(
                            getattr(pseudo_gt, "residual_quantile", 0.75)
                        ),
                    }
            else:
                raise ValueError(
                    "p2g.grid_query.count_mode must be 'legacy', "
                    "'learned_gumbel', 'learned_decoupled_st', or "
                    "'learned_gumbel_viewpt'"
                )
        lora_cfg = getattr(cfg, "utonia_lora", None)
        self.intensity_in_utonia = bool(
            getattr(lora_cfg, "enable", False)
        ) if lora_cfg is not None else False
        if self.intensity_in_utonia:
            self.intensity_mode = "utonia_xyzi"
            self.intensity_encoder = None
            self.intensity_out_dim = 0
        else:
            (
                self.intensity_mode,
                self.intensity_encoder,
                self.intensity_out_dim,
            ) = build_intensity_encoder(cfg)

    def forward(
        self,
        lidar_points,
        offset,
        pose,
        features,
        ptv3_input,
        mapper,
        *,
        compute_router_guide=False,
    ):
        device = features["feat"].device
        compute_router_guide = bool(compute_router_guide)
        if compute_router_guide and self.router_guide_kwargs is None:
            raise RuntimeError(
                "router pseudo-GT was requested without an enabled pseudo_gt config"
            )

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
        membership_list = [] if self.anchor_mode == "spherical" else None
        guide_data_list = [] if compute_router_guide else None
        for i in range(len(feat_list)):
            pts = pts_list[i]
            xyz = pts[:, :3]
            inten = pts[:, 3]
            if self.grid_seed_config is None:
                result = aggregate_points_to_cells_with_membership(
                    xyz, inten, grid_coord_list[i], feat_list[i],
                    coord_list[i], origins[i], mapper,
                )
                upos, ufeat, int5, occ, raw_count, membership = result
                membership_list.append(membership)
            else:
                (
                    count_mode, points_per_gaussian, k_max, exp, seed_mode,
                ) = self.grid_seed_config
                result = aggregate_points_to_cells_with_seeds(
                    xyz, inten, grid_coord_list[i], feat_list[i],
                    coord_list[i], origins[i], mapper,
                    points_per_gaussian, k_max, exp=exp,
                    count_mode=count_mode, seed_mode=seed_mode,
                    return_membership=compute_router_guide,
                )
                if compute_router_guide:
                    (
                        upos, ufeat, int5, occ, raw_count, membership, seed_data,
                    ) = result
                    guide = grid_inverse_range_pseudo_gt(
                        membership.points_sensor,
                        membership.token_index,
                        int(upos.shape[0]),
                        **self.router_guide_kwargs,
                    )
                    if not torch.equal(guide.raw_count, raw_count):
                        raise RuntimeError(
                            "grid pseudo-GT membership drifted from occupied-token "
                            "raw counts"
                        )
                    guide_data_list.append(guide)
                else:
                    upos, ufeat, int5, occ, raw_count, seed_data = result
                seed_data_list.append(seed_data)
            occ_gc = grid_coord_list[i].to(device)[occ]
            pos_list.append(upos)
            ufeat_list.append(ufeat)
            int5_list.append(int5)
            occ_list.append(occ)
            gc_list.append(occ_gc)
            raw_count_list.append(raw_count)

        if self.intensity_in_utonia:
            ifeat_list = None
        else:
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
            raw_memberships=membership_list,
            router_guides=guide_data_list,
        )
