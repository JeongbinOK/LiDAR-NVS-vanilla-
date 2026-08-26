"""Utonia/intensity feature fusion construction and execution."""
from __future__ import annotations

import torch
import torch.nn as nn

from .token_refiner import build_joint_refiner


def build_feature_fusion(cfg, *, utonia_dim: int, intensity_dim: int,
                         coord_scale: float):
    """Build fusion components while preserving Point2Gaus state-dict names."""
    agg_cfg = cfg.agg_mlp
    fusion_mlp = nn.Sequential(
        nn.Linear(utonia_dim + intensity_dim, int(agg_cfg.hidden_dim)),
        nn.SiLU(),
        nn.Linear(int(agg_cfg.hidden_dim), int(agg_cfg.hidden_dim)),
        nn.SiLU(),
        nn.Linear(int(agg_cfg.hidden_dim), int(agg_cfg.out_dim)),
    )
    refiner = build_joint_refiner(
        getattr(cfg, "joint_refiner", None),
        dim=int(agg_cfg.out_dim),
        coord_scale=coord_scale,
    )
    return fusion_mlp, refiner


def fuse_features(fusion_mlp, refiner, *, utonia_feature,
                  intensity_feature, position, grid_coord, offset):
    """Fuse two streams and optionally refine their joint token representation."""
    fused = fusion_mlp(
        torch.cat([utonia_feature, intensity_feature], dim=1)
    )
    if refiner is None:
        return fused
    if grid_coord is None:
        raise RuntimeError(
            "joint_refiner requires per-token grid_coord from the active builder"
        )
    return refiner(fused, position, grid_coord, offset)


__all__ = [
    "build_feature_fusion",
    "fuse_features",
]


