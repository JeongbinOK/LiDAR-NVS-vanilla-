"""Utonia/intensity feature fusion construction and execution."""
from __future__ import annotations

import torch
import torch.nn as nn

from .token_refiner import build_joint_refiner


class UtoniaResidualAdapter(nn.Module):
    """Dim-preserving residual adapter for frozen Utonia features.

    ``in_norm`` is kept before the bottleneck MLP. ``out_norm`` is optional so
    the adapter can be ablated without changing the residual MLP itself.
    """

    def __init__(self, dim: int, bottleneck: int | None = None,
                 out_norm: bool = True):
        super().__init__()
        width = int(bottleneck) if bottleneck else max(8, dim // 4)
        self.in_norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, width)
        self.act = nn.SiLU()
        self.up = nn.Linear(width, dim)
        self.out_norm = nn.LayerNorm(dim) if out_norm else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature):
        if feature.shape[0] == 0:
            return feature
        delta = self.up(self.act(self.down(self.in_norm(feature))))
        return self.out_norm(feature + delta)


def build_feature_fusion(cfg, *, utonia_dim: int, intensity_dim: int,
                         coord_scale: float):
    """Build fusion components while preserving Point2Gaus state-dict names."""
    adapter_cfg = getattr(cfg, "utonia_adapter", None)
    bottleneck = (
        int(getattr(adapter_cfg, "bottleneck", 0) or 0)
        if adapter_cfg is not None else 0
    )
    out_norm = bool(getattr(adapter_cfg, "out_norm", True)) \
        if adapter_cfg is not None else True
    adapter = UtoniaResidualAdapter(
        utonia_dim, bottleneck=bottleneck or None, out_norm=out_norm
    )
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
    return adapter, fusion_mlp, refiner


def fuse_features(adapter, fusion_mlp, refiner, *, utonia_feature,
                  intensity_feature, position, grid_coord, offset):
    """Fuse two streams and optionally refine their joint token representation."""
    fused = fusion_mlp(
        torch.cat([adapter(utonia_feature), intensity_feature], dim=1)
    )
    if refiner is None:
        return fused
    if grid_coord is None:
        raise RuntimeError(
            "joint_refiner requires per-token grid_coord from the active builder"
        )
    return refiner(fused, position, grid_coord, offset)


__all__ = [
    "UtoniaResidualAdapter",
    "build_feature_fusion",
    "fuse_features",
]
