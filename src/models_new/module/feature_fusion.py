"""Utonia/intensity feature fusion construction and execution."""
from __future__ import annotations

import torch
import torch.nn as nn

from .token_refiner import build_joint_refiner


class UtoniaResidualAdapter(nn.Module):
    """Dim-preserving residual adapter for frozen Utonia features."""

    def __init__(self, dim: int, bottleneck: int | None = None):
        super().__init__()
        width = int(bottleneck) if bottleneck else max(8, dim // 4)
        self.in_norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, width)
        self.act = nn.SiLU()
        self.up = nn.Linear(width, dim)
        self.out_norm = nn.LayerNorm(dim)
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
    adapter = UtoniaResidualAdapter(
        utonia_dim, bottleneck=bottleneck or None
    )
    agg_cfg = cfg.agg_mlp
    hidden_dim = int(agg_cfg.hidden_dim)
    hidden_layers = int(getattr(agg_cfg, "hidden_layers", 2))
    if hidden_layers < 1:
        raise ValueError("p2g.agg_mlp.hidden_layers must be at least one")
    fusion_layers = [
        nn.Linear(utonia_dim + intensity_dim, hidden_dim),
        nn.SiLU(),
    ]
    for _ in range(hidden_layers - 1):
        fusion_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
    fusion_layers.append(nn.Linear(hidden_dim, int(agg_cfg.out_dim)))
    fusion_mlp = nn.Sequential(*fusion_layers)
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
