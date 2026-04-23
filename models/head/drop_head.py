"""
Curvature-aware ray-drop head — Phase A3.4.

Takes the per-pixel rasterised LiDAR fields (latent, range, normal, curvature,
alpha_accum) and the per-pixel ray direction, runs a small MLP on the
alpha-normalised features, and returns a final drop probability of the form

    p_drop(u, v) = (1 - α_accum) + α_accum · σ(MLP(features))

where the first term captures pure geometric miss (no Gaussian intersected the
ray) and the second is the learned physical-drop probability conditioned on
material/incidence cues.

Inputs are expected in [B, C, H, W] layout. Each tensor produced by the CUDA
rasterizer is an alpha-weighted sum (∑ T_i α_i v_i); we divide by α_accum to
recover the per-pixel mean before feeding the MLP.

This module is intentionally pure-Python — the rasterizer kernels stay
backbone-agnostic, and the drop head is the only place that knows about the
QGS-Flow plan's drop decomposition.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_lidar_ray_grid(
    height: int,
    width: int,
    el_min_rad: float,
    el_max_rad: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-pixel unit ray direction in sensor frame.

    Mirrors the spherical projection used by the CUDA LiDAR-mode kernel
    (cuda_rasterizer/forward.cu, lidar_mode branch):
        az = (u + 0.5) / w_per_rad_az - π
        el = (v + 0.5) / h_per_rad_el + el_min
        d  = (sin(az)cos(el), cos(az)cos(el), sin(el))

    Returns a [3, H, W] tensor.
    """
    w_per_rad_az = width / (2.0 * math.pi)
    h_per_rad_el = height / (el_max_rad - el_min_rad)
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    az = u / w_per_rad_az - math.pi              # [W]
    el = v / h_per_rad_el + el_min_rad           # [H]
    cos_el = torch.cos(el)                        # [H]
    sin_el = torch.sin(el)                        # [H]
    sin_az = torch.sin(az)                        # [W]
    cos_az = torch.cos(az)                        # [W]
    dx = sin_az[None, :] * cos_el[:, None]        # [H, W]
    dy = cos_az[None, :] * cos_el[:, None]        # [H, W]
    dz = sin_el[:, None].expand(height, width)    # [H, W]
    return torch.stack([dx, dy, dz], dim=0)       # [3, H, W]


class DropHead(nn.Module):
    """Per-pixel MLP producing a learned drop probability.

    Args:
        latent_dim: width of the rasterised per-pixel latent feature.
        hidden_dim: hidden layer width for the two-layer MLP.

    Forward inputs (all [B, C, H, W] except `alpha_accum` which is [B, H, W]):
        latent       : [B, L, H, W]   alpha-weighted latent sum
        range_       : [B, H, W]      alpha-weighted range sum
        normal       : [B, 3, H, W]   alpha-weighted normal sum
        curvature    : [B, H, W]      alpha-weighted curvature sum
        alpha_accum  : [B, H, W]      Σ T_i α_i (geometric coverage)
        ray_dir      : [3, H, W] or [B, 3, H, W]  per-pixel unit ray direction

    Returns:
        drop_logit_phys : [B, H, W]   raw MLP output (pre-sigmoid)
        p_drop          : [B, H, W]   composed probability in [0, 1]

    The decomposition is

        p_hit       = α_accum
        p_drop_phys = σ(drop_logit_phys)
        p_drop      = (1 - p_hit) + p_hit · p_drop_phys
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Feature vector per pixel:
        #   latent (L) ∥ |κ_norm| (1) ∥ cos_inc (1) ∥ log(1+range_norm) (1) ∥ ray_dir (3)
        self.latent_dim = int(latent_dim)
        self.in_dim = self.latent_dim + 1 + 1 + 1 + 3
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent: Tensor,
        range_: Tensor,
        normal: Tensor,
        curvature: Tensor,
        alpha_accum: Tensor,
        ray_dir: Tensor,
        *,
        eps: float = 1e-3,
    ) -> Tuple[Tensor, Tensor]:
        if latent.dim() != 4 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must be [B, {self.latent_dim}, H, W], got {tuple(latent.shape)}"
            )
        B, _, H, W = latent.shape
        if alpha_accum.shape != (B, H, W):
            raise ValueError(
                f"alpha_accum must be [B, H, W]={B, H, W}, got {tuple(alpha_accum.shape)}"
            )
        if ray_dir.dim() == 3:
            ray_dir = ray_dir.unsqueeze(0).expand(B, 3, H, W)
        elif ray_dir.shape != (B, 3, H, W):
            raise ValueError(
                f"ray_dir must be [3, H, W] or [B, 3, H, W], got {tuple(ray_dir.shape)}"
            )

        alpha_safe = alpha_accum.clamp(min=eps)                 # [B, H, W]
        a1 = alpha_safe.unsqueeze(1)                            # [B, 1, H, W]

        latent_n   = latent / a1                                # [B, L, H, W]
        range_n    = range_ / alpha_safe                        # [B, H, W]
        normal_n   = F.normalize(normal / a1, p=2, dim=1, eps=1e-8)  # [B, 3, H, W]
        curv_n     = (curvature / alpha_safe).abs()             # [B, H, W]

        cos_inc = -(normal_n * ray_dir).sum(dim=1)              # [B, H, W]
        log_r   = torch.log1p(range_n.clamp(min=0.0))           # [B, H, W]

        feats = torch.cat([
            latent_n,                                            # L
            curv_n.unsqueeze(1),                                 # 1
            cos_inc.unsqueeze(1),                                # 1
            log_r.unsqueeze(1),                                  # 1
            ray_dir,                                             # 3
        ], dim=1)                                                # [B, in_dim, H, W]

        # Per-pixel MLP via permute + reshape; output reshaped back to [B, H, W].
        feats_flat = feats.permute(0, 2, 3, 1).reshape(B * H * W, self.in_dim)
        logit_flat = self.mlp(feats_flat).squeeze(-1)            # [B*H*W]
        drop_logit_phys = logit_flat.view(B, H, W)               # [B, H, W]

        p_hit = alpha_accum                                       # [B, H, W]
        p_drop_phys = torch.sigmoid(drop_logit_phys)              # [B, H, W]
        p_drop = (1.0 - p_hit) + p_hit * p_drop_phys              # [B, H, W]
        return drop_logit_phys, p_drop
