"""Phase A loss for pair-wise LiDAR rendering.

Loss family:
  - Depth         : L1 on expected depth (rendered.range) over GT hit rays
  - Intensity     : L1 on alpha-blended intensity over GT hit rays
  - Raydrop       : BCE on predicted drop probability over all rays
  - Depth distort : 2DGS depth distortion regulariser (CUDA-computed per pixel)
  - Normal        : Curvature-aware normal consistency loss (QGS eq.)

The loss deliberately avoids any direct alpha coverage regularizer. Collapse to
`alpha_accum -> 0` is instead penalised through the raydrop term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import diff_quadratic_rasterization as dq
from diff_quadratic_rasterization import LiDARRasterOutput
from nn.render_utils import build_gt_normal_map

# DISTORTION_OFFSET and CURV_DISTORTION_OFFSET are not re-exported by the
# Python wrapper, so compute from the base offset (channel_layout.h).
_DISTORTION_OFFSET = dq.NUM_CHANNELS + 5   # == DEPTH_OFFSET + 2


class QGSLoss(nn.Module):
    def __init__(
        self,
        *,
        w_depth: float = 1.0,
        w_intensity: float = 0.1,
        w_raydrop: float = 0.1,
        w_distortion: float = 0.05,
        w_normal: float = 0.05,
        alpha_eps: float = 1e-3,
        raydrop_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_depth = float(w_depth)
        self.w_intensity = float(w_intensity)
        self.w_raydrop = float(w_raydrop)
        self.w_distortion = float(w_distortion)
        self.w_normal = float(w_normal)
        self.alpha_eps = float(alpha_eps)
        self.raydrop_eps = float(raydrop_eps)

    def forward(
        self,
        rendered: LiDARRasterOutput,
        target: dict,
        drop_prob: Tensor | None = None,
        ray_grid: Tensor | None = None,
    ) -> dict:
        valid_mask = target["valid_mask"]
        if drop_prob is not None and drop_prob.dim() == valid_mask.dim() + 1 and drop_prob.shape[0] == 1:
            drop_prob = drop_prob[0]
        valid_f = valid_mask.to(dtype=rendered.range.dtype)
        n_valid = valid_f.sum().clamp(min=1.0)

        depth_gt = target["range_image"]
        intensity_gt = target["intensity_image"]
        # Train with expected (alpha-blended) depth — smooth gradients flow to all
        # Gaussians on the ray. Evaluation uses middepth (see eval_utils.py).
        depth_pred = rendered.range
        intensity_pred = rendered.intensity

        depth_loss = F.l1_loss(depth_pred[valid_mask], depth_gt[valid_mask]) if valid_mask.any() else rendered.range.new_zeros(())
        intensity_loss = (
            F.l1_loss(intensity_pred[valid_mask], intensity_gt[valid_mask])
            if valid_mask.any()
            else rendered.range.new_zeros(())
        )

        if drop_prob is None:
            drop_prob = 1.0 - rendered.alpha_accum.clamp(0.0, 1.0)
        drop_target = (~valid_mask).to(dtype=rendered.range.dtype)
        raydrop_loss = F.binary_cross_entropy(
            drop_prob.clamp(self.raydrop_eps, 1.0 - self.raydrop_eps),
            drop_target,
        )

        # Depth distortion: CUDA-computed ∑ᵢ∑ⱼ wᵢwⱼ|rᵢ−rⱼ| per pixel.
        # Keep compatibility with lightweight stubs that do not expose `raw`.
        distortion_loss = rendered.range.new_zeros(())
        if self.w_distortion > 0.0:
            raw = getattr(rendered, "raw", None)
            if raw is not None:
                distortion_map = raw[_DISTORTION_OFFSET]              # [H, W]
                distortion_loss = (
                    distortion_map[valid_mask].mean()
                    if valid_mask.any()
                    else distortion_map.new_zeros(())
                )

        # Curvature-guided normal consistency (QGS Eq. 18-20).
        # L_n(u,v)   = Σ_i ω_i (1 - n_i^T N)
        #            = α_accum(u,v) - <Σ_i ω_i n_i, N>
        # L_Kn(u,v)  = λ_K(K(u,v)) · L_n(u,v)
        # λ_K(K)     = 1 - sigmoid(ln(|K| + ε))
        #
        # Here N is the differential normal from the rendered depth map, not a
        # GT normal target. The rasterizer already returns the alpha-blended
        # normal sum Σ_i ω_i n_i and curvature map K(u,v)=Σ_i ω_i K_i.
        normal_loss = rendered.range.new_zeros(())
        pred_n_sum = getattr(rendered, "normal", None)
        kappa_map = getattr(rendered, "curvature", None)
        if self.w_normal > 0.0 and pred_n_sum is not None and kappa_map is not None and ray_grid is not None:
            depth_for_normal = getattr(rendered, "middepth", rendered.range)
            render_valid = rendered.alpha_accum > self.alpha_eps
            diff_normal = build_gt_normal_map(depth_for_normal, render_valid, ray_grid)
            ref_n = diff_normal["normal_image"]
            ref_valid = diff_normal["normal_valid"]

            mask = render_valid & ref_valid
            if mask.any():
                dot_sum = (pred_n_sum * ref_n).sum(dim=0)
                residual = (rendered.alpha_accum - dot_sum).clamp_min(0.0)
                lam = 1.0 - torch.sigmoid(torch.log(kappa_map.abs().clamp(min=1e-6)))
                normal_loss = (lam[mask] * residual[mask]).mean()

        total = (
            self.w_depth * depth_loss
            + self.w_intensity * intensity_loss
            + self.w_raydrop * raydrop_loss
            + self.w_distortion * distortion_loss
            + self.w_normal * normal_loss
        )

        return {
            "total": total,
            "depth": depth_loss.detach(),
            "intensity": intensity_loss.detach(),
            "raydrop": raydrop_loss.detach(),
            "distortion": distortion_loss.detach(),
            "normal": normal_loss.detach(),
            "n_valid": n_valid.detach(),
            "valid_ratio": valid_f.mean().detach(),
        }
