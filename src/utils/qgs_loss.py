"""Phase A loss for pair-wise LiDAR rendering.

Loss family:
  - Depth         : mean of expected-depth and median-depth L1 over GT hit rays
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
        alpha_eps: float = 0.5,
        raydrop_eps: float = 1e-4,
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
        pred_hit_mask = rendered.alpha_accum > self.alpha_eps
        coverage = (pred_hit_mask & valid_mask).float().sum() / n_valid

        depth_gt = target["range_image"]
        intensity_gt = target["intensity_image"]
        # Train with both expected (alpha-blended) depth and median depth. The
        # expected term gives smooth gradients to all ray contributors, while
        # the median term aligns the supervised depth with eval-time geometry.
        depth_pred = rendered.range
        depth_median_pred = getattr(rendered, "middepth", rendered.range)
        intensity_pred = rendered.intensity

        if valid_mask.any():
            depth_range_loss = F.l1_loss(depth_pred[valid_mask], depth_gt[valid_mask])
            depth_median_loss = F.l1_loss(depth_median_pred[valid_mask], depth_gt[valid_mask])
            depth_loss = 0.5 * (depth_range_loss + depth_median_loss)
        else:
            depth_range_loss = rendered.range.new_zeros(())
            depth_median_loss = rendered.range.new_zeros(())
            depth_loss = rendered.range.new_zeros(())
        intensity_loss = (
            F.l1_loss(intensity_pred[valid_mask], intensity_gt[valid_mask])
            if valid_mask.any()
            else rendered.range.new_zeros(())
        )

        drop_target = (~valid_mask).to(dtype=rendered.range.dtype)
        raydrop_loss = F.binary_cross_entropy(
            drop_prob.clamp(self.raydrop_eps, 1.0 - self.raydrop_eps),
            drop_target,
        )
        if valid_mask.any():
            raydrop_loss_hit = F.binary_cross_entropy(
                drop_prob[valid_mask].clamp(self.raydrop_eps, 1.0 - self.raydrop_eps),
                drop_target[valid_mask],
            )
            drop_prob_hit_mean = drop_prob[valid_mask].mean()
            alpha_hit_mean = rendered.alpha_accum[valid_mask].mean()
            intensity_gt_mean = intensity_gt[valid_mask].mean()
            intensity_pred_mean = intensity_pred[valid_mask].mean()
        else:
            raydrop_loss_hit = rendered.range.new_zeros(())
            drop_prob_hit_mean = rendered.range.new_zeros(())
            alpha_hit_mean = rendered.range.new_zeros(())
            intensity_gt_mean = rendered.range.new_zeros(())
            intensity_pred_mean = rendered.range.new_zeros(())
        miss_mask = ~valid_mask
        if miss_mask.any():
            raydrop_loss_miss = F.binary_cross_entropy(
                drop_prob[miss_mask].clamp(self.raydrop_eps, 1.0 - self.raydrop_eps),
                drop_target[miss_mask],
            )
            drop_prob_miss_mean = drop_prob[miss_mask].mean()
            alpha_miss_mean = rendered.alpha_accum[miss_mask].mean()
        else:
            raydrop_loss_miss = rendered.range.new_zeros(())
            drop_prob_miss_mean = rendered.range.new_zeros(())
            alpha_miss_mean = rendered.range.new_zeros(())

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
                lam = 1.0 - torch.sigmoid(torch.log(kappa_map.abs().clamp(min=1e-4)))
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
            "depth_range": depth_range_loss.detach(),
            "depth_median": depth_median_loss.detach(),
            "intensity": intensity_loss.detach(),
            "raydrop": raydrop_loss.detach(),
            "distortion": distortion_loss.detach(),
            "normal": normal_loss.detach(),
            "n_valid": n_valid.detach(),
            "coverage": coverage.detach(),
            "raydrop_hit": raydrop_loss_hit.detach(),
            "raydrop_miss": raydrop_loss_miss.detach(),
            "drop_prob_hit_mean": drop_prob_hit_mean.detach(),
            "drop_prob_miss_mean": drop_prob_miss_mean.detach(),
            "alpha_hit_mean": alpha_hit_mean.detach(),
            "alpha_miss_mean": alpha_miss_mean.detach(),
            "intensity_gt_mean": intensity_gt_mean.detach(),
            "intensity_pred_mean": intensity_pred_mean.detach(),
        }
