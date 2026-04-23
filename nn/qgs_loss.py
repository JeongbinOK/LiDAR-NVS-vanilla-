"""Phase A loss for pair-wise LiDAR rendering.

Loss family:
  - Depth      : L1 on median depth over GT hit rays
  - Intensity  : L1 on alpha-normalised intensity over GT hit rays
  - Raydrop    : BCE on predicted drop probability over all rays

The loss deliberately avoids any direct alpha coverage regularizer. Collapse to
`alpha_accum -> 0` is instead penalised through the raydrop term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diff_quadratic_rasterization import LiDARRasterOutput


class QGSLoss(nn.Module):
    def __init__(
        self,
        *,
        w_depth: float = 1.0,
        w_intensity: float = 0.1,
        w_raydrop: float = 0.1,
        alpha_eps: float = 1e-3,
        raydrop_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_depth = float(w_depth)
        self.w_intensity = float(w_intensity)
        self.w_raydrop = float(w_raydrop)
        self.alpha_eps = float(alpha_eps)
        self.raydrop_eps = float(raydrop_eps)

    def forward(
        self,
        rendered: LiDARRasterOutput,
        target: dict,
        drop_prob: Tensor | None = None,
    ) -> dict:
        valid_mask = target["valid_mask"]
        if drop_prob is not None and drop_prob.dim() == valid_mask.dim() + 1 and drop_prob.shape[0] == 1:
            drop_prob = drop_prob[0]
        valid_f = valid_mask.to(dtype=rendered.range.dtype)
        n_valid = valid_f.sum().clamp(min=1.0)

        depth_gt = target["range_image"]
        intensity_gt = target["intensity_image"]
        depth_pred = rendered.middepth
        intensity_pred = rendered.intensity / rendered.alpha_accum.clamp(min=self.alpha_eps)

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

        total = (
            self.w_depth * depth_loss
            + self.w_intensity * intensity_loss
            + self.w_raydrop * raydrop_loss
        )

        return {
            "total": total,
            "depth": depth_loss.detach(),
            "intensity": intensity_loss.detach(),
            "raydrop": raydrop_loss.detach(),
            "n_valid": n_valid.detach(),
            "valid_ratio": valid_f.mean().detach(),
        }
