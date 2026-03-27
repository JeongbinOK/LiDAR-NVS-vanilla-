"""Self-supervised loss functions for neural clustering (no pseudo-GT, no rendering)."""

import torch
import torch.nn as nn


class ClusteringLoss(nn.Module):
    """Combined geometric self-supervision for 2D Gaussian clustering.

    L = w_surface * L_surface  (2D Gaussian NLL with maha + log_det)

    All supervision comes from the input point cloud geometry.
    """

    def __init__(self, w_surface: float = 1.0):
        super().__init__()
        self.w_surface = w_surface

    def forward(self, xyz: torch.Tensor, output: dict) -> dict:
        """
        Args:
            xyz: [N, 3] input point positions
            output: model output dict

        Returns:
            dict with individual losses and total
        """
        gaussians = output["gaussians"]
        assign_indices = output["assign_indices"]   # [N, top_k]
        assign_weights = output["assign_weights"]   # [N, top_k]

        mu = gaussians["mu"]    # [K, 3]
        s = gaussians["s"]      # [K, 2]
        n = gaussians["n"]      # [K, 3]
        u = gaussians["u"]      # [K, 3]
        v = gaussians["v"]      # [K, 3]

        N, top_k = assign_indices.shape

        # ==== L1: Surface Reconstruction (2D Gaussian NLL) ====
        mu_cands = mu[assign_indices]   # [N, k, 3]
        n_cands = n[assign_indices]     # [N, k, 3]
        u_cands = u[assign_indices]     # [N, k, 3]
        v_cands = v[assign_indices]     # [N, k, 3]
        s_cands = s[assign_indices].clamp(min=1e-4)  # [N, k, 2]
        xyz_exp = xyz.unsqueeze(1).expand(-1, top_k, -1)

        d = xyz_exp - mu_cands                                    # [N, k, 3]
        gamma_sq = (d * n_cands).sum(dim=-1).pow(2)               # [N, k]

        d_u = (d * u_cands).sum(dim=-1)                           # [N, k]
        d_v = (d * v_cands).sum(dim=-1)                           # [N, k]
        maha = (d_u / s_cands[:, :, 0]).pow(2) + (d_v / s_cands[:, :, 1]).pow(2)

        log_det = torch.log(s_cands[:, :, 0]) + torch.log(s_cands[:, :, 1])

        l_surface = (assign_weights * (gamma_sq + maha + log_det)).sum(dim=1).mean()

        # ==== L2: Assignment Compactness (monitoring only) ====
        dist_sq = d.pow(2).sum(dim=-1)  # [N, k]
        l_compact = (assign_weights * dist_sq).sum(dim=1).mean()

        # ==== L3: Scale Regularization (monitoring only) ====
        l_scale = (s[:, 0] * s[:, 1]).mean()

        # ==== Total (L_surface only — maha+log_det subsume compact/scale) ====
        total = self.w_surface * l_surface

        return {
            "total": total,
            "surface": l_surface,
            "compact": l_compact,
            "scale": l_scale,
            "s_mean": s.mean(),
            "s_max": s.max(),
        }

