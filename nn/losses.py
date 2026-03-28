"""Self-supervised loss functions for neural Gaussian clustering.

Supports both 2D surfel and 3D Gaussian primitives.
Alpha-weighted NLL + sparsity for automatic Gaussian count control.
"""

import torch
import torch.nn as nn

from nn.gaussian_head import quaternion_to_rotation_matrix


class ClusteringLoss(nn.Module):
    """Geometric self-supervision for Gaussian clustering.

    L = w_surface * L_surface(alpha-weighted NLL) + lambda_sparse * L_sparsity

    2D: gamma_sq + maha_2d + log_det_2d
    3D: maha_3d + log_det_3d (no normal comparison)
    """

    def __init__(
        self,
        w_surface: float = 1.0,
        lambda_sparse: float = 0.01,
        primitive: str = "2d",
        top_k_assign: int = 8,
    ):
        super().__init__()
        self.w_surface = w_surface
        self.lambda_sparse = lambda_sparse
        self.primitive = primitive
        self.top_k_assign = top_k_assign

    def forward(self, xyz: torch.Tensor, output: dict) -> dict:
        """
        Args:
            xyz: [N, 3] input point positions
            output: model output dict

        Returns:
            dict with individual losses and total
        """
        gaussians = output["gaussians"]
        assign_full = output["assign"]  # [N, K] dense

        mu = gaussians["mu"]        # [K, 3]
        s = gaussians["s"]          # [K, 2] or [K, 3]
        q = gaussians["q"]          # [K, 4]
        alpha = gaussians["alpha"]  # [K, 1]

        N, K = assign_full.shape

        # Sparsify: top-k assignments per point for memory efficiency
        top_k = min(self.top_k_assign, K)
        assign_topk_w, assign_topk_idx = assign_full.topk(top_k, dim=-1)  # [N, k], [N, k]

        # Gather Gaussian params for top-k candidates
        mu_cands = mu[assign_topk_idx]         # [N, k, 3]
        s_cands = s[assign_topk_idx].clamp(min=1e-4)
        q_cands = q[assign_topk_idx]           # [N, k, 4]
        alpha_cands = alpha[assign_topk_idx].squeeze(-1)  # [N, k]

        xyz_exp = xyz.unsqueeze(1).expand(-1, top_k, -1)
        d = xyz_exp - mu_cands  # [N, k, 3]

        if self.primitive == "2d":
            l_surface = self._nll_2d(d, s_cands, q_cands, alpha_cands,
                                     assign_topk_w, gaussians)
        else:
            l_surface = self._nll_3d(d, s_cands, q_cands, alpha_cands,
                                     assign_topk_w)

        # Sparsity: push unused Gaussians toward alpha=0
        l_sparsity = alpha.mean()

        # Total
        total = self.w_surface * l_surface + self.lambda_sparse * l_sparsity

        # Monitoring metrics
        dist_sq = d.pow(2).sum(dim=-1)
        l_compact = (assign_topk_w * dist_sq).sum(1).mean()
        l_scale = (s[:, 0] * s[:, 1]).mean() if s.shape[1] >= 2 else s.mean()
        alpha_active = (alpha.squeeze(-1) > 0.1).sum().float()

        return {
            "total": total,
            "surface": l_surface,
            "sparsity": l_sparsity,
            "compact": l_compact,
            "scale": l_scale,
            "s_mean": s.mean(),
            "s_max": s.max(),
            "alpha_active": alpha_active,
        }

    def _nll_2d(self, d, s_cands, q_cands, alpha_cands, assign_w, gaussians):
        """2D surfel NLL: gamma_sq + maha_2d + log_det_2d."""
        N, k = assign_w.shape

        # Derive u, v, n from quaternion candidates
        q_flat = q_cands.reshape(N * k, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        u_cands = R_flat[:, 0, :].reshape(N, k, 3)
        v_cands = R_flat[:, 1, :].reshape(N, k, 3)
        n_cands = torch.cross(u_cands, v_cands, dim=-1)

        # Off-plane distance (normal direction)
        gamma_sq = (d * n_cands).sum(dim=-1).pow(2)

        # In-plane Mahalanobis distance
        d_u = (d * u_cands).sum(dim=-1)
        d_v = (d * v_cands).sum(dim=-1)
        maha = (d_u / s_cands[:, :, 0]).pow(2) + (d_v / s_cands[:, :, 1]).pow(2)

        # Log determinant
        log_det = torch.log(s_cands[:, :, 0]) + torch.log(s_cands[:, :, 1])

        # Alpha-weighted NLL
        per_point = alpha_cands * (gamma_sq + maha + log_det)
        return (assign_w * per_point).sum(1).mean()

    def _nll_3d(self, d, s_cands, q_cands, alpha_cands, assign_w):
        """3D Gaussian NLL: maha_3d + log_det_3d (no normal comparison)."""
        N, k = assign_w.shape

        # Rotation matrix from quaternion
        K_flat = q_cands.shape[0] * q_cands.shape[1]
        q_flat = q_cands.reshape(K_flat, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)  # [N*k, 3, 3]
        R = R_flat.reshape(N, k, 3, 3)

        # Rotate d into Gaussian local frame
        d_local = torch.einsum('nkij,nkj->nki', R.transpose(-1, -2), d)  # [N, k, 3]

        # 3D Mahalanobis
        maha = (d_local / s_cands).pow(2).sum(dim=-1)  # [N, k]

        # Log determinant (3 axes)
        log_det = torch.log(s_cands).sum(dim=-1)  # [N, k]

        # Alpha-weighted NLL
        per_point = alpha_cands * (maha + log_det)
        return (assign_w * per_point).sum(1).mean()
