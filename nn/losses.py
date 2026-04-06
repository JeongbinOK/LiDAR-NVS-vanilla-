"""Self-supervised loss for neural Gaussian clustering.

Single loss: assignment-weighted Gaussian NLL.
Supports both 2D surfel and 3D Gaussian primitives.

L = mean_i [ sum_j  w_ij * NLL_ij ]

2D: NLL = gamma_sq + maha_2d + log_det_2d
3D: NLL = maha_3d + log_det_3d

The NLL is a proper MLE objective that is self-regularizing:
  - log_det prevents scale collapse (s → 0)
  - maha prevents scale explosion (s → ∞)
  - No auxiliary losses needed for geometric fitting
"""

import torch
import torch.nn as nn

from nn.gaussian_head import quaternion_to_rotation_matrix


class ClusteringLoss(nn.Module):
    """Geometric self-supervision for Gaussian clustering.

    L = assignment-weighted Gaussian NLL (single term).

    2D: gamma_sq + maha_2d + log_det_2d
    3D: maha_3d + log_det_3d
    """

    def __init__(
        self,
        primitive: str = "2d",
        top_k_assign: int = 8,
    ):
        super().__init__()
        self.primitive = primitive
        self.top_k_assign = top_k_assign

    def forward(self, xyz: torch.Tensor, output: dict) -> dict:
        gaussians = output["gaussians"]
        assign_full = output["assign"]  # [N, K] dense

        mu = gaussians["mu"]   # [K, 3]
        s = gaussians["s"]     # [K, 2] or [K, 3]
        q = gaussians["q"]     # [K, 4]

        N, K = assign_full.shape

        # Sparsify: top-k assignments per point for memory efficiency
        top_k = min(self.top_k_assign, K)
        assign_topk_w, assign_topk_idx = assign_full.topk(top_k, dim=-1)
        # Renormalize so weights sum to 1 per point (removes tau-dependent scale drift)
        assign_topk_w = assign_topk_w / assign_topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Gather Gaussian params for top-k candidates
        mu_cands = mu[assign_topk_idx]
        s_cands = s[assign_topk_idx].clamp(min=1e-4)
        q_cands = q[assign_topk_idx]

        xyz_exp = xyz.unsqueeze(1).expand(-1, top_k, -1)
        d = xyz_exp - mu_cands  # [N, k, 3]

        # Compute NLL
        if self.primitive == "2d":
            l_surface = self._nll_2d(d, s_cands, q_cands, assign_topk_w)
        else:
            l_surface = self._nll_3d(d, s_cands, q_cands, assign_topk_w)

        return {
            "total": l_surface,
            "surface": l_surface,
            "s_mean": s.mean(),
            "s_max": s.max(),
            "K": torch.tensor(K, dtype=torch.float32),
        }

    def _nll_2d(self, d, s_cands, q_cands, assign_w):
        """2D surfel NLL: gamma_sq + maha_2d + log_det_2d."""
        N, k = assign_w.shape

        q_flat = q_cands.reshape(N * k, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        u_cands = R_flat[:, 0, :].reshape(N, k, 3)
        v_cands = R_flat[:, 1, :].reshape(N, k, 3)
        n_cands = torch.cross(u_cands, v_cands, dim=-1)

        gamma_sq = (d * n_cands).sum(dim=-1).pow(2)
        # Normalize gamma_sq: sigma_perp = 0.5m matches 1m voxel scale,
        # keeping gamma_nll on the same order as maha at cluster boundaries.
        sigma_perp_sq = 0.25
        gamma_nll = (gamma_sq / sigma_perp_sq).clamp(max=1e4)
        d_u = (d * u_cands).sum(dim=-1)
        d_v = (d * v_cands).sum(dim=-1)
        maha = (d_u / s_cands[:, :, 0]).pow(2) + (d_v / s_cands[:, :, 1]).pow(2)
        
        # 2) Log-determinant of covariance is 2 * sum(ln(s))
        log_det = 2.0 * (torch.log(s_cands[:, :, 0]) + torch.log(s_cands[:, :, 1]))

        # 3) Remove .clamp(min=0) since continuous NLL can mathematically be negative
        nll = gamma_nll + maha + log_det  # [N, k]
        loss = (assign_w * nll).sum(1).mean()
        return loss

    def _nll_3d(self, d, s_cands, q_cands, assign_w):
        """3D Gaussian NLL: maha_3d + log_det_3d."""
        N, k = assign_w.shape

        K_flat = q_cands.shape[0] * q_cands.shape[1]
        q_flat = q_cands.reshape(K_flat, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        R = R_flat.reshape(N, k, 3, 3)

        d_local = torch.einsum('nkij,nkj->nki', R.transpose(-1, -2), d)
        maha = (d_local / s_cands).pow(2).sum(dim=-1)
        
        # 4) Log-determinant correction (factor of 2)
        log_det = 2.0 * torch.log(s_cands).sum(dim=-1)

        # 5) Remove .clamp(min=0) to allow gradient flow at small scales
        nll = maha + log_det
        loss = (assign_w * nll).sum(1).mean()
        return loss
