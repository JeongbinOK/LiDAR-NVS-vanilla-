"""Self-supervised loss for neural Gaussian clustering.

Single loss: assignment-weighted Gaussian NLL, computed per-Gaussian.

L = mean_k [ sum_m  w_km * NLL_km ]

For each Gaussian k, the top-M most-assigned points are gathered and their
NLL under that Gaussian is minimized. This ensures every Gaussian receives
gradient regardless of cluster size (no dead-cluster problem), and is
consistent with the per-Gaussian PCA warmstart in GaussianHead.

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
        top_m: int = 32,
    ):
        super().__init__()
        self.primitive = primitive
        self.top_m = top_m

    def forward(self, xyz: torch.Tensor, output: dict) -> dict:
        gaussians = output["gaussians"]
        assign_full = output["assign"]  # [N, K] dense

        mu = gaussians["mu"]   # [K, 3]
        s = gaussians["s"]     # [K, 2] or [K, 3]
        q = gaussians["q"]     # [K, 4]

        N, K = assign_full.shape

        # Per-Gaussian top-M: each Gaussian gathers its most-assigned points.
        # Ensures all K Gaussians receive gradient (no dead clusters).
        # .contiguous() avoids implicit temp copy inside topk on non-contiguous .T view.
        top_m = min(self.top_m, N)
        assign_topk_w, assign_topk_idx = assign_full.T.topk(top_m, dim=-1)  # [K, M]
        # Normalize per Gaussian so weights sum to 1 per cluster
        assign_topk_w = assign_topk_w / assign_topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Gather point coords and Gaussian params
        top_xyz = xyz[assign_topk_idx]                          # [K, M, 3]
        d = top_xyz - mu.unsqueeze(1)                           # [K, M, 3]
        s_exp = s.unsqueeze(1).expand(-1, top_m, -1).clamp(min=0.01)  # [K, M, s_dim]
        q_exp = q.unsqueeze(1).expand(-1, top_m, -1)            # [K, M, 4]

        # Compute NLL
        if self.primitive == "2d":
            l_surface = self._nll_2d(d, s_exp, q_exp, assign_topk_w)
        else:
            l_surface = self._nll_3d(d, s_exp, q_exp, assign_topk_w)

        return {
            "total": l_surface,
            "surface": l_surface,
            "s_mean": s.mean(),
            "s_max": s.max(),
            "K": torch.tensor(K, dtype=torch.float32),
        }

    def _nll_2d(self, d, s_cands, q_cands, assign_w):
        """2D surfel NLL: gamma_sq + maha_2d + log_det_2d."""
        K, M = assign_w.shape

        q_flat = q_cands.reshape(K * M, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        u_cands = R_flat[:, 0, :].reshape(K, M, 3)
        v_cands = R_flat[:, 1, :].reshape(K, M, 3)
        n_cands = torch.cross(u_cands, v_cands, dim=-1)

        gamma_sq = (d * n_cands).sum(dim=-1).pow(2)
        # Normalize gamma_sq: sigma_perp = 0.5m matches 1m voxel scale,
        # keeping gamma_nll on the same order as maha at cluster boundaries.
        sigma_perp_sq = 0.25
        gamma_nll = (gamma_sq / sigma_perp_sq).clamp(max=1e4)
        d_u = (d * u_cands).sum(dim=-1)
        d_v = (d * v_cands).sum(dim=-1)
        maha = (d_u / s_cands[:, :, 0]).pow(2) + (d_v / s_cands[:, :, 1]).pow(2)
        maha = maha.clamp(max=7500.0)

        # 2) Log-determinant of covariance is 2 * sum(ln(s))
        log_det = 2.0 * (torch.log(s_cands[:, :, 0]) + torch.log(s_cands[:, :, 1]))

        # 3) Remove .clamp(min=0) since continuous NLL can mathematically be negative
        nll = gamma_nll + maha + log_det  # [N, k]
        loss = (assign_w * nll).sum(1).mean()
        return loss

    def _nll_3d(self, d, s_cands, q_cands, assign_w):
        """3D Gaussian NLL: maha_3d + log_det_3d."""
        K, M = assign_w.shape

        q_flat = q_cands.reshape(K * M, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        R = R_flat.reshape(K, M, 3, 3)

        d_local = torch.einsum('nkij,nkj->nki', R.transpose(-1, -2), d)
        maha = (d_local / s_cands).pow(2).sum(dim=-1).clamp(max=7500.0)

        # 4) Log-determinant correction (factor of 2)
        log_det = 2.0 * torch.log(s_cands).sum(dim=-1)

        # 5) Remove .clamp(min=0) to allow gradient flow at small scales
        nll = maha + log_det
        loss = (assign_w * nll).sum(1).mean()
        return loss
