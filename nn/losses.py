"""Self-supervised loss functions for neural Gaussian clustering.

Supports both 2D surfel and 3D Gaussian primitives.

Loss design:
  L_surface:    assignment-weighted NLL (alpha detached — no gradient to alpha)
  L_alpha:      quality-based alpha supervision (good fit → alpha=1, bad → alpha=0)
  L_centerness: auxiliary loss for seed selection (teaches score_mlp)
  L_barrier:    log barrier prevents alpha collapse

This eliminates the surface↔barrier gradient conflict (cos=-0.96 before fix)
and provides gradient to the otherwise-dead centerness predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.gaussian_head import quaternion_to_rotation_matrix


class ClusteringLoss(nn.Module):
    """Geometric self-supervision for Gaussian clustering.

    L = w_surface * L_surface
      + lambda_alpha * L_alpha
      + lambda_center * L_centerness
      + lambda_barrier * L_barrier

    2D: gamma_sq + maha_2d + log_det_2d
    3D: maha_3d + log_det_3d (no normal comparison)
    """

    def __init__(
        self,
        w_surface: float = 1.0,
        lambda_alpha: float = 0.1,
        lambda_center: float = 0.1,
        lambda_barrier: float = 0.01,
        primitive: str = "2d",
        top_k_assign: int = 8,
    ):
        super().__init__()
        self.w_surface = w_surface
        self.lambda_alpha = lambda_alpha
        self.lambda_center = lambda_center
        self.lambda_barrier = lambda_barrier
        self.primitive = primitive
        self.top_k_assign = top_k_assign

    def forward(self, xyz: torch.Tensor, output: dict) -> dict:
        gaussians = output["gaussians"]
        assign_full = output["assign"]  # [N, K] dense

        mu = gaussians["mu"]        # [K, 3]
        s = gaussians["s"]          # [K, 2] or [K, 3]
        q = gaussians["q"]          # [K, 4]
        alpha = gaussians["alpha"]  # [K, 1]

        N, K = assign_full.shape

        # Sparsify: top-k assignments per point for memory efficiency
        top_k = min(self.top_k_assign, K)
        assign_topk_w, assign_topk_idx = assign_full.topk(top_k, dim=-1)

        # Gather Gaussian params for top-k candidates
        mu_cands = mu[assign_topk_idx]
        s_cands = s[assign_topk_idx].clamp(min=1e-4)
        q_cands = q[assign_topk_idx]
        alpha_cands = alpha[assign_topk_idx].squeeze(-1)  # [N, k]

        xyz_exp = xyz.unsqueeze(1).expand(-1, top_k, -1)
        d = xyz_exp - mu_cands  # [N, k, 3]

        # ---- L_surface: NLL with alpha DETACHED ----
        # Gradient flows to mu, q, s but NOT alpha (eliminates conflict)
        alpha_det = alpha_cands.detach()
        if self.primitive == "2d":
            l_surface, nll_per_point = self._nll_2d(d, s_cands, q_cands, alpha_det, assign_topk_w)
        else:
            l_surface, nll_per_point = self._nll_3d(d, s_cands, q_cands, alpha_det, assign_topk_w)

        # ---- L_alpha: quality-based alpha target ----
        # Good-fit Gaussians (low NLL) → target=1, bad-fit → target=0
        with torch.no_grad():
            # Per-Gaussian average NLL via assignment weights
            # nll_per_point: [N, k], assign_topk_w: [N, k]
            # Use weighted NLL mapped back to full K
            nll_accum = torch.zeros(K, device=xyz.device)
            w_accum = torch.zeros(K, device=xyz.device)
            nll_flat = (assign_topk_w * nll_per_point).reshape(-1)
            w_flat = assign_topk_w.reshape(-1)
            idx_flat = assign_topk_idx.reshape(-1)
            nll_accum.scatter_add_(0, idx_flat, nll_flat)
            w_accum.scatter_add_(0, idx_flat, w_flat)
            avg_nll = nll_accum / w_accum.clamp(min=1e-8)  # [K]
            nll_median = avg_nll.median().clamp(min=0.1)
            # sigmoid centered at median: above median → low target, below → high
            alpha_target = torch.sigmoid(3.0 * (1.0 - avg_nll / nll_median))

        l_alpha = F.mse_loss(alpha.squeeze(-1), alpha_target)

        # ---- L_centerness: auxiliary for seed selection ----
        l_centerness = self._centerness_loss(output)

        # ---- L_barrier: log barrier on alpha ----
        alpha_f32 = alpha.float().clamp(min=1e-6)
        l_barrier = -torch.log(alpha_f32).mean()

        # Total
        total = (self.w_surface * l_surface
                 + self.lambda_alpha * l_alpha
                 + self.lambda_center * l_centerness
                 + self.lambda_barrier * l_barrier)

        # Monitoring
        alpha_active = (alpha.squeeze(-1) > 0.1).sum().float()

        return {
            "total": total,
            "surface": l_surface,
            "alpha_loss": l_alpha,
            "centerness": l_centerness,
            "barrier": l_barrier,
            "s_mean": s.mean(),
            "s_max": s.max(),
            "alpha_active": alpha_active,
        }

    def _centerness_loss(self, output):
        """Teach centerness predictor which points are near cluster centers."""
        vote_xyz = output["vote_xyz"]    # [N, 3]
        centerness = output["centerness"]  # [N]
        assign = output["assign"]        # [N, K]
        centers = output["centers"]      # [K, 3]

        # Each point's assigned center
        hard = assign.argmax(dim=1)  # [N]
        dist = (vote_xyz - centers[hard]).norm(dim=1)  # [N]

        # Target: exp(-dist/sigma), closer to center → higher centerness
        with torch.no_grad():
            sigma = dist.median().clamp(min=0.1)
            target = torch.exp(-dist / sigma)

        # Compute outside autocast (BCE is unsafe under AMP)
        with torch.amp.autocast("cuda", enabled=False):
            return F.binary_cross_entropy(centerness.float(), target.detach().float())

    def _nll_2d(self, d, s_cands, q_cands, alpha_det, assign_w):
        """2D surfel NLL: gamma_sq + maha_2d + log_det_2d.
        Returns (loss, nll_per_point) for alpha target computation."""
        N, k = assign_w.shape

        q_flat = q_cands.reshape(N * k, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        u_cands = R_flat[:, 0, :].reshape(N, k, 3)
        v_cands = R_flat[:, 1, :].reshape(N, k, 3)
        n_cands = torch.cross(u_cands, v_cands, dim=-1)

        gamma_sq = (d * n_cands).sum(dim=-1).pow(2)
        d_u = (d * u_cands).sum(dim=-1)
        d_v = (d * v_cands).sum(dim=-1)
        maha = (d_u / s_cands[:, :, 0]).pow(2) + (d_v / s_cands[:, :, 1]).pow(2)
        log_det = torch.log(s_cands[:, :, 0]) + torch.log(s_cands[:, :, 1])

        nll = (gamma_sq + maha + log_det).clamp(min=0)  # [N, k]
        per_point = alpha_det * nll
        loss = (assign_w * per_point).sum(1).mean()
        return loss, nll

    def _nll_3d(self, d, s_cands, q_cands, alpha_det, assign_w):
        """3D Gaussian NLL: maha_3d + log_det_3d."""
        N, k = assign_w.shape

        K_flat = q_cands.shape[0] * q_cands.shape[1]
        q_flat = q_cands.reshape(K_flat, 4)
        R_flat = quaternion_to_rotation_matrix(q_flat)
        R = R_flat.reshape(N, k, 3, 3)

        d_local = torch.einsum('nkij,nkj->nki', R.transpose(-1, -2), d)
        maha = (d_local / s_cands).pow(2).sum(dim=-1)
        log_det = torch.log(s_cands).sum(dim=-1)

        nll = (maha + log_det).clamp(min=0)
        per_point = alpha_det * nll
        loss = (assign_w * per_point).sum(1).mean()
        return loss, nll
