"""Stage 4: Gaussian parameter prediction with PCA initialization.

Supports both 2D surfel (s[K,2]) and 3D Gaussian (s[K,3]) primitives.
Uses the dense assign matrix [N, K] from differentiable soft clustering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert batched 3x3 rotation matrices to quaternions [w, x, y, z].

    Uses kornia when available for numerical stability,
    with a manual fallback.

    Args:
        R: [B, 3, 3] rotation matrices

    Returns:
        q: [B, 4] quaternions (w, x, y, z)
    """
    try:
        from kornia.geometry.conversions import rotation_matrix_to_quaternion
        # kornia returns (x, y, z, w) — convert to (w, x, y, z)
        q_xyzw = rotation_matrix_to_quaternion(R)
        q = torch.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], dim=1)
        return q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    except ImportError:
        pass

    # Fallback: Shepperd's method
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask1 = trace > 0
    if mask1.any():
        s = (trace[mask1] + 1.0).sqrt() * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = (1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]).clamp(min=0).sqrt() * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s.clamp(min=1e-8)
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s.clamp(min=1e-8)
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s.clamp(min=1e-8)

    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = (1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]).clamp(min=0).sqrt() * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s.clamp(min=1e-8)
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s.clamp(min=1e-8)
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s.clamp(min=1e-8)

    mask4 = ~mask1 & ~mask2 & ~mask3
    if mask4.any():
        s = (1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]).clamp(min=0).sqrt() * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s.clamp(min=1e-8)
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s.clamp(min=1e-8)
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s.clamp(min=1e-8)
        q[mask4, 3] = 0.25 * s

    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return q

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(-1, 3, 3)
    return R


class GaussianParameterHead(nn.Module):
    """Predict Gaussian parameters per cluster from refined center features.

    Args:
        dim: feature dimension
        primitive: "2d" for surfel (s[K,2]) or "3d" for volumetric (s[K,3])
        pca_topk: number of top assigned points per cluster for PCA
    """

    def __init__(self, dim: int = 64, primitive: str = "2d", pca_topk: int = 128):
        super().__init__()
        self.primitive = primitive
        self.pca_topk = pca_topk

        s_dim = 2 if primitive == "2d" else 3

        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 3))
        self.mlp_q = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 4))
        self.mlp_s = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, s_dim))
        # Zero-init residual heads so initial output = PCA
        for mlp in [self.mlp_mu, self.mlp_q, self.mlp_s]:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

        self.register_buffer("_eye3", torch.eye(3).unsqueeze(0))

    def _pca_warmstart(self, centers, assign, vote_xyz):
        """Compute PCA initialization from weighted covariance.

        Uses top-M assigned points per cluster for memory efficiency.
        """
        K = centers.shape[0]
        M = min(self.pca_topk, assign.shape[0])
        device = centers.device

        # Top-M points per cluster by assignment weight
        _, top_idx = assign.T.topk(M, dim=-1)     # [K, M]
        top_xyz = vote_xyz[top_idx]                # [K, M, 3]
        top_w = assign.T.gather(1, top_idx)        # [K, M]

        # Weighted covariance
        d = top_xyz - centers.unsqueeze(1)          # [K, M, 3]
        w_norm = top_w / top_w.sum(1, keepdim=True).clamp(min=1e-8)
        cov = torch.einsum('kmi,kmj,km->kij', d, d, w_norm)  # [K, 3, 3]
        cov = cov + 1e-6 * self._eye3

        # SVD
        U, S, Vh = torch.linalg.svd(cov)

        u_pca = Vh[:, 0, :]
        v_pca = Vh[:, 1, :]
        n_pca = torch.cross(u_pca, v_pca, dim=-1)

        # Orient normals toward sensor origin
        flip = torch.where(
            (n_pca * (-centers)).sum(1, keepdim=True) < 0,
            torch.tensor(-1.0, device=device),
            torch.tensor(1.0, device=device),
        )
        n_pca = n_pca * flip
        v_pca = v_pca * flip

        rot_pca = torch.stack([u_pca, v_pca, n_pca], dim=1)
        q_pca = _rotation_matrix_to_quaternion(rot_pca)

        if self.primitive == "2d":
            s_pca = S[:, :2].clamp(min=1e-8).sqrt()
        else:
            s_pca = S.clamp(min=1e-8).sqrt()

        return q_pca, s_pca

    def forward(
        self,
        center_feats: torch.Tensor,
        centers: torch.Tensor,
        assign: torch.Tensor,
        vote_xyz: torch.Tensor,
    ) -> dict:
        """
        Args:
            center_feats: [K, D] refined center features
            centers: [K, 3] cluster center positions
            assign: [N, K] soft assignment matrix
            vote_xyz: [N, 3] voted point positions

        Returns:
            dict with mu, q, s, alpha, and (for 2D) u, v, n
        """
        # PCA warm start (no gradient)
        with torch.no_grad():
            q_pca, s_pca = self._pca_warmstart(centers, assign, vote_xyz)

        # Residual prediction
        mu = centers + self.mlp_mu(center_feats)
        q = F.normalize(q_pca + self.mlp_q(center_feats), dim=-1)
        s = (s_pca * torch.exp(self.mlp_s(center_feats).clamp(-3, 3))).clamp(min=0.1)
        alpha = torch.ones(centers.shape[0], 1, device=centers.device)

        # Rotation matrix and derived vectors
        R = quaternion_to_rotation_matrix(q)
        u_out = R[:, 0, :]
        v_out = R[:, 1, :]
        n_out = torch.cross(u_out, v_out, dim=-1)

        result = {
            "mu": mu,
            "q": q,
            "s": s,
            "alpha": alpha,
        }

        if self.primitive == "2d":
            result.update({"u": u_out, "v": v_out, "n": n_out})

        return result
