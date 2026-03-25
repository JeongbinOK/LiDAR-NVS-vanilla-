"""Module D: Gaussian parameter prediction with differentiable PCA initialization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussian_fit import _rotation_matrix_to_quaternion


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Convention: rows of R are the local frame basis vectors (u, v, n).

    Args:
        q: [B, 4] unit quaternions

    Returns:
        R: [B, 3, 3] rotation matrices
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(-1, 3, 3)

    return R


class GaussianParameterHead(nn.Module):
    """Predict 2D Gaussian surfel parameters per cluster.

    Uses differentiable weighted PCA for initial rotation/scaling,
    then per-parameter MLPs predict residual corrections.
    Normal n is deterministically derived from quaternion q: n = u x v.
    """

    def __init__(self, dim: int = 64):
        super().__init__()

        self.mlp_mu = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 3))
        self.mlp_q = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 4))
        self.mlp_s = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 2))
        self.mlp_alpha = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))

        # Initialize residual heads near zero so initial output ≈ PCA
        for mlp in [self.mlp_mu, self.mlp_q, self.mlp_s]:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

        # Cached identity matrix (moved to correct device with model)
        self.register_buffer("_eye3", torch.eye(3).unsqueeze(0))

    def forward(self, features: torch.Tensor, xyz: torch.Tensor,
                assign_indices: torch.Tensor, assign_weights: torch.Tensor,
                seed_indices: torch.Tensor) -> dict:
        """
        Args:
            features: [N, D] per-point features
            xyz: [N, 3] point positions
            assign_indices: [N, top_k] indices into seed array (0..K-1)
            assign_weights: [N, top_k] soft assignment weights
            seed_indices: [K] seed indices (determines K)

        Returns:
            dict with mu [K,3], q [K,4], s [K,2], n [K,3], alpha [K,1],
            u [K,3], v [K,3]
        """
        N, D = features.shape
        K = seed_indices.shape[0]
        top_k = assign_indices.shape[1]
        device = features.device

        # ---- Weighted aggregation via scatter ----
        flat_idx = assign_indices.reshape(-1)   # [N*top_k]
        flat_w = assign_weights.reshape(-1)     # [N*top_k]

        flat_feats = features.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, D)
        flat_xyz = xyz.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, 3)

        weight_sum = torch.zeros(K, device=device)
        weight_sum.scatter_add_(0, flat_idx, flat_w)
        ws = weight_sum.clamp(min=1e-8)

        # Cluster features
        wf = flat_w.unsqueeze(1) * flat_feats
        cluster_feats = torch.zeros(K, D, device=device)
        cluster_feats.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, D), wf)
        cluster_feats = cluster_feats / ws.unsqueeze(1)

        # Cluster centroids (anchor for residual mu)
        wxyz = flat_w.unsqueeze(1) * flat_xyz
        mu_weighted = torch.zeros(K, 3, device=device)
        mu_weighted.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, 3), wxyz)
        mu_weighted = mu_weighted / ws.unsqueeze(1)

        # ---- PCA warm start (no gradient — SVD backward is unstable) ----
        with torch.no_grad():
            centered = flat_xyz - mu_weighted[flat_idx]
            w_outer = flat_w.unsqueeze(1).unsqueeze(2) * (
                centered.unsqueeze(2) * centered.unsqueeze(1)
            )  # [N*top_k, 3, 3]

            cov = torch.zeros(K, 3, 3, device=device)
            cov.scatter_add_(0, flat_idx.view(-1, 1, 1).expand(-1, 3, 3), w_outer)
            cov = cov / ws.view(-1, 1, 1)
            cov = cov + 1e-6 * self._eye3

            U, S, Vh = torch.linalg.svd(cov)

            u_pca = Vh[:, 0, :]   # [K, 3]
            v_pca = Vh[:, 1, :]   # [K, 3]
            n_pca = torch.cross(u_pca, v_pca, dim=-1)

            # Orient normals toward sensor origin
            flip_sign = torch.where(
                (n_pca * (-mu_weighted)).sum(dim=1, keepdim=True) < 0,
                torch.tensor(-1.0, device=device),
                torch.tensor(1.0, device=device),
            )
            n_pca = n_pca * flip_sign
            v_pca = v_pca * flip_sign

            rot_pca = torch.stack([u_pca, v_pca, n_pca], dim=1)  # [K, 3, 3]
            q_pca = _rotation_matrix_to_quaternion(rot_pca)
            s_pca = S[:, :2].clamp(min=1e-8).sqrt()

        # ---- Residual prediction ----
        mu = mu_weighted + self.mlp_mu(cluster_feats)

        q = F.normalize(q_pca + self.mlp_q(cluster_feats), dim=-1)

        s = F.softplus(torch.log(s_pca) + self.mlp_s(cluster_feats))

        alpha = torch.sigmoid(self.mlp_alpha(cluster_feats))

        # ---- Normal from quaternion: n = u x v ----
        R = quaternion_to_rotation_matrix(q)
        u_out = R[:, 0, :]
        v_out = R[:, 1, :]
        n_out = torch.cross(u_out, v_out, dim=-1)

        return {
            "mu": mu,
            "q": q,
            "s": s,
            "n": n_out,
            "alpha": alpha,
            "u": u_out,
            "v": v_out,
        }
