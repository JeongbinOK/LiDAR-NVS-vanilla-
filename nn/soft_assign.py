"""Module C: Soft assignment with learned affinity and geometric bias."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import _knn_points


class SoftAssignment(nn.Module):
    """Differentiable soft assignment of points to seed clusters.

    Two-stage process:
      1. Spatial KNN selects top-k candidate seeds per point (hard filter).
      2. Learned affinity + geometric bias scores candidates;
         Gumbel-Softmax produces differentiable soft weights.
    """

    def __init__(self, dim: int = 64, top_k: int = 8, lambda_normal: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.lambda_normal = lambda_normal

        # Learned affinity: [f_i; f_j; f_i - f_j] -> scalar
        self.affinity_mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, features: torch.Tensor, xyz: torch.Tensor,
                normals: torch.Tensor, seed_indices: torch.Tensor,
                tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [N, D] per-point features
            xyz: [N, 3] point positions
            normals: [N, 3] PCA-estimated normals (for geometric bias)
            seed_indices: [K] seed point indices into the point cloud
            tau: Gumbel-Softmax temperature

        Returns:
            assign_indices: [N, top_k] indices into seed array (0..K-1)
            assign_weights: [N, top_k] soft assignment weights (sum ~1)
        """
        K = seed_indices.shape[0]

        seed_xyz = xyz[seed_indices]
        seed_features = features[seed_indices]
        seed_normals = normals[seed_indices]

        # 1) Spatial KNN: top_k nearest seeds per point
        k = min(self.top_k, K)
        _, knn_idx = _knn_points(
            xyz.unsqueeze(0), seed_xyz.unsqueeze(0), k=k
        )
        knn_idx = knn_idx.squeeze(0)  # [N, k]

        # Gather candidate seed data
        f_j = seed_features[knn_idx]   # [N, k, D]
        xyz_j = seed_xyz[knn_idx]      # [N, k, 3]
        n_j = seed_normals[knn_idx]    # [N, k, 3]

        # 2) Learned affinity
        f_i = features.unsqueeze(1).expand_as(f_j)      # [N, k, D]
        interaction = torch.cat([f_i, f_j, f_i - f_j], dim=-1)  # [N, k, 3D]
        affinity = self.affinity_mlp(interaction).squeeze(-1)     # [N, k]

        # 3) Geometric bias: distance + normal consistency
        xyz_i = xyz.unsqueeze(1).expand_as(xyz_j)
        n_i = normals.unsqueeze(1).expand_as(n_j)

        dist_sq = (xyz_i - xyz_j).pow(2).sum(dim=-1)          # [N, k]
        sigma_sq = dist_sq.mean().clamp(min=1e-4).detach()     # scalar, no grad
        normal_sim = (n_i * n_j).sum(dim=-1).abs()             # [N, k]

        geo_bias = -dist_sq / sigma_sq - self.lambda_normal * (1.0 - normal_sim)

        logits = affinity + geo_bias

        # 4) Gumbel-Softmax (train) / argmax (eval)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            hard_idx = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(1, hard_idx, 1.0)

        return knn_idx, weights
