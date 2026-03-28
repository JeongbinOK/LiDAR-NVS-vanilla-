"""Stage 3: Differentiable Soft Clustering.

Iterative soft k-means using attention-based assignment.
Fully differentiable — gradients flow through soft assignments
back to vote positions and backbone features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffSoftClustering(nn.Module):
    """Differentiable soft clustering via iterative attention-based k-means.

    Centers are initialized from voted seed positions, then refined
    through T iterations of soft assignment + weighted mean update.
    """

    def __init__(self, num_iters: int = 4, feature_weight: float = 0.1):
        super().__init__()
        self.num_iters = num_iters
        self.feature_weight = feature_weight

    def forward(
        self,
        vote_xyz: torch.Tensor,
        features: torch.Tensor,
        seed_idx: torch.Tensor,
        tau: float = 1.0,
    ) -> dict:
        """
        Args:
            vote_xyz: [N, 3] voted positions (xyz + offset)
            features: [N, D] per-point backbone features
            seed_idx: [K] indices of initial seed points
            tau: softmax temperature (lower = harder assignment)

        Returns:
            dict with centers, center_feats, assign
        """
        N, D = features.shape
        K = seed_idx.shape[0]

        centers = vote_xyz[seed_idx].clone()       # [K, 3]
        center_feats = features[seed_idx].clone()  # [K, D]

        for _ in range(self.num_iters):
            # Spatial distance
            spatial_dist = torch.cdist(vote_xyz, centers)  # [N, K]

            # Feature distance (optional, weighted)
            if self.feature_weight > 0:
                feat_dist = torch.cdist(features, center_feats)  # [N, K]
                dist = spatial_dist + self.feature_weight * feat_dist
            else:
                dist = spatial_dist

            # Soft assignment via attention
            assign = F.softmax(-dist.pow(2) / tau, dim=-1)  # [N, K]

            # Weighted center update
            w = assign.sum(dim=0).clamp(min=1e-8)  # [K]
            centers = (assign.T @ vote_xyz) / w.unsqueeze(-1)       # [K, 3]
            center_feats = (assign.T @ features) / w.unsqueeze(-1)  # [K, D]

        return {
            "centers": centers,           # [K, 3]
            "center_feats": center_feats, # [K, D]
            "assign": assign,             # [N, K]
            "weight_sum": w,              # [K]
        }
