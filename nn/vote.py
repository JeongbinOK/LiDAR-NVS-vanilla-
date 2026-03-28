"""Stage 2: Voxel-based center seeding with learned offsets.

Each point predicts an offset to its cluster center. Offset-adjusted
positions (vote_xyz) are voxelized, and per-voxel mean pooling produces
seed centers and features.

Gradient path:
  L_surface → mu → centers (soft clustering) → voxel_centers (scatter_mean)
  → vote_xyz → offset_mlp → backbone features

scatter_mean is differentiable. Voxel assignment is discrete (like
PTv3's own voxelization), but values within each voxel get gradient.
"""

import torch
import torch.nn as nn


class VoxelCenterPredictor(nn.Module):
    """Predict Gaussian seed centers via offset voting + voxel pooling.

    Per-point offset MLP adjusts positions, then voxelization groups
    nearby voted positions. scatter_mean produces seed centers [V, 3]
    and aggregated features [V, D] where V = number of occupied voxels.
    """

    def __init__(self, dim: int = 64, voxel_size: float = 0.3):
        super().__init__()
        self.voxel_size = voxel_size

        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
        )

        # Zero-init: initial vote_xyz = xyz (no offset)
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
    ) -> dict:
        """
        Args:
            features: [N, D] per-point backbone features
            xyz: [N, 3] point positions

        Returns:
            dict with vote_xyz, offset, voxel_centers, voxel_feats, voxel_ids
        """
        N, D = features.shape
        device = features.device

        offset = self.offset_mlp(features)    # [N, 3]
        vote_xyz = xyz + offset               # [N, 3]

        # Voxelize vote_xyz
        voxel_coords = torch.floor(vote_xyz.detach() / self.voxel_size).long()

        # Shift to non-negative for linear indexing
        coord_min = voxel_coords.min(dim=0).values
        voxel_coords = voxel_coords - coord_min
        dims = voxel_coords.max(dim=0).values + 1

        # Collision-free linear hash
        voxel_linear = (voxel_coords[:, 0] * dims[1] * dims[2]
                        + voxel_coords[:, 1] * dims[2]
                        + voxel_coords[:, 2])

        unique_linear, voxel_ids = torch.unique(voxel_linear, return_inverse=True)
        V = unique_linear.shape[0]

        # scatter_mean for positions and features (differentiable)
        expand_ids_3 = voxel_ids.unsqueeze(1).expand(-1, 3)
        expand_ids_D = voxel_ids.unsqueeze(1).expand(-1, D)

        voxel_xyz_sum = torch.zeros(V, 3, device=device, dtype=vote_xyz.dtype)
        voxel_feat_sum = torch.zeros(V, D, device=device, dtype=features.dtype)
        voxel_count = torch.zeros(V, device=device, dtype=vote_xyz.dtype)

        voxel_xyz_sum.scatter_add_(0, expand_ids_3, vote_xyz)
        voxel_feat_sum.scatter_add_(0, expand_ids_D, features)
        voxel_count.scatter_add_(
            0, voxel_ids, torch.ones(N, device=device, dtype=vote_xyz.dtype)
        )

        voxel_count = voxel_count.clamp(min=1)
        voxel_centers = voxel_xyz_sum / voxel_count.unsqueeze(1)  # [V, 3]
        voxel_feats = voxel_feat_sum / voxel_count.unsqueeze(1)   # [V, D]

        return {
            "vote_xyz": vote_xyz,           # [N, 3]
            "offset": offset,               # [N, 3]
            "voxel_centers": voxel_centers,  # [V, 3]
            "voxel_feats": voxel_feats,      # [V, D]
            "voxel_ids": voxel_ids,          # [N] point-to-voxel mapping
        }
