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

import math

import torch
import torch.nn as nn


class VoxelCenterPredictor(nn.Module):
    """Predict Gaussian seed centers via offset voting + voxel pooling.

    Per-point offset MLP adjusts positions, then voxelization groups
    nearby voted positions. scatter_mean produces seed centers [V, 3]
    and aggregated features [V, D] where V = number of occupied voxels.

    If V > max_K, coarse-grid deduplication reduces seeds to at most max_K:
    fine voxels are grouped into a coarser grid (scale = ceil(sqrt(V/max_K))),
    and the highest-point-count representative is kept per coarse cell.
    """

    def __init__(self, dim: int = 64, voxel_size: float = 0.3, max_K: int = 8000):
        super().__init__()
        self.voxel_size = voxel_size
        self.max_K = max_K

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

        # Coarse-grid deduplication: if V > max_K, group fine voxels into a
        # coarser grid and keep the highest-point-count representative per cell.
        # scale = ceil((V/max_K)^(1/3)) for 3D grouping.
        #
        # Phase 1: coarse-grid selects one spatial representative per region
        #          → C < max_K (spatial coverage, but under-utilises budget)
        # Phase 2: fill remaining (max_K - C) slots with highest-count voxels
        #          not already selected → utilise full max_K budget
        if V > self.max_K:
            scale = max(2, math.ceil((V / self.max_K) ** (1 / 3)))
            coarse = torch.floor(voxel_centers.detach() / (self.voxel_size * scale)).long()
            coarse = coarse - coarse.min(0).values
            cdims = coarse.max(0).values + 1
            coarse_lin = (coarse[:, 0] * cdims[1] + coarse[:, 1]) * cdims[2] + coarse[:, 2]

            # Sort by (coarse_lin asc, voxel_count desc) via composite integer key
            max_cnt = voxel_count.long().max()
            sort_key = coarse_lin * (max_cnt + 1) + (max_cnt - voxel_count.long())
            order = sort_key.argsort()

            # First occurrence in sorted order = highest-count voxel per coarse cell
            sorted_coarse = coarse_lin[order]
            is_first = torch.cat([
                torch.ones(1, dtype=torch.bool, device=device),
                sorted_coarse[1:] != sorted_coarse[:-1],
            ])
            keep_coarse = order[is_first]  # [C] spatial representatives

            if keep_coarse.shape[0] >= self.max_K:
                # More coarse cells than budget: keep top by count
                keep = keep_coarse[voxel_count[keep_coarse].topk(self.max_K).indices]
            else:
                # Phase 2: fill remaining budget with highest-count non-selected voxels
                n_fill = self.max_K - keep_coarse.shape[0]
                selected_mask = torch.zeros(V, dtype=torch.bool, device=device)
                selected_mask[keep_coarse] = True
                remaining = (~selected_mask).nonzero(as_tuple=True)[0]
                if remaining.shape[0] > 0:
                    n_fill = min(n_fill, remaining.shape[0])
                    fill = remaining[voxel_count[remaining].topk(n_fill).indices]
                    keep = torch.cat([keep_coarse, fill])
                else:
                    keep = keep_coarse

            voxel_centers = voxel_centers[keep]
            voxel_feats = voxel_feats[keep]
            V = voxel_centers.shape[0]

        return {
            "vote_xyz": vote_xyz,            # [N, 3]
            "offset": offset,                # [N, 3]
            "voxel_centers": voxel_centers,  # [V, 3]  V <= max_K after capping
            "voxel_feats": voxel_feats,      # [V, D]
            # NOTE: voxel_ids maps points to the ORIGINAL (pre-cap) fine voxel
            # indices. Do NOT use voxel_ids to index into voxel_centers/voxel_feats
            # when capping has occurred (V_original > max_K).
            "voxel_ids": voxel_ids,          # [N] indices into pre-cap voxel space
        }
