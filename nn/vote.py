"""Stage 2: Learned Center Prediction via offset voting.

Each point predicts (1) an offset to its cluster center and
(2) a centerness score indicating confidence. Seeds are selected
as the top-K voted positions by centerness score.
"""

import torch
import torch.nn as nn


class LearnedCenterPredictor(nn.Module):
    """Predict Gaussian center candidates via per-point voting.

    Each point votes for where its cluster center should be by predicting
    a 3D offset. Centerness score indicates how confident the vote is.
    Top-K by centerness produces seed locations at voted positions.
    """

    def __init__(self, dim: int = 64, target_cluster_size: int = 30):
        super().__init__()
        self.target_cluster_size = target_cluster_size

        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

        # Zero-init offset: initial vote_xyz = xyz (no offset)
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        tau: float = 1.0,
    ) -> dict:
        """
        Args:
            features: [N, D] per-point backbone features
            xyz: [N, 3] point positions
            tau: Gumbel noise temperature (annealed 1.0 -> 0.1)

        Returns:
            dict with vote_xyz, centerness, offset, seed_idx, K
        """
        N = features.shape[0]
        device = features.device

        offset = self.offset_mlp(features)                    # [N, 3]
        score = self.score_mlp(features).squeeze(-1)          # [N]
        centerness = torch.sigmoid(score)                     # [N]
        vote_xyz = xyz + offset                               # [N, 3]

        K = max(N // self.target_cluster_size, 10)

        if self.training:
            # Gumbel noise for exploration during training
            gumbel = -torch.log(
                -torch.log(torch.rand(N, device=device).clamp(1e-8, 1.0)).clamp(1e-8)
            )
            _, seed_idx = (score + gumbel * tau).topk(K)
        else:
            _, seed_idx = centerness.topk(K)

        return {
            "vote_xyz": vote_xyz,       # [N, 3]
            "centerness": centerness,   # [N]
            "offset": offset,           # [N, 3]
            "seed_idx": seed_idx,       # [K]
            "K": K,
        }
