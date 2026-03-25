"""Module B: Learned seed generation with Gumbel top-K and ball-query NMS."""

import torch
import torch.nn as nn


class SeedGenerator(nn.Module):
    """Predict per-point seediness scores and select cluster seed points.

    Scores each point as a potential cluster center via a lightweight MLP.
    Training uses Gumbel noise for differentiable top-K exploration;
    inference uses deterministic top-K. Ball-query NMS removes redundant seeds.
    """

    def __init__(self, dim: int = 64, k_max: int = 1500,
                 nms_radius_factor: float = 0.8, target_cluster_size: int = 30):
        super().__init__()
        self.k_max = k_max
        self.nms_radius_factor = nms_radius_factor
        self.target_cluster_size = target_cluster_size

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, xyz: torch.Tensor,
                tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [N, D] per-point features from backbone
            xyz: [N, 3] point positions
            tau: Gumbel noise temperature (annealed 1.0 -> 0.1)

        Returns:
            seed_indices: [K_final] indices into point cloud (after NMS)
            seed_scores: [K_final] scores of selected seeds
            all_scores: [N] seediness scores for all points
            raw_logits: [N] pre-sigmoid logits (for score_proj in soft_assign)
        """
        N = features.shape[0]
        device = features.device

        raw_logits = self.mlp(features).squeeze(-1)        # [N]
        all_scores = torch.sigmoid(raw_logits)              # [N]

        k = min(self.k_max, N)

        if self.training:
            gumbel = -torch.log(
                -torch.log(torch.rand(N, device=device).clamp(1e-8, 1.0)).clamp(1e-8)
            )
            perturbed = all_scores + gumbel * tau
            _, topk_indices = perturbed.topk(k)
        else:
            _, topk_indices = all_scores.topk(k)

        # Ball-query NMS to remove spatially redundant seeds
        nms_radius = self._compute_nms_radius(xyz)
        keep = self._ball_query_nms(
            xyz[topk_indices], all_scores[topk_indices], nms_radius
        )

        seed_indices = topk_indices[keep]

        # Adaptive K cap for train/eval consistency
        k_cap = max(N // self.target_cluster_size, 10)
        if seed_indices.shape[0] > k_cap:
            _, cap_idx = all_scores[seed_indices].topk(k_cap)
            seed_indices = seed_indices[cap_idx]

        seed_scores = all_scores[seed_indices]

        return seed_indices, seed_scores, all_scores, raw_logits

    def _compute_nms_radius(self, xyz: torch.Tensor) -> float:
        """Adaptive NMS radius from point density, robust to outliers.

        Uses 2nd-98th percentile range instead of full bbox to avoid
        inflated volume from long-range LiDAR outliers.
        """
        N = xyz.shape[0]
        p02 = xyz.quantile(0.02, dim=0)
        p98 = xyz.quantile(0.98, dim=0)
        effective_range = (p98 - p02).clamp(min=1.0)
        effective_vol = effective_range.prod().item()
        target_clusters = max(N / self.target_cluster_size, 1)
        voxel_vol = effective_vol / target_clusters
        voxel_size = max(voxel_vol ** (1.0 / 3.0), 0.1)
        return voxel_size * self.nms_radius_factor

    @staticmethod
    def _ball_query_nms(xyz: torch.Tensor, scores: torch.Tensor,
                        radius: float) -> torch.Tensor:
        """Greedy NMS on GPU: suppress lower-scored seeds within radius.

        Uses vectorized mask operations instead of CPU loop.

        Returns:
            keep: [K_final] indices into the input arrays of kept seeds.
        """
        K = xyz.shape[0]
        if K == 0:
            return torch.tensor([], dtype=torch.long, device=xyz.device)

        device = xyz.device

        # Pairwise adjacency mask (GPU)
        dists = torch.cdist(xyz.unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)
        adjacent = dists < radius  # [K, K] bool
        del dists

        # Greedy NMS in score-descending order (all on GPU)
        order = scores.argsort(descending=True).cpu()  # loop var on CPU
        alive = torch.ones(K, dtype=torch.bool, device=device)
        keep = torch.empty(K, dtype=torch.long, device=device)
        n_keep = 0

        for i in order.tolist():
            if not alive[i].item():
                continue
            keep[n_keep] = i
            n_keep += 1
            # Suppress neighbors; un-suppress self (already kept)
            alive &= ~adjacent[i]
            alive[i] = True

        return keep[:n_keep]
