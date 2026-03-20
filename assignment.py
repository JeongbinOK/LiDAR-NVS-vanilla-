"""Stage 3: Normal-aware point-to-seed assignment."""

import torch

from .config import ClusteringConfig
from .geometry import _knn_points


def assign_points(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    seed_indices: torch.Tensor,
    cfg: ClusteringConfig,
) -> torch.Tensor:
    """Normal-aware assignment of points to nearest seeds.

    Args:
        xyz: [N, 3] all points
        normals: [N, 3] per-point normals
        seed_indices: [S] indices into xyz for seed points
        cfg: configuration

    Returns:
        assignments: [N] cluster ID (index into seed_indices) for each point
    """
    seeds = xyz[seed_indices]          # [S, 3]
    seed_normals = normals[seed_indices]  # [S, 3]

    # Auto-tune sigma_spatial: median distance to nearest seed
    _, nearest_idx = _knn_points(xyz.unsqueeze(0), seeds.unsqueeze(0), k=1)
    nearest_dists = (xyz - seeds[nearest_idx.squeeze()]).pow(2).sum(dim=1)
    sigma_sq = nearest_dists.median().clamp(min=1e-4)

    # Find C=candidate_k nearest seeds for each point
    _, candidate_ids = _knn_points(
        xyz.unsqueeze(0), seeds.unsqueeze(0), k=cfg.candidate_k
    )
    candidate_ids = candidate_ids.squeeze(0)  # [N, C]

    # Spatial cost: ||p - s||^2 / sigma^2
    candidate_seeds = seeds[candidate_ids]  # [N, C, 3]
    spatial_cost = (xyz.unsqueeze(1) - candidate_seeds).pow(2).sum(dim=-1) / sigma_sq

    # Normal cost: 1 - (n_p · n_s)^2
    candidate_normals = seed_normals[candidate_ids]  # [N, C, 3]
    dot_prod = (normals.unsqueeze(1) * candidate_normals).sum(dim=-1)  # [N, C]
    normal_cost = 1.0 - dot_prod.pow(2)

    # Combined cost
    cost = spatial_cost + cfg.lambda_normal * normal_cost  # [N, C]

    # Assign to minimum cost candidate
    best_local = cost.argmin(dim=1)  # [N]
    assignments = candidate_ids.gather(1, best_local.unsqueeze(1)).squeeze(1)  # [N]

    return assignments
