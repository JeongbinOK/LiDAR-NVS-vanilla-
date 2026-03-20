"""Stage 2: Adaptive seed placement via voxel-based seeding + curvature augmentation."""

import torch

from .config import ClusteringConfig


def compute_adaptive_voxel_size(xyz: torch.Tensor, target_cluster_size: int) -> float:
    """Compute voxel size to yield ~target_cluster_size points per voxel on average."""
    N = xyz.shape[0]
    # Bounding box volume
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    bbox_vol = (maxs - mins).prod().item()
    # target_clusters = N / target_cluster_size
    # voxel_vol = bbox_vol / target_clusters
    # voxel_size = voxel_vol^(1/3)
    target_clusters = max(N / target_cluster_size, 1)
    voxel_vol = bbox_vol / target_clusters
    voxel_size = max(voxel_vol ** (1.0 / 3.0), 0.1)
    return voxel_size


def voxel_seeding(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Select one seed per occupied voxel (closest to voxel centroid).

    Returns:
        seed_indices: [S] indices into xyz
    """
    voxel_coords = torch.floor(xyz / voxel_size).long()  # [N, 3]

    # Encode voxel coords to unique IDs
    mins = voxel_coords.min(dim=0).values
    shifted = voxel_coords - mins  # non-negative
    dims = shifted.max(dim=0).values + 1
    voxel_ids = (shifted[:, 0] * dims[1] * dims[2]
                 + shifted[:, 1] * dims[2]
                 + shifted[:, 2])  # [N]

    # For each unique voxel, find point closest to voxel centroid
    unique_ids, inverse = torch.unique(voxel_ids, return_inverse=True)
    num_voxels = unique_ids.shape[0]

    # Scatter mean to get centroids
    centroids = torch.zeros(num_voxels, 3, device=xyz.device)
    counts = torch.zeros(num_voxels, device=xyz.device)
    centroids.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), xyz)
    counts.scatter_add_(0, inverse, torch.ones(xyz.shape[0], device=xyz.device))
    centroids /= counts.unsqueeze(1).clamp(min=1)

    # For each point, distance to its voxel centroid
    dist_to_centroid = (xyz - centroids[inverse]).pow(2).sum(dim=1)  # [N]

    # Per-voxel argmin
    seed_indices = torch.zeros(num_voxels, dtype=torch.long, device=xyz.device)
    # Initialize with large distance
    min_dists = torch.full((num_voxels,), float("inf"), device=xyz.device)

    # Process — scatter_min not available in vanilla PyTorch, use loop-free approach
    # For each voxel, we need the point with minimum distance
    # Use: for ties, just pick any
    for i in range(num_voxels):
        voxel_mask = inverse == i
        voxel_dists = dist_to_centroid[voxel_mask]
        voxel_points = voxel_mask.nonzero(as_tuple=True)[0]
        seed_indices[i] = voxel_points[voxel_dists.argmin()]

    return seed_indices


def _voxel_seeding_fast(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Vectorized voxel seeding using sorting."""
    voxel_coords = torch.floor(xyz / voxel_size).long()  # [N, 3]
    mins = voxel_coords.min(dim=0).values
    shifted = voxel_coords - mins
    dims = shifted.max(dim=0).values + 1
    voxel_ids = (shifted[:, 0] * dims[1] * dims[2]
                 + shifted[:, 1] * dims[2]
                 + shifted[:, 2])

    unique_ids, inverse = torch.unique(voxel_ids, return_inverse=True)
    num_voxels = unique_ids.shape[0]

    # Compute centroids
    centroids = torch.zeros(num_voxels, 3, device=xyz.device)
    counts = torch.zeros(num_voxels, device=xyz.device)
    centroids.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), xyz)
    counts.scatter_add_(0, inverse, torch.ones(xyz.shape[0], device=xyz.device))
    centroids /= counts.unsqueeze(1).clamp(min=1)

    dist_to_centroid = (xyz - centroids[inverse]).pow(2).sum(dim=1)

    # Sort by (voxel_id, distance) to get per-voxel closest point
    sort_key = inverse.float() * 1e10 + dist_to_centroid
    sorted_order = sort_key.argsort()
    sorted_voxel = inverse[sorted_order]

    # First occurrence of each voxel in sorted order = closest point
    change_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    change_mask[1:] = sorted_voxel[1:] != sorted_voxel[:-1]
    seed_indices = sorted_order[change_mask]

    return seed_indices


def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """GPU farthest point sampling.

    Args:
        xyz: [M, 3] points to sample from
        n_samples: number of points to select

    Returns:
        indices: [n_samples] indices into xyz
    """
    M = xyz.shape[0]
    if n_samples >= M:
        return torch.arange(M, device=xyz.device)

    indices = torch.zeros(n_samples, dtype=torch.long, device=xyz.device)
    distances = torch.full((M,), float("inf"), device=xyz.device)

    # Start from random point
    indices[0] = torch.randint(0, M, (1,), device=xyz.device)

    for i in range(1, n_samples):
        last = xyz[indices[i - 1]].unsqueeze(0)  # [1, 3]
        d = (xyz - last).pow(2).sum(dim=1)        # [M]
        distances = torch.min(distances, d)
        indices[i] = distances.argmax()

    return indices


def compute_seeds(
    xyz: torch.Tensor,
    curvature: torch.Tensor,
    cfg: ClusteringConfig,
) -> torch.Tensor:
    """Stage 2: Adaptive seed placement.

    Tier 1: Voxel-based seeds
    Tier 2: Extra FPS seeds in high-curvature regions

    Args:
        xyz: [N, 3]
        curvature: [N]
        cfg: configuration

    Returns:
        seed_indices: [S] indices into xyz
    """
    # Tier 1: Voxel seeding
    voxel_size = compute_adaptive_voxel_size(xyz, cfg.target_cluster_size)
    voxel_seeds = _voxel_seeding_fast(xyz, voxel_size)

    # Tier 2: High-curvature FPS augmentation
    high_curv_mask = curvature > curvature.quantile(cfg.curvature_quantile)
    high_curv_xyz = xyz[high_curv_mask]

    if high_curv_xyz.shape[0] > 0:
        n_extra = max(int(voxel_seeds.shape[0] * cfg.fps_extra_ratio), 1)
        n_extra = min(n_extra, high_curv_xyz.shape[0])
        extra_local = farthest_point_sampling(high_curv_xyz, n_extra)
        # Map back to global indices
        global_indices = high_curv_mask.nonzero(as_tuple=True)[0]
        extra_seeds = global_indices[extra_local]
        # Merge, removing duplicates
        all_seeds = torch.cat([voxel_seeds, extra_seeds])
        all_seeds = torch.unique(all_seeds)
    else:
        all_seeds = voxel_seeds

    return all_seeds
