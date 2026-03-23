"""Ground detection and ground-aware adaptive seeding."""

import torch

from .config import ClusteringConfig
from .seeding import compute_adaptive_voxel_size, farthest_point_sampling, voxel_seeding


def detect_ground(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    cfg: ClusteringConfig,
) -> torch.Tensor:
    """Detect ground points using normal direction + RANSAC.

    Args:
        xyz: [N, 3] point positions
        normals: [N, 3] per-point normals
        cfg: configuration

    Returns:
        ground_mask: [N] boolean mask
    """
    z_min, z_max = cfg.ground_z_range

    # Pre-filter by z range
    z_candidates = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)

    # Normal-based: ground normals should be roughly vertical
    up = torch.tensor([0.0, 0.0, 1.0], device=xyz.device)
    cos_angle = (normals * up).sum(dim=1).abs()
    normal_candidates = cos_angle > cfg.ground_normal_thresh

    candidates = z_candidates & normal_candidates

    if candidates.sum() < 10:
        return torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)

    # RANSAC plane fitting on candidate points
    ground_mask = _ransac_plane(xyz, candidates, n_iters=50, dist_thresh=0.15)

    return ground_mask


def _ransac_plane(
    xyz: torch.Tensor,
    candidates: torch.Tensor,
    n_iters: int = 50,
    dist_thresh: float = 0.15,
) -> torch.Tensor:
    """RANSAC plane fitting on candidate points.

    Returns:
        inlier_mask: [N] boolean mask of ground inliers
    """
    cand_idx = candidates.nonzero(as_tuple=True)[0]
    cand_pts = xyz[cand_idx]
    M = cand_pts.shape[0]

    best_count = 0
    best_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)

    for _ in range(n_iters):
        sample = torch.randint(0, M, (3,), device=xyz.device)
        p0, p1, p2 = cand_pts[sample[0]], cand_pts[sample[1]], cand_pts[sample[2]]

        normal = torch.cross(p1 - p0, p2 - p0)
        norm_len = normal.norm()
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len

        dists = ((cand_pts - p0) * normal).sum(dim=1).abs()
        inliers = dists < dist_thresh
        count = inliers.sum().item()

        if count > best_count:
            best_count = count
            best_mask.zero_()
            best_mask[cand_idx[inliers]] = True

    return best_mask


def ground_aware_seeds(
    xyz: torch.Tensor,
    curvature: torch.Tensor,
    normals: torch.Tensor,
    cfg: ClusteringConfig,
) -> tuple:
    """Compute seeds with ground-aware voxel sizing.

    Ground regions use larger voxels (fewer, bigger clusters).
    Non-ground regions use normal voxels with curvature FPS augmentation.

    Returns:
        (seed_indices, ground_mask): [S] global indices, [N] bool mask
    """
    ground_mask = detect_ground(xyz, normals, cfg)
    base_voxel_size = compute_adaptive_voxel_size(xyz, cfg.target_cluster_size)

    all_seeds = []

    # Ground seeds — larger voxels
    ground_idx = ground_mask.nonzero(as_tuple=True)[0]
    if ground_idx.shape[0] > 0:
        ground_voxel_size = base_voxel_size * cfg.ground_voxel_scale
        local_seeds = voxel_seeding(xyz[ground_idx], ground_voxel_size)
        all_seeds.append(ground_idx[local_seeds])

    # Non-ground seeds — normal voxels + FPS augmentation
    non_ground_idx = (~ground_mask).nonzero(as_tuple=True)[0]
    if non_ground_idx.shape[0] > 0:
        ng_xyz = xyz[non_ground_idx]
        ng_curv = curvature[non_ground_idx]

        local_voxel_seeds = voxel_seeding(ng_xyz, base_voxel_size)
        all_seeds.append(non_ground_idx[local_voxel_seeds])

        # FPS augmentation in high-curvature regions
        if ng_curv.shape[0] > 1:
            high_curv_mask = ng_curv > ng_curv.quantile(cfg.curvature_quantile)
            high_curv_local = high_curv_mask.nonzero(as_tuple=True)[0]
            if high_curv_local.shape[0] > 0:
                n_extra = max(int(local_voxel_seeds.shape[0] * cfg.fps_extra_ratio), 1)
                n_extra = min(n_extra, high_curv_local.shape[0])
                fps_local = farthest_point_sampling(ng_xyz[high_curv_local], n_extra)
                all_seeds.append(non_ground_idx[high_curv_local[fps_local]])

    if all_seeds:
        seed_indices = torch.unique(torch.cat(all_seeds))
    else:
        seed_indices = torch.arange(min(10, xyz.shape[0]), device=xyz.device)

    return seed_indices, ground_mask
