"""
Miscellaneous utilities: seed, ego mask, point-to-range-image projection.
"""
import random
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ego_mask(points, radius=2.5):
    """Remove points within `radius` of the ego vehicle origin.

    Args:
        points: (N, 4) xyz + intensity.
        radius: exclusion radius in meters.

    Returns:
        Filtered points (M, 4).
    """
    dist = torch.norm(points[:, :3], dim=1)
    return points[dist > radius]


def points_to_pano(points, vfov_deg, hfov_deg, H, W):
    """Project (N, 4) xyz+intensity points to a panoramic range image.

    Args:
        points: (N, 4) tensor — x, y, z, intensity.
        vfov_deg: [vfov_min, vfov_max] vertical FOV in degrees.
        hfov_deg: [hfov_min, hfov_max] horizontal FOV in degrees.
        H: range image height (number of beams).
        W: range image width.

    Returns:
        range_img: (5, H, W) — channels: (depth, x, y, z, intensity).
        valid_mask: (H, W) bool — which pixels have valid points.
        uv_coords: (N, 2) — (u, v) pixel coordinates for each input point
                   (-1 for points outside FOV).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]
    depth = torch.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Spherical angles (nuScenes LiDAR: x=right, y=forward, z=up)
    phi = torch.atan2(x, y)  # horizontal angle [-pi, pi]
    theta = torch.asin(z / depth.clamp(min=1e-6))  # vertical angle

    phi_deg = phi * 180.0 / torch.pi
    theta_deg = theta * 180.0 / torch.pi

    # Filter by vertical FOV
    valid = (theta_deg >= vfov_deg[0]) & (theta_deg <= vfov_deg[1]) & (depth > 0.1)

    # Pixel coordinates
    u = ((phi_deg - hfov_deg[0]) / (hfov_deg[1] - hfov_deg[0]) * W).long()
    v = ((vfov_deg[1] - theta_deg) / (vfov_deg[1] - vfov_deg[0]) * H).long()

    # Clamp to valid range
    u = u.clamp(0, W - 1)
    v = v.clamp(0, H - 1)

    # Store per-point uv coords (-1 for invalid)
    uv_coords = torch.full((points.shape[0], 2), -1, dtype=torch.long,
                           device=points.device)
    uv_coords[valid, 0] = u[valid]
    uv_coords[valid, 1] = v[valid]

    # Z-buffer: keep closest point per pixel
    range_img = torch.zeros(5, H, W, device=points.device, dtype=points.dtype)
    zbuf = torch.full((H, W), float('inf'), device=points.device, dtype=points.dtype)

    valid_u = u[valid]
    valid_v = v[valid]
    valid_depth = depth[valid]
    valid_x = x[valid]
    valid_y = y[valid]
    valid_z = z[valid]
    valid_intensity = intensity[valid]

    # Sort by depth descending so scatter writes closest last
    sort_idx = torch.argsort(valid_depth, descending=True)
    valid_u = valid_u[sort_idx]
    valid_v = valid_v[sort_idx]
    valid_depth = valid_depth[sort_idx]
    valid_x = valid_x[sort_idx]
    valid_y = valid_y[sort_idx]
    valid_z = valid_z[sort_idx]
    valid_intensity = valid_intensity[sort_idx]

    # Linear indices for scatter
    lin_idx = valid_v * W + valid_u

    # Scatter — last write wins (closest since sorted descending)
    range_flat = range_img.view(5, -1)
    range_flat[0].scatter_(0, lin_idx, valid_depth)
    range_flat[1].scatter_(0, lin_idx, valid_x)
    range_flat[2].scatter_(0, lin_idx, valid_y)
    range_flat[3].scatter_(0, lin_idx, valid_z)
    range_flat[4].scatter_(0, lin_idx, valid_intensity)

    valid_mask = range_img[0] > 0

    return range_img, valid_mask, uv_coords


def pano_uv_to_grid(uv_coords, H, W):
    """Convert integer (u, v) pixel coordinates to normalized [-1, 1] grid
    for F.grid_sample.

    Args:
        uv_coords: (N, 2) — column u, row v in pixel space.
        H, W: range image dimensions.

    Returns:
        grid: (1, 1, N, 2) normalized coordinates for grid_sample.
    """
    grid = torch.zeros_like(uv_coords, dtype=torch.float32)
    grid[:, 0] = 2.0 * uv_coords[:, 0].float() / (W - 1) - 1.0  # u -> x
    grid[:, 1] = 2.0 * uv_coords[:, 1].float() / (H - 1) - 1.0  # v -> y
    return grid.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
