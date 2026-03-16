"""
Miscellaneous utilities: seed, ego mask, point-to-range-image projection,
beam direction computation, and feature warping grid construction.
"""
import math
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


def compute_beam_dirs(H, W, vfov_deg, hfov_deg):
    """Compute unit beam direction vectors for each pixel in a range image.

    This is the inverse of the angle→pixel mapping in ``points_to_pano``.
    nuScenes convention: x=right, y=forward, z=up.

    Args:
        H, W: range image dimensions.
        vfov_deg: [vfov_min, vfov_max] in degrees.
        hfov_deg: [hfov_min, hfov_max] in degrees.

    Returns:
        beam_dirs: [3, H, W] unit direction vectors (x, y, z).
    """
    # Pixel indices
    u = torch.arange(W, dtype=torch.float32)  # [W]
    v = torch.arange(H, dtype=torch.float32)  # [H]

    # Inverse of: u = (phi_deg - hfov[0]) / (hfov[1] - hfov[0]) * W
    phi_deg = hfov_deg[0] + u * (hfov_deg[1] - hfov_deg[0]) / W
    # Inverse of: v = (vfov[1] - theta_deg) / (vfov[1] - vfov[0]) * H
    theta_deg = vfov_deg[1] - v * (vfov_deg[1] - vfov_deg[0]) / H

    phi = phi_deg * (math.pi / 180.0)      # [W]
    theta = theta_deg * (math.pi / 180.0)   # [H]

    # Broadcast: theta[H,1] x phi[1,W]
    theta = theta[:, None]  # [H, 1]
    phi = phi[None, :]      # [1, W]

    cos_theta = torch.cos(theta)
    x = torch.sin(phi) * cos_theta   # [H, W]
    y = torch.cos(phi) * cos_theta   # [H, W]
    z = torch.sin(theta).expand_as(x)  # [H, W]

    beam_dirs = torch.stack([x, y, z], dim=0)  # [3, H, W]
    return beam_dirs


def compute_warp_grid(depth_0, beam_dirs, T_0_to_1, H, W, vfov_deg, hfov_deg):
    """Compute a warp grid that maps Frame 0 RI pixels to Frame 1 RI pixels.

    For each pixel (u,v) in ri_0:
      1) Reconstruct 3D point in Frame 0: P0 = depth * beam_dir(u,v)
      2) Transform to Frame 1: P1 = T_0_to_1 @ P0
      3) Project P1 to Frame 1 RI pixel coordinates
      4) Normalize to [-1, 1] for grid_sample

    Args:
        depth_0: [B, 1, H, W] depth channel of ri_0.
        beam_dirs: [3, H, W] precomputed unit beam directions.
        T_0_to_1: [B, 4, 4] transform from frame 0 to frame 1.
        H, W: range image dimensions.
        vfov_deg: [vfov_min, vfov_max] in degrees.
        hfov_deg: [hfov_min, hfov_max] in degrees.

    Returns:
        warp_grid: [B, H, W, 2] normalized grid for F.grid_sample.
    """
    B = depth_0.shape[0]
    device = depth_0.device

    # 1) Reconstruct 3D points in Frame 0: [B, 3, H, W]
    dirs = beam_dirs.to(device)  # [3, H, W]
    pts_3d = depth_0 * dirs.unsqueeze(0)  # [B, 3, H, W]

    # 2) Transform to Frame 1 coordinates
    # Reshape to [B, 3, H*W] for matrix multiply
    pts_flat = pts_3d.reshape(B, 3, H * W)  # [B, 3, N]
    ones = torch.ones(B, 1, H * W, device=device, dtype=pts_flat.dtype)
    pts_homo = torch.cat([pts_flat, ones], dim=1)  # [B, 4, N]

    # T_0_to_1 @ pts_homo: [B, 4, 4] @ [B, 4, N] -> [B, 4, N]
    pts_1 = torch.bmm(T_0_to_1, pts_homo)  # [B, 4, N]
    x1 = pts_1[:, 0]  # [B, N]
    y1 = pts_1[:, 1]
    z1 = pts_1[:, 2]

    # 3) Spherical projection in Frame 1
    depth_1 = torch.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2).clamp(min=1e-6)
    phi_1 = torch.atan2(x1, y1)  # [B, N]
    theta_1 = torch.asin((z1 / depth_1).clamp(-1, 1))

    phi_1_deg = phi_1 * (180.0 / math.pi)
    theta_1_deg = theta_1 * (180.0 / math.pi)

    # Pixel coordinates in Frame 1 RI
    u1 = (phi_1_deg - hfov_deg[0]) / (hfov_deg[1] - hfov_deg[0]) * W
    v1 = (vfov_deg[1] - theta_1_deg) / (vfov_deg[1] - vfov_deg[0]) * H

    # Normalize to [-1, 1] for grid_sample
    u1_norm = 2.0 * u1 / max(W - 1, 1) - 1.0
    v1_norm = 2.0 * v1 / max(H - 1, 1) - 1.0

    # 4) Handle invalid pixels: depth=0 in ri_0 → set grid to -10 (out of bounds)
    invalid = (depth_0.reshape(B, H * W) < 1e-6)
    u1_norm[invalid] = -10.0
    v1_norm[invalid] = -10.0

    # Stack and reshape: [B, N, 2] -> [B, H, W, 2]
    warp_grid = torch.stack([u1_norm, v1_norm], dim=-1)  # [B, N, 2]
    warp_grid = warp_grid.reshape(B, H, W, 2)

    return warp_grid


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
