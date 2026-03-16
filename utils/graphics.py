"""
Panoramic geometry utilities.
Adapted from GS-LiDAR (ICLR 2025) utils/graphics_utils.py.
"""
import torch
import torch.nn.functional as F


def pano_to_lidar(range_image, vfov, hfov):
    """Convert panoramic depth map to 3D point cloud.

    Args:
        range_image: (1, H, W) or (H, W) depth map.
        vfov: (vfov_min, vfov_max) in degrees.
        hfov: (hfov_min, hfov_max) in degrees.

    Returns:
        points_xyz: (M, 3) valid 3D points.
    """
    if range_image.dim() == 2:
        range_image = range_image.unsqueeze(0)

    mask = range_image > 0
    H, W = range_image.shape[-2:]

    theta, phi = torch.meshgrid(
        torch.arange(H, device=range_image.device, dtype=range_image.dtype),
        torch.arange(W, device=range_image.device, dtype=range_image.dtype),
        indexing="ij",
    )

    vert_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / H * vert_range) * torch.pi / 180

    horiz_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / W * horiz_range) * torch.pi / 180

    # GS-LiDAR convention (nuscenes_loader w2l axis swap applied):
    #   x_cam = sin(theta) * sin(phi)
    #   z_cam = sin(theta) * cos(phi)
    #   y_cam = -cos(theta)
    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)

    points_xyz = (directions * range_image)[:, mask[0]].permute(1, 0)
    return points_xyz


def depth_to_normal(range_image, vfov, hfov):
    """Compute surface normals from panoramic depth map.

    Args:
        range_image: (1, H, W) depth map.
        vfov: (vfov_min, vfov_max) in degrees.
        hfov: (hfov_min, hfov_max) in degrees.

    Returns:
        normal_map: (3, H, W) surface normals.
    """
    H, W = range_image.shape[-2:]

    theta, phi = torch.meshgrid(
        torch.arange(H, device=range_image.device, dtype=range_image.dtype),
        torch.arange(W, device=range_image.device, dtype=range_image.dtype),
        indexing="ij",
    )

    vert_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / H * vert_range) * torch.pi / 180

    horiz_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / W * horiz_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)

    points = directions * range_image
    output = torch.zeros_like(points)
    ddx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    ddy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = F.normalize(torch.cross(ddx, ddy, dim=0), dim=0)
    output[:, 1:-1, 1:-1] = normal_map
    return output
