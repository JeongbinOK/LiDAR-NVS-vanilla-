import torch
import math
import numpy as np
import torch.nn.functional as F


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fx, fy, w, h):
    top = cy / fy * znear
    bottom = -(h - cy) / fy * znear

    left = -(w - cx) / fx * znear
    right = cx / fx * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _pano_theta_phi(range_image, vfov, hfov, row_to_theta=None):
    panorama_height, panorama_width = range_image.shape[-2:]
    row, col = torch.meshgrid(
        torch.arange(panorama_height, device=range_image.device),
        torch.arange(panorama_width, device=range_image.device),
        indexing="ij",
    )
    if row_to_theta is None:
        vertical_degree_range = vfov[1] - vfov[0]
        theta = (90 - vfov[1] + row / panorama_height * vertical_degree_range) * torch.pi / 180
    else:
        row_to_theta = torch.as_tensor(
            row_to_theta,
            device=range_image.device,
            dtype=range_image.dtype,
        )
        if row_to_theta.numel() != panorama_height:
            raise ValueError(
                f"row_to_theta length {row_to_theta.numel()} does not match height {panorama_height}"
            )
        theta = row_to_theta.view(panorama_height, 1).expand(panorama_height, panorama_width)

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + col / panorama_width * horizontal_degree_range) * torch.pi / 180
    return theta, phi


def pano_to_lidar(range_image, vfov, hfov, row_to_theta=None):
    mask = range_image > 0

    theta, phi = _pano_theta_phi(range_image, vfov, hfov, row_to_theta)

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)

    points_xyz = (directions * range_image)[:, mask[0]].permute(1, 0)

    return points_xyz


LIDAR4D_CD_MAX_RANGE_M = 80.0
LIDAR4D_RAYDROP_THRESHOLD = 0.5


def lidar4d_range_image_to_points(
    range_image,
    vfov,
    hfov,
    row_to_theta=None,
    *,
    raydrop=None,
    raydrop_threshold=LIDAR4D_RAYDROP_THRESHOLD,
    min_range=0.0,
    max_range=LIDAR4D_CD_MAX_RANGE_M,
):
    """Build the point set used by this repo's LiDAR4D-style CD protocol.

    Official LiDAR4D removes predicted no-return rays with a hard 0.5 mask and
    back-projects the remaining non-zero range pixels; its GT range-view
    preprocessing excludes returns at or beyond 80 metres. GS-LiDAR makes the
    80 m filtering explicit for both point sets. We use that symmetric rule so
    training and evaluation cannot disagree on support. ``raydrop`` follows
    this repository's convention (1 means no return), which is the inverse of
    LiDAR4D's official return-mask convention.

    The binary masks are intentionally detached: Chamfer gradients should flow
    to retained depth values, not through the discontinuous raydrop/range test.
    ``min_range`` is optional because official LiDAR4D does not apply a separate
    near-range crop when constructing its evaluation point clouds.
    """
    if range_image.ndim != 3 or range_image.shape[0] != 1:
        raise ValueError(
            "range_image must have shape [1, H, W], got "
            f"{tuple(range_image.shape)}"
        )
    if raydrop is not None and raydrop.shape != range_image.shape:
        raise ValueError(
            "raydrop must match range_image shape, got "
            f"{tuple(raydrop.shape)} vs {tuple(range_image.shape)}"
        )

    valid = range_image.detach() > float(min_range)
    if max_range is not None:
        valid = valid & (range_image.detach() < float(max_range))
    if raydrop is not None:
        valid = valid & (raydrop.detach() <= float(raydrop_threshold))

    masked_range = range_image * valid.to(dtype=range_image.dtype)
    return pano_to_lidar(
        masked_range,
        vfov,
        hfov,
        row_to_theta=row_to_theta,
    )
