"""Rendering helpers for QGS-Flow Phase A.

Provides:
  - `rotmat_to_quat`        : differentiable [...,3,3] → [...,4] (w,x,y,z)
  - `build_target_lidar_image`: bin a LiDAR sweep onto the spherical grid used
                                 by the rasterizer (first-hit range, first-hit intensity)
  - `build_gt_normal_map`   : estimate per-pixel GT surface normals from a range image
  - `render_primitives`     : wrap a primitive dict into a `LiDARRasterizer` call
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

import diff_quadratic_rasterization as dq

LiDARRasterOutput = dq.LiDARRasterOutput
LiDARRasterizer = dq.LiDARRasterizer
MIDDEPTH_OFFSET = int(getattr(dq, "MIDDEPTH_OFFSET", dq.NUM_CHANNELS + 6))
make_lidar_settings = dq.make_lidar_settings


# ---------------------------------------------------------------------------
# Rotation conversion
# ---------------------------------------------------------------------------

def rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrices to (w, x, y, z) unit quaternions.

    Branch-by-trace formulation. For Phase A the head's R is built as
    R_init @ R_residual with a near-identity residual, so the trace-positive
    branch dominates. The fallback branches are kept for numerical safety
    (180-deg rotations) and to prevent NaN gradients via clamp.

    Args:
        R: [..., 3, 3] rotation matrices.
    Returns:
        [..., 4] unit quaternions (w, x, y, z).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must end in (3,3); got {tuple(R.shape)}")
    eps = 1e-8

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    t0 = 1 + m00 + m11 + m22
    t1 = 1 + m00 - m11 - m22
    t2 = 1 - m00 + m11 - m22
    t3 = 1 - m00 - m11 + m22

    s0 = (t0.clamp(min=eps)).sqrt() * 2          # 4w
    s1 = (t1.clamp(min=eps)).sqrt() * 2          # 4x
    s2 = (t2.clamp(min=eps)).sqrt() * 2          # 4y
    s3 = (t3.clamp(min=eps)).sqrt() * 2          # 4z

    q0 = torch.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], dim=-1)
    q1 = torch.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], dim=-1)
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], dim=-1)
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], dim=-1)

    ts = torch.stack([t0, t1, t2, t3], dim=-1)              # [..., 4]
    idx = ts.argmax(dim=-1)                                 # [...]

    qs = torch.stack([q0, q1, q2, q3], dim=-2)              # [..., 4, 4]
    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, 4)
    q = torch.gather(qs, dim=-2, index=gather_idx).squeeze(-2)   # [..., 4]
    return F.normalize(q, p=2, dim=-1)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert (w, x, y, z) quaternions to rotation matrices."""
    if q.shape[-1] != 4:
        raise ValueError(f"q must end in 4; got {tuple(q.shape)}")
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    one = torch.ones_like(w)
    return torch.stack([
        one - 2 * (yy + zz),  2 * (xy - wz),       2 * (xz + wy),
        2 * (xy + wz),         one - 2 * (xx + zz),  2 * (yz - wx),
        2 * (xz - wy),         2 * (yz + wx),         one - 2 * (xx + yy),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


# ---------------------------------------------------------------------------
# Target image construction
# ---------------------------------------------------------------------------

def build_target_lidar_image(
    xyz: Tensor,
    intensity: Tensor,
    *,
    height: int,
    width: int,
    el_min_rad: float,
    el_max_rad: float,
    r_near: float = 0.2,
    r_far: float = 70.0,
) -> dict:
    """Bin a sensor-frame point cloud onto the spherical pixel grid.

    The pixel mapping mirrors `make_lidar_ray_grid` and the CUDA forward kernel:
        u_int = round((az + π) * (W / 2π) - 0.5)
        v_int = round((el - el_min) * (H / (el_max - el_min)) - 0.5)

    Per pixel we record the *first-hit* range (min over all points falling in
    the pixel) and the intensity of that same first-hit point. Intensity is
    expected pre-normalised to [0, 1] (raw nuScenes intensity divided by 255).

    Args:
        xyz: [N, 3]  sensor-frame points.
        intensity: [N]  in [0, 1].
        height/width: spherical image dimensions.
        el_min_rad/el_max_rad: vertical FOV in radians.
        r_near/r_far: range bounds — points outside this band are skipped.

    Returns dict:
        'range_image':     [H, W]  zeroed where no point hit.
        'intensity_image': [H, W]  zeroed where no point hit (first-hit point's intensity).
        'valid_mask':      [H, W] bool.
    """
    if xyz.shape[0] != intensity.shape[0]:
        raise ValueError(
            f"xyz N={xyz.shape[0]} but intensity N={intensity.shape[0]}"
        )
    device = xyz.device
    dtype = xyz.dtype
    HW = height * width

    x, y, z = xyz.unbind(dim=-1)
    r = xyz.norm(dim=-1)
    az = torch.atan2(x, y)                                   # [-π, π]
    xy = (x * x + y * y).clamp(min=1e-12).sqrt()
    el = torch.atan2(z, xy)

    w_per_rad_az = width / (2.0 * math.pi)
    h_per_rad_el = height / (el_max_rad - el_min_rad)

    u_f = (az + math.pi) * w_per_rad_az - 0.5
    v_f = (el - el_min_rad) * h_per_rad_el - 0.5

    u_int = u_f.round().long().clamp(0, width - 1)
    v_int = v_f.round().long().clamp(0, height - 1)

    in_fov = (
        (el >= el_min_rad) & (el <= el_max_rad) &
        (r >= r_near) & (r <= r_far)
    )

    pix = v_int * width + u_int                              # [N]
    pix_v = pix[in_fov]
    r_v = r[in_fov]
    int_v = intensity[in_fov]

    range_flat = torch.full((HW,), float("inf"), dtype=dtype, device=device)
    range_flat.scatter_reduce_(0, pix_v, r_v, reduce="amin", include_self=True)

    # First-hit intensity: pick the same return as first-hit range.
    # For duplicate pixel indices, use deterministic tie-break by earliest point.
    intensity_flat = torch.zeros(HW, dtype=dtype, device=device)
    if pix_v.numel() > 0:
        first_r = range_flat[pix_v]                                  # [M]
        is_first = torch.isclose(r_v, first_r, rtol=0.0, atol=1e-6) # [M]
        local_idx = torch.arange(pix_v.shape[0], device=device, dtype=torch.long)
        sentinel = pix_v.shape[0]
        cand_idx = torch.where(is_first, local_idx, local_idx.new_full(local_idx.shape, sentinel))

        first_idx_flat = local_idx.new_full((HW,), sentinel)
        first_idx_flat.scatter_reduce_(0, pix_v, cand_idx, reduce="amin", include_self=True)

        has_hit = first_idx_flat < sentinel
        intensity_flat[has_hit] = int_v[first_idx_flat[has_hit]]

    valid_flat = range_flat.isfinite()
    range_flat = torch.where(valid_flat, range_flat, torch.zeros_like(range_flat))

    return {
        "range_image":     range_flat.view(height, width),
        "intensity_image": intensity_flat.view(height, width),
        "valid_mask":      valid_flat.view(height, width),
    }


# ---------------------------------------------------------------------------
# GT normal map
# ---------------------------------------------------------------------------

def build_gt_normal_map(
    range_image: Tensor,
    valid_mask: Tensor,
    ray_grid: Tensor,
) -> dict:
    """Estimate per-pixel surface normals from a range image via finite differences.

    Unprojects each pixel to 3D (P = ray_dir * r), then computes the cross
    product of horizontal and vertical central-difference tangent vectors.
    Azimuth (W) is treated as circular; elevation (H) uses replicate padding.

    Args:
        range_image: [H, W]  first-hit range in metres (0 where invalid).
        valid_mask:  [H, W]  bool, True where a LiDAR return exists.
        ray_grid:    [3, H, W]  unit ray directions in sensor frame.

    Returns dict:
        'normal_image': [3, H, W]  unit normals oriented toward sensor.
        'normal_valid': [H, W]  bool — valid where pixel and both horizontal
                                and vertical neighbours are valid.
    """
    H, W = range_image.shape

    # Unproject: [3, H, W]
    pts = ray_grid * range_image.unsqueeze(0)

    # Horizontal central difference — azimuth is circular, so wrap edges.
    pts_h = torch.cat([pts[:, :, -1:], pts, pts[:, :, :1]], dim=2)  # [3, H, W+2]
    t_h = pts_h[:, :, 2:] - pts_h[:, :, :-2]                        # [3, H, W]

    # Vertical central difference — elevation is NOT circular, replicate at edges.
    pts_v = F.pad(pts, (0, 0, 1, 1), mode='replicate')               # [3, H+2, W]
    t_v = pts_v[:, 2:, :] - pts_v[:, :-2, :]                         # [3, H, W]

    # Surface normal via cross product, normalise.
    n = torch.linalg.cross(t_h, t_v, dim=0)                          # [3, H, W]
    n_norm = n.norm(dim=0, keepdim=True).clamp(min=1e-8)
    n = n / n_norm

    # Orient normals toward sensor (dot with inward direction = -ray_grid should be > 0).
    cos_angle = (n * (-ray_grid)).sum(dim=0, keepdim=True)            # [1, H, W]
    n = torch.where(cos_angle < 0, -n, n)

    # Normal validity: require both horizontal and vertical neighbours.
    vm_left  = torch.cat([valid_mask[:, -1:], valid_mask[:, :-1]], dim=1)
    vm_right = torch.cat([valid_mask[:, 1:],  valid_mask[:, :1]], dim=1)
    vm_up = torch.zeros_like(valid_mask)
    vm_down = torch.zeros_like(valid_mask)
    vm_up[1:, :] = valid_mask[:-1, :]
    vm_down[:-1, :] = valid_mask[1:, :]
    normal_valid = valid_mask & vm_left & vm_right & vm_up & vm_down # [H, W]

    return {
        "normal_image": n,
        "normal_valid": normal_valid,
    }


# ---------------------------------------------------------------------------
# Ray grid helper (moved from models/head/drop_head.py)
# ---------------------------------------------------------------------------

def make_lidar_ray_grid(
    height: int,
    width: int,
    el_min_rad: float,
    el_max_rad: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-pixel unit ray direction in sensor frame.

    Mirrors the spherical projection used by the CUDA LiDAR-mode kernel
    (cuda_rasterizer/forward.cu, lidar_mode branch):
        az = (u + 0.5) / w_per_rad_az - π
        el = (v + 0.5) / h_per_rad_el + el_min
        d  = (sin(az)cos(el), cos(az)cos(el), sin(el))

    Returns a [3, H, W] tensor.
    """
    w_per_rad_az = width / (2.0 * math.pi)
    h_per_rad_el = height / (el_max_rad - el_min_rad)
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    az = u / w_per_rad_az - math.pi
    el = v / h_per_rad_el + el_min_rad
    cos_el = torch.cos(el)
    sin_el = torch.sin(el)
    sin_az = torch.sin(az)
    cos_az = torch.cos(az)
    dx = sin_az[None, :] * cos_el[:, None]
    dy = cos_az[None, :] * cos_el[:, None]
    dz = sin_el[:, None].expand(height, width)
    return torch.stack([dx, dy, dz], dim=0)


# ---------------------------------------------------------------------------
# Render wrapper
# ---------------------------------------------------------------------------

def render_primitives(
    primitives: dict,
    *,
    height: int,
    width: int,
    el_min_rad: float,
    el_max_rad: float,
    sigma: float = 3.0,
    scale_modifier: float = 1.0,
    r_near: float = 0.2,
    r_far: float = 70.0,
    viewmatrix: Optional[Tensor] = None,
    campos: Optional[Tensor] = None,
) -> LiDARRasterOutput:
    """Run the LiDAR rasterizer on a primitive dict.

    Args:
        primitives: dict with keys
            'means3D'   [N, 3]
            'scales'    [N, 3]
            'rotations' [N, 4]   (w, x, y, z)
            'opacities' [N, 1]   in (0, 1)
            'intensity' [N]      in [0, 1]
            'latent'    [N, L]   L == LIDAR_LATENT_DIM
            'raydrop'   [N]      per-Gaussian raydrop in [0, 0.5]
        height/width: spherical image dimensions.
        el_min_rad/el_max_rad: vertical FOV.
        viewmatrix: [4,4]  world → sensor. Defaults to identity (sensor frame).
        campos:     [3]    sensor centre in world. Defaults to origin.

    Returns:
        LiDARRasterOutput.
    """
    device = primitives["means3D"].device
    if viewmatrix is None:
        viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    if campos is None:
        campos = torch.zeros(3, device=device, dtype=torch.float32)

    settings = make_lidar_settings(
        image_height=height,
        image_width=width,
        el_min_rad=el_min_rad,
        el_max_rad=el_max_rad,
        viewmatrix=viewmatrix,
        campos=campos,
        sigma=sigma,
        scale_modifier=scale_modifier,
        r_near=r_near,
        r_far=r_far,
    )

    means3D = primitives["means3D"]
    means2D = torch.zeros_like(means3D, requires_grad=means3D.requires_grad)
    return LiDARRasterizer(settings)(
        means3D=means3D,
        means2D=means2D,
        opacities=primitives["opacities"],
        scales=primitives["scales"],
        rotations=primitives["rotations"],
        intensity=primitives["intensity"],
        latent=primitives["latent"],
        raydrop=primitives["raydrop"],
    )
