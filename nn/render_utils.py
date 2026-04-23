"""Rendering helpers for QGS-Flow Phase A.

Provides:
  - `rotmat_to_quat`        : differentiable [...,3,3] → [...,4] (w,x,y,z)
  - `build_target_lidar_image`: bin a LiDAR sweep onto the spherical grid used
                                 by the rasterizer (first-hit range, mean intensity)
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
    the pixel) and an alpha-blend-style mean intensity. Intensity is expected
    pre-normalised to [0, 1] (raw nuScenes intensity divided by 255).

    Args:
        xyz: [N, 3]  sensor-frame points.
        intensity: [N]  in [0, 1].
        height/width: spherical image dimensions.
        el_min_rad/el_max_rad: vertical FOV in radians.
        r_near/r_far: range bounds — points outside this band are skipped.

    Returns dict:
        'range_image':     [H, W]  zeroed where no point hit.
        'intensity_image': [H, W]  zeroed where no point hit.
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

    int_sum = torch.zeros(HW, dtype=dtype, device=device)
    cnt = torch.zeros(HW, dtype=dtype, device=device)
    int_sum.scatter_add_(0, pix_v, int_v)
    cnt.scatter_add_(0, pix_v, torch.ones_like(int_v))
    intensity_flat = int_sum / cnt.clamp(min=1.0)

    valid_flat = range_flat.isfinite()
    range_flat = torch.where(valid_flat, range_flat, torch.zeros_like(range_flat))

    return {
        "range_image":     range_flat.view(height, width),
        "intensity_image": intensity_flat.view(height, width),
        "valid_mask":      valid_flat.view(height, width),
    }


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
    )
