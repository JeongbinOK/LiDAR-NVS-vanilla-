"""Rendering helpers for QGS-Flow Phase A.

Provides:
  - `rotmat_to_quat`        : differentiable [...,3,3] → [...,4] (w,x,y,z)
  - `quat_to_rotmat`        : differentiable [...,4] → [...,3,3]
  - `build_gt_normal_map`   : estimate per-pixel GT surface normals from a range image
  - `render_primitives`     : wrap a primitive dict into a `LiDARRasterizer` call

GT range/intensity map generation and the spherical ray grid live in
`nn/lidar_geometry.py` (single source of truth for row ↔ elevation ↔ ring).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

import diff_quadratic_rasterization as dq

from nn.lidar_geometry import get_effective_el_bounds, get_row_to_elevation_rad

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
# Render wrapper
# ---------------------------------------------------------------------------

def render_primitives(
    primitives: dict,
    cfg,
    *,
    scale_modifier: float = 1.0,
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
        cfg: QGSConfig — provides lidar_height/width, ring_to_elevation_deg,
             lidar_sigma, r_near, r_far.
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

    el_min_eff, el_max_eff = get_effective_el_bounds(cfg)
    row_to_el = get_row_to_elevation_rad(cfg, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=int(cfg.lidar_height),
        image_width=int(cfg.lidar_width),
        el_min_rad=el_min_eff,
        el_max_rad=el_max_eff,
        row_to_elevation_rad=row_to_el,
        viewmatrix=viewmatrix,
        campos=campos,
        sigma=float(cfg.lidar_sigma),
        scale_modifier=scale_modifier,
        r_near=float(cfg.r_near),
        r_far=float(cfg.r_far),
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
