"""
Pure-Python mirror of cuda_rasterizer/spherical.h.

Used to validate the CUDA helper math without needing to round-trip through
the GPU kernel. The two implementations MUST stay in sync — any change to
spherical.h must also be reflected here, and vice versa.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

PI = math.pi
TWO_PI = 2.0 * math.pi
HALF_PI = 0.5 * math.pi
SPH_EPS = 1e-8


def project_to_sphere(p: torch.Tensor) -> torch.Tensor:
    """
    p: [..., 3] in sensor frame (x=right, y=forward, z=up).
    Returns [..., 3] = (r, az, el) where
        r  = ||p||
        az = atan2(x, y)        ∈ [-π, π]
        el = atan2(z, sqrt(x²+y²)) ∈ [-π/2, π/2]
    """
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = p.norm(dim=-1)
    az = torch.atan2(x, y)
    xy = torch.sqrt(x * x + y * y)
    el = torch.atan2(z, xy)
    return torch.stack([r, az, el], dim=-1)


def spherical_to_pixel(az: torch.Tensor, el: torch.Tensor,
                       cam_intr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map (az, el) -> (u, v) given LiDAR intrinsics
        cam_intr = [el_min, el_max, w_per_rad_az, h_per_rad_el].
    """
    el_min       = cam_intr[0]
    w_per_rad_az = cam_intr[2]
    h_per_rad_el = cam_intr[3]
    u = (az + PI) * w_per_rad_az
    v = (el - el_min) * h_per_rad_el
    return u, v


def is_in_spherical_frustum(p: torch.Tensor,
                            cam_intr: torch.Tensor,
                            r_near: float = 0.2,
                            r_far: float = 100.0) -> torch.Tensor:
    """
    Element-wise frustum test (vertical FOV + range gate).
    p: [..., 3] sensor-frame.  Returns [...] bool.
    """
    res = project_to_sphere(p)
    r, _, el = res[..., 0], res[..., 1], res[..., 2]
    el_min, el_max = cam_intr[0], cam_intr[1]
    return (r >= r_near) & (r <= r_far) & (el >= el_min) & (el <= el_max)


def aabb_spherical(p: torch.Tensor, R_eff: torch.Tensor):
    """
    Spherical AABB of a ball of radius R_eff centred at p (sensor frame).

    Args:
        p:     [N, 3]
        R_eff: [N]

    Returns dict with:
        az_min, az_max : [N]
        el_min, el_max : [N]
        wrapped        : [N] bool — True if box straddles ±π in azimuth.
    """
    r = p.norm(dim=-1)
    res = project_to_sphere(p)
    az_c, el_c = res[..., 1], res[..., 2]

    safe_r = r.clamp(min=SPH_EPS)
    sin_half = (R_eff / safe_r).clamp(max=1.0)
    theta_half = torch.asin(sin_half)

    el_min = (el_c - theta_half).clamp(min=-HALF_PI)
    el_max = (el_c + theta_half).clamp(max=HALF_PI)

    cos_el = torch.cos(el_c).abs().clamp(min=SPH_EPS)
    full_circle = sin_half >= cos_el
    az_half = torch.where(
        full_circle,
        torch.full_like(theta_half, PI),
        torch.asin((sin_half / cos_el).clamp(max=1.0)),
    )

    az_min = az_c - az_half
    az_max = az_c + az_half

    wrapped = torch.zeros_like(r, dtype=torch.bool)

    # Wrap az_min < -π
    wrap_lo = az_min < -PI
    az_min = torch.where(wrap_lo, az_min + TWO_PI, az_min)
    wrapped = wrapped | wrap_lo

    # Wrap az_max > π
    wrap_hi = az_max > PI
    az_max = torch.where(wrap_hi, az_max - TWO_PI, az_max)
    wrapped = wrapped | wrap_hi

    # If full circle, override to [-π, π], clear wrapped flag.
    az_min = torch.where(full_circle, torch.full_like(az_min, -PI), az_min)
    az_max = torch.where(full_circle, torch.full_like(az_max,  PI), az_max)
    wrapped = wrapped & ~full_circle

    return {
        "az_min": az_min, "az_max": az_max,
        "el_min": el_min, "el_max": el_max,
        "wrapped": wrapped,
    }
