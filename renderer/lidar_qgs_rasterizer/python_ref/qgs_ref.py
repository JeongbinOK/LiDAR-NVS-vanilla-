"""Double precision oracle for the LiDAR QGS math contract.

This module intentionally favors readable equations over CUDA parity structure.
It is used by CPU tests to lock down the math before comparing against the GPU
rasterizer.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class RayHit(NamedTuple):
    hit: torch.Tensor
    depth: torch.Tensor
    point_local: torch.Tensor
    normal_local: torch.Tensor
    curvature: torch.Tensor
    gaussian_weight: torch.Tensor


def pixel_center_ray(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    el_min: float,
    el_max: float,
) -> torch.Tensor:
    """Return sensor-frame ray direction for LiDAR pixel centers."""
    dtype = u.dtype
    device = u.device
    two_pi = torch.as_tensor(2.0 * torch.pi, dtype=dtype, device=device)
    pi = torch.as_tensor(torch.pi, dtype=dtype, device=device)
    el_min_t = torch.as_tensor(el_min, dtype=dtype, device=device)
    el_max_t = torch.as_tensor(el_max, dtype=dtype, device=device)
    az = (u + 0.5) / (image_width / two_pi) - pi
    el = (v + 0.5) / (image_height / (el_max_t - el_min_t)) + el_min_t
    cos_el = torch.cos(el)
    return torch.stack(
        [torch.sin(az) * cos_el, torch.cos(az) * cos_el, torch.sin(el)],
        dim=-1,
    )


def quaternion_to_matrix(q_raw: torch.Tensor) -> torch.Tensor:
    """Quaternion `(w, x, y, z)` to rotation matrix with raw normalize."""
    q = q_raw / q_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def compute_view2gaussian(
    mean: torch.Tensor,
    rotation: torch.Tensor,
    viewmatrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the row-vector-style view-to-Gaussian transform used by CUDA."""
    if viewmatrix is None:
        viewmatrix = torch.eye(4, dtype=mean.dtype, device=mean.device)
    R = quaternion_to_matrix(rotation)
    G2W = torch.eye(4, dtype=mean.dtype, device=mean.device)
    G2W[:3, :3] = R
    G2W[3, :3] = mean
    G2V = viewmatrix @ G2W
    R_t = G2V[:3, :3].transpose(0, 1)
    t = G2V[3, :3]
    out = torch.eye(4, dtype=mean.dtype, device=mean.device)
    out[:3, :3] = R_t
    out[3, :3] = -(R_t @ t)
    return out


def signed_qgs_coefficients(scales: torch.Tensor) -> torch.Tensor:
    """Return `(kx, ky, kz)` for QGS local implicit coefficients."""
    sign12 = torch.sign(scales[..., :2])
    sign12 = torch.where(sign12 == 0, torch.ones_like(sign12), sign12)
    kxy = sign12 / scales[..., :2].abs().clamp(min=1e-8).square()
    s3 = scales[..., 2]
    s3_sign = torch.where(s3 < 0.0, -torch.ones_like(s3), torch.ones_like(s3))
    s3_safe = torch.where(s3.abs() < 1e-8, s3_sign * 1e-8, s3)
    kz = torch.reciprocal(s3_safe)
    return torch.cat([kxy, kz.unsqueeze(-1)], dim=-1)


def qgs_geodesic_length(l: torch.Tensor, a: torch.Tensor, *, eps: float = 1e-10) -> torch.Tensor:
    """QGS Eq. 10 geodesic length along a quadratic curve."""
    a_safe = torch.where(torch.abs(a) < eps, torch.ones_like(a), a)
    u = 2.0 * a_safe * l
    sqrt_term = torch.sqrt(u * u + 1.0)
    curved = torch.log(sqrt_term + u) / (4.0 * a_safe) + 0.5 * l * sqrt_term
    return torch.where(torch.abs(a) < eps, l, curved)


def intersect_qgs_ray(
    ray_direction: torch.Tensor,
    view2gaussian: torch.Tensor,
    scales: torch.Tensor,
    *,
    sigma: float = 3.0,
    r_near: float = 0.2,
    r_far: float = 100.0,
    eps: float = 1e-10,
) -> RayHit:
    """Intersect one sensor ray with one local signed QGS surface."""
    coeff = signed_qgs_coefficients(scales)
    kx, ky, kz = coeff.unbind(dim=-1)
    o = view2gaussian[3, :3]
    d = view2gaussian[:3, :3] @ ray_direction

    A = kx * d[0] * d[0] + ky * d[1] * d[1]
    B = 2.0 * (kx * o[0] * d[0] + ky * o[1] * d[1]) - kz * d[2]
    C = kx * o[0] * o[0] + ky * o[1] * o[1] - kz * o[2]

    candidates = []
    if torch.abs(A).item() < eps:
        if torch.abs(B).item() >= eps:
            candidates.append(-C / B)
    else:
        disc = B * B - 4.0 * A * C
        if disc.item() >= 0.0:
            sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
            candidates.extend([(-B - sqrt_disc) / (2.0 * A), (-B + sqrt_disc) / (2.0 * A)])

    valid_hits = []
    for t in candidates:
        if not bool((t > r_near) & (t < r_far)):
            continue
        p = o + t * d
        p_norm_2 = p[0] * p[0] + p[1] * p[1] + 1e-12
        p_norm = torch.sqrt(p_norm_2)
        cos2 = p[0] * p[0] / p_norm_2
        sin2 = p[1] * p[1] / p_norm_2
        a = scales[2] * (kx * cos2 + ky * sin2)
        s = qgs_geodesic_length(p_norm, a, eps=eps)
        r0_2 = torch.reciprocal(
            cos2 / scales[0].abs().clamp(min=1e-8).square()
            + sin2 / scales[1].abs().clamp(min=1e-8).square()
        )
        if bool(s * s <= r0_2 * sigma * sigma):
            valid_hits.append((t, p, torch.exp(-(s * s) / (2.0 * r0_2))))

    if not valid_hits:
        zero = torch.zeros((), dtype=ray_direction.dtype, device=ray_direction.device)
        return RayHit(
            hit=torch.tensor(False, device=ray_direction.device),
            depth=zero,
            point_local=torch.zeros(3, dtype=ray_direction.dtype, device=ray_direction.device),
            normal_local=torch.zeros(3, dtype=ray_direction.dtype, device=ray_direction.device),
            curvature=zero,
            gaussian_weight=zero,
        )

    depth_values = torch.stack([item[0] for item in valid_hits])
    hit_index = torch.argmin(depth_values)
    depth, p, gaussian_weight = valid_hits[int(hit_index)]
    n = torch.stack([2.0 * kx * p[0], 2.0 * ky * p[1], -kz])
    n = n / n.norm().clamp(min=1e-12)
    n = torch.where((n * d).sum() > 0.0, -n, n)
    coeff_x = scales[2] * kx
    coeff_y = scales[2] * ky
    den = 1.0 + 4.0 * (coeff_x.square() * p[0].square() + coeff_y.square() * p[1].square())
    curvature = 4.0 * coeff_x * coeff_y / den.square()
    return RayHit(
        hit=torch.tensor(True, device=ray_direction.device),
        depth=depth,
        point_local=p,
        normal_local=n,
        curvature=curvature,
        gaussian_weight=gaussian_weight,
    )


def render_single_pixel(
    ray_direction: torch.Tensor,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    intensity: torch.Tensor,
    latent: torch.Tensor,
    *,
    viewmatrix: torch.Tensor | None = None,
    sigma: float = 3.0,
    r_near: float = 0.2,
    r_far: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Small differentiable front-to-back renderer for one LiDAR pixel."""
    hits = []
    for i in range(means.shape[0]):
        v2g = compute_view2gaussian(means[i], rotations[i], viewmatrix)
        hit = intersect_qgs_ray(
            ray_direction,
            v2g,
            scales[i],
            sigma=sigma,
            r_near=r_near,
            r_far=r_far,
        )
        if bool(hit.hit):
            hits.append((hit.depth, i, hit))

    hits.sort(key=lambda item: float(item[0].detach()))
    T = torch.ones((), dtype=means.dtype, device=means.device)
    range_out = torch.zeros((), dtype=means.dtype, device=means.device)
    intensity_out = torch.zeros((), dtype=means.dtype, device=means.device)
    normal_out = torch.zeros(3, dtype=means.dtype, device=means.device)
    curvature_out = torch.zeros((), dtype=means.dtype, device=means.device)
    latent_out = torch.zeros(latent.shape[1], dtype=means.dtype, device=means.device)
    alpha_accum = torch.zeros((), dtype=means.dtype, device=means.device)
    middepth = torch.zeros((), dtype=means.dtype, device=means.device)

    for depth, i, hit in hits:
        alpha = (opacities[i].reshape(()) * hit.gaussian_weight).clamp(0.0, 0.99)
        if bool(alpha < (1.0 / 255.0)):
            continue
        weight = T * alpha
        range_out = range_out + weight * depth
        intensity_out = intensity_out + weight * intensity[i]
        normal_out = normal_out + weight * hit.normal_local
        curvature_out = curvature_out + weight * hit.curvature
        latent_out = latent_out + weight * latent[i]
        alpha_accum = alpha_accum + weight
        middepth = torch.where(alpha_accum >= 0.5, torch.where(middepth == 0.0, depth, middepth), middepth)
        T = T * (1.0 - alpha)

    return {
        "range": range_out,
        "middepth": middepth,
        "intensity": intensity_out,
        "alpha_accum": alpha_accum,
        "normal": normal_out,
        "curvature": curvature_out,
        "latent": latent_out,
    }
