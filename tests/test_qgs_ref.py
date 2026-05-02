from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "renderer" / "lidar_qgs_rasterizer"))

from python_ref.qgs_ref import (  # noqa: E402
    compute_view2gaussian,
    intersect_qgs_ray,
    pixel_center_ray,
    qgs_geodesic_length,
    render_single_pixel,
    signed_qgs_coefficients,
)


def test_pixel_center_ray_sensor_axes():
    ray = pixel_center_ray(
        torch.tensor(511.5, dtype=torch.float64),
        torch.tensor(15.5, dtype=torch.float64),
        image_width=1024,
        image_height=32,
        el_min=math.radians(-10.0),
        el_max=math.radians(10.0),
    )
    assert torch.allclose(ray, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-6)


def test_intersect_axis_aligned_qgs_root_and_normal():
    dtype = torch.float64
    mean = torch.tensor([0.0, 5.0, 0.0], dtype=dtype)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    v2g = compute_view2gaussian(mean, quat)
    ray = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    scales = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)

    hit = intersect_qgs_ray(ray, v2g, scales)

    assert bool(hit.hit)
    assert torch.allclose(hit.depth, torch.tensor(5.0, dtype=dtype), atol=1e-8)
    assert torch.allclose(hit.point_local, torch.zeros(3, dtype=dtype), atol=1e-8)
    assert torch.allclose(hit.normal_local, torch.tensor([0.0, 0.0, -1.0], dtype=dtype), atol=1e-8)


def test_qgs_coefficients_use_s3_as_height_amplitude():
    dtype = torch.float64
    scales = torch.tensor([-2.0, 4.0, 2.0], dtype=dtype)

    coeff = signed_qgs_coefficients(scales)

    assert torch.allclose(coeff, torch.tensor([-0.25, 0.0625, 0.5], dtype=dtype))


def test_intersect_reports_signed_gaussian_curvature():
    dtype = torch.float64
    mean = torch.tensor([0.0, 5.0, 0.0], dtype=dtype)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    v2g = compute_view2gaussian(mean, quat)
    ray = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    scales = torch.tensor([-2.0, 4.0, 2.0], dtype=dtype)

    hit = intersect_qgs_ray(ray, v2g, scales)

    assert bool(hit.hit)
    assert torch.allclose(hit.curvature, torch.tensor(-0.25, dtype=dtype))


def test_qgs_geodesic_length_reduces_to_plane_metric():
    dtype = torch.float64
    l = torch.tensor(0.75, dtype=dtype)
    a = torch.tensor(0.0, dtype=dtype)

    assert torch.allclose(qgs_geodesic_length(l, a), l)


def test_intersect_linear_root_case():
    dtype = torch.float64
    v2g = torch.eye(4, dtype=dtype)
    v2g[3, :3] = torch.tensor([0.0, 0.0, -2.0], dtype=dtype)
    ray = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    scales = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)

    hit = intersect_qgs_ray(ray, v2g, scales)

    assert bool(hit.hit)
    assert torch.allclose(hit.depth, torch.tensor(2.0, dtype=dtype), atol=1e-8)


def test_geodesic_gaussian_alpha_falls_off_inside_support():
    dtype = torch.float64
    mean = torch.tensor([0.0, 5.0, 0.0], dtype=dtype)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    v2g = compute_view2gaussian(mean, quat)
    ray = torch.tensor([0.1, 1.0, 0.1], dtype=dtype)
    ray = ray / ray.norm()
    scales = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)

    hit = intersect_qgs_ray(ray, v2g, scales)

    assert bool(hit.hit)
    assert 0.0 < float(hit.gaussian_weight) < 1.0


def test_single_pixel_front_to_back_blending():
    dtype = torch.float64
    means = torch.tensor([[0.0, 4.0, 0.0], [0.0, 6.0, 0.0]], dtype=dtype)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    scales = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype)
    opacities = torch.tensor([[0.25], [0.5]], dtype=dtype)
    intensity = torch.tensor([0.2, 0.8], dtype=dtype)
    latent = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    ray = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)

    out = render_single_pixel(ray, means, rotations, scales, opacities, intensity, latent)

    w0 = torch.tensor(0.25, dtype=dtype)
    w1 = torch.tensor(0.75 * 0.5, dtype=dtype)
    assert torch.allclose(out["alpha_accum"], w0 + w1)
    assert torch.allclose(out["range"], w0 * 4.0 + w1 * 6.0)
    assert torch.allclose(out["intensity"], w0 * 0.2 + w1 * 0.8)
    assert torch.allclose(out["latent"], w0 * latent[0] + w1 * latent[1])
    assert torch.allclose(out["middepth"], torch.tensor(6.0, dtype=dtype))


def test_oracle_autograd_matches_double_finite_difference():
    dtype = torch.float64
    means = torch.tensor([[0.15, 5.2, 0.05]], dtype=dtype, requires_grad=True)
    rotations = torch.tensor([[0.9987502604, 0.0, 0.0499791693, 0.0]], dtype=dtype, requires_grad=True)
    scales = torch.tensor([[0.9, 0.8, 1.1]], dtype=dtype, requires_grad=True)
    opacities = torch.tensor([[0.6]], dtype=dtype, requires_grad=True)
    intensity = torch.tensor([0.3], dtype=dtype, requires_grad=True)
    latent = torch.tensor([[0.1, 0.2]], dtype=dtype, requires_grad=True)
    ray = torch.tensor([0.02, 0.9996, 0.02], dtype=dtype)
    ray = ray / ray.norm()

    def loss_fn(m):
        out = render_single_pixel(ray, m, rotations, scales, opacities, intensity, latent)
        return out["range"] + 0.25 * out["alpha_accum"] + 0.1 * out["intensity"]

    loss = loss_fn(means)
    loss.backward()

    eps = 1e-5
    with torch.no_grad():
        plus = means.detach().clone()
        minus = means.detach().clone()
        plus[0, 1] += eps
        minus[0, 1] -= eps
        fd = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)

    assert torch.allclose(means.grad[0, 1], fd, rtol=1e-5, atol=1e-6)
