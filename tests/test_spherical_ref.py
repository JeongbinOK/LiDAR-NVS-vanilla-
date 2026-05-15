"""
A3.2.b — Validate `renderer/lidar_qgs_rasterizer/python_ref/spherical_ref.py`
against ground-truth analytical cases.

This Python reference mirrors `cuda_rasterizer/spherical.h` exactly. Once the
LiDAR-mode CUDA kernel is wired up (A3.2.c), a follow-up test will round-trip
through the GPU helper and assert numerical agreement with this file. Both
implementations MUST stay in sync.

Sensor frame convention (locked):
  x = right, y = forward, z = up
  az = atan2(x, y)              ∈ [-π, π],   0 = +y forward, +π/2 = +x right
  el = atan2(z, sqrt(x²+y²))    ∈ [-π/2, π/2]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "renderer" / "lidar_qgs_rasterizer"))

from python_ref.spherical_ref import (  # noqa: E402
    HALF_PI,
    PI,
    aabb_spherical,
    is_in_spherical_frustum,
    project_to_sphere,
    spherical_to_pixel,
)


def _uniform_row_table(el_min_deg: float, el_max_deg: float, H: int) -> torch.Tensor:
    """Uniformly-spaced row centers, mimicking the legacy linear FOV.

    Used by tests that don't care about HDL-32E specifics — only that the
    boundary semantics (el_min_eff → v=0, el_max_eff → v=H) hold.
    """
    el_min = math.radians(el_min_deg)
    el_max = math.radians(el_max_deg)
    step = (el_max - el_min) / H
    # Row k center elevation: el_min + (k + 0.5) * step. Then
    # el_min_eff = row[0] - 0.5*step = el_min ✓
    # el_max_eff = row[H-1] + 0.5*step = el_max ✓
    centers = el_min + (torch.arange(H, dtype=torch.float64) + 0.5) * step
    return centers.to(torch.float32)


def _intr_with_row(W: int, H: int, el_min_deg: float, el_max_deg: float):
    """cam_intr + row table for a uniform-FOV setup."""
    row_el = _uniform_row_table(el_min_deg, el_max_deg, H)
    el_min_eff = (row_el[0] - 0.5 * (row_el[1] - row_el[0])).item()
    el_max_eff = (row_el[-1] + 0.5 * (row_el[-1] - row_el[-2])).item()
    cam_intr = torch.tensor([
        el_min_eff,
        el_max_eff,
        W / (2.0 * math.pi),
        float(H),
    ], dtype=torch.float32)
    return cam_intr, row_el


# ---------------------------------------------------------------------------
# project_to_sphere — known-answer ground truths
# ---------------------------------------------------------------------------

class TestProjectToSphere:
    def test_forward_axis(self):
        """+y (forward) → az=0, el=0."""
        p = torch.tensor([[0.0, 1.0, 0.0]])
        out = project_to_sphere(p)
        assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)

    def test_right_axis(self):
        """+x (right) → az=+π/2, el=0."""
        p = torch.tensor([[1.0, 0.0, 0.0]])
        out = project_to_sphere(p)
        assert torch.allclose(out, torch.tensor([[1.0, HALF_PI, 0.0]]), atol=1e-6)

    def test_left_axis(self):
        """-x (left) → az=-π/2, el=0."""
        p = torch.tensor([[-1.0, 0.0, 0.0]])
        out = project_to_sphere(p)
        assert torch.allclose(out, torch.tensor([[1.0, -HALF_PI, 0.0]]), atol=1e-6)

    def test_backward_axis_positive_x_side(self):
        """-y (back), tiny +x bias → az ≈ +π."""
        p = torch.tensor([[1e-6, -1.0, 0.0]])
        out = project_to_sphere(p)
        # atan2(+ε, -1) → +π
        assert math.isclose(out[0, 1].item(), PI, abs_tol=1e-4)
        assert math.isclose(out[0, 2].item(), 0.0, abs_tol=1e-4)

    def test_zenith(self):
        """+z (up) → el=+π/2 (az is undefined; we just check el and r)."""
        p = torch.tensor([[0.0, 0.0, 1.0]])
        out = project_to_sphere(p)
        assert math.isclose(out[0, 0].item(), 1.0, abs_tol=1e-6)
        assert math.isclose(out[0, 2].item(), HALF_PI, abs_tol=1e-6)

    def test_nadir(self):
        """-z (down) → el=-π/2."""
        p = torch.tensor([[0.0, 0.0, -1.0]])
        out = project_to_sphere(p)
        assert math.isclose(out[0, 2].item(), -HALF_PI, abs_tol=1e-6)

    def test_diagonal_45(self):
        """(1, 1, 0) / √2 → r=1, az=π/4, el=0."""
        p = torch.tensor([[1.0, 1.0, 0.0]]) / math.sqrt(2.0)
        out = project_to_sphere(p)
        assert torch.allclose(
            out, torch.tensor([[1.0, math.pi / 4, 0.0]]), atol=1e-6
        )

    def test_elevation_30deg(self):
        """xy = √3/2 horizontal, z = 1/2 → el = 30°."""
        p = torch.tensor([[0.0, math.sqrt(3.0) / 2.0, 0.5]])
        out = project_to_sphere(p)
        assert math.isclose(out[0, 0].item(), 1.0, abs_tol=1e-6)
        assert math.isclose(out[0, 2].item(), math.radians(30.0), abs_tol=1e-6)

    def test_batch(self):
        """Batched input keeps the leading dims intact."""
        p = torch.randn(4, 7, 3)
        out = project_to_sphere(p)
        assert out.shape == (4, 7, 3)
        # r ≥ 0
        assert (out[..., 0] >= 0).all()
        # az ∈ [-π, π]
        assert (out[..., 1] >= -PI - 1e-6).all() and (out[..., 1] <= PI + 1e-6).all()
        # el ∈ [-π/2, π/2]
        assert (out[..., 2] >= -HALF_PI - 1e-6).all() and (out[..., 2] <= HALF_PI + 1e-6).all()


# ---------------------------------------------------------------------------
# spherical_to_pixel — pixel grid sanity
# ---------------------------------------------------------------------------

class TestSphericalToPixel:
    def _intr(self, W=1024, H=32, el_min_deg=-30.0, el_max_deg=10.0):
        return _intr_with_row(W, H, el_min_deg, el_max_deg)

    def test_az_minus_pi_maps_to_u_zero(self):
        cam, row = self._intr()
        u, v = spherical_to_pixel(torch.tensor(-PI), torch.tensor(0.0), cam, row)
        assert math.isclose(u.item(), 0.0, abs_tol=1e-6)

    def test_az_plus_pi_maps_to_u_W(self):
        cam, row = self._intr(W=1024)
        u, _ = spherical_to_pixel(torch.tensor(PI), torch.tensor(0.0), cam, row)
        assert math.isclose(u.item(), 1024.0, abs_tol=1e-4)

    def test_az_zero_maps_to_u_half_W(self):
        cam, row = self._intr(W=2048)
        u, _ = spherical_to_pixel(torch.tensor(0.0), torch.tensor(0.0), cam, row)
        assert math.isclose(u.item(), 1024.0, abs_tol=1e-4)

    def test_el_min_eff_maps_to_v_zero(self):
        cam, row = self._intr(el_min_deg=-30.0, el_max_deg=10.0)
        _, v = spherical_to_pixel(torch.tensor(0.0), cam[0].clone().detach(), cam, row)
        assert math.isclose(v.item(), 0.0, abs_tol=1e-5)

    def test_el_max_eff_maps_to_v_H(self):
        cam, row = self._intr(H=32, el_min_deg=-30.0, el_max_deg=10.0)
        _, v = spherical_to_pixel(torch.tensor(0.0), cam[1].clone().detach(), cam, row)
        assert math.isclose(v.item(), 32.0, abs_tol=1e-4)

    def test_row_center_round_trip(self):
        """v(row_to_el[k]) == k + 0.5 by construction, for every row."""
        cam, row = self._intr(H=32, el_min_deg=-30.0, el_max_deg=10.0)
        for k in range(row.shape[0]):
            _, v = spherical_to_pixel(torch.tensor(0.0), row[k].clone().detach(),
                                      cam, row)
            assert math.isclose(v.item(), k + 0.5, abs_tol=1e-5), \
                f"row {k}: v={v.item()}"


# ---------------------------------------------------------------------------
# is_in_spherical_frustum — vertical FOV + range gate
# ---------------------------------------------------------------------------

class TestFrustum:
    def _intr(self):
        # nuScenes 32-beam: el ∈ [-30°, +10°]. Frustum bounds match el_min/max
        # in cam_intr[0]/[1] (effective bounds in the new convention).
        return torch.tensor([
            math.radians(-30.0),
            math.radians(10.0),
            1024.0 / (2.0 * math.pi),
            32.0,
        ])

    def test_in_band_passes(self):
        # 5m forward, slightly above horizon
        p = torch.tensor([[0.0, 5.0, 0.5]])
        assert is_in_spherical_frustum(p, self._intr()).item()

    def test_too_high_fails(self):
        # Straight up — el ≈ +90° > el_max
        p = torch.tensor([[0.0, 0.1, 5.0]])
        assert not is_in_spherical_frustum(p, self._intr()).item()

    def test_too_low_fails(self):
        # Steeply below — el ≈ -88°
        p = torch.tensor([[0.0, 0.1, -5.0]])
        assert not is_in_spherical_frustum(p, self._intr()).item()

    def test_too_close_fails(self):
        # Inside r_near=0.2
        p = torch.tensor([[0.0, 0.05, 0.0]])
        assert not is_in_spherical_frustum(p, self._intr(),
                                           r_near=0.2, r_far=100.0).item()

    def test_too_far_fails(self):
        # Beyond r_far=100
        p = torch.tensor([[0.0, 200.0, 0.0]])
        assert not is_in_spherical_frustum(p, self._intr(),
                                           r_near=0.2, r_far=100.0).item()


# ---------------------------------------------------------------------------
# aabb_spherical — angular bbox of a ball of radius R_eff
# ---------------------------------------------------------------------------

class TestAabbSpherical:
    def _assert_angles_inside(self, out, az, el):
        if out["wrapped"].item():
            az_ok = (az >= out["az_min"].item() - 1e-6) | (az <= out["az_max"].item() + 1e-6)
        else:
            az_ok = (az >= out["az_min"].item() - 1e-6) & (az <= out["az_max"].item() + 1e-6)
        el_ok = (el >= out["el_min"].item() - 1e-6) & (el <= out["el_max"].item() + 1e-6)
        assert az_ok.all(), (az[~az_ok][:8], out)
        assert el_ok.all(), (el[~el_ok][:8], out)

    def test_small_ball_no_wrap(self):
        """Ball of R=0.1 at 5m forward → small extent, no wrap."""
        p = torch.tensor([[0.0, 5.0, 0.0]])
        R_eff = torch.tensor([0.1])
        out = aabb_spherical(p, R_eff)

        # theta_half = asin(0.1/5) ≈ 0.02 rad
        expected = math.asin(0.1 / 5.0)
        assert torch.allclose(out["az_max"] - out["az_min"],
                              torch.tensor([2 * expected]), atol=1e-5)
        assert torch.allclose(out["el_max"] - out["el_min"],
                              torch.tensor([2 * expected]), atol=1e-5)
        assert not out["wrapped"].item()

    def test_centered_at_az_zero_no_wrap(self):
        p = torch.tensor([[0.0, 5.0, 0.0]])
        R_eff = torch.tensor([0.1])
        out = aabb_spherical(p, R_eff)
        # Centred on az=0 → symmetric
        assert torch.allclose(out["az_min"], -out["az_max"], atol=1e-6)

    def test_wrap_near_seam_minus(self):
        """Ball just inside az = -π+ε → az_min < -π → wraps."""
        # Place at (-ε, -y) so az ≈ -π + tiny
        p = torch.tensor([[-0.01, -5.0, 0.0]])
        R_eff = torch.tensor([0.5])
        out = aabb_spherical(p, R_eff)
        # az_c is near -π. theta_half ≈ asin(0.1) ≈ 0.1 → az_min ≈ -π - 0.1 → wraps to ~+π
        assert out["wrapped"].item()
        # After wrap, az_min > az_max (the box spans the seam)
        assert out["az_min"].item() > out["az_max"].item()

    def test_wrap_near_seam_plus(self):
        """Ball just inside az = +π-ε → az_max > +π → wraps."""
        p = torch.tensor([[0.01, -5.0, 0.0]])
        R_eff = torch.tensor([0.5])
        out = aabb_spherical(p, R_eff)
        assert out["wrapped"].item()
        assert out["az_min"].item() > out["az_max"].item()

    def test_R_equals_r_at_equator_gives_full_azimuth_superset(self):
        """R_eff = r at equator includes both poles, so a conservative azimuth AABB is full circle."""
        p = torch.tensor([[0.0, 1.0, 0.0]])
        R_eff = torch.tensor([1.0])
        out = aabb_spherical(p, R_eff)
        assert torch.isclose(out["az_min"], torch.tensor(-PI), atol=1e-6)
        assert torch.isclose(out["az_max"], torch.tensor(PI),  atol=1e-6)
        # Elevation extent is symmetric ±π/2 around 0 → fully clamped
        assert torch.isclose(out["el_min"], torch.tensor(-HALF_PI), atol=1e-6)
        assert torch.isclose(out["el_max"], torch.tensor(HALF_PI),  atol=1e-6)
        assert not out["wrapped"].item()

    def test_full_circle_when_cap_reaches_pole(self):
        """A cap that reaches a pole must cover all azimuths."""
        el_target = math.radians(80.0)
        r = 5.0
        x = 0.0
        y = r * math.cos(el_target)
        z = r * math.sin(el_target)
        p = torch.tensor([[x, y, z]])
        R_eff = torch.tensor([0.6 * r])
        out = aabb_spherical(p, R_eff)
        assert torch.isclose(out["az_min"], torch.tensor(-PI), atol=1e-6), out["az_min"]
        assert torch.isclose(out["az_max"], torch.tensor(PI),  atol=1e-6), out["az_max"]
        assert not out["wrapped"].item()

    def test_az_extent_uses_exact_spherical_cap_formula_inside_lidar_fov(self):
        """Within the [-30°, 10°] LiDAR FOV, the exact cap formula is still wider than the small-angle approximation."""
        el_target = math.radians(-29.0)
        r = 5.0
        p = torch.tensor([[0.0, r * math.cos(el_target), r * math.sin(el_target)]])
        R_eff = torch.tensor([0.4 * r])
        out = aabb_spherical(p, R_eff)

        sin_half = 0.4
        theta_half = math.asin(sin_half)
        exact_az_half = math.asin(sin_half / math.cos(el_target))
        old_small_angle = theta_half / math.cos(el_target)
        assert exact_az_half > old_small_angle
        assert torch.isclose(out["az_min"], torch.tensor(-exact_az_half), atol=1e-6)
        assert torch.isclose(out["az_max"], torch.tensor(exact_az_half), atol=1e-6)

    def test_aabb_contains_sampled_ball_points_inside_lidar_fov(self):
        """The angular box must be a superset of projected points from the 3D candidate ball."""
        torch.manual_seed(3)
        el_target = math.radians(-28.0)
        r = 6.0
        center = torch.tensor([[0.0, r * math.cos(el_target), r * math.sin(el_target)]])
        R_eff = torch.tensor([0.35 * r])
        out = aabb_spherical(center, R_eff)

        samples = torch.randn(20000, 3)
        samples = samples / samples.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        radius = torch.rand(20000, 1).pow(1.0 / 3.0) * R_eff.item()
        points = center + samples * radius
        projected = project_to_sphere(points)
        self._assert_angles_inside(out, projected[:, 1], projected[:, 2])

    def test_elevation_clamp_high(self):
        """Centre near zenith → el_max clamps at +π/2."""
        p = torch.tensor([[0.0, 0.1, 5.0]])  # el ≈ +88°
        R_eff = torch.tensor([1.0])
        out = aabb_spherical(p, R_eff)
        assert out["el_max"].item() <= HALF_PI + 1e-6

    def test_elevation_clamp_low(self):
        """Centre near nadir → el_min clamps at -π/2."""
        p = torch.tensor([[0.0, 0.1, -5.0]])
        R_eff = torch.tensor([1.0])
        out = aabb_spherical(p, R_eff)
        assert out["el_min"].item() >= -HALF_PI - 1e-6

    def test_az_extent_grows_near_pole(self):
        """At higher |el|, az_half = theta_half / cos(el) is larger than at equator."""
        p_eq = torch.tensor([[0.0, 5.0, 0.0]])           # el ≈ 0
        p_hi = torch.tensor([[0.0, 1.0, 4.0]])           # el ≈ 76°
        R_eff = torch.tensor([0.5])
        out_eq = aabb_spherical(p_eq, R_eff)
        out_hi = aabb_spherical(p_hi, R_eff)
        ext_eq = (out_eq["az_max"] - out_eq["az_min"]).item()
        ext_hi = (out_hi["az_max"] - out_hi["az_min"]).item()
        # Near pole the azimuthal extent is much larger (or wraps).
        # If hi wraps, az_min > az_max — handle that by computing wrapped extent.
        if out_hi["wrapped"].item():
            ext_hi = (out_hi["az_max"] + 2 * PI - out_hi["az_min"]).item()
            ext_hi = min(ext_hi, 2 * PI)
        assert ext_hi > ext_eq

    def test_batch_consistency(self):
        """Batched call equals scalar calls for each entry."""
        torch.manual_seed(0)
        p = torch.randn(8, 3) * 2.0 + torch.tensor([0.0, 5.0, 0.0])
        R_eff = torch.rand(8) * 0.3 + 0.05
        out = aabb_spherical(p, R_eff)
        for i in range(8):
            single = aabb_spherical(p[i:i+1], R_eff[i:i+1])
            for k in ("az_min", "az_max", "el_min", "el_max"):
                assert torch.allclose(out[k][i], single[k][0], atol=1e-6), k
            assert out["wrapped"][i].item() == single["wrapped"][0].item()


# ---------------------------------------------------------------------------
# Round-trip: project then pixel, check pixel ∈ [0, W) × [0, H) for in-frustum points
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_random_in_frustum_points_yield_valid_pixels(self):
        torch.manual_seed(1)
        W, H = 1024, 32
        cam_intr, row_el = _intr_with_row(W, H, -30.0, 10.0)
        el_min = float(cam_intr[0])
        el_max = float(cam_intr[1])

        # Sample r, az, el uniformly in valid ranges, then construct (x,y,z)
        N = 200
        r = torch.rand(N) * 50.0 + 1.0
        az = (torch.rand(N) * 2 - 1) * (PI - 1e-3)
        el = torch.rand(N) * (el_max - el_min) + el_min
        cos_el = torch.cos(el)
        x = r * cos_el * torch.sin(az)
        y = r * cos_el * torch.cos(az)
        z = r * torch.sin(el)
        p = torch.stack([x, y, z], dim=-1)

        # All should be in frustum
        in_fr = is_in_spherical_frustum(p, cam_intr, r_near=0.2, r_far=100.0)
        assert in_fr.all(), f"only {in_fr.float().mean():.2%} in frustum"

        # Project and check pixel bounds
        sph = project_to_sphere(p)
        u, v = spherical_to_pixel(sph[..., 1], sph[..., 2], cam_intr, row_el)
        assert (u >= 0).all() and (u <= W + 1e-3).all()
        assert (v >= 0).all() and (v <= H + 1e-3).all()
