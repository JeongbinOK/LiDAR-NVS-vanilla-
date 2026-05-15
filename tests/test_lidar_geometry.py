"""Unit tests for nn/lidar_geometry.py — single source of truth for LiDAR mapping."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from nn.lidar_geometry import (
    get_effective_el_bounds,
    get_ring_at_row,
    get_ring_to_row,
    get_row_to_elevation_rad,
    make_lidar_ray_grid,
    points_to_lidar_maps,
    points_to_v,
    range_map_to_points,
    row_to_ray_direction,
    v_to_el,
)


# nuScenes LIDAR_TOP — `ring` is pre-sorted by elevation (ascending), not the
# raw HDL-32E firing order. So ring index == row index (v=0=bottom convention).
HDL32E_RING_TO_EL_DEG = (
    -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33,
    -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
     -9.33,  -8.00,  -6.66,  -5.33,  -4.00,  -2.67,  -1.33,   0.00,
      1.33,   2.67,   4.00,   5.33,   6.67,   8.00,   9.33,  10.67,
)


def _make_cfg(width: int = 1085, r_near: float = 0.2, r_far: float = 70.0):
    return SimpleNamespace(
        ring_to_elevation_deg=HDL32E_RING_TO_EL_DEG,
        lidar_height=32,
        lidar_width=width,
        r_near=r_near,
        r_far=r_far,
    )


# ---------------------------------------------------------------------------
# Table sorting & ring↔row mappings
# ---------------------------------------------------------------------------

def test_row_to_elevation_is_ascending():
    cfg = _make_cfg()
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    assert row_el.shape == (32,)
    diffs = row_el[1:] - row_el[:-1]
    assert (diffs > 0).all(), "row_to_el must be strictly ascending"
    # endpoints in deg
    assert math.degrees(row_el[0].item()) == pytest.approx(-30.67, abs=1e-6)
    assert math.degrees(row_el[-1].item()) == pytest.approx(10.67, abs=1e-6)


def test_ring_to_row_is_identity_for_presorted_table():
    cfg = _make_cfg()
    ring_to_row = get_ring_to_row(cfg)
    ring_at_row = get_ring_at_row(cfg)
    assert ring_to_row.shape == (32,)
    assert ring_at_row.shape == (32,)
    # nuScenes' ring channel is already sorted ascending by elevation, so the
    # config table is sorted and the row mapping is identity.
    expected = torch.arange(32, dtype=torch.int64)
    assert torch.equal(ring_to_row.cpu(), expected)
    assert torch.equal(ring_at_row.cpu(), expected)


# ---------------------------------------------------------------------------
# v(el) ↔ el(v) round-trip
# ---------------------------------------------------------------------------

def test_row_center_round_trip():
    """Core identity: v(row_to_el[k]) == k + 0.5 for every row."""
    cfg = _make_cfg()
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    H = row_el.shape[0]
    v_at_centers = points_to_v(row_el, cfg)
    expected = torch.arange(H, dtype=torch.float64) + 0.5
    assert torch.allclose(v_at_centers, expected, atol=1e-9)


def test_v_to_el_at_centers_returns_row_elevation():
    cfg = _make_cfg()
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    H = row_el.shape[0]
    v_centers = torch.arange(H, dtype=torch.float64) + 0.5
    el_back = v_to_el(v_centers, cfg)
    assert torch.allclose(el_back, row_el, atol=1e-9)


def test_points_to_v_is_strict_monotonic():
    cfg = _make_cfg()
    torch.manual_seed(0)
    el_min, el_max = get_effective_el_bounds(cfg)
    el = torch.rand(2000, dtype=torch.float64) * (el_max - el_min) + el_min
    el, _ = torch.sort(el)
    v = points_to_v(el, cfg)
    assert (v[1:] - v[:-1] > 0).all()


def test_v_to_el_is_strict_monotonic():
    cfg = _make_cfg()
    torch.manual_seed(0)
    v = torch.rand(2000, dtype=torch.float64) * 32.0
    v, _ = torch.sort(v)
    el = v_to_el(v, cfg)
    assert (el[1:] - el[:-1] > 0).all()


def test_effective_bounds_map_to_v0_and_vH():
    cfg = _make_cfg()
    el_min_eff, el_max_eff = get_effective_el_bounds(cfg)
    el_t = torch.tensor([el_min_eff, el_max_eff], dtype=torch.float64)
    v = points_to_v(el_t, cfg)
    assert v[0].item() == pytest.approx(0.0, abs=1e-9)
    assert v[1].item() == pytest.approx(32.0, abs=1e-9)


def test_round_trip_random():
    """v_to_el(points_to_v(el)) == el for el inside the row span."""
    cfg = _make_cfg()
    torch.manual_seed(1)
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    el = torch.rand(1000, dtype=torch.float64) * (row_el[-1] - row_el[0]) + row_el[0]
    v = points_to_v(el, cfg)
    el_back = v_to_el(v, cfg)
    assert torch.allclose(el_back, el, atol=1e-9)


# ---------------------------------------------------------------------------
# Ray construction
# ---------------------------------------------------------------------------

def test_make_lidar_ray_grid_uses_row_elevations():
    cfg = _make_cfg(width=64)
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float32)
    grid = make_lidar_ray_grid(cfg, device="cpu", dtype=torch.float32)
    assert grid.shape == (3, 32, 64)
    # ||ray|| should be 1
    norms = grid.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # z-component of row k should be sin(row_el[k]), independent of u
    expected_z = torch.sin(row_el).unsqueeze(-1).expand(32, 64)
    assert torch.allclose(grid[2], expected_z, atol=1e-6)


def test_row_to_ray_direction_matches_grid():
    cfg = _make_cfg(width=64)
    grid = make_lidar_ray_grid(cfg, device="cpu", dtype=torch.float32)
    v = torch.tensor([0, 15, 31])
    u = torch.tensor([0, 32, 63])
    rays = row_to_ray_direction(v, u, cfg, device="cpu", dtype=torch.float32)
    assert rays.shape == (3, 3)
    for i, (vi, ui) in enumerate(zip(v.tolist(), u.tolist())):
        expected = grid[:, vi, ui]
        assert torch.allclose(rays[i], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Points → maps → points round-trip
# ---------------------------------------------------------------------------

def _make_point_at(row_el_k: float, az_rad: float, r: float, dtype=torch.float32):
    cos_el = math.cos(row_el_k)
    sin_el = math.sin(row_el_k)
    x = math.sin(az_rad) * cos_el * r
    y = math.cos(az_rad) * cos_el * r
    z = sin_el * r
    return torch.tensor([[x, y, z]], dtype=dtype)


def test_points_to_lidar_maps_ring_path():
    """row k receives the point only when its ring = ring_at_row[k]."""
    cfg = _make_cfg(width=64)
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    ring_at_row = get_ring_at_row(cfg)
    H, W = 32, 64

    az = 0.5  # arbitrary
    u_expected = int(round((az + math.pi) * (W / (2.0 * math.pi)) - 0.5))
    u_expected = max(0, min(W - 1, u_expected))

    for k in range(H):
        xyz = _make_point_at(float(row_el[k]), az, 10.0, dtype=torch.float32)
        intensity = torch.tensor([0.5], dtype=torch.float32)
        ring = torch.tensor([int(ring_at_row[k])], dtype=torch.int64)
        out = points_to_lidar_maps(xyz, intensity, cfg, ring=ring)
        assert bool(out["valid_mask"][k, u_expected]), f"row {k} should be valid"
        assert out["range_image"][k, u_expected].item() == pytest.approx(10.0, abs=1e-4)
        assert out["intensity_image"][k, u_expected].item() == pytest.approx(0.5, abs=1e-6)
        # only one pixel hit
        assert int(out["valid_mask"].sum()) == 1


def test_points_to_lidar_maps_no_ring_path_uses_nearest_beam():
    cfg = _make_cfg(width=64)
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    H, W = 32, 64
    az = -1.2

    for k in range(H):
        xyz = _make_point_at(float(row_el[k]), az, 5.0, dtype=torch.float32)
        intensity = torch.tensor([0.7], dtype=torch.float32)
        out = points_to_lidar_maps(xyz, intensity, cfg, ring=None)
        u_expected = int(round((az + math.pi) * (W / (2.0 * math.pi)) - 0.5))
        u_expected = max(0, min(W - 1, u_expected))
        assert bool(out["valid_mask"][k, u_expected])
        assert out["range_image"][k, u_expected].item() == pytest.approx(5.0, abs=1e-4)


def test_points_to_lidar_maps_first_hit_keeps_smallest_range():
    cfg = _make_cfg(width=8)
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    ring_at_row = get_ring_at_row(cfg)
    az = 0.0
    # Two points colliding at the same pixel (same row, same azimuth bin),
    # different ranges.
    k = 10
    el_k = float(row_el[k])
    p_near = _make_point_at(el_k, az, 5.0, dtype=torch.float32)
    p_far = _make_point_at(el_k, az, 20.0, dtype=torch.float32)
    xyz = torch.cat([p_far, p_near], dim=0)
    intensity = torch.tensor([0.2, 0.9], dtype=torch.float32)
    ring = torch.full((2,), int(ring_at_row[k]), dtype=torch.int64)
    out = points_to_lidar_maps(xyz, intensity, cfg, ring=ring)
    u_expected = 4  # az=0 with W=8 → u_int=4
    assert out["range_image"][k, u_expected].item() == pytest.approx(5.0, abs=1e-4)
    assert out["intensity_image"][k, u_expected].item() == pytest.approx(0.9, abs=1e-6)


def test_range_map_to_points_round_trip():
    cfg = _make_cfg(width=64)
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float32)
    ring_at_row = get_ring_at_row(cfg)
    H, W = 32, 64
    # one point per row at varying azimuth and range
    az = torch.linspace(-2.0, 2.0, H, dtype=torch.float32)
    r = torch.linspace(3.0, 30.0, H, dtype=torch.float32)
    pts = []
    rings = []
    for k in range(H):
        pts.append(_make_point_at(float(row_el[k]), float(az[k]), float(r[k])))
        rings.append(int(ring_at_row[k]))
    xyz = torch.cat(pts, dim=0)
    intensity = torch.full((H,), 0.5, dtype=torch.float32)
    ring = torch.tensor(rings, dtype=torch.int64)
    out = points_to_lidar_maps(xyz, intensity, cfg, ring=ring)
    pts_back = range_map_to_points(out["range_image"], out["valid_mask"], cfg)
    # M = H valid points; each ought to closely match input modulo azimuth bin snap.
    assert pts_back.shape == (H, 3)
    # ranges should agree exactly (we just round-trip ray * r)
    r_back = pts_back.norm(dim=-1)
    r_sorted, _ = torch.sort(r_back)
    r_in, _ = torch.sort(r)
    assert torch.allclose(r_sorted, r_in, atol=1e-3)
