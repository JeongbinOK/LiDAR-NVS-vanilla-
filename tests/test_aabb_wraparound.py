"""CUDA per-Gaussian AABB vs Python reference (spherical).

The CUDA rasterizer returns `aabb` with shape [N, 4] in **tile** coordinates
(one tile = BLOCK_X × BLOCK_Y = 16 × 16 pixels):
    aabb[i] = (rect_min_x, rect_min_y, rect_max_x, rect_max_y).

This file checks behavioural equivalence between the CUDA path and the Python
reference `spherical_ref.aabb_spherical` for the wraparound logic at azimuth
seam (±π), at the pole, and at the sensor origin.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "renderer" / "lidar_qgs_rasterizer"))

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Must match cuda_rasterizer/config.h
BLOCK_X = 16
BLOCK_Y = 16
SIGMA = 3.0  # default in make_lidar_settings


def _grid_dims(image_height: int, image_width: int) -> tuple[int, int]:
    grid_x = (image_width  + BLOCK_X - 1) // BLOCK_X
    grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
    return grid_x, grid_y


def _run_lidar(
    means3D: torch.Tensor,
    *,
    scale: float,
    image_height: int = 32,
    image_width: int = 128,
    el_min: float = math.radians(-30.0),
    el_max: float = math.radians(30.0),
):
    """Forward through the LiDAR rasterizer and return (aabb [N,4], radii [N])."""
    from diff_quadratic_rasterization import (
        LIDAR_LATENT_DIM,
        LiDARRasterizer,
        make_lidar_settings,
    )
    from tests._lidar_test_helpers import uniform_row_elevation_rad

    device = means3D.device
    N = means3D.shape[0]
    means2D = torch.zeros_like(means3D)
    scales = torch.full((N, 3), scale, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((N, 1), 0.75, device=device, dtype=torch.float32)
    intensity = torch.full((N,), 0.5, device=device, dtype=torch.float32)
    latent = torch.zeros((N, LIDAR_LATENT_DIM), device=device, dtype=torch.float32)
    raydrop = torch.zeros(N, device=device, dtype=torch.float32)

    row_to_el = uniform_row_elevation_rad(el_min, el_max, image_height).to(device)
    settings = make_lidar_settings(
        image_height=image_height,
        image_width=image_width,
        el_min_rad=el_min,
        el_max_rad=el_max,
        row_to_elevation_rad=row_to_el,
        viewmatrix=torch.eye(4, device=device, dtype=torch.float32),
        campos=torch.zeros(3, device=device, dtype=torch.float32),
        sigma=SIGMA,
    )
    out = LiDARRasterizer(settings)(
        means3D=means3D, means2D=means2D, opacities=opacities,
        scales=scales, rotations=rotations,
        intensity=intensity, latent=latent, raydrop=raydrop,
    )
    return out.aabb, out.radii


def _python_ref_wrapped(p_view: torch.Tensor, R_eff: torch.Tensor) -> torch.Tensor:
    """Return [N] bool: would Python ref classify each Gaussian as wrapped?"""
    from python_ref.spherical_ref import aabb_spherical
    return aabb_spherical(p_view, R_eff)["wrapped"]


# ---------------------------------------------------------------------------
# 1. Forward axis (no wrap)
# ---------------------------------------------------------------------------
@cuda
def test_forward_axis_no_wrap():
    """Gaussian at +y forward direction → small rect, no wrap."""
    H, W = 32, 128
    grid_x, _ = _grid_dims(H, W)

    device = "cuda"
    means3D = torch.tensor([[0.0, 5.0, 0.0]], device=device, dtype=torch.float32)
    scale = 0.18
    aabb, radii = _run_lidar(means3D, scale=scale, image_height=H, image_width=W)

    assert radii[0] > 0, "forward Gaussian must be visible"
    rmin_x, _rmin_y, rmax_x, _rmax_y = aabb[0].tolist()
    width = int(rmax_x - rmin_x)
    assert 0 <= rmin_x and rmax_x <= grid_x
    # az_c=0 is far from ±π seam — must not wrap → width should not span full image.
    assert width < grid_x, (
        f"forward-axis Gaussian must not span all {grid_x} tiles, got width={width}"
    )


# ---------------------------------------------------------------------------
# 2. Seam (-y ≈ ±π) → wrap
# ---------------------------------------------------------------------------
@cuda
def test_negative_y_seam_wraps():
    """Gaussian at -y (az_c≈±π) with non-trivial extent → CUDA must use full-width rect."""
    H, W = 32, 128
    grid_x, _ = _grid_dims(H, W)

    device = "cuda"
    # Slightly off ±π so the projected center isn't exactly at seam.
    p = torch.tensor([[0.05, -5.0, 0.0]], device=device, dtype=torch.float32)
    scale = 0.5  # large enough that the half-angle straddles seam from r=5
    aabb, radii = _run_lidar(p, scale=scale, image_height=H, image_width=W)

    assert radii[0] > 0
    rmin_x, _rmin_y, rmax_x, _rmax_y = aabb[0].tolist()

    # Cross-check with Python ref: did it set wrapped?
    R_eff = torch.tensor([SIGMA * scale], device=device)
    wrapped = _python_ref_wrapped(p, R_eff)
    assert bool(wrapped[0]), (
        "test premise broken: Python ref did not classify this Gaussian as wrapped"
    )
    # CUDA forward.cu widens wrapped boxes to cover the full azimuth range.
    assert int(rmin_x) == 0 and int(rmax_x) == grid_x, (
        f"wrapped Gaussian must span full grid_x={grid_x}, got [{rmin_x}, {rmax_x}]"
    )


# ---------------------------------------------------------------------------
# 3. Pole (high elevation): full-circle azimuth
# ---------------------------------------------------------------------------
@cuda
def test_high_elevation_full_circle():
    """Near-pole Gaussian → R_eff/r * 1/cos(el) ≥ 1 → Python ref returns full circle.

    The Gaussian must still be partially within the vertical FOV to be visible.
    """
    H, W = 32, 128
    grid_x, _ = _grid_dims(H, W)

    device = "cuda"
    # el_max = 30°. Place Gaussian near upper edge with a large scale so its
    # angular cap covers all azimuths.
    el = math.radians(28.0)
    r = 4.0
    z = r * math.sin(el)
    xy = r * math.cos(el)
    p = torch.tensor([[0.0, xy, z]], device=device, dtype=torch.float32)
    scale = 1.5
    aabb, radii = _run_lidar(p, scale=scale, image_height=H, image_width=W,
                             el_min=math.radians(-30.0), el_max=math.radians(30.0))

    R_eff = torch.tensor([SIGMA * scale], device=device)
    py = _python_ref_wrapped(p, R_eff)  # full_circle is masked out as "not wrapped"
    # full_circle is signaled in ref by az_min=-π, az_max=π and wrapped=False.
    from python_ref.spherical_ref import aabb_spherical
    ref = aabb_spherical(p, R_eff)
    is_full_circle = (
        torch.isclose(ref["az_min"][0], torch.tensor(-math.pi)) and
        torch.isclose(ref["az_max"][0], torch.tensor(math.pi))
    )
    assert is_full_circle, "test premise broken: ref did not return full-circle az"

    # In the CUDA kernel a full-circle cap is treated like wrap (rectangle widened
    # to [0, W] horizontally). Some near-pole geometries may be entirely above the
    # FOV cap and get culled — accept either visible+full-width or fully culled.
    if radii[0] > 0:
        rmin_x, _rmin_y, rmax_x, _rmax_y = aabb[0].tolist()
        assert int(rmin_x) == 0 and int(rmax_x) == grid_x, (
            f"full-circle Gaussian must span all tiles or be culled, got "
            f"[{rmin_x}, {rmax_x}] (grid_x={grid_x})"
        )


# ---------------------------------------------------------------------------
# 4. Sensor-origin singularity
# ---------------------------------------------------------------------------
@cuda
def test_near_origin_full_sphere():
    """Gaussian very close to sensor origin → R_eff > r → enclosed origin.

    Python ref returns full-sphere (az = [-π, π], el = [-π/2, π/2], wrapped=False).
    CUDA either widens to full or culls if fully behind r_near.
    """
    H, W = 32, 128
    grid_x, _ = _grid_dims(H, W)

    device = "cuda"
    # Place at r=0.4 with R_eff (~sigma*0.5)=1.5 > r → ball encloses origin.
    p = torch.tensor([[0.0, 0.4, 0.0]], device=device, dtype=torch.float32)
    scale = 0.5
    aabb, radii = _run_lidar(p, scale=scale, image_height=H, image_width=W,
                             el_min=math.radians(-30.0), el_max=math.radians(30.0))

    if radii[0] > 0:
        rmin_x, _rmin_y, rmax_x, _rmax_y = aabb[0].tolist()
        assert int(rmin_x) == 0 and int(rmax_x) == grid_x, (
            f"origin-enclosing Gaussian must span all tiles, got [{rmin_x}, {rmax_x}]"
        )


# ---------------------------------------------------------------------------
# 5. Random batch — wrap-flag agreement with Python reference
# ---------------------------------------------------------------------------
@cuda
def test_random_batch_wrap_flag_matches_python_ref():
    """Generate random Gaussians; CUDA aabb must match Python ref's wrap classification.

    For each Gaussian:
      * Python `wrapped=True` ⇒ CUDA rect must span all grid_x tiles.
      * Python `wrapped=False` AND not full-circle ⇒ CUDA rect width < grid_x.
    Culled (radii<=0) Gaussians are skipped.
    """
    H, W = 32, 128
    grid_x, _ = _grid_dims(H, W)

    device = "cuda"
    torch.manual_seed(2026)
    N = 32
    # Sample around the seam with biased scales to get a mix of wrap/no-wrap.
    az = torch.rand(N) * 2 * math.pi - math.pi  # [-π, π]
    el = (torch.rand(N) * 2 - 1) * math.radians(20.0)
    r = 2.0 + torch.rand(N) * 6.0
    x = r * torch.cos(el) * torch.sin(az)
    y = r * torch.cos(el) * torch.cos(az)
    z = r * torch.sin(el)
    means3D = torch.stack([x, y, z], dim=-1).to(device=device, dtype=torch.float32)
    scale = 0.4
    aabb, radii = _run_lidar(means3D, scale=scale, image_height=H, image_width=W,
                             el_min=math.radians(-30.0), el_max=math.radians(30.0))

    R_eff = torch.full((N,), SIGMA * scale, device=device, dtype=torch.float32)
    from python_ref.spherical_ref import aabb_spherical
    ref = aabb_spherical(means3D.cpu(), R_eff.cpu())
    wrapped = ref["wrapped"]
    is_full_circle = (
        torch.isclose(ref["az_min"], torch.tensor(-math.pi)) &
        torch.isclose(ref["az_max"], torch.tensor( math.pi))
    )

    cuda_aabb = aabb.cpu()
    cuda_radii = radii.cpu()
    n_visible = 0
    for i in range(N):
        if cuda_radii[i].item() <= 0:
            continue
        n_visible += 1
        rmin_x = int(cuda_aabb[i, 0].item())
        rmax_x = int(cuda_aabb[i, 2].item())
        full_width = (rmin_x == 0 and rmax_x == grid_x)
        if bool(wrapped[i]) or bool(is_full_circle[i]):
            assert full_width, (
                f"i={i}: ref wrapped/full-circle but CUDA rect=[{rmin_x},{rmax_x}], "
                f"grid_x={grid_x}; az_c={math.atan2(x[i].item(), y[i].item()):.3f}"
            )
        else:
            assert not full_width, (
                f"i={i}: ref non-wrapped but CUDA spans full width "
                f"[{rmin_x},{rmax_x}] (grid_x={grid_x})"
            )
    assert n_visible >= N // 2, f"too many culled ({N - n_visible}/{N})"
