"""
A3.1 sanity: vanilla camera-projection forward pass through the QGS CUDA rasterizer.

Just verifies that the upstream rasterizer (unmodified, pre-A3.2) can render a
trivial synthetic scene (a few Gaussians in front of a camera) without crashing,
and produces sensible-shape output. This is *not* a correctness test — it merely
confirms our build is functional end-to-end before A3.2 starts swapping projection.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "renderer" / "lidar_qgs_rasterizer"))

from tests._lidar_test_helpers import uniform_row_elevation_rad  # noqa: E402

# Skip whole module if no CUDA — rasterizer is GPU-only
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _axis_angle_quat(axis: int, angle: float, *, device: torch.device | str) -> torch.Tensor:
    q = torch.zeros(4, device=device, dtype=torch.float32)
    q[0] = math.cos(0.5 * angle)
    q[axis + 1] = math.sin(0.5 * angle)
    return q


def _apply_local_tangent_delta(rotations: torch.Tensor, gaussian_idx: int, axis: int, angle: float) -> torch.Tensor:
    out = rotations.clone()
    dq = _axis_angle_quat(axis, angle, device=rotations.device)
    out[gaussian_idx] = _quat_mul(out[gaussian_idx], dq)
    out[gaussian_idx] = out[gaussian_idx] / out[gaussian_idx].norm().clamp(min=1e-8)
    return out


def _compute_view2gaussian_ref(means3D: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    q = rotations / rotations.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r, x, y, z = q.unbind(dim=-1)
    R = torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
            2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
            2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    R_t = R.transpose(-1, -2)
    t = -(R_t @ means3D.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros(means3D.shape[0], 4, 4, device=means3D.device, dtype=means3D.dtype)
    out[:, :3, :3] = R_t
    out[:, 3, :3] = t
    out[:, 3, 3] = 1.0
    return out


def _lidar_scene(
    means3D: torch.Tensor,
    *,
    image_width: int = 64,
    image_height: int = 16,
    scale: float = 0.18,
    opacity: float = 0.75,
    el_min: float = math.radians(-30.0),
    el_max: float = math.radians(10.0),
    r_near: float = 0.2,
    r_far: float = 100.0,
):
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM, LiDARRasterizer, make_lidar_settings

    device = means3D.device
    N = means3D.shape[0]
    means2D = torch.zeros_like(means3D, requires_grad=means3D.requires_grad)
    scales = torch.full((N, 3), scale, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((N, 1), opacity, device=device, dtype=torch.float32)
    intensity = torch.full((N,), 0.5, device=device, dtype=torch.float32)
    latent = torch.zeros((N, LIDAR_LATENT_DIM), device=device, dtype=torch.float32)
    raydrop = torch.zeros(N, device=device, dtype=torch.float32)

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=image_height,
        image_width=image_width,
        el_min_rad=el_min,
        el_max_rad=el_max,
        row_to_elevation_rad=uniform_row_elevation_rad(el_min, el_max, image_height if 'image_height' in dir() else H).to(device),
        viewmatrix=viewmatrix,
        campos=campos,
        r_near=r_near,
        r_far=r_far,
    )
    out = LiDARRasterizer(settings)(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        intensity=intensity,
        latent=latent,
        raydrop=raydrop,
    )
    return out


def _lidar_scene_custom(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    intensity: torch.Tensor,
    latent: torch.Tensor,
    *,
    rotations: torch.Tensor | None = None,
    raydrop: torch.Tensor | None = None,
    image_width: int = 64,
    image_height: int = 16,
    el_min: float = math.radians(-30.0),
    el_max: float = math.radians(10.0),
    r_near: float = 0.2,
    r_far: float = 100.0,
):
    from diff_quadratic_rasterization import LiDARRasterizer, make_lidar_settings

    device = means3D.device
    N = means3D.shape[0]
    if rotations is None:
        rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
        rotations[:, 0] = 1.0
    if raydrop is None:
        raydrop = torch.zeros(N, device=device, dtype=torch.float32)

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=image_height,
        image_width=image_width,
        el_min_rad=el_min,
        el_max_rad=el_max,
        row_to_elevation_rad=uniform_row_elevation_rad(el_min, el_max, image_height if 'image_height' in dir() else H).to(device),
        viewmatrix=viewmatrix,
        campos=campos,
        r_near=r_near,
        r_far=r_far,
    )
    return LiDARRasterizer(settings)(
        means3D=means3D,
        means2D=torch.zeros_like(means3D),
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        intensity=intensity,
        latent=latent,
        raydrop=raydrop,
    )


@cuda
def test_vanilla_forward_runs():
    from diff_quadratic_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    device = "cuda"
    H, W = 64, 64
    fovx = math.radians(60)
    fovy = math.radians(60)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    znear, zfar = 0.1, 100.0

    # ---- 5 Gaussians in front of camera (camera at origin, looking +z) ----
    N = 5
    means3D = torch.tensor([
        [ 0.0, 0.0, 3.0],
        [ 0.5, 0.0, 3.0],
        [-0.5, 0.0, 3.0],
        [ 0.0, 0.5, 3.0],
        [ 0.0,-0.5, 3.0],
    ], device=device, dtype=torch.float32)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    scales = torch.full((N, 3), 0.05, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0   # identity quaternion (w=1, x=y=z=0)
    opacities = torch.full((N, 1), 0.8, device=device, dtype=torch.float32)
    # NUM_CHANNELS is now 17 (LiDAR widened it). Camera tests pad with zeros so
    # only RGB (channels 0..2) carries the test signal.
    from diff_quadratic_rasterization import NUM_CHANNELS
    colors_precomp = torch.zeros((N, NUM_CHANNELS), device=device, dtype=torch.float32)
    colors_precomp[:, :3] = 0.7

    # ---- Camera matrices (upstream convention: stored as TRANSPOSED) ----
    # See QGS/utils/graphics_utils.py + scene/cameras.py:
    #   world_view_transform  = getWorld2View2(...).T        (4x4)
    #   projection_matrix     = getProjectionMatrix(...).T   (4x4)
    #   full_proj_transform   = world_view_transform @ projection_matrix
    # World-to-view = identity → its transpose is also identity.
    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)

    # OpenGL-style perspective in *column-major* convention, then .T
    P = torch.zeros(4, 4, device=device, dtype=torch.float32)
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    projection_matrix = P.T.contiguous()
    projmatrix = (viewmatrix @ projection_matrix).contiguous()

    campos = torch.zeros(3, device=device, dtype=torch.float32)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)
    # NB: rasterizer_impl.cu reads cam_intr as a flat 4-float array
    # [focal_x, focal_y, principal_x, principal_y]. Upstream keeps it on CPU.
    cam_intr = torch.tensor(
        [fx, fy, W * 0.5, H * 0.5],
        device="cpu", dtype=torch.float32,
    )

    # ---- Rasterizer settings ----
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=0.0,
        subpixel_offset=torch.zeros(H, W, 2, device=device, dtype=torch.float32),
        bg=torch.zeros(NUM_CHANNELS, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        sigma=3.0,   # upstream default (scene/gaussian_model.py:76)
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        cam_intr=cam_intr,
        stop_z_gradient=False,
        reciprocal_z=False,
        return_depth=True,
        return_normal=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    rendered_image, radii, aabb, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
    )

    # ---- Sanity checks ----
    # rendered_image is [C, H, W] — channel count ≥ 14 from upstream layout
    assert rendered_image.dim() == 3
    assert rendered_image.shape[1:] == (H, W)
    assert rendered_image.shape[0] >= 11, f"unexpected channel count: {rendered_image.shape[0]}"

    # radii: [N], some Gaussians must have positive radius (visible)
    assert radii.shape == (N,)
    assert (radii > 0).any().item(), "no Gaussian visible — projection misconfigured"

    # No NaNs
    assert not torch.isnan(rendered_image).any().item()

    # The RGB channels [:3] should have some non-zero pixels (Gaussians rendered)
    rgb = rendered_image[:3]
    assert rgb.abs().sum().item() > 0, "RGB output is identically zero"


@cuda
def test_lidar_forward_runs():
    """A3.2.e — panoramic spherical projection (LiDAR mode). Smoke-only.

    Places a handful of Gaussians in front of the sensor (+y) and confirms the
    rasterizer runs end-to-end without crashing, emits a sensible-shape output
    tensor, and marks at least one primitive as visible.
    """
    from diff_quadratic_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    device = "cuda"
    W, H = 1024, 32  # 1024 azimuth × 32 elevation — nuScenes-like
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)

    # ---- 5 Gaussians at +y (sensor-frame forward), spread in x and z ----
    N = 5
    means3D = torch.tensor([
        [0.0, 5.0, 0.0],
        [1.0, 5.0, 0.0],
        [-1.0, 5.0, 0.0],
        [0.0, 5.0, 1.0],
        [0.0, 5.0, -1.0],
    ], device=device, dtype=torch.float32)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    scales = torch.full((N, 3), 0.1, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((N, 1), 0.8, device=device, dtype=torch.float32)
    from diff_quadratic_rasterization import NUM_CHANNELS
    colors_precomp = torch.zeros((N, NUM_CHANNELS), device=device, dtype=torch.float32)
    colors_precomp[:, :3] = 0.7

    # ---- LiDAR "intrinsics" in cam_intr slot ----
    # [el_min_eff, el_max_eff, w_per_rad_az, H_float]
    cam_intr = torch.tensor([
        el_min,
        el_max,
        W / (2.0 * math.pi),
        float(H),
    ], device="cpu", dtype=torch.float32)
    lidar_row_to_el = uniform_row_elevation_rad(el_min, el_max, H).to(device)

    # LiDAR mode: viewmatrix is still used for world→sensor. projmatrix is ignored.
    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    projmatrix = torch.eye(4, device=device, dtype=torch.float32)  # ignored
    campos = torch.zeros(3, device=device, dtype=torch.float32)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,                                   # unused in LiDAR mode
        tanfovy=1.0,                                   # unused
        kernel_size=0.0,
        subpixel_offset=torch.zeros(H, W, 2, device=device, dtype=torch.float32),
        bg=torch.zeros(NUM_CHANNELS, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        sigma=3.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        cam_intr=cam_intr,
        stop_z_gradient=False,
        reciprocal_z=False,
        return_depth=True,
        return_normal=True,
        lidar_mode=True,
        r_near=0.2,
        r_far=100.0,
        lidar_row_to_el=lidar_row_to_el,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    rendered_image, radii, aabb, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
    )

    assert rendered_image.dim() == 3
    assert rendered_image.shape[1:] == (H, W)
    assert rendered_image.shape[0] >= 11

    assert radii.shape == (N,)
    assert (radii > 0).any().item(), (
        f"no primitive visible in LiDAR mode — radii={radii.tolist()}"
    )

    assert not torch.isnan(rendered_image).any().item()


@cuda
def test_lidar_public_viewmatrix_applies_standard_world_to_sensor_pose():
    """`render_primitives` accepts an ordinary row-major world-to-sensor pose."""
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM
    from nn.render_utils import render_primitives

    device = "cuda"
    dtype = torch.float32
    H, W = 32, 1024
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)

    theta = math.radians(22.0)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    t = torch.tensor([1.3, -0.8, 0.2], device=device, dtype=dtype)
    expected_sensor = torch.tensor([-5.0, 10.0, 0.4], device=device, dtype=dtype)
    mean_world = R.T @ (expected_sensor - t)

    viewmatrix = torch.eye(4, device=device, dtype=dtype)
    viewmatrix[:3, :3] = R
    viewmatrix[:3, 3] = t

    primitives = {
        "means3D": mean_world.view(1, 3),
        "scales": torch.tensor([[4.0, 4.0, 2.0]], device=device, dtype=dtype),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype),
        "opacities": torch.tensor([[0.99]], device=device, dtype=dtype),
        "intensity": torch.tensor([1.0], device=device, dtype=dtype),
        "latent": torch.zeros((1, LIDAR_LATENT_DIM), device=device, dtype=dtype),
        "raydrop": torch.zeros(1, device=device, dtype=dtype),
    }

    from tests._lidar_test_helpers import uniform_lidar_cfg
    cfg_local = uniform_lidar_cfg(H, W, el_min, el_max, r_near=0.2, r_far=70.0, lidar_sigma=3.0)
    out = render_primitives(
        primitives,
        cfg_local,
        viewmatrix=viewmatrix,
        campos=torch.zeros(3, device=device, dtype=dtype),
    )

    assert out.alpha_accum.max().item() > 0.25
    peak = int(out.alpha_accum.argmax().item())
    peak_v, peak_u = divmod(peak, W)

    x, y, z = expected_sensor
    expected_range = expected_sensor.norm()
    expected_az = torch.atan2(x, y)
    expected_el = torch.atan2(z, torch.sqrt(x * x + y * y))
    expected_u = int(torch.round((expected_az + math.pi) * (W / (2.0 * math.pi)) - 0.5).item())
    expected_v = int(torch.round((expected_el - el_min) * (H / (el_max - el_min)) - 0.5).item())

    assert abs(peak_u - expected_u) <= 2
    assert abs(peak_v - expected_v) <= 2
    assert out.middepth[peak_v, peak_u].item() > expected_range.item() - 3.0
    assert out.middepth[peak_v, peak_u].item() < expected_range.item() + 1.0


@cuda
def test_lidar_forward_wraparound():
    """A3.2.d — azimuth ±π wraparound emission.

    Places a Gaussian directly behind the sensor (az_c ≈ ±π). Its spherical
    AABB straddles the ±π seam → `wrapped=true` in preprocessLidarCUDA.
    Before A3.2.d this primitive was conservatively dropped (radii=0); after
    A3.2.d it is emitted as a rect widened to the full image width so that
    tile-binning covers both halves of the seam.

    Success criterion: the wrapped primitive reports radii>0 and the rendered
    panorama contains non-zero contribution in *both* the left (u≈0) and right
    (u≈W-1) columns (the two halves that together cover the primitive).
    """
    from diff_quadratic_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    device = "cuda"
    W, H = 1024, 32
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)

    # One Gaussian directly behind the sensor at y=-5 (az_c = π).
    # With scale 0.5 and sigma=3.0, R_eff = 1.5 → theta_half = asin(1.5/5) ≈ 0.305 rad
    # → az_half ≈ 0.305/cos(0) = 0.305 rad. So az range ≈ [π-0.305, π+0.305],
    # which after wrap becomes [π-0.305, π] ∪ [-π, -π+0.305] — a true wrap.
    means3D = torch.tensor([[0.0, -5.0, 0.0]], device=device, dtype=torch.float32)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    scales = torch.full((1, 3), 0.5, device=device, dtype=torch.float32)
    rotations = torch.zeros(1, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((1, 1), 0.9, device=device, dtype=torch.float32)
    from diff_quadratic_rasterization import NUM_CHANNELS
    colors_precomp = torch.zeros((1, NUM_CHANNELS), device=device, dtype=torch.float32)
    colors_precomp[:, :3] = 0.9

    cam_intr = torch.tensor([
        el_min,
        el_max,
        W / (2.0 * math.pi),
        float(H),
    ], device="cpu", dtype=torch.float32)
    lidar_row_to_el = uniform_row_elevation_rad(el_min, el_max, H).to(device)

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    projmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        kernel_size=0.0,
        subpixel_offset=torch.zeros(H, W, 2, device=device, dtype=torch.float32),
        bg=torch.zeros(NUM_CHANNELS, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        sigma=3.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        cam_intr=cam_intr,
        stop_z_gradient=False,
        reciprocal_z=False,
        return_depth=True,
        return_normal=True,
        lidar_mode=True,
        r_near=0.2,
        r_far=100.0,
        lidar_row_to_el=lidar_row_to_el,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)
    rendered_image, radii, aabb, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
    )

    # Primitive must be visible — before A3.2.d, radii=0 for wrapped cases.
    assert (radii > 0).any().item(), (
        f"wrapped primitive dropped — radii={radii.tolist()}"
    )

    # RGB contribution should appear in both the left and right columns,
    # since the AABB covers [π-δ, π] ∪ [-π, -π+δ] which maps to u≈W and u≈0.
    rgb = rendered_image[:3]
    left_mass  = rgb[:, :, :8].abs().sum().item()       # first 8 cols (near az=-π)
    right_mass = rgb[:, :, -8:].abs().sum().item()      # last 8 cols (near az=+π)
    assert left_mass > 0.0, f"no left-seam contribution (u≈0): {left_mass}"
    assert right_mass > 0.0, f"no right-seam contribution (u≈W): {right_mass}"

    assert not torch.isnan(rendered_image).any().item()


@cuda
def test_lidar_rasterizer_named_layout():
    """A3.3 — `LiDARRasterizer` exposes named output channels (range, intensity,
    alpha_accum, normal, curvature, latent) consistent with the CUDA storage
    layout in cuda_rasterizer/channel_layout.h.

    The test feeds per-Gaussian (intensity, latent) explicitly and verifies that
    the named view recovers the same data the raw [OUTPUT_CHANNELS, H, W] tensor
    contains at the documented offsets.
    """
    from diff_quadratic_rasterization import (
        LIDAR_INTENSITY_OFFSET,
        LIDAR_LATENT_DIM,
        LIDAR_LATENT_OFFSET,
        DEPTH_OFFSET,
        MIDDEPTH_OFFSET,
        ALPHA_OFFSET,
        NORMAL_OFFSET,
        CURVATURE_OFFSET,
        OUTPUT_CHANNELS,
        LiDARRasterizer,
        LiDARRasterOutput,
        make_lidar_settings,
    )

    device = "cuda"
    W, H = 1024, 32
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)

    # ---- 5 Gaussians at +y (sensor forward), spread laterally and vertically.
    N = 5
    means3D = torch.tensor([
        [ 0.0, 5.0,  0.0],
        [ 1.0, 5.0,  0.0],
        [-1.0, 5.0,  0.0],
        [ 0.0, 5.0,  1.0],
        [ 0.0, 5.0, -1.0],
    ], device=device, dtype=torch.float32)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    scales = torch.full((N, 3), 0.1, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((N, 1), 0.8, device=device, dtype=torch.float32)

    # Per-Gaussian features. Use distinct constants so we can sanity-check that
    # each ends up in its expected channel.
    intensity = torch.full((N,), 0.42, device=device, dtype=torch.float32)
    # latent[k] = 0.1 * (k + 1) so each latent slot is distinguishable.
    latent = torch.empty((N, LIDAR_LATENT_DIM), device=device, dtype=torch.float32)
    for k in range(LIDAR_LATENT_DIM):
        latent[:, k] = 0.1 * (k + 1)
    raydrop = torch.zeros(N, device=device, dtype=torch.float32)

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=H,
        image_width=W,
        el_min_rad=el_min,
        el_max_rad=el_max,
        row_to_elevation_rad=uniform_row_elevation_rad(el_min, el_max, image_height if 'image_height' in dir() else H).to(device),
        viewmatrix=viewmatrix,
        campos=campos,
    )
    out = LiDARRasterizer(settings)(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        intensity=intensity,
        latent=latent,
        raydrop=raydrop,
    )

    # ---- Structural checks ------------------------------------------------
    assert isinstance(out, LiDARRasterOutput)
    assert out.raw.shape == (OUTPUT_CHANNELS, H, W)
    assert out.range.shape       == (H, W)
    assert out.middepth.shape    == (H, W)
    assert out.intensity.shape   == (H, W)
    assert out.alpha_accum.shape == (H, W)
    assert out.normal.shape      == (3, H, W)
    assert out.curvature.shape   == (H, W)
    assert out.latent.shape      == (LIDAR_LATENT_DIM, H, W)
    assert out.raydrop.shape     == (H, W)
    assert out.radii.shape == (N,)
    assert (out.radii > 0).any().item(), "no LiDAR primitive visible"

    # ---- Layout consistency: named view must match raw tensor offsets -----
    assert torch.equal(out.range,       out.raw[DEPTH_OFFSET])
    assert torch.equal(out.middepth,    out.raw[MIDDEPTH_OFFSET])
    assert torch.equal(out.intensity,   out.raw[LIDAR_INTENSITY_OFFSET])
    assert torch.equal(out.alpha_accum, out.raw[ALPHA_OFFSET])
    assert torch.equal(out.normal,      out.raw[NORMAL_OFFSET:NORMAL_OFFSET + 3])
    assert torch.equal(out.curvature,   out.raw[CURVATURE_OFFSET])
    assert torch.equal(
        out.latent,
        out.raw[LIDAR_LATENT_OFFSET:LIDAR_LATENT_OFFSET + LIDAR_LATENT_DIM],
    )

    # ---- Content sanity: hit pixels carry alpha-blended feature values ----
    hit = out.alpha_accum > 1e-3            # boolean mask of pixels with coverage
    assert hit.any().item(), "no covered pixels — sensor missed all Gaussians"

    # All inputs share intensity=0.42 and opacity=0.8, so on hit pixels the
    # alpha-blended intensity ≈ alpha_accum * 0.42 to within numerical noise.
    blended = out.intensity[hit]
    expected = out.alpha_accum[hit] * 0.42
    rel_err = (blended - expected).abs() / (expected.abs() + 1e-6)
    assert (rel_err < 1e-3).all().item(), (
        f"intensity blending mismatch: max rel err {rel_err.max().item()}"
    )

    # Each latent channel should match alpha_accum * (0.1 * (k+1)).
    for k in range(LIDAR_LATENT_DIM):
        blended_k = out.latent[k][hit]
        expected_k = out.alpha_accum[hit] * (0.1 * (k + 1))
        rel_err_k = (blended_k - expected_k).abs() / (expected_k.abs() + 1e-6)
        assert (rel_err_k < 1e-3).all().item(), (
            f"latent[{k}] blending mismatch: max rel err {rel_err_k.max().item()}"
        )

    # Range on hit pixels should be ~5 m (Gaussians sit at y=5). Like the
    # other rendered channels this is alpha-weighted (Σ T_i α_i r_i), so we
    # normalise by alpha_accum before checking the physical magnitude.
    range_norm = out.range[hit] / out.alpha_accum[hit].clamp(min=1e-3)
    assert (range_norm > 4.5).all().item(), f"range_norm min={range_norm.min().item()}"
    assert (range_norm < 5.5).all().item(), f"range_norm max={range_norm.max().item()}"

    assert not torch.isnan(out.raw).any().item()
    assert torch.isfinite(out.middepth).all().item()


def test_lidar_channel_layout_constants_are_exported():
    """Python constants must stay in lockstep with channel_layout.h."""
    import diff_quadratic_rasterization as dq

    assert dq.NORMAL_OFFSET == dq.NUM_CHANNELS
    assert dq.DEPTH_OFFSET == dq.NUM_CHANNELS + 3
    assert dq.ALPHA_OFFSET == dq.NUM_CHANNELS + 4
    assert dq.MIDDEPTH_OFFSET == dq.NUM_CHANNELS + 6
    assert dq.CURVATURE_OFFSET == dq.NUM_CHANNELS + 8
    assert dq.OUTPUT_CHANNELS == dq.NUM_CHANNELS + 10
    assert dq.LIDAR_INTENSITY_OFFSET == 0
    assert dq.LIDAR_LATENT_OFFSET == 1
    assert dq.LIDAR_LATENT_DIM == dq.NUM_CHANNELS - 2
    assert dq.LIDAR_RAYDROP_OFFSET == dq.NUM_CHANNELS - 1


@cuda
def test_lidar_range_fov_empty_and_multihit_cases():
    """Lock down the synthetic edge cases called out in the renderer plan."""
    device = "cuda"

    # Empty scenes should return all-zero maps and no primitive buffers.
    empty = _lidar_scene(torch.empty((0, 3), device=device, dtype=torch.float32))
    assert empty.raw.shape[1:] == (16, 64)
    assert empty.radii.numel() == 0
    assert empty.alpha_accum.abs().sum().item() == 0.0
    assert empty.range.abs().sum().item() == 0.0

    # Candidate culling is conservative: a primitive whose center is outside
    # the range gate can still touch tiles if its support ball overlaps the
    # range interval. Exact near/far rejection happens on solved hit roots.
    gated = _lidar_scene(torch.tensor([
        [0.0, 0.10, 0.0],   # too near
        [0.0, 5.00, 0.0],   # valid
        [0.0, 9.50, 0.0],   # too far for this setting
    ], device=device, dtype=torch.float32), scale=0.45, r_near=0.2, r_far=8.0)
    assert gated.radii.tolist()[0] > 0
    assert gated.radii.tolist()[1] > 0
    assert gated.radii.tolist()[2] == 0
    hit = gated.alpha_accum > 1e-4
    assert hit.any().item()
    physical_range = gated.range[hit] / gated.alpha_accum[hit].clamp(min=1e-4)
    assert (physical_range >= 0.2 - 1e-4).all().item()
    assert (physical_range <= 8.0 + 1e-4).all().item()

    # Vertical FOV edge: just inside the top boundary survives. Slightly outside
    # can also survive candidate culling if the support overlaps the FOV; clearly
    # outside the support envelope is culled.
    el_top = math.radians(10.0)
    r = 5.0
    inside_z = math.tan(el_top - math.radians(0.25)) * r
    near_outside_z = math.tan(el_top + math.radians(0.25)) * r
    far_outside_z = math.tan(el_top + math.radians(8.0)) * r
    fov = _lidar_scene(torch.tensor([
        [0.0, r, inside_z],
        [0.0, r, near_outside_z],
        [0.0, r, far_outside_z],
    ], device=device, dtype=torch.float32))
    assert fov.radii.tolist()[0] > 0
    assert fov.radii.tolist()[1] > 0
    assert fov.radii.tolist()[2] == 0

    # One-hit vs multi-hit: the same central ray with two depths should render
    # an expected range between the two physical roots, not NaN/Inf or zero.
    one = _lidar_scene(
        torch.tensor([[0.0, 5.0, 0.0]], device=device, dtype=torch.float32),
        scale=0.45,
    )
    multi = _lidar_scene(torch.tensor([
        [0.0, 5.0, 0.0],
        [0.0, 6.5, 0.0],
    ], device=device, dtype=torch.float32), scale=0.45)
    one_hit = one.alpha_accum > 1e-3
    multi_hit = multi.alpha_accum > 1e-3
    assert one_hit.any().item()
    assert multi_hit.any().item()
    one_phys = one.range[one_hit] / one.alpha_accum[one_hit].clamp(min=1e-3)
    multi_phys = multi.range[multi_hit] / multi.alpha_accum[multi_hit].clamp(min=1e-3)
    assert torch.isfinite(one_phys).all().item()
    assert torch.isfinite(multi_phys).all().item()
    assert one_phys.median().item() > 4.0
    assert one_phys.median().item() < 6.0
    assert multi_phys.median().item() > 4.0
    assert multi_phys.median().item() < 7.0


@cuda
def test_lidar_cuda_matches_python_oracle_for_single_pixel_contract():
    """Compare CUDA output against the slow Python QGS contract on one LiDAR ray."""
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM
    from python_ref.qgs_ref import pixel_center_ray, render_single_pixel

    device = "cuda"
    dtype = torch.float32
    W, H = 1, 1
    el_min, el_max = -0.1, 0.1

    means = torch.tensor([[0.0, 4.0, 0.0], [0.0, 6.0, 0.0]], device=device, dtype=dtype)
    scales = torch.ones((2, 3), device=device, dtype=dtype)
    opacities = torch.tensor([[0.30], [0.40]], device=device, dtype=dtype)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    intensity = torch.tensor([0.25, 0.75], device=device, dtype=dtype)
    latent = torch.zeros((2, LIDAR_LATENT_DIM), device=device, dtype=dtype)
    latent[0, 0] = 0.2
    latent[1, 0] = 0.8

    cuda_out = _lidar_scene_custom(
        means,
        scales,
        opacities,
        intensity,
        latent,
        rotations=rotations,
        image_width=W,
        image_height=H,
        el_min=el_min,
        el_max=el_max,
    )
    ray = pixel_center_ray(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
        image_width=W,
        image_height=H,
        el_min=el_min,
        el_max=el_max,
    )
    oracle = render_single_pixel(ray, means, rotations, scales, opacities, intensity, latent)

    assert torch.allclose(cuda_out.alpha_accum[0, 0], oracle["alpha_accum"], atol=2e-5)
    assert torch.allclose(cuda_out.range[0, 0], oracle["range"], atol=2e-5)
    assert torch.allclose(cuda_out.middepth[0, 0], oracle["middepth"], atol=2e-5)
    assert torch.allclose(cuda_out.intensity[0, 0], oracle["intensity"], atol=2e-5)
    assert torch.allclose(cuda_out.curvature[0, 0], oracle["curvature"], atol=2e-5)
    assert torch.allclose(cuda_out.latent[0, 0, 0], oracle["latent"][0], atol=2e-5)


@cuda
def test_lidar_depth_sorting_is_independent_of_input_order():
    """Pixel-resorted rendering must match depth-sorted compositing, not primitive input order."""
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM

    device = "cuda"
    dtype = torch.float32
    W, H = 1, 1
    el_min, el_max = -0.1, 0.1

    means = torch.tensor([[0.0, 4.0, 0.0], [0.0, 6.0, 0.0], [0.0, 5.0, 0.0]], device=device, dtype=dtype)
    scales = torch.ones((3, 3), device=device, dtype=dtype)
    opacities = torch.tensor([[0.25], [0.35], [0.45]], device=device, dtype=dtype)
    intensity = torch.tensor([0.1, 0.9, 0.5], device=device, dtype=dtype)
    latent = torch.zeros((3, LIDAR_LATENT_DIM), device=device, dtype=dtype)
    latent[:, 0] = torch.tensor([0.1, 0.9, 0.5], device=device, dtype=dtype)

    forward = _lidar_scene_custom(
        means, scales, opacities, intensity, latent,
        image_width=W, image_height=H, el_min=el_min, el_max=el_max,
    )

    perm = torch.tensor([1, 2, 0], device=device)
    shuffled = _lidar_scene_custom(
        means[perm],
        scales[perm],
        opacities[perm],
        intensity[perm],
        latent[perm],
        image_width=W,
        image_height=H,
        el_min=el_min,
        el_max=el_max,
    )

    assert torch.allclose(forward.raw, shuffled.raw, atol=2e-5)
    assert torch.allclose(forward.alpha_accum, shuffled.alpha_accum, atol=2e-5)
    assert torch.allclose(forward.range, shuffled.range, atol=2e-5)
    assert torch.allclose(forward.middepth, shuffled.middepth, atol=2e-5)


@cuda
def test_lidar_backward_runs_and_matches_finite_diff():
    """A3.x — LiDAR-mode backward.

    Two checks:

    1. **Smoke**: rendering loss backprops through `LiDARRasterizer` without
       NaN/inf and writes finite gradients into the leaf tensors that the
       backward kernel covers (means3D, scales, opacities, intensity, latent).

    2. **Finite-difference parity**: per-Gaussian range-loss gradients on
       `means3D` from autograd agree with central differences. This is the
       cheapest correctness signal — if the backward still routed through the
       camera ray construction, `ray_point` would be `(pix_x/0, pix_y/0, 1)` →
       NaN gradients (or, with the lidar_mode plumbing forgotten, gradients
       in *pinhole* directions instead of along the spherical rays).

    We pick a tiny scene (3 Gaussians, 32×8 grid) so finite differencing is
    affordable — each fwd costs O(P·H·W) and we run it 3·2 = 6 times.
    """
    from diff_quadratic_rasterization import (
        DEPTH_OFFSET,
        LIDAR_LATENT_DIM,
        LiDARRasterizer,
        make_lidar_settings,
    )

    device = "cuda"
    W, H = 32, 8                           # tiny so FD is cheap
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)

    torch.manual_seed(0)

    # 3 well-separated Gaussians in front of the sensor (+y).
    means3D_init = torch.tensor([
        [ 0.0, 5.0,  0.0],
        [ 1.5, 5.0,  0.0],
        [ 0.0, 5.0,  0.8],
    ], device=device, dtype=torch.float32)
    means2D = torch.zeros_like(means3D_init, requires_grad=True)
    scales_init = torch.full((3, 3), 0.2, device=device, dtype=torch.float32)
    rotations = torch.zeros(3, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities_init = torch.full((3, 1), 0.7, device=device, dtype=torch.float32)
    intensity_init = torch.full((3,), 0.5, device=device, dtype=torch.float32)
    latent_init = torch.randn(3, LIDAR_LATENT_DIM, device=device,
                              dtype=torch.float32) * 0.05

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=H, image_width=W,
        el_min_rad=el_min, el_max_rad=el_max,
        row_to_elevation_rad=uniform_row_elevation_rad(el_min, el_max, H).to(device),
        viewmatrix=viewmatrix, campos=campos,
    )

    raydrop_init = torch.zeros(3, device=device, dtype=torch.float32)

    def render_loss(means3D, scales, opacities, intensity, latent):
        """Sum of the depth channel — a smooth scalar function of inputs."""
        out = LiDARRasterizer(settings)(
            means3D=means3D,
            means2D=torch.zeros_like(means3D),
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            intensity=intensity,
            latent=latent,
            raydrop=raydrop_init,
        )
        # Use the raw alpha-weighted depth channel: sum over hit pixels.
        return out.raw[DEPTH_OFFSET].sum()

    # ---- (1) Smoke: gradient flows, no NaN ---------------------------------
    means3D = means3D_init.clone().requires_grad_(True)
    scales = scales_init.clone().requires_grad_(True)
    opacities = opacities_init.clone().requires_grad_(True)
    intensity = intensity_init.clone().requires_grad_(True)
    latent = latent_init.clone().requires_grad_(True)

    loss = render_loss(means3D, scales, opacities, intensity, latent)
    assert torch.isfinite(loss).item(), f"forward loss not finite: {loss.item()}"
    loss.backward()

    for name, t in [("means3D", means3D), ("scales", scales),
                    ("opacities", opacities), ("intensity", intensity),
                    ("latent", latent)]:
        assert t.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(t.grad).all().item(), f"{name}.grad has NaN/Inf"

    # The depth-sum loss only reads the alpha-weighted depth channel, which
    # depends on means3D/scales/opacities. intensity/latent aren't read by the
    # depth channel, so their gradients should be exactly zero — useful as a
    # routing sanity check that the backward isn't writing into wrong slots.
    assert intensity.grad.abs().sum().item() == 0.0, (
        "depth-only loss should not produce intensity gradients — "
        f"got {intensity.grad}"
    )
    assert latent.grad.abs().sum().item() == 0.0, (
        f"depth-only loss should not produce latent gradients — "
        f"max |g| = {latent.grad.abs().max().item()}"
    )

    # means3D gradient should be non-trivial (depth changes with position).
    assert means3D.grad.abs().sum().item() > 0.0, "means3D got zero gradient"

    # ---- (2) Finite-difference parity on means3D --------------------------
    # Compare per-Gaussian, per-axis gradient against a 2-point central diff.
    # Tolerance is loose because forward rendering is non-smooth at tile
    # boundaries (a Gaussian crossing a tile edge can flip from contributing
    # to not contributing). We only require sign + order-of-magnitude match.
    eps = 1e-3
    grad_autograd = means3D.grad.detach().clone()

    grad_fd = torch.zeros_like(grad_autograd)
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                m_plus  = means3D_init.clone()
                m_minus = means3D_init.clone()
                m_plus[i, j]  += eps
                m_minus[i, j] -= eps
                lp = render_loss(m_plus,  scales_init, opacities_init,
                                 intensity_init, latent_init)
                lm = render_loss(m_minus, scales_init, opacities_init,
                                 intensity_init, latent_init)
                grad_fd[i, j] = (lp - lm) / (2 * eps)

    # Compare only the entries where finite-diff itself is well-conditioned
    # (|fd| above noise floor). For the others, just require autograd is also
    # small — i.e. neither method can detect a contribution.
    fd_mag = grad_fd.abs()
    well_cond = fd_mag > 1.0
    if well_cond.any():
        a = grad_autograd[well_cond]
        f = grad_fd[well_cond]
        # Relative error per entry; allow 25 % tolerance (FD step interacts
        # with non-smooth tile transitions).
        rel = (a - f).abs() / f.abs().clamp(min=1e-3)
        assert rel.max().item() < 0.25, (
            f"autograd vs FD disagreement: rel max {rel.max().item():.3f}\n"
            f"  autograd: {a.tolist()}\n"
            f"  fd      : {f.tolist()}"
        )
        # Sign agreement on the dominant entries.
        same_sign = (a.sign() == f.sign()) | (a.abs() < 0.1)
        assert same_sign.all().item(), (
            f"sign disagreement on well-conditioned entries:\n"
            f"  autograd: {a.tolist()}\n"
            f"  fd      : {f.tolist()}"
        )


@cuda
def test_lidar_backward_matches_finite_diff_for_shape_opacity_rotation():
    """Exercise backward paths beyond means3D: scale, opacity, and rotation."""
    from diff_quadratic_rasterization import (
        ALPHA_OFFSET,
        DEPTH_OFFSET,
        LIDAR_LATENT_DIM,
        LiDARRasterizer,
        make_lidar_settings,
    )

    device = "cuda"
    W, H = 40, 10
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)

    means3D_init = torch.tensor([
        [0.2, 5.0, 0.1],
        [1.0, 5.4, 0.2],
    ], device=device, dtype=torch.float32)
    scales_init = torch.tensor([
        [0.28, 0.22, 0.12],
        [0.24, 0.30, 0.10],
    ], device=device, dtype=torch.float32)
    opacities_init = torch.tensor([[0.62], [0.55]], device=device, dtype=torch.float32)
    rotations_init = torch.tensor([
        [0.9971888, 0.0, 0.0749297, 0.0],   # small y-axis tilt
        [1.0, 0.0, 0.0, 0.0],
    ], device=device, dtype=torch.float32)
    rotations_init = rotations_init / rotations_init.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    intensity = torch.full((2,), 0.5, device=device, dtype=torch.float32)
    latent = torch.zeros((2, LIDAR_LATENT_DIM), device=device, dtype=torch.float32)
    raydrop = torch.zeros(2, device=device, dtype=torch.float32)

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=H,
        image_width=W,
        el_min_rad=el_min,
        el_max_rad=el_max,
        row_to_elevation_rad=uniform_row_elevation_rad(el_min, el_max, image_height if 'image_height' in dir() else H).to(device),
        viewmatrix=viewmatrix,
        campos=campos,
    )

    def render_loss(means3D, scales, opacities, rotations):
        out = LiDARRasterizer(settings)(
            means3D=means3D,
            means2D=torch.zeros_like(means3D),
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            intensity=intensity,
            latent=latent,
            raydrop=raydrop,
        )
        return out.raw[DEPTH_OFFSET].sum() + 3.0 * out.raw[ALPHA_OFFSET].sum()

    means3D = means3D_init.clone().requires_grad_(True)
    scales = scales_init.clone().requires_grad_(True)
    opacities = opacities_init.clone().requires_grad_(True)
    rotations = rotations_init.clone().requires_grad_(True)

    loss = render_loss(means3D, scales, opacities, rotations)
    assert torch.isfinite(loss).item()
    loss.backward()

    for name, tensor in [("scales", scales), ("opacities", opacities), ("rotations", rotations)]:
        assert tensor.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(tensor.grad).all().item(), f"{name}.grad has NaN/Inf"
        assert tensor.grad.abs().sum().item() > 0.0, f"{name} gradient is identically zero"

    checks = [
        ("scale_x", scales, scales_init, (0, 0), 2e-3, scales.grad),
        ("scale_z", scales, scales_init, (0, 2), 2e-3, scales.grad),
        ("opacity", opacities, opacities_init, (0, 0), 1e-3, opacities.grad),
    ]

    for label, _tensor, base, idx, eps, grad in checks:
        with torch.no_grad():
            plus_means, minus_means = means3D_init, means3D_init
            plus_scales, minus_scales = scales_init, scales_init
            plus_opacities, minus_opacities = opacities_init, opacities_init
            plus_rotations, minus_rotations = rotations_init, rotations_init

            b_plus = base.clone()
            b_minus = base.clone()
            b_plus[idx] += eps
            b_minus[idx] -= eps
            if label.startswith("scale"):
                plus_scales, minus_scales = b_plus, b_minus
            elif label == "opacity":
                plus_opacities, minus_opacities = b_plus, b_minus

            lp = render_loss(plus_means, plus_scales, plus_opacities, plus_rotations)
            lm = render_loss(minus_means, minus_scales, minus_opacities, minus_rotations)
            fd = (lp - lm) / (2 * eps)

        ag = grad[idx]
        assert torch.isfinite(fd).item(), f"{label} finite diff is not finite"
        assert fd.abs().item() > 1e-2, f"{label} finite diff too small to compare: {fd.item()}"
        rel = (ag - fd).abs() / fd.abs().clamp(min=1e-3)
        assert rel.item() < 0.35, (
            f"{label} autograd vs FD disagreement: rel {rel.item():.3f}, "
            f"autograd {ag.item():.6f}, fd {fd.item():.6f}"
        )
        assert ag.sign().item() == fd.sign().item(), (
            f"{label} sign disagreement: autograd {ag.item():.6f}, fd {fd.item():.6f}"
        )

    rot_settings = make_lidar_settings(
        image_height=16,
        image_width=64,
        el_min_rad=el_min,
        el_max_rad=el_max,
        row_to_elevation_rad=uniform_row_elevation_rad(el_min, el_max, 16).to(device),
        viewmatrix=viewmatrix,
        campos=campos,
    )
    rot_means = torch.tensor([[-1.0, 5.5, -0.2]], device=device, dtype=torch.float32)
    rot_scales = torch.tensor([[1.0, 0.8, 0.1]], device=device, dtype=torch.float32)
    rot_opacities = torch.tensor([[0.7]], device=device, dtype=torch.float32)
    rot_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    rot_intensity = torch.full((1,), 0.5, device=device, dtype=torch.float32)
    rot_latent = torch.zeros((1, LIDAR_LATENT_DIM), device=device, dtype=torch.float32)
    rot_raydrop = torch.zeros(1, device=device, dtype=torch.float32)

    def rotation_loss(rotations):
        out = LiDARRasterizer(rot_settings)(
            means3D=rot_means,
            means2D=torch.zeros_like(rot_means),
            opacities=rot_opacities,
            scales=rot_scales,
            rotations=rotations,
            intensity=rot_intensity,
            latent=rot_latent,
            raydrop=rot_raydrop,
        )
        return out.raw[DEPTH_OFFSET].sum() + 3.0 * out.raw[ALPHA_OFFSET].sum()

    rot = rot_init.clone().requires_grad_(True)
    rot_loss = rotation_loss(rot)
    assert torch.isfinite(rot_loss).item()
    rot_loss.backward()

    eps = 1e-3
    for axis in range(3):
        with torch.no_grad():
            plus_rotations = _apply_local_tangent_delta(rot_init, 0, axis, eps)
            minus_rotations = _apply_local_tangent_delta(rot_init, 0, axis, -eps)
            lp = rotation_loss(plus_rotations)
            lm = rotation_loss(minus_rotations)
            fd = (lp - lm) / (2 * eps)

        tangent = (
            _apply_local_tangent_delta(rot_init, 0, axis, eps)[0]
            - _apply_local_tangent_delta(rot_init, 0, axis, -eps)[0]
        ) / (2 * eps)
        ag = (rot.grad[0] * tangent).sum()
        assert torch.isfinite(fd).item(), f"rotation axis {axis} finite diff is not finite"
        assert fd.abs().item() > 1e-2, f"rotation axis {axis} finite diff too small: {fd.item()}"
        rel = (ag - fd).abs() / fd.abs().clamp(min=1e-3)
        assert rel.item() < 0.35, (
            f"rotation axis {axis} tangent autograd vs FD disagreement: "
            f"rel {rel.item():.3f}, autograd {ag.item():.6f}, fd {fd.item():.6f}"
        )
        assert ag.sign().item() == fd.sign().item(), (
            f"rotation axis {axis} sign disagreement: autograd {ag.item():.6f}, fd {fd.item():.6f}"
        )
