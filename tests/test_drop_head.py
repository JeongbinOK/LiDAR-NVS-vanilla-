"""
A3.4 — DropHead unit tests.

Covers:
  * Output shape and value range (p_drop ∈ [0, 1]).
  * The geometric-miss invariant: pixels with α_accum = 0 must produce
    p_drop = 1 regardless of MLP output.
  * Smoke integration with the LiDAR rasterizer: run the full pipeline
    (rasterise → drop head) end-to-end on a synthetic scene and check that
    well-covered pixels live in [0, 1] and uncovered pixels stay near 1.
  * make_lidar_ray_grid produces unit vectors agreeing with the kernel's
    spherical formula.
"""

from __future__ import annotations

import math

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# CPU tests (no CUDA needed) — pure-Python module behaviour.
# ---------------------------------------------------------------------------


def test_make_lidar_ray_grid_unit_vectors():
    from nn.render_utils import make_lidar_ray_grid

    H, W = 8, 32
    el_min = math.radians(-30.0)
    el_max = math.radians(+10.0)
    ray = make_lidar_ray_grid(H, W, el_min, el_max)
    assert ray.shape == (3, H, W)

    norms = ray.pow(2).sum(dim=0).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        f"ray dirs not unit length: max |‖d‖-1| = "
        f"{(norms - 1).abs().max().item():.2e}"
    )


def test_make_lidar_ray_grid_matches_kernel_formula():
    """Hand-compute one pixel's ray direction and compare against the helper."""
    from nn.render_utils import make_lidar_ray_grid

    H, W = 16, 64
    el_min, el_max = math.radians(-30.0), math.radians(+10.0)
    ray = make_lidar_ray_grid(H, W, el_min, el_max)

    # Pixel (u=10, v=5)
    u, v = 10, 5
    w_per_rad_az = W / (2.0 * math.pi)
    h_per_rad_el = H / (el_max - el_min)
    az = (u + 0.5) / w_per_rad_az - math.pi
    el = (v + 0.5) / h_per_rad_el + el_min
    expected = torch.tensor([
        math.sin(az) * math.cos(el),
        math.cos(az) * math.cos(el),
        math.sin(el),
    ])
    assert torch.allclose(ray[:, v, u], expected, atol=1e-6)


@pytest.mark.skip(reason="DropHead removed; replaced by per-Gaussian raydrop")
def test_drop_head_geometric_miss_invariant():
    """Pixels with α_accum = 0 must yield p_drop = 1 exactly."""
    from models.head import DropHead

    L = 8
    H, W = 4, 8
    head = DropHead(latent_dim=L)

    latent = torch.randn(1, L, H, W)
    range_ = torch.zeros(1, H, W)
    normal = torch.zeros(1, 3, H, W)
    curvature = torch.zeros(1, H, W)
    alpha_accum = torch.zeros(1, H, W)              # full miss
    ray_dir = torch.zeros(3, H, W); ray_dir[1] = 1  # +y forward

    _, p_drop = head(latent, range_, normal, curvature, alpha_accum, ray_dir)
    assert p_drop.shape == (1, H, W)
    assert torch.allclose(p_drop, torch.ones_like(p_drop)), (
        f"miss pixels not 1: min={p_drop.min()} max={p_drop.max()}"
    )


@pytest.mark.skip(reason="DropHead removed; replaced by per-Gaussian raydrop")
def test_drop_head_output_range_and_shape():
    """Random input → p_drop ∈ [0, 1] and shape matches alpha_accum."""
    from models.head import DropHead

    torch.manual_seed(0)
    L = 16
    H, W = 8, 16
    head = DropHead(latent_dim=L)

    B = 2
    latent = torch.randn(B, L, H, W)
    range_ = torch.rand(B, H, W) * 50.0
    normal = torch.randn(B, 3, H, W)
    curvature = torch.randn(B, H, W) * 0.1
    alpha_accum = torch.rand(B, H, W)
    ray_dir = torch.randn(3, H, W)
    ray_dir = ray_dir / ray_dir.pow(2).sum(0).sqrt().clamp(min=1e-8)

    drop_logit, p_drop = head(latent, range_, normal, curvature, alpha_accum, ray_dir)
    assert drop_logit.shape == (B, H, W)
    assert p_drop.shape == (B, H, W)
    assert (p_drop >= 0.0).all() and (p_drop <= 1.0 + 1e-6).all()


@pytest.mark.skip(reason="DropHead removed; replaced by per-Gaussian raydrop")
def test_drop_head_input_validation():
    from models.head import DropHead

    head = DropHead(latent_dim=4)
    with pytest.raises(ValueError, match="latent must be"):
        head(
            latent=torch.zeros(1, 8, 4, 8),         # wrong L
            range_=torch.zeros(1, 4, 8),
            normal=torch.zeros(1, 3, 4, 8),
            curvature=torch.zeros(1, 4, 8),
            alpha_accum=torch.zeros(1, 4, 8),
            ray_dir=torch.zeros(3, 4, 8),
        )


@pytest.mark.skip(reason="DropHead removed; replaced by per-Gaussian raydrop")
def test_drop_head_gradient_flows_to_mlp():
    """A trivial loss on p_drop must produce non-zero MLP gradients."""
    from models.head import DropHead

    torch.manual_seed(0)
    L = 4
    H, W = 4, 8
    head = DropHead(latent_dim=L)

    latent = torch.randn(1, L, H, W)
    range_ = torch.rand(1, H, W) * 10.0
    normal = torch.randn(1, 3, H, W)
    curvature = torch.randn(1, H, W) * 0.1
    alpha_accum = torch.full((1, H, W), 0.5)        # nonzero hit so MLP path matters
    ray_dir = torch.zeros(3, H, W); ray_dir[1] = 1

    _, p_drop = head(latent, range_, normal, curvature, alpha_accum, ray_dir)
    p_drop.mean().backward()

    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads), "no MLP grads — head is dead"


@pytest.mark.skip(reason="DropHead removed; replaced by per-Gaussian raydrop")
def test_drop_head_mlp_features_are_raw_premultiplied_values():
    from models.head import DropHead

    head = DropHead(latent_dim=2)
    captured = {}

    def capture_input(_module, inputs):
        captured["x"] = inputs[0].detach().clone()

    handle = head.mlp[0].register_forward_pre_hook(capture_input)
    try:
        latent = torch.tensor([[[[0.2]], [[0.4]]]])
        range_ = torch.tensor([[[3.0]]])
        normal = torch.tensor([[[[0.0]], [[0.5]], [[0.0]]]])
        curvature = torch.tensor([[[-0.125]]])
        alpha_accum = torch.tensor([[[0.25]]])
        ray_dir = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])

        head(latent, range_, normal, curvature, alpha_accum, ray_dir)
    finally:
        handle.remove()

    expected = torch.tensor([[
        0.2,
        0.4,
        0.125,
        -1.0,
        math.log1p(3.0),
        0.0,
        1.0,
        0.0,
    ]])
    assert torch.allclose(captured["x"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end integration with the CUDA rasterizer.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="DropHead removed; replaced by per-Gaussian raydrop")
@cuda
def test_drop_head_with_lidar_rasterizer():
    """Wire LiDARRasterizer output into DropHead and check the composition."""
    from diff_quadratic_rasterization import (
        LIDAR_LATENT_DIM,
        LiDARRasterizer,
        make_lidar_settings,
    )
    from models.head import DropHead, make_lidar_ray_grid

    device = "cuda"
    W, H = 1024, 32
    el_min, el_max = math.radians(-30.0), math.radians(+10.0)

    # 5 Gaussians in front of the sensor (+y).
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
    intensity = torch.full((N,), 0.42, device=device, dtype=torch.float32)
    latent = torch.randn(N, LIDAR_LATENT_DIM, device=device, dtype=torch.float32) * 0.1

    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    settings = make_lidar_settings(
        image_height=H, image_width=W,
        el_min_rad=el_min, el_max_rad=el_max,
        viewmatrix=viewmatrix, campos=campos,
    )
    out = LiDARRasterizer(settings)(
        means3D=means3D, means2D=means2D, opacities=opacities,
        scales=scales, rotations=rotations,
        intensity=intensity, latent=latent,
    )

    # Promote each per-image field to [B=1, ...] for the drop head.
    head = DropHead(latent_dim=LIDAR_LATENT_DIM).to(device)
    ray_dir = make_lidar_ray_grid(H, W, el_min, el_max, device=device)

    drop_logit, p_drop = head(
        latent      = out.latent.unsqueeze(0),       # [1, L, H, W]
        range_      = out.range.unsqueeze(0),        # [1, H, W]
        normal      = out.normal.unsqueeze(0),       # [1, 3, H, W]
        curvature   = out.curvature.unsqueeze(0),    # [1, H, W]
        alpha_accum = out.alpha_accum.unsqueeze(0),  # [1, H, W]
        ray_dir     = ray_dir,                       # [3, H, W]
    )

    assert p_drop.shape == (1, H, W)
    assert (p_drop >= 0.0).all() and (p_drop <= 1.0 + 1e-6).all()

    # Pixels with no coverage must keep p_drop ≈ 1 (geometric miss term).
    miss = out.alpha_accum < 1e-4
    assert miss.any(), "test scene has no miss pixels — bad coverage geometry"
    assert torch.allclose(
        p_drop[0][miss], torch.ones_like(p_drop[0][miss]), atol=1e-5
    ), f"miss pixel p_drop drifted: max err {(p_drop[0][miss] - 1).abs().max().item():.2e}"

    # Pixels with strong coverage must allow p_drop strictly below 1
    # (otherwise the learned MLP path is dead). With random init this is
    # essentially guaranteed; we check it nonetheless as a smoke signal.
    hit = out.alpha_accum > 0.5
    if hit.any():
        assert p_drop[0][hit].min() < 1.0 - 1e-3, (
            "no hit pixel produced a sub-1 drop — MLP appears stuck"
        )

    assert not torch.isnan(p_drop).any().item()
