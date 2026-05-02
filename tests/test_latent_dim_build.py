"""Build-time latent dim consistency.

Verifies that the rasterizer build's exported channel constants are
self-consistent for whatever LIDAR_LATENT_DIM the user compiled with, and
that the LiDARRasterizer pack/unpack accepts the matching shape.

To rebuild with a different latent dim:
    LIDAR_LATENT_DIM=<L> pip install -e renderer/lidar_qgs_rasterizer \\
        --force-reinstall --no-deps --no-build-isolation
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


def test_channel_constants_self_consistent():
    from diff_quadratic_rasterization import (
        ALPHA_OFFSET,
        CURVATURE_OFFSET,
        DEPTH_OFFSET,
        LIDAR_INTENSITY_OFFSET,
        LIDAR_LATENT_DIM,
        LIDAR_LATENT_OFFSET,
        MIDDEPTH_OFFSET,
        NORMAL_OFFSET,
        NUM_CHANNELS,
        OUTPUT_CHANNELS,
    )

    # Channel layout invariants from cuda_rasterizer/channel_layout.h.
    assert NUM_CHANNELS >= 1, "must have at least intensity channel"
    assert LIDAR_LATENT_DIM == NUM_CHANNELS - 1
    assert OUTPUT_CHANNELS == NUM_CHANNELS + 10
    assert LIDAR_INTENSITY_OFFSET == 0
    assert LIDAR_LATENT_OFFSET == 1
    assert NORMAL_OFFSET == NUM_CHANNELS
    assert DEPTH_OFFSET == NUM_CHANNELS + 3
    assert ALPHA_OFFSET == NUM_CHANNELS + 4
    assert MIDDEPTH_OFFSET == NUM_CHANNELS + 6
    assert CURVATURE_OFFSET == NUM_CHANNELS + 8


@cuda
def test_pack_unpack_roundtrip_at_built_dim():
    """Forward call with the build's LIDAR_LATENT_DIM produces matching output shape."""
    import math as _m

    from diff_quadratic_rasterization import (
        LIDAR_LATENT_DIM,
        OUTPUT_CHANNELS,
        LiDARRasterizer,
        make_lidar_settings,
    )

    device = "cuda"
    H, W = 16, 64
    N = 4
    means3D = torch.tensor(
        [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0], [-1.0, 5.0, 0.0], [0.0, 6.0, 0.5]],
        device=device, dtype=torch.float32,
    )
    means2D = torch.zeros_like(means3D)
    scales = torch.full((N, 3), 0.18, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.full((N, 1), 0.75, device=device, dtype=torch.float32)
    intensity = torch.full((N,), 0.5, device=device, dtype=torch.float32)
    latent = torch.randn(N, LIDAR_LATENT_DIM, device=device, dtype=torch.float32)

    settings = make_lidar_settings(
        image_height=H, image_width=W,
        el_min_rad=_m.radians(-30.0), el_max_rad=_m.radians(10.0),
        viewmatrix=torch.eye(4, device=device, dtype=torch.float32),
        campos=torch.zeros(3, device=device, dtype=torch.float32),
    )
    out = LiDARRasterizer(settings)(
        means3D=means3D, means2D=means2D, opacities=opacities,
        scales=scales, rotations=rotations,
        intensity=intensity, latent=latent,
    )
    assert out.raw.shape == (OUTPUT_CHANNELS, H, W)
    assert out.latent.shape == (LIDAR_LATENT_DIM, H, W)
    assert out.intensity.shape == (H, W)
    assert out.range.shape == (H, W)
    assert out.normal.shape == (3, H, W)


@cuda
def test_wrong_latent_dim_raises_helpful_error():
    """Mismatched latent dim must raise a ValueError that names the env var."""
    import math as _m

    from diff_quadratic_rasterization import (
        LIDAR_LATENT_DIM,
        LiDARRasterizer,
        make_lidar_settings,
    )

    device = "cuda"
    H, W = 8, 32
    N = 2
    means3D = torch.tensor(
        [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]],
        device=device, dtype=torch.float32,
    )
    settings = make_lidar_settings(
        image_height=H, image_width=W,
        el_min_rad=_m.radians(-30.0), el_max_rad=_m.radians(10.0),
        viewmatrix=torch.eye(4, device=device, dtype=torch.float32),
        campos=torch.zeros(3, device=device, dtype=torch.float32),
    )
    rasterizer = LiDARRasterizer(settings)

    wrong_L = LIDAR_LATENT_DIM + 4   # any value that differs from the build
    bad_latent = torch.zeros(N, wrong_L, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"LIDAR_LATENT_DIM"):
        rasterizer(
            means3D=means3D,
            means2D=torch.zeros_like(means3D),
            opacities=torch.full((N, 1), 0.5, device=device, dtype=torch.float32),
            scales=torch.full((N, 3), 0.18, device=device, dtype=torch.float32),
            rotations=torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]],
                                   device=device, dtype=torch.float32),
            intensity=torch.full((N,), 0.5, device=device, dtype=torch.float32),
            latent=bad_latent,
        )
