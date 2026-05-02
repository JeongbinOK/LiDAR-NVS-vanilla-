"""Finite-difference gradcheck for the QGS LiDAR CUDA backward pass.

The CUDA kernel runs in float32 only, so torch.autograd.gradcheck (which
expects float64) is unsuitable. We compute analytical gradients via the
autograd Function and compare them to centered finite differences of the
forward output, with a tolerance loose enough for float32 (~5%).

Tested inputs:
  * means3D       (per-Gaussian world position)
  * scales        (signed s1, s2, s3)
  * rotations     (unit quaternion, perturbed in tangent space)
  * opacities     (per-Gaussian alpha)
  * intensity     (per-Gaussian scalar — packed into colors_precomp)
  * latent        (per-Gaussian feature — packed into colors_precomp)

Plus a sanity check that gradients are exactly zero for Gaussians sitting
beyond r_far (the recently added range gate in backward.cu).
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

H, W = 16, 64
EL_MIN = math.radians(-20.0)
EL_MAX = math.radians(20.0)


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


def _build_scene(*, device: str = "cuda", N: int = 3, r_far: float = 100.0):
    """A minimal LiDAR scene with N Gaussians clustered in front of the sensor."""
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM

    means3D = torch.tensor(
        [[0.0, 5.0, 0.0],
         [0.6, 5.0, 0.2],
         [-0.4, 5.5, -0.1]][:N],
        device=device, dtype=torch.float32,
    )
    means2D = torch.zeros_like(means3D)
    scales = torch.full((N, 3), 0.20, device=device, dtype=torch.float32)
    # tiny azimuth tilts so different Gaussians have distinct quat behavior
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [math.cos(0.05), 0.0, 0.0, math.sin(0.05)],
         [math.cos(-0.04), 0.0, math.sin(-0.04), 0.0]][:N],
        device=device, dtype=torch.float32,
    )
    opacities = torch.full((N, 1), 0.7, device=device, dtype=torch.float32)
    intensity = torch.full((N,), 0.5, device=device, dtype=torch.float32)
    latent = (
        0.1 * torch.randn(N, LIDAR_LATENT_DIM, device=device, generator=_gen(0))
    ).to(torch.float32)
    return {
        "means3D": means3D, "means2D": means2D,
        "scales": scales, "rotations": rotations,
        "opacities": opacities, "intensity": intensity, "latent": latent,
        "r_far": r_far,
    }


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def _forward(params: dict) -> torch.Tensor:
    """Run forward pass and return a scalar loss for backprop / FD comparison."""
    from diff_quadratic_rasterization import LiDARRasterizer, make_lidar_settings

    device = params["means3D"].device
    settings = make_lidar_settings(
        image_height=H, image_width=W,
        el_min_rad=EL_MIN, el_max_rad=EL_MAX,
        viewmatrix=torch.eye(4, device=device, dtype=torch.float32),
        campos=torch.zeros(3, device=device, dtype=torch.float32),
        r_far=params.get("r_far", 100.0),
    )
    out = LiDARRasterizer(settings)(
        means3D=params["means3D"],
        means2D=params["means2D"],
        opacities=params["opacities"],
        scales=params["scales"],
        rotations=params["rotations"],
        intensity=params["intensity"],
        latent=params["latent"],
    )
    # Aggregate output channels with non-zero coefficients on each — guarantees
    # gradients propagate to means/scales/rotations (range/normal),
    # to opacities/intensity (intensity), and to latent (latent).
    return (
        out.range.sum()
        + 0.5 * out.intensity.sum()
        + 0.3 * out.normal.abs().sum()
        + 0.2 * out.latent.sum()
    )


def _analytic_grad(params: dict, key: str) -> torch.Tensor:
    """Compute autograd gradient of the scalar loss w.r.t. params[key]."""
    p2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
    p2[key] = p2[key].detach().clone().requires_grad_(True)
    loss = _forward(p2)
    grad = torch.autograd.grad(loss, p2[key], retain_graph=False, create_graph=False)[0]
    return grad.detach()


def _fd_grad_scalar_perturb(params: dict, key: str, eps: float = 1e-3) -> torch.Tensor:
    """Centered finite difference of every scalar element of params[key]."""
    base = params[key].detach().clone()
    fd = torch.zeros_like(base)
    flat = base.view(-1)
    for i in range(flat.numel()):
        e = torch.zeros_like(flat)
        e[i] = eps
        plus = flat + e
        minus = flat - e
        p_plus = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        p_minus = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        p_plus[key] = plus.view_as(base).contiguous()
        p_minus[key] = minus.view_as(base).contiguous()
        with torch.no_grad():
            f_plus = _forward(p_plus).item()
            f_minus = _forward(p_minus).item()
        fd.view(-1)[i] = (f_plus - f_minus) / (2 * eps)
    return fd


def _assert_close(grad_a: torch.Tensor, grad_fd: torch.Tensor, *, tag: str,
                  rtol: float = 5e-2, atol: float = 5e-3):
    abs_err = (grad_a - grad_fd).abs()
    scale = grad_fd.abs().max().clamp(min=1e-6)
    rel = abs_err.max() / scale
    msg = (
        f"{tag}: max_abs_err={abs_err.max().item():.4e}, "
        f"max_rel={rel.item():.3f}, "
        f"||analytic||={grad_a.norm().item():.3e}, ||fd||={grad_fd.norm().item():.3e}"
    )
    assert (abs_err <= atol + rtol * grad_fd.abs()).all() or rel.item() < rtol, msg


# ---------------------------------------------------------------------------
# Per-input gradient tests
# ---------------------------------------------------------------------------
@cuda
def test_grad_means3D_finite_diff():
    p = _build_scene()
    g_a = _analytic_grad(p, "means3D")
    g_fd = _fd_grad_scalar_perturb(p, "means3D", eps=2e-3)
    _assert_close(g_a, g_fd, tag="means3D")


@cuda
def test_grad_scales_finite_diff():
    p = _build_scene()
    g_a = _analytic_grad(p, "scales")
    g_fd = _fd_grad_scalar_perturb(p, "scales", eps=2e-3)
    _assert_close(g_a, g_fd, tag="scales")


@cuda
def test_grad_opacities_finite_diff():
    p = _build_scene()
    g_a = _analytic_grad(p, "opacities")
    g_fd = _fd_grad_scalar_perturb(p, "opacities", eps=1e-3)
    _assert_close(g_a, g_fd, tag="opacities")


@cuda
def test_grad_intensity_finite_diff():
    p = _build_scene()
    g_a = _analytic_grad(p, "intensity")
    g_fd = _fd_grad_scalar_perturb(p, "intensity", eps=1e-3)
    _assert_close(g_a, g_fd, tag="intensity")


@cuda
def test_grad_latent_finite_diff():
    p = _build_scene()
    g_a = _analytic_grad(p, "latent")
    g_fd = _fd_grad_scalar_perturb(p, "latent", eps=1e-3)
    _assert_close(g_a, g_fd, tag="latent")


# ---------------------------------------------------------------------------
# Rotations: tangent-space FD (perturb on the unit-quaternion manifold).
# A scalar perturbation of one quaternion component leaves the manifold and
# is masked by the kernel's defensive normalization, so we instead perturb
# along the three local axes and compare to the analytic raw-component grad
# projected onto the same tangent direction.
# ---------------------------------------------------------------------------
@cuda
def test_grad_rotations_tangent_finite_diff():
    p = _build_scene()
    N = p["rotations"].shape[0]
    device = p["rotations"].device
    eps = 1e-3

    # Analytic gradient w.r.t. raw quaternion components.
    g_a_raw = _analytic_grad(p, "rotations")  # [N, 4]

    # For each Gaussian and each tangent axis, the tangent-perturbation
    # derivative dL/dθ ≈ (L(q ⊗ exp(θ·e/2)) - L(q ⊗ exp(-θ·e/2))) / (2eps).
    # This must match the analytic raw gradient contracted with the same
    # tangent direction in the renormalized-quat manifold.
    for axis in range(3):
        for i in range(N):
            # tangent perturbation: small rotation around axis 'axis'
            dq_plus = torch.zeros(4, device=device, dtype=torch.float32)
            dq_plus[0] = math.cos(0.5 * eps)
            dq_plus[axis + 1] = math.sin(0.5 * eps)
            dq_minus = dq_plus.clone()
            dq_minus[axis + 1] = -dq_minus[axis + 1]

            rot_plus = p["rotations"].detach().clone()
            rot_plus[i] = _quat_mul(rot_plus[i], dq_plus)
            rot_plus[i] = rot_plus[i] / rot_plus[i].norm().clamp(min=1e-8)

            rot_minus = p["rotations"].detach().clone()
            rot_minus[i] = _quat_mul(rot_minus[i], dq_minus)
            rot_minus[i] = rot_minus[i] / rot_minus[i].norm().clamp(min=1e-8)

            with torch.no_grad():
                p_plus = {**p, "rotations": rot_plus}
                p_minus = {**p, "rotations": rot_minus}
                f_plus = _forward(p_plus).item()
                f_minus = _forward(p_minus).item()
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            # Project analytic raw-quat gradient onto the same tangent direction:
            # tangent vector at q in raw-quat coords is 0.5 * (q ⊗ (0, e_axis)).
            q = p["rotations"][i]
            ev = torch.zeros(4, device=device, dtype=torch.float32)
            ev[axis + 1] = 1.0
            tangent = 0.5 * _quat_mul(q.unsqueeze(0), ev.unsqueeze(0)).squeeze(0)
            analytic_tangent = (g_a_raw[i] * tangent).sum().item()

            err = abs(analytic_tangent - fd_tangent)
            scale = max(abs(fd_tangent), 1e-4)
            assert err / scale < 0.10, (
                f"rotations[i={i},axis={axis}]: analytic={analytic_tangent:.4e}, "
                f"fd={fd_tangent:.4e}, rel_err={err/scale:.3f}"
            )


# ---------------------------------------------------------------------------
# r_far gating: a Gaussian outside the range gate must receive exactly zero gradient.
# ---------------------------------------------------------------------------
@cuda
def test_r_far_gating_zeros_gradient():
    """One Gaussian inside r_far, one outside → outside one's grad must be 0."""
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM

    device = "cuda"
    means3D = torch.tensor(
        [[0.0, 5.0, 0.0],     # inside
         [0.0, 200.0, 0.0]],  # outside r_far=100
        device=device, dtype=torch.float32,
    )
    N = 2
    p = {
        "means3D": means3D,
        "means2D": torch.zeros_like(means3D),
        "scales": torch.full((N, 3), 0.20, device=device, dtype=torch.float32),
        "rotations": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            device=device, dtype=torch.float32,
        ),
        "opacities": torch.full((N, 1), 0.7, device=device, dtype=torch.float32),
        "intensity": torch.full((N,), 0.5, device=device, dtype=torch.float32),
        "latent": torch.zeros(N, LIDAR_LATENT_DIM, device=device, dtype=torch.float32),
        "r_far": 100.0,
    }

    g_means = _analytic_grad(p, "means3D")
    g_scales = _analytic_grad(p, "scales")
    g_op = _analytic_grad(p, "opacities")

    # Inside Gaussian must produce non-trivial gradient.
    assert g_means[0].abs().sum() > 0, "inside Gaussian got zero grad — kernel issue"
    # Outside Gaussian must be exactly zero (range-gated in backward).
    assert g_means[1].abs().sum().item() == 0.0, (
        f"outside-r_far Gaussian leaked gradient: {g_means[1].tolist()}"
    )
    assert g_scales[1].abs().sum().item() == 0.0
    assert g_op[1].abs().sum().item() == 0.0
