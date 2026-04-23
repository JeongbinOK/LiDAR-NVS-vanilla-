"""Unit tests for the fixed-center geometry-first QGS head."""

from __future__ import annotations

import copy
import math

import torch

from models.head.qgs_head import QGSHead


def _make_geom_init(B: int, N: int, device: str = "cpu", dtype=torch.float32) -> dict:
    s_init = torch.tensor([0.4, 0.2, 0.05], device=device, dtype=dtype).expand(B, N, 3).contiguous()
    tangent_aniso = torch.full((B, N), math.log(2.0), device=device, dtype=dtype)
    curvature_aniso = torch.full((B, N), 0.25, device=device, dtype=dtype)
    return {
        "c_init": torch.randn(B, N, 3, device=device, dtype=dtype),
        "R_init": torch.eye(3, device=device, dtype=dtype).expand(B, N, 3, 3).contiguous(),
        "s_init": s_init,
        "fit_quality": torch.rand(B, N, 4, device=device, dtype=dtype),
        "use_geom_init": torch.ones(B, N, device=device, dtype=torch.bool),
        "kappa1_init": torch.full((B, N), 0.35, device=device, dtype=dtype),
        "kappa2_init": torch.full((B, N), 0.15, device=device, dtype=dtype),
        "tangent_aniso": tangent_aniso,
        "curvature_aniso": curvature_aniso,
    }


def _zero_module(module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()


def test_fixed_center_removes_center_residual_outputs():
    head = QGSHead(feature_dim=8, latent_dim=4, hidden_dim=16, gated=True, center_mode="fixed")
    feats = torch.randn(1, 6, 8)
    geom = _make_geom_init(1, 6)
    intensity = torch.full((1, 6), 0.42)

    out = head(feats, geom, torch.zeros(1, 6), intensity_input=intensity)

    assert torch.allclose(out["center"], geom["c_init"])
    assert "delta_c" not in out["aux"]
    assert "delta_quat" not in out["aux"]


def test_geometry_head_output_excludes_dead_quaternion_w_channel():
    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False, center_mode="fixed")
    assert head.geometry_head.mlp[-1].out_features == 6


def test_subfloor_forces_scalar_gates_to_one():
    torch.manual_seed(0)
    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=True, center_mode="fixed")
    feats = torch.randn(1, 10, 8)
    geom = _make_geom_init(1, 10)
    geom["use_geom_init"][0, :3] = False
    intensity = torch.full((1, 10), 0.42)

    out = head(feats, geom, torch.zeros(1, 10), intensity_input=intensity)

    assert torch.allclose(out["aux"]["g_rot"][0, :3], torch.ones(3))
    assert torch.allclose(out["aux"]["g_scale"][0, :3], torch.ones(3))
    assert (out["aux"]["g_rot"][0, 3:] < 1.0).all()
    assert (out["aux"]["g_scale"][0, 3:] < 1.0).all()


def test_omega_local_respects_axis_bounds():
    head = QGSHead(
        feature_dim=8,
        latent_dim=2,
        hidden_dim=16,
        gated=False,
        center_mode="fixed",
        rot_tilt_deg=10.0,
        rot_spin_deg=30.0,
    )
    feats = torch.randn(1, 12, 8) * 1000.0
    geom = _make_geom_init(1, 12)
    intensity = torch.full((1, 12), 0.42)

    out = head(feats, geom, torch.zeros(1, 12), intensity_input=intensity)
    omega = out["aux"]["omega_local"].abs()
    bounds = torch.tensor(
        [math.radians(10.0), math.radians(10.0), math.radians(30.0)]
    ).view(1, 1, 3)

    assert torch.all(omega <= bounds + 1e-6)


def test_scales_are_always_ordered():
    torch.manual_seed(1)
    head = QGSHead(feature_dim=16, latent_dim=4, hidden_dim=32, gated=True, center_mode="fixed")
    feats = torch.randn(2, 32, 16)
    geom = _make_geom_init(2, 32)
    intensity = torch.full((2, 32), 0.42)

    out = head(feats, geom, torch.zeros(2, 32), intensity_input=intensity)

    assert torch.all(out["s"][..., 0] >= out["s"][..., 1])


def test_gap_increase_raises_scale_ratio():
    feats = torch.zeros(1, 4, 8)
    geom = _make_geom_init(1, 4)
    intensity = torch.full((1, 4), 0.42)

    head_lo = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False, center_mode="fixed")
    head_hi = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False, center_mode="fixed")
    _zero_module(head_lo)
    _zero_module(head_hi)

    with torch.no_grad():
        head_lo.geometry_head.mlp[-1].bias[4] = -4.0
        head_hi.geometry_head.mlp[-1].bias[4] = 4.0

    out_lo = head_lo(feats, geom, torch.zeros(1, 4), intensity_input=intensity)
    out_hi = head_hi(feats, geom, torch.zeros(1, 4), intensity_input=intensity)
    ratio_lo = (out_lo["s"][..., 0] / out_lo["s"][..., 1]).mean()
    ratio_hi = (out_hi["s"][..., 0] / out_hi["s"][..., 1]).mean()

    assert ratio_hi > ratio_lo


def test_s3_residual_is_sign_preserving():
    feats = torch.zeros(1, 2, 8)
    geom = _make_geom_init(1, 2)
    geom["s_init"][0, 0, 2] = 0.05
    geom["s_init"][0, 1, 2] = -0.05
    intensity = torch.full((1, 2), 0.42)

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False, center_mode="fixed")
    _zero_module(head)
    with torch.no_grad():
        head.geometry_head.mlp[-1].bias[5] = 5.0

    out = head(feats, geom, torch.zeros(1, 2), intensity_input=intensity)

    assert out["s"][0, 0, 2] > 0
    assert out["s"][0, 1, 2] < 0


def test_intensity_head_consumes_final_geometry():
    feats = torch.zeros(1, 4, 8)
    geom_a = _make_geom_init(1, 4)
    geom_b = copy.deepcopy(geom_a)
    geom_b["s_init"][..., 0] = 0.8
    intensity = torch.full((1, 4), 0.42)

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=8, gated=False, center_mode="fixed")
    _zero_module(head)
    with torch.no_grad():
        first = head.intensity_head.mlp[0]
        second = head.intensity_head.mlp[2]
        last = head.intensity_head.mlp[4]
        first.weight[0, 8] = 1.0   # log s1 from geometry summary
        second.weight[0, 0] = 1.0
        last.weight[0, 0] = 1.0

    out_a = head(feats, geom_a, torch.zeros(1, 4), intensity_input=intensity)
    out_b = head(feats, geom_b, torch.zeros(1, 4), intensity_input=intensity)

    assert not torch.allclose(out_a["intensity"], out_b["intensity"])


def test_alpha_head_consumes_final_geometry():
    feats = torch.zeros(1, 4, 8)
    geom_a = _make_geom_init(1, 4)
    geom_b = copy.deepcopy(geom_a)
    geom_b["s_init"][..., 0] = 0.8
    intensity = torch.full((1, 4), 0.42)

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=8, gated=False, center_mode="fixed")
    _zero_module(head)
    with torch.no_grad():
        first = head.alpha_head.mlp[0]
        second = head.alpha_head.mlp[2]
        last = head.alpha_head.mlp[4]
        first.weight[0, 8] = 1.0   # log s1 from geometry summary
        second.weight[0, 0] = 1.0
        last.weight[0, 0] = 1.0

    out_a = head(feats, geom_a, torch.zeros(1, 4), intensity_input=intensity)
    out_b = head(feats, geom_b, torch.zeros(1, 4), intensity_input=intensity)

    assert not torch.allclose(out_a["alpha"], out_b["alpha"])


def test_backward_runs_through_all_outputs():
    torch.manual_seed(2)
    head = QGSHead(feature_dim=8, latent_dim=3, hidden_dim=16, gated=True, center_mode="fixed")
    feats = torch.randn(1, 5, 8, requires_grad=True)
    geom = _make_geom_init(1, 5)
    geom["use_geom_init"][0, :2] = False
    intensity = torch.full((1, 5), 0.42)

    out = head(feats, geom, torch.ones(1, 5), intensity_input=intensity)
    loss = (
        out["R"].pow(2).sum()
        + out["s"].pow(2).sum()
        + out["alpha"].sum()
        + out["intensity"].sum()
        + out["latent"].pow(2).sum()
    )
    loss.backward()

    assert feats.grad is not None
    assert not torch.isnan(feats.grad).any()
