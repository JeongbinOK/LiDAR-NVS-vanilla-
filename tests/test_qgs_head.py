"""Unit tests for the local-tangent signed-QGS head."""

from __future__ import annotations

import copy
import math

import torch

import models.head as head_module
from models.head import LatentHead
from models.head.qgs_head import QGSHead


def _make_geom_init(B: int, N: int, device: str = "cpu", dtype=torch.float32) -> dict:
    s_init = torch.tensor([0.4, -0.2, 0.05], device=device, dtype=dtype).expand(B, N, 3).contiguous()
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


def test_zero_center_residual_keeps_analytic_center():
    head = QGSHead(feature_dim=8, latent_dim=4, hidden_dim=16, gated=True)
    _zero_module(head)
    feats = torch.randn(1, 6, 8)
    geom = _make_geom_init(1, 6)
    intensity = torch.full((1, 6), 0.42)

    out = head(feats, geom, torch.zeros(1, 6), intensity_input=intensity)

    assert torch.allclose(out["center"], geom["c_init"])
    assert "delta_c" in out["aux"]
    assert "delta_quat" not in out["aux"]


def test_geometry_head_output_contract():
    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False)
    assert head.geometry_head.mlp[-1].out_features == 9


def test_subfloor_forces_scalar_gates_to_one():
    torch.manual_seed(0)
    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=True)
    feats = torch.randn(1, 10, 8)
    geom = _make_geom_init(1, 10)
    geom["use_geom_init"][0, :3] = False
    intensity = torch.full((1, 10), 0.42)

    out = head(feats, geom, torch.zeros(1, 10), intensity_input=intensity)

    assert torch.allclose(out["aux"]["g_rot"][0, :3], torch.ones(3))
    assert torch.allclose(out["aux"]["g_center"][0, :3], torch.ones(3))
    assert torch.allclose(out["aux"]["g_scale"][0, :3], torch.ones(3))
    assert (out["aux"]["g_rot"][0, 3:] < 1.0).all()
    assert (out["aux"]["g_center"][0, 3:] < 1.0).all()
    assert (out["aux"]["g_scale"][0, 3:] < 1.0).all()


def test_omega_local_respects_axis_bounds():
    head = QGSHead(
        feature_dim=8,
        latent_dim=2,
        hidden_dim=16,
        gated=False,
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


def test_center_residual_respects_component_bound():
    head = QGSHead(
        feature_dim=8,
        latent_dim=2,
        hidden_dim=16,
        gated=False,
        center_bound=0.3,
    )
    feats = torch.randn(1, 12, 8) * 1000.0
    geom = _make_geom_init(1, 12)
    intensity = torch.full((1, 12), 0.42)

    out = head(feats, geom, torch.zeros(1, 12), intensity_input=intensity)
    delta_c = out["aux"]["delta_c"].abs()

    assert torch.all(delta_c <= 0.3 + 1e-6)


def test_scale_magnitudes_are_always_ordered():
    torch.manual_seed(1)
    head = QGSHead(feature_dim=16, latent_dim=4, hidden_dim=32, gated=True)
    feats = torch.randn(2, 32, 16)
    geom = _make_geom_init(2, 32)
    intensity = torch.full((2, 32), 0.42)

    out = head(feats, geom, torch.zeros(2, 32), intensity_input=intensity)

    assert torch.all(out["s"][..., 0].abs() >= out["s"][..., 1].abs())


def test_gap_increase_raises_scale_ratio():
    feats = torch.zeros(1, 4, 8)
    geom = _make_geom_init(1, 4)
    intensity = torch.full((1, 4), 0.42)

    head_lo = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False)
    head_hi = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False)
    _zero_module(head_lo)
    _zero_module(head_hi)

    with torch.no_grad():
        head_lo.geometry_head.mlp[-1].bias[7] = -4.0
        head_hi.geometry_head.mlp[-1].bias[7] = 4.0

    out_lo = head_lo(feats, geom, torch.zeros(1, 4), intensity_input=intensity)
    out_hi = head_hi(feats, geom, torch.zeros(1, 4), intensity_input=intensity)
    ratio_lo = (out_lo["s"][..., 0].abs() / out_lo["s"][..., 1].abs()).mean()
    ratio_hi = (out_hi["s"][..., 0].abs() / out_hi["s"][..., 1].abs()).mean()

    assert ratio_hi > ratio_lo


def test_s1_s2_signature_is_sign_preserving():
    feats = torch.zeros(1, 2, 8)
    geom = _make_geom_init(1, 2)
    geom["s_init"][0, 0, :2] = torch.tensor([0.4, -0.2])
    geom["s_init"][0, 1, :2] = torch.tensor([-0.4, 0.2])
    intensity = torch.full((1, 2), 0.42)

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False)
    _zero_module(head)

    out = head(feats, geom, torch.zeros(1, 2), intensity_input=intensity)

    assert out["s"][0, 0, 2] > 0
    assert out["s"][0, 0, 0] > 0
    assert out["s"][0, 0, 1] < 0
    assert out["s"][0, 1, 0] < 0
    assert out["s"][0, 1, 1] > 0


def test_s3_is_positive_curvature_magnitude():
    feats = torch.zeros(1, 2, 8)
    geom = _make_geom_init(1, 2)
    geom["s_init"][0, 0, 2] = 0.05
    geom["s_init"][0, 1, 2] = -0.05
    intensity = torch.full((1, 2), 0.42)

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=16, gated=False)
    _zero_module(head)
    with torch.no_grad():
        head.geometry_head.mlp[-1].bias[8] = 5.0

    out = head(feats, geom, torch.zeros(1, 2), intensity_input=intensity)

    assert (out["s"][..., 2] > 0).all()


def test_intensity_head_consumes_final_geometry():
    feats = torch.zeros(1, 4, 8)
    geom_a = _make_geom_init(1, 4)
    geom_b = copy.deepcopy(geom_a)
    geom_b["s_init"][..., 0] = 0.8
    intensity = torch.full((1, 4), 0.42)

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=8, gated=False)
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

    head = QGSHead(feature_dim=8, latent_dim=2, hidden_dim=8, gated=False)
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
    head = QGSHead(feature_dim=8, latent_dim=3, hidden_dim=16, gated=True)
    feats = torch.randn(1, 5, 8, requires_grad=True)
    geom = _make_geom_init(1, 5)
    geom["use_geom_init"][0, :2] = False
    intensity = torch.full((1, 5), 0.42)

    out = head(feats, geom, torch.ones(1, 5), intensity_input=intensity)
    loss = (
        out["R"].pow(2).sum()
        + out["center"].pow(2).sum()
        + out["s"].pow(2).sum()
        + out["alpha"].sum()
        + out["intensity"].sum()
        + out["latent"].pow(2).sum()
    )
    loss.backward()

    assert feats.grad is not None
    assert not torch.isnan(feats.grad).any()


def test_latent_head_is_exported_without_appearance_alias():
    assert LatentHead is head_module.LatentHead
    assert not hasattr(head_module, "AppearanceHead")
