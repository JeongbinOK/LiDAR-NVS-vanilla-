"""
Geometry-first QGS head with fixed-center semantics.

The head now keeps the query point as the primitive center anchor and predicts
only rotation / scale residuals for geometry. Appearance (`alpha`, `latent`)
and intensity are handled by separate heads, with intensity explicitly
conditioned on the final geometry.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

_EPS = 1e-8


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    """Build a small SiLU MLP."""
    assert depth in (1, 2), "depth must be 1 or 2"
    if depth == 2:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _skew(v: Tensor) -> Tensor:
    """Return the skew-symmetric matrix `v^` for batched 3-vectors."""
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def _exp_so3(omega: Tensor) -> Tensor:
    """Rodrigues exp-map from local axis-angle to rotation matrices."""
    theta = omega.norm(dim=-1, keepdim=True)
    theta2 = theta * theta
    theta4 = theta2 * theta2

    A = torch.where(
        theta > 1e-4,
        torch.sin(theta) / theta.clamp(min=_EPS),
        1.0 - theta2 / 6.0 + theta4 / 120.0,
    )
    B = torch.where(
        theta > 1e-4,
        (1.0 - torch.cos(theta)) / theta2.clamp(min=_EPS),
        0.5 - theta2 / 24.0 + theta4 / 720.0,
    )

    K = _skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    eye = eye.view(*([1] * (omega.dim() - 1)), 3, 3).expand_as(K)
    return eye + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)


def _nonzero_sign(x: Tensor) -> Tensor:
    """Return +/-1 with zero mapped to +1."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


class GeometryHead(nn.Module):
    """Predict local rotation / ordered-scale residuals and scalar gates."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        gated: bool = True,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.gated = bool(gated)
        self.mlp = _make_mlp(feature_dim + 1, hidden_dim, 6, depth=2)
        if self.gated:
            self.gate = _make_mlp(6, max(hidden_dim // 2, 16), 2, depth=1)
            nn.init.constant_(self.gate[-1].bias, gate_bias_init)

    def forward(
        self,
        features: Tensor,
        gate_input: Tensor,
        is_dynamic_flag: Tensor,
        use_geom_init: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dyn = is_dynamic_flag.unsqueeze(-1)
        raw = self.mlp(torch.cat([features, dyn], dim=-1))

        if self.gated:
            gates = torch.sigmoid(self.gate(gate_input))
        else:
            gates = torch.ones(
                *features.shape[:-1],
                2,
                dtype=features.dtype,
                device=features.device,
            )

        subfloor = ~use_geom_init
        if subfloor.any():
            gates = gates.clone()
            gates[subfloor] = 1.0

        return raw, gates[..., 0], gates[..., 1]


class AlphaHead(nn.Module):
    """Predict opacity logit after geometry is fixed."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # geometry summary (6) + dynamic flag (1)
        self.mlp = _make_mlp(feature_dim + 7, hidden_dim, 1, depth=2)

    def forward(
        self,
        features: Tensor,
        geometry_summary: Tensor,
        is_dynamic_flag: Tensor,
    ) -> Tensor:
        dyn = is_dynamic_flag.unsqueeze(-1)
        inp = torch.cat([features, geometry_summary, dyn], dim=-1)
        return self.mlp(inp).squeeze(-1)


class AppearanceHead(nn.Module):
    """Predict latent appearance channels from backbone features."""

    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.mlp = _make_mlp(feature_dim + 1, hidden_dim, latent_dim, depth=2)

    def forward(self, features: Tensor, is_dynamic_flag: Tensor) -> Tensor:
        dyn = is_dynamic_flag.unsqueeze(-1)
        return self.mlp(torch.cat([features, dyn], dim=-1))


class IntensityHead(nn.Module):
    """Predict an intensity logit residual after geometry is fixed."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # geometry summary (6) + dynamic flag (1) + intensity anchor (1)
        self.mlp = _make_mlp(feature_dim + 8, hidden_dim, 1, depth=2)

    def forward(
        self,
        features: Tensor,
        geometry_summary: Tensor,
        is_dynamic_flag: Tensor,
        intensity_input: Tensor,
    ) -> Tensor:
        dyn = is_dynamic_flag.unsqueeze(-1)
        intensity_anchor = intensity_input.unsqueeze(-1)
        inp = torch.cat([features, geometry_summary, dyn, intensity_anchor], dim=-1)
        return self.mlp(inp).squeeze(-1)


class QGSHead(nn.Module):
    """Fixed-center geometry-first QGS predictor."""

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        gated: bool = True,
        gate_bias_init: float = -2.0,
        alpha_bias_init: float = 2.2,
        intensity_residual: bool = True,
        center_mode: str = "fixed",
        rot_tilt_deg: float = 10.0,
        rot_spin_deg: float = 30.0,
        scale_log_mean_bound: float | None = None,
        scale_log_gap_bound: float | None = None,
        s3_log_bound: float | None = None,
        s3_fallback_abs: float = 0.05,
    ) -> None:
        super().__init__()

        center_mode = str(center_mode).lower()
        if center_mode != "fixed":
            raise ValueError(
                f"QGSHead now only supports fixed centers; got center_mode={center_mode!r}"
            )

        self.center_mode = center_mode
        self.alpha_bias_init = float(alpha_bias_init)
        self.intensity_residual = bool(intensity_residual)
        self.rot_tilt_bound = math.radians(float(rot_tilt_deg))
        self.rot_spin_bound = math.radians(float(rot_spin_deg))
        self.scale_log_mean_bound = (
            float(scale_log_mean_bound) if scale_log_mean_bound is not None else math.log(2.0)
        )
        self.scale_log_gap_bound = (
            float(scale_log_gap_bound) if scale_log_gap_bound is not None else math.log(1.5)
        )
        self.s3_log_bound = float(s3_log_bound) if s3_log_bound is not None else math.log(1.5)
        self.s3_fallback_abs = float(s3_fallback_abs)
        self.s12_fallback = 0.05

        self.geometry_head = GeometryHead(
            feature_dim,
            hidden_dim,
            gated=gated,
            gate_bias_init=gate_bias_init,
        )
        self.alpha_head = AlphaHead(feature_dim, hidden_dim)
        self.appearance_head = AppearanceHead(feature_dim, hidden_dim, latent_dim)
        self.intensity_head = IntensityHead(feature_dim, hidden_dim)

    def forward(
        self,
        features: Tensor,
        geom_init: dict,
        is_dynamic_flag: Tensor,
        intensity_input: Tensor | None = None,
    ) -> dict:
        c_init = geom_init["c_init"]
        R_init = geom_init["R_init"]
        s_init = geom_init["s_init"]
        fit_quality = geom_init["fit_quality"]
        use_geom_init = geom_init["use_geom_init"]
        tangent_aniso = geom_init["tangent_aniso"]
        curvature_aniso = geom_init["curvature_aniso"]

        if intensity_input is None:
            raise ValueError("QGSHead requires intensity_input for the intensity residual path")

        gate_input = torch.cat(
            [
                fit_quality,
                tangent_aniso.unsqueeze(-1),
                curvature_aniso.unsqueeze(-1),
            ],
            dim=-1,
        )

        raw_geom, g_rot, g_scale = self.geometry_head(
            features,
            gate_input,
            is_dynamic_flag,
            use_geom_init,
        )

        raw_omega = raw_geom[..., :3]
        raw_mu = raw_geom[..., 3]
        raw_gap = raw_geom[..., 4]
        raw_s3 = raw_geom[..., 5]

        omega_bound = raw_geom.new_tensor(
            [self.rot_tilt_bound, self.rot_tilt_bound, self.rot_spin_bound]
        )
        omega_local = g_rot.unsqueeze(-1) * torch.tanh(raw_omega) * omega_bound
        R = R_init @ _exp_so3(omega_local)

        s1_base = torch.where(
            use_geom_init,
            s_init[..., 0].clamp(min=_EPS),
            torch.full_like(s_init[..., 0], self.s12_fallback),
        )
        s2_base = torch.where(
            use_geom_init,
            s_init[..., 1].clamp(min=_EPS),
            torch.full_like(s_init[..., 1], self.s12_fallback),
        )

        mu_init = 0.5 * (torch.log(s1_base) + torch.log(s2_base))
        gap_init = torch.log(s1_base) - torch.log(s2_base)

        delta_mu = g_scale * torch.tanh(raw_mu) * self.scale_log_mean_bound
        delta_gap = g_scale * torch.tanh(raw_gap) * self.scale_log_gap_bound
        mu = mu_init + delta_mu
        gap = torch.clamp(gap_init + delta_gap, min=0.0)

        s1 = torch.exp(mu + 0.5 * gap)
        s2 = torch.exp(mu - 0.5 * gap)

        s3_init = s_init[..., 2]
        s3_sign = torch.where(use_geom_init, _nonzero_sign(s3_init), torch.ones_like(s3_init))
        s3_abs_base = torch.where(
            use_geom_init,
            s3_init.abs().clamp(min=_EPS),
            torch.full_like(s3_init, self.s3_fallback_abs),
        )
        delta_log_abs_s3 = g_scale * torch.tanh(raw_s3) * self.s3_log_bound
        s3 = s3_sign * s3_abs_base * torch.exp(delta_log_abs_s3)

        normal = R[..., :, 2]
        geometry_summary = torch.cat(
            [
                torch.log(s1.clamp(min=_EPS)).unsqueeze(-1),
                torch.log(s2.clamp(min=_EPS)).unsqueeze(-1),
                s3.unsqueeze(-1),
                normal,
            ],
            dim=-1,
        )
        logit_alpha = self.alpha_head(features, geometry_summary, is_dynamic_flag)
        alpha = torch.sigmoid(logit_alpha + self.alpha_bias_init)
        latent = self.appearance_head(features, is_dynamic_flag)
        delta_logit_intensity = self.intensity_head(
            features,
            geometry_summary,
            is_dynamic_flag,
            intensity_input,
        )

        if self.intensity_residual:
            i_safe = intensity_input.clamp(1e-3, 1.0 - 1e-3)
            i_anchor_logit = torch.log(i_safe / (1.0 - i_safe))
            intensity = torch.sigmoid(i_anchor_logit + delta_logit_intensity)
        else:
            intensity = torch.sigmoid(delta_logit_intensity)

        return {
            "center": c_init,
            "R": R,
            "s": torch.stack([s1, s2, s3], dim=-1),
            "alpha": alpha,
            "intensity": intensity,
            "latent": latent,
            "aux": {
                "g_rot": g_rot,
                "g_scale": g_scale,
                "omega_local": omega_local,
                "delta_mu": delta_mu,
                "delta_gap": delta_gap,
                "delta_log_abs_s3": delta_log_abs_s3,
                "delta_logit_intensity": delta_logit_intensity,
                "tangent_aniso": tangent_aniso,
                "curvature_aniso": curvature_aniso,
                "used_init": use_geom_init,
                "gate_input": gate_input,
                "center_mode": self.center_mode,
            },
        }
