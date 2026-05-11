"""
Geometry-first QGS head for local-tangent signed QGS initialisation.

The analytic quadric fit supplies the primitive centre, tangent frame, signed
tangent support, and curvature magnitude.  The head predicts bounded residuals
for centre, rotation, and scale while preserving the `s1/s2` surface signature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

_EPS = 1e-8
_INIT_SUMMARY_DIM = 13


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
    """Predict local centre / rotation / ordered-scale residuals."""

    def __init__(
        self,
        feature_dim: int,
        init_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.mlp = _make_mlp(feature_dim + init_feature_dim + 1, hidden_dim, 9, depth=2)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        features: Tensor,
        init_features: Tensor,
        is_dynamic_flag: Tensor,
    ) -> Tensor:
        dyn = is_dynamic_flag.unsqueeze(-1)
        return self.mlp(torch.cat([features, init_features, dyn], dim=-1))


class InitEncoder(nn.Module):
    """Project per-anchor init quality/shape summary before residual prediction."""

    def __init__(
        self,
        summary_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, init_summary: Tensor) -> Tensor:
        return self.mlp(init_summary)


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


class RaydropHead(nn.Module):
    """Predict per-Gaussian raydrop probability capped at 0.5.

    Cap at 0.5 structurally blocks the miss-BCE cheat path where
    Σ T_i α_i r_i → 1 with high alpha_accum; the only way rendered_raydrop
    can reach 1.0 on miss pixels is via T_N → 1 (desired).
    """

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # geometry summary (6) + dynamic flag (1)
        self.mlp = _make_mlp(feature_dim + 7, hidden_dim, 1, depth=2)
        nn.init.constant_(self.mlp[-1].bias, -2.5)

    def forward(
        self,
        features: Tensor,
        geometry_summary: Tensor,
        is_dynamic_flag: Tensor,
    ) -> Tensor:
        dyn = is_dynamic_flag.unsqueeze(-1)
        inp = torch.cat([features, geometry_summary, dyn], dim=-1)
        return 0.5 * torch.sigmoid(self.mlp(inp).squeeze(-1))


class LatentHead(nn.Module):
    """Predict latent channels from backbone features."""

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
    """Local-tangent geometry-first QGS predictor."""

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        alpha_bias_init: float = 2.2,
        intensity_residual: bool = True,
        center_bound: float = 0.3,
        rot_tilt_deg: float = 10.0,
        rot_spin_deg: float = 30.0,
        scale_log_mean_bound: float | None = None,
        scale_log_gap_bound: float | None = None,
        s3_log_bound: float | None = None,
        s3_fallback_abs: float = 0.05,
        init_summary_dim: int = _INIT_SUMMARY_DIM,
        init_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.init_summary_dim = int(init_summary_dim)
        self.init_feature_dim = int(init_feature_dim)
        self.alpha_bias_init = float(alpha_bias_init)
        self.intensity_residual = bool(intensity_residual)
        self.center_bound = float(center_bound)
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

        self.init_encoder = InitEncoder(
            self.init_summary_dim,
            hidden_dim,
            self.init_feature_dim,
        )
        self.geometry_head = GeometryHead(
            feature_dim,
            self.init_feature_dim,
            hidden_dim,
        )
        self.alpha_head = AlphaHead(feature_dim, hidden_dim)
        self.raydrop_head = RaydropHead(feature_dim, hidden_dim)
        self.latent_head = LatentHead(feature_dim, hidden_dim, latent_dim)
        self.intensity_head = IntensityHead(feature_dim, hidden_dim)

    def forward(
        self,
        features: Tensor,
        geom_init: dict,
        is_dynamic_flag: Tensor,
        intensity_input: Tensor | None = None,
        init_summary: Tensor | None = None,
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
        if init_summary is None:
            init_summary = features.new_zeros(*features.shape[:2], self.init_summary_dim)
        if init_summary.shape[:2] != features.shape[:2] or init_summary.shape[-1] != self.init_summary_dim:
            raise ValueError(
                "init_summary must be [B, N, "
                f"{self.init_summary_dim}], got {tuple(init_summary.shape)}"
            )
        init_features = self.init_encoder(init_summary.to(device=features.device, dtype=features.dtype))

        raw_geom = self.geometry_head(
            features,
            init_features,
            is_dynamic_flag,
        )

        raw_omega = raw_geom[..., :3]
        raw_delta_c = raw_geom[..., 3:6]
        raw_mu = raw_geom[..., 6]
        raw_gap = raw_geom[..., 7]
        raw_s3 = raw_geom[..., 8]

        omega_bound = raw_geom.new_tensor(
            [self.rot_tilt_bound, self.rot_tilt_bound, self.rot_spin_bound]
        )
        omega_local = torch.tanh(raw_omega) * omega_bound
        R = R_init @ _exp_so3(omega_local)

        delta_c = torch.tanh(raw_delta_c) * self.center_bound
        center = c_init + delta_c

        s1_sign = torch.where(use_geom_init, _nonzero_sign(s_init[..., 0]), torch.ones_like(s_init[..., 0]))
        s2_sign = torch.where(use_geom_init, _nonzero_sign(s_init[..., 1]), torch.ones_like(s_init[..., 1]))
        s1_base = torch.where(
            use_geom_init,
            s_init[..., 0].abs().clamp(min=_EPS),
            torch.full_like(s_init[..., 0], self.s12_fallback),
        )
        s2_base = torch.where(
            use_geom_init,
            s_init[..., 1].abs().clamp(min=_EPS),
            torch.full_like(s_init[..., 1], self.s12_fallback),
        )

        mu_init = 0.5 * (torch.log(s1_base) + torch.log(s2_base))
        gap_init = (torch.log(s1_base) - torch.log(s2_base)).clamp(min=0.0)

        delta_mu = torch.tanh(raw_mu) * self.scale_log_mean_bound
        delta_gap = torch.tanh(raw_gap) * self.scale_log_gap_bound
        mu = mu_init + delta_mu
        gap = torch.clamp(gap_init + delta_gap, min=0.0)

        s1_abs = torch.exp(mu + 0.5 * gap)
        s2_abs = torch.exp(mu - 0.5 * gap)
        s1 = s1_sign * s1_abs
        s2 = s2_sign * s2_abs

        s3_init = s_init[..., 2]
        s3_abs_base = torch.where(
            use_geom_init,
            s3_init.abs().clamp(min=_EPS),
            torch.full_like(s3_init, self.s3_fallback_abs),
        )
        delta_log_abs_s3 = torch.tanh(raw_s3) * self.s3_log_bound
        s3 = s3_abs_base * torch.exp(delta_log_abs_s3)

        normal = R[..., :, 2]
        geometry_summary = torch.cat(
            [
                torch.log(s1_abs.clamp(min=_EPS)).unsqueeze(-1),
                torch.log(s2_abs.clamp(min=_EPS)).unsqueeze(-1),
                torch.log(s3.clamp(min=_EPS)).unsqueeze(-1),
                normal,
            ],
            dim=-1,
        )
        logit_alpha = self.alpha_head(features, geometry_summary, is_dynamic_flag)
        alpha = torch.sigmoid(logit_alpha + self.alpha_bias_init)
        raydrop = self.raydrop_head(features, geometry_summary, is_dynamic_flag)
        latent = self.latent_head(features, is_dynamic_flag)
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
            "center": center,
            "R": R,
            "s": torch.stack([s1, s2, s3], dim=-1),
            "alpha": alpha,
            "raydrop": raydrop,
            "intensity": intensity,
            "latent": latent,
            "aux": {
                "omega_local": omega_local,
                "delta_c": delta_c,
                "delta_mu": delta_mu,
                "delta_gap": delta_gap,
                "delta_log_abs_s3": delta_log_abs_s3,
                "delta_logit_intensity": delta_logit_intensity,
                "tangent_aniso": tangent_aniso,
                "curvature_aniso": curvature_aniso,
                "used_init": use_geom_init,
                "fit_quality": fit_quality,
                "init_summary": init_summary,
                "init_features": init_features,
            },
        }
