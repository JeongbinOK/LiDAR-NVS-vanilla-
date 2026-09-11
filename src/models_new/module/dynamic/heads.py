"""Gaussian attribute and velocity prediction heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ...utils.attention import FourierPositionEncoder3D
from .common import _cfg_get

class GaussianAttributeHead(nn.Module):
    """Time-invariant 2DGS decoder driven by the shared token feature.

    ``gaussians_per_token`` slots share the token's feature but nothing else:
    each owns its own rows of every attribute projection, including the bounded
    position offset, so slots seeded at the same coordinate still learn
    independent geometry. At the default of one slot this is exactly the V3
    head, down to the parameter shapes saved in a checkpoint.

    ``seed_conditioned`` additionally concatenates every slot's seed delta to
    the normalized feature, widening the head's input by
    ``3 * gaussians_per_token``. This is what the adaptive-K router's trunk
    does with its own ``3 * K_max`` channels, so a fixed-count head can carry
    the same geometric conditioning without the router.

    ``trunk`` puts one ``Linear(in, dim) -> SiLU`` in front of the attribute
    projections, which is the other half of what the router's decoder does and
    the only nonlinearity either version has. Without it the whole head is a
    single affine map of the normalized feature.

    Both default to off: every V1-V11.1 checkpoint was saved from the plain
    affine form and must restore unchanged.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1, seed_conditioned: bool = False,
                 trunk: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.offset_bound = float(offset_bound)
        self.gaussians_per_token = int(gaussians_per_token)
        if self.gaussians_per_token < 1:
            raise ValueError("gaussians_per_token must be at least one")
        self.seed_conditioned = bool(seed_conditioned)
        self.seed_dim = (
            3 * self.gaussians_per_token if self.seed_conditioned else 0
        )
        self.output_norm = nn.LayerNorm(self.dim)
        # A trunk absorbs the conditioned input and hands every attribute the
        # same dim-wide hidden activation, exactly as the router's does.
        self.trunk = (
            nn.Sequential(nn.Linear(self.dim + self.seed_dim, self.dim), nn.SiLU())
            if bool(trunk) else None
        )
        head_in = self.dim if self.trunk is not None else self.dim + self.seed_dim

        self.sizes = {
            name: int(getattr(gs_params, name))
            for name in ("shs", "opacity", "scaling", "rotation")
        }
        self.sizes["offset"] = int(getattr(gs_params, "offset", 0) or 0)
        if self.sizes["offset"] != 3:
            raise ValueError("Dynamic 2DGS requires p2g.gs_params.offset=3")
        self.heads = nn.ModuleDict({
            name: nn.Linear(head_in, width * self.gaussians_per_token)
            for name, width in self.sizes.items()
        })
        self._initialize_outputs(cfg)

    def _initialize_outputs(self, cfg):
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)
        opacity = float(_cfg_get(cfg, "initial_opacity", 0.2))
        opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(
            self.heads["opacity"].bias,
            math.log(opacity / (1.0 - opacity)),
        )
        scale = float(_cfg_get(cfg, "initial_scale_m", 0.3))
        raw_scale = math.log(math.expm1(max(scale, 1.0e-4)))
        nn.init.constant_(self.heads["scaling"].bias, raw_scale)
        if self.sizes["rotation"] != 4:
            raise ValueError("Dynamic 2DGS requires a 4D quaternion rotation")
        with torch.no_grad():
            # Every slot starts at the identity quaternion.
            self.heads["rotation"].bias[
                ::self.sizes["rotation"]
            ] = 1.0
        nn.init.zeros_(self.heads["offset"].weight)
        nn.init.zeros_(self.heads["offset"].bias)

    def _unpack(self, value, width):
        """(N, G*width) -> (N*G, width), token-major so slot order is stable."""
        if self.gaussians_per_token == 1:
            return value
        return value.reshape(value.shape[0] * self.gaussians_per_token, width)

    def _head_input(self, decoded, seed_delta):
        """Normalized feature, with every slot's raw metric seed delta appended.

        The deltas stay in metres rather than being normalized, which is how
        the router trunk consumes its own copy of the same quantity.
        """
        if not self.seed_conditioned:
            return decoded
        if seed_delta is None:
            raise ValueError(
                "a seed-conditioned Gaussian head requires seed deltas"
            )
        expected = (decoded.shape[0], self.gaussians_per_token, 3)
        if tuple(seed_delta.shape) != expected:
            raise ValueError(
                f"seed deltas must have shape {expected}, got "
                f"{tuple(seed_delta.shape)}"
            )
        return torch.cat([
            decoded,
            seed_delta.to(dtype=decoded.dtype).reshape(
                decoded.shape[0], self.seed_dim
            ),
        ], dim=-1)

    def _decode(self, feature, seed_delta=None):
        decoded = self.output_norm(feature)
        head_input = self._head_input(decoded, seed_delta)
        if self.trunk is not None:
            head_input = self.trunk(head_input)
        raw = {
            name: self._unpack(head(head_input), self.sizes[name])
            for name, head in self.heads.items()
        }
        offset = self.offset_bound * torch.tanh(raw.pop("offset"))
        # The third value stays the normalized feature, not the conditioned
        # input, so downstream consumers keep reading a dim-wide token.
        return raw, offset, decoded

    def forward(self, feature, seed_delta=None):
        return self._decode(feature, seed_delta)

class SeedConditionedGaussianAttributeHead(GaussianAttributeHead):
    """Legacy V1 decoder retained for its checkpoint contract."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__(cfg, gs_params, dim, offset_bound)
        self.cell_size_m = float(_cfg_get(cfg, "cell_size_m", 0.8))
        geom_dim = int(_cfg_get(cfg, "geometry_dim", 64))
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, geom_dim),
            nn.SiLU(),
            nn.Linear(geom_dim, geom_dim),
        )
        self.feature_norm = nn.LayerNorm(self.dim)
        self.refine = nn.Sequential(
            nn.Linear(self.dim + geom_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feature, seed_delta_ref):
        geom = self.geometry_encoder(seed_delta_ref / self.cell_size_m)
        decoded = feature + self.refine(torch.cat([
            self.feature_norm(feature), geom,
        ], dim=-1))
        return self._decode(decoded)

class MotionConditionedVelocityHead(nn.Module):
    """Legacy v1 direct-velocity decoder kept for checkpoint compatibility."""

    def __init__(self, cfg, dim: int, time_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.max_speed_mps = float(_cfg_get(cfg, "max_speed_mps", 25.0))
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        duration_dim = int(_cfg_get(cfg, "duration_embedding_dim", 32))
        position_frequencies = int(_cfg_get(cfg, "position_frequencies", 4))
        position_range_m = float(_cfg_get(cfg, "position_range_m", 110.0))
        self.position_encoder = FourierPositionEncoder3D(
            position_frequencies, position_range_m
        )
        self.duration_encoder = nn.Sequential(
            nn.Linear(3, duration_dim),
            nn.SiLU(),
            nn.Linear(duration_dim, duration_dim),
        )
        self.feature_norm = nn.LayerNorm(self.dim)
        self.delta_norm = nn.LayerNorm(self.dim)
        input_dim = (
            2 * self.dim + int(time_dim) + duration_dim
            + self.position_encoder.out_dim
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(input_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(
        self,
        refined_feature,
        temporal_delta,
        time_embedding,
        duration_sec,
        position_ref,
    ):
        duration = duration_sec.reshape(-1, 1).clamp_min(1.0e-4)
        duration_stat = torch.cat([
            duration - 1.0,
            duration.log(),
            duration.reciprocal() - 1.0,
        ], dim=-1)
        duration_embedding = self.duration_encoder(duration_stat)
        motion_feature = self.motion_refiner(torch.cat([
            self.feature_norm(refined_feature),
            self.delta_norm(temporal_delta),
            time_embedding,
            duration_embedding,
            self.position_encoder(position_ref),
        ], dim=-1))
        velocity = self.max_speed_mps * torch.tanh(self.velocity(motion_feature))
        return velocity

class PhysicalVelocityHead(nn.Module):
    """Predict an unbounded physical velocity term in metres/second.

    V3 deliberately consumes the exact same shared token feature as the Gaussian
    head and uses this term as the complete velocity. V4 uses it as an additive
    residual over attention-derived ``velocity_init``. The final Linear starts at
    exactly zero; unlike the historical heads, no position encoding, tanh, or
    hand-chosen speed cap changes the output semantics or saturates its gradients.
    """

    # Whether ``forward`` takes ``v_init`` as its second argument.
    reads_velocity_init = False


    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.feature_norm = nn.LayerNorm(self.dim)
        self.motion_refiner = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature):
        motion_feature = self.motion_refiner(
            self.feature_norm(refined_feature)
        )
        return self.velocity(motion_feature)

class InitConditionedVelocityHead(nn.Module):
    """Predict a V6 residual from the refined feature and proposal velocity.

    ``velocity_init`` is scaled into the normalized feature domain and
    concatenated after feature LayerNorm.  The conditioning copy is detached so
    the residual head cannot reshape the proposal through an indirect shortcut;
    the additive ``velocity_init`` path still carries the final-velocity
    gradient into the proposal itself.

    ``residual_hidden_dim`` picks the refiner width. Omitting it keeps the
    historical two ``dim``-wide layers so existing V6 checkpoints rebuild
    unchanged; a positive value builds one bottleneck layer instead, i.e.
    ``Linear(dim+3, h) -> SiLU -> Linear(h, 3)``. The offset is a 3-vector
    residual over a proposal that already carries the motion, so the wide
    variant spends ~9x the parameters on a target the regularizer holds near
    zero anyway.
    """

    # Whether ``forward`` takes ``v_init`` as its second argument.
    reads_velocity_init = True


    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.init_condition_scale_mps = float(
            _cfg_get(cfg, "init_condition_scale_mps", 10.0)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if self.init_condition_scale_mps <= 0.0:
            raise ValueError("motion.init_condition_scale_mps must be positive")

        hidden_dim = _cfg_get(cfg, "residual_hidden_dim", None)
        if hidden_dim is not None and int(hidden_dim) <= 0:
            raise ValueError(
                "motion.residual_hidden_dim must be positive when set"
            )
        self.residual_hidden_dim = (
            None if hidden_dim is None else int(hidden_dim)
        )

        self.feature_norm = nn.LayerNorm(self.dim)
        if self.residual_hidden_dim is None:
            self.motion_refiner = nn.Sequential(
                nn.Linear(self.dim + 3, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.dim),
                nn.SiLU(),
            )
            output_dim = self.dim
        else:
            self.motion_refiner = nn.Sequential(
                nn.Linear(self.dim + 3, self.residual_hidden_dim),
                nn.SiLU(),
            )
            output_dim = self.residual_hidden_dim
        self.velocity = nn.Linear(output_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature, velocity_init):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V6 velocity"
            )
        if velocity_init.shape != (refined_feature.shape[0], 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")

        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        normalized_feature = self.feature_norm(refined_feature)
        normalized_init = (
            condition / self.init_condition_scale_mps
        ).to(dtype=normalized_feature.dtype)
        motion_feature = self.motion_refiner(torch.cat([
            normalized_feature,
            normalized_init,
        ], dim=-1))
        return self.velocity(motion_feature)

class EmbeddedInitVelocityHead(nn.Module):
    """V8 offset from the refined token and a detached velocity initializer.

    The current estimate is embedded explicitly because the dense feature
    value path does not contain the metric coordinate expectation itself.  The
    conditioning copy is detached; the additive ``v_init + v_offset`` path
    outside this module remains the only offset-loss route into matching.
    """

    # Whether ``forward`` takes ``v_init`` as its second argument.
    reads_velocity_init = True


    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.velocity_embedding_dim = int(
            _cfg_get(cfg, "velocity_embedding_dim", 32)
        )
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if self.velocity_embedding_dim <= 0:
            raise ValueError("V8 velocity_embedding_dim must be positive")
        if self.residual_hidden_dim <= 0:
            raise ValueError("V8 residual_hidden_dim must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, self.velocity_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.velocity_embedding_dim,
                self.velocity_embedding_dim,
            ),
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(
                self.dim + self.velocity_embedding_dim,
                self.residual_hidden_dim,
            ),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature, velocity_init):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V8 velocity"
            )
        if velocity_init.shape != (refined_feature.shape[0], 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")
        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        feature = self.feature_norm(refined_feature)
        velocity_embedding = self.velocity_encoder(
            condition.to(dtype=feature.dtype)
        )
        motion_feature = self.motion_refiner(torch.cat([
            feature,
            velocity_embedding,
        ], dim=-1))
        return self.velocity(motion_feature)

class FeatureOnlyVelocityOffsetHead(nn.Module):
    """V7.2 compact offset from only the time-refined token feature.

    The complete path is ``LN(feature) -> Linear -> SiLU -> Linear(3)``:
    exactly one nonlinear activation and no explicit initializer or duration
    conditioning. V7.2 exposes time to the preceding cross-attention instead,
    while the detached initializer Q warp makes the refined feature depend on
    the current correspondence lookup. The final Linear is zero initialized,
    so a fresh model still starts exactly at ``v_final = v_init``.
    """

    # Whether ``forward`` takes ``v_init`` as its second argument.
    reads_velocity_init = False


    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        if self.dim <= 0 or self.residual_hidden_dim <= 0:
            raise ValueError("V7.2 feature-only offset dimensions must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.motion_refiner = nn.Sequential(
            nn.Linear(self.dim, self.residual_hidden_dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V7.2 velocity"
            )
        motion_feature = self.motion_refiner(
            self.feature_norm(refined_feature)
        )
        return self.velocity(motion_feature)

class EmbeddedInitDurationVelocityHead(nn.Module):
    """V7 additive offset head with explicit velocity and duration embeddings.

    The input contract is exactly ``LN(refined_feature)`` concatenated with
    ``MLP(v_init): 3 -> Dv -> SiLU -> Dv`` and
    ``MLP(Fourier4(duration)): 8 -> Dt -> SiLU -> Dt``. Duration Fourier
    features contain sin/cos only (no raw scalar), and physical seconds are
    kept raw when ``duration_reference_sec=1``. The conditioning copy of
    ``v_init`` is detached by default; its additive path outside this module
    still trains the proposal.
    """

    # Whether ``forward`` takes ``v_init`` as its second argument.
    reads_velocity_init = True


    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.velocity_embedding_dim = int(
            _cfg_get(cfg, "velocity_embedding_dim", 32)
        )
        self.duration_embedding_dim = int(
            _cfg_get(cfg, "duration_embedding_dim", 16)
        )
        self.duration_frequency_count = int(
            _cfg_get(cfg, "duration_frequencies", 4)
        )
        self.duration_reference_sec = float(
            _cfg_get(cfg, "duration_reference_sec", 1.0)
        )
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if min(
            self.velocity_embedding_dim,
            self.duration_embedding_dim,
            self.duration_frequency_count,
            self.residual_hidden_dim,
        ) <= 0:
            raise ValueError("V7 motion embedding dimensions must be positive")
        if self.duration_reference_sec <= 0.0:
            raise ValueError("motion.duration_reference_sec must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, self.velocity_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.velocity_embedding_dim,
                self.velocity_embedding_dim,
            ),
        )
        duration_input_dim = 2 * self.duration_frequency_count
        self.duration_encoder = nn.Sequential(
            nn.Linear(duration_input_dim, self.duration_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.duration_embedding_dim,
                self.duration_embedding_dim,
            ),
        )
        self.register_buffer(
            "duration_frequency_bands",
            math.pi * 2.0 ** torch.arange(
                self.duration_frequency_count, dtype=torch.float32
            ),
            persistent=False,
        )
        head_input_dim = (
            self.dim
            + self.velocity_embedding_dim
            + self.duration_embedding_dim
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(head_input_dim, self.residual_hidden_dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def _duration_fourier(self, duration_sec):
        normalized = (
            duration_sec.reshape(-1, 1).float()
            / self.duration_reference_sec
        )
        phase = normalized * self.duration_frequency_bands.reshape(1, -1)
        # Keep the stated 8D layout for four bands: all sin, then all cos.
        return torch.cat([phase.sin(), phase.cos()], dim=-1)

    def forward(self, refined_feature, velocity_init, duration_sec):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V7 velocity"
            )
        n_tokens = refined_feature.shape[0]
        if velocity_init.shape != (n_tokens, 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")
        if duration_sec.shape != (n_tokens,):
            raise ValueError("duration_sec must provide one scalar per token")
        if not bool(torch.all(
            torch.isfinite(duration_sec) & (duration_sec > 0.0)
        )):
            raise ValueError("V7 velocity head requires positive finite duration")

        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        feature = self.feature_norm(refined_feature)
        velocity_embedding = self.velocity_encoder(
            condition.to(dtype=feature.dtype)
        )
        duration_embedding = self.duration_encoder(
            self._duration_fourier(duration_sec).to(dtype=feature.dtype)
        )
        motion_feature = self.motion_refiner(torch.cat([
            feature,
            velocity_embedding,
            duration_embedding,
        ], dim=-1))
        return self.velocity(motion_feature)

__all__ = [
    "GaussianAttributeHead",
    "SeedConditionedGaussianAttributeHead",
    "MotionConditionedVelocityHead",
    "PhysicalVelocityHead",
    "InitConditionedVelocityHead",
    "EmbeddedInitVelocityHead",
    "FeatureOnlyVelocityOffsetHead",
    "EmbeddedInitDurationVelocityHead",
 ]
