"""Dynamic Gaussian backend composition."""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn

from ...utils import boxes as box_utils
from .attention_matching import ConsensusAttentionMotionMatcher
from .common import _cfg_get
from .heads import (
    EmbeddedInitDurationVelocityHead, EmbeddedInitVelocityHead,
    FeatureOnlyVelocityOffsetHead, GaussianAttributeHead,
    InitConditionedVelocityHead, MotionConditionedVelocityHead,
    PhysicalVelocityHead, SeedConditionedGaussianAttributeHead,
)
from .motion_proposals import (
    ProjectedDenseMotionProposal, SparseMotionProposal,
    StraightThroughTop4MotionProposal,
)
from .temporal import (
    GroupedGainBarrierCrossAttention,
    LayerWeightedDistanceBiasCrossAttention,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    ParallelBidirectionalCrossAttention, TimeConditionedParallelCrossAttention,
)

class DynamicGaussianBackend(nn.Module):
    """Legacy direct-velocity v1 backend."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1):
        super().__init__()
        self.cfg = cfg
        self.dim = int(dim)
        if int(gaussians_per_token) != 1:
            # This head refines the token feature with its seed's own offset
            # geometry, which is defined at token resolution only.
            raise ValueError(
                "the legacy V1 backend emits exactly one Gaussian per token"
            )
        self.gaussians_per_token = 1
        self.temporal = ParallelBidirectionalCrossAttention(cfg.temporal, self.dim)
        gaussian_cfg = _cfg_get(cfg, "gaussian_head", None)
        self.gaussian_head = SeedConditionedGaussianAttributeHead(
            gaussian_cfg, gs_params, self.dim, offset_bound
        )
        self.velocity_head = MotionConditionedVelocityHead(
            cfg.motion,
            self.dim,
            self.temporal.time_encoder.out_dim,
        )

    def _check_fixed_count_seeds(self, seed_sensor, token_position_sensor):
        """Normalize the source seeds to ``(N, G, 3)`` and check the count.

        The token builder always emits a padded slot axis; a backend that emits
        ``G`` Gaussians per token needs exactly ``G`` of those slots filled, and
        a mismatch here means the config's Gaussian count and the seed contract
        have drifted apart.
        """
        if token_position_sensor.ndim != 2 or token_position_sensor.shape[1] != 3:
            raise ValueError("dynamic token positions must be (N,3)")
        slots = self.gaussians_per_token
        if seed_sensor.ndim == 2:
            seed_sensor = seed_sensor.unsqueeze(1)
        expected = (token_position_sensor.shape[0], slots, 3)
        if tuple(seed_sensor.shape) != expected:
            raise ValueError(
                f"Dynamic 2DGS expects {slots} seed(s) per token with shape "
                f"{expected}, got {tuple(seed_sensor.shape)}"
            )
        return seed_sensor

    def _expand_to_gaussians(self, value):
        """Repeat one per-token row for each of that token's Gaussians."""
        if self.gaussians_per_token == 1:
            return value
        return value.repeat_interleave(self.gaussians_per_token, dim=0)

    @staticmethod
    def _frame_bounds(token_offset, frame):
        start = int(token_offset[frame - 1]) if frame > 0 else 0
        return start, int(token_offset[frame])

    @staticmethod
    def _assemble_batch_gaussians(
        *,
        sample_count,
        batch_index,
        position,
        raw,
        velocity,
        time_sec,
        time_normalized,
        duration_sec,
        extra_fields=None,
    ):
        """Split the packed per-Gaussian tensors into one payload per sample.

        Every dynamic backend produces the same rendering contract regardless
        of how it obtained position, attributes, and velocity, so the split is
        shared. ``extra_fields`` carries whatever the variant additionally
        exposes -- motion diagnostics, and for the adaptive-K router the token
        each Gaussian came from. ``is_dynamic`` is a diagnostics field: nothing
        here performs bbox assignment or gates rendering.
        """
        extra_fields = extra_fields or {}
        batch_gaussians = []
        for batch_id in range(sample_count):
            rows = (batch_index == batch_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                batch_gaussians.append(None)
                continue
            n = int(rows.numel())
            motion_mask = torch.ones(n, dtype=torch.bool, device=rows.device)
            batch_gaussians.append({
                "position": position[rows],
                "coord": position[rows],
                "coord_ref": position[rows],
                "shs": raw["shs"][rows],
                "opacity": raw["opacity"][rows],
                "scaling": raw["scaling"][rows],
                "rotation": raw["rotation"][rows],
                "velocity": velocity[rows],
                **{name: value[rows] for name, value in extra_fields.items()},
                "source_time_sec": time_sec[rows],
                "source_time_normalized": time_normalized[rows],
                "window_duration_sec": duration_sec[rows],
                "is_dynamic": motion_mask,
                "instance_id": torch.full(
                    (n,), -1, dtype=torch.long, device=rows.device
                ),
                "bg_mask": ~motion_mask,
                "fg_masks": {},
                "frame_bboxes": [],
            })
        return batch_gaussians

    @staticmethod
    def _signed_pair_delta_t(geometry):
        """Forward-time travel interval for each token's source frame.

        Dynamic inputs contain exactly two endpoint frames per sample, and
        their physical timestamps are relative to the same window start. Local
        frame 0 therefore travels forward by ``+duration`` and frame 1 backward
        by ``-duration``, which is what turns both endpoints' displacements into
        one forward-time velocity convention.
        """
        local_frame = geometry["local_frame"]
        if not bool(torch.all((local_frame == 0) | (local_frame == 1))):
            raise ValueError(
                "dynamic pair velocity requires endpoint indices 0/1"
            )
        duration = geometry["duration_sec"]
        if not bool(torch.all(torch.isfinite(duration) & (duration > 0.0))):
            raise ValueError(
                "dynamic pair velocity requires a positive finite duration"
            )
        return torch.where(local_frame == 0, duration, -duration)

    def _reference_geometry_and_time(
        self,
        token_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        seed_delta_sensor=None,
    ):
        device, dtype = token_sensor.device, token_sensor.dtype
        token_ref = torch.empty_like(token_sensor)
        seed_ref = torch.empty_like(seed_sensor)
        seed_delta_ref = (
            torch.empty_like(seed_delta_sensor)
            if seed_delta_sensor is not None else None
        )
        if seed_delta_sensor is not None and seed_delta_sensor.shape != seed_sensor.shape:
            raise ValueError("seed_delta_sensor must have the same shape as seeds")
        token_time_norm = torch.empty(token_sensor.shape[0], device=device, dtype=dtype)
        token_time_sec = torch.empty_like(token_time_norm)
        token_duration = torch.empty_like(token_time_norm)
        token_batch = torch.empty(
            token_sensor.shape[0], device=device, dtype=torch.long
        )
        local_frame_index = torch.empty_like(token_batch)
        seen = defaultdict(int)
        for global_frame, batch_id_value in enumerate(frame_batch_idx.tolist()):
            batch_id = int(batch_id_value)
            local_frame = seen[batch_id]
            seen[batch_id] += 1
            start, end = self._frame_bounds(token_offset, global_frame)
            pose = pose_list[batch_id][local_frame].to(device=device, dtype=dtype)
            token_ref[start:end] = box_utils.apply_pose(
                token_sensor[start:end], pose
            )
            seed_frame = seed_sensor[start:end]
            seed_ref[start:end] = box_utils.apply_pose(
                seed_frame.reshape(-1, 3), pose
            ).reshape_as(seed_frame)
            if seed_delta_ref is not None:
                delta_frame = seed_delta_sensor[start:end]
                center_frame = seed_frame - delta_frame
                center_ref = box_utils.apply_pose(
                    center_frame.reshape(-1, 3), pose
                ).reshape_as(center_frame)
                seed_delta_ref[start:end] = seed_ref[start:end] - center_ref
            t_norm = timestamps_normalized[batch_id].to(device=device, dtype=dtype)
            t_sec = timestamps_sec[batch_id].to(device=device, dtype=dtype)
            if local_frame >= t_norm.numel() or local_frame >= t_sec.numel():
                raise ValueError("source timestamp count does not match input frames")
            token_time_norm[start:end] = t_norm[local_frame]
            token_time_sec[start:end] = t_sec[local_frame]
            duration = torch.as_tensor(
                window_duration_sec[batch_id], device=device, dtype=dtype
            )
            token_duration[start:end] = duration
            token_batch[start:end] = batch_id
            local_frame_index[start:end] = local_frame
        return {
            "token_ref": token_ref,
            "seed_ref": seed_ref,
            "seed_delta_ref": seed_delta_ref,
            "time_normalized": token_time_norm,
            "time_sec": token_time_sec,
            "duration_sec": token_duration,
            "batch": token_batch,
            "local_frame": local_frame_index,
        }

    def forward(
        self,
        fused_feature,
        token_position_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
    ):
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(
                f"dynamic fused_feature must be (N,{self.dim})"
            )
        seed_sensor = self._check_fixed_count_seeds(
            seed_sensor, token_position_sensor
        )

        geometry = self._reference_geometry_and_time(
            token_position_sensor,
            seed_sensor,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
        )
        seed_ref = geometry["seed_ref"].reshape(-1, 3)
        refined, temporal_delta, time_embedding = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["time_normalized"],
        )
        raw, position_offset, _gaussian_feature = self.gaussian_head(
            refined,
            seed_ref - geometry["token_ref"],
        )
        position = seed_ref + position_offset
        velocity = self.velocity_head(
            refined,
            temporal_delta,
            time_embedding,
            geometry["duration_sec"],
            position,
        )

        batch_gaussians = self._assemble_batch_gaussians(
            sample_count=len(pose_list),
            batch_index=geometry["batch"],
            position=position,
            raw=raw,
            velocity=velocity,
            time_sec=geometry["time_sec"],
            time_normalized=geometry["time_normalized"],
            duration_sec=geometry["duration_sec"],
        )

        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": geometry["batch"],
        }

class PhysicalVelocityGaussianBackend(DynamicGaussianBackend):
    """Current bbox-free backend with physical-time conditioning and direct m/s."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1, gaussian_count_cfg=None):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.dim = int(dim)
        self.gaussians_per_token = int(gaussians_per_token)
        self.temporal = self._build_temporal(cfg)
        self.time_reference_sec = float(
            _cfg_get(cfg.temporal, "time_reference_sec", 1.0)
        )
        if self.time_reference_sec <= 0.0:
            raise ValueError("time_reference_sec must be positive")
        self.adaptive_count = gaussian_count_cfg is not None
        self.gaussian_head = self._build_gaussian_head(
            cfg, gs_params, offset_bound, gaussian_count_cfg
        )
        self.velocity_head = self._build_velocity_head(cfg)

    def _build_temporal(self, cfg):
        """The cross-attention stack this variant refines its tokens with."""
        return TimeConditionedParallelCrossAttention(cfg.temporal, self.dim)

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        """The 2DGS attribute decoder, fixed-count unless a router is given.

        A backend that supports the learned-count router overrides this; every
        other one emits ``gaussians_per_token`` Gaussians from one head and
        rejects a router config it could not honour.
        """
        if gaussian_count_cfg is not None:
            raise ValueError(
                f"{type(self).__name__} emits a fixed number of Gaussians per "
                "token and cannot use p2g.grid_query.count_mode=learned_gumbel"
            )
        return GaussianAttributeHead(
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
            gaussians_per_token=self.gaussians_per_token,
        )

    def _build_velocity_head(self, cfg):
        """The module that turns a refined token into metres per second."""
        return PhysicalVelocityHead(cfg.motion, self.dim)

    def _temporal_coordinate(self, geometry):
        # A fixed reference unit preserves cadence: endpoints 0.8 s apart are
        # encoded as [0, 0.8], not collapsed to [0, 1].
        return geometry["time_sec"] / self.time_reference_sec

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
        )
        return refined, {}

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del geometry, position, temporal_fields
        return self.velocity_head(refined), {}

    def _predict_gaussian(self, refined, geometry):
        del geometry
        return self.gaussian_head(refined)

    def forward(
        self,
        fused_feature,
        token_position_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        motion_proposal_feature=None,
    ):
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        seed_sensor = self._check_fixed_count_seeds(
            seed_sensor, token_position_sensor
        )

        geometry = self._reference_geometry_and_time(
            token_position_sensor,
            seed_sensor,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
            motion_proposal_feature,
        )
        raw, position_offset, _gaussian_feature = self._predict_gaussian(
            refined, geometry
        )
        # Attributes and position are per Gaussian; correspondence, motion and
        # source timing are per token. With a fixed count per token the second
        # group is index-expanded, so a per-Gaussian mean over any of these
        # stays exactly the per-token mean.
        position = geometry["seed_ref"].reshape(-1, 3) + position_offset
        token_velocity, token_motion_fields = self._predict_motion(
            refined, geometry, position, temporal_fields
        )
        velocity = self._expand_to_gaussians(token_velocity)
        motion_fields = {
            name: self._expand_to_gaussians(value)
            for name, value in token_motion_fields.items()
        }

        batch_gaussians = self._assemble_batch_gaussians(
            sample_count=len(pose_list),
            batch_index=self._expand_to_gaussians(geometry["batch"]),
            position=position,
            raw=raw,
            velocity=velocity,
            time_sec=self._expand_to_gaussians(geometry["time_sec"]),
            time_normalized=self._expand_to_gaussians(
                geometry["time_normalized"]
            ),
            duration_sec=self._expand_to_gaussians(geometry["duration_sec"]),
            extra_fields=motion_fields,
        )
        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": self._expand_to_gaussians(geometry["batch"]),
        }

class AttentionInitializedVelocityGaussianBackend(PhysicalVelocityGaussianBackend):
    """V4 physical velocity with a four-head correspondence initialization.

    The selected heads remain part of the ordinary all-head feature update.
    Their final-layer Q/K probabilities are additionally applied to ref-frame
    token xyz to produce a source-to-opposite-frame displacement. Dividing by
    the signed source-to-opposite timestamp interval converts both endpoint
    directions to one forward-time velocity convention. The V3 velocity head
    is retained as a zero-initialized residual in metres/second.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1):
        super().__init__(
            cfg, gs_params, dim, offset_bound,
            gaussians_per_token=gaussians_per_token,
        )
        if self.temporal.motion_head_count <= 0:
            raise ValueError(
                "V4 attention velocity requires temporal.motion_head_count > 0"
            )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        refined, delta_p_init = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            return_motion_displacement=True,
        )
        return refined, {"delta_p_init": delta_p_init}

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        delta_p_init = temporal_fields["delta_p_init"]
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_residual = self.velocity_head(refined)
        velocity = velocity_init + velocity_residual
        return velocity, {
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_init": velocity_init,
            "velocity_residual": velocity_residual,
        }

class LayerMixtureMotionMixin:
    """The correspondence-to-velocity contract V10, V11 and V11.1 all share.

    The temporal module mixes one coordinate readout per layer and returns the
    displacement alongside the refined feature. Dividing that displacement by
    the signed endpoint interval turns both source frames into one forward-time
    ``v_init``, and the variant's own offset head corrects it. Only how the
    readout is *produced* differs between the three, and that lives entirely in
    the temporal module each one builds.
    """

    def _velocity_offset(self, refined, velocity_init):
        """The additive correction applied on top of ``v_init``.

        V10's offset head reads only the refined token; V11 and V11.1 embed the
        initializer they are correcting, because head 0's attended feature
        channels never carry the metric coordinate expectation itself. The head
        declares which it is, so no backend has to restate it.
        """
        if self.velocity_head.reads_velocity_init:
            return self.velocity_head(refined, velocity_init)
        return self.velocity_head(refined)

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        return self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            geometry["duration_sec"],
        )

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self._velocity_offset(refined, velocity_init)
        velocity = velocity_init + velocity_offset
        return velocity, {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "matched_position": temporal_fields["matched_position"],
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": velocity_init,
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
            "motion_layer_logits": temporal_fields["motion_layer_logits"],
            "motion_layer_weights": temporal_fields["motion_layer_weights"],
        }

class LayerWeightedAttentionVelocityGaussianBackend(
    LayerMixtureMotionMixin, PhysicalVelocityGaussianBackend
):
    """V10 all-layer initializer with config-selected Gaussian count.

    The temporal module returns the layer-weighted coordinate expectation.
    Dividing its signed displacement by the physical endpoint interval produces
    one token-level ``v_init``. The shipped config routes the normalized refined
    token through ``learned_gumbel`` and emits K={1,2,3} children; a legacy count
    config instead uses the fixed-count Gaussian head. Every child of a token
    shares its velocity, while the additive correction stays at token resolution.
    """

    def __init__(
        self,
        cfg,
        gs_params,
        dim: int,
        offset_bound: float,
        gaussian_count_cfg=None,
        *,
        gaussians_per_token: int = 1,
    ):
        """Preserve the historical fifth positional adaptive-count argument."""
        super().__init__(
            cfg,
            gs_params,
            dim,
            offset_bound,
            gaussians_per_token=gaussians_per_token,
            gaussian_count_cfg=gaussian_count_cfg,
        )

    def _build_gaussian_head(self, cfg, gs_params, offset_bound,
                             gaussian_count_cfg):
        """The learned-count router, or the shared fixed-count head.

        ``p2g.grid_query.count_mode`` decides. With ``learned_gumbel`` the
        refined token routes through the grid ``GridSlotHead`` and only the
        selected K-specific joint head runs; otherwise this backend behaves
        like every other fixed-count variant and the temporal stack is the only
        thing that distinguishes it.
        """
        if gaussian_count_cfg is None:
            return super()._build_gaussian_head(
                cfg, gs_params, offset_bound, None
            )
        from ..grid_query_head import GridSlotHead

        self.gaussian_output_norm = nn.LayerNorm(self.dim)
        head = GridSlotHead(gaussian_count_cfg, gs_params, dim=self.dim)
        if head.count_mode != "learned_gumbel":
            raise ValueError("adaptive-K Gaussian count must use learned_gumbel")
        if head.k_max != 3:
            raise ValueError("adaptive-K Gaussian count requires K_max=3")
        self.offset_bound = float(offset_bound)
        self.gaussian_param_names = [
            "shs", "opacity", "scaling", "rotation", "offset",
        ]
        self.gaussian_param_sizes = [
            int(getattr(gs_params, name)) for name in self.gaussian_param_names
        ]
        if self.gaussian_param_sizes[-2:] != [4, 3]:
            raise ValueError(
                "adaptive-K Gaussian head requires rotation=4 and offset=3"
            )
        if sum(self.gaussian_param_sizes) != head.param_dim:
            raise ValueError("adaptive-K Gaussian parameter widths disagree")
        self._initialize_adaptive_gaussian_heads(
            _cfg_get(cfg, "gaussian_head", None), head
        )
        return head

    def _build_temporal(self, cfg):
        """The temporal module whose head probabilities become ``v_init``."""
        return LayerWeightedDistanceBiasCrossAttention(cfg.temporal, self.dim)

    def _build_velocity_head(self, cfg):
        """The additive correction applied on top of ``v_init``."""
        return FeatureOnlyVelocityOffsetHead(cfg.motion, self.dim)

    def _initialize_adaptive_gaussian_heads(self, cfg, head=None):
        """Give every K expert the existing dynamic-head output priors."""
        head = self.gaussian_head if head is None else head
        opacity = float(_cfg_get(cfg, "initial_opacity", 0.2))
        opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
        opacity_bias = math.log(opacity / (1.0 - opacity))
        scale = float(_cfg_get(cfg, "initial_scale_m", 0.3))
        scale_bias = math.log(math.expm1(max(scale, 1.0e-4)))

        sh_size, opacity_size, scale_size, rotation_size, offset_size = (
            self.gaussian_param_sizes
        )
        opacity_start = sh_size
        scale_start = opacity_start + opacity_size
        rotation_start = scale_start + scale_size
        offset_start = rotation_start + rotation_size
        block_bias = head.k_heads[0].bias.new_zeros(head.param_dim)
        block_bias[opacity_start:scale_start] = opacity_bias
        block_bias[scale_start:rotation_start] = scale_bias
        block_bias[rotation_start] = 1.0

        for k, k_head in enumerate(head.k_heads, start=1):
            nn.init.normal_(k_head.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                k_head.bias.copy_(block_bias.repeat(k))
                for slot in range(k):
                    row_start = slot * head.param_dim + offset_start
                    row_end = row_start + offset_size
                    k_head.weight[row_start:row_end].zero_()

    def _predict_adaptive_gaussians(self, refined, geometry, token_offset):
        seed_ref = geometry["seed_ref"]
        expected_seed_shape = (
            refined.shape[0],
            self.gaussian_head.k_max,
            self.gaussian_head.k_max,
            3,
        )
        if tuple(seed_ref.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K seed bank must have shape "
                f"{expected_seed_shape}, got {tuple(seed_ref.shape)}"
            )
        gaussian_feature = self.gaussian_output_norm(refined)
        delta_ref = geometry.get("seed_delta_ref")
        if delta_ref is None or tuple(delta_ref.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires a transformed K-specific seed-delta bank"
            )
        raw_params, packing = self.gaussian_head(
            gaussian_feature,
            None,
            delta_ref,
            token_offset,
        )
        selected_index = packing["anchor_k"] - 1
        token_rows = torch.arange(
            refined.shape[0], device=refined.device
        )
        selected_seed = seed_ref[token_rows, selected_index]
        anchor_index = packing["anchor_index"]
        slot_index = packing["slot_index"]
        packed_seed = selected_seed[anchor_index, slot_index]

        parts = torch.split(raw_params, self.gaussian_param_sizes, dim=-1)
        raw = dict(zip(self.gaussian_param_names, parts))
        position = packed_seed + self.offset_bound * torch.tanh(
            raw.pop("offset")
        )
        gradient_weight = self.gaussian_head.gradient_weight(
            packing["slot_k"], position.dtype
        )
        if (
            self.gaussian_head.grad_balance == "sqrt_k"
            and self.gaussian_head.grad_balance_scope == "output"
        ):
            from ..gaussian_assembly import gradient_scale_identity

            position = gradient_scale_identity(position, gradient_weight)
            raw = {
                name: gradient_scale_identity(value, gradient_weight)
                for name, value in raw.items()
            }
        return raw, position, packing, gradient_weight

    def forward(
        self,
        fused_feature,
        token_position_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        motion_proposal_feature=None,
        seed_delta_sensor=None,
    ):
        if not self.adaptive_count:
            # Without a router this backend is an ordinary fixed-count variant
            # whose temporal stack happens to mix layers; the shared forward
            # already implements that.
            if seed_delta_sensor is not None:
                raise ValueError(
                    "a fixed Gaussian count needs no K-specific seed-delta bank"
                )
            return super().forward(
                fused_feature,
                token_position_sensor,
                seed_sensor,
                token_offset,
                frame_batch_idx,
                pose_list,
                timestamps_normalized,
                timestamps_sec,
                window_duration_sec,
                motion_proposal_feature=motion_proposal_feature,
            )
        del motion_proposal_feature
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        expected_seed_shape = (
            fused_feature.shape[0],
            self.gaussian_head.k_max,
            self.gaussian_head.k_max,
            3,
        )
        if tuple(seed_sensor.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires the learned-count seed bank with shape "
                f"{expected_seed_shape}, got {tuple(seed_sensor.shape)}"
            )
        if seed_delta_sensor is None or tuple(
            seed_delta_sensor.shape
        ) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires a learned-count seed-delta bank with shape "
                f"{expected_seed_shape}"
            )
        if token_position_sensor.shape != (fused_feature.shape[0], 3):
            raise ValueError("adaptive-K token positions must align with features")

        geometry = self._reference_geometry_and_time(
            token_position_sensor,
            seed_sensor,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
            seed_delta_sensor=seed_delta_sensor,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
        )
        raw, position, packing, gradient_weight = (
            self._predict_adaptive_gaussians(
                refined, geometry, token_offset
            )
        )

        token_velocity, token_motion_fields = self._predict_motion(
            refined, geometry, None, temporal_fields
        )
        anchor_index = packing["anchor_index"]
        velocity = token_velocity[anchor_index]
        motion_fields = {
            name: value[anchor_index]
            for name, value in token_motion_fields.items()
        }
        if (
            self.gaussian_head.grad_balance == "sqrt_k"
            and self.gaussian_head.grad_balance_scope == "output"
        ):
            from ..gaussian_assembly import gradient_scale_identity

            velocity = gradient_scale_identity(velocity, gradient_weight)

        packed_batch = geometry["batch"][anchor_index]
        packed_time_sec = geometry["time_sec"][anchor_index]
        packed_time_normalized = geometry["time_normalized"][anchor_index]
        packed_duration = geometry["duration_sec"][anchor_index]
        packed_selected_k = packing["anchor_k"][anchor_index]

        batch_gaussians = self._assemble_batch_gaussians(
            sample_count=len(pose_list),
            batch_index=packed_batch,
            position=position,
            raw=raw,
            velocity=velocity,
            time_sec=packed_time_sec,
            time_normalized=packed_time_normalized,
            duration_sec=packed_duration,
            extra_fields={
                **motion_fields,
                "source_token_index": anchor_index,
                "selected_k": packed_selected_k,
                "gaussian_slot_index": packing["slot_index"],
            },
        )

        routing_stats = {
            "k_logits": packing["k_logits"].detach(),
            "selected_k": packing["anchor_k"].detach(),
            "token_position_sensor": token_position_sensor.detach(),
            "is_dynamic": torch.ones(
                fused_feature.shape[0],
                dtype=torch.bool,
                device=fused_feature.device,
            ),
        }
        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": packed_batch,
            "routing_stats": routing_stats,
        }

class MaxSpeedBarrierVelocityGaussianBackend(
    LayerWeightedAttentionVelocityGaussianBackend
):
    """V11: V10's count-selectable backend with split position encoding.

    The shipped config uses V10's ``learned_gumbel`` K={1,2,3} router, while a
    legacy count config selects the fixed-count head. Both keep one token velocity
    shared by its Gaussian children. Correspondence comes from head 0 of each
    layer under the max-speed soft barrier, and plain learned scalars mix the
    layer readouts.
    """

    def _build_temporal(self, cfg):
        return MaxSpeedBarrierLayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )

    def _build_velocity_head(self, cfg):
        """V8's head: ``[LN(f'), MLP(v_init; 3->32->32)] -> 96 -> 3``.

        The refined feature carries head 0's attended *feature* channels, never
        the metric coordinate expectation itself, so the offset head cannot see
        what it is correcting unless the initializer is embedded explicitly.
        """
        return EmbeddedInitVelocityHead(cfg.motion, self.dim)

class SingleGaussianBarrierVelocityGaussianBackend(
    LayerMixtureMotionMixin, PhysicalVelocityGaussianBackend
):
    """V11.1: V11's temporal stack with a fixed Gaussian count.

    The shipped config emits one medoid-seeded Gaussian per occupied token, and
    ``p2g.grid_query.K_max`` can select another fixed K. Repeated token fields are
    index-expanded by the same K, preserving token-mean motion diagnostics.

    1. **The QK-Norm gain is split per head group** (see
       :class:`GroupedGainBarrierCrossAttention`).

    2. **The adaptive-K router is absent.** The plain
       :class:`GaussianAttributeHead` owns one parameter block per configured
       slot, with no count predictor, opacity gate, or per-K gradient balancing.
    """

    def _build_temporal(self, cfg):
        return GroupedGainBarrierCrossAttention(cfg.temporal, self.dim)

    def _build_velocity_head(self, cfg):
        """V8's head: ``[LN(f'), MLP(v_init; 3->32->32)] -> 96 -> 3``.

        The refined feature carries head 0's attended *feature* channels, never
        the metric coordinate expectation itself, so the offset head cannot see
        what it is correcting unless the initializer is embedded explicitly.
        """
        return EmbeddedInitVelocityHead(cfg.motion, self.dim)

class ConsensusAttentionVelocityGaussianBackend(
    AttentionInitializedVelocityGaussianBackend
):
    """V8 final-attention consensus matching plus an additive offset.

    Heads ``[0, motion_head_count)`` remain ordinary dense feature-attention
    heads.  After the final layer has computed that dense update, their Q/K
    probabilities are averaged and one shared K->M reciprocal support is used
    only to read opposite-frame coordinates.  There is no independent LoRA
    descriptor branch, distance bias, dustbin, or query-coordinate warp.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1):
        super().__init__(
            cfg, gs_params, dim, offset_bound,
            gaussians_per_token=gaussians_per_token,
        )
        self.motion_matcher = ConsensusAttentionMotionMatcher(
            cfg.motion_matching
        )
        self.velocity_head = EmbeddedInitVelocityHead(
            cfg.motion, self.dim
        )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        refined, matching_fields = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            return_motion_displacement=True,
            motion_matcher=self.motion_matcher,
        )
        return refined, matching_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self.velocity_head(
            refined,
            velocity_init,
        )
        velocity = velocity_init + velocity_offset
        fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": (
                temporal_fields["delta_p_match"]
                / pair_delta_t_sec.unsqueeze(-1)
            ),
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        for name in (
            "matched_position",
            "motion_top1_probability",
            "motion_effective_support",
            "motion_reciprocal_probability",
            "motion_selected_forward_probability",
            "motion_consensus_entropy",
            "motion_head_js_divergence",
            "motion_candidate_probability_mass",
            "motion_selection_log_margin",
            "motion_candidate_support",
            "motion_match_support",
        ):
            fields[name] = temporal_fields[name]
        return velocity, fields

class ProposalInitializedVelocityGaussianBackend(PhysicalVelocityGaussianBackend):
    """V6 velocity from a sparse proposal plus an init-aware residual.

    Correspondence is computed from the same Utonia tokens used by the temporal
    trunk and before the temporal module adds time. The ordinary
    temporal stack refines attributes and supplies the feature for a velocity
    residual conditioned on the dustbin-aware proposal velocity. Dustbin
    probability remains a soft proposal-confidence gate; it no longer gates the
    residual itself.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 proposal_dim: int | None = None,
                 gaussians_per_token: int = 1):
        super().__init__(
            cfg, gs_params, dim, offset_bound,
            gaussians_per_token=gaussians_per_token,
        )
        if self.temporal.motion_head_count != 0:
            raise ValueError(
                "V6 motion proposal is independent; temporal.motion_head_count "
                "must be omitted or zero"
            )
        self.motion_proposal = SparseMotionProposal(
            cfg.motion_proposal,
            self.dim if proposal_dim is None else int(proposal_dim),
        )
        # Old V6 checkpoints do not carry this flag and retain their original
        # feature-only head and dustbin-gated fallback exactly. Fresh configs
        # opt into the init-conditioned additive residual explicitly.
        self.init_conditioned_residual = bool(
            _cfg_get(cfg.motion, "init_conditioned_residual", False)
        )
        if self.init_conditioned_residual:
            self.velocity_head = InitConditionedVelocityHead(
                cfg.motion, self.dim
            )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        if motion_proposal_feature is None:
            raise ValueError("V6 requires Utonia motion proposal features")
        # This call is intentionally before self.temporal: descriptors never see
        # endpoint time embedding or cross-frame
        # feature attention.
        proposal_fields = self.motion_proposal(
            motion_proposal_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["duration_sec"],
        )
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
        )
        return refined, proposal_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_match = temporal_fields["delta_p_match"]
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_match = delta_p_match / pair_delta_t_sec.unsqueeze(-1)
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        if self.init_conditioned_residual:
            velocity_offset = self.velocity_head(refined, velocity_init)
            velocity = velocity_init + velocity_offset
        else:
            # Checkpoint compatibility for the original V6 formulation.
            velocity_offset = self.velocity_head(refined)
            velocity = (
                velocity_init
                + temporal_fields["p_unmatched"].unsqueeze(-1) * velocity_offset
            )

        fields = {
            "delta_p_match": delta_p_match,
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": velocity_match,
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        fields.update({
            name: temporal_fields[name]
            for name in (
                "matched_position",
                "motion_top1_probability",
                "motion_effective_support",
                "motion_reciprocal_probability",
                "motion_dustbin_similarity",
                "motion_search_speed_mps",
                "match_probability",
                "p_unmatched",
            )
        })
        return velocity, fields

class WarpedProposalVelocityGaussianBackend(
    ProposalInitializedVelocityGaussianBackend
):
    """V7 proposal-warped attention and duration-aware velocity offset.

    The hard-gated initializer moves only the RoPE query coordinate to
    ``x_query + stopgrad(v_init) * signed_dt``. Keys stay at their observed
    reference-frame coordinates. There is no multiplicative confidence gate on
    cross-attention and V7 constructs no temporal time embedding. A confidently
    unmatched token therefore keeps its query at the observed position.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 proposal_dim: int | None = None,
                 gaussians_per_token: int = 1):
        super().__init__(
            cfg,
            gs_params,
            dim,
            offset_bound,
            proposal_dim=proposal_dim,
            gaussians_per_token=gaussians_per_token,
        )
        if self.temporal.use_time_embedding:
            raise ValueError("V7 temporal attention must disable time embedding")
        if self.motion_proposal.descriptor_mode != "direct_l2":
            raise ValueError("V7 requires direct_l2 motion descriptors")
        if self.motion_proposal.dustbin_mode != "evidence_mlp":
            raise ValueError("V7 requires evidence_mlp unmatched prediction")
        self.velocity_head = EmbeddedInitDurationVelocityHead(
            cfg.motion, self.dim
        )

    def _temporal_attention_time(self, geometry):
        # Historical V7/V7.1 intentionally keep temporal feature conditioning
        # disabled. V7.2 overrides this hook with raw relative seconds.
        return None

    def _velocity_offset(self, refined, velocity_init, geometry):
        return self.velocity_head(
            refined,
            velocity_init,
            geometry["duration_sec"],
        )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        if motion_proposal_feature is None:
            raise ValueError("V7 requires Utonia motion proposal features")
        proposal_fields = self.motion_proposal(
            motion_proposal_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["duration_sec"],
        )
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        velocity_match = (
            proposal_fields["delta_p_match"]
            / pair_delta_t_sec.unsqueeze(-1)
        )
        velocity_init = (
            proposal_fields["delta_p_init"]
            / pair_delta_t_sec.unsqueeze(-1)
        )
        # V7/V7.1 use their binary dustbin gate here; V7.2 supplies its exact
        # normalized Top-4 forward displacement. Detaching keeps attention loss
        # from reshaping the proposal through its own lookup coordinate, while
        # final-velocity loss still trains the proposal through the STE below.
        query_position_ref = (
            geometry["token_ref"]
            + velocity_init.detach() * pair_delta_t_sec.unsqueeze(-1)
        )
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_attention_time(geometry),
            query_position_ref=query_position_ref,
        )
        proposal_fields["pair_delta_t_sec"] = pair_delta_t_sec
        proposal_fields["velocity_match"] = velocity_match
        return refined, proposal_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = temporal_fields["pair_delta_t_sec"]
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self._velocity_offset(
            refined, velocity_init, geometry
        )
        velocity = velocity_init + velocity_offset

        fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": temporal_fields["velocity_match"],
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        for name in (
            "matched_position",
            "motion_top1_probability",
            "motion_effective_support",
            "motion_reciprocal_probability",
            "motion_candidate_entropy",
            "motion_best_candidate_displacement_m",
            "motion_reciprocal_probability_sum",
            "motion_mean_displacement_m",
            "motion_candidate_spread_m",
            "motion_normalized_mean_displacement",
            "motion_normalized_candidate_spread",
            "motion_dustbin_logit",
            "motion_soft_match_probability",
            "motion_hard_reject",
            "motion_search_speed_mps",
            "motion_candidate_probability_mass",
            "motion_selection_log_margin",
            "motion_hard_soft_displacement_cosine",
            "motion_hard_soft_displacement_norm_ratio",
            "motion_selected_distance_prior_penalty",
            "motion_match_support",
            "match_probability",
            "p_unmatched",
        ):
            if name in temporal_fields:
                fields[name] = temporal_fields[name]
        return velocity, fields

class StraightThroughProposalVelocityGaussianBackend(
    WarpedProposalVelocityGaussianBackend
):
    """V7.2: direct Top-4 STE, time-aware Q warp, and feature-only offset.

    The matcher remains time-free and consumes raw LoRA-adapted Utonia tokens.
    Temporal refinement then adds the historical V5 endpoint-time embedding
    before all four cross-attention layers while retaining the detached hard
    initializer Q-coordinate warp. The additive offset sees only the refined
    feature. Correspondence uses direct Utonia cosine, one fixed distance
    envelope, no reciprocal or unmatched terms, and a full-key softmax backward
    surrogate for the normalized Top-4 forward.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 proposal_dim: int | None = None,
                 gaussians_per_token: int = 1):
        # Skip the V6/V7 SparseMotionProposal constructor and its dustbin/search
        # parameters while retaining the common physical Gaussian trunk.
        PhysicalVelocityGaussianBackend.__init__(
            self, cfg, gs_params, dim, offset_bound,
            gaussians_per_token=gaussians_per_token,
        )
        if self.temporal.motion_head_count != 0:
            raise ValueError(
                "V7.2 motion proposal is independent; "
                "temporal.motion_head_count must be omitted or zero"
            )
        if not self.temporal.use_time_embedding:
            raise ValueError("V7.2 temporal attention must enable time embedding")
        self.motion_proposal = StraightThroughTop4MotionProposal(
            cfg.motion_proposal,
            self.dim if proposal_dim is None else int(proposal_dim),
        )
        self.velocity_head = FeatureOnlyVelocityOffsetHead(
            cfg.motion, self.dim
        )

    def _temporal_attention_time(self, geometry):
        return self._temporal_coordinate(geometry)

    def _velocity_offset(self, refined, velocity_init, geometry):
        # Q-warped, time-conditioned attention has already made the refined
        # feature depend on correspondence and cadence. Keep the final residual
        # decoder feature-only; v_init retains its separate additive path.
        del velocity_init, geometry
        return self.velocity_head(refined)

class PostAttentionProposalVelocityGaussianBackend(
    PhysicalVelocityGaussianBackend
):
    """V9: correspondence after temporal refinement, dense and offset-free.

    The temporal stack is the plain V3/V5 form -- endpoint time added once as
    ``f = f + E(t)``, twelve bidirectional cross-attention layers, and no
    query-coordinate warp, motion head, or matcher hook.  Its output ``f'``
    then feeds two independent consumers: the dense projected-descriptor
    proposal that produces ``v_init``, and the feature-only offset head that
    produces ``v_offset``.  Final motion is ``v_init + v_offset``.

    This is the structural difference from V6/V7/V7.2, whose proposals run
    *before* attention on raw LoRA Utonia tokens and therefore need a detached
    warp to feed correspondence back into the trunk.  Here the trunk feeds the
    matcher directly, so correspondence gradient reaches the same twelve layers
    that render the Gaussians and no detached path is required.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 gaussians_per_token: int = 1):
        super().__init__(
            cfg, gs_params, dim, offset_bound,
            gaussians_per_token=gaussians_per_token,
        )
        if self.temporal.motion_head_count != 0:
            raise ValueError(
                "V9 motion proposal is a separate descriptor branch; "
                "temporal.motion_head_count must be omitted or zero"
            )
        if not self.temporal.use_time_embedding:
            raise ValueError("V9 temporal attention must enable time embedding")
        self.motion_proposal = ProjectedDenseMotionProposal(
            cfg.motion_proposal, self.dim
        )
        self.velocity_head = FeatureOnlyVelocityOffsetHead(
            cfg.motion, self.dim
        )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        # V9 matches on the refined feature, so the raw pre-attention Utonia
        # tokens V6/V7 need are deliberately unused here.
        del motion_proposal_feature
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
        )
        proposal_fields = self.motion_proposal(
            refined,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["duration_sec"],
        )
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        proposal_fields["pair_delta_t_sec"] = pair_delta_t_sec
        proposal_fields["velocity_match"] = (
            proposal_fields["delta_p_match"] / pair_delta_t_sec.unsqueeze(-1)
        )
        return refined, proposal_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del geometry, position
        pair_delta_t_sec = temporal_fields["pair_delta_t_sec"]
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self.velocity_head(refined)
        velocity = velocity_init + velocity_offset

        fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": temporal_fields["velocity_match"],
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        for name in (
            "matched_position",
            "motion_top1_probability",
            "motion_effective_support",
            "motion_candidate_probability_mass",
            "motion_selection_log_margin",
            "motion_hard_soft_displacement_cosine",
            "motion_hard_soft_displacement_norm_ratio",
            "motion_selected_distance_prior_penalty",
        ):
            if name in temporal_fields:
                fields[name] = temporal_fields[name]
        return velocity, fields

__all__ = [
    "DynamicGaussianBackend",
    "PhysicalVelocityGaussianBackend",
    "AttentionInitializedVelocityGaussianBackend",
    "LayerMixtureMotionMixin",
    "LayerWeightedAttentionVelocityGaussianBackend",
    "MaxSpeedBarrierVelocityGaussianBackend",
    "SingleGaussianBarrierVelocityGaussianBackend",
    "ConsensusAttentionVelocityGaussianBackend",
    "ProposalInitializedVelocityGaussianBackend",
    "WarpedProposalVelocityGaussianBackend",
    "StraightThroughProposalVelocityGaussianBackend",
    "PostAttentionProposalVelocityGaussianBackend",
 ]
