from __future__ import annotations

import os

import torch
import torch.nn as nn

from .. import utonia
from ..utonia.lora import (
    adapt_utonia_embedding_to_xyzi,
    build_xyzi_utonia_input,
    freeze_utonia_and_enable_active_lora,
    inject_utonia_lora,
    resolve_utonia_lora_settings,
    set_active_lora_mode,
)
from .anchor_modes import (
    build_grid_gaussian_seeds,
    build_spherical_gaussian_seeds,
)
from .builders import build_token_builder
from .builders.common import GridSeedConfig, UtoniaGridMapper
from .feature_fusion import build_feature_fusion, fuse_features
from .gaussian_assembly import (
    assemble_batch_gaussians,
    gradient_scale_identity,
    refresh_coord_ref_after_offset,
)


def _uses_learned_count(cfg) -> bool:
    """Whether ``p2g.grid_query`` asks the router to predict the count."""
    grid_query = getattr(cfg, "grid_query", None)
    if grid_query is None:
        return False
    count_mode = str(getattr(grid_query, "count_mode", "legacy")).lower()
    return count_mode in GridSeedConfig.LEARNED_COUNT_MODES


class Point2Gaus(nn.Module):
    """Point to Gaussian."""

    def __init__(self, cfg, dynamic_cfg=None, dynamic_variant=None):
        super().__init__()
        self.cfg = cfg
        self.dynamic_cfg = dynamic_cfg
        self.dynamic_variant = dynamic_variant

        self.freeze_utonia = bool(getattr(cfg, "freeze_utonia", True))
        self.utonia_lora_settings = resolve_utonia_lora_settings(cfg)
        self.utonia_lora_enabled = self.utonia_lora_settings.enabled
        if self.utonia_lora_enabled and not self.freeze_utonia:
            raise ValueError(
                "p2g.utonia_lora.enable=true requires p2g.freeze_utonia=true; "
                "LoRA fine-tunes adapters while the pretrained Utonia base "
                "stays frozen"
            )
        # Load the frozen Utonia encoder from a repo-local checkpoint so the weights
        # and their embedded architecture config live under the project (not ~/.cache).
        # The architecture config is also dumped to config/utonia_pretrained.yaml for
        # inspection. Falls back to a HuggingFace download if the local file is absent.
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        _default_ckpt = os.path.join(_repo_root, "checkpoints", "utonia", "utonia.pth")
        utonia_ckpt = getattr(cfg, "utonia_ckpt", None) or _default_ckpt
        utonia_name = utonia_ckpt if os.path.isfile(utonia_ckpt) else "utonia"
        self.feature_extractor = utonia.load(
            utonia_name,
            repo_id="Pointcept/Utonia",
            custom_config={"freeze_encoder": True} if self.freeze_utonia else None,
        )
        self.utonia_coord_scale = 0.2
        self.utonia_input_grid_size = 0.01
        self.utonia_feature_stage = int(
            getattr(cfg, "utonia_feature_stage", len(self.feature_extractor.enc) - 1)
        )
        self._validate_utonia_feature_stage()
        self.utonia_stride_factor = self._infer_utonia_stride_factor()
        self.utonia_feature_grid_size = self._infer_utonia_feature_grid_size()
        self.utonia_feature_dim = self._infer_utonia_feature_dim()

        if self.utonia_lora_enabled:
            # Load the untouched 9D checkpoint before migrating its functional
            # [xyz, 0, ..., 0] stem to [xyz, intensity]. Both the new intensity
            # base column and every LoRA up projection start at zero.
            old_input_channels = adapt_utonia_embedding_to_xyzi(
                self.feature_extractor
            )
        self.utonia_lora_report = inject_utonia_lora(
            self.feature_extractor, self.utonia_lora_settings
        )
        self.utonia_lora_trainable_parameters = 0
        if self.utonia_lora_enabled:
            self.utonia_lora_trainable_parameters = (
                freeze_utonia_and_enable_active_lora(
                    self.feature_extractor, self._active_utonia_lora_roots()
                )
            )
            print(
                "Utonia LoRA: "
                f"input={old_input_channels}->4 (xyzi), "
                f"linear={self.utonia_lora_report.linear_modules} "
                f"rank={self.utonia_lora_settings.linear_rank}, "
                f"subm_conv3d={self.utonia_lora_report.conv_modules} "
                f"rank={self.utonia_lora_settings.conv_rank}, "
                "active_trainable="
                f"{self.utonia_lora_trainable_parameters / 1e6:.2f}M"
            )
        if self.freeze_utonia:
            self.feature_extractor.eval()
            if self.utonia_lora_enabled:
                set_active_lora_mode(
                    self._active_utonia_lora_roots(), self.training
                )

        # Both anchor modes start from the exact same occupied Utonia-grid tokens.
        self.grid_mapper = UtoniaGridMapper(
            self.utonia_coord_scale, self.utonia_input_grid_size, self.utonia_stride_factor
        )
        # p2g.grid_query decides how many Gaussians a token becomes: absent or
        # count_mode=legacy gives every token the same K_max (default one),
        # while a learned count_mode hands the decision to the router. Nothing
        # about that choice is pinned to the variant name.
        self.dynamic_adaptive_count = (
            self.dynamic_cfg is not None and _uses_learned_count(cfg)
        )
        self.anchor_mode, self.anchor_builder = build_token_builder(
            cfg,
            fixed_count_seeds=(
                self.dynamic_cfg is not None
                and not self.dynamic_adaptive_count
            ),
        )
        self.dynamic_gaussians_per_token = (
            self.anchor_builder.grid_seed_config.gaussians_per_token
            if self.dynamic_cfg is not None and not self.dynamic_adaptive_count
            else 1
        )

        self.agg_mlp = cfg.agg_mlp
        # Intensity width is part of the shared token-builder contract.
        intensity_out_dim = getattr(self.anchor_builder, "intensity_out_dim", None)
        if intensity_out_dim is None:
            raise RuntimeError("the shared token builder must expose intensity_out_dim")
        if self.utonia_lora_enabled:
            if int(intensity_out_dim) != 0:
                raise RuntimeError(
                    "Utonia XYZI LoRA must not construct a separate intensity "
                    "encoder"
                )
            self.agg_in_dim = self.utonia_feature_dim
            self.utonia_adapter = None
            self.intensity_agg_mlp = None
            self.joint_refiner = None
            self.trunk_dim = self.utonia_feature_dim
        else:
            self.agg_in_dim = self.utonia_feature_dim + int(intensity_out_dim)
            (
                self.utonia_adapter,
                self.intensity_agg_mlp,
                self.joint_refiner,
            ) = build_feature_fusion(
                cfg,
                utonia_dim=self.utonia_feature_dim,
                intensity_dim=int(intensity_out_dim),
                coord_scale=self.utonia_coord_scale,
            )
            self.trunk_dim = int(self.agg_mlp.out_dim)

        # Gaussian head size is derived purely from gs_params (shs+opacity+scaling+
        # rotation [+offset]). offset>0 adds the bounded position-offset output.
        self.offset_size = int(getattr(cfg.gs_params, "offset", 0) or 0)
        self.use_offset = self.offset_size > 0
        self.offset_bound = float(getattr(cfg, "head_offset_bound", 0.8))
        self.gs_param_sizes = [
            int(cfg.gs_params.shs),
            int(cfg.gs_params.opacity),
            int(cfg.gs_params.scaling),
            int(cfg.gs_params.rotation),
        ] + ([self.offset_size] if self.use_offset else [])
        gs_out_dim = sum(self.gs_param_sizes)
        trunk_dim = self.trunk_dim

        # Dynamic 2DGS shares occupied-grid construction and then registers only
        # its own temporal/GS/motion backend. In LoRA mode its input is the direct
        # 432D Utonia XYZI feature; the legacy intensity fusion and local refiner
        # modules are not constructed.
        if self.dynamic_cfg is not None:
            if self.anchor_mode != "grid":
                raise ValueError("Dynamic 2DGS requires p2g.anchor_mode='grid'")
            from src.config_loader import (
                DYNAMIC_VARIANT_V1,
                DYNAMIC_VARIANT_V3,
                DYNAMIC_VARIANT_V3_1,
                DYNAMIC_VARIANT_V4,
                DYNAMIC_VARIANT_V5,
                DYNAMIC_VARIANT_V6,
                DYNAMIC_VARIANT_V7,
                DYNAMIC_VARIANT_V7_1,
                DYNAMIC_VARIANT_V7_2,
                DYNAMIC_VARIANT_V8,
                DYNAMIC_VARIANT_V9,
                DYNAMIC_VARIANT_V10,
                DYNAMIC_VARIANT_V11,
                DYNAMIC_VARIANT_V11_1,
                DYNAMIC_VARIANT_V11_2,
                DYNAMIC_VARIANT_V11_3,
                DYNAMIC_VARIANT_V11_4,
            )
            from .dynamic_gaussian import (
                AttentionInitializedVelocityGaussianBackend,
                ConsensusAttentionVelocityGaussianBackend,
                DynamicGaussianBackend,
                FinalFeatureBarrierVelocityGaussianBackend,
                LayerWeightedAttentionVelocityGaussianBackend,
                MaxSpeedBarrierVelocityGaussianBackend,
                SingleGaussianBarrierVelocityGaussianBackend,
                PhysicalVelocityGaussianBackend,
                PostAttentionProposalVelocityGaussianBackend,
                ProposalInitializedVelocityGaussianBackend,
                SeedConditionedBarrierVelocityGaussianBackend,
                StraightThroughProposalVelocityGaussianBackend,
                WarpedProposalVelocityGaussianBackend,
            )

            # V3.1 differs from V3 only by its objective. V4 preserves the same
            # parameter structure but activates attention-derived initialization.
            # V5 shares V4's backend and differs only by temporal config keys.
            # V6 has an independent Siamese proposal. V7/V7.1 use its direct-L2
            # sparse successor. V7.2 uses a direct Top-4 forward/full-key STE
            # matcher, then V5-style endpoint-time refinement and a compact
            # feature-only velocity offset. V9 keeps a standalone proposal
            # branch but runs it after temporal refinement on the refined
            # feature, so it needs no raw pre-attention Utonia stream. V10 reads
            # every head at every temporal layer and learns a global-token
            # softmax over the resulting dense coordinate expectations. V11
            # keeps that adaptive-K backend but reads head 0 only, under a
            # max-speed barrier, and mixes layers with plain learned scalars.
            # V11.2 moves correspondence out of the stack entirely: every
            # layer head gets RoPE so the loop is one FlashAttention call, and
            # one dedicated readout head builds the initializer from the
            # complete final feature under V11's barrier. V11.3 keeps V11's
            # stack and correspondence but replaces the router with a fixed
            # count, and rebuilds the seed conditioning the router's trunk
            # used to supply on the plain Gaussian head. V11.4 is V11 with
            # only the hinge shape changed, so it shares V11's class and
            # differs by dynamic_2dgs.temporal config alone.
            backend_cls = {
                DYNAMIC_VARIANT_V1: DynamicGaussianBackend,
                DYNAMIC_VARIANT_V3: PhysicalVelocityGaussianBackend,
                DYNAMIC_VARIANT_V3_1: PhysicalVelocityGaussianBackend,
                DYNAMIC_VARIANT_V4: AttentionInitializedVelocityGaussianBackend,
                DYNAMIC_VARIANT_V5: AttentionInitializedVelocityGaussianBackend,
                DYNAMIC_VARIANT_V6: ProposalInitializedVelocityGaussianBackend,
                DYNAMIC_VARIANT_V7: WarpedProposalVelocityGaussianBackend,
                DYNAMIC_VARIANT_V7_1: WarpedProposalVelocityGaussianBackend,
                DYNAMIC_VARIANT_V7_2: (
                    StraightThroughProposalVelocityGaussianBackend
                ),
                DYNAMIC_VARIANT_V8: ConsensusAttentionVelocityGaussianBackend,
                DYNAMIC_VARIANT_V9: (
                    PostAttentionProposalVelocityGaussianBackend
                ),
                DYNAMIC_VARIANT_V10: (
                    LayerWeightedAttentionVelocityGaussianBackend
                ),
                DYNAMIC_VARIANT_V11: MaxSpeedBarrierVelocityGaussianBackend,
                DYNAMIC_VARIANT_V11_1: (
                    SingleGaussianBarrierVelocityGaussianBackend
                ),
                DYNAMIC_VARIANT_V11_2: (
                    FinalFeatureBarrierVelocityGaussianBackend
                ),
                DYNAMIC_VARIANT_V11_3: (
                    SeedConditionedBarrierVelocityGaussianBackend
                ),
                DYNAMIC_VARIANT_V11_4: MaxSpeedBarrierVelocityGaussianBackend,
            }.get(self.dynamic_variant)
            if backend_cls is None:
                raise ValueError(
                    f"Unsupported dynamic model variant={self.dynamic_variant!r}"
                )
            self._uses_motion_proposal = self.dynamic_variant in (
                DYNAMIC_VARIANT_V6,
                DYNAMIC_VARIANT_V7,
                DYNAMIC_VARIANT_V7_1,
                DYNAMIC_VARIANT_V7_2,
            )
            backend_kwargs = {}
            if self._uses_motion_proposal:
                backend_kwargs["proposal_dim"] = self.utonia_feature_dim
            if self.dynamic_adaptive_count:
                backend_kwargs["gaussian_count_cfg"] = cfg.grid_query
            else:
                backend_kwargs["gaussians_per_token"] = (
                    self.dynamic_gaussians_per_token
                )
            self.dynamic_backend = backend_cls(
                self.dynamic_cfg,
                cfg.gs_params,
                dim=trunk_dim,
                offset_bound=self.offset_bound,
                **backend_kwargs,
            )
            return

        # Both anchor modes share the token temporal-fusion stage (background
        # 0.8 m radius cross-frame attention + per-instance self-attention).
        # The attribute keeps its grid-era name so grid checkpoints load
        # unchanged; spherical runs it without seed geometry.
        from .grid_query_head import GridTemporalAggregator

        grid_query_cfg = getattr(cfg, "grid_query", None)
        if grid_query_cfg is None:
            raise ValueError(
                "p2g.grid_query config block is required for the shared temporal fusion"
            )
        self.grid_temporal_agg = GridTemporalAggregator(
            grid_query_cfg, dim=trunk_dim,
            r_far=float(getattr(cfg, "r_far", 70.0)),
        )

        if self.anchor_mode == "spherical":
            from .spherical_query_head import SphericalQueryHead
            squery_cfg = getattr(cfg, "squery", None)
            if squery_cfg is None:
                raise ValueError("p2g.squery config block is required for anchor_mode='spherical'")
            self.squery_cfg = squery_cfg
            self.squery_head = SphericalQueryHead(
                squery_cfg, dim=trunk_dim,
                r_far=float(getattr(cfg, "r_far", 70.0)),
                ring_to_elevation_deg=getattr(
                    cfg, "ring_to_elevation_deg", None
                ),
            )
            if self.squery_head.count_mode == "legacy":
                self.gs_predictor = nn.Sequential(
                    nn.Linear(trunk_dim + 3, trunk_dim),
                    nn.SiLU(),
                    nn.Linear(trunk_dim, gs_out_dim),
                )
            else:
                from .grid_query_head import GridSlotHead

                # Reuse the grid learned-count implementation exactly for
                # hard Gumbel-ST routing, K-specific joint heads, opacity-gate
                # router gradients, and per-K gradient balancing. Only its
                # router input width differs: [f_anchor, e_statistic].
                self.squery_slot_head = GridSlotHead(
                    squery_cfg,
                    cfg.gs_params,
                    dim=trunk_dim,
                    router_dim=self.squery_head.router_dim,
                )
        else:  # grid; validated by build_token_builder
            from .grid_query_head import GridSlotHead

            self.grid_slot_head = GridSlotHead(
                grid_query_cfg, cfg.gs_params, dim=trunk_dim,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_utonia", False):
            self.feature_extractor.eval()
            if getattr(self, "utonia_lora_enabled", False):
                set_active_lora_mode(self._active_utonia_lora_roots(), mode)
        return self

    def _active_utonia_lora_roots(self):
        """Modules executed by the configured early-exit encoder stage.

        Adapters are installed throughout Utonia, but stages after the selected
        feature stage stay frozen so DDP never sees trainable unused parameters.
        """
        roots = [self.feature_extractor.embedding]
        roots.extend(
            self.feature_extractor.enc[index]
            for index in range(self.utonia_feature_stage + 1)
        )
        return roots

    def _validate_utonia_feature_stage(self):
        max_stage = len(self.feature_extractor.enc) - 1
        if self.utonia_feature_stage < 0 or self.utonia_feature_stage > max_stage:
            raise ValueError(
                f"p2g.utonia_feature_stage must be in [0, {max_stage}], "
                f"got {self.utonia_feature_stage}"
            )

    def _infer_utonia_stride_factor(self, stage: int = None):
        stage = self.utonia_feature_stage if stage is None else int(stage)
        stride_factor = 1
        if getattr(self.feature_extractor, "enc_mode", False):
            for stage_idx in range(1, stage + 1):
                down = getattr(self.feature_extractor.enc[stage_idx], "down", None)
                if down is not None:
                    stride_factor *= int(down.stride)
        return stride_factor

    def _infer_utonia_feature_grid_size(self):
        return (
            self.utonia_input_grid_size
            / self.utonia_coord_scale
            * self.utonia_stride_factor
        )

    def _infer_utonia_feature_dim(self):
        stage_module = self.feature_extractor.enc[self.utonia_feature_stage]
        for module in reversed(list(stage_module.modules())):
            if hasattr(module, "channels"):
                return int(module.channels)
            if hasattr(module, "out_channels"):
                return int(module.out_channels)
        raise RuntimeError(f"Could not infer Utonia feature dim for stage {self.utonia_feature_stage}")

    def _forward_utonia_features(self, ptv3_input):
        if self.utonia_lora_enabled:
            ptv3_input = build_xyzi_utonia_input(ptv3_input)
        max_stage = len(self.feature_extractor.enc) - 1
        if self.utonia_feature_stage == max_stage:
            return self.feature_extractor(ptv3_input)

        point = utonia.structure.Point(ptv3_input)
        point = self.feature_extractor.embedding(point)
        point.serialization(
            order=self.feature_extractor.order,
            shuffle_orders=self.feature_extractor.shuffle_orders,
        )
        point.sparsify()

        for stage_idx in range(self.utonia_feature_stage + 1):
            point = self.feature_extractor.enc[stage_idx](point)
        return point

    def split_gs_params(self, feat):
        parts = torch.split(feat, self.gs_param_sizes, dim=-1)
        keys = ["shs", "opacity", "scaling", "rotation"]
        if self.use_offset:
            keys.append("offset")
        return dict(zip(keys, parts))

    def forward(self, _input, target_pose=None, target_timestamps=None):
        """The anchor mode is fixed at construction from ``cfg.anchor_mode``, and
        train/eval behaviour follows ``nn.Module.training``, so this module needs
        neither a split name nor a step counter from the caller.

        ``target_pose``/``target_timestamps`` are the render-time ``gt["pose"]``
        and ``gt["timestamps"]``, which index the same target views. Only the
        viewpoint-conditioned count router consumes them.
        """
        lidar_points = _input.get("lidar_points_sensor", _input["lidar_points"])
        offset = _input["offset"]
        point_batch_idx = _input["batch_idx"]
        pose = _input["pose"]
        # Bboxes remain in the dataloader for legacy runs, but the dynamic
        # backend does not even read their keys. This keeps its model contract
        # genuinely bbox-free and lets the loader be simplified independently
        # later without touching the representation.
        bbox = _input["bbox"] if self.dynamic_cfg is None else None
        bbox_instance_ids = (
            _input.get("bbox_instance_ids") if self.dynamic_cfg is None else None
        )
        features = self._forward_utonia_features(_input["ptv3_input"])

        # Shared contract for both anchor modes: occupied Cartesian tokens and
        # optional grid-only variable-K seed geometry.
        token_batch = self.anchor_builder(
            lidar_points, offset, pose, features, _input["ptv3_input"], self.grid_mapper
        )
        pos_list = token_batch.positions
        ufeat_list = token_batch.utonia_features
        ifeat_list = token_batch.intensity_features
        if self.utonia_lora_enabled and ifeat_list is not None:
            raise RuntimeError(
                "Utonia XYZI LoRA unexpectedly received separate intensity "
                "features"
            )
        if not self.utonia_lora_enabled and ifeat_list is None:
            raise RuntimeError("legacy fusion requires separate intensity features")
        gc_list = token_batch.grid_coords
        grid_seed_list = token_batch.grid_seeds
        raw_membership_list = token_batch.raw_memberships
        if self.anchor_mode == "grid" and grid_seed_list is None:
            raise RuntimeError("grid token builder did not return seed data")
        if self.anchor_mode == "spherical" and grid_seed_list is not None:
            raise RuntimeError("spherical token builder unexpectedly constructed grid seeds")
        if self.anchor_mode == "spherical" and raw_membership_list is None:
            raise RuntimeError("spherical token builder did not return raw-token membership")
        if self.anchor_mode == "grid" and raw_membership_list is not None:
            raise RuntimeError("grid token builder unexpectedly returned raw-token membership")

        n_frames = len(pos_list)

        # 프레임별 batch index
        frame_starts    = torch.cat([torch.tensor([0], device=offset.device), offset[:-1]])
        frame_batch_idx = point_batch_idx[frame_starts.long()]  # (n_frames,)

        all_pos         = []
        all_ufeat       = []
        all_ifeat       = [] if ifeat_list is not None else None
        all_gc          = []
        all_offset      = []
        all_frame_batch = []
        all_bbox        = []   # List[Tensor(B_f, 7)], 프레임별
        all_bbox_iids   = []
        all_raw_point_sensor = []
        all_raw_token_index = []
        local_frame_counter = {}
        cumsum = 0

        for i in range(n_frames):
            pos_i = pos_list[i]
            n_i   = pos_i.shape[0]

            b         = frame_batch_idx[i].item()
            local_f   = local_frame_counter.get(b, 0)
            local_frame_counter[b] = local_f + 1

            all_pos.append(pos_i)
            all_ufeat.append(ufeat_list[i])
            if all_ifeat is not None:
                all_ifeat.append(ifeat_list[i])
            all_gc.append(gc_list[i])
            all_frame_batch.append(b)
            if self.dynamic_cfg is None:
                all_bbox.append(bbox[b][local_f])   # Tensor(B_f, 7)
                if bbox_instance_ids is not None:
                    all_bbox_iids.append(bbox_instance_ids[b][local_f])
            if raw_membership_list is not None:
                membership = raw_membership_list[i]
                all_raw_point_sensor.append(membership.points_sensor)
                all_raw_token_index.append(membership.token_index + cumsum)
            cumsum += n_i
            all_offset.append(torch.tensor(cumsum, device=pos_i.device))

        all_pos       = torch.cat(all_pos,   dim=0)   # (N_valid, 3)
        all_ufeat     = torch.cat(all_ufeat, dim=0)   # (N_valid, C_stage)
        if all_ifeat is not None:
            all_ifeat = torch.cat(all_ifeat, dim=0)   # (N_valid, D)
        all_gc        = torch.cat(all_gc, dim=0)      # (N_valid, 3) token grid_coord
        new_offset    = torch.stack(all_offset)        # (n_frames,)
        frame_batch_idx = torch.tensor(
            all_frame_batch, dtype=torch.long, device=all_ufeat.device
        )  # (n_frames,)

        if self.anchor_mode == "grid":
            all_seed_sensor = torch.cat(
                [item.seed_sensor for item in grid_seed_list], dim=0
            )
            all_delta_sensor = torch.cat(
                [item.delta_sensor for item in grid_seed_list], dim=0
            )
            if self.dynamic_cfg is not None:
                all_anchor_k = None
            elif self.grid_slot_head.count_mode == "legacy":
                if any(item.anchor_k is None for item in grid_seed_list):
                    raise RuntimeError(
                        "legacy grid seed data must provide anchor_k"
                    )
                all_anchor_k = torch.cat(
                    [item.anchor_k for item in grid_seed_list], dim=0
                )
            else:
                if any(item.anchor_k is not None for item in grid_seed_list):
                    raise RuntimeError(
                        "learned-count seed data must defer anchor_k prediction"
                    )
                all_anchor_k = None
        else:
            all_raw_point_sensor = torch.cat(all_raw_point_sensor, dim=0)
            all_raw_token_index = torch.cat(all_raw_token_index, dim=0)

        if all_ufeat.shape[1] != self.utonia_feature_dim:
            raise RuntimeError(
                f"Utonia stage {self.utonia_feature_stage} feature dim mismatch: "
                f"expected {self.utonia_feature_dim}, got {all_ufeat.shape[1]}"
            )

        if self.utonia_lora_enabled:
            # Direct path: Utonia(XYZI)+LoRA -> temporal cross-attention.
            # There is no adapter, concat/fusion MLP, or local token refiner.
            agg_feat_i = all_ufeat
        else:
            agg_feat_i = fuse_features(
                self.utonia_adapter,
                self.intensity_agg_mlp,
                self.joint_refiner,
                utonia_feature=all_ufeat,
                intensity_feature=all_ifeat,
                position=all_pos,
                grid_coord=all_gc,
                offset=new_offset,
            )

        if self.dynamic_cfg is not None:
            timestamps_sec = _input.get("timestamps_sec")
            window_duration_sec = _input.get("window_duration_sec")
            if timestamps_sec is None or window_duration_sec is None:
                raise KeyError(
                    "Dynamic 2DGS requires timestamps_sec and "
                    "window_duration_sec from the dataloader"
                )
            dynamic_kwargs = {}
            if self._uses_motion_proposal:
                # The matcher consumes the same frozen-base, LoRA-adapted
                # Utonia token as temporal attention. Variant-specific matching
                # then decides whether to project it or L2-normalize it directly.
                dynamic_kwargs["motion_proposal_feature"] = all_ufeat
            # The router selects a candidate row from the seed bank, and a
            # seed-conditioned fixed-count head reads its slots directly.
            # Either way the decoder needs the own-frame seed deltas.
            if self.dynamic_adaptive_count or getattr(
                self.dynamic_backend, "seed_conditioned_head", False
            ):
                dynamic_kwargs["seed_delta_sensor"] = all_delta_sensor
            dynamic_out = self.dynamic_backend(
                agg_feat_i,
                all_pos,
                all_seed_sensor,
                new_offset,
                frame_batch_idx,
                pose,
                _input["timestamps"],
                timestamps_sec,
                window_duration_sec,
                **dynamic_kwargs,
            )
            routing_stats = dynamic_out.pop("routing_stats", None)
            return {
                **dynamic_out,
                "pose": pose,
                "timestamps": _input["timestamps"],
                "timestamps_sec": timestamps_sec,
                "window_duration_sec": window_duration_sec,
                "routing_stats": routing_stats,
                # V10/V11/V11.2 learn K only from rendering through the opacity
                # STE gate. No count-budget objective is constructed or returned.
                "routing_budget_logits": None,
            }

        if self.anchor_mode == "spherical":
            seeds = build_spherical_gaussian_seeds(
                self.grid_temporal_agg,
                self.squery_head,
                agg_feat_i,
                all_pos,
                all_raw_point_sensor,
                all_raw_token_index,
                new_offset,
                frame_batch_idx,
                pose,
                bbox,
                bbox_instance_ids,
                _input.get("timestamps"),
                slot_head=getattr(self, "squery_slot_head", None),
            )
        else:
            seeds = build_grid_gaussian_seeds(
                self.grid_temporal_agg,
                self.grid_slot_head,
                agg_feat_i,
                all_pos,
                all_anchor_k,
                all_seed_sensor,
                all_delta_sensor,
                new_offset,
                frame_batch_idx,
                pose,
                bbox,
                bbox_instance_ids,
                _input.get("timestamps"),
                target_pose=target_pose,
                target_timestamps=target_timestamps,
            )

        # Legacy spherical retains the shared predictor. Grid and learned
        # spherical already return tensors in the exact split_gs_params layout.
        if seeds.raw_params is None:
            if seeds.feature is None:
                raise RuntimeError("Gaussian seeds provide neither features nor raw parameters")
            if seeds.delta is None or seeds.delta.shape != (seeds.feature.shape[0], 3):
                raise RuntimeError("spherical Gaussian seeds must provide one 3D delta per feature")
            delta = seeds.delta.to(device=seeds.feature.device, dtype=seeds.feature.dtype)
            gs_feat = self.gs_predictor(torch.cat([seeds.feature, delta], dim=-1))
        else:
            gs_feat = seeds.raw_params
        gs_raw = self.split_gs_params(gs_feat)
        out_coord = seeds.position
        if self.use_offset:
            out_coord = out_coord + self.offset_bound * torch.tanh(gs_raw.pop("offset"))

        # Grid's variable-K balancing is forward-identical and is applied once at
        # this common renderer boundary. Spherical has no gradient weight.
        if seeds.gradient_weight is not None:
            out_coord = gradient_scale_identity(out_coord, seeds.gradient_weight)
            for key in ("shs", "opacity", "scaling", "rotation"):
                gs_raw[key] = gradient_scale_identity(
                    gs_raw[key], seeds.gradient_weight
                )

        agg_meta = refresh_coord_ref_after_offset(
            out_coord, seeds.metadata, seeds.frame_offset
        )
        batch_gaussians = assemble_batch_gaussians(
            gs_raw,
            out_coord,
            agg_meta,
            seeds.frame_offset,
            all_frame_batch,
            all_bbox,
            all_bbox_iids,
            bbox_instance_ids is not None,
            out_coord.device,
        )
        gauss_counts = torch.diff(
            seeds.frame_offset,
            prepend=seeds.frame_offset.new_zeros(1),
        )
        gaussian_batch = torch.repeat_interleave(
            frame_batch_idx.to(seeds.frame_offset.device), gauss_counts
        )

        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": gaussian_batch,
            "pose": pose,
            "timestamps": _input["timestamps"],
            "routing_stats": seeds.routing_stats,
            "routing_budget_logits": seeds.routing_budget_logits,
        }
