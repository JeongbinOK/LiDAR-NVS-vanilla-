from __future__ import annotations

import os

import torch
import torch.nn as nn

from .. import utonia
from .anchor_modes import (
    build_grid_gaussian_seeds,
    build_spherical_gaussian_seeds,
)
from .builders import build_token_builder
from .builders.common import UtoniaGridMapper
from .feature_fusion import build_feature_fusion, fuse_features
from .gaussian_assembly import (
    assemble_batch_gaussians,
    gradient_scale_identity,
    refresh_coord_ref_after_offset,
)


class Point2Gaus(nn.Module):
    """Point to Gaussian."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.freeze_utonia = bool(getattr(cfg, "freeze_utonia", True))
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
        if self.freeze_utonia:
            self.feature_extractor.eval()
        self.utonia_coord_scale = 0.2
        self.utonia_input_grid_size = 0.01
        self.utonia_feature_stage = int(
            getattr(cfg, "utonia_feature_stage", len(self.feature_extractor.enc) - 1)
        )
        self._validate_utonia_feature_stage()

        lora_cfg = getattr(cfg, "utonia_lora", None)
        self.utonia_lora_enabled = bool(
            lora_cfg is not None and getattr(lora_cfg, "enabled", False)
        )
        if self.utonia_lora_enabled:
            lora_rank = int(getattr(lora_cfg, "rank", 16))
            n_lora = utonia.lora.inject_lora(
                self.feature_extractor,
                max_stage=self.utonia_feature_stage,
                rank=lora_rank,
                alpha=getattr(lora_cfg, "alpha", None),
            )
            print(f"Injected Utonia LoRA into {n_lora} layers (rank={lora_rank})")

        self.utonia_stride_factor = self._infer_utonia_stride_factor()
        self.utonia_feature_grid_size = self._infer_utonia_feature_grid_size()
        self.utonia_feature_dim = self._infer_utonia_feature_dim()

        # Both anchor modes start from the exact same occupied Utonia-grid tokens.
        self.grid_mapper = UtoniaGridMapper(
            self.utonia_coord_scale, self.utonia_input_grid_size, self.utonia_stride_factor
        )
        self.anchor_mode, self.anchor_builder = build_token_builder(cfg)

        self.agg_mlp = cfg.agg_mlp
        # Intensity width is part of the shared token-builder contract.
        intensity_out_dim = getattr(self.anchor_builder, "intensity_out_dim", None)
        if intensity_out_dim is None:
            raise RuntimeError("the shared token builder must expose intensity_out_dim")
        self.agg_in_dim = self.utonia_feature_dim + int(intensity_out_dim)
        (
            self.intensity_agg_mlp,
            self.joint_refiner,
        ) = build_feature_fusion(
            cfg,
            utonia_dim=self.utonia_feature_dim,
            intensity_dim=int(intensity_out_dim),
            coord_scale=self.utonia_coord_scale,
        )

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
        trunk_dim = int(self.agg_mlp.out_dim)

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
            grid_query_cfg, dim=self.agg_mlp.out_dim,
            r_far=float(getattr(cfg, "r_far", 70.0)),
        )

        if self.anchor_mode == "spherical":
            from .spherical_query_head import SphericalQueryHead
            squery_cfg = getattr(cfg, "squery", None)
            if squery_cfg is None:
                raise ValueError("p2g.squery config block is required for anchor_mode='spherical'")
            self.squery_cfg = squery_cfg
            self.squery_head = SphericalQueryHead(
                squery_cfg, dim=self.agg_mlp.out_dim,
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
                    dim=self.agg_mlp.out_dim,
                    router_dim=self.squery_head.router_dim,
                )
        else:  # grid; validated by build_token_builder
            from .grid_query_head import GridSlotHead

            self.grid_slot_head = GridSlotHead(
                grid_query_cfg, cfg.gs_params, dim=self.agg_mlp.out_dim,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_utonia", False):
            self.feature_extractor.eval()
        return self

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
        bbox = _input["bbox"]
        bbox_instance_ids = _input.get("bbox_instance_ids")
        features = self._forward_utonia_features(_input["ptv3_input"])

        # Shared contract for both anchor modes: occupied Cartesian tokens and
        # optional grid-only variable-K seed geometry.
        token_batch = self.anchor_builder(
            lidar_points, offset, pose, features, _input["ptv3_input"], self.grid_mapper
        )
        pos_list = token_batch.positions
        ufeat_list = token_batch.utonia_features
        ifeat_list = token_batch.intensity_features
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
        all_ifeat       = []
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
            all_ifeat.append(ifeat_list[i])
            all_gc.append(gc_list[i])
            all_frame_batch.append(b)
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
        all_ifeat     = torch.cat(all_ifeat, dim=0)   # (N_valid, D)
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
            if self.grid_slot_head.count_mode == "legacy":
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

        agg_feat_i = fuse_features(
            self.intensity_agg_mlp,
            self.joint_refiner,
            utonia_feature=all_ufeat,
            intensity_feature=all_ifeat,
            position=all_pos,
            grid_coord=all_gc,
            offset=new_offset,
        )

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
