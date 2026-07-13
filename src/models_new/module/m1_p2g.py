from __future__ import annotations

import os

import torch
import torch.nn as nn

from .. import utonia
from .builders import build_anchor_builder
from .builders.common import UtoniaGridMapper
from .feature_fusion import build_feature_fusion, fuse_features
from .gaussian_assembly import (
    assemble_batch_gaussians,
    refresh_coord_ref_after_offset,
)
from .temporal_aggregator import TimeAgg


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
        self.utonia_stride_factor = self._infer_utonia_stride_factor()
        self.utonia_feature_grid_size = self._infer_utonia_feature_grid_size()
        self.utonia_feature_dim = self._infer_utonia_feature_dim()

        # Utonia bottleneck grid mapper shared by all builder implementations.
        self.grid_mapper = UtoniaGridMapper(
            self.utonia_coord_scale, self.utonia_input_grid_size, self.utonia_stride_factor
        )
        (
            self.anchor_mode,
            self.anchor_builder,
            self.builder_returns_grid_coord,
        ) = build_anchor_builder(cfg)

        self.agg_mlp = cfg.agg_mlp
        # Intensity width is part of the selected builder contract.
        intensity_out_dim = getattr(self.anchor_builder, "intensity_out_dim", None)
        if intensity_out_dim is None:
            ie_cfg = getattr(cfg, "intensity_encoder", None)
            refiner_cfg = getattr(cfg, "joint_refiner", None)
            if (ie_cfg is not None and str(getattr(ie_cfg, "type", "")) == "ptv3") or (
                refiner_cfg is not None and bool(getattr(refiner_cfg, "enable", False))
            ):
                raise ValueError(
                    "anchor_mode='spherical_legacy' (SphericalAnchorBuilder) does not "
                    "support intensity_encoder.type='ptv3' or joint_refiner (it emits no "
                    "per-token grid_coord); use anchor_mode 'grid' or 'spherical'.")
            intensity_out_dim = int(cfg.int_proj.out_dim)
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

        if self.anchor_mode == "spherical":
            from .spherical_query_head import SphericalQueryHead
            squery_cfg = getattr(cfg, "squery", None)
            if squery_cfg is None:
                raise ValueError("p2g.squery config block is required for anchor_mode='spherical'")
            self.squery_cfg = squery_cfg
            self.squery_head = SphericalQueryHead(
                squery_cfg, dim=self.agg_mlp.out_dim,
                r_far=float(getattr(cfg, "r_far", 70.0)),
            )
            self.time_agg = None   # replaced by the query head in this mode
        else:
            self.time_agg = TimeAgg(dim=self.agg_mlp.out_dim, num_heads= 8, k_bg= 8, k_fg= 16)

        # Gaussian head size is derived purely from gs_params (shs+opacity+scaling+
        # rotation [+offset]). Edit gs_params to control the head. offset>0 adds the
        # position-offset head; 0 disables it for every anchor mode.
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
        # Input dim is dictated by the trunk width (agg_mlp.out_dim; TimeAgg preserves
        # dim), so the gs head can never desync when the trunk width changes.
        trunk_dim = int(self.agg_mlp.out_dim)
        self.gs_predictor = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim, gs_out_dim),
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

    def forward(self, _input, batch_idx, mode):
        lidar_points = _input.get("lidar_points_sensor", _input["lidar_points"])
        offset = _input["offset"]
        batch_idx = _input["batch_idx"]
        pose = _input["pose"]
        bbox = _input["bbox"]
        bbox_instance_ids = _input.get("bbox_instance_ids")


        features = self._forward_utonia_features(_input["ptv3_input"])

        # Builder contract: per-frame position/Utonia/intensity, plus grid_coord
        # for token builders used by the optional joint refiner.
        builder_out = self.anchor_builder(
            lidar_points, offset, pose, features, _input["ptv3_input"], self.grid_mapper
        )
        expected_items = 4 if self.builder_returns_grid_coord else 3
        if len(builder_out) != expected_items:
            raise RuntimeError(
                f"anchor builder for mode={self.anchor_mode!r} returned "
                f"{len(builder_out)} items; expected {expected_items}"
            )
        if self.builder_returns_grid_coord:
            pos_list, ufeat_list, ifeat_list, gc_list = builder_out
        else:
            pos_list, ufeat_list, ifeat_list = builder_out
            gc_list = None

        n_frames = len(pos_list)

        # 프레임별 batch index
        frame_starts    = torch.cat([torch.tensor([0], device=offset.device), offset[:-1]])
        frame_batch_idx = batch_idx[frame_starts.long()]  # (n_frames,)

        all_pos         = []
        all_ufeat       = []
        all_ifeat       = []
        all_gc          = [] if gc_list is not None else None
        all_offset      = []
        all_batch       = []
        all_frame_batch = []
        all_bbox        = []   # List[Tensor(B_f, 7)], 프레임별
        all_bbox_iids   = []
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
            if all_gc is not None:
                all_gc.append(gc_list[i])
            all_batch.append(torch.full((n_i,), b, dtype=torch.long, device=pos_i.device))
            all_frame_batch.append(b)
            all_bbox.append(bbox[b][local_f])   # Tensor(B_f, 7)
            if bbox_instance_ids is not None:
                all_bbox_iids.append(bbox_instance_ids[b][local_f])
            cumsum += n_i
            all_offset.append(torch.tensor(cumsum, device=pos_i.device))

        all_pos       = torch.cat(all_pos,   dim=0)   # (N_valid, 3)
        all_ufeat     = torch.cat(all_ufeat, dim=0)   # (N_valid, C_stage)
        all_ifeat     = torch.cat(all_ifeat, dim=0)   # (N_valid, D)
        if all_gc is not None:
            all_gc    = torch.cat(all_gc, dim=0)      # (N_valid, 3) token grid_coord
        new_batch     = torch.cat(all_batch, dim=0)   # (N_valid,)
        new_offset    = torch.stack(all_offset)        # (n_frames,)
        frame_batch_idx = torch.tensor(
            all_frame_batch, dtype=torch.long, device=all_ufeat.device
        )  # (n_frames,)

        if all_ufeat.shape[1] != self.utonia_feature_dim:
            raise RuntimeError(
                f"Utonia stage {self.utonia_feature_stage} feature dim mismatch: "
                f"expected {self.utonia_feature_dim}, got {all_ufeat.shape[1]}"
            )

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

        if self.anchor_mode == "spherical":
            # Spherical query path (replaces TimeAgg): K learnable queries per
            # (theta, phi, log r) anchor cross-attend to the anchor's token
            # neighbourhood; each query yields one gaussian whose seed position is
            # the attention-weighted K/V position (bg: ref frame, fg: box-local).
            out_feat, p_init, _r_anchor, gauss_offset, agg_meta = self.squery_head(
                agg_feat_i, all_pos, new_offset, frame_batch_idx, pose, bbox,
                bbox_instance_ids,
            )
            gs_feat = self.gs_predictor(out_feat)
            gs_raw = self.split_gs_params(gs_feat)
            out_coord = p_init
            if self.use_offset:
                # Same bounded offset contract as grid/legacy modes: each xyz
                # component is limited to +/- head_offset_bound meters.
                out_coord = out_coord + self.offset_bound * torch.tanh(gs_raw.pop("offset"))
            agg_meta = refresh_coord_ref_after_offset(out_coord, agg_meta, gauss_offset)
            batch_gaussians = assemble_batch_gaussians(
                gs_raw, out_coord, agg_meta, gauss_offset, all_frame_batch,
                all_bbox, all_bbox_iids, bbox_instance_ids is not None, out_feat.device,
            )
            gauss_counts = torch.diff(gauss_offset, prepend=gauss_offset.new_zeros(1))
            new_batch_g = torch.repeat_interleave(
                frame_batch_idx.to(gauss_offset.device), gauss_counts)
            return {
                "batch_gaussians": batch_gaussians,
                "gaussians": batch_gaussians,  # List[dict], 배치별 (구조는 grid 경로와 동일)
                "batch": new_batch_g,
                "pose": pose,
                "timestamps": _input["timestamps"],
            }

        # time_agg: out_coord is the Gaussian position coordinate. Dynamic points
        # are represented in object-local coordinates; static points stay in ref frame.
        out_feat, out_coord, agg_meta = self.time_agg(
            agg_feat_i, all_pos, new_offset,
            frame_batch_idx, pose, bbox, bbox_instance_ids
        )

        # GS 예측
        gs_feat = self.gs_predictor(out_feat)
        gs_raw = self.split_gs_params(gs_feat)
        # gs_raw: dict of (N_valid, ...) tensors

        if self.use_offset:
            out_coord = out_coord + self.offset_bound * torch.tanh(gs_raw.pop("offset"))
        agg_meta = refresh_coord_ref_after_offset(out_coord, agg_meta, new_offset)

        # 배치별로 분리 + box_assign 붙이기 (모든 anchor_mode 공유 헬퍼)
        batch_gaussians = assemble_batch_gaussians(
            gs_raw, out_coord, agg_meta, new_offset, all_frame_batch,
            all_bbox, all_bbox_iids, bbox_instance_ids is not None, out_feat.device,
        )

        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,  # List[dict], 배치별
            # batch_gaussians[b] 구조:
            #   "position"     : (Nb, 3)
            #   "opacity"      : (Nb, 1)
            #   "scale"        : (Nb, 2)
            #   "rotation"     : (Nb, 4)
            #   "shs" : (Nb, 16) -> L =3 임.
            #   "coord"        : (Nb, 3)  frame_0 좌표계
            #   "box_assign"   : (Nb,)    -1=bg, 0~B-1=box index
            #   "bg_mask"      : (Nb,)    bool
            #   "fg_masks"     : {box_id: (Nb,) bool}
            #   "frame_bboxes"  : List[dict] 프레임별
            #       [local_f]["center"] : (B_f, 3)
            #       [local_f]["size"]   : (B_f, 3)
            #       [local_f]["yaw"]    : (B_f,)
            #       [local_f]["bbox"]   : (B_f, 7)
            # bbox바더서 linear 이동할 준비. 
            "batch":  new_batch,
            "pose": pose,
            "timestamps": _input["timestamps"],
        }
