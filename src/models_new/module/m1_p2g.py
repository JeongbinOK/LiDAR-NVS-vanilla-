from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# from diff_quadratic_rasterization import LIDAR_LATENT_DIM
# from models.geometry import decompose_scene
# from models.geometry.voxel_anchor import (
#     DynamicVoxelAnchorBuilder,
#     VoxelAnchorBuilder,
#     VoxelAnchorOutput,
# )
# from models.head.qgs_head import QGSHead
# from models.ptv3.wrapper import PTv3Backbone, validate_context_type
# from ..utils.lidar_geometry import make_lidar_ray_grid, points_to_lidar_maps
# from ..utils.qgs_loss import QGSLoss
# from ..utils.render_utils import (
#     build_gt_normal_map,
#     quat_to_rotmat,
#     render_primitives,
#     rotmat_to_quat,
# )
#from ..utonia.model import PointTransformerV3
from .. import utonia
from ..utils.attention import LocalAttentionFlash
from .builders.common import UtoniaGridMapper
import numpy as np
import os
import torch
import numpy as np
import open3d as o3d


def split_by_offset(tensor, offset):
    splits = []
    start = 0
    for end in offset:
        end = int(end)
        splits.append(tensor[start:end])
        start = end
    return splits


def save_pcd(points, color, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.tile(color, (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd)


def save_all_batches(x, voxelized_points, features, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    lidar_points = x["lidar_points"][:, :3]
    offset = x["offset"]                          # 프레임별 offset (원본 포인트)
    feat_offset = features["offset"]              # utonia 출력 offset (GridSample 후)
    
    anchor_points = voxelized_points["anchor_points"]
    coords = features["coord"] /0.2            # scale 복원

    lidar_list  = split_by_offset(lidar_points, offset)
    anchor_list = anchor_points
    coord_list  = split_by_offset(coords, feat_offset)  # feat_offset 사용



    for i in range(len(offset)):

        print("원본 coord range:", lidar_list[i].min(), lidar_list[i].max())
        print("transform 후 coord range:", (coord_list[i]*0.2).min(), (coord_list[i]*0.2).max())
        print("/ 0.2 후:", (coord_list[i] ).min(), (coord_list[i] ).max())

        lidar_np  = lidar_list[i].cpu().numpy()
        anchor_np = anchor_list[i].cpu().numpy()
        coord_np  = coord_list[i].cpu().numpy()

        all_points = np.concatenate([lidar_np, anchor_np, coord_np], axis=0)
        all_colors = np.concatenate([
            np.tile([0, 0, 1], (lidar_np.shape[0],  1)),   # 파란색 = lidar 원본
            np.tile([1, 0, 0], (anchor_np.shape[0], 1)),   # 빨간색 = anchor (voxelizer)
            np.tile([0, 1, 0], (coord_np.shape[0],  1)),   # 초록색 = utonia 출력
        ], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(f"{save_dir}/batch_{i}_all.ply", pcd)

        save_pcd(lidar_np,  [0, 0, 1], f"{save_dir}/batch_{i}_lidar.ply")
        save_pcd(anchor_np, [1, 0, 0], f"{save_dir}/batch_{i}_anchor.ply")
        save_pcd(coord_np,  [0, 1, 0], f"{save_dir}/batch_{i}_coord.ply")

        print(f"[Saved] batch {i} | lidar={lidar_np.shape[0]} anchor={anchor_np.shape[0]} coord={coord_np.shape[0]}")
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
        self.utonia_stride_factor = self._infer_utonia_stride_factor()
        self.utonia_feature_grid_size = self._infer_utonia_feature_grid_size()

        # Utonia bottleneck grid mapper (shared by both builders).
        self.grid_mapper = UtoniaGridMapper(
            self.utonia_coord_scale, self.utonia_input_grid_size, self.utonia_stride_factor
        )
        # Anchor/primitive builder, selected by config (only the active one imported).
        # Each builder owns its IntensityEncoder (5D->64D) and utonia injection, and
        # returns per-frame (position[xyz], utonia_feat[576], intensity_feat[64]).
        self.anchor_mode = str(getattr(cfg, "anchor_mode", "spherical"))
        if self.anchor_mode == "grid":
            from .builders.grid_intensity import GridIntensityBuilder as _Builder
        else:
            from .builders.spherical_anchor import SphericalAnchorBuilder as _Builder
        self.anchor_builder = _Builder(cfg)

        self.agg_mlp = cfg.agg_mlp
        self.intensity_agg_mlp = nn.Sequential(
            nn.Linear(self.agg_mlp.in_dim, self.agg_mlp.hidden_dim), # 576 + 64 (640) -> 256  
            nn.SiLU(),
            nn.Linear(self.agg_mlp.hidden_dim, self.agg_mlp.hidden_dim), #256 -> 256
            nn.SiLU(),
            nn.Linear(self.agg_mlp.hidden_dim, self.agg_mlp.out_dim), # 256 -> 128 
        )

        self.time_agg = TimeAgg(dim=self.agg_mlp.out_dim, num_heads= 8, k_bg= 8, k_fg= 16)

        self.gs = cfg.gs
        # Gaussian head size is derived purely from gs_params (shs+opacity+scaling+
        # rotation [+offset]); cfg.gs.out_dim is unused. Edit gs_params to control the
        # head. offset>0 adds the position-offset head (set 3 for grid); 0 = none (spherical).
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
        self.gs_predictor = nn.Sequential(
            nn.Linear(self.gs.in_dim, self.gs.in_dim),
            nn.SiLU(),
            nn.Linear(self.gs.in_dim, gs_out_dim),
        )
        #self.feature_extractor = PointTransformerV3(cfg=self.cfg, finetune = True).from_pretrained("Pointcept/Utonia")
        #self.feature_condition = Conditionor(cfg=self.cfg)
        #self.gaussian_predictor = Predictor(cfg=self.cfg)
        # scene_id : {frame_id: {track_id(str), track_label: class(str), score: 3 float, xyz: float, size: 3 float, rotation: 4 float}}
        # yaw -> quat. code 

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_utonia", False):
            self.feature_extractor.eval()
        return self

    def _infer_utonia_stride_factor(self):
        stride_factor = 1
        if getattr(self.feature_extractor, "enc_mode", False):
            for module in self.feature_extractor.modules():
                if module.__class__.__name__ == "GridPooling":
                    stride_factor *= int(module.stride)
        return stride_factor

    def _infer_utonia_feature_grid_size(self):
        return (
            self.utonia_input_grid_size
            / self.utonia_coord_scale
            * self.utonia_stride_factor
        )

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


        features = self.feature_extractor(_input["ptv3_input"])

        # Builder -> per-frame (position[Mi,3], utonia_feat[Mi,576], intensity_feat[Mi,64])
        pos_list, ufeat_list, ifeat_list = self.anchor_builder(
            lidar_points, offset, pose, features, _input["ptv3_input"], self.grid_mapper
        )

        n_frames = len(pos_list)

        # 프레임별 batch index
        frame_starts    = torch.cat([torch.tensor([0], device=offset.device), offset[:-1]])
        frame_batch_idx = batch_idx[frame_starts.long()]  # (n_frames,)

        all_pos         = []
        all_ufeat       = []
        all_ifeat       = []
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
            all_batch.append(torch.full((n_i,), b, dtype=torch.long, device=pos_i.device))
            all_frame_batch.append(b)
            all_bbox.append(bbox[b][local_f])   # Tensor(B_f, 7)
            if bbox_instance_ids is not None:
                all_bbox_iids.append(bbox_instance_ids[b][local_f])
            cumsum += n_i
            all_offset.append(torch.tensor(cumsum, device=pos_i.device))

        all_pos       = torch.cat(all_pos,   dim=0)   # (N_valid, 3)
        all_ufeat     = torch.cat(all_ufeat, dim=0)   # (N_valid, 576)
        all_ifeat     = torch.cat(all_ifeat, dim=0)   # (N_valid, 64)
        new_batch     = torch.cat(all_batch, dim=0)   # (N_valid,)
        new_offset    = torch.stack(all_offset)        # (n_frames,)
        frame_batch_idx = torch.tensor(
            all_frame_batch, dtype=torch.long, device=all_ufeat.device
        )  # (n_frames,)

        # concat[utonia 576, intensity 64] = 640 -> agg_mlp -> 240
        agg_feat_i = self.intensity_agg_mlp(torch.cat([all_ufeat, all_ifeat], dim=1))

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

        # 배치별로 분리 + box_assign 붙이기
        n_batch = max(all_frame_batch) + 1
        batch_frame_map = {}
        for global_f, b in enumerate(all_frame_batch):
            batch_frame_map.setdefault(b, []).append(global_f)

        batch_gaussians = []
        for b in range(n_batch):
            frame_indices = batch_frame_map.get(b, [])

            # 이 배치의 anchor 범위
            b_start = (int(new_offset[frame_indices[0] - 1])
                    if frame_indices[0] > 0 and all_frame_batch[frame_indices[0] - 1] == b
                    else 0)
            b_end   = int(new_offset[frame_indices[-1]])
            b_slice = torch.arange(b_start, b_end, device=out_feat.device)

            b_box_assign = agg_meta["box_assign"][b_slice]   # (Nb,) -1=bg, 0~B-1=box
            b_instance_id = agg_meta["instance_id"][b_slice]
            b_is_dynamic = agg_meta["is_dynamic"][b_slice]

            b_gs = {k: v[b_slice] for k, v in gs_raw.items()}
            b_gs["position"] = out_coord[b_slice]
            b_gs["box_assign"] = b_box_assign    # (Nb,) 어떤 bbox에서 나왔는지
            b_gs["coord"]      = out_coord[b_slice]  # (Nb, 3) frame_0 좌표계
            b_gs["coord_ref"]  = agg_meta["coord_ref"][b_slice]
            b_gs["instance_id"] = b_instance_id
            b_gs["is_dynamic"] = b_is_dynamic

            # bg / fg 분리 인덱스
            b_gs["bg_mask"]  = ~b_is_dynamic          # (Nb,) bool
            b_gs["fg_masks"] = {
                int(inst_id): (b_instance_id == inst_id) & b_is_dynamic
                for inst_id in b_instance_id[b_is_dynamic].unique().tolist()
            }  # {instance_id: (Nb,) bool}
            b_gs["frame_bboxes"] = []
            for local_f, global_f in enumerate(frame_indices):
                bbox_f = all_bbox[global_f]   # Tensor(B_f, 7)
                bbox_ref_f = agg_meta["bbox_ref_by_frame"][global_f]
                if bbox_instance_ids is not None:
                    bbox_iids_f = all_bbox_iids[global_f]
                else:
                    bbox_iids_f = torch.arange(bbox_f.shape[0], device=out_feat.device, dtype=torch.long)
                b_gs["frame_bboxes"].append({
                    "center": bbox_f[:, :3],   # (B_f, 3)
                    "size":   bbox_f[:, 3:6],  # (B_f, 3)
                    "yaw":    bbox_f[:, 6],    # (B_f,)
                    "bbox":   bbox_f,
                    "bbox_ref": bbox_ref_f,
                    "instance_id": bbox_iids_f.to(device=out_feat.device),
                })
            batch_gaussians.append(b_gs)

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






class TimeAgg(nn.Module):
    def __init__(self, dim, num_heads=8, k_bg=8, k_fg=16, attn_drop=0.0):
        super().__init__()
        self.k_bg = k_bg
        self.k_fg = k_fg
        self.bg_attn = LocalAttentionFlash(dim, num_heads, attn_drop)
        self.fg_attn = LocalAttentionFlash(dim, num_heads, attn_drop)


    def apply_pose(self, points, pose):
        """
        points : (N, 3)
        pose   : (4, 4)  frame_i → frame_0
        """
        N = points.shape[0]
        ones = torch.ones(N, 1, device=points.device, dtype=points.dtype)
        pts_h = torch.cat([points, ones], dim=-1)   # (N, 4)
        return (pose @ pts_h.T).T[:, :3]            # (N, 3)

    def yaw_from_pose(self, pose):
        return torch.atan2(pose[1, 0], pose[0, 0])

    def transform_boxes_to_ref(self, boxes, pose):
        if boxes.shape[0] == 0:
            return boxes
        out = boxes.clone()
        out[:, :3] = self.apply_pose(boxes[:, :3], pose)
        out[:, 6] = boxes[:, 6] + self.yaw_from_pose(pose)
        return out

    def points_to_box_local(self, points_ref, box_ref):
        center = box_ref[:3]
        yaw = box_ref[6]
        shifted = points_ref - center.unsqueeze(0)
        cos_y = torch.cos(-yaw)
        sin_y = torch.sin(-yaw)
        local_x = cos_y * shifted[:, 0] - sin_y * shifted[:, 1]
        local_y = sin_y * shifted[:, 0] + cos_y * shifted[:, 1]
        return torch.stack([local_x, local_y, shifted[:, 2]], dim=-1)

    def _common_instance_ids(self, first_ids, last_ids):
        if first_ids is None or last_ids is None:
            return set()
        first = {int(x) for x in first_ids.tolist()}
        last = {int(x) for x in last_ids.tolist()}
        return first & last


    def point_in_box(self, anchor, box):
        """
        anchor : (N, 3)
        box    : (B_f, 7) x,y,z,w,l,h,yaw  sensor frame
        return : (N,) long  -1=background, 0~B_f-1=box index
        """
        N, B_f = anchor.shape[0], box.shape[0]
        if B_f == 0:
            return torch.full((N,), -1, dtype=torch.long, device=anchor.device)

        cx, cy, cz = box[:, 0], box[:, 1], box[:, 2]
        w, l, h    = box[:, 3], box[:, 4], box[:, 5]
        yaw        = box[:, 6]

        dx = anchor[:, 0].unsqueeze(1) - cx.unsqueeze(0)  # (N, B_f)
        dy = anchor[:, 1].unsqueeze(1) - cy.unsqueeze(0)
        dz = anchor[:, 2].unsqueeze(1) - cz.unsqueeze(0)

        cos_y  = torch.cos(-yaw).unsqueeze(0)
        sin_y  = torch.sin(-yaw).unsqueeze(0)
        local_x = cos_y * dx - sin_y * dy
        local_y = sin_y * dx + cos_y * dy

        inside = (
            (local_x.abs() <= w.unsqueeze(0) / 2) &
            (local_y.abs() <= l.unsqueeze(0) / 2) &
            (dz.abs()      <= h.unsqueeze(0) / 2)
        )  # (N, B_f)

        box_idx = torch.full((N,), -1, dtype=torch.long, device=anchor.device)
        for b in range(B_f):
            mask = inside[:, b] & (box_idx == -1)
            box_idx[mask] = b
        return box_idx


    def forward(self, feat, anchor, new_offset, frame_batch_idx, pose_list, bbox_list, bbox_instance_ids_list=None):
        """
        feat            : (N_valid, C)
        anchor          : (N_valid, 3)  각 프레임의 sensor frame 좌표
        new_offset      : (n_frames,)   프레임별 cumsum
        frame_batch_idx : (n_frames,)   프레임별 batch index
        pose_list       : List[Tensor(V, 4, 4)]  배치별 pose
        bbox_list       : List[List[Tensor(B_f, 7)]]  배치별, 프레임별 bbox
        """
        device = feat.device
        N, C   = feat.shape
        out_feat  = torch.zeros_like(feat)
        out_coord = torch.zeros_like(anchor)
        box_assign_out = torch.full((N,), -1, dtype=torch.long, device=device)
        instance_id_out = torch.full((N,), -1, dtype=torch.long, device=device)
        is_dynamic_out = torch.zeros((N,), dtype=torch.bool, device=device)
        coord_ref_out = torch.zeros_like(anchor)
        bbox_ref_by_frame = [None for _ in range(len(frame_batch_idx))]

        # 배치별로 처리
        batch_frame_map = {}
        for global_f, b in enumerate(frame_batch_idx.tolist()):
            batch_frame_map.setdefault(int(b), []).append(global_f)

        for b, frame_indices in batch_frame_map.items():
            pose_b = pose_list[b]    # Tensor(V, 4, 4)
            bbox_b = bbox_list[b]    # List[Tensor(B_f, 7)]
            bbox_iids_b = bbox_instance_ids_list[b] if bbox_instance_ids_list is not None else None
            if bbox_iids_b is not None and len(bbox_iids_b) >= 2:
                common_ids = self._common_instance_ids(bbox_iids_b[0], bbox_iids_b[-1])
            else:
                common_ids = set()

            #1. 각 프레임 anchor를 frame_0 좌표계로 변환 
            frame_feats   = []
            frame_coords  = []  # frame_0 좌표계
            frame_attn_coords = []
            frame_g_idx   = []  # global index
            frame_is_dynamic = []
            frame_box_assign = []
            frame_instance_id = []

            for local_f, global_f in enumerate(frame_indices):
                prev  = int(new_offset[global_f - 1]) if global_f > 0 and frame_batch_idx[global_f - 1] == b else 0
                end   = int(new_offset[global_f])
                g_idx = torch.arange(prev, end, device=device)

                feat_f   = feat[prev:end]     # (Nf, C)
                anchor_f = anchor[prev:end]   # (Nf, 3) sensor frame
                pose_f    = pose_b[local_f].to(device)   # (4, 4)
                coord_f0  = self.apply_pose(anchor_f, pose_f)  # (Nf, 3) frame_0 좌표계
                bbox_sensor_f = bbox_b[local_f].to(device)
                bbox_ref_f = self.transform_boxes_to_ref(bbox_sensor_f, pose_f)
                bbox_ref_by_frame[global_f] = bbox_ref_f

                if bbox_iids_b is not None:
                    bbox_iids_f = bbox_iids_b[local_f].to(device)
                else:
                    bbox_iids_f = torch.arange(bbox_sensor_f.shape[0], device=device, dtype=torch.long)

                box_assign_f = self.point_in_box(anchor_f, bbox_sensor_f)
                instance_id_f = torch.full((anchor_f.shape[0],), -1, dtype=torch.long, device=device)
                valid_box = box_assign_f >= 0
                if valid_box.any() and bbox_iids_f.numel() > 0:
                    instance_id_f[valid_box] = bbox_iids_f[box_assign_f[valid_box]]

                is_dynamic_f = torch.zeros_like(valid_box)
                for inst_id in common_ids:
                    is_dynamic_f |= instance_id_f == int(inst_id)

                attn_coord_f = coord_f0.clone()
                for box_i in box_assign_f[is_dynamic_f & (box_assign_f >= 0)].unique().tolist():
                    box_i = int(box_i)
                    mask = is_dynamic_f & (box_assign_f == box_i)
                    if mask.any():
                        attn_coord_f[mask] = self.points_to_box_local(coord_f0[mask], bbox_ref_f[box_i])

                frame_feats.append(feat_f)
                frame_coords.append(coord_f0)
                frame_attn_coords.append(attn_coord_f)
                frame_g_idx.append(g_idx)
                frame_is_dynamic.append(is_dynamic_f)
                frame_box_assign.append(box_assign_f)
                frame_instance_id.append(instance_id_f)

            #2. 전체 anchor를 bbox 기준으로 분류 (frame_0 좌표계) 
            all_feat_b   = torch.cat(frame_feats,  dim=0)   # (Nb, C)
            all_coord_b  = torch.cat(frame_coords, dim=0)   # (Nb, 3) frame_0
            all_coord_attn_b = torch.cat(frame_attn_coords, dim=0)
            all_g_idx_b  = torch.cat(frame_g_idx,  dim=0)   # (Nb,)
            all_dynamic_b = torch.cat(frame_is_dynamic, dim=0)
            all_box_assign_b = torch.cat(frame_box_assign, dim=0)
            all_instance_id_b = torch.cat(frame_instance_id, dim=0)

            coord_ref_out[all_g_idx_b] = all_coord_b
            is_dynamic_out[all_g_idx_b] = all_dynamic_b
            instance_id_out[all_g_idx_b] = all_instance_id_b
            box_assign_out[all_g_idx_b] = torch.where(
                all_dynamic_b,
                all_box_assign_b,
                torch.full_like(all_box_assign_b, -1),
            )

            # 4. background / foreground KNN attention 
            bg_mask  = ~all_dynamic_b   # (Nb,)

            # background
            if bg_mask.any():
                bg_feat  = all_feat_b[bg_mask]          # (N_bg, C)
                bg_coord = all_coord_attn_b[bg_mask]   # (N_bg, 3)
                bg_g_idx = all_g_idx_b[bg_mask]

                bg_out = self.bg_attn(bg_feat, bg_coord, k=self.k_bg)
                out_feat[bg_g_idx]  = bg_out
                out_coord[bg_g_idx] = bg_coord

            # foreground: instance별
            for inst_id in all_instance_id_b[all_dynamic_b].unique().tolist():
                inst_id = int(inst_id)
                fg_mask = all_dynamic_b & (all_instance_id_b == inst_id)

                fg_feat  = all_feat_b[fg_mask]
                fg_coord = all_coord_attn_b[fg_mask]
                fg_g_idx = all_g_idx_b[fg_mask]

                fg_out = self.fg_attn(fg_feat, fg_coord, k=self.k_fg)
                out_feat[fg_g_idx]  = fg_out
                out_coord[fg_g_idx] = fg_coord

        meta = {
            "box_assign": box_assign_out,
            "instance_id": instance_id_out,
            "is_dynamic": is_dynamic_out,
            "coord_ref": coord_ref_out,
            "bbox_ref_by_frame": bbox_ref_by_frame,
        }
        return out_feat, out_coord, meta   # (N_valid, C), (N_valid, 3), metadata
