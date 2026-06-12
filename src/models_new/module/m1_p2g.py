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
from ..utils.coord import Voxelizer
from ..utils.camera import intensity_dir
from ..utils.attention import LocalAttentionFlash
import numpy as np
import os
import torch
import numpy as np
import open3d as o3d
from ..utils.camera import cameraList_from_camInfos

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

        self.voxelizer = Voxelizer(cfg=self.cfg, max_frames=4)
        self.feature_extractor = utonia.load("utonia", repo_id="Pointcept/Utonia")
        

        self.int_proj = cfg.int_proj
        self.intensity_proj = nn.Linear(self.int_proj.in_dim, self.int_proj.out_dim) # 5-> 64
        self.intensity_norm = nn.LayerNorm(self.int_proj.out_dim) 

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
        self.gs_predictor = nn.Sequential(
            nn.Linear(self.gs.in_dim, self.gs.in_dim),
            nn.SiLU(),
            nn.Linear(self.gs.in_dim, self.gs.self.out_dim),
        )
        #self.feature_extractor = PointTransformerV3(cfg=self.cfg, finetune = True).from_pretrained("Pointcept/Utonia")
        #self.feature_condition = Conditionor(cfg=self.cfg)
        #self.gaussian_predictor = Predictor(cfg=self.cfg)
        # scene_id : {frame_id: {track_id(str), track_label: class(str), score: 3 float, xyz: float, size: 3 float, rotation: 4 float}}
        # yaw -> quat. code 

    def utonia_cond(self, anchor_points, grid_coord, voxel_feats, voxel_coords, sparse_shape, grid_size=0.16):
        

        M = anchor_points.shape[0]
        D = voxel_feats.shape[1]

        # 1. coord_min (sparsify 기준과 동일하게)
        coord_min = voxel_coords.min(0)[0]  # (3,)

        # 2. anchor를 grid index로 변환
        anchor_grid = torch.div(
            anchor_points - coord_min, grid_size, rounding_mode="trunc"
        ).long()  # (M, 3)

        # 3. anchor의 voxel 내 상대 위치로 8방향 결정
        anchor_local = (anchor_points - coord_min) / grid_size - anchor_grid.float()  # (M, 3), 0~1
        sign = torch.sign(anchor_local - 0.5).long()  # (M, 3), -1 or +1
        sx, sy, sz = sign[:, 0], sign[:, 1], sign[:, 2]
        zero = torch.zeros(M, dtype=torch.long, device=anchor_points.device)

        # (M, 8, 3) - 2x2x2 정육면체 offset
        offsets_per_anchor = torch.stack([
            torch.stack([ox, oy, oz], dim=-1)
            for ox in [zero, sx]
            for oy in [zero, sy]
            for oz in [zero, sz]
        ], dim=1)  # (M, 8, 3)

        # 4. sparse_shape 기반 lookup table
        #sparse_shape = point.sparse_shape  # [SX, SY, SZ]
        SX, SY, SZ = sparse_shape[0], sparse_shape[1], sparse_shape[2]

        linear_idx = (
            grid_coord[:, 0] * SY * SZ +
            grid_coord[:, 1] * SZ +
            grid_coord[:, 2]
        )  # (V,)
        table = torch.full((SX * SY * SZ,), -1, dtype=torch.long, device=anchor_points.device)
        table[linear_idx] = torch.arange(len(grid_coord), device=anchor_points.device)

        # 5. 8개 neighbor linear index
        neighbor_grids = anchor_grid[:, None, :] + offsets_per_anchor  # (M, 8, 3)
        nx = neighbor_grids[:, :, 0].clamp(0, SX - 1)
        ny = neighbor_grids[:, :, 1].clamp(0, SY - 1)
        nz = neighbor_grids[:, :, 2].clamp(0, SZ - 1)
        neighbor_linear = nx * SY * SZ + ny * SZ + nz  # (M, 8)

        # 6. lookup: voxel index (-1이면 empty)
        neighbor_vi = table[neighbor_linear]  # (M, 8)
        valid_mask = neighbor_vi >= 0         # (M, 8)

        # 7. anchor 중 8개 모두 empty인 것 제거
        has_any = valid_mask.any(dim=-1)      # (M,)
        keep_mask = has_any                   # (M,)

        anchor_points = anchor_points[keep_mask]
        neighbor_vi   = neighbor_vi[keep_mask]
        valid_mask    = valid_mask[keep_mask]
        M_new = anchor_points.shape[0]

        # 8. 벡터화 IDW
        safe_vi = neighbor_vi.clamp(min=0)                                 # (M_new, 8)
        neighbor_centers = voxel_coords[safe_vi]                           # (M_new, 8, 3)
        dist = torch.norm(
            neighbor_centers - anchor_points[:, None, :], dim=-1
        )                                                                   # (M_new, 8)

        dist = dist.masked_fill(~valid_mask, float('inf'))
        w = 1.0 / (dist + 1e-9)
        w = w.masked_fill(~valid_mask, 0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)                        # (M_new, 8)

        out = (w[:, :, None] * voxel_feats[safe_vi]).sum(dim=1)            # (M_new, D)

        return out, keep_mask  # feature (M_new, D)
    

    def intensity_agg(self, feat, intensity):
        feat_i = self.intensity_proj(intensity)
        agg_feat_i = self.intensity_agg_mlp(torch.cat([feat, self.intensity_norm(feat_i)],dim=1))
        return agg_feat_i

    def forward(self, x, batch_idx, mode):
        lidar_points, offset, batch_idx, pose, bbox = x["lidar_points"], x["offset"], x["batch_idx"],  x["pose"], x["bbox"]


        voxelized_points  = self.voxelizer(lidar_points,offset,pose, mode="sphere")

        features = self.feature_extractor(x["ptv3_input"]) 
        grid_coords = features["grid_coord"]   # (V, 3) int
        utonia_feat = features["feat"]         # (V, D)
        feat_coord  = features["coord"] /0.2       # (V, 3)
        feat_offset = features["offset"]       # utonia offset
        sparse_shape = features["sparse_shape"]

        grid_coord_list = split_by_offset(grid_coords, feat_offset)  # List[Tensor(Vi, 3)]
        feat_list       = split_by_offset(utonia_feat, feat_offset)  # List[Tensor(Vi, D)]
        coord_list      = split_by_offset(feat_coord,  feat_offset)  # List[Tensor(Vi, 3)]
        #sparse_shape    = split_by_offset(sparse_shape, feat_offset)
        anchor_list     = voxelized_points["anchor_points"]                 # List[Tensor(Ui, 3)]
        mean_i_list     = voxelized_points["mean_i"]                        # List[Tensor(Ui,)]
        var_i_list      = voxelized_points["var_i"]     

        n_frames = len(anchor_list)

        # 프레임별 batch index
        frame_starts    = torch.cat([torch.tensor([0], device=offset.device), offset[:-1]])
        frame_batch_idx = batch_idx[frame_starts.long()]  # (n_frames,)

        all_feat        = []
        all_intensity   = []
        all_offset      = []
        all_batch       = []
        all_anchor      = []
        all_frame_batch = []
        all_bbox        = []   # List[Tensor(B_f, 7)], 프레임별
        local_frame_counter = {}
        cumsum = 0

        for i in range(n_frames):
            anchor = anchor_list[i]

            anchor_feat, keep_mask = self.utonia_cond(
                anchor, grid_coord_list[i], feat_list[i], coord_list[i], sparse_shape
            )

            mean_i    = mean_i_list[i][keep_mask]
            var_i     = var_i_list[i][keep_mask]
            direction = intensity_dir(anchor, keep_mask)

            b         = frame_batch_idx[i].item()
            local_f   = local_frame_counter.get(b, 0)
            local_frame_counter[b] = local_f + 1

            b_tensor     = torch.full(
                (anchor_feat.shape[0],), b,
                dtype=torch.long, device=anchor.device
            )
            valid_anchor = anchor[keep_mask]

            all_feat.append(anchor_feat)
            all_intensity.append(
                torch.cat([mean_i.unsqueeze(-1), var_i.unsqueeze(-1), direction], dim=1)
            )
            all_batch.append(b_tensor)
            all_anchor.append(valid_anchor)
            all_frame_batch.append(b)
            all_bbox.append(bbox[b][local_f])   # Tensor(B_f, 7)
            cumsum += anchor_feat.shape[0]
            all_offset.append(torch.tensor(cumsum, device=anchor.device))

        all_feat      = torch.cat(all_feat,      dim=0)   # (N_valid, D)
        all_intensity = torch.cat(all_intensity, dim=0)   # (N_valid, 5)
        new_batch     = torch.cat(all_batch,     dim=0)   # (N_valid,)
        new_offset    = torch.stack(all_offset)            # (n_frames,)
        all_anchor    = torch.cat(all_anchor,    dim=0)   # (N_valid, 3)
        frame_batch_idx = torch.tensor(
            all_frame_batch, dtype=torch.long, device=all_feat.device
        )  # (n_frames,)

        agg_feat_i = self.intensity_agg(all_feat, all_intensity)  # (N_valid, C)

        # time_agg: out_feat, out_coord, box_assign 반환
        # box_assign: (N_valid,) -1=background, 0~B_f-1=box index
        out_feat, out_coord, box_assign = self.time_agg(
            agg_feat_i, all_anchor, new_offset,
            frame_batch_idx, x["pose"], all_bbox
        )

        # GS 예측
        gs_raw = self.gs_predictor(out_feat, out_coord)
        # gs_raw: dict of (N_valid, ...) tensors

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

            b_box_assign = box_assign[b_slice]   # (Nb,) -1=bg, 0~B-1=box

            b_gs = {k: v[b_slice] for k, v in gs_raw.items()}
            b_gs["box_assign"] = b_box_assign    # (Nb,) 어떤 bbox에서 나왔는지
            b_gs["coord"]      = out_coord[b_slice]  # (Nb, 3) frame_0 좌표계

            # bg / fg 분리 인덱스
            b_gs["bg_mask"]  = b_box_assign == -1          # (Nb,) bool
            b_gs["fg_masks"] = {
                int(box_i): (b_box_assign == box_i)
                for box_i in b_box_assign[b_box_assign >= 0].unique().tolist()
            }  # {box_id: (Nb,) bool}
            b_gs["frame_bboxes"] = []
            for local_f, global_f in enumerate(frame_indices):
                bbox_f = all_bbox[global_f]   # Tensor(B_f, 7)
                b_gs["frame_bboxes"].append({
                    "center": bbox_f[:, :3],   # (B_f, 3)
                    "size":   bbox_f[:, 3:6],  # (B_f, 3)
                    "yaw":    bbox_f[:, 6],    # (B_f,)
                })
            batch_gaussians.append(b_gs)

        return {
            "gaussians": batch_gaussians,  # List[dict], 배치별
            # batch_gaussians[b] 구조:
            #   "position"     : (Nb, 3)
            #   "opacity"      : (Nb, 1)
            #   "scale"        : (Nb, 2)
            #   "rotation"     : (Nb, 4)
            #   "intensity_sh" : (Nb, 16) -> L =3 임.
            #   "raydrop_sh"   : (Nb, 16)
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


    def forward(self, feat, anchor, new_offset, frame_batch_idx, pose_list, bbox_list):
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

        # 배치별로 처리
        batch_frame_map = {}
        for global_f, b in enumerate(frame_batch_idx.tolist()):
            batch_frame_map.setdefault(int(b), []).append(global_f)

        for b, frame_indices in batch_frame_map.items():
            pose_b = pose_list[b]    # Tensor(V, 4, 4)
            bbox_b = bbox_list[b]    # List[Tensor(B_f, 7)]

            #1. 각 프레임 anchor를 frame_0 좌표계로 변환 
            frame_feats   = []
            frame_coords  = []  # frame_0 좌표계
            frame_g_idx   = []  # global index
            frame_bbox    = []  # 각 프레임의 bbox (sensor frame)

            for local_f, global_f in enumerate(frame_indices):
                prev  = int(new_offset[global_f - 1]) if global_f > 0 and frame_batch_idx[global_f - 1] == b else 0
                end   = int(new_offset[global_f])
                g_idx = torch.arange(prev, end, device=device)

                feat_f   = feat[prev:end]     # (Nf, C)
                anchor_f = anchor[prev:end]   # (Nf, 3) sensor frame
                pose_f    = pose_b[local_f].to(device)   # (4, 4)
                coord_f0  = self.apply_pose(anchor_f, pose_f)  # (Nf, 3) frame_0 좌표계

                frame_feats.append(feat_f)
                frame_coords.append(coord_f0)
                frame_g_idx.append(g_idx)
                frame_bbox.append(bbox_b[local_f].to(device))  # (B_f, 7) sensor frame

            # frame_0 bbox (local_f=0 기준)
            bbox_f0 = frame_bbox[0]   # (B_f0, 7) frame_0의 bbox

            #2. 전체 anchor를 bbox 기준으로 분류 (frame_0 좌표계) 
            all_feat_b   = torch.cat(frame_feats,  dim=0)   # (Nb, C)
            all_coord_b  = torch.cat(frame_coords, dim=0)   # (Nb, 3) frame_0
            all_g_idx_b  = torch.cat(frame_g_idx,  dim=0)   # (Nb,)

            # frame_0 좌표계에서 bbox 분류
            box_assign = self.point_in_box(all_coord_b, bbox_f0)  # (Nb,) -1 or box_idx

            # 3. foreground: bbox 상대 좌표로 변환 
            # 각 프레임의 anchor를 해당 프레임 bbox 중심 기준 offset으로 표현
            all_coord_aligned = all_coord_b.clone()

            if bbox_f0.shape[0] > 0:
                # 프레임별로 처리
                offset_start = 0
                for local_f, (feat_f, coord_f0, g_idx) in enumerate(
                    zip(frame_feats, frame_coords, frame_g_idx)
                ):
                    Nf = feat_f.shape[0]
                    coord_slice = all_coord_b[offset_start:offset_start + Nf]  # frame_0 좌표

                    if local_f == 0:
                        offset_start += Nf
                        continue

                    # 이 프레임의 bbox (sensor frame)
                    bbox_fi = frame_bbox[local_f]   # (B_fi, 7)

                    # frame_i anchor의 sensor frame 좌표
                    prev = int(new_offset[frame_indices[local_f] - 1]) if frame_indices[local_f] > 0 and frame_batch_idx[frame_indices[local_f] - 1] == b else 0
                    end  = int(new_offset[frame_indices[local_f]])
                    anchor_fi_sensor = anchor[prev:end]   # (Nf, 3) sensor frame

                    # sensor frame에서 bbox 분류
                    box_assign_fi = self.point_in_box(anchor_fi_sensor, bbox_fi)  # (Nf,)

                    for box_i in box_assign_fi[box_assign_fi >= 0].unique().tolist():
                        box_i    = int(box_i)
                        fg_mask  = box_assign_fi == box_i

                        if box_i >= bbox_f0.shape[0]:
                            offset_start += Nf
                            continue

                        # frame_i bbox 중심 (sensor frame)
                        center_fi = bbox_fi[box_i, :3]   # (3,)
                        center_f0 = bbox_f0[box_i, :3].to(device)  # (3,)
                        center_fi_in_f0 = self.apply_pose(
                            center_fi.unsqueeze(0),
                            pose_b[local_f].to(device)
                        ).squeeze(0)  # (3,)

                        # offset 보정: frame_i bbox → frame_0 bbox
                        delta = center_f0 - center_fi_in_f0  # (3,)

                        fg_global = offset_start + fg_mask.nonzero(as_tuple=True)[0]
                        all_coord_aligned[fg_global] = coord_slice[fg_mask] + delta

                    offset_start += Nf

            # 4. background / foreground KNN attention 
            bg_mask  = box_assign == -1   # (Nb,)

            # background
            if bg_mask.any():
                bg_feat  = all_feat_b[bg_mask]          # (N_bg, C)
                bg_coord = all_coord_aligned[bg_mask]   # (N_bg, 3)
                bg_g_idx = all_g_idx_b[bg_mask]

                bg_out = self.bg_attn(bg_feat, bg_coord, k=self.k_bg)
                out_feat[bg_g_idx]  = bg_out
                out_coord[bg_g_idx] = bg_coord

            # foreground: bbox별
            for box_i in box_assign[box_assign >= 0].unique().tolist():
                box_i   = int(box_i)
                fg_mask = box_assign == box_i

                fg_feat  = all_feat_b[fg_mask]
                fg_coord = all_coord_aligned[fg_mask]
                fg_g_idx = all_g_idx_b[fg_mask]

                fg_out = self.fg_attn(fg_feat, fg_coord, k=self.k_fg)
                out_feat[fg_g_idx]  = fg_out
                out_coord[fg_g_idx] = fg_coord

        return out_feat, out_coord   # (N_valid, C), (N_valid, 3)