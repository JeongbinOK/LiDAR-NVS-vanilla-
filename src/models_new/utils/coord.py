import math
from dataclasses import dataclass

import torch
from torch import Tensor
import pytorch_lightning as L

def ptv3_2_batch(points, offsets, batch_idx):
    # points (N,3), offset(B*V_i), batch_idx(N)
    sizes = torch.diff(torch.cat([torch.tensor([0], device=offset.device), offset]))
    frames = torch.split(points, sizes.tolist())

    frame_batch_idx = []
    start = 0
    for end in offset:
        frame_batch_idx.append(batch_idx[start].item())
        start = end

    batch_frames = []
    current_batch = frame_batch_idx[0]
    temp = []

    for f, b in zip(frames, frame_batch_idx): #루프 frame num만 돔 
        if b != current_batch:
            batch_frames.append(temp)
            temp = []
            current_batch = b
        temp.append(f)
    batch_frames.append(temp)

    #batch_frames: List[List[Tensor]]  # B , V_i , (Ni, 3)
    return batch_frames    

def batch_2_ptv3(batch):
    #B, V_i, (Ni,3)
    all_points = []
    offsets = []
    batch_indices = []

    cum_sum = 0
    frame_count = 0

    for b, frames in enumerate(batch):
        for f in frames:
            n = f.shape[0]

            all_points.append(f)
            cum_sum += n
            offsets.append(cum_sum)

            batch_indices.append(
                torch.full((n,), b, dtype=torch.long, device=f.device)
            )

            frame_count += 1

    points = torch.cat(all_points, dim=0)
    offset = torch.tensor(offsets, device=points.device)
    batch_idx = torch.cat(batch_indices, dim=0)

    return points, offset, batch_idx




# class SphericalVoxelOutput:
#     """Per-valid-voxel aggregates."""

#     query_xyz: Tensor      # [M, 3]   voxel mean position (k-NN query anchor)
#     n_points: Tensor       # [M]      long, point count per voxel
#     i_mean: Tensor         # [M]      float, mean intensity (zero if intensity=None)
#     i_std: Tensor          # [M]      float, std intensity (zero if intensity=None)
#     src_ratio: Tensor      # [M]      float, LiDAR_1 share in [0, 1] (zero if src=None)
#     point_voxel: Tensor    # [N]      long, voxel index in [0, M) or -1 if filtered
#     voxel_hash: Tensor     # [M]      long, raw spherical bin hash (debug / dedup)

class Voxelizerargs:
    dphi_deg= 3.0
    dtheta_deg= 4.0
    dr_m= 3.0
    max_radius_m = 100.0
    query_voxel_min_points= 1
    max_extent = 20.0
    voxel_size = 0.004  
class Voxelizer(L.LightningModule):
    def __init__(
        # self,
        # dphi_deg: float = 3.0,
        # dtheta_deg: float = 4.0,
        # dr_m: float = 3.0,
        # query_voxel_min_points: int = 1,
        # k_min: int | None = None,
        self,
        cfg,
        max_frames
    ): 
        super().__init__()
        self.cfg = cfg
        self.voxel_cfg = Voxelizerargs
        self.dphi = math.radians(self.voxel_cfg.dphi_deg)
        self.dtheta = math.radians(self.voxel_cfg.dtheta_deg)
        self.dr = self.voxel_cfg.dr_m
        self.max_radius = self.voxel_cfg.max_radius_m
        # if k_min is not None:
        #     query_voxel_min_points = k_min
        self.query_voxel_min_points = self.voxel_cfg.query_voxel_min_points


        #For Spherical
        self.n_phi = math.ceil(2 * math.pi / self.dphi) + 1
        self.n_theta = math.ceil(math.pi / self.dtheta) + 1
        self.n_r = math.ceil(self.max_radius / self.dr)
        self._stride_theta = self.n_phi # &&& 3도 + 4도 + 3m -> 실험으로 증명 말고 깔끔하게 얼버부리기?
        self._stride_r = self.n_phi * self.n_theta
        self._sphere_bins_per_frame = self.n_r * self._stride_r
    
        #for Cartesian
        self.voxel_size = self.voxel_cfg.voxel_size # &&& 3도 + 4도 + 3m -> 실험으로 증명 말고 깔끔하게 얼버부리기?
        self.max_extent = self.voxel_cfg.max_extent
        self.n_grid = math.ceil(2 * self.max_extent / self.voxel_size) + 1 

    def sphere_bin_indices(self, xyz):
        r = xyz.norm(dim=-1)
        safe_r = r.clamp(min=1e-6)
        phi = torch.atan2(xyz[..., 1], xyz[..., 0])                 # [-π, π]
        theta = torch.asin((xyz[..., 2] / safe_r).clamp(-1.0, 1.0)) # [-π/2, π/2]

        iphi = torch.floor((phi + math.pi) / self.dphi).long()
        itheta = torch.floor((theta + math.pi / 2) / self.dtheta).long()
        ir = torch.floor(r / self.dr).long()
        voxel_hash = ir * self._stride_r + itheta * self._stride_theta + iphi
        return voxel_hash, r

    def _empty_output(self, n_frames, device, dtype):
        empty_xyz = torch.zeros((0, 3), device=device, dtype=dtype)
        empty_scalar = torch.zeros((0,), device=device, dtype=dtype)
        empty_hash = torch.zeros((0,), device=device, dtype=torch.long)
        return {
            "anchor_points": [empty_xyz for _ in range(n_frames)],
            "counts": [empty_scalar for _ in range(n_frames)],
            "mean_i": [empty_scalar for _ in range(n_frames)],
            "var_i": [empty_scalar for _ in range(n_frames)],
            "voxel_hash": [empty_hash for _ in range(n_frames)]
        }

    def _forward_sphere_dense(self, points, offset):
        dtype = points.dtype
        device = points.device
        n_frames = int(offset.numel())
        sizes = torch.diff(
            torch.cat([torch.zeros(1, device=device, dtype=offset.dtype), offset])
        )
        if points.shape[0] == 0:
            return self._empty_output(n_frames, device, dtype)

        voxel_hash, r = self.sphere_bin_indices(points[:, :3])
        frame_idx = torch.repeat_interleave(
            torch.arange(n_frames, device=device, dtype=torch.long),
            sizes.long(),
        )

        valid = (r < self.max_radius) & (voxel_hash >= 0) & (voxel_hash < self._sphere_bins_per_frame)
        if not valid.any():
            return self._empty_output(n_frames, device, dtype)

        valid_points = points[valid]
        valid_frame_idx = frame_idx[valid]
        valid_hash = voxel_hash[valid]
        dense_key = valid_frame_idx * self._sphere_bins_per_frame + valid_hash
        total_bins = n_frames * self._sphere_bins_per_frame

        counts = torch.zeros(total_bins, device=device, dtype=dtype)
        ones = torch.ones_like(dense_key, dtype=dtype)
        counts.scatter_add_(0, dense_key, ones)

        sum_xyz = torch.zeros((total_bins, 3), device=device, dtype=dtype)
        sum_xyz.index_add_(0, dense_key, valid_points[:, :3])

        intensity = valid_points[:, 3]
        sum_i = torch.zeros(total_bins, device=device, dtype=dtype)
        sum_i.scatter_add_(0, dense_key, intensity)
        sum_i2 = torch.zeros(total_bins, device=device, dtype=dtype)
        sum_i2.scatter_add_(0, dense_key, intensity * intensity)

        occupied = counts > 0
        occupied_idx = occupied.nonzero(as_tuple=True)[0]
        cnt_f = counts[occupied_idx]
        mean_xyz = sum_xyz[occupied_idx] / cnt_f.unsqueeze(-1)
        mean_i = sum_i[occupied_idx] / cnt_f
        var_i = (sum_i2[occupied_idx] / cnt_f - mean_i * mean_i).clamp(min=0.0)

        occupied_frame = torch.div(occupied_idx, self._sphere_bins_per_frame, rounding_mode="floor")
        occupied_hash = occupied_idx - occupied_frame * self._sphere_bins_per_frame
        frame_voxel_counts = torch.bincount(occupied_frame, minlength=n_frames).tolist()

        return {
            "anchor_points": list(torch.split(mean_xyz, frame_voxel_counts, dim=0)),
            "counts": list(torch.split(cnt_f, frame_voxel_counts, dim=0)),
            "mean_i": list(torch.split(mean_i, frame_voxel_counts, dim=0)),
            "var_i": list(torch.split(var_i, frame_voxel_counts, dim=0)),
            "voxel_hash": list(torch.split(occupied_hash, frame_voxel_counts, dim=0)),
        }

    def cart_bin_indices(self, xyz: Tensor) -> Tensor:
        idx = torch.floor((xyz + self.max_extent) / self.voxel_size).long()
        idx = idx.clamp(min=0, max=self.n_grid - 1)
        return idx[..., 0] * (self.n_grid * self.n_grid) + idx[..., 1] * self.n_grid + idx[..., 2]


    def forward(self, points, offset, pose, mode="sphere"):
        #순서: b1_f1, b1_f2, b2_f1, b2_f2, ... ,bn_fn
        dtype = points.dtype
        device = points.device
        offset = offset.to(device=device)
        n_frames = int(offset.numel())
        if mode == "sphere":
            return self._forward_sphere_dense(points, offset)

        voxel_hash, _ = self.cart_bin_indices(points[:,:3]) #N
        sizes = torch.diff(
            torch.cat([torch.zeros(1, device=device, dtype=offset.dtype), offset])
        )
        if points.shape[0] == 0:
            return self._empty_output(n_frames, device, dtype)

        frame_idx = torch.repeat_interleave(
            torch.arange(n_frames, device=device, dtype=torch.long),
            sizes.long(),
        )
        hash_stride = voxel_hash.max().clamp_min(0) + 1
        frame_voxel_hash = frame_idx * hash_stride + voxel_hash

        unique_frame_hash, inverse, counts = torch.unique(
            frame_voxel_hash,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        U = unique_frame_hash.shape[0]
        cnt_f = counts.to(dtype=dtype)

        sum_xyz = torch.zeros((U, 3), device=device, dtype=dtype)
        sum_xyz.index_add_(0, inverse, points[:, :3])
        mean_xyz = sum_xyz / cnt_f.unsqueeze(-1)

        i = points[:, 3]
        sum_i = torch.zeros(U, device=device, dtype=dtype)
        sum_i.index_add_(0, inverse, i)
        sum_i2 = torch.zeros(U, device=device, dtype=dtype)
        sum_i2.index_add_(0, inverse, i * i)
        mean_i = sum_i / cnt_f
        var_i = (sum_i2 / cnt_f - mean_i * mean_i).clamp(min=0.0)

        unique_frame_idx = torch.div(unique_frame_hash, hash_stride, rounding_mode="floor")
        unique_hash = unique_frame_hash - unique_frame_idx * hash_stride
        frame_voxel_counts = torch.bincount(unique_frame_idx, minlength=n_frames).tolist()

        anchor_points_list = list(torch.split(mean_xyz, frame_voxel_counts, dim=0))
        counts_list = list(torch.split(cnt_f, frame_voxel_counts, dim=0))
        intensity_mean_list = list(torch.split(mean_i, frame_voxel_counts, dim=0))
        intensity_var_list = list(torch.split(var_i, frame_voxel_counts, dim=0))
        voxel_hash_list = list(torch.split(unique_hash, frame_voxel_counts, dim=0))
        return {
            "anchor_points": anchor_points_list,
            "counts": counts_list,
            "mean_i": intensity_mean_list,
            "var_i": intensity_var_list,
            "voxel_hash": voxel_hash_list  
        }

    # def __call__(
    #     self,
    #     xyz: Tensor,
    #     intensity: Tensor | None = None,
    #     src: Tensor | None = None,
    # ) -> SphericalVoxelOutput:
    #     """Voxelize a single point cloud (no batch dim).

    #     Args:
    #         xyz:       [N, 3]  static points in LiDAR_0 frame.
    #         intensity: [N]     optional, normalised in [0, 1].
    #         src:       [N]     optional, 0 for LiDAR_0 / 1 for LiDAR_1.

    #     Returns SphericalVoxelOutput with M valid voxels.
    #     """
    #     if xyz.dim() != 2 or xyz.shape[-1] != 3:
    #         raise ValueError(f"xyz must be [N, 3], got {tuple(xyz.shape)}")
    #     device = xyz.device
    #     dtype = xyz.dtype
    #     N = xyz.shape[0]
    #     if N == 0:
    #         empty1 = torch.zeros((0,), device=device, dtype=dtype)
    #         return SphericalVoxelOutput(
    #             query_xyz=torch.zeros((0, 3), device=device, dtype=dtype),
    #             n_points=torch.zeros((0,), device=device, dtype=torch.long),
    #             i_mean=empty1,
    #             i_std=empty1,
    #             src_ratio=empty1,
    #             point_voxel=torch.zeros((0,), device=device, dtype=torch.long),
    #             voxel_hash=torch.zeros((0,), device=device, dtype=torch.long),
    #         )

    #     voxel_hash, _r = self._bin_indices(xyz)
    #     unique_hash, inverse, counts = torch.unique(
    #         voxel_hash, return_inverse=True, return_counts=True
    #     )
    #     U = unique_hash.shape[0]

    #     # ----- Aggregate per-voxel sums (index_add for portability — no torch_scatter)
    #     cnt_f = counts.to(dtype=dtype)
    #     sum_xyz = torch.zeros((U, 3), device=device, dtype=dtype)
    #     sum_xyz.index_add_(0, inverse, xyz)
    #     mean_xyz = sum_xyz / cnt_f.unsqueeze(-1)

    #     if intensity is not None:
    #         i = intensity.to(dtype=dtype)
    #         sum_i = torch.zeros(U, device=device, dtype=dtype)
    #         sum_i.index_add_(0, inverse, i)
    #         sum_i2 = torch.zeros(U, device=device, dtype=dtype)
    #         sum_i2.index_add_(0, inverse, i * i)
    #         mean_i = sum_i / cnt_f
    #         var_i = (sum_i2 / cnt_f - mean_i * mean_i).clamp(min=0.0)
    #         std_i = var_i.sqrt()
    #     else:
    #         mean_i = torch.zeros(U, device=device, dtype=dtype)
    #         std_i = torch.zeros(U, device=device, dtype=dtype)

    #     if src is not None:
    #         s = src.to(dtype=dtype)
    #         sum_s = torch.zeros(U, device=device, dtype=dtype)
    #         sum_s.index_add_(0, inverse, s)
    #         src_ratio_all = sum_s / cnt_f
    #     else:
    #         src_ratio_all = torch.zeros(U, device=device, dtype=dtype)

    #     # ----- Query filter: keep voxels that should produce k-NN queries.
    #     valid = counts >= self.query_voxel_min_points
    #     keep = valid.nonzero(as_tuple=False).squeeze(-1)
    #     M = keep.numel()

    #     old_to_new = torch.full((U,), -1, dtype=torch.long, device=device)
    #     old_to_new[keep] = torch.arange(M, device=device, dtype=torch.long)
    #     point_voxel = old_to_new[inverse]  # [N], -1 if filtered

    #     return SphericalVoxelOutput(
    #         query_xyz=mean_xyz.index_select(0, keep),
    #         n_points=counts.index_select(0, keep),
    #         i_mean=mean_i.index_select(0, keep),
    #         i_std=std_i.index_select(0, keep),
    #         src_ratio=src_ratio_all.index_select(0, keep),
    #         point_voxel=point_voxel,
    #         voxel_hash=unique_hash.index_select(0, keep),
    #     )
