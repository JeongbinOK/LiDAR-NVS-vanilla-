"""Cartesian voxelization for dynamic instance bbox-local point clouds.

Each tracked dynamic instance has its bbox-local canonical point cloud
(`decompose_scene` in `models/geometry/decomposition.py`). Within an instance's
canonical space (origin = bbox center, +x = forward), we voxelize with a
**cartesian** grid sized to the bbox: `voxel_size = max(bbox_dim) / 8` so each
instance gets ~5³ = 125 voxels regardless of object size (plan §-0).

Returns the same structured output as `SphericalVoxelizer` so downstream
consumers (`VoxelAnchorBuilder`-style pipelines) can reuse the schema.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .spherical_voxel import SphericalVoxelOutput


class CartesianVoxelizer:
    """Cartesian voxelization in instance-canonical space.

    Args:
        voxel_size: float — bin size in metres. Typical: max(bbox_dim) / 8.
        query_voxel_min_points: voxels with fewer than this many points do not
            produce query positions.
        k_min: backward-compatible alias for query_voxel_min_points.
        max_extent: half-range (m) used to offset cartesian indices so the
            voxel hash stays non-negative. Should comfortably bound the bbox.
    """

    def __init__(
        self,
        voxel_size: float,
        query_voxel_min_points: int = 1,
        k_min: int | None = None,
        max_extent: float = 20.0,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        self.voxel_size = float(voxel_size)
        if k_min is not None:
            query_voxel_min_points = k_min
        self.query_voxel_min_points = int(query_voxel_min_points)
        self.max_extent = float(max_extent)
        self.n_grid = int(math.ceil(2 * self.max_extent / self.voxel_size)) + 1

    def _bin_indices(self, xyz: Tensor) -> Tensor:
        idx = torch.floor((xyz + self.max_extent) / self.voxel_size).long()
        idx = idx.clamp(min=0, max=self.n_grid - 1)
        return idx[..., 0] * (self.n_grid * self.n_grid) + idx[..., 1] * self.n_grid + idx[..., 2]

    @torch.no_grad()
    def __call__(
        self,
        xyz: Tensor,
        intensity: Tensor | None = None,
        src: Tensor | None = None,
    ) -> SphericalVoxelOutput:
        """Voxelize one canonical-space point cloud.

        Args:
            xyz:       [N, 3]  canonical (bbox-local) coords.
            intensity: [N]     optional, normalised in [0, 1].
            src:       [N]     optional, 0 for frame0 / 1 for frame1
                               (or any [0,1] continuous time scalar).

        Returns SphericalVoxelOutput with M valid voxels.
        """
        if xyz.dim() != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [N, 3], got {tuple(xyz.shape)}")
        device = xyz.device
        dtype = xyz.dtype
        N = xyz.shape[0]

        if N == 0:
            empty1 = torch.zeros((0,), device=device, dtype=dtype)
            return SphericalVoxelOutput(
                query_xyz=torch.zeros((0, 3), device=device, dtype=dtype),
                n_points=torch.zeros((0,), device=device, dtype=torch.long),
                i_mean=empty1,
                i_std=empty1,
                src_ratio=empty1,
                point_voxel=torch.zeros((0,), device=device, dtype=torch.long),
                voxel_hash=torch.zeros((0,), device=device, dtype=torch.long),
            )

        voxel_hash = self._bin_indices(xyz)
        unique_hash, inverse, counts = torch.unique(
            voxel_hash, return_inverse=True, return_counts=True
        )
        U = unique_hash.shape[0]

        cnt_f = counts.to(dtype=dtype)
        sum_xyz = torch.zeros((U, 3), device=device, dtype=dtype)
        sum_xyz.index_add_(0, inverse, xyz)
        mean_xyz = sum_xyz / cnt_f.unsqueeze(-1)

        if intensity is not None:
            i = intensity.to(dtype=dtype)
            sum_i = torch.zeros(U, device=device, dtype=dtype)
            sum_i.index_add_(0, inverse, i)
            sum_i2 = torch.zeros(U, device=device, dtype=dtype)
            sum_i2.index_add_(0, inverse, i * i)
            mean_i = sum_i / cnt_f
            var_i = (sum_i2 / cnt_f - mean_i * mean_i).clamp(min=0.0)
            std_i = var_i.sqrt()
        else:
            mean_i = torch.zeros(U, device=device, dtype=dtype)
            std_i = torch.zeros(U, device=device, dtype=dtype)

        if src is not None:
            s = src.to(dtype=dtype)
            sum_s = torch.zeros(U, device=device, dtype=dtype)
            sum_s.index_add_(0, inverse, s)
            src_ratio_all = sum_s / cnt_f
        else:
            src_ratio_all = torch.zeros(U, device=device, dtype=dtype)

        valid = counts >= self.query_voxel_min_points
        keep = valid.nonzero(as_tuple=False).squeeze(-1)
        M = keep.numel()

        old_to_new = torch.full((U,), -1, dtype=torch.long, device=device)
        old_to_new[keep] = torch.arange(M, device=device, dtype=torch.long)
        point_voxel = old_to_new[inverse]

        return SphericalVoxelOutput(
            query_xyz=mean_xyz.index_select(0, keep),
            n_points=counts.index_select(0, keep),
            i_mean=mean_i.index_select(0, keep),
            i_std=std_i.index_select(0, keep),
            src_ratio=src_ratio_all.index_select(0, keep),
            point_voxel=point_voxel,
            voxel_hash=unique_hash.index_select(0, keep),
        )
