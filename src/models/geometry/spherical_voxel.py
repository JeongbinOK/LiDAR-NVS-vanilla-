"""LiDAR_0-origin spherical voxelization for static point pool.

Bins points by (azimuth, elevation, range) in LiDAR_0 frame and aggregates
per-voxel statistics. Used as the first step of voxel-anchored QGS:

    static points  -->  SphericalVoxelizer  -->  per-voxel query positions
                                                  + per-voxel feature stats
    (point cloud)        (this module)            (next: k-NN + quad_fit)

Paradigm decisions (plan §-0):
  * Static-only — dynamic instances handled by a separate bbox-cartesian voxelizer.
  * Query voxelization is separate from Gaussian filtering. Voxels with at
    least `query_voxel_min_points` points produce query positions; the
    downstream anchor builder applies the k_eff >= k_min Gaussian filter.
  * The voxel-mean coordinate is *not* the gaussian center. It is the k-NN query
    position; the gaussian center is the quadric tangent point produced by
    `fit_local_quadrics` downstream.

Default voxel size (Phase A 통계 결과로 확정):
    Δφ = 3°, Δθ = 4°, Δr = 3 m  →  median 18 pts / valid voxel, 56% valid ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SphericalVoxelOutput:
    """Per-valid-voxel aggregates."""

    query_xyz: Tensor      # [M, 3]   voxel mean position (k-NN query anchor)
    n_points: Tensor       # [M]      long, point count per voxel
    i_mean: Tensor         # [M]      float, mean intensity (zero if intensity=None)
    i_std: Tensor          # [M]      float, std intensity (zero if intensity=None)
    src_ratio: Tensor      # [M]      float, LiDAR_1 share in [0, 1] (zero if src=None)
    point_voxel: Tensor    # [N]      long, voxel index in [0, M) or -1 if filtered
    voxel_hash: Tensor     # [M]      long, raw spherical bin hash (debug / dedup)


class SphericalVoxelizer:
    """Spherical voxelization in LiDAR_0 sensor frame.

    Args:
        dphi_deg, dtheta_deg, dr_m: bin sizes.
        query_voxel_min_points: voxels with fewer than this many points do not
            produce query positions.
        k_min: backward-compatible alias for query_voxel_min_points.
    """

    def __init__(
        self,
        dphi_deg: float = 3.0,
        dtheta_deg: float = 4.0,
        dr_m: float = 3.0,
        query_voxel_min_points: int = 1,
        k_min: int | None = None,
    ) -> None:
        self.dphi = math.radians(float(dphi_deg))
        self.dtheta = math.radians(float(dtheta_deg))
        self.dr = float(dr_m)
        if k_min is not None:
            query_voxel_min_points = k_min
        self.query_voxel_min_points = int(query_voxel_min_points)
        self.n_phi = int(math.ceil(2 * math.pi / self.dphi)) + 1
        self.n_theta = int(math.ceil(math.pi / self.dtheta)) + 1
        self._stride_theta = self.n_phi
        self._stride_r = self.n_phi * self.n_theta

    def _bin_indices(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        r = xyz.norm(dim=-1)
        safe_r = r.clamp(min=1e-6)
        phi = torch.atan2(xyz[..., 1], xyz[..., 0])                 # [-π, π]
        theta = torch.asin((xyz[..., 2] / safe_r).clamp(-1.0, 1.0)) # [-π/2, π/2]

        iphi = torch.floor((phi + math.pi) / self.dphi).long()
        itheta = torch.floor((theta + math.pi / 2) / self.dtheta).long()
        ir = torch.floor(r / self.dr).long()
        voxel_hash = ir * self._stride_r + itheta * self._stride_theta + iphi
        return voxel_hash, r

    @torch.no_grad()
    def __call__(
        self,
        xyz: Tensor,
        intensity: Tensor | None = None,
        src: Tensor | None = None,
    ) -> SphericalVoxelOutput:
        """Voxelize a single point cloud (no batch dim).

        Args:
            xyz:       [N, 3]  static points in LiDAR_0 frame.
            intensity: [N]     optional, normalised in [0, 1].
            src:       [N]     optional, 0 for LiDAR_0 / 1 for LiDAR_1.

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

        voxel_hash, _r = self._bin_indices(xyz)
        unique_hash, inverse, counts = torch.unique(
            voxel_hash, return_inverse=True, return_counts=True
        )
        U = unique_hash.shape[0]

        # ----- Aggregate per-voxel sums (index_add for portability — no torch_scatter)
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

        # ----- Query filter: keep voxels that should produce k-NN queries.
        valid = counts >= self.query_voxel_min_points
        keep = valid.nonzero(as_tuple=False).squeeze(-1)
        M = keep.numel()

        old_to_new = torch.full((U,), -1, dtype=torch.long, device=device)
        old_to_new[keep] = torch.arange(M, device=device, dtype=torch.long)
        point_voxel = old_to_new[inverse]  # [N], -1 if filtered

        return SphericalVoxelOutput(
            query_xyz=mean_xyz.index_select(0, keep),
            n_points=counts.index_select(0, keep),
            i_mean=mean_i.index_select(0, keep),
            i_std=std_i.index_select(0, keep),
            src_ratio=src_ratio_all.index_select(0, keep),
            point_voxel=point_voxel,
            voxel_hash=unique_hash.index_select(0, keep),
        )
