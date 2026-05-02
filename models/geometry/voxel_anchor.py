"""Voxel-anchored quadric Gaussian builders (static + dynamic).

Pipeline (plan §-0):
                 voxelize  →  k-NN  →  quad_fit  →  2-stage filter  →  22-ch token

Static branch (`VoxelAnchorBuilder`):
    LiDAR_0-origin spherical voxel, candidates = whole static cloud.

Dynamic branch (`DynamicVoxelAnchorBuilder`):
    Per-instance bbox-local cartesian voxel (size = max(bbox_dim) / 8),
    candidates = same instance only (instance-separated k-NN, locked design).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .cartesian_voxel import CartesianVoxelizer
from .knn_radius import hybrid_radius_knn
from .quadric_fit import fit_local_quadrics
from .spherical_voxel import SphericalVoxelizer, SphericalVoxelOutput


_LOG_EPS = 1e-6
_TOKEN_CENTER_SCALE_M = 50.0
_TOKEN_LOG_SCALE_SPAN = 4.0
_TOKEN_KAPPA_SCALE = 5.0
_TOKEN_QUALITY_SCALE = 2.0
_TOKEN_ANISO_SCALE = 2.0


@dataclass(frozen=True)
class VoxelAnchorOutput:
    """Per-anchor (post-filter) tensors."""

    c_init: Tensor          # [M', 3]
    R_init: Tensor          # [M', 3, 3]
    s_init: Tensor          # [M', 3]
    kappa1: Tensor          # [M']
    kappa2: Tensor          # [M']
    fit_quality: Tensor     # [M', 4]
    tangent_aniso: Tensor   # [M']
    curvature_aniso: Tensor # [M']
    n_points: Tensor        # [M']
    i_mean: Tensor          # [M']
    i_std: Tensor           # [M']
    src_ratio: Tensor       # [M']
    token: Tensor           # [M', 22]
    use_geom_init: Tensor   # [M']
    k_eff: Tensor           # [M']
    diagnostics: dict       # query/filter counts for this builder call


def _make_empty(
    device: torch.device,
    dtype: torch.dtype,
    diagnostics: dict | None = None,
    token_width: int = 22,
) -> VoxelAnchorOutput:
    z1 = torch.zeros((0,), device=device, dtype=dtype)
    z3 = torch.zeros((0, 3), device=device, dtype=dtype)
    return VoxelAnchorOutput(
        c_init=z3,
        R_init=torch.zeros((0, 3, 3), device=device, dtype=dtype),
        s_init=z3,
        kappa1=z1,
        kappa2=z1,
        fit_quality=torch.zeros((0, 4), device=device, dtype=dtype),
        tangent_aniso=z1,
        curvature_aniso=z1,
        n_points=torch.zeros((0,), device=device, dtype=torch.long),
        i_mean=z1,
        i_std=z1,
        src_ratio=z1,
        token=torch.zeros((0, token_width), device=device, dtype=dtype),
        use_geom_init=torch.zeros((0,), device=device, dtype=torch.bool),
        k_eff=torch.zeros((0,), device=device, dtype=torch.long),
        diagnostics=diagnostics or {
            "query_voxels": 0,
            "k_eff_pass": 0,
            "residual_pass": 0,
            "final_anchors": 0,
            "fallback_reason": "empty",
        },
    )


def _normalize_anchor_token_parts(
    *,
    c_init: Tensor,
    R_6d: Tensor,
    log_s: Tensor,
    kappa: Tensor,
    fit_quality: Tensor,
    aniso: Tensor,
    i_stat: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Normalize only PTv3 token inputs; raw QGS init tensors stay unchanged."""
    c_token = (c_init / _TOKEN_CENTER_SCALE_M).clamp(-2.0, 2.0)
    r_token = R_6d.clamp(-1.0, 1.0)
    log_s_token = (log_s / _TOKEN_LOG_SCALE_SPAN).clamp(-2.0, 2.0)
    kappa_token = (kappa / _TOKEN_KAPPA_SCALE).clamp(-2.0, 2.0)
    quality_token = (fit_quality / _TOKEN_QUALITY_SCALE).clamp(0.0, 2.0)
    aniso_token = (aniso / _TOKEN_ANISO_SCALE).clamp(0.0, 2.0)
    intensity_token = i_stat.clamp(0.0, 1.0)
    return (
        c_token,
        r_token,
        log_s_token,
        kappa_token,
        quality_token,
        aniso_token,
        intensity_token,
    )


def build_anchor_token(
    *,
    c_init: Tensor,
    R_init: Tensor,
    s_init: Tensor,
    kappa1: Tensor,
    kappa2: Tensor,
    fit_quality: Tensor,
    tangent_aniso: Tensor,
    curvature_aniso: Tensor,
    i_mean: Tensor,
    i_std: Tensor,
    token_variant: str,
) -> Tensor:
    """Build the normalized PTv3 anchor token while preserving raw init tensors."""
    R_6d = torch.cat([R_init[..., 0], R_init[..., 1]], dim=-1)
    log_s = torch.log(s_init.abs().clamp(min=_LOG_EPS))
    kappa = torch.stack([kappa1, kappa2], dim=-1)
    aniso = torch.stack([tangent_aniso, curvature_aniso], dim=-1)
    i_stat = torch.stack([i_mean, i_std], dim=-1)
    token_fit_quality = fit_quality
    token_i_stat = i_stat
    if token_variant == "no_fit_quality":
        token_fit_quality = torch.zeros_like(token_fit_quality)
    elif token_variant == "no_intensity_stats":
        token_i_stat = torch.zeros_like(token_i_stat)
    elif token_variant not in {"full", "with_normal"}:
        raise ValueError(f"unsupported anchor token_variant {token_variant!r}")

    token_parts = list(_normalize_anchor_token_parts(
        c_init=c_init,
        R_6d=R_6d,
        log_s=log_s,
        kappa=kappa,
        fit_quality=token_fit_quality,
        aniso=aniso,
        i_stat=token_i_stat,
    ))
    if token_variant == "with_normal":
        token_parts.append(R_init[..., 2].clamp(-1.0, 1.0))
    token = torch.cat(token_parts, dim=-1)
    expected_width = 25 if token_variant == "with_normal" else 22
    assert token.shape[-1] == expected_width, (
        f"token width {token.shape[-1]} != {expected_width}"
    )
    return token


def _quad_fit_and_token(
    vox: SphericalVoxelOutput,
    candidates_xyz: Tensor,
    *,
    k_min: int,
    k_target: int,
    residual_threshold: float,
    filter_mode: str,
    planarity_threshold: float,
    token_variant: str,
    knn_chunk_size: int,
    r_max_fn: Callable[[Tensor], Tensor],
) -> VoxelAnchorOutput:
    """k-NN + quad_fit + 2-stage filter + 22-ch token build."""
    device = candidates_xyz.device
    dtype = candidates_xyz.dtype
    token_width = 25 if token_variant == "with_normal" else 22

    knn = hybrid_radius_knn(
        points=vox.query_xyz,
        candidates=candidates_xyz,
        k_target=k_target,
        r_max_fn=r_max_fn,
        k_min=k_min,
        chunk_size=knn_chunk_size,
    )
    idx_knn = knn["idx"].clamp(min=0)
    nbrs = candidates_xyz[idx_knn]                  # [M, K, 3]
    k_eff = knn["k_eff"]
    k_eff_mask = k_eff >= k_min

    geom = fit_local_quadrics(
        points=vox.query_xyz.unsqueeze(0),
        neighbors=nbrs.unsqueeze(0),
        k_eff=k_eff.unsqueeze(0),
        k_min=k_min,
        k_target=k_target,
    )
    c_init = geom["c_init"][0]
    R_init = geom["R_init"][0]
    s_init = geom["s_init"][0]
    fit_quality = geom["fit_quality"][0]
    use_geom = geom["use_geom_init"][0]
    kappa1 = geom["kappa1_init"][0]
    kappa2 = geom["kappa2_init"][0]
    tangent_aniso = geom["tangent_aniso"][0]
    curvature_aniso = geom["curvature_aniso"][0]

    residual = fit_quality[:, 0]
    residual_mask = residual < residual_threshold
    if filter_mode == "residual":
        filter_mask = residual_mask
    elif filter_mode == "residual_planarity":
        planarity = fit_quality[:, 1]
        filter_mask = residual_mask & (planarity < planarity_threshold)
    else:
        raise ValueError(f"unsupported anchor filter_mode {filter_mode!r}")
    pass_mask = k_eff_mask & filter_mask
    diagnostics = {
        "query_voxels": int(vox.query_xyz.shape[0]),
        "k_eff_pass": int(k_eff_mask.sum().item()),
        "residual_pass": int((k_eff_mask & residual_mask).sum().item()),
        "final_anchors": int(pass_mask.sum().item()),
        "use_geom_init_pass": 0,
        "filter_mode": filter_mode,
        "token_variant": token_variant,
        "fallback_reason": None,
    }
    keep = pass_mask.nonzero(as_tuple=False).squeeze(-1)
    if keep.numel() == 0:
        diagnostics["fallback_reason"] = (
            "k_eff_lt_k_min" if diagnostics["k_eff_pass"] == 0 else "residual_ge_threshold"
        )
        return _make_empty(device, dtype, diagnostics, token_width=token_width)

    c_init = c_init.index_select(0, keep)
    R_init = R_init.index_select(0, keep)
    s_init = s_init.index_select(0, keep)
    fit_quality = fit_quality.index_select(0, keep)
    use_geom = use_geom.index_select(0, keep)
    k_eff = k_eff.index_select(0, keep)
    kappa1 = kappa1.index_select(0, keep)
    kappa2 = kappa2.index_select(0, keep)
    tangent_aniso = tangent_aniso.index_select(0, keep)
    curvature_aniso = curvature_aniso.index_select(0, keep)
    n_points = vox.n_points.index_select(0, keep)
    i_mean = vox.i_mean.index_select(0, keep)
    i_std = vox.i_std.index_select(0, keep)
    src_ratio = vox.src_ratio.index_select(0, keep)
    diagnostics["use_geom_init_pass"] = int(use_geom.sum().item())

    token = build_anchor_token(
        c_init=c_init,
        R_init=R_init,
        s_init=s_init,
        kappa1=kappa1,
        kappa2=kappa2,
        fit_quality=fit_quality,
        tangent_aniso=tangent_aniso,
        curvature_aniso=curvature_aniso,
        i_mean=i_mean,
        i_std=i_std,
        token_variant=token_variant,
    )

    return VoxelAnchorOutput(
        c_init=c_init,
        R_init=R_init,
        s_init=s_init,
        kappa1=kappa1,
        kappa2=kappa2,
        fit_quality=fit_quality,
        tangent_aniso=tangent_aniso,
        curvature_aniso=curvature_aniso,
        n_points=n_points,
        i_mean=i_mean,
        i_std=i_std,
        src_ratio=src_ratio,
        token=token,
        use_geom_init=use_geom,
        k_eff=k_eff,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Static branch — spherical voxel anchored at LiDAR_0 origin
# ---------------------------------------------------------------------------

class VoxelAnchorBuilder:
    """Static-branch builder. Spherical voxel + global candidate pool."""

    def __init__(
        self,
        dphi_deg: float = 3.0,
        dtheta_deg: float = 4.0,
        dr_m: float = 3.0,
        k_min: int = 8,
        k_target: int = 16,
        residual_threshold: float = 0.806,
        filter_mode: str = "residual",
        planarity_threshold: float = 0.2,
        token_variant: str = "full",
        knn_chunk_size: int = 256,
        knn_r_min: float = 0.5,
        knn_r_max: float = 8.0,
        query_voxel_min_points: int = 1,
    ) -> None:
        self.voxelizer = SphericalVoxelizer(
            dphi_deg=dphi_deg,
            dtheta_deg=dtheta_deg,
            dr_m=dr_m,
            query_voxel_min_points=query_voxel_min_points,
        )
        self.k_min = int(k_min)
        self.query_voxel_min_points = int(query_voxel_min_points)
        self.k_target = int(k_target)
        self.residual_threshold = float(residual_threshold)
        self.filter_mode = str(filter_mode)
        self.planarity_threshold = float(planarity_threshold)
        self.token_variant = str(token_variant)
        self.knn_chunk_size = int(knn_chunk_size)
        self._dphi_rad = math.radians(float(dphi_deg))
        self._dtheta_rad = math.radians(float(dtheta_deg))
        self._dr = float(dr_m)
        self._r_min = float(knn_r_min)
        self._r_max = float(knn_r_max)

    def _voxel_r_max(self, q: Tensor) -> Tensor:
        r = q.norm(dim=-1)
        dphi_r = self._dphi_rad * r
        dtheta_r = self._dtheta_rad * r
        dr = torch.full_like(r, self._dr)
        diag = torch.sqrt(dphi_r * dphi_r + dtheta_r * dtheta_r + dr * dr)
        return diag.clamp(min=self._r_min, max=self._r_max)

    @torch.no_grad()
    def __call__(self, xyz: Tensor, intensity: Tensor, src: Tensor) -> VoxelAnchorOutput:
        token_width = 25 if self.token_variant == "with_normal" else 22
        if xyz.shape[0] == 0:
            return _make_empty(xyz.device, xyz.dtype, token_width=token_width)
        vox = self.voxelizer(xyz, intensity=intensity, src=src)
        if vox.query_xyz.shape[0] == 0:
            return _make_empty(xyz.device, xyz.dtype, {"query_voxels": 0, "k_eff_pass": 0, "residual_pass": 0, "final_anchors": 0, "fallback_reason": "no_query_voxels"}, token_width=token_width)
        return _quad_fit_and_token(
            vox,
            candidates_xyz=xyz,
            k_min=self.k_min,
            k_target=self.k_target,
            residual_threshold=self.residual_threshold,
            filter_mode=self.filter_mode,
            planarity_threshold=self.planarity_threshold,
            token_variant=self.token_variant,
            knn_chunk_size=self.knn_chunk_size,
            r_max_fn=self._voxel_r_max,
        )


# ---------------------------------------------------------------------------
# Dynamic branch — per-instance bbox-local cartesian voxel
# ---------------------------------------------------------------------------

class DynamicVoxelAnchorBuilder:
    """Dynamic-branch builder.

    For each tracked instance from `decompose_scene`, voxelize its canonical
    point cloud with cartesian cells of size `max(bbox_dim) / divisor`. k-NN
    candidates are restricted to the same instance (locked design Q11).
    """

    def __init__(
        self,
        bbox_voxel_divisor: int = 8,
        bbox_voxel_min: float = 0.05,
        bbox_voxel_max_extent: float = 20.0,
        k_min: int = 8,
        k_target: int = 16,
        residual_threshold: float = 0.806,
        filter_mode: str = "residual",
        planarity_threshold: float = 0.2,
        token_variant: str = "full",
        knn_chunk_size: int = 256,
        r_max_pad_factor: float = 2.0,
        query_voxel_min_points: int = 1,
    ) -> None:
        self.bbox_voxel_divisor = int(bbox_voxel_divisor)
        self.bbox_voxel_min = float(bbox_voxel_min)
        self.bbox_voxel_max_extent = float(bbox_voxel_max_extent)
        self.k_min = int(k_min)
        self.query_voxel_min_points = int(query_voxel_min_points)
        self.k_target = int(k_target)
        self.residual_threshold = float(residual_threshold)
        self.filter_mode = str(filter_mode)
        self.planarity_threshold = float(planarity_threshold)
        self.token_variant = str(token_variant)
        self.knn_chunk_size = int(knn_chunk_size)
        self.r_max_pad_factor = float(r_max_pad_factor)

    def _voxel_size_for_bbox(self, box: Tensor) -> float:
        """voxel_size = max(w, l, h) / divisor, clamped to a positive floor."""
        max_dim = float(box[3:6].max().item())
        return max(max_dim / self.bbox_voxel_divisor, self.bbox_voxel_min)

    @torch.no_grad()
    def __call__(self, instances: list[dict]) -> list[tuple[int, VoxelAnchorOutput]]:
        """Build anchors for every dynamic instance.

        Args:
            instances: dynamic_list from `decompose_scene` — each dict has
                'instance_id', 'canonical_xyz', 'canonical_intensity',
                'canonical_time', 'box_0' ([7] = x,y,z,w,l,h,yaw).

        Returns:
            list of (instance_id, VoxelAnchorOutput). Empty output ⇒ instance
            failed both filters; caller should fall back to static.
        """
        results: list[tuple[int, VoxelAnchorOutput]] = []
        token_width = 25 if self.token_variant == "with_normal" else 22
        for inst in instances:
            iid = int(inst["instance_id"])
            xyz = inst["canonical_xyz"]
            intensity = inst["canonical_intensity"]
            time = inst["canonical_time"]
            box0 = inst["box_0"]
            device = xyz.device
            dtype = xyz.dtype

            if xyz.shape[0] == 0:
                results.append((iid, _make_empty(device, dtype, {"query_voxels": 0, "k_eff_pass": 0, "residual_pass": 0, "final_anchors": 0, "fallback_reason": "no_instance_points"}, token_width=token_width)))
                continue

            voxel_size = self._voxel_size_for_bbox(box0)
            voxelizer = CartesianVoxelizer(
                voxel_size=voxel_size,
                query_voxel_min_points=self.query_voxel_min_points,
                max_extent=self.bbox_voxel_max_extent,
            )
            vox = voxelizer(xyz, intensity=intensity, src=time)
            if vox.query_xyz.shape[0] == 0:
                results.append((iid, _make_empty(device, dtype, {"query_voxels": 0, "k_eff_pass": 0, "residual_pass": 0, "final_anchors": 0, "fallback_reason": "no_query_voxels"}, token_width=token_width)))
                continue

            # r_max bounded by voxel diagonal × pad factor — instance is small,
            # so a constant radius works well across all queries.
            r_max_value = voxel_size * math.sqrt(3.0) * self.r_max_pad_factor

            def _const_r_max(q: Tensor, _v=r_max_value) -> Tensor:
                return torch.full(
                    (q.shape[0],), _v, device=q.device, dtype=q.dtype
                )

            out = _quad_fit_and_token(
                vox,
                candidates_xyz=xyz,
                k_min=self.k_min,
                k_target=self.k_target,
                residual_threshold=self.residual_threshold,
                filter_mode=self.filter_mode,
                planarity_threshold=self.planarity_threshold,
                token_variant=self.token_variant,
                knn_chunk_size=self.knn_chunk_size,
                r_max_fn=_const_r_max,
            )
            results.append((iid, out))
        return results
