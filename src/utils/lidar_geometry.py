"""Single source of truth for LiDAR row ↔ elevation ↔ ring mappings.

All consumers (renderer ray grid, GT range/intensity map, range→point unprojection,
CUDA rasterizer host-side setup) must use this module. The mapping is nonuniform —
each row's elevation is read from the sensor's actual firing table (HDL-32E for
nuScenes), not a linear `linspace(el_min, el_max, H)`.

Row convention (row_bottom0): `row_to_el[0]` is the LOWEST elevation, `row_to_el[H-1]`
is the highest. This matches the existing CUDA / python_ref / tests convention
where v=0 is the bottom of the panorama and v=H-1 is the top.

Math:
  - Row k center elevation:    row_to_el[k]              (k=0..H-1)
  - Forward v(el):              piecewise-linear through {(row_to_el[k], k+0.5)}
                                with edge segments extrapolated by local slope.
  - Inverse el(v):              inverse of v(el); v=k+0.5 ↔ el=row_to_el[k].
  - Effective FOV bounds:       el_min_eff = row_to_el[0]   - 0.5*(row_to_el[1]  -row_to_el[0])
                                el_max_eff = row_to_el[H-1] + 0.5*(row_to_el[H-1]-row_to_el[H-2])
                                (These correspond to v=0 and v=H.)
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Cached primitives. Keys must be hashable, so we accept the raw tuple/device
# instead of the (unhashable) dataclass.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _row_to_elevation_rad_cached(
    ring_to_el_deg: tuple,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    arr = torch.tensor(ring_to_el_deg, device=device, dtype=dtype)
    sorted_el, _ = torch.sort(arr)
    return torch.deg2rad(sorted_el)


@functools.lru_cache(maxsize=16)
def _ring_at_row_cached(
    ring_to_el_deg: tuple,
    device: torch.device,
) -> Tensor:
    """ring_at_row[k] = ring index emitted at row k (row_bottom0)."""
    arr = torch.tensor(ring_to_el_deg, device=device, dtype=torch.float64)
    return torch.argsort(arr).to(torch.int64)


@functools.lru_cache(maxsize=16)
def _ring_to_row_cached(
    ring_to_el_deg: tuple,
    device: torch.device,
) -> Tensor:
    """ring_to_row[r] = row position of ring r (row_bottom0)."""
    ring_at_row = _ring_at_row_cached(ring_to_el_deg, device)
    ring_to_row = torch.empty_like(ring_at_row)
    H = ring_at_row.shape[0]
    ring_to_row[ring_at_row] = torch.arange(H, device=device, dtype=torch.int64)
    return ring_to_row


def _ensure_torch_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


# ---------------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------------

def get_row_to_elevation_rad(
    cfg,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-row center elevation in radians, ascending (row_bottom0)."""
    dev = _ensure_torch_device(device)
    return _row_to_elevation_rad_cached(tuple(cfg.ring_to_elevation_deg), dev, dtype)


def get_ring_to_row(
    cfg,
    device: torch.device | str = "cpu",
) -> Tensor:
    """ring_to_row[r] = row position of ring r."""
    dev = _ensure_torch_device(device)
    return _ring_to_row_cached(tuple(cfg.ring_to_elevation_deg), dev)


def get_ring_at_row(
    cfg,
    device: torch.device | str = "cpu",
) -> Tensor:
    """ring_at_row[k] = ring index emitted at row k."""
    dev = _ensure_torch_device(device)
    return _ring_at_row_cached(tuple(cfg.ring_to_elevation_deg), dev)


def get_effective_el_bounds(cfg) -> tuple[float, float]:
    """(el_min_eff_rad, el_max_eff_rad) — extrapolated to v=0 and v=H."""
    row_el = get_row_to_elevation_rad(cfg, device="cpu", dtype=torch.float64)
    el_min = (row_el[0] - 0.5 * (row_el[1] - row_el[0])).item()
    el_max = (row_el[-1] + 0.5 * (row_el[-1] - row_el[-2])).item()
    return float(el_min), float(el_max)


# ---------------------------------------------------------------------------
# Piecewise-linear v ↔ el
# ---------------------------------------------------------------------------

def points_to_v(el_rad: Tensor, cfg) -> Tensor:
    """Forward map: elevation [rad] → continuous v.

    Piecewise-linear through control points {(row_to_el[k], k+0.5)}, with edge
    segments extrapolated by local slope (so el=el_min_eff → v=0 and
    el=el_max_eff → v=H). Caller is responsible for frustum reject.
    """
    row_el = get_row_to_elevation_rad(cfg, el_rad.device, el_rad.dtype)
    H = row_el.shape[0]
    # idx where el would insert: el ≤ row_el[idx]
    idx = torch.searchsorted(row_el, el_rad, right=False)
    # segment k such that row_el[k] ≤ el ≤ row_el[k+1], clamped to [0, H-2]
    seg = (idx - 1).clamp(min=0, max=H - 2)
    e_lo = row_el[seg]
    e_hi = row_el[seg + 1]
    return seg.to(el_rad.dtype) + 0.5 + (el_rad - e_lo) / (e_hi - e_lo)


def v_to_el(v_continuous: Tensor, cfg) -> Tensor:
    """Inverse map: continuous v → elevation [rad].

    By construction v=k+0.5 returns row_to_el[k] exactly. Outside [0.5, H-0.5]
    we extrapolate using the first/last segment slope (matches points_to_v).
    """
    row_el = get_row_to_elevation_rad(cfg, v_continuous.device, v_continuous.dtype)
    H = row_el.shape[0]
    seg = (v_continuous - 0.5).floor().to(torch.long).clamp(min=0, max=H - 2)
    e_lo = row_el[seg]
    e_hi = row_el[seg + 1]
    return e_lo + (v_continuous - seg.to(v_continuous.dtype) - 0.5) * (e_hi - e_lo)


# ---------------------------------------------------------------------------
# Ray construction
# ---------------------------------------------------------------------------

def row_to_ray_direction(
    v_idx,
    u_idx,
    cfg,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-(v,u) unit ray direction in sensor frame (x=right, y=forward, z=up).

    v_idx is interpreted as an integer row index (lookup row_to_el directly).
    For continuous v, call v_to_el explicitly. u_idx is integer column.
    """
    dev = _ensure_torch_device(device)
    row_el = get_row_to_elevation_rad(cfg, dev, dtype)
    W = int(cfg.lidar_width)
    v_idx_t = torch.as_tensor(v_idx, device=dev, dtype=torch.long)
    u_idx_t = torch.as_tensor(u_idx, device=dev, dtype=dtype)
    el = row_el[v_idx_t]
    az = (u_idx_t + 0.5) * (2.0 * math.pi / W) - math.pi
    cos_el = torch.cos(el)
    return torch.stack([
        torch.sin(az) * cos_el,
        torch.cos(az) * cos_el,
        torch.sin(el),
    ], dim=-1)


def make_lidar_ray_grid(
    cfg,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """[3, H, W] unit ray directions; each row k uses row_to_el[k], az uniform."""
    dev = _ensure_torch_device(device)
    row_el = get_row_to_elevation_rad(cfg, dev, dtype)
    H = int(cfg.lidar_height)
    W = int(cfg.lidar_width)
    u = torch.arange(W, device=dev, dtype=dtype) + 0.5
    az = u * (2.0 * math.pi / W) - math.pi
    cos_el = torch.cos(row_el)
    sin_el = torch.sin(row_el)
    cos_az = torch.cos(az)
    sin_az = torch.sin(az)
    dx = sin_az[None, :] * cos_el[:, None]
    dy = cos_az[None, :] * cos_el[:, None]
    dz = sin_el[:, None].expand(H, W)
    return torch.stack([dx, dy, dz], dim=0)


# ---------------------------------------------------------------------------
# Point cloud ↔ panoramic map
# ---------------------------------------------------------------------------

def points_to_lidar_maps(
    xyz: Tensor,
    intensity: Tensor,
    cfg,
    *,
    ring: Optional[Tensor] = None,
) -> dict:
    """Bin a sensor-frame point cloud onto the spherical pixel grid.

    Row assignment:
      - If `ring` is provided: row = ring_to_row[ring] (HDL-32E hardware truth).
      - Otherwise: nearest beam by elevation (argmin |row_to_el - el|).
    Azimuth: uniform `u = round((az + π) * W/(2π) - 0.5)`.
    First-hit rule: per pixel keep the smallest range and that point's intensity.

    Args:
        xyz [N, 3]: sensor-frame points (x=right, y=forward, z=up).
        intensity [N]: in [0, 1].
        cfg: QGSConfig.
        ring [N] int (optional): nuScenes ring channel.

    Returns dict {range_image [H,W], intensity_image [H,W], valid_mask [H,W] bool}.
    """
    if xyz.shape[0] != intensity.shape[0]:
        raise ValueError(
            f"xyz N={xyz.shape[0]} but intensity N={intensity.shape[0]}"
        )
    device = xyz.device
    dtype = xyz.dtype
    H = int(cfg.lidar_height)
    W = int(cfg.lidar_width)
    r_near = float(cfg.r_near)
    r_far = float(cfg.r_far)
    HW = H * W

    row_el = get_row_to_elevation_rad(cfg, device, dtype)             # [H]
    el_min_eff, el_max_eff = get_effective_el_bounds(cfg)

    x, y, z = xyz.unbind(dim=-1)
    r = xyz.norm(dim=-1)
    az = torch.atan2(x, y)
    xy = (x * x + y * y).clamp(min=1e-12).sqrt()
    el = torch.atan2(z, xy)

    w_per_rad_az = W / (2.0 * math.pi)
    u_f = (az + math.pi) * w_per_rad_az - 0.5
    u_int = u_f.round().long().clamp(0, W - 1)

    if ring is not None:
        if ring.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"ring N={ring.shape[0]} but xyz N={xyz.shape[0]}"
            )
        ring_long = ring.to(torch.int64).to(device)
        valid_ring = (ring_long >= 0) & (ring_long < H)
        ring_safe = ring_long.clamp(0, H - 1)
        v_int = get_ring_to_row(cfg, device)[ring_safe]
    else:
        valid_ring = torch.ones_like(r, dtype=torch.bool)
        diff = (el.unsqueeze(-1) - row_el.unsqueeze(0)).abs()         # [N, H]
        v_int = diff.argmin(dim=-1)

    in_fov = (
        (r >= r_near) & (r <= r_far) &
        (el >= el_min_eff) & (el <= el_max_eff) &
        valid_ring
    )

    pix = v_int * W + u_int
    pix_v = pix[in_fov]
    r_v = r[in_fov]
    int_v = intensity[in_fov]

    range_flat = torch.full((HW,), float("inf"), dtype=dtype, device=device)
    range_flat.scatter_reduce_(0, pix_v, r_v, reduce="amin", include_self=True)

    intensity_flat = torch.zeros(HW, dtype=dtype, device=device)
    if pix_v.numel() > 0:
        first_r = range_flat[pix_v]
        is_first = torch.isclose(r_v, first_r, rtol=0.0, atol=1e-6)
        local_idx = torch.arange(pix_v.shape[0], device=device, dtype=torch.long)
        sentinel = pix_v.shape[0]
        cand_idx = torch.where(
            is_first, local_idx, local_idx.new_full(local_idx.shape, sentinel)
        )
        first_idx_flat = local_idx.new_full((HW,), sentinel)
        first_idx_flat.scatter_reduce_(0, pix_v, cand_idx, reduce="amin", include_self=True)
        has_hit = first_idx_flat < sentinel
        intensity_flat[has_hit] = int_v[first_idx_flat[has_hit]]

    valid_flat = range_flat.isfinite()
    range_flat = torch.where(valid_flat, range_flat, torch.zeros_like(range_flat))

    return {
        "range_image":     range_flat.view(H, W),
        "intensity_image": intensity_flat.view(H, W),
        "valid_mask":      valid_flat.view(H, W),
    }


def range_map_to_points(
    range_image: Tensor,
    valid_mask: Tensor,
    cfg,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Unproject a panoramic range image into sensor-frame xyz [M, 3]."""
    dev = device if device is not None else range_image.device
    dt = dtype if dtype is not None else range_image.dtype
    ray = make_lidar_ray_grid(cfg, dev, dt)                       # [3, H, W]
    pts = (ray * range_image.unsqueeze(0)).permute(1, 2, 0)       # [H, W, 3]
    return pts[valid_mask]
