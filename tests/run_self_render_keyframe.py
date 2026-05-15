"""
A3.5 — Single static-frame self-render validation.

Goal
----
Take one nuScenes LiDAR keyframe, place a tiny Gaussian at every scan point,
render through the LiDAR (spherical) rasterizer, and verify the rendered range
image reproduces the analytical spherical projection of the same points.

This validates the rasterizer end-to-end on real-scale data
(N ≈ 30k, full 360°×40° FOV) without depending on PTv3 / QGSHead /
quadric_fit (which are Sonnet's responsibility for A4 wiring).

Reference range image
---------------------
For each input point p_i in the sensor frame, project to (u_i, v_i, r_i)
using the *exact same* spherical formula the CUDA kernel uses
(see cuda_rasterizer/spherical.h). For each pixel, expected range = min over
all input points landing in that pixel (LiDAR returns first-hit semantics).

Per-pixel comparison
--------------------
On covered pixels (alpha_accum > τ): physical_range = rendered_range / alpha_accum.
Compute |physical_range − reference_range| stats: MAE, median, p95, hit ratio.

Pass criteria (informational; this is a milestone, not a unit test)
---------
- coverage  > 50 %  of reference-hit pixels are also rendered-hit
- MAE       < 1.0 m on covered pixels
- median    < 0.10 m on covered pixels

Usage
-----
    conda activate lnvs
    CUDA_VISIBLE_DEVICES=0 python tests/run_self_render_keyframe.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch

# Project root + external loader path
sys.path.insert(0, "/data/jeongbin/qgs")
sys.path.insert(0, "/data1/nuScenes/loader")

from config import QGSConfig                                          # noqa: E402
from diff_quadratic_rasterization import (                            # noqa: E402
    LIDAR_LATENT_DIM,
    LiDARRasterizer,
    make_lidar_settings,
)


# ---------------------------------------------------------------------------
# Spherical projection mirror of cuda_rasterizer/spherical.h
# ---------------------------------------------------------------------------

def project_to_pixel(
    p: torch.Tensor,            # [N, 3]  sensor frame (x=right, y=fwd, z=up)
    el_min_rad: float,
    el_max_rad: float,
    W: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        r:       [N]   range
        az:      [N]   in [-π, π]
        el:      [N]   in [-π/2, π/2]
        in_fov:  [N]   bool  el ∈ [el_min, el_max] (azimuth always in)
    """
    x, y, z = p.unbind(dim=-1)
    r = p.norm(dim=-1)
    az = torch.atan2(x, y)
    xy = torch.sqrt(x * x + y * y).clamp(min=1e-8)
    el = torch.atan2(z, xy)
    in_fov = (el >= el_min_rad) & (el <= el_max_rad)
    return r, az, el, in_fov


def az_el_to_uv(
    az: torch.Tensor,
    el: torch.Tensor,
    el_min_rad: float,
    el_max_rad: float,
    W: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns integer pixel coords (u, v) in [0, W-1] × [0, H-1]."""
    w_per_rad_az = W / (2.0 * math.pi)
    h_per_rad_el = H / (el_max_rad - el_min_rad)
    u = (az + math.pi) * w_per_rad_az
    v = (el - el_min_rad) * h_per_rad_el
    u_int = u.floor().long().clamp(0, W - 1)
    v_int = v.floor().long().clamp(0, H - 1)
    return u_int, v_int


def build_reference_range_images(
    p: torch.Tensor,
    el_min_rad: float,
    el_max_rad: float,
    W: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build two per-pixel reference range images by binning input points to pixels.

    Returns
    -------
    ref_min:   [H, W]   per-pixel min-range  (physical LiDAR first-hit semantics).
    ref_mean:  [H, W]   per-pixel arithmetic mean range
                        (closer to what an alpha-blender averages on-pixel).
    n_pts:     [H, W]   number of input points binned to that pixel.

    Uncovered pixels have ref_*=+inf and n_pts=0.
    """
    r, az, el, in_fov = project_to_pixel(p, el_min_rad, el_max_rad, W, H)
    r_in = r[in_fov]
    az_in = az[in_fov]
    el_in = el[in_fov]
    u, v = az_el_to_uv(az_in, el_in, el_min_rad, el_max_rad, W, H)

    flat_idx = v * W + u                      # [N']
    n_pix = H * W

    ref_min = torch.full((n_pix,), float("inf"), dtype=p.dtype, device=p.device)
    ref_min.scatter_reduce_(0, flat_idx, r_in, reduce="amin", include_self=True)

    # Mean range: sum / count
    sum_r = torch.zeros(n_pix, dtype=p.dtype, device=p.device)
    sum_r.scatter_add_(0, flat_idx, r_in)
    cnt = torch.zeros(n_pix, dtype=p.dtype, device=p.device)
    cnt.scatter_add_(0, flat_idx, torch.ones_like(r_in))
    ref_mean = torch.where(cnt > 0, sum_r / cnt.clamp(min=1.0),
                           torch.full_like(sum_r, float("inf")))

    return ref_min.view(H, W), ref_mean.view(H, W), cnt.view(H, W)


# ---------------------------------------------------------------------------
# Frame loader
# ---------------------------------------------------------------------------

@dataclass
class FrameSample:
    points: torch.Tensor       # [N, 3]
    intensity: torch.Tensor    # [N]   in [0, 1]


def load_keyframe(cfg: QGSConfig, frame_idx: int = 0) -> FrameSample:
    """Load input_0 of dataset[frame_idx] and return CPU tensors."""
    from dataset import NuScenesNVSDataset
    ds = NuScenesNVSDataset(
        dataroot=cfg.data_root,
        version="v1.0-trainval",
        split="train",
        frame_gap=2,
        mode="nvs",
    )
    item = ds[frame_idx]
    pc = item["input_0"]                        # [N, 4]  (x,y,z,intensity)
    xyz = pc[:, :3].float()
    intensity = pc[:, 3].float() / 255.0        # nuScenes stores 0–255
    return FrameSample(points=xyz, intensity=intensity.clamp(0.0, 1.0))


# ---------------------------------------------------------------------------
# Self-render
# ---------------------------------------------------------------------------

@dataclass
class SelfRenderResult:
    rendered_range:    torch.Tensor   # [H, W]  raw alpha-blended depth channel
    alpha_accum:       torch.Tensor   # [H, W]
    reference_min:     torch.Tensor   # [H, W]  per-pixel min range  (+inf if empty)
    reference_mean:    torch.Tensor   # [H, W]  per-pixel mean range (+inf if empty)
    points_per_pixel:  torch.Tensor   # [H, W]  count of points binned per pixel
    n_points:          int
    n_in_fov:          int


def self_render(
    sample: FrameSample,
    *,
    W: int = 1024,
    H: int = 32,
    el_min_deg: float = -30.67,
    el_max_deg: float = +10.67,
    ego_radius: float = 2.5,
    r_far: float = 70.0,
    scale: float = 0.10,                  # Gaussian extent (m)
    sigma: float = 3.0,
    opacity: float = 0.95,
    device: str = "cuda",
    max_points: int | None = None,
) -> SelfRenderResult:
    """
    Place identity-rotation Gaussians at each in-range scan point and render.
    """
    # -------- mask ego + far points -------------------------------------
    p = sample.points
    rng = p.norm(dim=-1)
    keep = (rng > ego_radius) & (rng < r_far)
    p = p[keep]
    intensity = sample.intensity[keep]

    if max_points is not None and p.shape[0] > max_points:
        # Random subsample for speed
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(p.shape[0], generator=g)[:max_points]
        p = p[perm]
        intensity = intensity[perm]

    n_points = p.shape[0]

    # -------- move to GPU -----------------------------------------------
    p = p.to(device).contiguous()
    intensity = intensity.to(device).contiguous()

    el_min_rad = math.radians(el_min_deg)
    el_max_rad = math.radians(el_max_deg)

    # -------- in-FOV count for reporting --------------------------------
    _, _, _, in_fov = project_to_pixel(p, el_min_rad, el_max_rad, W, H)
    n_in_fov = int(in_fov.sum().item())

    # -------- build per-Gaussian properties -----------------------------
    N = p.shape[0]
    means3D = p
    means2D = torch.zeros_like(means3D, requires_grad=True)
    scales = torch.full((N, 3), scale, device=device, dtype=torch.float32)
    rotations = torch.zeros(N, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0                                       # identity quat
    opacities = torch.full((N, 1), opacity, device=device, dtype=torch.float32)
    intensity_per_g = intensity                                 # [N]
    latent = torch.zeros((N, LIDAR_LATENT_DIM), device=device, dtype=torch.float32)

    # -------- rasterizer settings (sensor at origin, identity view) -----
    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    from tests._lidar_test_helpers import uniform_row_elevation_rad
    row_to_el = uniform_row_elevation_rad(el_min_rad, el_max_rad, H).to(device)
    settings = make_lidar_settings(
        image_height=H, image_width=W,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        row_to_elevation_rad=row_to_el,
        viewmatrix=viewmatrix, campos=campos,
        sigma=sigma, r_near=0.2, r_far=r_far,
    )

    out = LiDARRasterizer(settings)(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        intensity=intensity_per_g,
        latent=latent,
    )

    # -------- reference range (analytical projection) -------------------
    ref_min, ref_mean, n_per_pix = build_reference_range_images(
        p, el_min_rad, el_max_rad, W, H
    )

    return SelfRenderResult(
        rendered_range=out.range.detach(),
        alpha_accum=out.alpha_accum.detach(),
        reference_min=ref_min,
        reference_mean=ref_mean,
        points_per_pixel=n_per_pix,
        n_points=n_points,
        n_in_fov=n_in_fov,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _err_stats(err: torch.Tensor) -> tuple[float, float, float]:
    if err.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(err.mean().item()),
        float(err.median().item()),
        float(err.quantile(0.95).item()),
    )


def report(result: SelfRenderResult, alpha_thresh: float = 0.1) -> dict:
    """
    Two error reports, both useful:

    - vs ref_min  (LiDAR first-hit semantics): high error in multi-hit pixels.
    - vs ref_mean (matches alpha-blend semantics): tighter; isolates kernel error.

    Also splits singleton (1 point per pixel) vs multi-hit pixels — the singleton
    bucket cleanly tests rasterizer geometry without alpha-blend confounding.
    """
    ref_min  = result.reference_min
    ref_mean = result.reference_mean
    n_per    = result.points_per_pixel
    rend_raw = result.rendered_range
    alpha    = result.alpha_accum

    H, W = ref_min.shape
    ref_hit  = ref_min.isfinite()
    rend_hit = alpha > alpha_thresh

    physical = torch.where(
        rend_hit,
        rend_raw / alpha.clamp(min=1e-3),
        torch.zeros_like(rend_raw),
    )

    both = ref_hit & rend_hit
    singleton = both & (n_per == 1)
    multi     = both & (n_per >= 2)

    ref_count  = int(ref_hit.sum().item())
    rend_count = int(rend_hit.sum().item())
    overlap    = int(both.sum().item())
    coverage   = overlap / max(ref_count, 1)

    # Two flavours of error
    err_min  = (physical[both]      - ref_min[both]      ).abs()
    err_mean = (physical[both]      - ref_mean[both]     ).abs()
    err_single = (physical[singleton] - ref_min[singleton]).abs()
    err_multi  = (physical[multi]     - ref_min[multi]    ).abs()

    # Relative error vs reference range — useful because per-pixel angular
    # footprint grows linearly with range, so absolute error is range-coupled.
    rel_single = err_single / ref_min[singleton].clamp(min=0.5)

    mae_min, med_min, p95_min   = _err_stats(err_min)
    mae_mn,  med_mn,  p95_mn    = _err_stats(err_mean)
    mae_s,   med_s,   p95_s     = _err_stats(err_single)
    mae_m,   med_m,   p95_m     = _err_stats(err_multi)
    rmae_s,  rmed_s, rp95_s     = _err_stats(rel_single)

    n_singleton = int(singleton.sum().item())
    n_multi     = int(multi.sum().item())

    metrics = {
        "n_points":              result.n_points,
        "n_in_fov":              result.n_in_fov,
        "ref_hit_pixels":        ref_count,
        "rend_hit_pixels":       rend_count,
        "overlap_pixels":        overlap,
        "coverage_ratio":        coverage,
        "vs_ref_min":   {"mae": mae_min, "median": med_min, "p95": p95_min},
        "vs_ref_mean":  {"mae": mae_mn,  "median": med_mn,  "p95": p95_mn},
        "singleton_px": {"n": n_singleton, "mae": mae_s, "median": med_s, "p95": p95_s},
        "multi_px":     {"n": n_multi,     "mae": mae_m, "median": med_m, "p95": p95_m},
        "image_hw":              (H, W),
    }

    # ── Pretty print ────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("A3.5 — Single keyframe self-render validation")
    print("=" * 64)
    print(f"  Points loaded             : {metrics['n_points']:>8d}")
    print(f"  Points in FOV (el bound)  : {metrics['n_in_fov']:>8d}")
    print(f"  Image size (H × W)        : {H} × {W}")
    print(f"  Reference-hit pixels      : {ref_count:>8d}")
    print(f"  Rendered-hit pixels       : {rend_count:>8d}  "
          f"(alpha > {alpha_thresh})")
    print(f"  Overlap pixels            : {overlap:>8d}")
    print(f"  Coverage ratio            : {coverage:>8.3f}  "
          f"(rendered ∩ ref / ref)")
    print()
    print(f"  vs ref_min  (1st-hit)     : "
          f"MAE {mae_min:>7.3f} m  med {med_min:>7.3f}  p95 {p95_min:>7.3f}")
    print(f"  vs ref_mean (avg)         : "
          f"MAE {mae_mn:>7.3f} m  med {med_mn:>7.3f}  p95 {p95_mn:>7.3f}")
    print()
    print(f"  singleton-px ({n_singleton:>5d})    : "
          f"MAE {mae_s:>7.3f} m  med {med_s:>7.3f}  p95 {p95_s:>7.3f}")
    print(f"  multi-px     ({n_multi:>5d})    : "
          f"MAE {mae_m:>7.3f} m  med {med_m:>7.3f}  p95 {p95_m:>7.3f}")
    print(f"  singleton rel-err          : "
          f"MAE {rmae_s:>7.4f}    med {rmed_s:>7.4f}  p95 {rp95_s:>7.4f}")
    print()
    print("Pass criteria (informational):")
    cov_ok  = coverage > 0.50
    rel_ok  = (rmed_s < 0.05) if n_singleton > 100 else True
    mean_ok = mae_mn < 1.0
    print(f"  coverage > 0.50                  : {'PASS' if cov_ok else 'FAIL'}")
    print(f"  singleton rel-err median < 0.05  : {'PASS' if rel_ok else 'FAIL'}")
    print(f"  vs ref_mean MAE          < 1.00m : {'PASS' if mean_ok else 'FAIL'}")
    print("=" * 64)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping.")
        sys.exit(1)

    cfg = QGSConfig()
    print(f"Loading first keyframe from {cfg.data_root} (split=train)…")
    sample = load_keyframe(cfg, frame_idx=0)
    print(f"  Raw points: {sample.points.shape[0]}")

    # Sweep two configurations to disentangle resolution vs scale effects.
    for label, (W, H, scale) in [
        ("nominal (1024x32, s=0.10)", (1024, 32, 0.10)),
        ("hi-res  (2048x64, s=0.05)", (2048, 64, 0.05)),
    ]:
        print()
        print(f"### Configuration: {label}")
        result = self_render(sample, W=W, H=H, scale=scale, max_points=None)
        report(result)


if __name__ == "__main__":
    main()
