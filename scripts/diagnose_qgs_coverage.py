"""Diagnose per-ray QGS coverage for one dataset pair.

The report focuses on geometry coverage, not training loss.  It renders either
raw voxel-anchor initial QGS primitives or model-predicted primitives, then
classifies GT-hit rays whose accumulated alpha stays below a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QGSConfig
from models.geometry import decompose_scene
from models.geometry.voxel_anchor import VoxelAnchorOutput
from nn.eval_utils import (
    _ensure_2d_pose,
    _box1_to_frame0_pose,
    _box_to_pose,
    _make_dynamic_anchor_builder,
    _make_static_anchor_builder,
    _sensor_visibility_mask,
    concat_primitives,
    filter_frame_points,
    load_cfg_from_checkpoint,
    resolve_bbox_json,
    transform_primitives,
)
from nn.model import QGSModel
from nn.lidar_geometry import (
    get_effective_el_bounds,
    make_lidar_ray_grid,
    points_to_lidar_maps,
    points_to_v,
)
from nn.render_utils import render_primitives, rotmat_to_quat


@dataclass(frozen=True)
class FrameBundle:
    name: str
    primitives: dict
    contexts: list[dict]
    target: dict
    rendered: object
    viewmatrix: torch.Tensor


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_dataset(cfg: QGSConfig, split_name: str):
    target_split = cfg.train_split if split_name == "train" else cfg.eval_split
    sys.path.insert(0, os.path.join(os.path.expanduser(cfg.data_root), "loader"))
    from dataset import NuScenesNVSDataset  # noqa: WPS433

    bbox_json = resolve_bbox_json(cfg, target_split)
    return NuScenesNVSDataset(
        dataroot=os.path.expanduser(cfg.data_root),
        version=cfg.nuscenes_version,
        split=target_split,
        frame_gap=cfg.frame_gap,
        mode=cfg.dataset_mode,
        bbox_json_path=bbox_json if (cfg.dataset_mode == "bbox" and bbox_json) else None,
    )


def _anchor_to_init_primitives(
    anchor_out: VoxelAnchorOutput,
    *,
    latent_dim: int,
    alpha: float,
) -> dict | None:
    n = int(anchor_out.c_init.shape[0])
    if n == 0:
        return None
    device = anchor_out.c_init.device
    dtype = anchor_out.c_init.dtype
    return {
        "means3D": anchor_out.c_init,
        "scales": anchor_out.s_init,
        "rotations": rotmat_to_quat(anchor_out.R_init),
        "opacities": torch.full((n, 1), float(alpha), device=device, dtype=dtype),
        "intensity": anchor_out.i_mean.clamp(0.0, 1.0),
        "latent": torch.zeros((n, latent_dim), device=device, dtype=dtype),
        "features": torch.zeros((n, 0), device=device, dtype=dtype),
        "aux": {},
        "geom_init": {
            "c_init": anchor_out.c_init.unsqueeze(0),
            "R_init": anchor_out.R_init.unsqueeze(0),
            "s_init": anchor_out.s_init.unsqueeze(0),
            "fit_quality": anchor_out.fit_quality.unsqueeze(0),
            "use_geom_init": anchor_out.use_geom_init.unsqueeze(0),
            "kappa1_init": anchor_out.kappa1.unsqueeze(0),
            "kappa2_init": anchor_out.kappa2.unsqueeze(0),
            "tangent_aniso": anchor_out.tangent_aniso.unsqueeze(0),
            "curvature_aniso": anchor_out.curvature_aniso.unsqueeze(0),
        },
        "k_eff": anchor_out.k_eff,
        "diagnostics": {"memory_stages": []},
    }


def _primitive_extent_summary(primitives: dict) -> dict:
    means = primitives["means3D"].detach()
    scales = primitives["scales"].detach()
    if means.numel() == 0:
        return {}
    r = means.norm(dim=-1)
    abs_scales = scales.abs()
    out = {
        "range_m": _tensor_stats(r),
        "abs_s1_m": _tensor_stats(abs_scales[:, 0]),
        "abs_s2_m": _tensor_stats(abs_scales[:, 1]),
        "s3_m": _tensor_stats(abs_scales[:, 2]),
        "s1_over_range": _tensor_stats(abs_scales[:, 0] / r.clamp(min=1e-6)),
        "s2_over_range": _tensor_stats(abs_scales[:, 1] / r.clamp(min=1e-6)),
    }
    geom = primitives.get("geom_init", {})
    fit_quality = geom.get("fit_quality")
    if torch.is_tensor(fit_quality) and fit_quality.numel():
        fq = fit_quality[0].detach()
        out["fit_residual"] = _tensor_stats(fq[:, 0])
        out["fit_planarity"] = _tensor_stats(fq[:, 1])
    return out


def _tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"n": 0}
    return {
        "n": int(x.numel()),
        "min": float(x.min().item()),
        "p05": float(torch.quantile(x, 0.05).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "mean": float(x.mean().item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "max": float(x.max().item()),
    }


def _build_scene_and_primitives(
    *,
    cfg: QGSConfig,
    sample: dict,
    device: torch.device,
    mode: str,
    model: QGSModel | None,
    init_alpha: float,
) -> tuple[list[dict], list[dict], dict, dict]:
    p0 = sample["input_0"].to(device)
    p1 = sample["input_1"].to(device)
    rel_input_1_pose = _ensure_2d_pose(sample["input_1_pose"]).to(device)

    xyz0 = p0[:, :3].float()
    xyz1 = p1[:, :3].float()
    i0 = (p0[:, 3].float() / 255.0).clamp(0.0, 1.0)
    i1 = (p1[:, 3].float() / 255.0).clamp(0.0, 1.0)
    xyz0, i0 = filter_frame_points(xyz0, i0, cfg)
    xyz1, i1 = filter_frame_points(xyz1, i1, cfg)

    boxes_0 = sample.get("boxes_0", torch.empty(0, 7)).to(device)
    boxes_1 = sample.get("boxes_1", torch.empty(0, 7)).to(device)
    instance_ids_0 = sample.get("instance_ids_0", torch.empty(0, dtype=torch.long)).to(device)
    instance_ids_1 = sample.get("instance_ids_1", torch.empty(0, dtype=torch.long)).to(device)

    if boxes_0.numel() and boxes_1.numel():
        scene = decompose_scene(
            xyz0, xyz1, i0, i1,
            boxes_0, boxes_1,
            instance_ids_0, instance_ids_1,
            rel_input_1_pose,
        )
    else:
        p1_in_frame0 = (rel_input_1_pose[:3, :3] @ xyz1.T).T + rel_input_1_pose[:3, 3]
        scene = {
            "static_xyz": torch.cat([xyz0, p1_in_frame0], dim=0),
            "static_intensity": torch.cat([i0, i1], dim=0),
            "static_time": torch.cat([
                torch.zeros(xyz0.shape[0], device=device, dtype=xyz0.dtype),
                torch.ones(xyz1.shape[0], device=device, dtype=xyz1.dtype),
            ], dim=0),
            "dynamic": [],
            "untracked_stats": {},
        }

    if cfg.primitive_mode != "voxel_anchor":
        raise ValueError("diagnose_qgs_coverage currently expects primitive_mode='voxel_anchor'")

    frame0_contexts: list[dict] = []
    frame1_contexts: list[dict] = []
    anchor_summary = {
        "input_points": int(scene["static_xyz"].shape[0])
        + sum(int(d["canonical_xyz"].shape[0]) for d in scene["dynamic"]),
        "query_voxels": 0,
        "static_primitives": 0,
        "dynamic_primitives": 0,
        "fallback_instance_count": 0,
        "fallback_point_count": 0,
        "static_init_count": 0,
        "static_fallback_count": 0,
        "dynamic_init_count": 0,
        "dynamic_fallback_count": 0,
    }

    def realize(anchor_out: VoxelAnchorOutput, context_type: str) -> dict | None:
        if mode == "init":
            return _anchor_to_init_primitives(
                anchor_out,
                latent_dim=int(cfg.lidar_latent_dim),
                alpha=init_alpha,
            )
        if model is None:
            raise ValueError("model mode requires a model")
        return model.forward_anchor_context(
            anchor_out,
            context_type=context_type,
            return_diagnostics=False,
        )

    dyn_anchor_outputs = []
    dyn_prims_list = []
    if scene["dynamic"]:
        dynamic_builder = _make_dynamic_anchor_builder(cfg)
        dyn_anchor_pairs = dynamic_builder(scene["dynamic"])
        dyn_anchor_outputs = [out for _, out in dyn_anchor_pairs]
        if mode == "model":
            if model is None:
                raise ValueError("model mode requires a model")
            dyn_prims_list = model.forward_anchor_contexts_batched(
                dyn_anchor_outputs,
                context_type="dynamic",
            )
        else:
            dyn_prims_list = [
                _anchor_to_init_primitives(out, latent_dim=int(cfg.lidar_latent_dim), alpha=init_alpha)
                for out in dyn_anchor_outputs
            ]

    static_xyz_parts = [scene["static_xyz"].to(device)]
    static_i_parts = [scene["static_intensity"].to(device)]
    static_t_parts = [scene["static_time"].to(device)]
    for dyn_idx, (dyn, anchor_out, dyn_prims) in enumerate(
        zip(scene["dynamic"], dyn_anchor_outputs, dyn_prims_list)
    ):
        anchor_summary["query_voxels"] += int(anchor_out.diagnostics.get("query_voxels", 0))
        if dyn_prims is None:
            fallback_n = int(dyn["fallback_xyz"].shape[0])
            anchor_summary["fallback_instance_count"] += 1
            anchor_summary["fallback_point_count"] += fallback_n
            static_xyz_parts.append(dyn["fallback_xyz"].to(device))
            static_i_parts.append(dyn["fallback_intensity"].to(device))
            static_t_parts.append(dyn["fallback_time"].to(device))
            continue
        anchor_summary["dynamic_primitives"] += int(dyn_prims["means3D"].shape[0])
        anchor_summary["dynamic_init_count"] += int(anchor_out.use_geom_init.sum().item())
        anchor_summary["dynamic_fallback_count"] += int((~anchor_out.use_geom_init).sum().item())
        context = {
            "name": f"dynamic_{dyn_idx}",
            "type": "dynamic",
            "anchor_diagnostics": anchor_out.diagnostics,
            "primitive_extent": _primitive_extent_summary(dyn_prims),
        }
        if dyn.get("box_0") is not None:
            frame0_contexts.append({
                "name": f"dynamic_{dyn_idx}",
                "context": context,
                "primitives": transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device))),
            })
        if dyn.get("box_1") is not None:
            frame1_contexts.append({
                "name": f"dynamic_{dyn_idx}",
                "context": context,
                "primitives": transform_primitives(
                    dyn_prims,
                    _box1_to_frame0_pose(dyn["box_1"].to(device), rel_input_1_pose),
                ),
            })

    static_xyz = torch.cat(static_xyz_parts, dim=0)
    static_i = torch.cat(static_i_parts, dim=0)
    static_t = torch.cat(static_t_parts, dim=0)
    static_anchor_out = _make_static_anchor_builder(cfg)(
        static_xyz,
        static_i,
        static_t,
        pose_frame1_in_frame0=rel_input_1_pose,
    )
    anchor_summary["query_voxels"] += int(static_anchor_out.diagnostics.get("query_voxels", 0))
    static_prims = realize(static_anchor_out, "static")
    if static_prims is not None:
        anchor_summary["static_primitives"] = int(static_prims["means3D"].shape[0])
        anchor_summary["static_init_count"] = int(static_anchor_out.use_geom_init.sum().item())
        anchor_summary["static_fallback_count"] = int((~static_anchor_out.use_geom_init).sum().item())
        static_context = {
            "name": "static",
            "type": "static",
            "anchor_diagnostics": static_anchor_out.diagnostics,
            "primitive_extent": _primitive_extent_summary(static_prims),
        }
        frame0_contexts.append({"name": "static", "context": static_context, "primitives": static_prims})
        frame1_contexts.append({"name": "static", "context": static_context, "primitives": static_prims})

    anchor_summary["final_primitives"] = (
        int(anchor_summary["static_primitives"]) + int(anchor_summary["dynamic_primitives"])
    )
    anchor_summary["compression_ratio"] = (
        anchor_summary["final_primitives"] / max(anchor_summary["input_points"], 1)
    )

    if not frame0_contexts or not frame1_contexts:
        raise RuntimeError("no primitives were produced")

    return frame0_contexts, frame1_contexts, {
        "xyz0": xyz0,
        "xyz1": xyz1,
        "i0": i0,
        "i1": i1,
        "rel_input_1_pose": rel_input_1_pose,
    }, anchor_summary


def _render_frame_bundle(
    *,
    cfg: QGSConfig,
    name: str,
    contexts: list[dict],
    target_xyz: torch.Tensor,
    target_i: torch.Tensor,
    viewmatrix: torch.Tensor,
    campos: torch.Tensor,
    scale_modifier: float,
) -> FrameBundle:
    device = target_xyz.device
    primitives = concat_primitives([ctx["primitives"] for ctx in contexts])
    target = points_to_lidar_maps(target_xyz, target_i, cfg)
    rendered = render_primitives(
        primitives,
        cfg,
        scale_modifier=scale_modifier,
        viewmatrix=viewmatrix,
        campos=campos,
    )
    return FrameBundle(
        name=name,
        primitives=primitives,
        contexts=contexts,
        target=target,
        rendered=rendered,
        viewmatrix=viewmatrix.to(device=device, dtype=target_xyz.dtype),
    )


def _project_to_lidar_pixels(
    xyz_world: torch.Tensor,
    viewmatrix: torch.Tensor,
    cfg: QGSConfig,
) -> dict:
    xyz = (viewmatrix[:3, :3] @ xyz_world.T).T + viewmatrix[:3, 3]
    x, y, z = xyz.unbind(dim=-1)
    r = xyz.norm(dim=-1)
    xy = (x * x + y * y).clamp(min=1e-12).sqrt()
    az = torch.atan2(x, y)
    el = torch.atan2(z, xy)
    el_min, el_max = get_effective_el_bounds(cfg)
    u = (az + math.pi) * (cfg.lidar_width / (2.0 * math.pi)) - 0.5
    v = points_to_v(el, cfg) - 0.5
    visible = (
        torch.isfinite(r)
        & (r >= cfg.r_near)
        & (r <= cfg.r_far)
        & (el >= el_min)
        & (el <= el_max)
    )
    return {"sensor_xyz": xyz, "range": r, "az": az, "el": el, "u": u, "v": v, "visible": visible}


def _nearest_gaussian_to_rays(
    *,
    ray_v: torch.Tensor,
    ray_u: torch.Tensor,
    ray_range: torch.Tensor,
    prim_u: torch.Tensor,
    prim_v: torch.Tensor,
    prim_range: torch.Tensor,
    prim_radius: torch.Tensor,
    width: int,
    chunk: int,
) -> dict:
    n_rays = int(ray_u.shape[0])
    n_prim = int(prim_u.shape[0])
    if n_rays == 0 or n_prim == 0:
        device = ray_u.device
        dtype = ray_u.dtype
        return {
            "idx": torch.full((n_rays,), -1, device=device, dtype=torch.long),
            "pixel_dist": torch.full((n_rays,), float("inf"), device=device, dtype=dtype),
            "du": torch.full((n_rays,), float("inf"), device=device, dtype=dtype),
            "dv": torch.full((n_rays,), float("inf"), device=device, dtype=dtype),
            "range_gap": torch.full((n_rays,), float("inf"), device=device, dtype=dtype),
            "radius": torch.zeros((n_rays,), device=device, dtype=dtype),
        }

    best_dist = torch.full((n_rays,), float("inf"), device=ray_u.device, dtype=ray_u.dtype)
    best_idx = torch.full((n_rays,), -1, device=ray_u.device, dtype=torch.long)
    best_du = torch.zeros_like(best_dist)
    best_dv = torch.zeros_like(best_dist)
    for start in range(0, n_rays, chunk):
        end = min(start + chunk, n_rays)
        du = (ray_u[start:end].unsqueeze(1) - prim_u.unsqueeze(0)).abs()
        du = torch.minimum(du, width - du)
        dv = (ray_v[start:end].unsqueeze(1) - prim_v.unsqueeze(0)).abs()
        d2 = du * du + dv * dv
        d, idx = d2.min(dim=1)
        d = d.sqrt()
        best_dist[start:end] = d
        best_idx[start:end] = idx
        best_du[start:end] = du.gather(1, idx.unsqueeze(1)).squeeze(1)
        best_dv[start:end] = dv.gather(1, idx.unsqueeze(1)).squeeze(1)

    safe_idx = best_idx.clamp(min=0)
    range_gap = prim_range[safe_idx] - ray_range
    nearest_radius = prim_radius[safe_idx].to(dtype=ray_u.dtype)
    range_gap = torch.where(best_idx >= 0, range_gap, torch.full_like(range_gap, float("inf")))
    nearest_radius = torch.where(best_idx >= 0, nearest_radius, torch.zeros_like(nearest_radius))
    return {
        "idx": best_idx,
        "pixel_dist": best_dist,
        "du": best_du,
        "dv": best_dv,
        "range_gap": range_gap,
        "radius": nearest_radius,
    }


def _bin_counts(values: torch.Tensor, bins: list[float]) -> list[int]:
    if values.numel() == 0:
        return [0 for _ in range(len(bins) - 1)]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        out.append(int(((values >= lo) & (values < hi)).sum().item()))
    return out


def _top_counts(mask: torch.Tensor, top_k: int = 8) -> list[dict]:
    if mask.numel() == 0:
        return []
    counts = mask.long().sum(dim=1)
    k = min(top_k, int(counts.numel()))
    if k == 0:
        return []
    vals, idx = torch.topk(counts, k=k)
    return [{"index": int(i.item()), "count": int(v.item())} for v, i in zip(vals, idx)]


def _context_gaussian_stats(
    contexts: list[dict],
    rendered,
    viewmatrix: torch.Tensor,
    cfg: QGSConfig,
) -> list[dict]:
    offset = 0
    out = []
    el_min_rad, el_max_rad = get_effective_el_bounds(cfg)
    for ctx in contexts:
        n = int(ctx["primitives"]["means3D"].shape[0])
        radii = rendered.radii[offset:offset + n]
        touched = rendered.n_touched[offset:offset + n]
        visible = _sensor_visibility_mask(
            ctx["primitives"]["means3D"],
            viewmatrix,
            el_min_rad=el_min_rad,
            el_max_rad=el_max_rad,
            r_near=cfg.r_near,
            r_far=cfg.r_far,
        )
        out.append({
            "name": ctx["name"],
            "type": ctx["context"]["type"],
            "n": n,
            "visible": int(visible.sum().item()),
            "positive_radius": int((radii > 0).sum().item()),
            "touched": int((touched > 0).sum().item()),
            "untouched_positive_radius": int(((radii > 0) & (touched <= 0)).sum().item()),
            "radius_px": _tensor_stats(radii.float()),
            "n_touched_tiles": _tensor_stats(touched.float()),
            "primitive_extent": ctx["context"].get("primitive_extent", {}),
            "anchor_diagnostics": ctx["context"].get("anchor_diagnostics", {}),
        })
        offset += n
    return out


def _analyze_frame(
    frame: FrameBundle,
    cfg: QGSConfig,
    *,
    alpha_eps: float,
    nearest_chunk: int,
) -> dict:
    target = frame.target
    rendered = frame.rendered
    gt_valid = target["valid_mask"]
    alpha = rendered.alpha_accum
    covered = gt_valid & (alpha > alpha_eps)
    uncovered = gt_valid & ~covered
    false_coverage = (~gt_valid) & (alpha > alpha_eps)

    proj = _project_to_lidar_pixels(frame.primitives["means3D"], frame.viewmatrix, cfg)
    prim_keep = proj["visible"] & (rendered.radii > 0)
    if prim_keep.any():
        prim_u = proj["u"][prim_keep].float()
        prim_v = proj["v"][prim_keep].float()
        prim_range = proj["range"][prim_keep].float()
        prim_radius = rendered.radii[prim_keep].float()
    else:
        device = alpha.device
        prim_u = torch.zeros((0,), device=device)
        prim_v = torch.zeros((0,), device=device)
        prim_range = torch.zeros((0,), device=device)
        prim_radius = torch.zeros((0,), device=device)

    touched_keep = prim_keep & (rendered.n_touched > 0)
    if touched_keep.any():
        touched_u = proj["u"][touched_keep].float()
        touched_v = proj["v"][touched_keep].float()
        touched_range = proj["range"][touched_keep].float()
        touched_radius = rendered.radii[touched_keep].float()
    else:
        device = alpha.device
        touched_u = torch.zeros((0,), device=device)
        touched_v = torch.zeros((0,), device=device)
        touched_range = torch.zeros((0,), device=device)
        touched_radius = torch.zeros((0,), device=device)

    v_idx, u_idx = torch.where(uncovered)
    ray_range = target["range_image"][uncovered].float()
    nearest = _nearest_gaussian_to_rays(
        ray_v=v_idx.float(),
        ray_u=u_idx.float(),
        ray_range=ray_range,
        prim_u=prim_u,
        prim_v=prim_v,
        prim_range=prim_range,
        prim_radius=prim_radius,
        width=cfg.lidar_width,
        chunk=nearest_chunk,
    )
    nearest_touched = _nearest_gaussian_to_rays(
        ray_v=v_idx.float(),
        ray_u=u_idx.float(),
        ray_range=ray_range,
        prim_u=touched_u,
        prim_v=touched_v,
        prim_range=touched_range,
        prim_radius=touched_radius,
        width=cfg.lidar_width,
        chunk=nearest_chunk,
    )
    pixel_dist = nearest["pixel_dist"]
    radius = nearest["radius"]
    range_gap_abs = nearest["range_gap"].abs()
    close_angular = pixel_dist <= torch.maximum(radius + 1.0, torch.full_like(radius, 2.0))
    close_depth = range_gap_abs <= torch.maximum(1.5 * torch.ones_like(range_gap_abs), 0.10 * ray_range)
    no_near = nearest["idx"] < 0
    angular_gap = (~no_near) & ~close_angular
    depth_gap = (~no_near) & close_angular & ~close_depth
    footprint_or_opacity = (~no_near) & close_angular & close_depth

    row_uncovered = uncovered.long().sum(dim=1)
    row_gt = gt_valid.long().sum(dim=1).clamp(min=1)
    row_rate = row_uncovered.float() / row_gt.float()
    top_rows = _top_counts(uncovered, top_k=8)
    for item in top_rows:
        row = item["index"]
        item["gt_hits"] = int(gt_valid[row].sum().item())
        item["uncovered_rate"] = float(row_rate[row].item())

    col_bins = 36
    col_bin_idx = torch.div(u_idx * col_bins, cfg.lidar_width, rounding_mode="floor").clamp(0, col_bins - 1)
    col_counts = torch.bincount(col_bin_idx, minlength=col_bins)
    top_cols_vals, top_cols_idx = torch.topk(col_counts, k=min(8, col_bins))

    range_bins = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 1e9]
    alpha_on_gt = alpha[gt_valid].float()
    summary = {
        "gt_hits": int(gt_valid.sum().item()),
        "covered_gt_hits": int(covered.sum().item()),
        "uncovered_gt_hits": int(uncovered.sum().item()),
        "gt_coverage_recall": float(covered.sum().item() / max(gt_valid.sum().item(), 1)),
        "false_coverage_rays": int(false_coverage.sum().item()),
        "false_coverage_rate_on_empty": float(false_coverage.sum().item() / max((~gt_valid).sum().item(), 1)),
        "alpha_on_gt": _tensor_stats(alpha_on_gt),
        "alpha_on_empty": _tensor_stats(alpha[(~gt_valid)].float()),
        "rendered_range_on_gt": _tensor_stats(rendered.middepth[gt_valid].float()),
        "gt_range": _tensor_stats(target["range_image"][gt_valid].float()),
        "depth_abs_err_on_gt": _tensor_stats((rendered.middepth[gt_valid] - target["range_image"][gt_valid]).abs().float()),
        "uncovered_range_bins_m": {
            "bins": range_bins,
            "counts": _bin_counts(ray_range, range_bins),
        },
        "uncovered_top_rows": top_rows,
        "uncovered_top_azimuth_bins_10deg": [
            {"bin": int(i.item()), "count": int(v.item())}
            for v, i in zip(top_cols_vals, top_cols_idx)
            if int(v.item()) > 0
        ],
        "nearest_visible_positive_gaussian": {
            "candidate_count": int(prim_u.shape[0]),
            "pixel_dist": _tensor_stats(pixel_dist),
            "abs_range_gap_m": _tensor_stats(range_gap_abs),
            "nearest_radius_px": _tensor_stats(radius),
        },
        "nearest_touched_gaussian": {
            "candidate_count": int(touched_u.shape[0]),
            "pixel_dist": _tensor_stats(nearest_touched["pixel_dist"]),
            "abs_range_gap_m": _tensor_stats(nearest_touched["range_gap"].abs()),
            "nearest_radius_px": _tensor_stats(nearest_touched["radius"]),
        },
        "uncovered_classification": {
            "no_visible_positive_gaussian": int(no_near.sum().item()),
            "angular_gap": int(angular_gap.sum().item()),
            "depth_gap": int(depth_gap.sum().item()),
            "footprint_or_opacity_after_close_match": int(footprint_or_opacity.sum().item()),
        },
        "gaussians": {
            "generated": int(frame.primitives["means3D"].shape[0]),
            "visible_center": int(proj["visible"].sum().item()),
            "positive_radius": int((rendered.radii > 0).sum().item()),
            "touched": int((rendered.n_touched > 0).sum().item()),
            "untouched_positive_radius": int(((rendered.radii > 0) & (rendered.n_touched <= 0)).sum().item()),
            "radius_px": _tensor_stats(rendered.radii.float()),
            "n_touched_tiles": _tensor_stats(rendered.n_touched.float()),
            "primitive_extent": _primitive_extent_summary(frame.primitives),
            "contexts": _context_gaussian_stats(frame.contexts, rendered, frame.viewmatrix, cfg),
        },
    }
    return summary


def _run_case(
    *,
    cfg: QGSConfig,
    sample: dict,
    device: torch.device,
    mode: str,
    model: QGSModel | None,
    init_alpha: float,
    scale_modifier: float,
    alpha_eps: float,
    nearest_chunk: int,
) -> dict:
    frame0_contexts, frame1_contexts, frame_inputs, anchor_summary = _build_scene_and_primitives(
        cfg=cfg,
        sample=sample,
        device=device,
        mode=mode,
        model=model,
        init_alpha=init_alpha,
    )
    rel_pose = frame_inputs["rel_input_1_pose"]
    inv_pose = torch.linalg.inv(rel_pose)
    frame0 = _render_frame_bundle(
        cfg=cfg,
        name="frame0",
        contexts=frame0_contexts,
        target_xyz=frame_inputs["xyz0"],
        target_i=frame_inputs["i0"],
        viewmatrix=torch.eye(4, device=device, dtype=frame_inputs["xyz0"].dtype),
        campos=torch.zeros(3, device=device, dtype=frame_inputs["xyz0"].dtype),
        scale_modifier=scale_modifier,
    )
    frame1 = _render_frame_bundle(
        cfg=cfg,
        name="frame1",
        contexts=frame1_contexts,
        target_xyz=frame_inputs["xyz1"],
        target_i=frame_inputs["i1"],
        viewmatrix=inv_pose,
        campos=rel_pose[:3, 3],
        scale_modifier=scale_modifier,
    )
    return {
        "mode": mode,
        "scale_modifier": float(scale_modifier),
        "quadric_gamma": float(cfg.quadric_gamma),
        "lidar_sigma": float(cfg.lidar_sigma),
        "alpha_eps": float(alpha_eps),
        "anchor_summary": anchor_summary,
        "frame0": _analyze_frame(frame0, cfg, alpha_eps=alpha_eps, nearest_chunk=nearest_chunk),
        "frame1": _analyze_frame(frame1, cfg, alpha_eps=alpha_eps, nearest_chunk=nearest_chunk),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pair-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    parser.add_argument("--mode", choices=["init", "model"], default="init")
    parser.add_argument("--quadric-gammas", default="")
    parser.add_argument("--scale-modifiers", default="1.0")
    parser.add_argument("--lidar-sigma", type=float, default=None)
    parser.add_argument("--alpha-eps", type=float, default=1e-3)
    parser.add_argument("--init-alpha", type=float, default=0.9)
    parser.add_argument("--nearest-chunk", type=int, default=2048)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_cfg_from_checkpoint(args.checkpoint) if args.checkpoint else QGSConfig()
    cfg.device = args.device
    if args.lidar_sigma is not None:
        cfg.lidar_sigma = float(args.lidar_sigma)
    gammas = _parse_csv_floats(args.quadric_gammas) if args.quadric_gammas else [float(cfg.quadric_gamma)]
    scale_modifiers = _parse_csv_floats(args.scale_modifiers)

    dataset = _load_dataset(cfg, args.split)
    sample = dataset[args.pair_idx]

    model = None
    if args.mode == "model":
        model = QGSModel(cfg).to(device)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

    cases = []
    with torch.no_grad():
        for gamma in gammas:
            cfg.quadric_gamma = float(gamma)
            for scale_modifier in scale_modifiers:
                case = _run_case(
                    cfg=cfg,
                    sample=sample,
                    device=device,
                    mode=args.mode,
                    model=model,
                    init_alpha=float(args.init_alpha),
                    scale_modifier=float(scale_modifier),
                    alpha_eps=float(args.alpha_eps),
                    nearest_chunk=int(args.nearest_chunk),
                )
                cases.append(case)

    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "pair_idx": int(args.pair_idx),
        "device": args.device,
        "primitive_mode": cfg.primitive_mode,
        "lidar_height": cfg.lidar_height,
        "lidar_width": cfg.lidar_width,
        "cases": cases,
    }
    text = json.dumps(report, indent=2)
    if args.quiet:
        print(f"wrote {len(cases)} cases for pair_idx={args.pair_idx} mode={args.mode}")
        for case in cases:
            parts = [
                f"gamma={case['quadric_gamma']}",
                f"scale={case['scale_modifier']}",
            ]
            for frame_name in ("frame0", "frame1"):
                frame = case[frame_name]
                parts.append(
                    f"{frame_name}:recall={frame['gt_coverage_recall']:.4f}"
                    f",uncovered={frame['uncovered_gt_hits']}"
                    f",false={frame['false_coverage_rays']}"
                    f",touched={frame['gaussians']['touched']}"
                )
            print(" ".join(parts))
    else:
        print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
