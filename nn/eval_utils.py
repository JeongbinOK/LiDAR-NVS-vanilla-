"""Pair-wise evaluation helpers for Phase A QGS."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from config import QGSConfig
from models.geometry import decompose_scene
from models.head import make_lidar_ray_grid
from nn.qgs_loss import QGSLoss
from nn.render_utils import quat_to_rotmat, render_primitives, rotmat_to_quat, build_target_lidar_image


def load_cfg_from_checkpoint(checkpoint_path: str) -> QGSConfig:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    run_dir = os.path.dirname(ckpt_dir)
    config_path = os.path.join(run_dir, "configs", "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    allowed = QGSConfig.__dataclass_fields__.keys()
    return QGSConfig(**{k: v for k, v in raw.items() if k in allowed})


def resolve_bbox_json(cfg: QGSConfig, split: str | None = None) -> str | None:
    target_split = split or cfg.train_split
    bbox_json = cfg.bbox_json_path.format(split=target_split) if cfg.bbox_json_path else ""
    if not bbox_json:
        return None
    data_root = os.path.expanduser(cfg.data_root)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(bbox_json):
        candidates = [os.path.join(data_root, bbox_json), os.path.join(repo_root, bbox_json)]
        bbox_json = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    return bbox_json if os.path.isfile(bbox_json) else None


def _ensure_2d_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.dim() == 3 and pose.shape[0] == 1:
        return pose[0]
    return pose


def filter_frame_points(xyz: torch.Tensor, intensity: torch.Tensor, cfg: QGSConfig):
    r = xyz.norm(dim=1)
    keep = (r > cfg.ego_radius) & (r < cfg.r_far)
    return xyz[keep], intensity[keep]


def _cuda_memory_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "alloc_mb": round(float(torch.cuda.memory_allocated(device) / 1024**2), 1),
        "reserved_mb": round(float(torch.cuda.memory_reserved(device) / 1024**2), 1),
        "max_alloc_mb": round(float(torch.cuda.max_memory_allocated(device) / 1024**2), 1),
        "max_reserved_mb": round(float(torch.cuda.max_memory_reserved(device) / 1024**2), 1),
    }


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _record_memory(stage: str, device: torch.device, records: list[dict]) -> None:
    if device.type != "cuda":
        return
    records.append({"stage": stage, **_cuda_memory_snapshot(device)})


def _peak_from_records(records: list[dict]) -> dict:
    if not records:
        return {"max_alloc_mb": 0.0, "max_reserved_mb": 0.0}
    return {
        "max_alloc_mb": max(float(r.get("max_alloc_mb", 0.0)) for r in records),
        "max_reserved_mb": max(float(r.get("max_reserved_mb", 0.0)) for r in records),
    }


def _record_memory_with_peak(
    stage: str,
    device: torch.device,
    records: list[dict],
    *,
    peak_records: list[dict] | None = None,
) -> None:
    if device.type != "cuda":
        return
    snapshot = _cuda_memory_snapshot(device)
    if peak_records:
        peak = _peak_from_records(peak_records)
        snapshot["max_alloc_mb"] = max(snapshot["max_alloc_mb"], peak["max_alloc_mb"])
        snapshot["max_reserved_mb"] = max(snapshot["max_reserved_mb"], peak["max_reserved_mb"])
    records.append({"stage": stage, **snapshot})


def _box_to_pose(box: torch.Tensor) -> torch.Tensor:
    c = torch.cos(box[6])
    s = torch.sin(box[6])
    pose = torch.eye(4, device=box.device, dtype=box.dtype)
    pose[0, 0] = c
    pose[0, 1] = -s
    pose[1, 0] = s
    pose[1, 1] = c
    pose[0, 3] = box[0]
    pose[1, 3] = box[1]
    pose[2, 3] = box[2]
    return pose


def transform_primitives(primitives: dict, T: torch.Tensor) -> dict:
    R = quat_to_rotmat(primitives["rotations"])
    R_out = T[:3, :3] @ R
    t = T[:3, 3]
    means = (T[:3, :3] @ primitives["means3D"].T).T + t
    return {
        "means3D": means,
        "scales": primitives["scales"],
        "rotations": rotmat_to_quat(R_out),
        "opacities": primitives["opacities"],
        "intensity": primitives["intensity"],
        "latent": primitives["latent"],
    }


def concat_primitives(primitives: list[dict]) -> dict:
    keys = ["means3D", "scales", "rotations", "opacities", "intensity", "latent"]
    return {key: torch.cat([p[key] for p in primitives], dim=0) for key in keys}


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.mean().item())


def _safe_max(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.max().item())


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x, q).item())


def _context_diagnostics(name: str, primitives: dict, num_points: int) -> dict:
    aux = primitives["aux"]
    geom = primitives["geom_init"]
    g_rot = aux["g_rot"][0]
    g_scale = aux["g_scale"][0]
    omega_local = aux["omega_local"][0]
    delta_mu = aux["delta_mu"][0]
    delta_gap = aux["delta_gap"][0]
    delta_log_abs_s3 = aux["delta_log_abs_s3"][0]
    alpha = primitives["opacities"].squeeze(-1)
    k_eff = primitives["k_eff"].float()
    used_init = geom["use_geom_init"][0].float()
    tangent_aniso = geom["tangent_aniso"][0]
    curvature_aniso = geom["curvature_aniso"][0]
    kappa1 = geom["kappa1_init"][0]
    kappa2 = geom["kappa2_init"][0]
    scales = primitives["scales"]
    omega_deg = omega_local.abs() * (180.0 / math.pi)
    return {
        "name": name,
        "n_input_points": int(num_points),
        "n_generated": int(primitives["means3D"].shape[0]),
        "skipped": False,
        "skip_reason": None,
        "center_mode": aux.get("center_mode", "fixed"),
        "used_geom_ratio": float(used_init.mean().item()),
        "all_points_subfloor": bool((used_init == 0).all().item()),
        "k_eff_mean": _safe_mean(k_eff),
        "k_eff_p95": _safe_quantile(k_eff, 0.95),
        "g_rot_mean": _safe_mean(g_rot),
        "g_rot_p95": _safe_quantile(g_rot, 0.95),
        "g_scale_mean": _safe_mean(g_scale),
        "g_scale_p95": _safe_quantile(g_scale, 0.95),
        "omega_abs_max_deg": _safe_max(omega_deg),
        "omega_tilt_abs_max_deg": _safe_max(omega_deg[..., :2]),
        "omega_spin_abs_max_deg": _safe_max(omega_deg[..., 2]),
        "delta_mu_abs_mean": _safe_mean(delta_mu.abs()),
        "delta_mu_abs_max": _safe_max(delta_mu.abs()),
        "delta_gap_abs_mean": _safe_mean(delta_gap.abs()),
        "delta_gap_abs_max": _safe_max(delta_gap.abs()),
        "delta_log_abs_s3_abs_mean": _safe_mean(delta_log_abs_s3.abs()),
        "delta_log_abs_s3_abs_max": _safe_max(delta_log_abs_s3.abs()),
        "tangent_aniso_mean": _safe_mean(tangent_aniso),
        "tangent_aniso_p95": _safe_quantile(tangent_aniso, 0.95),
        "curvature_aniso_mean": _safe_mean(curvature_aniso),
        "curvature_aniso_p95": _safe_quantile(curvature_aniso, 0.95),
        "kappa1_abs_mean": _safe_mean(kappa1.abs()),
        "kappa2_abs_mean": _safe_mean(kappa2.abs()),
        "ordered_scale_violations": int((scales[:, 0] < scales[:, 1]).sum().item()),
        "alpha_mean": _safe_mean(alpha),
        "alpha_p95": _safe_quantile(alpha, 0.95),
        "memory_stages": primitives.get("diagnostics", {}).get("memory_stages", []),
    }


def _skipped_context_diagnostics(name: str, num_points: int, reason: str) -> dict:
    return {
        "name": name,
        "n_input_points": int(num_points),
        "n_generated": 0,
        "skipped": True,
        "skip_reason": reason,
        "center_mode": None,
        "used_geom_ratio": None,
        "all_points_subfloor": None,
        "k_eff_mean": None,
        "k_eff_p95": None,
        "g_rot_mean": None,
        "g_rot_p95": None,
        "g_scale_mean": None,
        "g_scale_p95": None,
        "omega_abs_max_deg": None,
        "omega_tilt_abs_max_deg": None,
        "omega_spin_abs_max_deg": None,
        "delta_mu_abs_mean": None,
        "delta_mu_abs_max": None,
        "delta_gap_abs_mean": None,
        "delta_gap_abs_max": None,
        "delta_log_abs_s3_abs_mean": None,
        "delta_log_abs_s3_abs_max": None,
        "tangent_aniso_mean": None,
        "tangent_aniso_p95": None,
        "curvature_aniso_mean": None,
        "curvature_aniso_p95": None,
        "kappa1_abs_mean": None,
        "kappa2_abs_mean": None,
        "ordered_scale_violations": None,
        "alpha_mean": None,
        "alpha_p95": None,
        "memory_stages": [],
    }


def build_context_primitives(
    model,
    cfg: QGSConfig,
    name: str,
    xyz: torch.Tensor,
    intensity_norm: torch.Tensor,
    *,
    time_scalar: torch.Tensor | None,
    ego_motion: torch.Tensor,
    is_dynamic_flag: torch.Tensor,
) -> tuple[dict | None, dict | None]:
    if xyz.shape[0] < cfg.knn_k_min:
        return None, None
    primitives = model.forward_context(
        xyz,
        intensity_norm,
        time_scalar=time_scalar,
        ego_motion=ego_motion,
        is_dynamic_flag=is_dynamic_flag,
        neighbor_xyz=xyz,
        return_diagnostics=True,
    )
    return primitives, _context_diagnostics(name, primitives, xyz.shape[0])


def _sensor_visibility_mask(
    means3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    *,
    el_min_rad: float,
    el_max_rad: float,
    r_near: float,
    r_far: float,
) -> torch.Tensor:
    if means3D.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=means3D.device)
    sensor_xyz = (viewmatrix[:3, :3] @ means3D.T).T + viewmatrix[:3, 3]
    x, y, z = sensor_xyz.unbind(dim=-1)
    r = sensor_xyz.norm(dim=-1)
    xy = torch.sqrt((x * x + y * y).clamp(min=1e-12))
    el = torch.atan2(z, xy)
    return (
        torch.isfinite(r)
        & (r >= r_near)
        & (r <= r_far)
        & (el >= el_min_rad)
        & (el <= el_max_rad)
    )


def gaussian_slice_stats(
    primitives: dict,
    radii: torch.Tensor,
    n_touched: torch.Tensor,
    viewmatrix: torch.Tensor,
    cfg: QGSConfig,
) -> dict:
    el_min_rad = math.radians(cfg.lidar_el_min_deg)
    el_max_rad = math.radians(cfg.lidar_el_max_deg)
    visible = _sensor_visibility_mask(
        primitives["means3D"],
        viewmatrix,
        el_min_rad=el_min_rad,
        el_max_rad=el_max_rad,
        r_near=cfg.r_near,
        r_far=cfg.r_far,
    )
    positive_radius = radii > 0
    touched = n_touched > 0
    n_generated = int(primitives["means3D"].shape[0])
    n_visible = int(visible.sum().item())
    n_radius = int(positive_radius.sum().item())
    n_touched_total = int(touched.sum().item())
    return {
        "n_generated": n_generated,
        "n_frustum_visible": n_visible,
        "n_positive_radius": n_radius,
        "n_touched": n_touched_total,
        "n_dropped_out_of_fov": n_generated - n_visible,
        "n_dropped_zero_radius": max(n_visible - n_radius, 0),
        "n_dropped_untouched": max(n_radius - n_touched_total, 0),
        "n_dropped_raster": n_generated - n_touched_total,
    }


def frame_gaussian_stats(
    frame_contexts: list[dict],
    rendered,
    viewmatrix: torch.Tensor,
    cfg: QGSConfig,
) -> dict:
    context_stats = []
    offset = 0
    totals = {
        "n_generated": 0,
        "n_frustum_visible": 0,
        "n_positive_radius": 0,
        "n_touched": 0,
        "n_dropped_out_of_fov": 0,
        "n_dropped_zero_radius": 0,
        "n_dropped_untouched": 0,
        "n_dropped_raster": 0,
    }
    for ctx in frame_contexts:
        n_ctx = ctx["primitives"]["means3D"].shape[0]
        stats = gaussian_slice_stats(
            ctx["primitives"],
            rendered.radii[offset:offset + n_ctx],
            rendered.n_touched[offset:offset + n_ctx],
            viewmatrix,
            cfg,
        )
        context_stats.append({
            "name": ctx["name"],
            "context": ctx["context"],
            "gaussians": stats,
        })
        offset += n_ctx
        for key, value in stats.items():
            totals[key] += value
    return {"overall": totals, "contexts": context_stats}


def range_image_to_points(ray_grid: torch.Tensor, range_image: torch.Tensor, valid_mask: torch.Tensor):
    pts = (ray_grid * range_image.unsqueeze(0)).permute(1, 2, 0)
    return pts[valid_mask]


def approx_chamfer(pred_pts: torch.Tensor, gt_pts: torch.Tensor, max_points: int = 2048) -> float:
    if pred_pts.numel() == 0 or gt_pts.numel() == 0:
        return float("inf")
    if pred_pts.shape[0] > max_points:
        pred_pts = pred_pts[torch.randperm(pred_pts.shape[0], device=pred_pts.device)[:max_points]]
    if gt_pts.shape[0] > max_points:
        gt_pts = gt_pts[torch.randperm(gt_pts.shape[0], device=gt_pts.device)[:max_points]]
    d = torch.cdist(pred_pts, gt_pts)
    return float(0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()))


def frame_metrics(
    name: str,
    frame_data: dict,
    ray_grid: torch.Tensor,
    hit_threshold: float,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rendered = frame_data["rendered"]
    target = frame_data["target"]
    drop = frame_data["drop"]
    loss = frame_data["loss"]

    gt_valid = target["valid_mask"]
    pred_valid = drop < (1.0 - hit_threshold)
    intersect = pred_valid & gt_valid

    alpha_safe = rendered.alpha_accum.clamp(min=1e-3)
    pred_intensity = rendered.intensity / alpha_safe
    pred_range = rendered.middepth

    gt_pts = range_image_to_points(ray_grid, target["range_image"], gt_valid)
    pred_pts = range_image_to_points(ray_grid, pred_range, pred_valid)

    precision = float(intersect.sum().item() / max(pred_valid.sum().item(), 1))
    recall = float(intersect.sum().item() / max(gt_valid.sum().item(), 1))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "frame": name,
        "loss_total": float(loss["total"].item()),
        "loss_depth": float(loss["depth"].item()),
        "loss_intensity": float(loss["intensity"].item()),
        "loss_raydrop": float(loss["raydrop"].item()),
        "gt_hits": int(gt_valid.sum().item()),
        "pred_hits": int(pred_valid.sum().item()),
        "n_gt_drop_rays": int((~gt_valid).sum().item()),
        "n_pred_drop_rays": int((~pred_valid).sum().item()),
        "hit_precision": precision,
        "hit_recall": recall,
        "hit_f1": f1,
        "depth_mae_on_gt": float((pred_range[gt_valid] - target["range_image"][gt_valid]).abs().mean().item()),
        "depth_mae_on_intersect": float((pred_range[intersect] - target["range_image"][intersect]).abs().mean().item()) if intersect.any() else float("inf"),
        "intensity_l1_on_gt": float((pred_intensity[gt_valid] - target["intensity_image"][gt_valid]).abs().mean().item()),
        "alpha_mean_on_gt": float(rendered.alpha_accum[gt_valid].mean().item()),
        "drop_mean_on_gt": float(drop[gt_valid].mean().item()),
        "approx_chamfer_m": approx_chamfer(pred_pts, gt_pts),
    }
    return metrics, pred_pts, gt_pts, pred_intensity[pred_valid], target["intensity_image"][gt_valid]


def write_ply(path: Path, xyz: torch.Tensor, intensity: torch.Tensor | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_cpu = xyz.detach().cpu().float()
    intensity_cpu = None if intensity is None else intensity.detach().cpu().float()
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz_cpu.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if intensity_cpu is not None:
            f.write("property float intensity\n")
        f.write("end_header\n")
        if intensity_cpu is None:
            for p in xyz_cpu:
                f.write(f"{p[0].item()} {p[1].item()} {p[2].item()}\n")
        else:
            for p, i in zip(xyz_cpu, intensity_cpu):
                f.write(f"{p[0].item()} {p[1].item()} {p[2].item()} {i.item()}\n")


def evaluate_pair_sample(
    model,
    drop_head,
    loss_fn: QGSLoss,
    sample: dict,
    *,
    device: str,
    cfg: QGSConfig,
    hit_threshold: float = 0.5,
) -> dict:
    device_t = torch.device(device)
    memory_records: list[dict] = []

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

    _reset_cuda_peak(device_t)
    _record_memory("start", device_t, memory_records)
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
    _record_memory("scene_decomposition", device_t, memory_records)

    e_dir = rel_input_1_pose[:3, 3].to(dtype=xyz0.dtype, device=device)
    inv_pose = torch.linalg.inv(rel_input_1_pose)

    frame0_contexts: list[dict] = []
    frame1_contexts: list[dict] = []
    context_summaries: list[dict] = []

    _reset_cuda_peak(device_t)
    static_prims, static_diag = build_context_primitives(
        model,
        cfg,
        "static",
        scene["static_xyz"].to(device),
        scene["static_intensity"].to(device),
        time_scalar=scene["static_time"].to(device),
        ego_motion=e_dir,
        is_dynamic_flag=torch.zeros(scene["static_xyz"].shape[0], device=device, dtype=xyz0.dtype),
    )
    _record_memory_with_peak(
        "static_context",
        device_t,
        memory_records,
        peak_records=None if static_diag is None else static_diag.get("memory_stages", []),
    )
    if static_prims is not None and static_diag is not None:
        frame0_contexts.append({"name": "static", "primitives": static_prims, "context": static_diag})
        frame1_contexts.append({"name": "static", "primitives": static_prims, "context": static_diag})
        context_summaries.append(static_diag)
    else:
        context_summaries.append(
            _skipped_context_diagnostics(
                "static",
                scene["static_xyz"].shape[0],
                f"n_input_points<{cfg.knn_k_min}",
            )
        )

    for dyn_idx, dyn in enumerate(scene["dynamic"]):
        _reset_cuda_peak(device_t)
        dyn_prims, dyn_diag = build_context_primitives(
            model,
            cfg,
            f"dynamic_{dyn_idx}",
            dyn["canonical_xyz"].to(device),
            dyn["canonical_intensity"].to(device),
            time_scalar=dyn["canonical_time"].to(device),
            ego_motion=e_dir,
            is_dynamic_flag=torch.ones(dyn["canonical_xyz"].shape[0], device=device, dtype=xyz0.dtype),
        )
        _record_memory_with_peak(
            f"dynamic_context_{dyn_idx}",
            device_t,
            memory_records,
            peak_records=None if dyn_diag is None else dyn_diag.get("memory_stages", []),
        )
        if dyn_prims is None or dyn_diag is None:
            context_summaries.append(
                _skipped_context_diagnostics(
                    f"dynamic_{dyn_idx}",
                    dyn["canonical_xyz"].shape[0],
                    f"n_input_points<{cfg.knn_k_min}",
                )
            )
            continue
        context_summaries.append(dyn_diag)
        if dyn.get("box_0") is not None:
            frame0_contexts.append({
                "name": f"dynamic_{dyn_idx}",
                "primitives": transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device))),
                "context": dyn_diag,
            })
        if dyn.get("box_1") is not None:
            frame1_contexts.append({
                "name": f"dynamic_{dyn_idx}",
                "primitives": transform_primitives(dyn_prims, _box_to_pose(dyn["box_1"].to(device))),
                "context": dyn_diag,
            })

    if not frame0_contexts or not frame1_contexts:
        raise RuntimeError("no primitives were produced for evaluation")

    frame0 = concat_primitives([ctx["primitives"] for ctx in frame0_contexts])
    frame1 = concat_primitives([ctx["primitives"] for ctx in frame1_contexts])

    el_min_rad = math.radians(cfg.lidar_el_min_deg)
    el_max_rad = math.radians(cfg.lidar_el_max_deg)
    ray_grid = make_lidar_ray_grid(cfg.lidar_height, cfg.lidar_width, el_min_rad, el_max_rad, device=device)

    target0 = build_target_lidar_image(
        xyz0, i0,
        height=cfg.lidar_height, width=cfg.lidar_width,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        r_near=cfg.r_near, r_far=cfg.r_far,
    )
    target1 = build_target_lidar_image(
        xyz1, i1,
        height=cfg.lidar_height, width=cfg.lidar_width,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        r_near=cfg.r_near, r_far=cfg.r_far,
    )

    _reset_cuda_peak(device_t)
    rendered0 = render_primitives(
        frame0,
        height=cfg.lidar_height, width=cfg.lidar_width,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        sigma=cfg.lidar_sigma,
        r_near=cfg.r_near, r_far=cfg.r_far,
        viewmatrix=torch.eye(4, device=device, dtype=xyz0.dtype),
        campos=torch.zeros(3, device=device, dtype=xyz0.dtype),
    )
    _record_memory("render_frame0", device_t, memory_records)
    _reset_cuda_peak(device_t)
    rendered1 = render_primitives(
        frame1,
        height=cfg.lidar_height, width=cfg.lidar_width,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        sigma=cfg.lidar_sigma,
        r_near=cfg.r_near, r_far=cfg.r_far,
        viewmatrix=inv_pose,
        campos=rel_input_1_pose[:3, 3],
    )
    _record_memory("render_frame1", device_t, memory_records)

    _reset_cuda_peak(device_t)
    _, drop0 = drop_head(
        rendered0.latent.unsqueeze(0),
        rendered0.range.unsqueeze(0),
        rendered0.normal.unsqueeze(0),
        rendered0.curvature.unsqueeze(0),
        rendered0.alpha_accum.unsqueeze(0),
        ray_grid,
    )
    _, drop1 = drop_head(
        rendered1.latent.unsqueeze(0),
        rendered1.range.unsqueeze(0),
        rendered1.normal.unsqueeze(0),
        rendered1.curvature.unsqueeze(0),
        rendered1.alpha_accum.unsqueeze(0),
        ray_grid,
    )
    _record_memory("drop_head", device_t, memory_records)

    loss0 = loss_fn(rendered0, target0, drop0)
    loss1 = loss_fn(rendered1, target1, drop1)
    total_loss = loss0["total"] + loss1["total"]

    frame0_data = {
        "rendered": rendered0,
        "target": target0,
        "drop": drop0[0],
        "loss": loss0,
    }
    frame1_data = {
        "rendered": rendered1,
        "target": target1,
        "drop": drop1[0],
        "loss": loss1,
    }

    frame0_metrics, pred0_pts, gt0_pts, pred0_i, gt0_i = frame_metrics("frame0", frame0_data, ray_grid, hit_threshold)
    frame1_metrics, pred1_pts, gt1_pts, pred1_i, gt1_i = frame_metrics("frame1", frame1_data, ray_grid, hit_threshold)

    frame0_metrics["gaussians"] = frame_gaussian_stats(frame0_contexts, rendered0, torch.eye(4, device=device, dtype=xyz0.dtype), cfg)
    frame1_metrics["gaussians"] = frame_gaussian_stats(frame1_contexts, rendered1, inv_pose, cfg)

    context_peak_alloc_mb = max(
        (
            _peak_from_records(ctx["memory_stages"])["max_alloc_mb"]
            if ctx["memory_stages"] else 0.0
            for ctx in context_summaries
        ),
        default=0.0,
    )
    context_peak_reserved_mb = max(
        (
            _peak_from_records(ctx["memory_stages"])["max_reserved_mb"]
            if ctx["memory_stages"] else 0.0
            for ctx in context_summaries
        ),
        default=0.0,
    )
    outer_peak_alloc_mb = max((float(r.get("max_alloc_mb", 0.0)) for r in memory_records), default=0.0)
    outer_peak_reserved_mb = max((float(r.get("max_reserved_mb", 0.0)) for r in memory_records), default=0.0)

    summary = {
        "total_loss": float(total_loss.item()),
        "scene": {
            "filtered_points": {"frame0": int(xyz0.shape[0]), "frame1": int(xyz1.shape[0])},
            "static_points": int(scene["static_xyz"].shape[0]),
            "dynamic_instances": len(scene["dynamic"]),
            "realized_contexts": sum(0 if ctx["skipped"] else 1 for ctx in context_summaries),
            "skipped_contexts": sum(1 if ctx["skipped"] else 0 for ctx in context_summaries),
            "untracked_stats": scene["untracked_stats"],
            "contexts": context_summaries,
        },
        "memory": memory_records,
        "memory_overview": {
            "peak_alloc_mb": max(context_peak_alloc_mb, outer_peak_alloc_mb),
            "peak_reserved_mb": max(context_peak_reserved_mb, outer_peak_reserved_mb),
            "context_peak_alloc_mb": context_peak_alloc_mb,
            "context_peak_reserved_mb": context_peak_reserved_mb,
            "outer_peak_alloc_mb": outer_peak_alloc_mb,
            "outer_peak_reserved_mb": outer_peak_reserved_mb,
        },
        "frame0": frame0_metrics,
        "frame1": frame1_metrics,
    }

    return {
        "summary": summary,
        "artifacts": {
            "pred_frame0_pts": pred0_pts,
            "gt_frame0_pts": gt0_pts,
            "pred_frame1_pts": pred1_pts,
            "gt_frame1_pts": gt1_pts,
            "pred_frame0_intensity": pred0_i,
            "gt_frame0_intensity": gt0_i,
            "pred_frame1_intensity": pred1_i,
            "gt_frame1_intensity": gt1_i,
        },
    }
