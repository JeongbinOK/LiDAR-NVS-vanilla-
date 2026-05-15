"""Pair-wise evaluation helpers for Phase A QGS."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from config import QGSConfig
from models.geometry import decompose_scene
from models.geometry.voxel_anchor import DynamicVoxelAnchorBuilder, VoxelAnchorBuilder
from nn.lidar_geometry import (
    make_lidar_ray_grid,
    points_to_lidar_maps,
    range_map_to_points,
)
from nn.qgs_loss import QGSLoss
from nn.render_utils import quat_to_rotmat, render_primitives, rotmat_to_quat


def load_cfg_from_checkpoint(checkpoint_path: str) -> QGSConfig:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    run_dir = os.path.dirname(ckpt_dir)
    config_path = os.path.join(run_dir, "configs", "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    if "primitive_mode" not in raw:
        raw["primitive_mode"] = "per_point"
    if "ptv3_model_in_channels" not in raw:
        raw["ptv3_model_in_channels"] = raw.get("input_feature_dim", QGSConfig.input_feature_dim)
    if "ptv3_decoupled_stem" not in raw:
        raw["ptv3_decoupled_stem"] = False
    if "ptv3_pdnorm_bn" not in raw:
        raw["ptv3_pdnorm_bn"] = False
    if "ptv3_pdnorm_ln" not in raw:
        raw["ptv3_pdnorm_ln"] = False
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


def _box1_to_frame0_pose(box: torch.Tensor, rel_input_1_pose: torch.Tensor) -> torch.Tensor:
    rel = rel_input_1_pose.to(device=box.device, dtype=box.dtype)
    return rel @ _box_to_pose(box)


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


def _context_group_name(ctx: dict | str) -> str:
    if isinstance(ctx, dict):
        context_type = ctx.get("context_type")
        if context_type in {"static", "dynamic"}:
            return context_type
        name = str(ctx.get("name", ""))
    else:
        name = str(ctx)
    if name == "static":
        return "static"
    if name.startswith("dynamic"):
        return "dynamic"
    raise ValueError(f"cannot infer context group from {ctx!r}")


def _summarize_context_groups(context_summaries: list[dict]) -> dict:
    groups = {
        "static": {
            "realized_contexts": 0,
            "skipped_contexts": 0,
            "input_points": 0,
            "points_with_init": 0,
            "points_without_init": 0,
            "points_subfloor": 0,
        },
        "dynamic": {
            "realized_contexts": 0,
            "skipped_contexts": 0,
            "input_points": 0,
            "points_with_init": 0,
            "points_without_init": 0,
            "points_subfloor": 0,
        },
    }
    for ctx in context_summaries:
        group = groups[_context_group_name(ctx)]
        group["input_points"] += int(ctx["n_input_points"])
        if ctx["skipped"]:
            group["skipped_contexts"] += 1
        else:
            group["realized_contexts"] += 1
        group["points_with_init"] += int(ctx.get("n_points_with_init") or 0)
        group["points_without_init"] += int(ctx.get("n_points_without_init") or 0)
        group["points_subfloor"] += int(ctx.get("n_points_subfloor") or 0)
    return groups


def _context_diagnostics(
    name: str,
    context_type: str,
    primitives: dict,
    num_points: int,
) -> dict:
    aux = primitives["aux"]
    geom = primitives["geom_init"]
    omega_local = aux["omega_local"][0]
    delta_c = aux["delta_c"][0]
    delta_mu = aux["delta_mu"][0]
    delta_gap = aux["delta_gap"][0]
    delta_log_abs_s3 = aux["delta_log_abs_s3"][0]
    alpha = primitives["opacities"].squeeze(-1)
    k_eff = primitives["k_eff"].float()
    used_init_bool = geom["use_geom_init"][0]
    used_init = used_init_bool.float()
    tangent_aniso = geom["tangent_aniso"][0]
    curvature_aniso = geom["curvature_aniso"][0]
    kappa1 = geom["kappa1_init"][0]
    kappa2 = geom["kappa2_init"][0]
    scales = primitives["scales"]
    omega_deg = omega_local.abs() * (180.0 / math.pi)
    n_points_with_init = int(used_init_bool.sum().item())
    n_points_without_init = int((~used_init_bool).sum().item())
    return {
        "name": name,
        "context_type": context_type,
        "n_input_points": int(num_points),
        "n_generated": int(primitives["means3D"].shape[0]),
        "skipped": False,
        "skip_reason": None,
        "used_geom_ratio": float(used_init.mean().item()),
        "all_points_subfloor": bool(n_points_without_init == int(num_points)),
        "n_points_with_init": n_points_with_init,
        "n_points_without_init": n_points_without_init,
        "n_points_subfloor": n_points_without_init,
        "k_eff_mean": _safe_mean(k_eff),
        "k_eff_p95": _safe_quantile(k_eff, 0.95),
        "omega_abs_max_deg": _safe_max(omega_deg),
        "omega_tilt_abs_max_deg": _safe_max(omega_deg[..., :2]),
        "omega_spin_abs_max_deg": _safe_max(omega_deg[..., 2]),
        "delta_c_abs_mean": _safe_mean(delta_c.norm(dim=-1)),
        "delta_c_abs_max": _safe_max(delta_c.norm(dim=-1)),
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
        "ordered_scale_violations": int((scales[:, 0].abs() < scales[:, 1].abs()).sum().item()),
        "alpha_mean": _safe_mean(alpha),
        "alpha_p95": _safe_quantile(alpha, 0.95),
        "memory_stages": primitives.get("diagnostics", {}).get("memory_stages", []),
    }


def _skipped_context_diagnostics(
    name: str,
    context_type: str,
    num_points: int,
    reason: str,
) -> dict:
    return {
        "name": name,
        "context_type": context_type,
        "n_input_points": int(num_points),
        "n_generated": 0,
        "skipped": True,
        "skip_reason": reason,
        "used_geom_ratio": None,
        "all_points_subfloor": None,
        "n_points_with_init": None,
        "n_points_without_init": None,
        "n_points_subfloor": None,
        "k_eff_mean": None,
        "k_eff_p95": None,
        "omega_abs_max_deg": None,
        "omega_tilt_abs_max_deg": None,
        "omega_spin_abs_max_deg": None,
        "delta_c_abs_mean": None,
        "delta_c_abs_max": None,
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
    context_type: str,
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
        context_type=context_type,
        time_scalar=time_scalar,
        ego_motion=ego_motion,
        is_dynamic_flag=is_dynamic_flag,
        neighbor_xyz=xyz,
        return_diagnostics=True,
    )
    return primitives, _context_diagnostics(name, context_type, primitives, xyz.shape[0])


def _make_static_anchor_builder(cfg: QGSConfig) -> VoxelAnchorBuilder:
    return VoxelAnchorBuilder(
        k_min=cfg.knn_k_min,
        k_target=cfg.knn_k_target,
        residual_threshold=cfg.anchor_residual_threshold,
        filter_mode=cfg.anchor_filter_mode,
        planarity_threshold=cfg.anchor_planarity_threshold,
        token_variant=cfg.anchor_token_variant,
        knn_chunk_size=min(cfg.knn_chunk_size, 256),
        quadric_gamma=cfg.quadric_gamma,
        quadric_kappa_max=cfg.quadric_kappa_max,
        quadric_eps_lambda=cfg.quadric_eps_lambda,
        quadric_eps_kappa=cfg.quadric_eps_kappa,
        quadric_eps_s=cfg.quadric_eps_s,
        quadric_eps_s3=cfg.quadric_eps_s3,
    )


def _make_dynamic_anchor_builder(cfg: QGSConfig) -> DynamicVoxelAnchorBuilder:
    return DynamicVoxelAnchorBuilder(
        k_min=cfg.knn_k_min,
        k_target=cfg.knn_k_target,
        residual_threshold=cfg.anchor_residual_threshold,
        filter_mode=cfg.anchor_filter_mode,
        planarity_threshold=cfg.anchor_planarity_threshold,
        token_variant=cfg.anchor_token_variant,
        knn_chunk_size=min(cfg.knn_chunk_size, 256),
        quadric_gamma=cfg.quadric_gamma,
        quadric_kappa_max=cfg.quadric_kappa_max,
        quadric_eps_lambda=cfg.quadric_eps_lambda,
        quadric_eps_kappa=cfg.quadric_eps_kappa,
        quadric_eps_s=cfg.quadric_eps_s,
        quadric_eps_s3=cfg.quadric_eps_s3,
    )


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
    from nn.lidar_geometry import get_effective_el_bounds
    el_min_rad, el_max_rad = get_effective_el_bounds(cfg)
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

    pred_intensity = rendered.intensity
    pred_range = rendered.middepth

    def _unproject(rng: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (ray_grid * rng.unsqueeze(0)).permute(1, 2, 0)[mask]

    gt_pts = _unproject(target["range_image"], gt_valid)
    pred_pts = _unproject(pred_range, pred_valid)

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

    anchor_summary = {
        "primitive_mode": cfg.primitive_mode,
        "input_points": int(scene["static_xyz"].shape[0])
        + sum(int(d["canonical_xyz"].shape[0]) for d in scene["dynamic"]),
        "query_voxels": 0,
        "final_primitives": 0,
        "static_primitives": 0,
        "dynamic_primitives": 0,
        "fallback_instance_count": 0,
        "fallback_point_count": 0,
        "static_init_count": 0,
        "static_fallback_count": 0,
        "dynamic_init_count": 0,
        "dynamic_fallback_count": 0,
    }

    if cfg.primitive_mode == "per_point":
        _reset_cuda_peak(device_t)
        static_prims, static_diag = build_context_primitives(
            model,
            cfg,
            "static",
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
                "dynamic",
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
                skipped = _skipped_context_diagnostics(
                    f"dynamic_{dyn_idx}",
                    "dynamic",
                    dyn["canonical_xyz"].shape[0],
                    f"n_input_points<{cfg.knn_k_min}",
                )
                skipped["instance_id"] = int(dyn["instance_id"])
                context_summaries.append(skipped)
                continue
            dyn_diag["instance_id"] = int(dyn["instance_id"])
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
                    "primitives": transform_primitives(
                        dyn_prims,
                        _box1_to_frame0_pose(dyn["box_1"].to(device), rel_input_1_pose),
                    ),
                    "context": dyn_diag,
                })
    elif cfg.primitive_mode == "voxel_anchor":
        dyn_anchor_outputs = []
        dyn_prims_list = []
        if scene["dynamic"]:
            dynamic_builder = _make_dynamic_anchor_builder(cfg)
            dyn_anchor_pairs = dynamic_builder(scene["dynamic"])
            dyn_anchor_outputs = [out for _, out in dyn_anchor_pairs]
            dyn_prims_list = model.forward_anchor_contexts_batched(
                dyn_anchor_outputs,
                context_type="dynamic",
            )

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
                skipped = _skipped_context_diagnostics(
                    f"dynamic_{dyn_idx}",
                    "dynamic",
                    dyn["canonical_xyz"].shape[0],
                    anchor_out.diagnostics.get("fallback_reason", "empty_anchor"),
                )
                skipped["instance_id"] = int(dyn["instance_id"])
                skipped["anchor_diagnostics"] = anchor_out.diagnostics
                context_summaries.append(skipped)
                continue
            dyn_diag = _context_diagnostics(
                f"dynamic_{dyn_idx}",
                "dynamic",
                dyn_prims,
                anchor_out.diagnostics.get("query_voxels", dyn["canonical_xyz"].shape[0]),
            )
            dyn_diag["instance_id"] = int(dyn["instance_id"])
            dyn_diag["anchor_diagnostics"] = anchor_out.diagnostics
            context_summaries.append(dyn_diag)
            anchor_summary["dynamic_primitives"] += int(dyn_prims["means3D"].shape[0])
            anchor_summary["dynamic_init_count"] += int(anchor_out.use_geom_init.sum().item())
            anchor_summary["dynamic_fallback_count"] += int((~anchor_out.use_geom_init).sum().item())
            if dyn.get("box_0") is not None:
                frame0_contexts.append({
                    "name": f"dynamic_{dyn_idx}",
                    "primitives": transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device))),
                    "context": dyn_diag,
                })
            if dyn.get("box_1") is not None:
                frame1_contexts.append({
                    "name": f"dynamic_{dyn_idx}",
                    "primitives": transform_primitives(
                        dyn_prims,
                        _box1_to_frame0_pose(dyn["box_1"].to(device), rel_input_1_pose),
                    ),
                    "context": dyn_diag,
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
        static_prims = model.forward_anchor_context(
            static_anchor_out,
            context_type="static",
            return_diagnostics=True,
        )
        if static_prims is not None:
            static_diag = _context_diagnostics(
                "static",
                "static",
                static_prims,
                static_anchor_out.diagnostics.get("query_voxels", static_xyz.shape[0]),
            )
            frame0_contexts.append({"name": "static", "primitives": static_prims, "context": static_diag})
            frame1_contexts.append({"name": "static", "primitives": static_prims, "context": static_diag})
            context_summaries.append(static_diag)
            anchor_summary["static_primitives"] = int(static_prims["means3D"].shape[0])
            anchor_summary["static_init_count"] = int(static_anchor_out.use_geom_init.sum().item())
            anchor_summary["static_fallback_count"] = int((~static_anchor_out.use_geom_init).sum().item())
        else:
            context_summaries.append(
                _skipped_context_diagnostics(
                    "static",
                    "static",
                    static_xyz.shape[0],
                    static_anchor_out.diagnostics.get("fallback_reason", "empty_anchor"),
                )
            )
    else:
        raise ValueError(f"unsupported primitive_mode {cfg.primitive_mode!r}")

    if cfg.primitive_mode == "per_point":
        for ctx in context_summaries:
            if ctx.get("skipped"):
                continue
            if ctx.get("context_type") == "static":
                anchor_summary["static_primitives"] += int(ctx.get("n_generated", 0))
                anchor_summary["static_init_count"] += int(ctx.get("n_points_with_init", 0))
                anchor_summary["static_fallback_count"] += int(ctx.get("n_points_without_init", 0))
            elif ctx.get("context_type") == "dynamic":
                anchor_summary["dynamic_primitives"] += int(ctx.get("n_generated", 0))
                anchor_summary["dynamic_init_count"] += int(ctx.get("n_points_with_init", 0))
                anchor_summary["dynamic_fallback_count"] += int(ctx.get("n_points_without_init", 0))
        anchor_summary["query_voxels"] = anchor_summary["input_points"]

    anchor_summary["final_primitives"] = (
        anchor_summary["static_primitives"] + anchor_summary["dynamic_primitives"]
    )
    if anchor_summary["input_points"] > 0:
        anchor_summary["compression_ratio"] = (
            anchor_summary["final_primitives"] / anchor_summary["input_points"]
        )
    else:
        anchor_summary["compression_ratio"] = 0.0

    if not frame0_contexts or not frame1_contexts:
        raise RuntimeError("no primitives were produced for evaluation")

    frame0 = concat_primitives([ctx["primitives"] for ctx in frame0_contexts])
    frame1 = concat_primitives([ctx["primitives"] for ctx in frame1_contexts])

    ray_grid = make_lidar_ray_grid(cfg, device=device)

    target0 = points_to_lidar_maps(xyz0, i0, cfg)
    target1 = points_to_lidar_maps(xyz1, i1, cfg)

    _reset_cuda_peak(device_t)
    rendered0 = render_primitives(
        frame0,
        cfg,
        viewmatrix=torch.eye(4, device=device, dtype=xyz0.dtype),
        campos=torch.zeros(3, device=device, dtype=xyz0.dtype),
    )
    _record_memory("render_frame0", device_t, memory_records)
    _reset_cuda_peak(device_t)
    rendered1 = render_primitives(
        frame1,
        cfg,
        viewmatrix=inv_pose,
        campos=rel_input_1_pose[:3, 3],
    )
    _record_memory("render_frame1", device_t, memory_records)

    loss0 = loss_fn(rendered0, target0, rendered0.raydrop)
    loss1 = loss_fn(rendered1, target1, rendered1.raydrop)
    total_loss = loss0["total"] + loss1["total"]

    frame0_data = {
        "rendered": rendered0,
        "target": target0,
        "drop": rendered0.raydrop,
        "loss": loss0,
    }
    frame1_data = {
        "rendered": rendered1,
        "target": target1,
        "drop": rendered1.raydrop,
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
            "primitive_mode": cfg.primitive_mode,
            "filtered_points": {"frame0": int(xyz0.shape[0]), "frame1": int(xyz1.shape[0])},
            "static_points": int(scene["static_xyz"].shape[0]),
            "dynamic_instances": len(scene["dynamic"]),
            "primitive_summary": anchor_summary,
            "realized_contexts": sum(0 if ctx["skipped"] else 1 for ctx in context_summaries),
            "skipped_contexts": sum(1 if ctx["skipped"] else 0 for ctx in context_summaries),
            "context_groups": _summarize_context_groups(context_summaries),
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
