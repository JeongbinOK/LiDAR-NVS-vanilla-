"""Training loop for QGS-Flow shared-scene pair rendering.

Each training sample contains two LiDAR keyframes. The pipeline builds one
shared scene from the pair, predicts static and tracked-dynamic primitives,
renders both keyframes, and optimizes against the two keyframe LiDAR images.

Intermediate sweep supervision / arbitrary-time rendering is not wired here.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")

# expandable_segments mitigates fragmentation from .topk() on non-contiguous
# tensors (see git history). Must be set before `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import OFFICIAL_FULL_PTV3_BACKBONE_PARAMS, QGSConfig, resolve_lidar_latent_dim
sys.path.insert(0, os.path.join(os.path.expanduser(QGSConfig.data_root), "loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn

from models.geometry import decompose_scene
from models.geometry.voxel_anchor import DynamicVoxelAnchorBuilder, VoxelAnchorBuilder
from nn.model import QGSModel
from nn.qgs_loss import QGSLoss
from nn.lidar_geometry import (
    make_lidar_ray_grid,
    points_to_lidar_maps,
)
from nn.render_utils import (
    build_gt_normal_map,
    quat_to_rotmat,
    render_primitives,
    rotmat_to_quat,
)


# ---------------------------------------------------------------------------
# Output dir
# ---------------------------------------------------------------------------

def _make_run_dir(base: str = "outputs") -> str:
    """Create outputs/train_NNN/ with configs/ and ckpt/ subdirs."""
    os.makedirs(base, exist_ok=True)
    existing = [
        d for d in os.listdir(base)
        if d.startswith("train_") and os.path.isdir(os.path.join(base, d))
    ]
    indices = []
    for d in existing:
        try:
            indices.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_idx = max(indices, default=0) + 1
    run_dir = os.path.join(base, f"train_{next_idx:03d}")
    os.makedirs(os.path.join(run_dir, "configs"))
    os.makedirs(os.path.join(run_dir, "ckpt"))
    return run_dir


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _batch_item(batch, key, idx):
    value = batch[key]
    if isinstance(value, (list, tuple)):
        return value[idx]
    if torch.is_tensor(value):
        return value[idx]
    return value[idx]


def _ensure_2d_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.dim() == 3 and pose.shape[0] == 1:
        return pose[0]
    return pose


def _warmup_cosine_lr_scale(
    step: int,
    warmup_iters: int,
    total_iters: int,
    start_factor: float = 1e-3,
) -> float:
    warmup_iters = max(1, int(warmup_iters))
    total_iters = max(warmup_iters, int(total_iters))
    step = max(0, int(step))

    if step < warmup_iters:
        if warmup_iters == 1:
            return 1.0
        progress = step / (warmup_iters - 1)
        return start_factor + (1.0 - start_factor) * progress

    cosine_iters = max(1, total_iters - warmup_iters)
    if cosine_iters == 1:
        return 0.0

    cosine_step = min(step - warmup_iters, cosine_iters - 1)
    progress = cosine_step / (cosine_iters - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    start_factor: float = 1e-3,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_lr_scale(
            step, warmup_iters=warmup_iters, total_iters=total_iters,
            start_factor=start_factor,
        ),
    )


def _restore_scheduler_state(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    optimizer: torch.optim.Optimizer,
    scheduler_state_dict: dict,
    warmup_iters: int,
    total_iters: int,
) -> None:
    try:
        scheduler.load_state_dict(scheduler_state_dict)
        return
    except (KeyError, ValueError, RuntimeError):
        pass

    if not isinstance(scheduler_state_dict, dict) or "last_epoch" not in scheduler_state_dict:
        return

    last_epoch = int(scheduler_state_dict["last_epoch"])
    last_lr = scheduler_state_dict.get("_last_lr")
    if last_lr is None:
        scale = _warmup_cosine_lr_scale(last_epoch, warmup_iters, total_iters)
        last_lr = [base_lr * scale for base_lr in scheduler.base_lrs]

    scheduler.last_epoch = last_epoch
    scheduler._last_lr = list(last_lr)
    for group, lr in zip(optimizer.param_groups, last_lr):
        group["lr"] = lr


def _filter_frame_points(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    cfg: QGSConfig,
    ring: torch.Tensor | None = None,
):
    r = xyz.norm(dim=1)
    keep = (r > cfg.ego_radius) & (r < cfg.r_far)
    if ring is None:
        return xyz[keep], intensity[keep]
    return xyz[keep], intensity[keep], ring[keep]


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


def _transform_primitives(primitives: dict, T: torch.Tensor) -> dict:
    R = quat_to_rotmat(primitives["rotations"])
    R_out = T[:3, :3] @ R
    t = T[:3, 3]
    means = (T[:3, :3] @ primitives["means3D"].T).T + t
    out = dict(primitives)
    out["means3D"] = means
    out["rotations"] = rotmat_to_quat(R_out)
    return out


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


def _concat_primitives(primitives: list[dict]) -> dict:
    if not primitives:
        return {}
    keys = primitives[0].keys()
    out = {}
    for key in keys:
        values = [p[key] for p in primitives]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = values
    return out


@torch.no_grad()
def _primitive_residual_stats(primitives: list[dict]) -> dict[str, torch.Tensor]:
    """Aggregate QGSHead residual magnitudes for train-time diagnostics."""
    rows = []
    weights = []
    for prim in primitives:
        aux = prim.get("aux", {})
        if not aux:
            continue
        n = int(prim["means3D"].shape[0])
        if n == 0:
            continue
        omega_local_deg = aux["omega_local"][0] * (180.0 / math.pi)  # [..., 3]
        row = {
            "diag_omega_deg": omega_local_deg.norm(dim=-1).mean(),
            "diag_tilt_deg": omega_local_deg[..., :2].norm(dim=-1).mean(),   # M2: tilt (ω_x,ω_y)
            "diag_spin_deg": omega_local_deg[..., 2].abs().mean(),           # M2: spin (ω_z, gauge)
            "diag_delta_c": aux["delta_c"][0].norm(dim=-1).mean(),
            "diag_delta_mu": aux["delta_mu"][0].abs().mean(),
            "diag_delta_mu_sign": aux["delta_mu"][0].mean(),                 # M1: signed (A vs B)
            "diag_delta_gap": aux["delta_gap"][0].abs().mean(),
            "diag_delta_gap_sign": aux["delta_gap"][0].mean(),               # M1: signed
            "diag_delta_s3": aux["delta_log_abs_s3"][0].abs().mean(),
            "diag_delta_s3_sign": aux["delta_log_abs_s3"][0].mean(),         # M1: signed
            "diag_delta_int": aux["delta_logit_intensity"][0].abs().mean(),
        }
        rows.append(row)
        weights.append(n)

    if not rows:
        return {}

    total = float(sum(weights))
    stats = {}
    for key in rows[0]:
        vals = torch.stack([
            row[key] * (weight / total)
            for row, weight in zip(rows, weights)
        ])
        stats[key] = vals.sum()
    return stats


def _profile_sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_mark(profile: dict | None, name: str, start: float, device: torch.device | str) -> float:
    if profile is None:
        return time.perf_counter()
    _profile_sync(device)
    now = time.perf_counter()
    timings = profile.setdefault("timings_ms", {})
    timings[name] = timings.get(name, 0.0) + (now - start) * 1000.0
    return now


def _profile_count(profile: dict | None, name: str, value: int | float) -> None:
    if profile is None:
        return
    profile.setdefault("counts", {})[name] = int(value) if isinstance(value, int) else float(value)


def _summarize_bad_gradients(
    named_params: list[tuple[str, torch.nn.Parameter]],
    *,
    max_items: int = 64,
) -> str:
    """Return a compact summary of the first non-finite gradients."""
    parts = []
    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all()):
            continue
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        finite_abs_max = float(grad[finite_mask].abs().max().item()) if bool(finite_mask.any()) else float("nan")
        parts.append(
            f"{name}(nan={nan_count},inf={inf_count},finite_abs_max={finite_abs_max:.2e})"
        )
        if len(parts) >= max_items:
            break
    return "; ".join(parts) if parts else "unknown"


def _check_finite_prims(prims_list: list, label: str) -> None:
    """Print non-finite entries for all tensor fields across a list of primitive dicts."""
    for i, prims in enumerate(prims_list):
        if prims is None:
            continue
        for k, v in prims.items():
            if not torch.is_tensor(v):
                continue
            fm = torch.isfinite(v)
            if fm.all():
                continue
            nan_c = int(torch.isnan(v).sum())
            inf_c = int(torch.isinf(v).sum())
            fmax = float(v[fm].abs().max()) if fm.any() else float("nan")
            print(f"  [non-finite] {label}[{i}].{k}: shape={tuple(v.shape)} "
                  f"nan={nan_c} inf={inf_c} finite_abs_max={fmax:.2e}")


def _check_finite_render(rendered, label: str) -> None:
    """Print non-finite entries for key fields of a rendered output."""
    for attr in ("range", "intensity", "alpha_accum", "normal", "curvature"):
        v = getattr(rendered, attr, None)
        if v is None or not torch.is_tensor(v):
            continue
        fm = torch.isfinite(v)
        if fm.all():
            continue
        nan_c = int(torch.isnan(v).sum())
        inf_c = int(torch.isinf(v).sum())
        fmax = float(v[fm].abs().max()) if fm.any() else float("nan")
        print(f"  [non-finite] {label}.{attr}: shape={tuple(v.shape)} "
              f"nan={nan_c} inf={inf_c} finite_abs_max={fmax:.2e}")


def _make_bad_grad_trace(cfg) -> dict | None:
    if not getattr(cfg, "debug_bad_grad_trace", False):
        return None
    return {
        "records": [],
        "total_nonfinite_tensors": 0,
        "max_records": int(getattr(cfg, "debug_bad_grad_trace_max", 16)),
    }


def _grad_trace_record(trace: dict | None, label: str, tensor: torch.Tensor) -> None:
    if trace is None or not torch.is_tensor(tensor) or not tensor.requires_grad:
        return

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        grad_det = grad.detach()
        finite_mask = torch.isfinite(grad_det)
        if bool(finite_mask.all()):
            return grad

        trace["total_nonfinite_tensors"] = trace.get("total_nonfinite_tensors", 0) + 1
        records = trace.setdefault("records", [])
        max_records = int(trace.get("max_records", 16))
        if len(records) < max_records:
            nan_count = int(torch.isnan(grad_det).sum().item())
            inf_count = int(torch.isinf(grad_det).sum().item())
            finite_abs_max = (
                float(grad_det[finite_mask].abs().max().item())
                if bool(finite_mask.any())
                else float("nan")
            )
            bad_indices = torch.nonzero(~finite_mask, as_tuple=False)
            first_bad = tuple(int(x) for x in bad_indices[0].detach().cpu().tolist()) if bad_indices.numel() else ()
            records.append(
                f"{label}: shape={tuple(grad_det.shape)} first_bad={first_bad} "
                f"nan={nan_count} inf={inf_count} finite_abs_max={finite_abs_max:.2e}"
            )
        return grad

    tensor.register_hook(_hook)


def _attach_primitive_grad_trace(trace: dict | None, prims: dict, label: str) -> None:
    for key in ("means3D", "scales", "rotations", "opacities", "intensity", "latent", "raydrop"):
        _grad_trace_record(trace, f"{label}.{key}", prims.get(key))


def _attach_render_grad_trace(trace: dict | None, rendered, label: str) -> None:
    for attr in ("range", "middepth", "intensity", "raydrop", "alpha_accum", "normal", "curvature", "raw"):
        _grad_trace_record(trace, f"{label}.{attr}", getattr(rendered, attr, None))


def _summarize_bad_grad_trace(trace: dict | None) -> str:
    if trace is None:
        return ""
    records = trace.get("records", [])
    if not records:
        return "no non-finite primitive/render tensor grad hook fired"
    total = int(trace.get("total_nonfinite_tensors", len(records)))
    suffix = f"; ... +{total - len(records)} more" if total > len(records) else ""
    return "; ".join(records) + suffix


def process_pair(
    model,
    loss_fn,
    ray_dir,
    batch,
    idx,
    device,
    cfg,
    profile: dict | None = None,
    bad_grad_trace: dict | None = None,
):
    """Build a shared scene from a frame pair and render both keyframes."""
    _profile_sync(device)
    t_prof = time.perf_counter()
    p0 = _batch_item(batch, "input_0", idx).to(device)
    p1 = _batch_item(batch, "input_1", idx).to(device)
    rel_input_1_pose = _ensure_2d_pose(_batch_item(batch, "input_1_pose", idx)).to(device)

    xyz0 = p0[:, :3].float()
    i0_raw = p0[:, 3].float()
    i0 = (i0_raw / 255.0).clamp(0.0, 1.0)
    ring0 = p0[:, 4].to(torch.int64) if p0.shape[1] >= 5 else None
    xyz1 = p1[:, :3].float()
    i1_raw = p1[:, 3].float()
    i1 = (i1_raw / 255.0).clamp(0.0, 1.0)
    ring1 = p1[:, 4].to(torch.int64) if p1.shape[1] >= 5 else None

    boxes_0 = _batch_item(batch, "boxes_0", idx).to(device) if "boxes_0" in batch else torch.empty(0, 7, device=device)
    boxes_1 = _batch_item(batch, "boxes_1", idx).to(device) if "boxes_1" in batch else torch.empty(0, 7, device=device)
    instance_ids_0 = _batch_item(batch, "instance_ids_0", idx).to(device) if "instance_ids_0" in batch else torch.empty(0, dtype=torch.long, device=device)
    instance_ids_1 = _batch_item(batch, "instance_ids_1", idx).to(device) if "instance_ids_1" in batch else torch.empty(0, dtype=torch.long, device=device)

    if ring0 is not None:
        xyz0, i0, ring0 = _filter_frame_points(xyz0, i0, cfg, ring=ring0)
        xyz1, i1, ring1 = _filter_frame_points(xyz1, i1, cfg, ring=ring1)
    else:
        xyz0, i0 = _filter_frame_points(xyz0, i0, cfg)
        xyz1, i1 = _filter_frame_points(xyz1, i1, cfg)
    _profile_count(profile, "filtered_frame0_points", xyz0.shape[0])
    _profile_count(profile, "filtered_frame1_points", xyz1.shape[0])
    if xyz0.shape[0] < cfg.knn_k_min or xyz1.shape[0] < cfg.knn_k_min:
        return None
    t_prof = _profile_mark(profile, "load_filter", t_prof, device)

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
    _profile_count(profile, "static_points", scene["static_xyz"].shape[0])
    _profile_count(profile, "dynamic_instances", len(scene["dynamic"]))
    t_prof = _profile_mark(profile, "decompose_scene", t_prof, device)

    inv_pose = torch.linalg.inv(rel_input_1_pose)

    frame0_primitives = []
    frame1_primitives = []
    diagnostic_primitives = []

    dyn_list = scene["dynamic"]
    if cfg.primitive_mode == "voxel_anchor":
        dyn_anchor_outputs = []
        if dyn_list:
            dynamic_builder = _make_dynamic_anchor_builder(cfg)
            dyn_anchor_pairs = dynamic_builder(
                dyn_list,
                profile=profile,
                profile_prefix="dynamic_anchor",
            )
            dyn_anchor_outputs = [out for _, out in dyn_anchor_pairs]
            _profile_count(profile, "dynamic_query_voxels", sum(out.diagnostics.get("query_voxels", 0) for out in dyn_anchor_outputs))
            _profile_count(profile, "dynamic_final_anchors", sum(out.c_init.shape[0] for out in dyn_anchor_outputs))
            t_prof = _profile_mark(profile, "dynamic_anchor_build", t_prof, device)
            dyn_prims_list = model.forward_anchor_contexts_batched(
                dyn_anchor_outputs,
                context_type="dynamic",
            )
            t_prof = _profile_mark(profile, "dynamic_ptv3_head", t_prof, device)
        else:
            dyn_prims_list = []

        static_xyz_parts = [scene["static_xyz"].to(device)]
        static_i_parts = [scene["static_intensity"].to(device)]
        static_t_parts = [scene["static_time"].to(device)]
        for dyn, dyn_anchor_out, dyn_prims in zip(dyn_list, dyn_anchor_outputs, dyn_prims_list):
            if dyn_prims is None or dyn_anchor_out.c_init.shape[0] == 0:
                static_xyz_parts.append(dyn["fallback_xyz"].to(device))
                static_i_parts.append(dyn["fallback_intensity"].to(device))
                static_t_parts.append(dyn["fallback_time"].to(device))
                continue
            diagnostic_primitives.append(dyn_prims)
            if dyn.get("box_0") is not None:
                frame0_primitives.append(_transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device))))
            if dyn.get("box_1") is not None:
                frame1_primitives.append(_transform_primitives(
                    dyn_prims,
                    _box1_to_frame0_pose(dyn["box_1"].to(device), rel_input_1_pose),
                ))

        static_xyz = torch.cat(static_xyz_parts, dim=0)
        static_i = torch.cat(static_i_parts, dim=0)
        static_t = torch.cat(static_t_parts, dim=0)
        src = static_t
        static_anchor_out = _make_static_anchor_builder(cfg)(
            static_xyz,
            static_i,
            src,
            pose_frame1_in_frame0=rel_input_1_pose,
            profile=profile,
            profile_prefix="static_anchor",
        )
        _profile_count(profile, "static_query_voxels", static_anchor_out.diagnostics.get("query_voxels", 0))
        _profile_count(profile, "static_final_anchors", static_anchor_out.c_init.shape[0])
        t_prof = _profile_mark(profile, "static_anchor_build", t_prof, device)
        static_prims = model.forward_anchor_context(
            static_anchor_out,
            context_type="static",
        )
        t_prof = _profile_mark(profile, "static_ptv3_head", t_prof, device)
        if static_prims is not None:
            diagnostic_primitives.append(static_prims)
            frame0_primitives.append(static_prims)
            frame1_primitives.append(static_prims)
    else:
        raise ValueError(f"unsupported primitive_mode {cfg.primitive_mode!r}")

    if cfg.debug_finite_check and diagnostic_primitives:
        _check_finite_prims(diagnostic_primitives, "primitives")

    frame0 = _concat_primitives(frame0_primitives)
    frame1 = _concat_primitives(frame1_primitives)
    if not frame0 or not frame1:
        return None
    _profile_count(profile, "frame0_primitives", frame0["means3D"].shape[0])
    _profile_count(profile, "frame1_primitives", frame1["means3D"].shape[0])
    _attach_primitive_grad_trace(bad_grad_trace, frame0, f"pair{idx}.frame0.prims")
    _attach_primitive_grad_trace(bad_grad_trace, frame1, f"pair{idx}.frame1.prims")

    target0 = points_to_lidar_maps(xyz0, i0, cfg, ring=ring0)
    target0.update(build_gt_normal_map(
        target0["range_image"], target0["valid_mask"], ray_dir,
    ))
    target1 = points_to_lidar_maps(xyz1, i1, cfg, ring=ring1)
    target1.update(build_gt_normal_map(
        target1["range_image"], target1["valid_mask"], ray_dir,
    ))
    t_prof = _profile_mark(profile, "target_images", t_prof, device)

    rendered0 = render_primitives(
        frame0,
        cfg,
        viewmatrix=torch.eye(4, device=device, dtype=static_xyz.dtype),
        campos=torch.zeros(3, device=device, dtype=static_xyz.dtype),
    )
    t_prof = _profile_mark(profile, "render_frame0", t_prof, device)
    if cfg.debug_finite_check:
        _check_finite_render(rendered0, "rendered0")
    _attach_render_grad_trace(bad_grad_trace, rendered0, f"pair{idx}.rendered0")
    rendered1 = render_primitives(
        frame1,
        cfg,
        viewmatrix=inv_pose,
        campos=rel_input_1_pose[:3, 3],
    )
    t_prof = _profile_mark(profile, "render_frame1", t_prof, device)
    if cfg.debug_finite_check:
        _check_finite_render(rendered1, "rendered1")
    _attach_render_grad_trace(bad_grad_trace, rendered1, f"pair{idx}.rendered1")

    loss0 = loss_fn(rendered0, target0, rendered0.raydrop, ray_dir)
    loss1 = loss_fn(rendered1, target1, rendered1.raydrop, ray_dir)
    t_prof = _profile_mark(profile, "loss", t_prof, device)
    total = loss0["total"] + loss1["total"]
    loss_dict = {
        "total": total,
        "depth": 0.5 * (loss0["depth"] + loss1["depth"]),
        "depth_range": 0.5 * (loss0["depth_range"] + loss1["depth_range"]),
        "depth_median": 0.5 * (loss0["depth_median"] + loss1["depth_median"]),
        "intensity": 0.5 * (loss0["intensity"] + loss1["intensity"]),
        "raydrop": 0.5 * (loss0["raydrop"] + loss1["raydrop"]),
        "distortion": 0.5 * (loss0["distortion"] + loss1["distortion"]),
        "normal": 0.5 * (loss0["normal"] + loss1["normal"]),
        "n_valid": 0.5 * (loss0["n_valid"] + loss1["n_valid"]),
        "coverage": 0.5 * (loss0["coverage"] + loss1["coverage"]),
        "raydrop_hit": 0.5 * (loss0["raydrop_hit"] + loss1["raydrop_hit"]),
        "raydrop_miss": 0.5 * (loss0["raydrop_miss"] + loss1["raydrop_miss"]),
        "drop_prob_hit_mean": 0.5 * (loss0["drop_prob_hit_mean"] + loss1["drop_prob_hit_mean"]),
        "drop_prob_miss_mean": 0.5 * (loss0["drop_prob_miss_mean"] + loss1["drop_prob_miss_mean"]),
        "alpha_hit_mean": 0.5 * (loss0["alpha_hit_mean"] + loss1["alpha_hit_mean"]),
        "alpha_miss_mean": 0.5 * (loss0["alpha_miss_mean"] + loss1["alpha_miss_mean"]),
        "intensity_gt_mean": 0.5 * (loss0["intensity_gt_mean"] + loss1["intensity_gt_mean"]),
        "intensity_pred_mean": 0.5 * (loss0["intensity_pred_mean"] + loss1["intensity_pred_mean"]),
    }
    loss_dict.update(_primitive_residual_stats(diagnostic_primitives))
    return loss_dict


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: QGSConfig, overfit_frames: int = 0, resume: str = ""):
    device = cfg.device

    run_dir = _make_run_dir(os.path.join(os.path.dirname(__file__), "outputs"))
    ckpt_dir = os.path.join(run_dir, "ckpt")
    cfg_dir = os.path.join(run_dir, "configs")

    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    try:
        import subprocess
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_diff = subprocess.check_output(
            ["git", "diff", "--stat"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        branch_name = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info = {
            "commit_hash": commit_hash,
            "branch": branch_name,
            "is_dirty": bool(git_diff),
            "diff_stat": git_diff,
        }
        with open(os.path.join(cfg_dir, "git_info.json"), "w") as f:
            json.dump(git_info, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save git info: {e}")

    print(f"Run dir: {run_dir}")

    data_root = os.path.expanduser(cfg.data_root)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    bbox_json = cfg.bbox_json_path.format(split=cfg.train_split) if cfg.bbox_json_path else ""
    if bbox_json and not os.path.isabs(bbox_json):
        candidates = [os.path.join(data_root, bbox_json), os.path.join(repo_root, bbox_json)]
        bbox_json = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    if cfg.dataset_mode == "bbox" and bbox_json and not os.path.isfile(bbox_json):
        print(f"Warning: bbox_json_path={bbox_json} not found; falling back to GT annotations.")
        bbox_json = ""
    dataset = NuScenesNVSDataset(
        dataroot=data_root,
        version=cfg.nuscenes_version,
        split=cfg.train_split,
        frame_gap=cfg.frame_gap,
        mode=cfg.dataset_mode,
        bbox_json_path=bbox_json if (cfg.dataset_mode == "bbox" and bbox_json) else None,
    )
    if overfit_frames > 0:
        dataset.data_infos = dataset.data_infos[:overfit_frames]
        print(f"Overfit mode: {overfit_frames} pair(s)")

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=nvs_collate_fn, num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )

    model = QGSModel(cfg).to(device)
    loss_fn = QGSLoss(
        w_depth=cfg.loss_w_range,
        w_intensity=cfg.loss_w_intensity,
        w_raydrop=getattr(cfg, "loss_w_raydrop", 0.1),
        w_distortion=getattr(cfg, "loss_w_distortion", 0.05),
        w_normal=getattr(cfg, "loss_w_normal", 0.05),
        alpha_eps=cfg.loss_alpha_eps,
    ).to(device)
    ray_grid = make_lidar_ray_grid(cfg, device=device)
    if hasattr(model.backbone, "parameter_count"):
        backbone_core_params = model.backbone.parameter_count(include_projection=False)
        backbone_total_params = model.backbone.parameter_count(include_projection=True)
    else:
        backbone_core_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        backbone_total_params = backbone_core_params
    expected_backbone_params = cfg.expected_ptv3_backbone_params()
    branch_parts = []
    if cfg.ptv3_decoupled_stem:
        branch_parts.append("stem")
    if cfg.ptv3_pdnorm_bn:
        branch_parts.append("bn")
    if cfg.ptv3_pdnorm_ln:
        branch_parts.append("ln")
    branch_label = "+".join(branch_parts) if branch_parts else "none"
    print(
        "PTv3 official_full "
        f"(branches={branch_label}, backbone={backbone_core_params:,}, "
        f"wrapper={backbone_total_params:,}, official_base={OFFICIAL_FULL_PTV3_BACKBONE_PARAMS:,})"
    )
    if (
        expected_backbone_params is not None
        and backbone_core_params != expected_backbone_params
    ):
        raise RuntimeError(
            "PTv3 official_full expected "
            f"{expected_backbone_params:,} backbone params, got {backbone_core_params:,}"
        )

    # Per-module lr groups: PTv3 backbone needs lower lr (bf16+flash-attn);
    # head MLPs tolerate the higher lr. Falls back to cfg.lr if either is None.
    lr_b = cfg.lr_backbone if cfg.lr_backbone is not None else cfg.lr
    lr_h = cfg.lr_head     if cfg.lr_head     is not None else cfg.lr
    backbone_params = list(model.backbone.parameters())
    head_params     = [p for n, p in model.named_parameters()
                       if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_b, "name": "backbone"},
            {"params": head_params, "lr": lr_h, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
    )

    total_iters = max(1, cfg.num_epochs * max(1, len(dataloader)))
    warmup_iters = min(cfg.warmup_iters, max(1, total_iters // 4))
    scheduler = _build_warmup_cosine_scheduler(
        optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        start_factor=1e-3,
    )

    print("Model Parameters per Module:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<20}: {params:>12,}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,}")
    print(f"Dataset: {len(dataset)} pairs")
    print(f"Epochs: {cfg.num_epochs}, batch_size: {cfg.batch_size}")
    print(f"LiDAR image: {cfg.lidar_width}×{cfg.lidar_height} "
          f"el∈[{min(cfg.ring_to_elevation_deg):.2f}, {max(cfg.ring_to_elevation_deg):.2f}]°")
    print("-" * 60)

    best_loss = float("inf")
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            _restore_scheduler_state(
                scheduler,
                optimizer,
                ckpt["scheduler_state_dict"],
                warmup_iters=warmup_iters,
                total_iters=total_iters,
            )

        max_lrs = [lr_b, lr_h]
        for group, max_lr in zip(optimizer.param_groups, max_lrs):
            loaded_lr = group.get("lr", max_lr)
            if loaded_lr > max_lr * 1.01:
                print(
                    f"Warning: resume lr for {group.get('name', 'group')} "
                    f"was {loaded_lr:.3e}; clamping to config lr {max_lr:.3e}"
                )
                group["lr"] = max_lr
            group["initial_lr"] = max_lr
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from {resume} (epoch {ckpt['epoch']+1}, loss={best_loss:.4f})")
        print("-" * 60)

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_losses = []
        t0 = time.time()

        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            B = len(batch["input_0"])
            batch_loss = 0.0
            batch_loss_dicts = []

            optimizer.zero_grad()
            valid_pairs = 0
            bad_grad_trace = _make_bad_grad_trace(cfg)

            _anomaly_ctx = (
                torch.autograd.detect_anomaly()
                if cfg.debug_anomaly_batch >= 0 and batch_idx == cfg.debug_anomaly_batch
                else contextlib.nullcontext()
            )
            with _anomaly_ctx:
                for b in range(B):
                    out = process_pair(
                        model,
                        loss_fn,
                        ray_grid,
                        batch,
                        b,
                        device,
                        cfg,
                        bad_grad_trace=bad_grad_trace,
                    )
                    if out is None:
                        continue
                    if not torch.isfinite(out["total"]):
                        pbar.write(f"  NaN/inf loss at batch {batch_idx} pair {valid_pairs}")
                        continue
                    out["total"].backward()
                    valid_pairs += 1
                    batch_loss += out["total"].item()
                    batch_loss_dicts.append({k: float(v) for k, v in out.items()})

            if valid_pairs == 0:
                pbar.write(f"  All pairs invalid at batch {batch_idx}")
                optimizer.zero_grad()
                continue

            batch_loss /= valid_pairs
            if valid_pairs > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= valid_pairs

            all_params = list(model.parameters())
            has_bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in all_params
            )
            if has_bad_grad:
                named_params = list(model.named_parameters())
                bad_grad_summary = _summarize_bad_gradients(named_params)
                bad_grad_trace_summary = _summarize_bad_grad_trace(bad_grad_trace)
                avg_batch = {
                    k: sum(d[k] for d in batch_loss_dicts) / len(batch_loss_dicts)
                    for k in batch_loss_dicts[0]
                }
                pbar.write(
                    "  Bad grad at batch "
                    f"{batch_idx} "
                    f"loss={avg_batch['total']:.4f} depth={avg_batch['depth']:.3f} "
                    f"int={avg_batch['intensity']:.3f} drop={avg_batch['raydrop']:.3f} "
                    f"drop_hit={avg_batch.get('raydrop_hit', 0.0):.3f} "
                    f"drop_miss={avg_batch.get('raydrop_miss', 0.0):.3f} "
                    f"alpha_hit={avg_batch.get('alpha_hit_mean', 0.0):.3f} "
                    f"alpha_miss={avg_batch.get('alpha_miss_mean', 0.0):.3f} "
                    f"bad={bad_grad_summary}"
                )
                if bad_grad_trace_summary:
                    pbar.write(f"  Bad grad tensor trace: {bad_grad_trace_summary}")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=cfg.grad_clip)

            optimizer.step()
            scheduler.step()      # iter-based: warmup → cosine over total_iters

            avg_batch = {
                k: sum(d[k] for d in batch_loss_dicts) / len(batch_loss_dicts)
                for k in batch_loss_dicts[0]
            }
            epoch_losses.append(avg_batch)

            # Intermediate diagnostic log every 500 batches
            if (batch_idx + 1) % 500 == 0 and epoch_losses:
                mid_avg = {
                    k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                    for k in epoch_losses[0]
                }
                with open(os.path.join(run_dir, "train_metrics.jsonl"), "a") as _f:
                    _f.write(json.dumps({"epoch": epoch + 1, "batch": batch_idx + 1, **mid_avg}) + "\n")

            cur_lr_b = optimizer.param_groups[0]["lr"]
            cur_lr_h = optimizer.param_groups[1]["lr"]
            pbar.set_postfix(
                loss=f"{avg_batch['total']:.4f}",
                depth=f"{avg_batch['depth']:.3f}",
                int=f"{avg_batch['intensity']:.3f}",
                drop=f"{avg_batch['raydrop']:.3f}",
                dist=f"{avg_batch.get('distortion', 0.0):.4f}",
                nrm=f"{avg_batch.get('normal', 0.0):.4f}",
                dC=f"{avg_batch.get('diag_delta_c', 0.0):.3f}",
                dI=f"{avg_batch.get('diag_delta_int', 0.0):.3f}",
                lr=f"{cur_lr_b:.1e}/{cur_lr_h:.1e}",
            )

        dt = time.time() - t0
        if not epoch_losses:
            continue

        avg = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0]
        }
        log = (
            f"[{epoch+1:3d}/{cfg.num_epochs}] "
            f"loss={avg['total']:.4f} "
            f"depth={avg['depth']:.3f} "
            f"int={avg['intensity']:.3f} "
            f"raydrop={avg['raydrop']:.3f} "
            f"dist={avg.get('distortion', 0.0):.3f} "
            f"nrm={avg.get('normal', 0.0):.3f} "
            f"cov={avg.get('coverage', 0.0):.3f} "
            f"dC={avg.get('diag_delta_c', 0.0):.4f} "
            f"tilt={avg.get('diag_tilt_deg', 0.0):.2f}° spin={avg.get('diag_spin_deg', 0.0):.2f}° "
            f"dMu={avg.get('diag_delta_mu', 0.0):.4f}(sgn={avg.get('diag_delta_mu_sign', 0.0):+.3f}) "
            f"dGap={avg.get('diag_delta_gap', 0.0):.4f}(sgn={avg.get('diag_delta_gap_sign', 0.0):+.3f}) "
            f"dS3={avg.get('diag_delta_s3', 0.0):.4f}(sgn={avg.get('diag_delta_s3_sign', 0.0):+.3f}) "
            f"dI={avg.get('diag_delta_int', 0.0):.4f} "
            f"{dt:.1f}s"
        )
        tqdm.write(log)
        with open(os.path.join(run_dir, "train_metrics.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, **avg}) + "\n")

        model_has_nan = any(
            torch.isnan(p).any() or torch.isinf(p).any()
            for p in model.parameters()
        )
        if avg["total"] < best_loss and not model_has_nan:
            best_loss = avg["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_loss,
            }, os.path.join(ckpt_dir, "best_model.pt"))

        # scheduler.step() runs per-iter inside the batch loop now.

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg["total"],
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"))

    print(f"\nDone. Best loss: {best_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> QGSConfig:
    with open(path) as f:
        raw = json.load(f)
    allowed = {field.name for field in dataclasses.fields(QGSConfig)}
    return QGSConfig(**{k: v for k, v in raw.items() if k in allowed})


def _load_config_for_resume(checkpoint_path: str) -> QGSConfig | None:
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))),
        "configs",
        "config.json",
    )
    if not os.path.isfile(cfg_path):
        return None
    return _load_config(cfg_path)


def main():
    _defaults = QGSConfig()
    parser = argparse.ArgumentParser(description="Train QGS-Flow Phase A")
    parser.add_argument("--config", type=str, default="",
                        help="Path to a saved configs/config.json to use as the base config")
    parser.add_argument("--data-root", default=_defaults.data_root)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lidar-latent-dim", type=int, default=None)
    parser.add_argument("--overfit", type=int, default=0,
                        help="Overfit on N pairs (0 = full training)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--anchor-token-variant",
        choices=("full", "no_fit_quality", "no_intensity_stats", "with_normal"),
        default=None,
        help="Voxel-anchor token ablation variant",
    )
    parser.add_argument(
        "--anchor-filter-mode",
        choices=("residual", "residual_planarity"),
        default=None,
        help="Voxel-anchor hard-filter ablation mode",
    )
    parser.add_argument(
        "--anchor-residual-threshold",
        type=float,
        default=None,
        help="Voxel-anchor residual threshold tau_r",
    )
    parser.add_argument(
        "--anchor-planarity-threshold",
        type=float,
        default=None,
        help="Voxel-anchor planarity threshold for residual_planarity mode",
    )
    parser.add_argument(
        "--lidar-sigma",
        type=float,
        default=None,
        help="LiDAR rasterizer footprint sigma; larger values increase ray coverage",
    )
    # Residual saturation bounds (codex_geometry_residual_saturation_plan.md)
    parser.add_argument("--head-center-bound", type=float, default=None,
                        help="Max analytic-centre residual magnitude (m); default 0.3")
    parser.add_argument("--rot-tilt-deg", type=float, default=None,
                        help="Rotation tilt bound (deg); default 10")
    parser.add_argument("--rot-spin-deg", type=float, default=None,
                        help="Rotation spin bound (deg); default 30")
    parser.add_argument("--scale-log-mean-bound", type=float, default=None,
                        help="Scale log-mean residual bound; default ln(2)≈0.693")
    parser.add_argument("--scale-log-gap-bound", type=float, default=None,
                        help="Scale log-gap residual bound; default ln(1.5)≈0.405")
    parser.add_argument("--s3-log-bound", type=float, default=None,
                        help="s3 log residual bound; default ln(1.5)≈0.405")
    parser.add_argument(
        "--debug-finite-check",
        action="store_true",
        default=False,
        help="Log non-finite forward boundaries (primitives / rendered) in process_pair",
    )
    parser.add_argument(
        "--debug-anomaly-batch",
        type=int,
        default=None,
        help="Run torch.autograd.detect_anomaly on this batch_idx (-1 = off)",
    )
    parser.add_argument(
        "--no-debug-bad-grad-trace",
        action="store_true",
        default=False,
        help="Disable primitive/render backward hooks used to localize Bad grad batches",
    )
    parser.add_argument(
        "--debug-bad-grad-trace-max",
        type=int,
        default=None,
        help="Maximum primitive/render tensor-gradient records to print per Bad grad batch",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config else None
    if cfg is None and args.resume:
        cfg = _load_config_for_resume(args.resume)
    if cfg is None:
        cfg = QGSConfig()
    cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lidar_latent_dim is not None:
        cfg.lidar_latent_dim = resolve_lidar_latent_dim(args.lidar_latent_dim)
    if args.device is not None:
        cfg.device = args.device
    if args.anchor_token_variant is not None:
        cfg.anchor_token_variant = args.anchor_token_variant
        cfg.anchor_token_dim = 25 if args.anchor_token_variant == "with_normal" else 22
        cfg.ptv3_model_in_channels = cfg.resolved_input_channels()
    if args.anchor_filter_mode is not None:
        cfg.anchor_filter_mode = args.anchor_filter_mode
    if args.anchor_residual_threshold is not None:
        cfg.anchor_residual_threshold = args.anchor_residual_threshold
    if args.anchor_planarity_threshold is not None:
        cfg.anchor_planarity_threshold = args.anchor_planarity_threshold
    if args.lidar_sigma is not None:
        cfg.lidar_sigma = args.lidar_sigma
    if args.head_center_bound is not None:
        cfg.head_center_bound = args.head_center_bound
    if args.rot_tilt_deg is not None:
        cfg.rot_tilt_deg = args.rot_tilt_deg
    if args.rot_spin_deg is not None:
        cfg.rot_spin_deg = args.rot_spin_deg
    if args.scale_log_mean_bound is not None:
        cfg.scale_log_mean_bound = args.scale_log_mean_bound
    if args.scale_log_gap_bound is not None:
        cfg.scale_log_gap_bound = args.scale_log_gap_bound
    if args.s3_log_bound is not None:
        cfg.s3_log_bound = args.s3_log_bound
    if args.debug_finite_check:
        cfg.debug_finite_check = True
    if args.debug_anomaly_batch is not None:
        cfg.debug_anomaly_batch = args.debug_anomaly_batch
    if args.no_debug_bad_grad_trace:
        cfg.debug_bad_grad_trace = False
    if args.debug_bad_grad_trace_max is not None:
        cfg.debug_bad_grad_trace_max = args.debug_bad_grad_trace_max
    train(cfg, overfit_frames=args.overfit, resume=args.resume)


if __name__ == "__main__":
    main()
