"""Training loop for QGS-Flow shared-scene pair rendering.

Each training sample contains two LiDAR keyframes. The pipeline builds one
shared scene from the pair, predicts static and tracked-dynamic primitives,
renders both keyframes, and optimizes against the two keyframe LiDAR images.

Intermediate sweep supervision / arbitrary-time rendering is not wired here.
"""

import argparse
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

from config import OFFICIAL_FULL_PTV3_BACKBONE_PARAMS, QGSConfig
sys.path.insert(0, os.path.join(os.path.expanduser(QGSConfig.data_root), "loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn

from models.geometry import decompose_scene
from models.head import DropHead, make_lidar_ray_grid
from nn.model import QGSModel
from nn.qgs_loss import QGSLoss
from nn.render_utils import (
    build_target_lidar_image,
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


def _filter_frame_points(xyz: torch.Tensor, intensity: torch.Tensor, cfg: QGSConfig):
    r = xyz.norm(dim=1)
    keep = (r > cfg.ego_radius) & (r < cfg.r_far)
    return xyz[keep], intensity[keep]


def _build_scene_primitives(
    model: QGSModel,
    cfg: QGSConfig,
    xyz: torch.Tensor,
    intensity_norm: torch.Tensor,
    *,
    context_type: str,
    time_scalar: torch.Tensor | None,
    ego_motion: torch.Tensor,
    is_dynamic_flag: torch.Tensor,
) -> tuple[dict | None, torch.Tensor | None]:
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
    )
    return primitives, xyz


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


def process_pair(model, loss_fn, drop_head, ray_dir, batch, idx, device, cfg):
    """Build a shared scene from a frame pair and render both keyframes."""
    p0 = _batch_item(batch, "input_0", idx).to(device)
    p1 = _batch_item(batch, "input_1", idx).to(device)
    rel_input_1_pose = _ensure_2d_pose(_batch_item(batch, "input_1_pose", idx)).to(device)

    xyz0 = p0[:, :3].float()
    i0_raw = p0[:, 3].float()
    i0 = (i0_raw / 255.0).clamp(0.0, 1.0)
    xyz1 = p1[:, :3].float()
    i1_raw = p1[:, 3].float()
    i1 = (i1_raw / 255.0).clamp(0.0, 1.0)

    boxes_0 = _batch_item(batch, "boxes_0", idx).to(device) if "boxes_0" in batch else torch.empty(0, 7, device=device)
    boxes_1 = _batch_item(batch, "boxes_1", idx).to(device) if "boxes_1" in batch else torch.empty(0, 7, device=device)
    instance_ids_0 = _batch_item(batch, "instance_ids_0", idx).to(device) if "instance_ids_0" in batch else torch.empty(0, dtype=torch.long, device=device)
    instance_ids_1 = _batch_item(batch, "instance_ids_1", idx).to(device) if "instance_ids_1" in batch else torch.empty(0, dtype=torch.long, device=device)

    xyz0, i0 = _filter_frame_points(xyz0, i0, cfg)
    xyz1, i1 = _filter_frame_points(xyz1, i1, cfg)
    if xyz0.shape[0] < cfg.knn_k_min or xyz1.shape[0] < cfg.knn_k_min:
        return None

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

    e_dir = rel_input_1_pose[:3, 3].to(dtype=xyz0.dtype, device=device)
    inv_pose = torch.linalg.inv(rel_input_1_pose)

    static_xyz = scene["static_xyz"].to(device)
    static_i = scene["static_intensity"].to(device)
    static_t = scene["static_time"].to(device)
    static_flag = torch.zeros(static_xyz.shape[0], device=device, dtype=static_xyz.dtype)
    static_prims, _ = _build_scene_primitives(
        model, cfg, static_xyz, static_i,
        context_type="static",
        time_scalar=static_t, ego_motion=e_dir,
        is_dynamic_flag=static_flag,
    )

    frame0_primitives = []
    frame1_primitives = []

    if static_prims is not None:
        frame0_primitives.append(static_prims)
        frame1_primitives.append(static_prims)

    dyn_list = scene["dynamic"]
    if dyn_list:
        dyn_xyz_list = [d["canonical_xyz"].to(device) for d in dyn_list]
        dyn_int_list = [d["canonical_intensity"].to(device) for d in dyn_list]
        dyn_time_list = [d["canonical_time"].to(device) for d in dyn_list]
        dyn_prims_list = model.forward_contexts_batched(
            dyn_xyz_list, dyn_int_list,
            context_type="dynamic",
            time_list=dyn_time_list,
            ego_motion=e_dir,
        )
        for dyn, dyn_prims in zip(dyn_list, dyn_prims_list):
            if dyn_prims is None:
                continue
            if dyn.get("box_0") is not None:
                frame0_primitives.append(_transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device))))
            if dyn.get("box_1") is not None:
                frame1_primitives.append(_transform_primitives(dyn_prims, _box_to_pose(dyn["box_1"].to(device))))

    frame0 = _concat_primitives(frame0_primitives)
    frame1 = _concat_primitives(frame1_primitives)
    if not frame0 or not frame1:
        return None

    el_min_rad = math.radians(cfg.lidar_el_min_deg)
    el_max_rad = math.radians(cfg.lidar_el_max_deg)

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

    rendered0 = render_primitives(
        frame0,
        height=cfg.lidar_height, width=cfg.lidar_width,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        sigma=cfg.lidar_sigma,
        r_near=cfg.r_near, r_far=cfg.r_far,
        viewmatrix=torch.eye(4, device=device, dtype=static_xyz.dtype),
        campos=torch.zeros(3, device=device, dtype=static_xyz.dtype),
    )
    rendered1 = render_primitives(
        frame1,
        height=cfg.lidar_height, width=cfg.lidar_width,
        el_min_rad=el_min_rad, el_max_rad=el_max_rad,
        sigma=cfg.lidar_sigma,
        r_near=cfg.r_near, r_far=cfg.r_far,
        viewmatrix=inv_pose,
        campos=rel_input_1_pose[:3, 3],
    )

    drop0_logit, drop0 = drop_head(
        rendered0.latent.unsqueeze(0),
        rendered0.range.unsqueeze(0),
        rendered0.normal.unsqueeze(0),
        rendered0.curvature.unsqueeze(0),
        rendered0.alpha_accum.unsqueeze(0),
        ray_dir,
    )
    drop1_logit, drop1 = drop_head(
        rendered1.latent.unsqueeze(0),
        rendered1.range.unsqueeze(0),
        rendered1.normal.unsqueeze(0),
        rendered1.curvature.unsqueeze(0),
        rendered1.alpha_accum.unsqueeze(0),
        ray_dir,
    )

    loss0 = loss_fn(rendered0, target0, drop0)
    loss1 = loss_fn(rendered1, target1, drop1)
    total = loss0["total"] + loss1["total"]
    loss_dict = {
        "total": total,
        "depth": 0.5 * (loss0["depth"] + loss1["depth"]),
        "intensity": 0.5 * (loss0["intensity"] + loss1["intensity"]),
        "raydrop": 0.5 * (loss0["raydrop"] + loss1["raydrop"]),
        "n_valid": 0.5 * (loss0["n_valid"] + loss1["n_valid"]),
        "valid_ratio": 0.5 * (loss0["valid_ratio"] + loss1["valid_ratio"]),
    }
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
        alpha_eps=cfg.loss_alpha_eps,
    ).to(device)
    drop_head = DropHead(latent_dim=cfg.lidar_latent_dim).to(device)
    ray_grid = make_lidar_ray_grid(
        cfg.lidar_height,
        cfg.lidar_width,
        math.radians(cfg.lidar_el_min_deg),
        math.radians(cfg.lidar_el_max_deg),
        device=device,
    )
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
        f"wrapper={backbone_total_params:,}, base_8ch={OFFICIAL_FULL_PTV3_BACKBONE_PARAMS:,})"
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
    drop_head_params = list(drop_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_b, "name": "backbone"},
            {"params": head_params + drop_head_params, "lr": lr_h, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
    )

    total_iters = max(1, cfg.num_epochs * max(1, len(dataloader)))
    warmup_iters = min(cfg.warmup_iters, max(1, total_iters // 4))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_iters,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_iters - warmup_iters), eta_min=0.0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters],
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
          f"el∈[{cfg.lidar_el_min_deg}, {cfg.lidar_el_max_deg}]°")
    print("-" * 60)

    best_loss = float("inf")
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "drop_head_state_dict" in ckpt:
            drop_head.load_state_dict(ckpt["drop_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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

            for b in range(B):
                out = process_pair(model, loss_fn, drop_head, ray_grid, batch, b, device, cfg)
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
                for p in list(model.parameters()) + list(drop_head.parameters()):
                    if p.grad is not None:
                        p.grad /= valid_pairs

            all_params = list(model.parameters()) + list(drop_head.parameters())
            has_bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in all_params
            )
            if has_bad_grad:
                pbar.write(f"  Bad grad at batch {batch_idx}")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip)

            optimizer.step()
            scheduler.step()      # iter-based: warmup → cosine over total_iters

            avg_batch = {
                k: sum(d[k] for d in batch_loss_dicts) / len(batch_loss_dicts)
                for k in batch_loss_dicts[0]
            }
            epoch_losses.append(avg_batch)
            cur_lr_b = optimizer.param_groups[0]["lr"]
            cur_lr_h = optimizer.param_groups[1]["lr"]
            pbar.set_postfix(
                loss=f"{avg_batch['total']:.4f}",
                depth=f"{avg_batch['depth']:.3f}",
                raydrop=f"{avg_batch['raydrop']:.3f}",
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
            f"valid={avg.get('valid_ratio', 0.0):.3f} "
            f"{dt:.1f}s"
        )
        tqdm.write(log)

        model_has_nan = any(
            torch.isnan(p).any() or torch.isinf(p).any()
            for p in model.parameters()
        )
        if avg["total"] < best_loss and not model_has_nan:
            best_loss = avg["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "drop_head_state_dict": drop_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_loss,
            }, os.path.join(ckpt_dir, "best_model.pt"))

        # scheduler.step() runs per-iter inside the batch loop now.

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "drop_head_state_dict": drop_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg["total"],
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"))

    print(f"\nDone. Best loss: {best_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _defaults = QGSConfig()
    parser = argparse.ArgumentParser(description="Train QGS-Flow Phase A")
    parser.add_argument("--data-root", default=_defaults.data_root)
    parser.add_argument("--epochs", type=int, default=_defaults.num_epochs)
    parser.add_argument("--lr", type=float, default=_defaults.lr)
    parser.add_argument("--batch-size", type=int, default=_defaults.batch_size)
    parser.add_argument("--overfit", type=int, default=0,
                        help="Overfit on N pairs (0 = full training)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", default=_defaults.device)
    args = parser.parse_args()

    cfg = QGSConfig(
        data_root=args.data_root,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    train(cfg, overfit_frames=args.overfit, resume=args.resume)


if __name__ == "__main__":
    main()
