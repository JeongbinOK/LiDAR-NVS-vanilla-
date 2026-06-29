#!/usr/bin/env python3
"""Per-input midpoint evaluation for Utonia/GS-LiDAR style LiDAR metrics.

The script bypasses Lightning's test loop so every dataset item/window is kept as
one result row. For a 1s input window it renders only the midpoint key frame
(normalised timestamp closest to 0.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent if (SCRIPT_PATH.parent / "config").exists() else SCRIPT_PATH.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(REPO_ROOT / ".cache/torch_extensions"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.dataloader import dataset_dict  # noqa: E402
from src.dataloader.nuscene import multiframe_collate_fn  # noqa: E402
from src.models_new.module import GausRender, GausTemp, Point2Gaus  # noqa: E402
from src.models_new.utils.chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist  # noqa: E402
from src.models_new.utils.chamfer.fscore import fscore  # noqa: E402
from src.models_new.utils.graphics_utils import pano_to_lidar  # noqa: E402


GS_LIDAR_DEPTH_MAX = 80.0
GS_LIDAR_INTENSITY_MAX = 1.0
GS_LIDAR_MIN_VALUE = 1e-6
GS_LIDAR_POINT_NEAR = 0.2
GS_LIDAR_POINT_FAR = 80.0
GS_LIDAR_FSCORE_THRESHOLD = 0.05
RAYDROP_KEEP_THRESHOLD = 0.5


METRIC_KEYS = [
    "point_cd",
    "point_fscore",
    "point_precision",
    "point_recall",
    "depth_rmse",
    "depth_median_abs_error",
    "depth_ssim",
    "depth_psnr",
    "intensity_rmse",
    "intensity_ssim",
    "intensity_psnr",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate every test dataset input window at its midpoint key frame."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config/nuscene_train.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument(
        "--seq-ranges",
        default=None,
        help="Comma-separated 0.1s sequence ranges, e.g. 450-500,1250-1300.",
    )
    parser.add_argument("--seq-step", type=int, default=10)
    parser.add_argument("--seq-context", type=int, default=5)
    parser.add_argument("--seq-origin", choices=["global", "scene"], default="global")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--no-save-visualizations", action="store_true")
    parser.add_argument("--vis-every", type=int, default=1)
    parser.add_argument("--vis-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_known_args()


def parse_seq_ranges(text: str | None) -> list[tuple[int, int]]:
    if not text:
        return []

    ranges = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            value = int(chunk)
            ranges.append((value, value))
            continue
        start_s, end_s = chunk.split("-", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError(f"Invalid seq range {chunk!r}: end must be >= start.")
        ranges.append((start, end))
    return ranges


def target_seq_map_from_ranges(
    ranges: list[tuple[int, int]],
    *,
    step: int,
    context: int,
) -> dict[int, tuple[int, int]]:
    if step <= 0:
        raise ValueError("--seq-step must be positive.")
    if context <= 0:
        raise ValueError("--seq-context must be positive.")

    target_to_range = {}
    for start, end in ranges:
        # For 450-500 with step=10/context=5, targets are
        # 460, 470, 480, 490, so inputs stay inside the range.
        for target_seq in range(start + step, end, step):
            if target_seq - context < start or target_seq + context > end:
                continue
            target_to_range.setdefault(target_seq, (start, end))
    return target_to_range


def scene_seq_starts(dataset) -> list[int]:
    starts = []
    total = 0
    for frames in dataset.scene_frames:
        starts.append(total)
        total += (len(frames) - 1) // dataset.sample_hop + 1
    return starts


def _scene_seq_counts(dataset) -> list[int]:
    return [(len(frames) - 1) // dataset.sample_hop + 1 for frames in dataset.scene_frames]


def _find_scene_for_global_seq(scene_starts: list[int], scene_counts: list[int], target_seq: int) -> tuple[int, int] | None:
    for scene_idx, (start, count) in enumerate(zip(scene_starts, scene_counts)):
        if start <= target_seq < start + count:
            return scene_idx, target_seq - start
    return None


def build_eval_jobs(dataset, args: argparse.Namespace) -> tuple[list[dict], dict]:
    seq_ranges = parse_seq_ranges(args.seq_ranges)
    if not seq_ranges:
        end_index = len(dataset) if args.max_items is None else min(len(dataset), args.start_index + args.max_items)
        jobs = [{"item_idx": item_idx, "seq_meta": None} for item_idx in range(args.start_index, end_index)]
        return jobs, {
            "mode": "all_items",
            "start_index": args.start_index,
            "end_index": end_index,
            "num_dataset_items": len(dataset),
        }

    target_to_range = target_seq_map_from_ranges(
        seq_ranges,
        step=args.seq_step,
        context=args.seq_context,
    )
    scene_starts = scene_seq_starts(dataset)
    scene_counts = _scene_seq_counts(dataset)
    mid_window_index = dataset.window_frame_count // 2
    mid_seq_offset = dataset.window_sample_hops[mid_window_index] // dataset.sample_hop
    input_seq_span = dataset.window_sample_hops[-1] // dataset.sample_hop

    if args.seq_context != mid_seq_offset:
        raise ValueError(
            f"--seq-context={args.seq_context} does not match the dataset midpoint "
            f"offset {mid_seq_offset}. Change window/sample_gap config or use {mid_seq_offset}."
        )
    if args.seq_step != input_seq_span:
        raise ValueError(
            f"--seq-step={args.seq_step} does not match the dataset input span "
            f"{input_seq_span}. Change window/sample_gap config or use {input_seq_span}."
        )

    existing_item_by_anchor = {
        (int(scene_idx), int(anchor_idx)): item_idx
        for item_idx, (scene_idx, anchor_idx) in enumerate(dataset.index)
    }
    original_num_items = len(dataset.index)
    jobs = []
    seen_targets = set()
    skipped_targets = {}

    def add_job(scene_idx: int, target_seq_local: int, compare_seq: int, seq_range: tuple[int, int]) -> None:
        scene_count = scene_counts[scene_idx]
        input0_seq_local = target_seq_local - args.seq_context
        input1_seq_local = target_seq_local + args.seq_context
        if input0_seq_local < 0 or input1_seq_local >= scene_count:
            skipped_targets[compare_seq] = "context crosses scene boundary"
            return

        anchor_idx = input0_seq_local * dataset.sample_hop
        if anchor_idx + dataset.window_hop >= len(dataset.scene_frames[scene_idx]):
            skipped_targets[compare_seq] = "window exceeds scene frames"
            return

        anchor_key = (int(scene_idx), int(anchor_idx))
        item_idx = existing_item_by_anchor.get(anchor_key)
        custom_anchor = item_idx is None
        if item_idx is None:
            item_idx = len(dataset.index)
            dataset.index.append(anchor_key)
            existing_item_by_anchor[anchor_key] = item_idx

        scene_start = scene_starts[scene_idx]
        input0_seq_global = scene_start + input0_seq_local
        target_seq_global = scene_start + target_seq_local
        input1_seq_global = scene_start + input1_seq_local

        seen_targets.add(compare_seq)
        jobs.append(
            {
                "item_idx": item_idx,
                "seq_meta": {
                    "seq_origin": args.seq_origin,
                    "seq_range_start": int(seq_range[0]),
                    "seq_range_end": int(seq_range[1]),
                    "input0_seq": int(input0_seq_global if args.seq_origin == "global" else input0_seq_local),
                    "target_seq": int(compare_seq),
                    "input1_seq": int(input1_seq_global if args.seq_origin == "global" else input1_seq_local),
                    "input0_seq_global": int(input0_seq_global),
                    "target_seq_global": int(target_seq_global),
                    "input1_seq_global": int(input1_seq_global),
                    "input0_seq_local": int(input0_seq_local),
                    "target_seq_local": int(target_seq_local),
                    "input1_seq_local": int(input1_seq_local),
                    "custom_anchor": bool(custom_anchor),
                },
            }
        )

    for target_seq, seq_range in target_to_range.items():
        if args.seq_origin == "global":
            found = _find_scene_for_global_seq(scene_starts, scene_counts, target_seq)
            if found is None:
                skipped_targets[target_seq] = "target outside all scenes"
                continue
            scene_idx, target_seq_local = found
            add_job(scene_idx, target_seq_local, target_seq, seq_range)
        else:
            for scene_idx, scene_count in enumerate(scene_counts):
                if 0 <= target_seq < scene_count:
                    add_job(scene_idx, target_seq, target_seq, seq_range)

    missing_targets = sorted(set(target_to_range.keys()) - seen_targets)
    jobs = jobs[args.start_index :]
    if args.max_items is not None:
        jobs = jobs[: args.max_items]

    return jobs, {
        "mode": "seq_ranges",
        "seq_ranges": seq_ranges,
        "seq_origin": args.seq_origin,
        "seq_step": args.seq_step,
        "seq_context": args.seq_context,
        "requested_targets": sorted(target_to_range.keys()),
        "missing_targets": missing_targets,
        "skipped_targets": skipped_targets,
        "num_dataset_items": original_num_items,
        "num_custom_anchors": len(dataset.index) - original_num_items,
    }


def as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        # Camera objects intentionally stay on CPU; render() moves their tensors
        # as needed, and nn.Module.to() would not move their plain tensor attrs.
        return [v if v.__class__.__name__ == "Camera" else move_to_device(v, device) for v in obj]
    return obj


def load_prefixed_state(module: torch.nn.Module, state_dict: dict, prefix: str, *, strict: bool) -> None:
    prefix_len = len(prefix)
    sub_state = {k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not sub_state and strict:
        raise RuntimeError(f"No checkpoint weights found for prefix {prefix!r}")
    missing, unexpected = module.load_state_dict(sub_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"State load mismatch for {prefix}: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if missing or unexpected:
        print(
            f"[warn] non-strict load for {prefix}: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )


def build_model(cfg, device: torch.device, checkpoint: str | None, *, strict: bool, allow_random: bool):
    p2g = Point2Gaus(cfg.p2g).to(device)
    g2g = GausTemp(cfg.g2g).to(device)
    g2p = GausRender(cfg.g2p).to(device)

    ckpt_path = checkpoint or getattr(cfg.test, "ckpt_path", None)
    if ckpt_path:
        ckpt_path = str(ckpt_path)
        print(f"[load] checkpoint: {ckpt_path}", flush=True)
        checkpoint_obj = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
        load_prefixed_state(p2g, state_dict, "p2g_model.", strict=strict)
        load_prefixed_state(g2g, state_dict, "g2g_model.", strict=False)
        load_prefixed_state(g2p, state_dict, "g2p_model.", strict=False)
    elif allow_random:
        print("[warn] no checkpoint supplied; evaluating random initialisation", flush=True)
    else:
        raise RuntimeError(
            "No checkpoint supplied. Pass --checkpoint path/to/epoch.ckpt or set test.ckpt_path=..."
        )

    p2g.eval()
    g2g.eval()
    g2p.eval()
    return p2g, g2g, g2p


def split_scene_names(dataset) -> list[str]:
    # Mirrors NuScenesNVSDataset.__init__ so scene_idx can be decoded without
    # changing the dataset return contract.
    from nuscenes.utils.splits import create_splits_scenes

    split_scenes = create_splits_scenes()[dataset.split]
    filtered = [scene for scene in dataset.nusc.scene if scene["name"] in split_scenes]
    scene_names = []
    for scene in filtered:
        frames = []
        first_sample = dataset.nusc.get("sample", scene["first_sample_token"])
        curr = first_sample["data"]["LIDAR_TOP"]
        while curr:
            sd = dataset.nusc.get("sample_data", curr)
            frames.append(curr)
            curr = sd["next"]
        if len(frames) >= dataset.window_frame_count:
            scene_names.append(scene["name"])
    return scene_names


def parse_scene_number(scene_name: str | None) -> int | None:
    if not scene_name:
        return None
    match = re.search(r"(\d+)$", scene_name)
    return int(match.group(1)) if match else None


def select_midpoint_gt(gt: dict, target_timestamp: float = 0.5) -> tuple[dict, list[int]]:
    selected = []
    midpoint_cameras = []
    for cameras in gt["cameras"]:
        if not cameras:
            raise RuntimeError("GT camera list is empty.")
        idx = min(range(len(cameras)), key=lambda i: abs(float(cameras[i].timestamp) - target_timestamp))
        selected.append(idx)
        midpoint_cameras.append([cameras[idx]])

    gt_mid = dict(gt)
    gt_mid["cameras"] = midpoint_cameras
    if "timestamps" in gt and isinstance(gt["timestamps"], list):
        gt_mid["timestamps"] = [ts[idx : idx + 1] for ts, idx in zip(gt["timestamps"], selected)]
    if "window_indices" in gt and isinstance(gt["window_indices"], list):
        gt_mid["window_indices"] = [wi[idx : idx + 1] for wi, idx in zip(gt["window_indices"], selected)]
    return gt_mid, selected


def clamp_for_metric(x: torch.Tensor, min_value: float, max_value: float) -> np.ndarray:
    return x.detach().float().cpu().numpy().clip(min_value, max_value)


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((gt - pred) ** 2)))


def median_abs_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.median(np.abs(gt - pred)))


def psnr(pred: np.ndarray, gt: np.ndarray, max_value: float) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_value ** 2 / mse))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_2d = np.squeeze(pred)
    gt_2d = np.squeeze(gt)
    data_range = float(np.max(gt_2d) - np.min(gt_2d))
    if data_range <= 0.0:
        return 1.0 if np.allclose(pred_2d, gt_2d) else 0.0
    return float(structural_similarity(pred_2d, gt_2d, data_range=data_range))


def range_to_points(range_image: torch.Tensor, camera, scale_factor: float) -> torch.Tensor:
    # GS-LiDAR PointsMeter divides range maps by args.scale_factor before
    # pano_to_lidar. This repo normally uses scale_factor=1.0, but keep the
    # official convention for config overrides.
    range_image = range_image.detach().clone() / float(scale_factor)
    range_image = torch.where(
        range_image > GS_LIDAR_POINT_FAR,
        torch.zeros_like(range_image),
        range_image,
    )
    points = pano_to_lidar(
        range_image,
        camera.vfov,
        camera.hfov,
        row_to_theta=camera.row_to_theta.to(device=range_image.device, dtype=range_image.dtype),
    )
    if points.numel() == 0:
        return points.reshape(0, 3)
    return points[points.norm(dim=1) > GS_LIDAR_POINT_NEAR]


def point_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    camera,
    chamfer,
    scale_factor: float,
) -> dict:
    pred_pts = range_to_points(pred_depth, camera, scale_factor)
    gt_pts = range_to_points(gt_depth, camera, scale_factor)
    result = {
        "pred_points": int(pred_pts.shape[0]),
        "gt_points": int(gt_pts.shape[0]),
        "point_cd": float("nan"),
        "point_fscore": 0.0,
        "point_precision": 0.0,
        "point_recall": 0.0,
    }
    if pred_pts.numel() == 0 or gt_pts.numel() == 0:
        return result

    dist1, dist2, _, _ = chamfer(
        pred_pts.unsqueeze(0).contiguous(),
        gt_pts.unsqueeze(0).contiguous(),
    )
    fs, precision, recall = fscore(dist1, dist2, GS_LIDAR_FSCORE_THRESHOLD)
    result.update(
        point_cd=as_float(dist1.mean() + dist2.mean()),
        point_fscore=as_float(fs[0]),
        point_precision=as_float(precision[0]),
        point_recall=as_float(recall[0]),
    )
    return result


def compute_metrics(all_renders: dict, camera, chamfer, scale_factor: float) -> dict:
    pred_depth = all_renders["depth"][0, 0]
    gt_depth = all_renders["gt_depth"][0, 0]
    pred_intensity = all_renders["intensity_sh"][0, 0]
    gt_intensity = all_renders["gt_intensity_sh"][0, 0]
    pred_raydrop = all_renders["raydrop"][0, 0]

    pred_keep = (pred_raydrop.detach() <= RAYDROP_KEEP_THRESHOLD).to(dtype=pred_depth.dtype)
    pred_depth_raydrop = pred_depth * pred_keep
    pred_intensity_raydrop = pred_intensity * pred_keep

    # GS-LiDAR DepthMeter divides depth by args.scale_factor, then clamps to
    # [1e-6, 80] before RMSE/MedAE/SSIM/PSNR.
    depth_pred_np = clamp_for_metric(
        pred_depth_raydrop / float(scale_factor),
        GS_LIDAR_MIN_VALUE,
        GS_LIDAR_DEPTH_MAX,
    )
    depth_gt_np = clamp_for_metric(
        gt_depth / float(scale_factor),
        GS_LIDAR_MIN_VALUE,
        GS_LIDAR_DEPTH_MAX,
    )
    int_pred_np = clamp_for_metric(pred_intensity_raydrop, GS_LIDAR_MIN_VALUE, GS_LIDAR_INTENSITY_MAX)
    int_gt_np = clamp_for_metric(gt_intensity, GS_LIDAR_MIN_VALUE, GS_LIDAR_INTENSITY_MAX)

    metrics = {
        "depth_rmse": rmse(depth_pred_np, depth_gt_np),
        "depth_median_abs_error": median_abs_error(depth_pred_np, depth_gt_np),
        "depth_ssim": ssim(depth_pred_np, depth_gt_np),
        "depth_psnr": psnr(depth_pred_np, depth_gt_np, GS_LIDAR_DEPTH_MAX),
        "intensity_rmse": rmse(int_pred_np, int_gt_np),
        "intensity_ssim": ssim(int_pred_np, int_gt_np),
        "intensity_psnr": psnr(int_pred_np, int_gt_np, GS_LIDAR_INTENSITY_MAX),
        "pred_keep_ratio": as_float(pred_keep.mean()),
        "gt_keep_ratio": as_float((gt_depth > 0).float().mean()),
    }
    metrics.update(point_metrics(pred_depth_raydrop, gt_depth, camera, chamfer, scale_factor))
    return metrics


def metric_maps_for_visualization(all_renders: dict, scale_factor: float) -> dict[str, np.ndarray]:
    pred_depth = all_renders["depth"][0, 0]
    gt_depth = all_renders["gt_depth"][0, 0]
    pred_intensity = all_renders["intensity_sh"][0, 0]
    gt_intensity = all_renders["gt_intensity_sh"][0, 0]
    pred_raydrop = all_renders["raydrop"][0, 0]

    pred_keep = (pred_raydrop.detach() <= RAYDROP_KEEP_THRESHOLD).to(dtype=pred_depth.dtype)
    pred_depth_raydrop = pred_depth * pred_keep
    pred_intensity_raydrop = pred_intensity * pred_keep
    gt_keep = (gt_depth > 0).to(dtype=pred_depth.dtype)

    pred_depth_np = np.squeeze(
        (pred_depth_raydrop / float(scale_factor)).detach().float().cpu().numpy()
    )
    gt_depth_np = np.squeeze((gt_depth / float(scale_factor)).detach().float().cpu().numpy())
    pred_int_np = np.squeeze(pred_intensity_raydrop.detach().float().cpu().numpy())
    gt_int_np = np.squeeze(gt_intensity.detach().float().cpu().numpy())
    pred_keep_np = np.squeeze(pred_keep.detach().float().cpu().numpy())
    gt_keep_np = np.squeeze(gt_keep.detach().float().cpu().numpy())

    return {
        "pred_depth": pred_depth_np,
        "gt_depth": gt_depth_np,
        "depth_abs_error": np.abs(pred_depth_np - gt_depth_np),
        "pred_intensity": pred_int_np,
        "gt_intensity": gt_int_np,
        "intensity_abs_error": np.abs(pred_int_np - gt_int_np),
        "pred_keep": pred_keep_np,
        "gt_keep": gt_keep_np,
    }


def _imshow(ax, image: np.ndarray, title: str, *, cmap: str, vmin=None, vmax=None) -> None:
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def save_visualization(record: dict, all_renders: dict, output_dir: Path, scale_factor: float) -> str:
    maps = metric_maps_for_visualization(all_renders, scale_factor)
    vis_dir = output_dir / "visualizations" / record["sequence_id"]
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / f"{record['input_id']}.png"

    fig, axes = plt.subplots(3, 3, figsize=(15, 7), constrained_layout=True)
    fig.suptitle(
        f"{record['input_id']} | target={record['target_scene_time_s']:.2f}s | "
        f"CD={record['point_cd']:.4f} F={record['point_fscore']:.3f}",
        fontsize=11,
    )
    _imshow(axes[0, 0], maps["pred_depth"], "Pred Depth (raydrop)", cmap="viridis", vmin=0, vmax=GS_LIDAR_DEPTH_MAX)
    _imshow(axes[0, 1], maps["gt_depth"], "GT Depth", cmap="viridis", vmin=0, vmax=GS_LIDAR_DEPTH_MAX)
    _imshow(axes[0, 2], maps["depth_abs_error"], "Depth |error|", cmap="magma", vmin=0)
    _imshow(axes[1, 0], maps["pred_intensity"], "Pred Intensity (raydrop)", cmap="gray", vmin=0, vmax=1)
    _imshow(axes[1, 1], maps["gt_intensity"], "GT Intensity", cmap="gray", vmin=0, vmax=1)
    _imshow(axes[1, 2], maps["intensity_abs_error"], "Intensity |error|", cmap="magma", vmin=0, vmax=1)
    _imshow(axes[2, 0], maps["pred_keep"], "Pred Keep Mask", cmap="gray", vmin=0, vmax=1)
    _imshow(axes[2, 1], maps["gt_keep"], "GT Keep Mask", cmap="gray", vmin=0, vmax=1)
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        0.9,
        "\n".join(
            [
                f"Depth RMSE: {record['depth_rmse']:.4f}",
                f"Depth MedAE: {record['depth_median_abs_error']:.4f}",
                f"Depth SSIM: {record['depth_ssim']:.4f}",
                f"Depth PSNR: {record['depth_psnr']:.2f}",
                f"Intensity RMSE: {record['intensity_rmse']:.4f}",
                f"Intensity SSIM: {record['intensity_ssim']:.4f}",
                f"Intensity PSNR: {record['intensity_psnr']:.2f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(output_dir))


def metadata_for_item(
    dataset,
    scene_names: list[str],
    item_idx: int,
    selected_gt_idx: int,
    sequence_offset: int,
    seq_meta: dict | None = None,
) -> dict:
    scene_idx, anchor_idx = dataset.index[item_idx]
    scene_name = scene_names[scene_idx] if scene_idx < len(scene_names) else None
    sequence_id = f"sequence{scene_idx + sequence_offset:03d}"
    window = dataset._sample_frames(scene_idx, anchor_idx)
    gt_window_indices = dataset._select_gt_indices(len(window))
    input_window_indices = dataset._select_input_indices(len(window))
    target_window_index = int(gt_window_indices[selected_gt_idx])
    scene_start_us = dataset.scene_frames[scene_idx][0][2]

    def frame_meta(window_index: int) -> dict:
        lidar_token, sample_token, timestamp_us, is_key_frame = window[window_index]
        return {
            "lidar_token": lidar_token,
            "sample_token": sample_token,
            "scene_time_s": (timestamp_us - scene_start_us) / 1_000_000.0,
            "is_key_frame": bool(is_key_frame),
            "window_index": int(window_index),
        }

    input0 = frame_meta(input_window_indices[0])
    input1 = frame_meta(input_window_indices[-1])
    target = frame_meta(target_window_index)
    input_id = f"{sequence_id}_item{item_idx:06d}"
    if seq_meta is not None:
        input_id = f"{sequence_id}_seq{int(seq_meta['target_seq']):06d}"

    record = {
        "input_id": input_id,
        "sequence_id": sequence_id,
        "scene_idx": int(scene_idx),
        "scene_name": scene_name,
        "scene_number": parse_scene_number(scene_name),
        "anchor_idx": int(anchor_idx),
        "item_idx": int(item_idx),
        "target_gt_camera_index": int(selected_gt_idx),
        "input0_window_index": input0["window_index"],
        "input1_window_index": input1["window_index"],
        "target_window_index": target["window_index"],
        "input0_scene_time_s": input0["scene_time_s"],
        "input1_scene_time_s": input1["scene_time_s"],
        "target_scene_time_s": target["scene_time_s"],
        "input0_lidar_token": input0["lidar_token"],
        "input1_lidar_token": input1["lidar_token"],
        "target_lidar_token": target["lidar_token"],
        "input0_sample_token": input0["sample_token"],
        "input1_sample_token": input1["sample_token"],
        "target_sample_token": target["sample_token"],
        "target_is_key_frame": target["is_key_frame"],
    }
    if seq_meta is not None:
        record.update(seq_meta)
    return record


def finite_values(records: list[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def aggregate(records: list[dict]) -> dict:
    out = {}
    for key in METRIC_KEYS + ["pred_keep_ratio", "gt_keep_ratio", "pred_points", "gt_points"]:
        values = finite_values(records, key)
        out[key] = {
            "mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values)) if values else float("nan"),
            "count": len(values),
        }
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args, overrides = parse_args()
    set_seed(args.seed)
    config_path = Path(args.config).resolve()
    base_conf = OmegaConf.load(config_path)
    override_conf = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(base_conf, override_conf)

    split = args.split or getattr(cfg.data, "test_split", "test")
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(cfg.logger.dir)) / f"midpoint_test_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    dataset_cls = dataset_dict[cfg.data.dataset_name]
    dataset = dataset_cls(cfg=cfg.data, split=split)
    scene_names = split_scene_names(dataset)
    eval_jobs, plan_info = build_eval_jobs(dataset, args)

    if args.dry_run_plan:
        print(json.dumps(plan_info, indent=2), flush=True)
        for preview in eval_jobs[:20]:
            print(json.dumps(preview, allow_nan=True), flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Point cloud CD/F-score uses the CUDA Chamfer extension; run with --device cuda.")

    torch.set_float32_matmul_precision("high")
    p2g, g2g, g2p = build_model(
        cfg,
        device,
        args.checkpoint,
        strict=not args.non_strict,
        allow_random=args.allow_random_init,
    )
    chamfer = chamfer_3DDist().to(device)
    scale_factor = float(getattr(cfg.g2p, "scale_factor", 1.0) or 1.0)

    records = []
    jsonl_path = output_dir / "per_input_metrics.jsonl"
    start_time = time.time()
    saved_visualizations = 0
    save_visualizations = not args.no_save_visualizations

    print(
        f"[eval] split={split} jobs={len(eval_jobs)} dataset_items={len(dataset)} "
        f"output={output_dir}",
        flush=True,
    )
    print(
        f"[config] path={config_path} anchor_mode={cfg.p2g.anchor_mode} "
        f"scale_factor={scale_factor}",
        flush=True,
    )
    if plan_info["mode"] == "seq_ranges":
        print(
            f"[seq-plan] ranges={plan_info['seq_ranges']} origin={plan_info['seq_origin']} "
            f"step={plan_info['seq_step']} context={plan_info['seq_context']} "
            f"requested={len(plan_info['requested_targets'])} matched={len(eval_jobs)} "
            f"missing={len(plan_info['missing_targets'])}",
            flush=True,
        )
        if plan_info["missing_targets"]:
            print(f"[seq-plan] missing_targets={plan_info['missing_targets'][:20]}", flush=True)

    with jsonl_path.open("w") as jsonl:
        for eval_idx, job in enumerate(eval_jobs):
            item_idx = job["item_idx"]
            item = dataset[item_idx]
            batch = multiframe_collate_fn([item])
            gt_mid, selected_indices = select_midpoint_gt(batch["gt"])
            selected_gt_idx = selected_indices[0]
            target_camera = gt_mid["cameras"][0][0]

            batch_input = move_to_device(batch["input"], device)
            gt_mid = move_to_device(gt_mid, device)

            with torch.inference_mode():
                out = p2g(batch_input, batch_idx=item_idx, mode="test")
                out = g2g(out, batch_input["timestamps"])
                all_renders = g2p(out, gt_mid)
                metrics = compute_metrics(all_renders, target_camera, chamfer, scale_factor)

            record = metadata_for_item(
                dataset,
                scene_names,
                item_idx,
                selected_gt_idx,
                args.sequence_offset,
                job.get("seq_meta"),
            )
            record["target_normalized_timestamp"] = float(target_camera.timestamp)
            record.update(metrics)

            should_save_vis = (
                save_visualizations
                and args.vis_every > 0
                and (eval_idx % args.vis_every == 0)
                and (args.vis_max is None or saved_visualizations < args.vis_max)
            )
            if should_save_vis:
                record["visualization"] = save_visualization(
                    record,
                    all_renders,
                    output_dir,
                    scale_factor,
                )
                saved_visualizations += 1

            records.append(record)
            jsonl.write(json.dumps(record, allow_nan=True) + "\n")
            jsonl.flush()

            if args.print_every > 0 and (len(records) == 1 or len(records) % args.print_every == 0):
                elapsed = time.time() - start_time
                print(
                    f"[{len(records):6d}/{len(eval_jobs)}] "
                    f"{record['input_id']} "
                    f"CD={record['point_cd']:.6f} "
                    f"F={record['point_fscore']:.4f} "
                    f"D_RMSE={record['depth_rmse']:.4f} "
                    f"I_RMSE={record['intensity_rmse']:.4f} "
                    f"elapsed={elapsed/60.0:.1f}m",
                    flush=True,
                )

    write_csv(output_dir / "per_input_metrics.csv", records)
    summary = {
        "split": split,
        "num_items": len(records),
        "num_visualizations": saved_visualizations,
        "start_index": args.start_index,
        "end_index": args.start_index + len(eval_jobs),
        "checkpoint": args.checkpoint or getattr(cfg.test, "ckpt_path", None),
        "config": str(config_path),
        "overrides": overrides,
        "anchor_mode": str(cfg.p2g.anchor_mode),
        "gs_params": OmegaConf.to_container(cfg.p2g.gs_params, resolve=True),
        "seed": args.seed,
        "plan": plan_info,
        "metric_convention": {
            "source": "GS-LiDAR metrics_utils.py",
            "raydrop_keep_threshold": RAYDROP_KEEP_THRESHOLD,
            "depth_max": GS_LIDAR_DEPTH_MAX,
            "intensity_max": GS_LIDAR_INTENSITY_MAX,
            "fscore_threshold": GS_LIDAR_FSCORE_THRESHOLD,
            "scale_factor": scale_factor,
        },
        "aggregate": aggregate(records),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(f"[done] wrote {jsonl_path}", flush=True)
    print(f"[done] wrote {output_dir / 'per_input_metrics.csv'}", flush=True)
    print(f"[done] wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
