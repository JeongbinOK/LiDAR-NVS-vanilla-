"""Profile one or more QGS pair steps and report loss/coverage diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QGSConfig
from nn.lidar_geometry import make_lidar_ray_grid

DropHead = None  # legacy diagnostic; per-Gaussian raydrop replaces DropHead.
from nn.eval_utils import evaluate_pair_sample, load_cfg_from_checkpoint, resolve_bbox_json
from nn.model import QGSModel
from nn.qgs_loss import QGSLoss
from train import process_pair


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_single_batch(sample: dict) -> dict:
    batch = {}
    list_keys = {
        "input_0",
        "input_1",
        "boxes_0",
        "boxes_1",
        "instance_ids_0",
        "instance_ids_1",
    }
    for key, value in sample.items():
        if key in list_keys:
            batch[key] = [value]
        elif torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def _make_loss(cfg: QGSConfig, device: torch.device) -> QGSLoss:
    return QGSLoss(
        w_depth=cfg.loss_w_range,
        w_intensity=cfg.loss_w_intensity,
        w_raydrop=cfg.loss_w_raydrop,
        w_distortion=cfg.loss_w_distortion,
        w_normal=cfg.loss_w_normal,
        alpha_eps=cfg.loss_alpha_eps,
    ).to(device)


def _raw_intensity_stats(samples: list[dict]) -> dict:
    values = []
    for sample in samples:
        values.append(sample["input_0"][:, 3].float())
        values.append(sample["input_1"][:, 3].float())
    raw = torch.cat(values)
    rounded = torch.isclose(raw, raw.round(), rtol=0.0, atol=1e-6).float().mean()
    return {
        "dtype": str(raw.dtype),
        "min": float(raw.min().item()),
        "p01": float(torch.quantile(raw, 0.01).item()),
        "mean": float(raw.mean().item()),
        "p50": float(torch.quantile(raw, 0.50).item()),
        "p99": float(torch.quantile(raw, 0.99).item()),
        "max": float(raw.max().item()),
        "integer_like_ratio": float(rounded.item()),
        "normalized_mean": float((raw / 255.0).clamp(0.0, 1.0).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pair-idx", type=int, default=0)
    parser.add_argument("--num-pairs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--lidar-sigma", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_cfg_from_checkpoint(args.checkpoint) if args.checkpoint else QGSConfig()
    cfg.device = args.device
    if args.lidar_sigma is not None:
        cfg.lidar_sigma = float(args.lidar_sigma)
    target_split = cfg.train_split if args.split == "train" else cfg.eval_split
    sys.path.insert(0, os.path.join(os.path.expanduser(cfg.data_root), "loader"))
    from dataset import NuScenesNVSDataset  # noqa: WPS433

    bbox_json = resolve_bbox_json(cfg, target_split)
    dataset = NuScenesNVSDataset(
        dataroot=os.path.expanduser(cfg.data_root),
        version=cfg.nuscenes_version,
        split=target_split,
        frame_gap=cfg.frame_gap,
        mode=cfg.dataset_mode,
        bbox_json_path=bbox_json if (cfg.dataset_mode == "bbox" and bbox_json) else None,
    )
    pair_indices = [args.pair_idx + i for i in range(args.num_pairs)]
    samples = [dataset[i] for i in pair_indices]

    model = QGSModel(cfg).to(device)
    drop_head = DropHead(latent_dim=cfg.lidar_latent_dim).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "drop_head_state_dict" in ckpt:
            drop_head.load_state_dict(ckpt["drop_head_state_dict"], strict=False)

    loss_fn = _make_loss(cfg, device)
    ray_grid = make_lidar_ray_grid(cfg, device=device)

    pair_reports = []
    for pair_idx, sample in zip(pair_indices, samples):
        batch = _make_single_batch(sample)
        model.train()
        drop_head.train()
        model.zero_grad(set_to_none=True)
        drop_head.zero_grad(set_to_none=True)
        profile: dict = {}
        _sync(device)
        step_start = time.perf_counter()
        out = process_pair(model, loss_fn, drop_head, ray_grid, batch, 0, device, cfg, profile=profile)
        _sync(device)
        forward_ms = (time.perf_counter() - step_start) * 1000.0
        backward_ms = 0.0
        if out is not None and args.backward:
            _sync(device)
            backward_start = time.perf_counter()
            out["total"].backward()
            _sync(device)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

        model.eval()
        drop_head.eval()
        with torch.no_grad():
            eval_report = evaluate_pair_sample(
                model,
                drop_head,
                loss_fn,
                sample,
                device=args.device,
                cfg=cfg,
                hit_threshold=args.hit_threshold,
            )

        loss_report = {}
        if out is not None:
            loss_report = {k: float(v.detach().item()) for k, v in out.items() if torch.is_tensor(v)}
        pair_reports.append({
            "pair_idx": pair_idx,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "profile": profile,
            "loss": loss_report,
            "eval_summary": eval_report["summary"],
        })

    report = {
        "device": args.device,
        "checkpoint": args.checkpoint,
        "primitive_mode": cfg.primitive_mode,
        "lidar_sigma": cfg.lidar_sigma,
        "pairs": pair_reports,
        "raw_intensity": _raw_intensity_stats(samples),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
