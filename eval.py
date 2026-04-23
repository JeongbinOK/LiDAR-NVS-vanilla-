"""Pair-wise evaluation script for Quadratic Gaussian Splatting (QGS)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from config import QGSConfig
from models.head import DropHead
from nn.eval_utils import (
    evaluate_pair_sample,
    load_cfg_from_checkpoint,
    resolve_bbox_json,
    write_ply,
)
from nn.model import QGSModel
from nn.qgs_loss import QGSLoss


def _aggregate_pair_summaries(pair_summaries: list[dict]) -> dict:
    metric_paths = [
        ("frame0", "loss_total"),
        ("frame0", "loss_depth"),
        ("frame0", "loss_intensity"),
        ("frame0", "loss_raydrop"),
        ("frame0", "hit_precision"),
        ("frame0", "hit_recall"),
        ("frame0", "hit_f1"),
        ("frame0", "depth_mae_on_gt"),
        ("frame0", "approx_chamfer_m"),
        ("frame1", "loss_total"),
        ("frame1", "loss_depth"),
        ("frame1", "loss_intensity"),
        ("frame1", "loss_raydrop"),
        ("frame1", "hit_precision"),
        ("frame1", "hit_recall"),
        ("frame1", "hit_f1"),
        ("frame1", "depth_mae_on_gt"),
        ("frame1", "approx_chamfer_m"),
    ]
    aggregate = {"num_pairs": len(pair_summaries)}
    for frame_key, metric_key in metric_paths:
        values = [pair[frame_key][metric_key] for pair in pair_summaries]
        name = f"{frame_key}_{metric_key}"
        aggregate[name] = {
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Evaluate pair-wise QGS checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--pair-idx", type=int, default=None)
    parser.add_argument("--num-pairs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--center-mode", default="", choices=["", "fixed"], help="optional override")
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--save-pointclouds", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_cfg_from_checkpoint(args.checkpoint)
    cfg.device = args.device
    if args.data_root:
        cfg.data_root = args.data_root
    if args.center_mode:
        cfg.head_center_mode = args.center_mode

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

    if args.pair_idx is not None:
        pair_indices = [args.pair_idx]
    else:
        pair_indices = list(range(min(args.num_pairs, len(dataset))))

    model = QGSModel(cfg).to(args.device)
    drop_head = DropHead(latent_dim=cfg.lidar_latent_dim).to(args.device)
    loss_fn = QGSLoss(
        w_depth=cfg.loss_w_range,
        w_intensity=cfg.loss_w_intensity,
        w_raydrop=cfg.loss_w_raydrop,
        alpha_eps=cfg.loss_alpha_eps,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "drop_head_state_dict" in ckpt:
        drop_head.load_state_dict(ckpt["drop_head_state_dict"], strict=False)
    model.eval()
    drop_head.eval()

    ckpt_dir = Path(os.path.abspath(args.checkpoint)).parent
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ckpt_dir.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_summaries = []
    with torch.no_grad():
        for pair_idx in pair_indices:
            sample = dataset[pair_idx]
            result = evaluate_pair_sample(
                model,
                drop_head,
                loss_fn,
                sample,
                device=args.device,
                cfg=cfg,
                hit_threshold=args.hit_threshold,
            )

            summary = {
                "pair_idx": pair_idx,
                "checkpoint_epoch": ckpt.get("epoch"),
                "checkpoint_loss": ckpt.get("loss"),
                "center_mode": cfg.head_center_mode,
                **result["summary"],
            }
            pair_summaries.append(summary)

            pair_dir = out_dir / f"pair_{pair_idx:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            with open(pair_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open(pair_dir / "diagnostics.json", "w") as f:
                json.dump(summary["scene"]["contexts"], f, indent=2)

            if args.save_pointclouds:
                artifacts = result["artifacts"]
                write_ply(pair_dir / "pred_frame0.ply", artifacts["pred_frame0_pts"], artifacts["pred_frame0_intensity"])
                write_ply(pair_dir / "gt_frame0.ply", artifacts["gt_frame0_pts"], artifacts["gt_frame0_intensity"])
                write_ply(pair_dir / "pred_frame1.ply", artifacts["pred_frame1_pts"], artifacts["pred_frame1_intensity"])
                write_ply(pair_dir / "gt_frame1.ply", artifacts["gt_frame1_pts"], artifacts["gt_frame1_intensity"])

            print(
                f"[pair {pair_idx:04d}] "
                f"f0 depth={summary['frame0']['depth_mae_on_gt']:.3f} "
                f"f1 depth={summary['frame1']['depth_mae_on_gt']:.3f} "
                f"f0 touched={summary['frame0']['gaussians']['overall']['n_touched']} "
                f"f1 touched={summary['frame1']['gaussians']['overall']['n_touched']}"
            )

    aggregate = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_loss": ckpt.get("loss"),
        "center_mode": cfg.head_center_mode,
        "split": args.split,
        "pairs": pair_indices,
        "aggregate": _aggregate_pair_summaries(pair_summaries),
    }
    with open(out_dir / "aggregate_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(json.dumps(aggregate, indent=2))
    print(f"saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
