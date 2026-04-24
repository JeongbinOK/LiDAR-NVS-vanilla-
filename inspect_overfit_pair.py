"""Inspect one pair checkpoint and save pointcloud/diagnostic artifacts."""

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


def main():
    parser = argparse.ArgumentParser(description="Inspect pair-wise pointcloud quality for one checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-idx", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_cfg_from_checkpoint(args.checkpoint)
    cfg.device = args.device
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
    sample = dataset[args.pair_idx]

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

    with torch.no_grad():
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
        "pair_idx": args.pair_idx,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_loss": ckpt.get("loss"),
        **result["summary"],
    }

    ckpt_dir = Path(os.path.abspath(args.checkpoint)).parent
    out_dir = ckpt_dir.parent / f"inspect_pair_{args.pair_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = result["artifacts"]
    write_ply(out_dir / "pred_frame0.ply", artifacts["pred_frame0_pts"], artifacts["pred_frame0_intensity"])
    write_ply(out_dir / "gt_frame0.ply", artifacts["gt_frame0_pts"], artifacts["gt_frame0_intensity"])
    write_ply(out_dir / "pred_frame1.ply", artifacts["pred_frame1_pts"], artifacts["pred_frame1_intensity"])
    write_ply(out_dir / "gt_frame1.ply", artifacts["gt_frame1_pts"], artifacts["gt_frame1_intensity"])
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"saved pointclouds to {out_dir}")


if __name__ == "__main__":
    main()
