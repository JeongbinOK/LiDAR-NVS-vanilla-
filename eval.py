"""Evaluation script for Quadratic Gaussian Splatting (QGS)."""

import argparse
import os
import sys

import torch
import numpy as np

from config import QGSConfig
sys.path.insert(0, os.path.join(os.path.expanduser(QGSConfig.data_root), "loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn

from nn.model import QGSModel
# TODO: Import your QGSLoss here
# from nn.qgs_loss import QGSLoss


def evaluate_frame(model, loss_fn, pts, device, ego_radius):
    """Run model on a single frame, return metrics + output."""
    xyz = pts[:, :3].to(device)
    intensity = pts[:, 3:4].to(device)

    mask = torch.norm(xyz, dim=1) > ego_radius
    xyz, intensity = xyz[mask], intensity[mask]

    with torch.no_grad():
        output = model(xyz, intensity)
        
        # TODO: compute actual loss and metrics using loss_fn
        # loss_dict = loss_fn(xyz, output)
        metrics = {
            "dummy_loss": 0.0,
            "N": xyz.shape[0],
        }

    return metrics, output, xyz


def main():
    parser = argparse.ArgumentParser(description="Evaluate QGS model")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--data-root", default=os.path.expanduser("~/data/nuScenes"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--save-viz", action="store_true", help="Save BEV plots")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = QGSConfig(device=args.device)

    # Load model
    model = QGSModel(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, train_loss {ckpt['loss']:.4f}")

    # TODO: Instantiate QGSLoss
    loss_fn = None

    # Dataset
    dataset = NuScenesNVSDataset(
        dataroot=args.data_root, version="v1.0-trainval", split=args.split,
    )
    num_frames = min(args.num_frames, len(dataset))
    print(f"Evaluating {num_frames} frames from {args.split} split\n")

    # Eval loop
    all_metrics = []
    viz_dir = os.path.join("outputs", "eval_viz")
    if args.save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    for i in range(num_frames):
        sample = dataset[i]
        pts = sample["input_0"]  # [N, 4]

        metrics, output, xyz = evaluate_frame(
            model, loss_fn, pts, args.device, cfg.ego_radius,
        )
        all_metrics.append(metrics)

        status = (
            f"[{i+1:3d}/{num_frames}] "
            f"N={metrics['N']:5d} loss={metrics['dummy_loss']:.4f}"
        )
        print(status)

        if args.save_viz and i < 10:
            # TODO: Implement QGS specific plotting
            pass

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for key in ["dummy_loss", "N"]:
        vals = [m[key] for m in all_metrics]
        print(
            f"{key:<15} {np.mean(vals):10.4f} {np.std(vals):10.4f} "
            f"{np.min(vals):10.4f} {np.max(vals):10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

