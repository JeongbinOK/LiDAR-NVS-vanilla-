"""Evaluation script for neural 2D Gaussian clustering."""

import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.expanduser("~/data/nuScenes/loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn

from config import NeuralClusteringConfig
from nn.model import NeuralClusteringModel
from nn.losses import ClusteringLoss


def evaluate_frame(model, loss_fn, pts, device, tau, ego_radius):
    """Run model on a single frame, return metrics + output."""
    xyz = pts[:, :3].to(device)
    intensity = pts[:, 3:4].to(device)

    mask = torch.norm(xyz, dim=1) > ego_radius
    xyz, intensity = xyz[mask], intensity[mask]

    with torch.no_grad():
        output = model(xyz, intensity, tau=tau)
        loss_dict = loss_fn(xyz, output)

    gaussians = output["gaussians"]
    assign_indices = output["assign_indices"]
    assign_weights = output["assign_weights"]
    mu, n = gaussians["mu"], gaussians["n"]

    hard = assign_indices.gather(
        1, assign_weights.argmax(dim=1, keepdim=True),
    ).squeeze(1)

    gamma = ((xyz - mu[hard]) * n[hard]).sum(dim=1)
    gamma_rms = gamma.pow(2).mean().sqrt().item()
    coverage = (assign_weights.max(dim=1).values > 0.1).float().mean().item()

    metrics = {
        "loss": loss_dict["total"].item(),
        "surface": loss_dict["surface"].item(),
        "compact": loss_dict["compact"].item(),
        "scale": loss_dict["scale"].item(),
        "gamma_rms": gamma_rms,
        "coverage": coverage,
        "K": mu.shape[0],
        "N": xyz.shape[0],
    }
    return metrics, output, xyz


def save_bev(xyz, output, path, title=""):
    """Save BEV visualization of clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz_np = xyz.cpu().numpy()
    assign_indices = output["assign_indices"]
    assign_weights = output["assign_weights"]
    mu = output["gaussians"]["mu"].cpu().numpy()
    s = output["gaussians"]["s"].cpu().numpy()

    hard = assign_indices.gather(
        1, assign_weights.argmax(dim=1, keepdim=True),
    ).squeeze(1).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    scatter = ax.scatter(
        xyz_np[:, 0], xyz_np[:, 1],
        c=hard, cmap="tab20", s=0.3, alpha=0.6, rasterized=True,
    )
    ax.scatter(mu[:, 0], mu[:, 1], c="red", s=10, marker="x", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural clustering")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--data-root", default=os.path.expanduser("~/data/nuScenes"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.1, help="Gumbel tau for eval")
    parser.add_argument("--save-viz", action="store_true", help="Save BEV plots")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = NeuralClusteringConfig(device=args.device)

    # Load model
    model = NeuralClusteringModel(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, train_loss {ckpt['loss']:.4f}")

    loss_fn = ClusteringLoss(
        w_surface=cfg.w_surface,
        w_assign=cfg.w_assign,
        w_scale=cfg.w_scale,
    )

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
            model, loss_fn, pts, args.device, args.tau, cfg.ego_radius,
        )
        all_metrics.append(metrics)

        status = (
            f"[{i+1:3d}/{num_frames}] "
            f"N={metrics['N']:5d} K={metrics['K']:3d} "
            f"gamma={metrics['gamma_rms']:.4f} "
            f"S={metrics['surface']:.4f} "
            f"cov={metrics['coverage']:.3f}"
        )
        print(status)

        if args.save_viz and i < 10:
            save_bev(
                xyz, output,
                os.path.join(viz_dir, f"frame_{i:03d}.png"),
                title=f"Frame {i} | K={metrics['K']} gamma={metrics['gamma_rms']:.3f}",
            )

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for key in ["gamma_rms", "surface", "compact", "scale", "coverage", "K", "N"]:
        vals = [m[key] for m in all_metrics]
        print(
            f"{key:<15} {np.mean(vals):10.4f} {np.std(vals):10.4f} "
            f"{np.min(vals):10.4f} {np.max(vals):10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
