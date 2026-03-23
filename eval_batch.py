"""Batch evaluation script for the clustering pipeline."""

import argparse
import os
import sys

import numpy as np
import torch

from gaustering.config import ClusteringConfig
from gaustering.data_loader import list_lidar_files
from gaustering.pipeline import cluster_frame


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate gaustering pipeline")
    parser.add_argument(
        "--data-root",
        default=os.path.expanduser("~/data/nuScenes"),
        help="Path to nuScenes dataset root",
    )
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to evaluate")
    parser.add_argument("--save-dir", default="eval_results", help="Directory for output")
    args = parser.parse_args()

    files = list_lidar_files(args.data_root)
    if not files:
        print(f"No .pcd.bin files found under {args.data_root}")
        sys.exit(1)

    files = files[: args.num_frames]
    print(f"Evaluating {len(files)} frames...")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("-" * 60)

    cfg = ClusteringConfig()

    stats = {
        "gamma_rms_median": [],
        "gamma_rms_mean": [],
        "cluster_size_mean": [],
        "cluster_size_median": [],
        "num_gaussians": [],
        "coverage": [],
        "ground_ratio": [],
        "total_time": [],
        "gpu_peak_mb": [],
    }

    for i, path in enumerate(files):
        result = cluster_frame(path, cfg)

        gamma_rms = result["_gamma_rms"]
        counts = result["_counts"]
        coverage = result["point_mask"].float().mean().item()
        ground_ratio = result["ground_mask"].float().mean().item()

        stats["gamma_rms_median"].append(gamma_rms.median().item())
        stats["gamma_rms_mean"].append(gamma_rms.mean().item())
        stats["cluster_size_mean"].append(counts.float().mean().item())
        stats["cluster_size_median"].append(counts.float().median().item())
        stats["num_gaussians"].append(result["num_gaussians"])
        stats["coverage"].append(coverage)
        stats["ground_ratio"].append(ground_ratio)
        stats["total_time"].append(result["timings"]["total"])
        stats["gpu_peak_mb"].append(result["gpu_peak_mb"])

        print(
            f"  [{i+1}/{len(files)}] {os.path.basename(path)}: "
            f"G={result['num_gaussians']}, "
            f"gamma_rms={gamma_rms.median():.5f}, "
            f"cov={coverage:.4f}, "
            f"ground={ground_ratio:.2f}, "
            f"t={result['timings']['total']:.2f}s, "
            f"mem={result['gpu_peak_mb']:.0f}MB"
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Summary ({len(files)} frames)")
    print(f"{'='*60}")
    for key, values in stats.items():
        arr = np.array(values)
        print(
            f"  {key:25s}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
            f"min={arr.min():.4f}, max={arr.max():.4f}"
        )
    print(f"{'='*60}")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "batch_stats.npz")
    np.savez(out_path, **{k: np.array(v) for k, v in stats.items()})
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
