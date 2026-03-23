"""Test script: run pipeline on a single nuScenes frame and report results."""

import argparse
import os
import sys

import torch

from gaustering.config import ClusteringConfig
from gaustering.data_loader import list_lidar_files
from gaustering.pipeline import cluster_frame
from gaustering.visualization import plot_statistics, plot_bev, plot_quality_heatmap


def main():
    parser = argparse.ArgumentParser(description="Test gaustering on a nuScenes frame")
    parser.add_argument(
        "--data-root",
        default=os.path.expanduser("~/data/nuScenes"),
        help="Path to nuScenes dataset root",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to process")
    parser.add_argument("--save-dir", default=".", help="Directory to save output plots")
    args = parser.parse_args()

    files = list_lidar_files(args.data_root)
    if not files:
        print(f"No .pcd.bin files found under {args.data_root}/samples/LIDAR_TOP/")
        sys.exit(1)

    path = files[min(args.frame_idx, len(files) - 1)]
    print(f"File: {path}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("-" * 60)

    cfg = ClusteringConfig()
    result = cluster_frame(path, cfg)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Raw points:      {result['num_raw_points']}")
    print(f"  Filtered points: {result['num_filtered_points']}")
    print(f"  Gaussians:       {result['num_gaussians']}")

    # Coverage
    coverage = result["point_mask"].float().mean().item()
    print(f"  Coverage:        {coverage:.4f} ({coverage*100:.1f}%)")

    # Ground / non-ground breakdown
    ground_mask = result["ground_mask"]
    n_ground = ground_mask.sum().item()
    n_total = ground_mask.shape[0]
    print(f"  Ground points:   {n_ground} / {n_total} ({100*n_ground/n_total:.1f}%)")

    counts = result["_counts"]
    print(f"\n  Cluster size -- mean: {counts.float().mean():.1f}, "
          f"median: {counts.float().median():.1f}, "
          f"min: {counts.min().item()}, max: {counts.max().item()}")

    gamma_rms = result["_gamma_rms"]
    print(f"  gamma_rms -- mean: {gamma_rms.mean():.5f}, "
          f"median: {gamma_rms.median():.5f}, "
          f"max: {gamma_rms.max():.5f}")

    # Reparameterization error
    mask = result["point_mask"]
    xyz = result["source_xyz"][mask]
    a = result["point_assignments"][mask]
    alphas = result["alphas"][mask]
    betas = result["betas"][mask]
    gammas = result["gammas"][mask]

    mu = result["xyz"][a]
    tu = result["tangent_u"][a]
    tv = result["tangent_v"][a]
    tn = result["normal"][a]

    recon = mu + alphas.unsqueeze(1) * tu + betas.unsqueeze(1) * tv + gammas.unsqueeze(1) * tn
    error = (xyz - recon).pow(2).sum(dim=1).sqrt()
    print(f"  Reparam error -- mean: {error.mean():.8f}, max: {error.max():.8f}")

    # GPU memory
    print(f"  GPU peak memory: {result['gpu_peak_mb']:.1f} MB")

    # Timings
    print(f"\n  Timings:")
    for stage, t in result["timings"].items():
        print(f"    {stage:15s}: {t:.4f}s")
    print(f"{'='*60}")

    # Save plots
    os.makedirs(args.save_dir, exist_ok=True)

    stats_path = os.path.join(args.save_dir, "clustering_stats.png")
    print(f"\nSaving statistics plot to {stats_path} ...")
    plot_statistics(result, save_path=stats_path)

    bev_path = os.path.join(args.save_dir, "bev.png")
    print(f"Saving BEV plot to {bev_path} ...")
    plot_bev(result, save_path=bev_path)

    heatmap_path = os.path.join(args.save_dir, "quality_heatmap.png")
    print(f"Saving quality heatmap to {heatmap_path} ...")
    plot_quality_heatmap(result, save_path=heatmap_path)

    print("Done.")


if __name__ == "__main__":
    main()
