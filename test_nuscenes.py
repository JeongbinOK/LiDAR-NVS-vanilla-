"""Test script: run pipeline on a single nuScenes frame and report results."""

import sys
import torch

from gaustering.config import ClusteringConfig
from gaustering.data_loader import list_lidar_files
from gaustering.pipeline import cluster_frame
from gaustering.visualization import plot_statistics


def main():
    nuscenes_root = "/data1/nuScenes"
    files = list_lidar_files(nuscenes_root)
    if not files:
        print(f"No .pcd.bin files found under {nuscenes_root}/samples/LIDAR_TOP/")
        sys.exit(1)

    path = files[0]
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

    counts = result["_counts"]
    print(f"\n  Cluster size — mean: {counts.float().mean():.1f}, "
          f"median: {counts.float().median():.1f}, "
          f"min: {counts.min().item()}, max: {counts.max().item()}")

    gamma_rms = result["_gamma_rms"]
    print(f"  γ_rms — mean: {gamma_rms.mean():.5f}, "
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
    print(f"  Reparam error — mean: {error.mean():.8f}, max: {error.max():.8f}")

    # Timings
    print(f"\n  Timings:")
    for stage, t in result["timings"].items():
        print(f"    {stage:15s}: {t:.4f}s")
    print(f"{'='*60}")

    # Save statistics plot
    save_path = "clustering_stats.png"
    print(f"\nSaving statistics plot to {save_path} ...")
    plot_statistics(result, save_path=save_path)

    print("Done.")


if __name__ == "__main__":
    main()
