"""Visualization utilities for clustering results."""

import numpy as np
import torch


def visualize_clusters_open3d(result: dict, max_points: int = 50000):
    """Visualize point cloud colored by cluster + Gaussian centers with normals.

    Args:
        result: output from cluster_frame()
        max_points: subsample if more points
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed, skipping 3D visualization")
        return

    xyz = result["source_xyz"].cpu().numpy()
    assignments = result["point_assignments"].cpu().numpy()
    point_mask = result["point_mask"].cpu().numpy()

    # Color by cluster (random color per cluster)
    K = result["num_gaussians"]
    rng = np.random.RandomState(42)
    palette = rng.rand(K + 1, 3)
    palette[-1] = [0.3, 0.3, 0.3]  # unassigned points

    colors = np.zeros((xyz.shape[0], 3))
    for i in range(xyz.shape[0]):
        if point_mask[i] and assignments[i] >= 0:
            colors[i] = palette[assignments[i] % K]
        else:
            colors[i] = palette[-1]

    # Subsample for visualization
    if xyz.shape[0] > max_points:
        idx = rng.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
        colors = colors[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Gaussian centers
    centers = result["xyz"].cpu().numpy()
    normals_arr = result["normal"].cpu().numpy()

    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector(centers)
    center_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (centers.shape[0], 1))
    )

    # Normal arrows as line set
    lines = []
    line_points = []
    arrow_len = 0.5
    for i in range(centers.shape[0]):
        base = centers[i]
        tip = base + normals_arr[i] * arrow_len
        idx_base = len(line_points)
        line_points.append(base)
        line_points.append(tip)
        lines.append([idx_base, idx_base + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (len(lines), 1))
    )

    o3d.visualization.draw_geometries(
        [pcd, center_pcd, line_set],
        window_name="Gaustering: 2D Gaussian Clusters",
        width=1280,
        height=720,
    )


def plot_statistics(result: dict, save_path: str | None = None):
    """Plot clustering statistics with matplotlib.

    Args:
        result: output from cluster_frame()
        save_path: if provided, save figure instead of showing
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Cluster size distribution
    counts = result["_counts"].cpu().numpy()
    axes[0, 0].hist(counts, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Points per cluster")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Cluster Size Distribution (n={len(counts)})")
    axes[0, 0].axvline(counts.mean(), color="r", linestyle="--",
                        label=f"mean={counts.mean():.1f}")
    axes[0, 0].legend()

    # 2. Gamma RMS distribution
    gamma_rms = result["_gamma_rms"].cpu().numpy()
    axes[0, 1].hist(gamma_rms, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("γ_rms (off-plane residual)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("γ_rms Distribution")
    axes[0, 1].axvline(np.median(gamma_rms), color="r", linestyle="--",
                        label=f"median={np.median(gamma_rms):.4f}")
    axes[0, 1].legend()

    # 3. Reparameterization error
    xyz = result["source_xyz"]
    assignments = result["point_assignments"]
    mask = result["point_mask"]
    alphas = result["alphas"][mask]
    betas = result["betas"][mask]
    gammas_pt = result["gammas"][mask]
    a = assignments[mask]

    mu = result["xyz"][a]
    tu = result["tangent_u"][a]
    tv = result["tangent_v"][a]
    tn = result["normal"][a]

    reconstructed = mu + alphas.unsqueeze(1) * tu + betas.unsqueeze(1) * tv + gammas_pt.unsqueeze(1) * tn
    error = (xyz[mask] - reconstructed).pow(2).sum(dim=1).sqrt().cpu().numpy()

    axes[1, 0].hist(error, bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[1, 0].set_xlabel("Reconstruction error (meters)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Reparameterization Error (max={error.max():.6f})")

    # 4. Timing breakdown
    timings = result["timings"]
    stages = [k for k in timings if k != "total"]
    times = [timings[k] for k in stages]
    axes[1, 1].barh(stages, times, color="steelblue", edgecolor="black")
    axes[1, 1].set_xlabel("Time (seconds)")
    axes[1, 1].set_title(f"Pipeline Timing (total={timings['total']:.3f}s)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
