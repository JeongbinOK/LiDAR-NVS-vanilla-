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

    # Color by cluster (vectorized)
    K = result["num_gaussians"]
    rng = np.random.RandomState(42)
    palette = rng.rand(K + 1, 3)
    palette[-1] = [0.3, 0.3, 0.3]  # unassigned points

    # Vectorized coloring
    color_idx = np.full(xyz.shape[0], K, dtype=np.int64)  # default to unassigned
    valid = point_mask & (assignments >= 0)
    color_idx[valid] = assignments[valid] % K
    colors = palette[color_idx]

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

    # Normal arrows as line set (vectorized)
    arrow_len = 0.5
    tips = centers + normals_arr * arrow_len
    n_lines = centers.shape[0]
    line_points = np.empty((n_lines * 2, 3))
    line_points[0::2] = centers
    line_points[1::2] = tips
    lines = np.column_stack([np.arange(0, n_lines * 2, 2),
                             np.arange(1, n_lines * 2, 2)])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (n_lines, 1))
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
    axes[0, 1].set_xlabel("gamma_rms (off-plane residual)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("gamma_rms Distribution")
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


def plot_bev(result: dict, save_path: str | None = None):
    """Bird's Eye View: X-Y plane with cluster colors + surfel ellipse overlay.

    Args:
        result: output from cluster_frame()
        save_path: if provided, save figure instead of showing
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Point cloud colored by cluster
    xyz = result["source_xyz"].cpu().numpy()
    assignments = result["point_assignments"].cpu().numpy()
    point_mask = result["point_mask"].cpu().numpy()
    K = result["num_gaussians"]

    rng = np.random.RandomState(42)
    palette = rng.rand(K, 3)

    # Plot points
    valid = point_mask & (assignments >= 0)
    valid_xyz = xyz[valid]
    valid_colors = palette[assignments[valid] % K]
    ax.scatter(valid_xyz[:, 0], valid_xyz[:, 1], c=valid_colors, s=0.3, alpha=0.5)

    # Unassigned points in gray
    invalid = ~valid
    if invalid.any():
        ax.scatter(xyz[invalid, 0], xyz[invalid, 1], c="gray", s=0.1, alpha=0.2)

    # Overlay surfel ellipses
    centers = result["xyz"].cpu().numpy()
    scaling = result["scaling"].cpu().numpy()  # [K', 2]
    tangent_u = result["tangent_u"].cpu().numpy()  # [K', 3]

    for i in range(min(centers.shape[0], 2000)):  # limit for rendering speed
        cx, cy = centers[i, 0], centers[i, 1]
        # Ellipse axes from 2D scaling
        w = scaling[i, 0] * 2  # width = 2 * sigma
        h = scaling[i, 1] * 2
        # Rotation angle from tangent_u projected to XY plane
        angle = np.degrees(np.arctan2(tangent_u[i, 1], tangent_u[i, 0]))
        ell = Ellipse((cx, cy), w, h, angle=angle,
                      facecolor=palette[i % K], edgecolor="black",
                      linewidth=0.3, alpha=0.3)
        ax.add_patch(ell)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"BEV: {K} Gaussians from {xyz.shape[0]} points")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"BEV plot saved to {save_path}")
    else:
        plt.show()


def plot_quality_heatmap(result: dict, save_path: str | None = None):
    """BEV heatmap colored by gamma_rms (red=bad, green=good).

    Args:
        result: output from cluster_frame()
        save_path: if provided, save figure instead of showing
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    xyz = result["source_xyz"].cpu().numpy()
    assignments = result["point_assignments"].cpu().numpy()
    point_mask = result["point_mask"].cpu().numpy()
    gamma_rms = result["_gamma_rms"].cpu().numpy()  # [K']

    valid = point_mask & (assignments >= 0)
    valid_xyz = xyz[valid]
    valid_assignments = assignments[valid]

    # Map each point to its cluster's gamma_rms
    point_gamma = gamma_rms[valid_assignments]

    # Colormap: low gamma_rms = green, high = red
    vmin, vmax = 0.0, np.percentile(point_gamma, 95)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(valid_xyz[:, 0], valid_xyz[:, 1],
                    c=point_gamma, cmap="RdYlGn_r", norm=norm,
                    s=0.5, alpha=0.7)

    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn_r"), ax=ax)
    cbar.set_label("gamma_rms (off-plane residual)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Quality Heatmap (median gamma_rms={np.median(gamma_rms):.5f})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Quality heatmap saved to {save_path}")
    else:
        plt.show()
