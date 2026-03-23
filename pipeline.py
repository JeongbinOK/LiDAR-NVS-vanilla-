"""Full clustering pipeline: single entry point."""

import time

import torch

from .assignment import assign_points
from .config import ClusteringConfig
from .data_loader import load_pcd_bin
from .gaussian_fit import fit_2d_gaussians
from .geometry import estimate_local_geometry, preprocess
from .ground import ground_aware_seeds
from .refinement import refine_clusters
from .seeding import compute_seeds


def cluster_frame(
    path: str,
    cfg: ClusteringConfig | None = None,
) -> dict:
    """Run the full clustering pipeline on a single LiDAR frame.

    Args:
        path: path to .pcd.bin file
        cfg: optional config (uses defaults if None)

    Returns:
        dict with Gaussian parameters + metadata
    """
    if cfg is None:
        cfg = ClusteringConfig()

    timings = {}
    t0 = time.time()

    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load data
    data = load_pcd_bin(path, device=cfg.device)
    xyz_raw = data["xyz"]
    intensity_raw = data["intensity"]
    timings["load"] = time.time() - t0

    # Stage 0: Preprocessing
    t = time.time()
    mask = preprocess(xyz_raw, cfg)
    xyz = xyz_raw[mask]
    intensity = intensity_raw[mask]
    timings["preprocess"] = time.time() - t

    # Stage 1: Local geometry
    t = time.time()
    geo = estimate_local_geometry(xyz, cfg)
    timings["geometry"] = time.time() - t

    # Stage 2: Ground-aware seeding
    t = time.time()
    seed_indices, ground_mask = ground_aware_seeds(
        xyz, geo["curvature"], geo["normals"], cfg
    )
    timings["seeding"] = time.time() - t

    # Stage 3: Assignment (with curvature consistency)
    t = time.time()
    assignments = assign_points(
        xyz, geo["normals"], seed_indices, cfg,
        curvature=geo["curvature"],
    )
    timings["assignment"] = time.time() - t

    # Stage 4: Refinement (split/merge)
    t = time.time()
    assignments = refine_clusters(xyz, geo["normals"], assignments, cfg)
    timings["refinement"] = time.time() - t

    # Stage 5: 2D Gaussian fitting
    t = time.time()
    result = fit_2d_gaussians(xyz, geo["normals"], assignments, intensity, cfg)
    timings["fitting"] = time.time() - t

    timings["total"] = time.time() - t0

    # GPU memory tracking
    gpu_peak_mb = 0.0
    if torch.cuda.is_available():
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Add metadata
    result["timings"] = timings
    result["num_raw_points"] = xyz_raw.shape[0]
    result["num_filtered_points"] = xyz.shape[0]
    result["num_gaussians"] = result["xyz"].shape[0]
    result["source_xyz"] = xyz
    result["source_normals"] = geo["normals"]
    result["ground_mask"] = ground_mask
    result["gpu_peak_mb"] = gpu_peak_mb

    return result
