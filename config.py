"""Hyperparameters for the 2D Gaussian clustering pipeline."""

from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    # Stage 0: Preprocessing
    ego_radius: float = 2.5           # ego-vehicle mask radius (meters)
    sor_k: int = 10                   # statistical outlier removal: k neighbors
    sor_std_mul: float = 2.0          # SOR: mean + std_mul * sigma threshold

    # Stage 1: Local geometry
    knn_k: int = 30                   # neighbors for PCA / covariance

    # Stage 2: Seeding
    target_cluster_size: int = 30     # target points per cluster for voxel sizing
    curvature_quantile: float = 0.85  # top 15% curvature -> extra seeds
    fps_extra_ratio: float = 0.1      # extra FPS seeds = ratio * num_voxel_seeds

    # Stage 3: Assignment
    candidate_k: int = 20            # candidate seeds per point
    lambda_normal: float = 0.3       # weight for normal consistency cost
    lambda_curvature: float = 0.1    # weight for curvature consistency cost

    # Stage 4: Refinement
    max_refine_iters: int = 4        # split/merge iterations (even=split, odd=merge)
    split_gamma_rms: float = 0.05    # split if gamma_rms > threshold
    split_max_size: int = 80         # split if cluster size > this
    split_cond_number: float = 20.0  # split if condition number > this
    merge_angle_deg: float = 5.0     # merge if normal angle < this
    merge_max_dist_ratio: float = 2.0  # merge only if centroid dist < ratio * median NN dist
    convergence_threshold: float = 0.01  # stop if < 1% points reassigned

    # Stage 5: Gaussian fitting
    min_cluster_size: int = 5        # discard clusters smaller than this

    # Ground-aware processing
    ground_z_range: tuple = (-3.0, -1.0)  # z range for ground candidate points
    ground_normal_thresh: float = 0.9     # cos angle with up vector for ground
    ground_voxel_scale: float = 3.0       # voxel size multiplier for ground

    # Device
    device: str = "cuda"
