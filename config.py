"""Hyperparameters for the Quadratic Gaussian Splatting (QGS) pipeline."""

from dataclasses import dataclass


@dataclass
class QGSConfig:
    """Hyperparameters for the QGS pipeline."""

    # Backbone
    backbone_type: str = "ptv3"    # "ptv3" or "custom"
    feature_dim: int = 64
    input_feature_dim: int = 8      # xyz + intensity + time + e_dir

    # PTv3 backbone settings
    ptv3_grid_size: float = 0.1
    ptv3_stride: tuple = (2, 2)
    ptv3_enc_depths: tuple = (2, 2, 2)
    ptv3_enc_channels: tuple = (32, 64, 128)
    ptv3_enc_num_head: tuple = (2, 4, 8)
    ptv3_enc_patch_size: tuple = (1024, 1024, 1024)
    ptv3_dec_depths: tuple = (2, 2)
    ptv3_dec_channels: tuple = (64, 64)
    ptv3_dec_num_head: tuple = (4, 4)
    ptv3_dec_patch_size: tuple = (1024, 1024)
    ptv3_enable_flash: bool = True
    ptv3_conv_algo: str = "native"    # "auto" | "native" | "mask_implicit_gemm" | "mask_split_implicit_gemm"

    # Custom backbone fallback settings
    num_blocks: int = 3
    window_size: int = 48
    num_heads: int = 4

    # QGS head
    head_hidden_dim: int = 128
    use_gated_head: bool = True       # A2.2 flagship; False = A2.1 ablation
    lidar_latent_dim: int = 16        # must match diff_quadratic_rasterization.LIDAR_LATENT_DIM
    head_alpha_bias_init: float = 2.2     # sigmoid(2.2)≈0.90 — high initial coverage
    head_intensity_residual: bool = True  # intensity = sigmoid(logit + logit(input))
    head_center_mode: str = "fixed"       # fixed-center geometry-first head
    head_center_bound_min: float = 0.05   # legacy / unused in fixed-center mode
    head_center_bound_max: float = 0.5    # legacy / unused in fixed-center mode
    rot_tilt_deg: float = 10.0
    rot_spin_deg: float = 30.0
    scale_log_mean_bound: float = 0.6931471805599453   # ln(2)
    scale_log_gap_bound: float = 0.4054651081081644    # ln(1.5)
    s3_log_bound: float = 0.4054651081081644           # ln(1.5)
    s3_fallback_abs: float = 0.05

    # Local quadric init / k-NN
    knn_k_target: int = 16
    knn_k_min: int = 8
    knn_chunk_size: int = 1024

    # LiDAR rasterizer (spherical projection)
    lidar_height: int = 32
    lidar_width: int = 1024
    lidar_el_min_deg: float = -30.67   # nuScenes LIDAR_TOP
    lidar_el_max_deg: float = +10.67
    lidar_sigma: float = 1.5
    r_near: float = 0.2
    r_far: float = 70.0

    # Loss weights
    loss_w_range: float = 1.0
    loss_w_intensity: float = 0.1
    loss_alpha_eps: float = 1e-3
    loss_w_raydrop: float = 0.1

    # Preprocessing
    ego_radius: float = 2.5

    # Training
    lr: float = 5e-4                    # legacy single-lr; ignored when lr_backbone/lr_head set
    lr_backbone: float = 1e-4           # PTv3 + flash-attn (bf16) is sensitive to large lr
    lr_head: float = 5e-4               # head MLPs tolerate higher lr (lower than 1e-3 to dampen alpha-range cycle)
    warmup_iters: int = 500             # linear warmup over this many optimizer steps
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_epochs: int = 30
    batch_size: int = 2

    # Data
    data_root: str = "/data1/nuScenes"
    nuscenes_version: str = "v1.0-trainval"
    train_split: str = "train"
    eval_split: str = "val"
    frame_gap: int = 2                  # 1 → 0.5 s gap, 2 → 1.0 s gap (Phase B target)
    dataset_mode: str = "bbox"           # "nvs" | "bbox" (Phase A=nvs; Phase B switches to bbox)
    bbox_json_path: str = "bbox/tracking_{split}.json"    # {split} resolves to train_split/eval_split; "" = GT annotations
    num_workers: int = 4                # DataLoader worker count

    # Device
    device: str = "cuda"
