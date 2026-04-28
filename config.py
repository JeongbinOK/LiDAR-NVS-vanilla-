"""Hyperparameters for the Quadratic Gaussian Splatting (QGS) pipeline."""

from __future__ import annotations

from dataclasses import dataclass


OFFICIAL_FULL_PTV3_BACKBONE_PARAMS = 46_174_272


@dataclass
class QGSConfig:
    """Hyperparameters for the QGS pipeline."""

    # Backbone
    feature_dim: int = 64
    input_feature_dim: int = 8      # raw QGS feature contract: xyz + intensity + time + e_dir
    ptv3_model_in_channels: int = 8  # PTv3 now consumes the native 8D QGS feature contract directly

    # PTv3 default backbone settings (aligned to outputs/train_044)
    ptv3_grid_size: float = 0.1
    ptv3_stride: tuple = (2, 2, 2)  # full=(2, 2, 2, 2), mid=(2, 2, 2), small=(2, 2)
    ptv3_enc_depths: tuple = (2, 2, 2, 4)  # full=(2, 2, 2, 6, 2), mid=(2, 2, 2, 4), small=(2, 2, 4)
    ptv3_enc_channels: tuple = (32, 64, 128, 256)  # full=(32, 64, 128, 256, 512), mid=(32, 64, 128, 256), small=(32, 64, 128)
    ptv3_enc_num_head: tuple = (2, 4, 8, 16)  # full=(2, 4, 8, 16, 32), mid=(2, 4, 8, 16), small=(2, 4, 8)
    ptv3_enc_patch_size: tuple = (1024, 1024, 1024, 1024)  # full=(1024, 1024, 1024, 1024, 1024), mid=(1024, 1024, 1024, 1024), small=(1024, 1024, 1024)
    ptv3_dec_depths: tuple = (2, 2, 2)  # full=(2, 2, 2, 2), mid=(2, 2, 2), small=(2, 2)
    ptv3_dec_channels: tuple = (64, 64, 128)  # full=(64, 64, 128, 256), mid=(64, 64, 128), small=(64, 128)
    ptv3_dec_num_head: tuple = (4, 4, 8)  # full=(4, 4, 8, 16), mid=(4, 4, 8), small=(4, 8)
    ptv3_dec_patch_size: tuple = (1024, 1024, 1024)  # full=(1024, 1024, 1024, 1024), mid=(1024, 1024, 1024), small=(1024, 1024)
    ptv3_enable_flash: bool = True
    ptv3_conv_algo: str = "native"    # "auto" | "native" | "mask_implicit_gemm" | "mask_split_implicit_gemm"
    ptv3_batch_norm_eval: bool = True
    ptv3_decoupled_stem: bool = True
    ptv3_pdnorm_bn: bool = True
    ptv3_pdnorm_ln: bool = True
    ptv3_pdnorm_decouple: bool = True
    ptv3_condition_names: tuple = ("static", "dynamic")

    # QGS head
    head_hidden_dim: int = 128
    use_gated_head: bool = True       # A2.2 flagship; False = A2.1 ablation
    lidar_latent_dim: int = 16        # must match diff_quadratic_rasterization.LIDAR_LATENT_DIM
    head_alpha_bias_init: float = 2.2     # sigmoid(2.2)≈0.90 — high initial coverage
    head_intensity_residual: bool = True  # intensity = sigmoid(logit + logit(input))
    head_center_bound: float = 0.3        # max analytic-center residual magnitude (m)
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
    quadric_gamma: float = 1.0
    quadric_kappa_max: float = 5.0
    quadric_eps_lambda: float = 0.01
    quadric_eps_kappa: float = 1e-3
    quadric_eps_s: float = 1e-3
    quadric_eps_s3: float = 1e-4

    # LiDAR rasterizer (spherical projection)
    lidar_height: int = 32
    lidar_width: int = 1085  # 20Hz/46.08us -> ~1085 azimuth bins per revolution
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
    loss_w_distortion: float = 0.05   # 2DGS depth distortion regulariser
    loss_w_normal: float = 0.05       # QGS curvature-aware normal consistency

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

    def ptv3_backbone_kwargs(self) -> dict:
        return {
            "grid_size": float(self.ptv3_grid_size),
            "model_in_channels": int(self.ptv3_model_in_channels),
            "stride": tuple(self.ptv3_stride),
            "enc_depths": tuple(self.ptv3_enc_depths),
            "enc_channels": tuple(self.ptv3_enc_channels),
            "enc_num_head": tuple(self.ptv3_enc_num_head),
            "enc_patch_size": tuple(self.ptv3_enc_patch_size),
            "dec_depths": tuple(self.ptv3_dec_depths),
            "dec_channels": tuple(self.ptv3_dec_channels),
            "dec_num_head": tuple(self.ptv3_dec_num_head),
            "dec_patch_size": tuple(self.ptv3_dec_patch_size),
            "enable_flash": bool(self.ptv3_enable_flash),
            "conv_algo": self.ptv3_conv_algo,
            "batch_norm_eval": bool(self.ptv3_batch_norm_eval),
            "decoupled_stem": bool(self.ptv3_decoupled_stem),
            "pdnorm_bn": bool(self.ptv3_pdnorm_bn),
            "pdnorm_ln": bool(self.ptv3_pdnorm_ln),
            "pdnorm_decouple": bool(self.ptv3_pdnorm_decouple),
            "context_conditions": tuple(self.ptv3_condition_names),
            "pdnorm_conditions": tuple(self.ptv3_condition_names),
        }

    def expected_ptv3_backbone_params(self) -> int | None:
        if not (
            self.ptv3_decoupled_stem
            or self.ptv3_pdnorm_bn
            or self.ptv3_pdnorm_ln
        ):
            return OFFICIAL_FULL_PTV3_BACKBONE_PARAMS
        return None
