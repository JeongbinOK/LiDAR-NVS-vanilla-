"""Hyperparameters for the Quadratic Gaussian Splatting (QGS) pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

_logger = logging.getLogger(__name__)


OFFICIAL_FULL_PTV3_BACKBONE_PARAMS = 46_174_272


def resolve_lidar_latent_dim(lidar_latent_dim: int | None = None) -> int:
    """Resolve/check the latent width against the compiled rasterizer extension.
        CUDA rasterizer 가 빌드한 LIDAR_LATENT_DIM과 config의 lidar_latent_dim이 일치하는지 확인하고, None이면 빌드된 값 사용.
    """
    from diff_quadratic_rasterization import LIDAR_LATENT_DIM

    built_dim = int(LIDAR_LATENT_DIM)
    if lidar_latent_dim is None:
        return built_dim
    requested = int(lidar_latent_dim)
    if requested != built_dim:
        raise ValueError(
            f"lidar_latent_dim={requested} must equal "
            f"diff_quadratic_rasterization.LIDAR_LATENT_DIM={built_dim}. "
            "Rebuild the rasterizer with LIDAR_LATENT_DIM=<value> or remove "
            "the explicit override."
        )
    return requested


@dataclass
class QGSConfig:
    """Hyperparameters for the QGS pipeline."""
    #######################
    # Backbone
    feature_dim: int = 64
    primitive_mode: str = "voxel_anchor"  # voxel-anchor primitives only
    ptv3_model_in_channels: int | None = None  # None resolves to mode feature width; kept for legacy configs

    # PTv3 default backbone settings (aligned to outputs/train_044)
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
    lidar_latent_dim: int | None = None  # None resolves to compiled rasterizer LIDAR_LATENT_DIM
    head_alpha_bias_init: float = 0.0     # sigmoid(0.0)=0.50 — neutral initial coverage; 2.2 caused T overflow
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
    quadric_gamma: float = 1.73
    quadric_kappa_max: float = 5.0
    quadric_eps_lambda: float = 0.01
    quadric_eps_kappa: float = 1e-3
    quadric_eps_s: float = 1e-3
    quadric_eps_s3: float = 1e-4

    # Voxel-anchor primitive generation
    anchor_token_variant: str = "full"        # full|no_fit_quality|no_intensity_stats|with_normal
    anchor_filter_mode: str = "residual"      # residual|residual_planarity
    anchor_residual_threshold: float = 0.806  # tau_r for anchor residual filtering
    anchor_planarity_threshold: float = 0.2   # only used by residual_planarity filtering

    # LiDAR rasterizer (spherical projection)
    lidar_height: int = 32
    lidar_width: int = 1085  # 20Hz/46.08us -> ~1085 azimuth bins per revolution
    # Per-ring elevation table (deg). nuScenes LIDAR_TOP exposes the `ring`
    # channel **already sorted by elevation** (ascending, v=0=bottom): ring k
    # corresponds to the k-th lowest beam. This is NOT the HDL-32E manufacturer
    # firing order; it's the post-sort index nuScenes ships with each scan.
    # Verified empirically (scripts/diag_ring_elevation.py).
    #
    # row_to_elevation_deg, ring_to_row, ring_at_row are all derived in
    # nn.lidar_geometry from this single source of truth. Because the table is
    # already sorted, ring_to_row is the identity mapping.
    ring_to_elevation_deg: tuple = (
        -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33,
        -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
         -9.33,  -8.00,  -6.66,  -5.33,  -4.00,  -2.67,  -1.33,   0.00,
          1.33,   2.67,   4.00,   5.33,   6.67,   8.00,   9.33,  10.67,
    )

    # Legacy linear-FOV fields (deprecated; kept as tombstones so older
    # checkpoints' config.json reload without raising). The runtime path never
    # reads these — see __post_init__.
    lidar_el_min_deg: Optional[float] = None
    lidar_el_max_deg: Optional[float] = None
    lidar_sigma: float = 3.0
    r_near: float = 0.2
    r_far: float = 70.0

    # Loss weights
    loss_w_range: float = 1.0
    loss_w_intensity: float = 0.1
    loss_alpha_eps: float = 0.5
    loss_w_raydrop: float = 0.1
    loss_w_distortion: float = 0.05   # 2DGS depth distortion regulariser
    loss_w_normal: float = 0.05       # QGS curvature-aware normal consistency

    #Preprocessing
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

    # Debug
    debug_finite_check: bool = False  # log non-finite forward boundaries in process_pair
    debug_anomaly_batch: int = -1     # run set_detect_anomaly on this batch_idx (-1 = off)
    debug_bad_grad_trace: bool = True  # log first non-finite primitive/render grads when Bad grad happens
    debug_bad_grad_trace_max: int = 16  # max tensor-gradient records printed per bad batch

    def __post_init__(self) -> None:
        self.lidar_latent_dim = resolve_lidar_latent_dim(self.lidar_latent_dim)
        if len(self.ring_to_elevation_deg) != self.lidar_height:
            raise ValueError(
                f"len(ring_to_elevation_deg)={len(self.ring_to_elevation_deg)} must "
                f"equal lidar_height={self.lidar_height}"
            )
        if self.lidar_el_min_deg is not None or self.lidar_el_max_deg is not None:
            _logger.warning(
                "config.lidar_el_min_deg / lidar_el_max_deg are legacy tombstones; "
                "ignored. Per-row elevations are derived from ring_to_elevation_deg."
            )
            # Discard so downstream code never accidentally uses them.
            self.lidar_el_min_deg = None
            self.lidar_el_max_deg = None
        allowed_modes = {"voxel_anchor"}
        if self.primitive_mode not in allowed_modes:
            raise ValueError(
                f"primitive_mode must be one of {sorted(allowed_modes)}; "
                f"got {self.primitive_mode!r}"
            )
        allowed_token_variants = {
            "full",
            "no_fit_quality",
            "no_intensity_stats",
            "with_normal",
        }
        if self.anchor_token_variant not in allowed_token_variants:
            raise ValueError(
                "anchor_token_variant must be one of "
                f"{sorted(allowed_token_variants)}; got {self.anchor_token_variant!r}"
            )
        allowed_filter_modes = {"residual", "residual_planarity"}
        if self.anchor_filter_mode not in allowed_filter_modes:
            raise ValueError(
                f"anchor_filter_mode must be one of {sorted(allowed_filter_modes)}; "
                f"got {self.anchor_filter_mode!r}"
            )
        if self.anchor_token_variant == "with_normal":
            self.anchor_token_dim = 25
        elif self.anchor_token_dim != 22:
            raise ValueError(
                "anchor_token_dim must remain 22 except for "
                "anchor_token_variant='with_normal'"
            )
        if self.ptv3_model_in_channels is None:
            self.ptv3_model_in_channels = self.resolved_input_channels()

    def resolved_input_channels(self) -> int:
        return int(self.anchor_token_dim)

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

    #######################




    # Backbone
    
    anchor_token_dim: int = 22      # voxel-anchor token width consumed by PTv3 in voxel_anchor mode
    ptv3_model_in_channels: int | None = None  # None resolves to mode feature width; kept for legacy configs

    # PTv3 default backbone settings (aligned to outputs/train_044)
    ptv3_grid_size: float = 0.1
    ptv3_stride: tuple = (2, 2)  # full=(2, 2, 2, 2), mid=(2, 2, 2), small=(2, 2)
    ptv3_enc_depths: tuple = (2, 2, 4)  # full=(2, 2, 2, 6, 2), mid=(2, 2, 2, 4), small=(2, 2, 4)
    ptv3_enc_channels: tuple = (32, 64, 128)  # full=(32, 64, 128, 256, 512), mid=(32, 64, 128, 256), small=(32, 64, 128)
    ptv3_enc_num_head: tuple = (2, 4, 8)  # full=(2, 4, 8, 16, 32), mid=(2, 4, 8, 16), small=(2, 4, 8)
    ptv3_enc_patch_size: tuple = (1024, 1024, 1024)  # full=(1024, 1024, 1024, 1024, 1024), mid=(1024, 1024, 1024, 1024), small=(1024, 1024, 1024)
    ptv3_dec_depths: tuple = (2, 2)  # full=(2, 2, 2, 2), mid=(2, 2, 2), small=(2, 2)
    ptv3_dec_channels: tuple = (64, 128)  # full=(64, 64, 128, 256), mid=(64, 64, 128), small=(64, 128)
    ptv3_dec_num_head: tuple = (4, 8)  # full=(4, 4, 8, 16), mid=(4, 4, 8), small=(4, 8)
    ptv3_dec_patch_size: tuple = (1024, 1024)  # full=(1024, 1024, 1024, 1024), mid=(1024, 1024, 1024), small=(1024, 1024)
    

    # QGS head
    

    # Local quadric init / k-NN
 

    # Voxel-anchor primitive generation
    
    # LiDAR rasterizer (spherical projection)
    
    # Loss weights

   
    # Preprocessing
    
    # Training

    
    # Data
    

    # Device
    

    # Debug