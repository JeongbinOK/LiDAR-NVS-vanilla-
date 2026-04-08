"""Hyperparameters for the Quadratic Gaussian Splatting (QGS) pipeline."""

from dataclasses import dataclass


@dataclass
class QGSConfig:
    """Hyperparameters for the QGS pipeline."""

    # Backbone
    backbone_type: str = "ptv3"    # "ptv3" or "custom"
    feature_dim: int = 64

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

    # Custom backbone fallback settings
    num_blocks: int = 3
    window_size: int = 48
    num_heads: int = 4

    # TODO: Add specific QGS representation parameters here
    # Example:
    # qgs_max_primitives: int = 2000
    # qgs_feature_dim: int = 32

    # Preprocessing
    ego_radius: float = 2.5

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 30
    batch_size: int = 2

    # Data
    data_root: str = "/data1/nuScenes"
    loader_mode: str = "nvs"  # "nvs" | "bbox"
    bbox_json_path: str = ""  # path to bbox/tracking.json; empty = use GT annotations

    # Device
    device: str = "cuda"

