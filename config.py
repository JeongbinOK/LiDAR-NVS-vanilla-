"""Hyperparameters for the 2D Gaussian clustering pipeline."""

from dataclasses import dataclass


@dataclass
class NeuralClusteringConfig:
    """Hyperparameters for the neural clustering pipeline v2."""

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

    # Voxel seeding
    seed_voxel_size: float = 1.0

    # Differentiable clustering
    cluster_iters: int = 4
    cluster_feat_weight: float = 0.0  # 0.1은 backward에서 OOM 유발 (24GB 카드 기준)

    # Cross-attention refinement
    refine_layers: int = 2
    refine_heads: int = 4
    refine_local_topk: int = 64

    # Gaussian head
    primitive_type: str = "3d"     # "2d" or "3d"
    pca_topk: int = 32             # PCA 및 loss 모두 per-Gaussian top-M으로 사용

    # Preprocessing
    ego_radius: float = 2.5

    # Clustering temperature schedule
    cluster_tau_start: float = 1.0
    cluster_tau_end: float = 0.2

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 30
    batch_size: int = 2

    # Loss (top_k_assign은 현재 미사용; loss는 pca_topk를 top_m으로 사용)
    top_k_assign: int = 8

    # Data
    data_root: str = "~/data/nuScenes"

    # Device
    device: str = "cuda"
