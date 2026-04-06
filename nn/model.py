"""Neural Gaussian clustering model v2.

Pipeline: PTv3 Backbone → Voxel Seeding → Diff Clustering → Cross-Attn Refine → Gaussian Head
"""

import torch
import torch.nn as nn

from nn.ptv3.wrapper import PTv3Backbone
from nn.backbone import PointFeatureBackbone
from nn.vote import VoxelCenterPredictor
from nn.diff_cluster import DiffSoftClustering
from nn.refine import CrossAttentionRefiner
from nn.gaussian_head import GaussianParameterHead


class NeuralClusteringModel(nn.Module):
    """Neural Gaussian clustering in a single forward pass.

    Pipeline:
        1. Backbone: xyz + intensity → per-point features [N, D]
        2. Vote + Voxel: features → offset → voxel pooling → seeds [V, 3]
        3. Cluster: iterative soft k-means → centers, assignments [N, K]
        4. Refine: cross-attention → refined center features [K, D]
        5. Head: PCA + residual → Gaussian parameters
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.feature_dim

        # Stage 1: Backbone
        if getattr(cfg, 'backbone_type', 'custom') == 'ptv3':
            self.backbone = PTv3Backbone(
                in_channels=4,
                out_channels=D,
                grid_size=getattr(cfg, 'ptv3_grid_size', 0.1),
                stride=getattr(cfg, 'ptv3_stride', (2, 2)),
                enc_depths=getattr(cfg, 'ptv3_enc_depths', (2, 2, 2)),
                enc_channels=getattr(cfg, 'ptv3_enc_channels', (32, 64, 128)),
                enc_num_head=getattr(cfg, 'ptv3_enc_num_head', (2, 4, 8)),
                enc_patch_size=getattr(cfg, 'ptv3_enc_patch_size', (1024, 1024, 1024)),
                dec_depths=getattr(cfg, 'ptv3_dec_depths', (2, 2)),
                dec_channels=getattr(cfg, 'ptv3_dec_channels', (64, 64)),
                dec_num_head=getattr(cfg, 'ptv3_dec_num_head', (4, 4)),
                dec_patch_size=getattr(cfg, 'ptv3_dec_patch_size', (1024, 1024)),
                enable_flash=getattr(cfg, 'ptv3_enable_flash', True),
            )
        else:
            self.backbone = PointFeatureBackbone(
                dim=D, num_blocks=cfg.num_blocks,
                window_size=cfg.window_size, num_heads=cfg.num_heads,
            )

        # Stage 2: Voxel-based Center Prediction
        self.voter = VoxelCenterPredictor(
            dim=D,
            voxel_size=getattr(cfg, 'seed_voxel_size', 0.3),
            max_K=getattr(cfg, 'max_seed_K', 8000),
        )

        # Stage 3: Differentiable Soft Clustering
        self.clusterer = DiffSoftClustering(
            num_iters=getattr(cfg, 'cluster_iters', 4),
            feature_weight=getattr(cfg, 'cluster_feat_weight', 0.1),
        )

        # Stage 3.5: Cross-Attention Refinement
        self.refiner = CrossAttentionRefiner(
            dim=D,
            num_layers=getattr(cfg, 'refine_layers', 2),
            num_heads=getattr(cfg, 'refine_heads', 4),
            local_topk=getattr(cfg, 'refine_local_topk', 64),
        )

        # Stage 4: Gaussian Head
        self.gaussian_head = GaussianParameterHead(
            dim=D,
            primitive=getattr(cfg, 'primitive_type', '2d'),
            pca_topk=getattr(cfg, 'pca_topk', 128),
        )

    def forward(self, xyz: torch.Tensor, intensity: torch.Tensor,
                tau: float = 1.0) -> dict:
        """
        Args:
            xyz: [N, 3] point positions (ego-masked)
            intensity: [N] or [N, 1] per-point intensity
            tau: temperature for soft clustering

        Returns:
            dict with gaussians, assignments, voting info
        """
        # Stage 1: Backbone
        features = self.backbone(xyz, intensity)  # [N, D]

        # Stage 2: Voxel Seeding
        vote_out = self.voter(features, xyz)
        vote_xyz = vote_out["vote_xyz"]              # [N, 3]
        voxel_centers = vote_out["voxel_centers"]    # [V, 3]
        voxel_feats = vote_out["voxel_feats"]        # [V, D]

        # Stage 3: Differentiable Soft Clustering
        cluster_out = self.clusterer(
            xyz, features, voxel_centers, voxel_feats, tau,
        )
        centers = cluster_out["centers"]          # [K, 3]
        center_feats = cluster_out["center_feats"]  # [K, D]
        assign = cluster_out["assign"]            # [N, K]

        # Stage 3.5: Cross-Attention Refinement
        center_feats = self.refiner(center_feats, features, assign)

        # Stage 4: Gaussian Parameters
        gaussians = self.gaussian_head(
            center_feats, centers, assign, xyz,
        )

        return {
            "gaussians": gaussians,
            "assign": assign,             # [N, K] dense
            "centers": centers,           # [K, 3]
            "vote_xyz": vote_xyz,         # [N, 3]
            "offset": vote_out["offset"],          # [N, 3]
            "voxel_ids": vote_out["voxel_ids"],    # [N]
        }
