"""Neural Gaussian clustering model v2.

Pipeline: PTv3 Backbone → Offset Voting → Diff Clustering → Cross-Attn Refine → Gaussian Head
"""

import torch
import torch.nn as nn

from nn.ptv3.wrapper import PTv3Backbone
from nn.backbone import PointFeatureBackbone
from nn.vote import LearnedCenterPredictor
from nn.diff_cluster import DiffSoftClustering
from nn.refine import CrossAttentionRefiner
from nn.gaussian_head import GaussianParameterHead


class NeuralClusteringModel(nn.Module):
    """Neural Gaussian clustering in a single forward pass.

    Pipeline:
        1. Backbone: xyz + intensity → per-point features [N, D]
        2. Vote: features → offset, centerness → seed selection
        3. Cluster: iterative soft k-means → centers, assignments
        4. Refine: cross-attention → refined center features
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

        # Stage 2: Learned Center Prediction
        self.voter = LearnedCenterPredictor(
            dim=D,
            target_cluster_size=cfg.target_cluster_size,
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
            tau: temperature for Gumbel selection and soft clustering

        Returns:
            dict with gaussians, assignments, voting info
        """
        # Stage 1: Backbone
        features = self.backbone(xyz, intensity)  # [N, D]

        # Stage 2: Offset Voting
        vote_out = self.voter(features, xyz, tau)
        vote_xyz = vote_out["vote_xyz"]     # [N, 3]
        seed_idx = vote_out["seed_idx"]     # [K]

        # Stage 3: Differentiable Soft Clustering
        cluster_out = self.clusterer(vote_xyz, features, seed_idx, tau)
        centers = cluster_out["centers"]          # [K, 3]
        center_feats = cluster_out["center_feats"]  # [K, D]
        assign = cluster_out["assign"]            # [N, K]

        # Stage 3.5: Cross-Attention Refinement
        center_feats = self.refiner(center_feats, features, assign)

        # Stage 4: Gaussian Parameters
        gaussians = self.gaussian_head(
            center_feats, centers, assign, vote_xyz,
        )

        return {
            "gaussians": gaussians,
            "assign": assign,             # [N, K] dense
            "centers": centers,           # [K, 3]
            "vote_xyz": vote_xyz,         # [N, 3]
            "centerness": vote_out["centerness"],  # [N]
            "offset": vote_out["offset"],          # [N, 3]
            "seed_idx": seed_idx,                  # [K]
        }
