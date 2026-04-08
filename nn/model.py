"""Quadratic Gaussian Splatting (QGS) Model.

Pipeline: PTv3 Backbone → QGS Prediction
"""

import torch
import torch.nn as nn

from nn.ptv3.wrapper import PTv3Backbone
from nn.backbone import PointFeatureBackbone


class QGSModel(nn.Module):
    """Quadratic Gaussian Splatting in a single forward pass.

    Pipeline:
        1. Backbone: xyz + intensity → per-point features [N, D]
        2. QGS: features → quadratic gaussian primitives
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

        # TODO: Add QGS Network specific modules here
        # self.qgs_head = QGSHead(...)

    def forward(self, xyz: torch.Tensor, intensity: torch.Tensor, **kwargs) -> dict:
        """
        Args:
            xyz: [N, 3] point positions (ego-masked)
            intensity: [N] or [N, 1] per-point intensity

        Returns:
            dict with qgs predictions
        """
        # Stage 1: Backbone
        features = self.backbone(xyz, intensity)  # [N, D]

        # TODO: Implement QGS logic
        
        return {
            "features": features,
            "xyz": xyz,
            # "qgs_params": ...
        }

