"""Wrapper around PTv3 for our clustering pipeline.

Converts our (xyz, intensity) inputs to PTv3's expected dict format
and extracts per-point features from the output.
"""

import torch
import torch.nn as nn

from .model import PointTransformerV3


class PTv3Backbone(nn.Module):
    """PTv3 encoder-decoder as a drop-in backbone replacement.

    Input:  xyz [N, 3], intensity [N] or [N, 1]
    Output: features [N, D]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        grid_size: float = 0.1,
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        enable_flash: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_channels = out_channels

        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            cls_mode=False,
            enable_flash=enable_flash,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )

        # Project decoder output to desired dimension if needed
        dec_out_dim = dec_channels[0] if dec_channels else enc_channels[0]
        if dec_out_dim != out_channels:
            self.proj = nn.Linear(dec_out_dim, out_channels)
        else:
            self.proj = nn.Identity()

    def forward(self, xyz: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [N, 3] point positions
            intensity: [N] or [N, 1] per-point intensity

        Returns:
            features: [N, out_channels] per-point features
        """
        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(1)

        N = xyz.shape[0]
        device = xyz.device

        # Build PTv3 input dict
        feat = torch.cat([xyz, intensity], dim=1)  # [N, 4]
        data_dict = dict(
            coord=xyz.float(),
            feat=feat.float(),
            grid_size=self.grid_size,
            offset=torch.tensor([N], dtype=torch.long, device=device),
        )

        # Forward through PTv3
        point = self.ptv3(data_dict)

        # Extract per-point features from decoder output
        features = point.feat  # [N', D] (N' may differ due to grid sampling)

        # Project to desired dimension
        features = self.proj(features)

        return features
