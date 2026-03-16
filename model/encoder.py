"""
Range-view CNN encoder (ResNet-18 style + FPN) with sinusoidal time embedding.

Key design: uses stride=(1,2) for downsampling layers to preserve vertical
resolution (H=32 is only 32 laser beams — can't afford vertical downsampling).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet BasicBlock that supports tuple strides for asymmetric downsampling."""

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class FPN(nn.Module):
    """Feature Pyramid Network that merges multi-scale features to a single
    resolution with output dim ``out_dim``."""

    def __init__(self, in_channels_list, out_dim):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_dim, 1))
            self.output_convs.append(nn.Conv2d(out_dim, out_dim, 3, padding=1))

    def forward(self, features):
        # features: list [c1, c2, c3, c4] from fine to coarse
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]

        # Top-down pathway (coarse → fine)
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode="bilinear",
                align_corners=False)

        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Upsample all to finest resolution and sum
        target_size = outs[0].shape[2:]
        merged = sum(
            F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
            if o.shape[2:] != target_size else o
            for o in outs
        )
        return merged


class RangeViewEncoder(nn.Module):
    """ResNet-18 style encoder for 32×1080 range images.

    Uses stride=(1,2) in downsampling layers so that vertical resolution (H=32)
    is fully preserved. Only the width is reduced:
        layer1: [B, 64,  32, 1080]
        layer2: [B, 128, 32, 540]
        layer3: [B, 256, 32, 270]
        layer4: [B, 512, 32, 135]
    FPN merges everything back to [B, feat_dim, 32, 1080].
    """

    def __init__(self, in_channels=5, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim

        # Stem (no downsampling)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers — stride=(1,2): preserve height, halve width
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1, 2))
        self.layer3 = self._make_layer(128, 256, 2, stride=(1, 2))
        self.layer4 = self._make_layer(256, 512, 2, stride=(1, 2))

        self.fpn = FPN([64, 128, 256, 512], feat_dim)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, 5, H, W] range image.
        Returns:
            feat: [B, feat_dim, H, W].
        """
        x = self.relu(self.bn1(self.conv1(x)))

        c1 = self.layer1(x)   # [B, 64,  32, 1080]
        c2 = self.layer2(c1)  # [B, 128, 32, 540]
        c3 = self.layer3(c2)  # [B, 256, 32, 270]
        c4 = self.layer4(c3)  # [B, 512, 32, 135]

        feat = self.fpn([c1, c2, c3, c4])  # [B, feat_dim, 32, 1080]
        return feat


class TimeEmbedding(nn.Module):
    """Inject a scalar timestamp into spatial features via concat + 1x1 conv."""

    def __init__(self, feat_dim=128, time_embed_dim=32, max_freq=10):
        super().__init__()
        self.num_freqs = max_freq + 1
        raw_dim = 2 * self.num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, time_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.proj = nn.Conv2d(feat_dim + time_embed_dim, feat_dim, 1)

    def forward(self, t, feat):
        """
        Args:
            t: [B] scalar timestamps.
            feat: [B, C, H, W] encoder output.
        Returns:
            feat_t: [B, C, H, W] time-conditioned features.
        """
        B, C, H, W = feat.shape
        freqs = 2.0 ** torch.arange(self.num_freqs, device=t.device, dtype=t.dtype)
        angles = t[:, None] * freqs[None, :] * math.pi
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        time_emb = self.mlp(enc)

        time_emb = time_emb[:, :, None, None].expand(-1, -1, H, W)
        feat_t = torch.cat([feat, time_emb], dim=1)
        feat_t = self.proj(feat_t)
        return feat_t
