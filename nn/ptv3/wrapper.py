"""Wrapper around PTv3 for our clustering pipeline.

Converts our (xyz, intensity) inputs to PTv3's expected dict format
and extracts per-point features from the output.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from contextlib import contextmanager

from .model import PointTransformerV3


VALID_CONTEXT_TYPES = ("static", "dynamic")


def validate_context_type(context_type: str) -> str:
    normalized = str(context_type).strip().lower()
    if normalized not in VALID_CONTEXT_TYPES:
        raise ValueError(
            f"context_type must be one of {VALID_CONTEXT_TYPES}; got {context_type!r}"
        )
    return normalized


def _resolve_conv_algo(conv_algo: str | None):
    if conv_algo is None:
        return None
    name = str(conv_algo).strip().lower()
    if name in {"", "auto", "none"}:
        return None
    mapping = {
        "native": spconv.ConvAlgo.Native,
        "mask_implicit_gemm": spconv.ConvAlgo.MaskImplicitGemm,
        "maskimplicitgemm": spconv.ConvAlgo.MaskImplicitGemm,
        "mask_split_implicit_gemm": spconv.ConvAlgo.MaskSplitImplicitGemm,
        "masksplitimplicitgemm": spconv.ConvAlgo.MaskSplitImplicitGemm,
    }
    if name not in mapping:
        raise ValueError(
            f"unsupported PTv3 conv algo {conv_algo!r}; "
            "expected auto|native|mask_implicit_gemm|mask_split_implicit_gemm"
        )
    return mapping[name]


class PTv3Backbone(nn.Module):
    """PTv3 encoder-decoder as a drop-in backbone replacement.

    Input:  xyz [N, 3], point_features [N, C]
    Output: features [N, D]
    """

    supports_context_type = True

    def __init__(
        self,
        in_channels: int = 4,
        model_in_channels: int | None = None,
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
        conv_algo: str | None = None,
        batch_norm_eval: bool = False,
        decoupled_stem: bool = True,
        pdnorm_bn: bool = True,
        pdnorm_ln: bool = True,
        pdnorm_decouple: bool = True,
        context_conditions=VALID_CONTEXT_TYPES,
        pdnorm_conditions=VALID_CONTEXT_TYPES,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_channels = out_channels
        self.in_channels = int(in_channels)
        self.model_in_channels = int(model_in_channels if model_in_channels is not None else in_channels)
        if self.model_in_channels < self.in_channels:
            raise ValueError(
                f"model_in_channels must be >= in_channels; got "
                f"{self.model_in_channels} < {self.in_channels}"
            )
        self.conv_algo = conv_algo
        self.conv_algo_resolved = _resolve_conv_algo(conv_algo)
        self.batch_norm_eval = bool(batch_norm_eval)
        self.context_conditions = tuple(context_conditions)

        self.ptv3 = PointTransformerV3(
            in_channels=self.model_in_channels,
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
            decoupled_stem=decoupled_stem,
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_conditions=pdnorm_conditions,
            conv_algo=self.conv_algo_resolved,
        )

        # Project decoder output to desired dimension if needed
        dec_out_dim = dec_channels[0] if dec_channels else enc_channels[0]
        if dec_out_dim != out_channels:
            self.proj = nn.Linear(dec_out_dim, out_channels)
        else:
            self.proj = nn.Identity()

    @contextmanager
    def _temporary_batch_norm_eval(self):
        if not self.training or not self.batch_norm_eval:
            yield
            return

        bn_modules = [m for m in self.ptv3.modules() if isinstance(m, nn.BatchNorm1d)]
        states = [m.training for m in bn_modules]
        for module in bn_modules:
            module.eval()
        try:
            yield
        finally:
            for module, was_training in zip(bn_modules, states):
                module.train(was_training)

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: torch.Tensor,
        *,
        context_type: str | None = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz: [N, 3] point positions
            point_features: [N, C] raw per-point feature matrix. The feature
                dimension must match the external QGS contract `in_channels`.
            context_type: optional QGS context label. The official full PTv3
                backbone uses this to select the static/dynamic stem and norm
                branch. When omitted, the wrapper falls back to the first
                configured condition. `model_in_channels > in_channels` is kept
                only for legacy checkpoint compatibility.

        Returns:
            features: [N, out_channels] per-point features
        """
        if context_type is None:
            context_type = self.context_conditions[0]
        context_type = validate_context_type(context_type)
        if point_features.dim() == 1:
            point_features = point_features.unsqueeze(1)
        if point_features.dim() != 2:
            raise ValueError(
                f"point_features must be [N, C]; got {tuple(point_features.shape)}"
            )
        if point_features.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} feature channels, "
                f"got {point_features.shape[1]}"
            )

        N = xyz.shape[0]
        device = xyz.device

        # Build PTv3 input dict
        feat = point_features.float()
        if self.model_in_channels != self.in_channels:
            feat = torch.cat(
                [
                    feat,
                    feat.new_zeros(N, self.model_in_channels - self.in_channels),
                ],
                dim=1,
            )
        data_dict = dict(
            coord=xyz.float(),
            feat=feat,
            grid_size=self.grid_size,
            offset=torch.tensor([N], dtype=torch.long, device=device),
            condition=context_type,
        )

        # Forward through PTv3
        with self._temporary_batch_norm_eval():
            point = self.ptv3(data_dict)

        # Extract per-point features from decoder output
        features = point.feat  # [N', D] (N' may differ due to grid sampling)

        # Project to desired dimension
        features = self.proj(features)
        if features.shape[0] != N:
            raise RuntimeError(
                f"PTv3 backbone changed point count: {N} -> {features.shape[0]} — "
                "QGS expects dense per-input-point features."
            )

        return features

    def parameter_count(
        self,
        *,
        trainable_only: bool = True,
        include_projection: bool = True,
    ) -> int:
        def _count(module: nn.Module) -> int:
            params = module.parameters()
            if trainable_only:
                params = (p for p in params if p.requires_grad)
            return sum(p.numel() for p in params)

        total = _count(self.ptv3)
        if include_projection:
            total += _count(self.proj)
        return total
