"""Low-rank adapters for the pretrained Utonia sparse point encoder.

The pretrained checkpoint is kept frozen. Linear updates use the standard
``B(A(x))`` LoRA factorization. Utonia's only convolutional operator is a
3x3x3 ``SubMConv3d`` xCPE block, so its adapter first applies the same spatial
kernel while reducing channels to ``rank`` and then restores the output width
with a 1x1x1 sparse convolution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn
import spconv.pytorch as spconv


@dataclass(frozen=True)
class UtoniaLoRASettings:
    enabled: bool
    linear_rank: int = 16
    conv_rank: int = 16
    linear_alpha: float = 16.0
    conv_alpha: float = 16.0
    dropout: float = 0.0
    input_mode: str = "xyzi"


@dataclass(frozen=True)
class UtoniaLoRAReport:
    linear_modules: int
    conv_modules: int


def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def resolve_utonia_lora_settings(cfg) -> UtoniaLoRASettings:
    """Resolve and validate ``p2g.utonia_lora`` without requiring OmegaConf."""
    block = _cfg_get(cfg, "utonia_lora", None)
    enabled = bool(_cfg_get(block, "enable", False))
    if not enabled:
        return UtoniaLoRASettings(enabled=False)

    shared_rank = int(_cfg_get(block, "rank", 16))
    linear_rank = int(_cfg_get(block, "linear_rank", shared_rank))
    conv_rank = int(_cfg_get(block, "conv_rank", shared_rank))
    shared_alpha = float(_cfg_get(block, "alpha", shared_rank))
    linear_alpha = float(_cfg_get(block, "linear_alpha", shared_alpha))
    conv_alpha = float(_cfg_get(block, "conv_alpha", shared_alpha))
    dropout = float(_cfg_get(block, "dropout", 0.0))
    input_mode = str(_cfg_get(block, "input_mode", "xyzi")).lower()

    if linear_rank <= 0 or conv_rank <= 0:
        raise ValueError("p2g.utonia_lora ranks must be positive")
    if linear_alpha <= 0.0 or conv_alpha <= 0.0:
        raise ValueError("p2g.utonia_lora alpha values must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("p2g.utonia_lora.dropout must be in [0, 1)")
    if input_mode != "xyzi":
        raise ValueError(
            "p2g.utonia_lora.input_mode currently supports only 'xyzi'"
        )
    return UtoniaLoRASettings(
        enabled=True,
        linear_rank=linear_rank,
        conv_rank=conv_rank,
        linear_alpha=linear_alpha,
        conv_alpha=conv_alpha,
        dropout=dropout,
        input_mode=input_mode,
    )


class LoRALinear(nn.Module):
    """Frozen base linear plus a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, *, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = (
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        )
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, feature):
        base = self.base_layer(feature)
        delta = self.lora_up(self.lora_down(self.lora_dropout(feature)))
        return base + self.scaling * delta

    def set_lora_requires_grad(self, enabled: bool):
        for parameter in self.lora_down.parameters():
            parameter.requires_grad = bool(enabled)
        for parameter in self.lora_up.parameters():
            parameter.requires_grad = bool(enabled)


class LoRASubMConv3d(spconv.SparseModule):
    """Frozen sparse 3D convolution plus spatial-down/pointwise-up LoRA."""

    def __init__(self, base_layer: spconv.SubMConv3d, *, rank: int,
                 alpha: float, dropout: float = 0.0):
        super().__init__()
        if int(base_layer.groups) != 1:
            raise ValueError("Utonia Conv LoRA currently requires groups=1")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = (
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        )

        # A: preserve the base spatial receptive field while reducing channels.
        self.lora_down = spconv.SubMConv3d(
            base_layer.in_channels,
            self.rank,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=1,
            bias=False,
            indice_key=base_layer.indice_key,
            algo=getattr(base_layer, "algo", None),
        )
        # B: restore output channels without enlarging the receptive field.
        self.lora_up = spconv.SubMConv3d(
            self.rank,
            base_layer.out_channels,
            kernel_size=1,
            bias=False,
            indice_key=None,
            algo=getattr(base_layer, "algo", None),
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def in_channels(self):
        return self.base_layer.in_channels

    @property
    def out_channels(self):
        return self.base_layer.out_channels

    @property
    def kernel_size(self):
        return self.base_layer.kernel_size

    def forward(self, sparse_feature):
        base = self.base_layer(sparse_feature)
        lora_input = sparse_feature.replace_feature(
            self.lora_dropout(sparse_feature.features)
        )
        delta = self.lora_up(self.lora_down(lora_input))
        if base.features.shape != delta.features.shape:
            raise RuntimeError(
                "Utonia Conv LoRA changed the sparse output support unexpectedly"
            )
        return base.replace_feature(base.features + self.scaling * delta.features)

    def set_lora_requires_grad(self, enabled: bool):
        for parameter in self.lora_down.parameters():
            parameter.requires_grad = bool(enabled)
        for parameter in self.lora_up.parameters():
            parameter.requires_grad = bool(enabled)


_LORA_TYPES = (LoRALinear, LoRASubMConv3d)


def adapt_utonia_embedding_to_xyzi(feature_extractor) -> int:
    """Replace the checkpoint's 9D stem by an initially equivalent XYZI stem.

    The live dataloader supplied ``[xyz, zero_rgb, zero_normal]`` to the pretrained
    9D stem. Copying its xyz columns and zero-initializing the new intensity column
    therefore preserves the exact base function before LoRA starts updating.
    """
    stem = feature_extractor.embedding.stem
    base = stem._modules.get("linear")
    if not isinstance(base, nn.Linear):
        raise TypeError("expected Utonia embedding.stem.linear to be nn.Linear")
    old_in_channels = int(base.in_features)
    if old_in_channels == 4:
        return old_in_channels
    if old_in_channels < 3:
        raise ValueError(
            f"Utonia stem needs at least xyz columns, got {old_in_channels}"
        )

    replacement = nn.Linear(4, base.out_features, bias=base.bias is not None)
    replacement = replacement.to(device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:, :3].copy_(base.weight[:, :3])
        if base.bias is not None:
            replacement.bias.copy_(base.bias)
    stem._modules["linear"] = replacement
    feature_extractor.embedding.in_channels = 4
    return old_in_channels


def build_xyzi_utonia_input(ptv3_input):
    """Return a shallow input copy whose feature tensor is exactly ``[x,y,z,i]``."""
    if "coord" not in ptv3_input or "strength" not in ptv3_input:
        raise KeyError(
            "Utonia XYZI LoRA requires aligned ptv3_input['coord'] and ['strength']"
        )
    coord = ptv3_input["coord"]
    strength = ptv3_input["strength"]
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"expected ptv3 coord shape (N,3), got {tuple(coord.shape)}")
    if strength.ndim == 1:
        strength = strength.unsqueeze(-1)
    if strength.ndim != 2 or strength.shape != (coord.shape[0], 1):
        raise ValueError(
            "expected ptv3 strength shape (N,) or (N,1), got "
            f"{tuple(strength.shape)}"
        )
    result = dict(ptv3_input)
    result["feat"] = torch.cat(
        [coord, strength.to(device=coord.device, dtype=coord.dtype)], dim=-1
    )
    return result


def inject_utonia_lora(module: nn.Module,
                       settings: UtoniaLoRASettings) -> UtoniaLoRAReport:
    """Recursively attach adapters to every Linear and SubMConv3d in ``module``."""
    if not settings.enabled:
        return UtoniaLoRAReport(linear_modules=0, conv_modules=0)

    linear_count = 0
    conv_count = 0

    def _inject(parent):
        nonlocal linear_count, conv_count
        for name, child in list(parent.named_children()):
            if isinstance(child, _LORA_TYPES):
                continue
            if isinstance(child, nn.Linear):
                setattr(
                    parent,
                    name,
                    LoRALinear(
                        child,
                        rank=settings.linear_rank,
                        alpha=settings.linear_alpha,
                        dropout=settings.dropout,
                    ),
                )
                linear_count += 1
            elif isinstance(child, spconv.SubMConv3d):
                setattr(
                    parent,
                    name,
                    LoRASubMConv3d(
                        child,
                        rank=settings.conv_rank,
                        alpha=settings.conv_alpha,
                        dropout=settings.dropout,
                    ),
                )
                conv_count += 1
            else:
                _inject(child)

    _inject(module)
    return UtoniaLoRAReport(
        linear_modules=linear_count,
        conv_modules=conv_count,
    )


def freeze_utonia_and_enable_active_lora(feature_extractor: nn.Module,
                                         active_roots: Iterable[nn.Module]) -> int:
    """Freeze the base and enable adapters only on the actually executed stages."""
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    active_parameter_ids = set()
    for root in active_roots:
        for module in root.modules():
            if isinstance(module, _LORA_TYPES):
                module.set_lora_requires_grad(True)
                for parameter in module.lora_down.parameters():
                    active_parameter_ids.add(id(parameter))
                for parameter in module.lora_up.parameters():
                    active_parameter_ids.add(id(parameter))
    return sum(
        parameter.numel()
        for parameter in feature_extractor.parameters()
        if id(parameter) in active_parameter_ids
    )


def set_active_lora_mode(active_roots: Iterable[nn.Module], training: bool):
    """Keep the frozen backbone in eval mode while honoring adapter dropout mode."""
    for root in active_roots:
        for module in root.modules():
            if isinstance(module, _LORA_TYPES):
                module.train(bool(training))


__all__ = [
    "LoRALinear",
    "LoRASubMConv3d",
    "UtoniaLoRAReport",
    "UtoniaLoRASettings",
    "adapt_utonia_embedding_to_xyzi",
    "build_xyzi_utonia_input",
    "freeze_utonia_and_enable_active_lora",
    "inject_utonia_lora",
    "resolve_utonia_lora_settings",
    "set_active_lora_mode",
]
