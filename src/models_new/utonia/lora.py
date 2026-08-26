"""LoRA injection for the frozen Utonia (PointTransformerV3) backbone.

Wraps target Linear/SubMConv3d layers in-place with a frozen base path plus a
zero-initialized low-rank residual branch, so the model is output-identical to
the frozen baseline until the LoRA parameters are trained.
"""
from __future__ import annotations

import torch.nn as nn
from torch.nn.init import trunc_normal_
import spconv.pytorch as spconv


class LoRALinear(nn.Module):
    """Frozen nn.Linear + zero-initialized low-rank residual branch."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


class LoRASubMConv3d(spconv.modules.SparseModule):
    """Frozen 3x3x3 SubMConv3d + zero-initialized low-rank residual branch.

    Must subclass SparseModule (not plain nn.Module) so PointSequential's
    is_spconv_module dispatch routes SparseConvTensor input through this
    wrapper the same way it did through the bare frozen conv.
    """

    def __init__(self, base: spconv.SubMConv3d, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        # Must NOT reuse base.indice_key: that key is intentionally shared across
        # every block's cpe conv within a stage (same sparsity pattern -> one
        # rulebook). Stacking a differently-shaped (C->rank) conv onto that same
        # shared cache slot corrupts the implicit-gemm rulebook (empty-indices
        # assert on backward). Leave indice_key unset so spconv builds/caches
        # this conv's rulebook independently.
        self.down = spconv.SubMConv3d(base.in_channels, rank, kernel_size=3, bias=False)
        self.up = spconv.SubMConv3d(rank, base.out_channels, kernel_size=1, bias=False)
        trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        base_out = self.base(x)
        delta = self.up(self.down(x))
        return base_out.replace_feature(base_out.features + delta.features * self.scale)


def inject_lora(model, max_stage: int, rank: int, alpha: float | None = None) -> int:
    """Wrap embedding + enc[0..max_stage] Linear/SubMConv3d layers with LoRA in-place.

    Never touches model.dec (decoder), which this pipeline never calls.
    Returns the number of layers wrapped.
    """
    count = 0

    def _lin(module: nn.Linear) -> LoRALinear:
        nonlocal count
        count += 1
        return LoRALinear(module, rank, alpha)

    def _conv(module: spconv.SubMConv3d) -> LoRASubMConv3d:
        nonlocal count
        count += 1
        return LoRASubMConv3d(module, rank, alpha)

    model.embedding.stem.linear = _lin(model.embedding.stem.linear)

    for s in range(max_stage + 1):
        stage = model.enc[s]
        down = getattr(stage, "down", None)
        if down is not None:
            down.proj = _lin(down.proj)
        for name, block in stage.named_children():
            if not name.startswith("block"):
                continue
            block.attn.qkv = _lin(block.attn.qkv)
            block.attn.proj = _lin(block.attn.proj)
            mlp = block.mlp[0]
            mlp.fc1 = _lin(mlp.fc1)
            mlp.fc2 = _lin(mlp.fc2)
            block.cpe.add_module("0", _conv(block.cpe[0]))
            block.cpe.add_module("1", _lin(block.cpe[1]))

    return count
