"""Post-fusion joint token refiner.

After the Utonia (geometry) and intensity streams are fused pointwise into the
192D trunk, this module runs a few serialized RoPE attention blocks (reusing
Utonia's ``Block``, so each carries xCPE) over the fused tokens. This is where
"look at the same points together" actually happens: within one frame's tokens a
single grouping lets neighbouring geometry+intensity interact directly in one
patch, which the two independent encoder streams cannot do.

Attention stays WITHIN a frame: the Point is built with the per-frame ``offset``,
so flash var-len patches never cross frame boundaries (no dynamic-object mixing;
temporal fusion remains the SphericalQueryHead's job). ``coord = pos *
coord_scale`` reuses the Utonia coord scale (0.2), keeping the RoPE frequency scale
identical to the frozen backbone.

Near-identity start (safe given the NaN-collapse history): ``layer_scale`` scales
the attn/mlp residuals, but the xCPE residual is NOT layer-scaled (Block.forward:
``point.feat = shortcut + cpe(point).feat``), so the Linear inside each block's cpe
is zero-initialized -> ``LN(Linear(.)) = LN(0) = 0`` and the cpe contributes 0 at
init. So the whole refiner starts as ~identity on the trunk.
"""
from __future__ import annotations

import torch.nn as nn

from ..utonia.model import Block
from ..utonia.structure import Point


class JointTokenRefiner(nn.Module):
    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8,
                 patch_size: int = 1024, mlp_ratio: float = 2.0,
                 layer_scale: float = 1e-5, coord_scale: float = 0.2,
                 order=("z", "hilbert")):
        super().__init__()
        if dim % num_heads != 0 or (dim // num_heads) % 6 != 0:
            raise ValueError(
                f"joint_refiner dim={dim}/num_heads={num_heads} give head_dim="
                f"{dim / num_heads}; RoPE needs head_dim divisible by 6")
        self.coord_scale = float(coord_scale)
        self.order = tuple(order)
        self.blocks = nn.ModuleList([
            Block(
                channels=dim,
                num_heads=num_heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                layer_scale=layer_scale,
                order_index=i % len(self.order),
                cpe_indice_key="jref",
                enable_flash=True,
                # flash attention asserts both upcast flags are False; Block defaults
                # them True, so they must be passed explicitly for standalone Blocks.
                upcast_attention=False,
                upcast_softmax=False,
                drop_path=0.0,
            )
            for i in range(int(depth))
        ])
        # xCPE residual is not layer-scaled -> zero-init its Linear for identity start.
        for blk in self.blocks:
            cpe_linear = blk.cpe[1]
            nn.init.zeros_(cpe_linear.weight)
            nn.init.zeros_(cpe_linear.bias)

    def forward(self, feat, pos, grid_coord, offset):
        """feat (N,dim), pos (N,3) metric sensor-frame position, grid_coord (N,3)
        Utonia-token grid coords, offset (n_frames,) per-frame cumsum. Returns the
        refined (N,dim) trunk feature."""
        point = Point(dict(
            feat=feat,
            coord=pos * self.coord_scale,
            grid_coord=grid_coord,
            offset=offset,
        ))
        point.serialization(order=self.order, shuffle_orders=False)
        point.sparsify()
        for blk in self.blocks:
            point = blk(point)
        return point.feat


def build_joint_refiner(cfg, *, dim: int, coord_scale: float):
    """Build the optional configured post-fusion refiner."""
    if cfg is None or not bool(getattr(cfg, "enable", False)):
        return None
    return JointTokenRefiner(
        dim=dim,
        depth=int(getattr(cfg, "depth", 2)),
        num_heads=int(getattr(cfg, "num_heads", 8)),
        patch_size=int(getattr(cfg, "patch_size", 1024)),
        mlp_ratio=float(getattr(cfg, "mlp_ratio", 2.0)),
        layer_scale=float(getattr(cfg, "layer_scale", 1e-5)),
        coord_scale=coord_scale,
        order=tuple(getattr(cfg, "order", ("z", "hilbert"))),
    )


__all__ = ["JointTokenRefiner", "build_joint_refiner"]
