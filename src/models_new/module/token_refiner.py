"""Post-fusion joint token refiners.

``serial`` preserves the existing serialized RoPE-attention implementation,
including its xCPE branch. ``sparse`` uses exact occupied-cell local
self-attention without xCPE: every token attends to occupied cells in a
configurable odd 3D window around its own ``grid_coord``.  The lookup key includes
the frame id built from ``offset``, so neither implementation mixes frames;
temporal fusion remains the SphericalQueryHead's job.

The attention/MLP branches use ``layer_scale`` and start near identity.  The
serial refiner additionally zero-initializes xCPE's unscaled residual projection.
"""
from __future__ import annotations

from itertools import product

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..utonia.model import (
    Block,
    LayerScale,
    MLP,
    Point3DRoPE,
    flash_attn,
)
from ..utonia.structure import Point


class JointTokenRefiner(nn.Module):
    """Existing serialized-attention refiner (kept for checkpoint compatibility)."""

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


def _window_shape(value) -> tuple[int, int, int]:
    """Normalize an integer or xyz triplet into an odd local-window shape."""
    if isinstance(value, (int, float)):
        shape = (int(value),) * 3
    else:
        try:
            shape = tuple(int(v) for v in value)
        except TypeError as exc:
            raise ValueError(
                "joint_refiner.window_size must be an odd int or xyz triplet"
            ) from exc
    if len(shape) != 3 or any(v <= 0 or v % 2 == 0 for v in shape):
        raise ValueError(
            "joint_refiner.window_size must contain three positive odd values, "
            f"got {shape}"
        )
    return shape


@torch.no_grad()
def _sparse_window_neighbors(grid_coord, offset, neighbor_offsets):
    """Return ``(N, K)`` exact local-neighbor indices, with ``-1`` for holes.

    Hashes are built from ``(frame_id, x, y, z)`` and resolved with one sorted
    search, rather than an ``N x N`` distance matrix.  ``grid_coord`` must be
    unique within each frame, matching the occupied Utonia-token contract.
    """
    n_tokens = int(grid_coord.shape[0])
    if n_tokens == 0:
        return torch.empty(
            (0, int(neighbor_offsets.shape[0])),
            dtype=torch.long,
            device=grid_coord.device,
        )
    if offset.ndim != 1 or offset.numel() == 0:
        raise ValueError("joint_refiner offset must be a non-empty 1D tensor")
    if int(offset[-1].item()) != n_tokens:
        raise ValueError(
            f"joint_refiner offset ends at {int(offset[-1].item())}, "
            f"but received {n_tokens} tokens"
        )

    frame_counts = torch.diff(
        offset.long(), prepend=offset.new_zeros(1, dtype=torch.long)
    )
    if bool((frame_counts < 0).any().item()):
        raise ValueError("joint_refiner offset must be monotonically increasing")
    frame_id = torch.arange(
        offset.numel(), device=offset.device, dtype=torch.long
    ).repeat_interleave(frame_counts)

    coord = grid_coord.to(device=offset.device, dtype=torch.long)
    offsets = neighbor_offsets.to(device=coord.device, dtype=torch.long)
    radius = offsets.abs().amax(dim=0)
    lower = coord.amin(dim=0) - radius
    span = coord.amax(dim=0) - coord.amin(dim=0) + 1 + 2 * radius

    def encode(batch, xyz):
        shifted = xyz - lower
        return (
            ((batch * span[0] + shifted[..., 0]) * span[1] + shifted[..., 1])
            * span[2]
            + shifted[..., 2]
        )

    token_keys = encode(frame_id, coord)
    sorted_keys, order = torch.sort(token_keys)
    if n_tokens > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any().item()):
        raise ValueError(
            "joint_refiner sparse attention requires unique grid_coord per frame"
        )

    candidate_coord = coord[:, None, :] + offsets[None, :, :]
    candidate_keys = encode(frame_id[:, None], candidate_coord).reshape(-1)
    locations = torch.searchsorted(sorted_keys, candidate_keys)
    in_range = locations < n_tokens
    safe_locations = locations.clamp(max=n_tokens - 1)
    matched = in_range & (sorted_keys[safe_locations] == candidate_keys)
    neighbors = order[safe_locations]
    neighbors = neighbors.masked_fill(~matched, -1)
    return neighbors.view(n_tokens, offsets.shape[0])


class SparseLocalSelfAttention(nn.Module):
    """Exact occupied-voxel attention with one query sequence per token."""

    def __init__(self, dim: int, num_heads: int, max_neighbors: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 chunk_size: int = 4096):
        super().__init__()
        if dim % num_heads != 0 or (dim // num_heads) % 6 != 0:
            raise ValueError(
                f"joint_refiner dim={dim}/num_heads={num_heads} give head_dim="
                f"{dim / num_heads}; 3D RoPE needs head_dim divisible by 6"
            )
        if chunk_size <= 0:
            raise ValueError("joint_refiner.attn_chunk_size must be positive")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.max_neighbors = int(max_neighbors)
        self.chunk_size = int(chunk_size)
        self.scale = self.head_dim ** -0.5
        self.attn_drop = float(attn_drop)

        self.qkv = nn.Linear(dim, dim * 3)
        self.rope = Point3DRoPE(self.head_dim, base=10)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _manual_attention(self, q, k, v, neighbors):
        """Chunked CPU/non-flash fallback; uses fp32 scores and softmax."""
        chunks = []
        for start in range(0, q.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, q.shape[0])
            index = neighbors[start:end]
            valid = index >= 0
            safe_index = index.clamp_min(0)
            key = k[safe_index]
            value = v[safe_index]
            score = torch.einsum(
                "nhd,nkhd->nhk", q[start:end].float(), key.float()
            ) * self.scale
            score = score.masked_fill(~valid[:, None, :], float("-inf"))
            weight = torch.softmax(score, dim=-1)
            weight = F.dropout(weight, p=self.attn_drop, training=self.training)
            out = torch.einsum(
                "nhk,nkhd->nhd", weight.to(value.dtype), value
            )
            chunks.append(out)
        return torch.cat(chunks, dim=0)

    def _flash_attention(self, q, k, v, neighbors):
        valid = neighbors >= 0
        lengths = valid.sum(dim=1, dtype=torch.int32)
        if bool((lengths == 0).any().item()):
            raise RuntimeError("sparse local attention lost a token's self neighbor")
        kv_index = neighbors.masked_select(valid)
        cu_seqlens_q = torch.arange(
            q.shape[0] + 1, device=q.device, dtype=torch.int32
        )
        cu_seqlens_k = F.pad(
            torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0)
        )

        # Match the serialized path: FlashAttention runs in bf16 and the result
        # returns to the fusion trunk's input dtype.
        out = flash_attn.flash_attn_varlen_func(
            q=q.to(torch.bfloat16),
            k=k.index_select(0, kv_index).to(torch.bfloat16),
            v=v.index_select(0, kv_index).to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=self.max_neighbors,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )
        return out.to(q.dtype)

    def forward(self, feat, coord, neighbors):
        n_tokens = feat.shape[0]
        if n_tokens == 0:
            return feat
        qkv = self.qkv(feat).view(
            n_tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=1)
        q, k = self.rope(q, k, coord)
        if feat.is_cuda and flash_attn is not None:
            out = self._flash_attention(q, k, v, neighbors)
        else:
            out = self._manual_attention(q, k, v, neighbors)
        out = out.reshape(n_tokens, self.dim)
        return self.proj_drop(self.proj(out))


class SparseLocalBlock(nn.Module):
    """Pre-norm sparse-local attention/MLP residual block without xCPE."""

    def __init__(self, dim: int, num_heads: int, max_neighbors: int,
                 mlp_ratio: float, layer_scale: float, chunk_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseLocalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            max_neighbors=max_neighbors,
            chunk_size=chunk_size,
        )
        self.ls1 = LayerScale(dim, init_values=layer_scale)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_channels=dim,
            hidden_channels=int(dim * mlp_ratio),
            out_channels=dim,
        )
        self.ls2 = LayerScale(dim, init_values=layer_scale)

    def forward(self, feat, coord, neighbors):
        feat = feat + self.ls1(
            self.attn(self.norm1(feat), coord, neighbors)
        )
        feat = feat + self.ls2(self.mlp(self.norm2(feat)))
        return feat


class SparseLocalTokenRefiner(nn.Module):
    """Post-fusion refiner using an exact per-frame occupied-cell window."""

    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8,
                 window_size=3, mlp_ratio: float = 2.0,
                 layer_scale: float = 1e-5, coord_scale: float = 0.2,
                 attn_chunk_size: int = 4096):
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.window_size = _window_shape(window_size)
        radius = tuple((value - 1) // 2 for value in self.window_size)
        neighbor_offsets = torch.tensor(
            list(product(
                range(-radius[0], radius[0] + 1),
                range(-radius[1], radius[1] + 1),
                range(-radius[2], radius[2] + 1),
            )),
            dtype=torch.long,
        )
        self.register_buffer(
            "neighbor_offsets", neighbor_offsets, persistent=False
        )
        max_neighbors = int(neighbor_offsets.shape[0])
        self.blocks = nn.ModuleList([
            SparseLocalBlock(
                dim=dim,
                num_heads=num_heads,
                max_neighbors=max_neighbors,
                mlp_ratio=mlp_ratio,
                layer_scale=layer_scale,
                chunk_size=attn_chunk_size,
            )
            for _ in range(int(depth))
        ])

    def forward(self, feat, pos, grid_coord, offset):
        if feat.shape[0] == 0:
            return feat
        neighbors = _sparse_window_neighbors(
            grid_coord, offset, self.neighbor_offsets
        )
        coord = pos * self.coord_scale
        for block in self.blocks:
            feat = block(feat, coord, neighbors)
        return feat


def build_joint_refiner(cfg, *, dim: int, coord_scale: float):
    """Build the optional configured post-fusion refiner."""
    if cfg is None or not bool(getattr(cfg, "enable", False)):
        return None
    refiner_type = str(getattr(cfg, "type", "serial")).lower()
    common = dict(
        dim=dim,
        depth=int(getattr(cfg, "depth", 2)),
        num_heads=int(getattr(cfg, "num_heads", 8)),
        mlp_ratio=float(getattr(cfg, "mlp_ratio", 2.0)),
        layer_scale=float(getattr(cfg, "layer_scale", 1e-5)),
        coord_scale=coord_scale,
    )
    if refiner_type in {"serial", "serialized"}:
        return JointTokenRefiner(
            **common,
            patch_size=int(getattr(cfg, "patch_size", 1024)),
            order=tuple(getattr(cfg, "order", ("z", "hilbert"))),
        )
    if refiner_type == "sparse":
        return SparseLocalTokenRefiner(
            **common,
            window_size=getattr(cfg, "window_size", 3),
            attn_chunk_size=int(getattr(cfg, "attn_chunk_size", 4096)),
        )
    raise ValueError(
        "joint_refiner.type must be 'serial' or 'sparse', "
        f"got {refiner_type!r}"
    )


__all__ = [
    "JointTokenRefiner",
    "SparseLocalSelfAttention",
    "SparseLocalTokenRefiner",
    "build_joint_refiner",
]
