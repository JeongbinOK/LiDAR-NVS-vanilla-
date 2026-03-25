"""Module A: Point Feature Backbone with serialized point transformer.

Uses dual serialization (Z-order + Hilbert) following PTv3 design:
even blocks use Z-order, odd blocks use Hilbert curve, providing
complementary locality coverage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quantize_coords(xyz: torch.Tensor, num_bits: int = 10):
    """Quantize 3D coordinates to integer grid [0, 2^num_bits - 1]."""
    mins = xyz.min(dim=0).values
    span = (xyz.max(dim=0).values - mins).clamp(min=1e-6)
    coords = ((xyz - mins) / span * ((1 << num_bits) - 1)).long()
    return coords.clamp(0, (1 << num_bits) - 1)


def _interleave_bits(x, y, z, num_bits, device):
    """Interleave bits of 3 coordinates into a single key."""
    bits = torch.arange(num_bits, device=device, dtype=torch.long)
    return (
        (((x.unsqueeze(1) >> bits) & 1) << (3 * bits))
        | (((y.unsqueeze(1) >> bits) & 1) << (3 * bits + 1))
        | (((z.unsqueeze(1) >> bits) & 1) << (3 * bits + 2))
    ).sum(dim=1)


def z_order_key(xyz: torch.Tensor, num_bits: int = 10) -> torch.Tensor:
    """Compute Z-order (Morton) curve keys for 3D points.

    Interleaves bits of quantized x, y, z coordinates into a single 30-bit
    integer. Points nearby in 3D tend to have nearby keys, enabling
    locality-preserving 1D serialization via sort.

    Args:
        xyz: [N, 3] point positions
        num_bits: bits per coordinate axis (10 -> 30-bit key, fits int64)

    Returns:
        keys: [N] Z-order keys (int64)
    """
    coords = _quantize_coords(xyz, num_bits)
    return _interleave_bits(
        coords[:, 0], coords[:, 1], coords[:, 2], num_bits, xyz.device,
    )


def hilbert_curve_key(xyz: torch.Tensor, num_bits: int = 10) -> torch.Tensor:
    """Compute 3D Hilbert curve keys using the Skilling transpose algorithm.

    Hilbert curves have provably better locality preservation than Z-order:
    consecutive points along the curve differ in exactly one coordinate,
    reducing worst-case locality gaps at spatial boundaries.

    Args:
        xyz: [N, 3] point positions
        num_bits: bits per coordinate axis (10 -> 30-bit key, fits int64)

    Returns:
        keys: [N] Hilbert curve keys (int64)
    """
    coords = _quantize_coords(xyz, num_bits)
    x, y, z = coords[:, 0].clone(), coords[:, 1].clone(), coords[:, 2].clone()

    M = 1 << (num_bits - 1)

    # Phase 1: Inverse undo (Skilling 2004)
    Q = M
    while Q > 1:
        P = Q - 1

        # dim 0 (x): invert low bits if current bit is set
        mask = (x & Q) != 0
        x = torch.where(mask, x ^ P, x)

        # dim 1 (y): invert x if y-bit set, else exchange low bits of x,y
        mask = (y & Q) != 0
        t = (x ^ y) & P
        x = torch.where(mask, x ^ P, x ^ t)
        y = torch.where(mask, y, y ^ t)

        # dim 2 (z): invert x if z-bit set, else exchange low bits of x,z
        mask = (z & Q) != 0
        t = (x ^ z) & P
        x = torch.where(mask, x ^ P, x ^ t)
        z = torch.where(mask, z, z ^ t)

        Q >>= 1

    # Phase 2: Gray encode
    y = y ^ x
    z = z ^ y

    # Phase 3: Parity fix
    t = torch.zeros_like(x)
    Q = M
    while Q > 1:
        t = torch.where((z & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1

    x = x ^ t
    y = y ^ t
    z = z ^ t

    # Interleave the transposed coordinates to form the Hilbert index
    return _interleave_bits(x, y, z, num_bits, xyz.device)


class WindowedAttention(nn.Module):
    """Multi-head self-attention within local windows."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [num_windows, W, D]
            mask: [num_windows, W] boolean, True = valid (not padding)

        Returns:
            [num_windows, W, D]
        """
        B, W, D = x.shape
        qkv = self.qkv(x).reshape(B, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, W, d]
        q, k, v = qkv.unbind(0)

        attn_mask = None
        if mask is not None:
            # True in pad_mask = padded position -> fill with -inf
            attn_mask = torch.zeros(B, 1, W, W, device=x.device, dtype=x.dtype)
            pad_mask = ~mask
            attn_mask.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, W, D)
        return self.proj(out)


class SerializedTransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> residual -> LN -> FFN -> residual."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class PointFeatureBackbone(nn.Module):
    """Module A: Dual-serialization point transformer backbone (PTv3-inspired).

    Transforms raw point features (xyz + intensity) into per-point latent
    features via local self-attention. Even blocks use Z-order serialization,
    odd blocks use Hilbert curve, providing complementary locality coverage.
    Shifted windows (W/2) on odd blocks add cross-window information flow.
    """

    def __init__(self, dim: int = 64, num_blocks: int = 3,
                 window_size: int = 48, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.embed = nn.Sequential(
            nn.Linear(4, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            SerializedTransformerBlock(dim, num_heads)
            for _ in range(num_blocks)
        ])

    def _process_block(self, features, sort_order, unsort_order,
                       block, shift, N, W, device):
        """Sort → window → attention → unsort for one block."""
        sorted_feats = features[sort_order]

        if shift > 0:
            sorted_feats = torch.roll(sorted_feats, shift, dims=0)

        # Pad to multiple of W
        pad_size = (W - N % W) % W
        if pad_size > 0:
            sorted_feats = F.pad(sorted_feats, (0, 0, 0, pad_size))

        num_windows = sorted_feats.shape[0] // W
        windows = sorted_feats.view(num_windows, W, self.dim)

        # Validity mask (False for padding tokens)
        mask = torch.ones(num_windows, W, dtype=torch.bool, device=device)
        if pad_size > 0:
            mask[-1, W - pad_size:] = False

        windows = block(windows, mask)
        sorted_feats = windows.reshape(-1, self.dim)[:N]

        if shift > 0:
            sorted_feats = torch.roll(sorted_feats, -shift, dims=0)

        return sorted_feats[unsort_order]

    def forward(self, xyz: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [N, 3] point positions
            intensity: [N] or [N, 1] per-point intensity

        Returns:
            features: [N, D] per-point features
        """
        N = xyz.shape[0]
        W = self.window_size
        device = xyz.device

        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(1)

        # Initial embedding
        features = self.embed(torch.cat([xyz, intensity], dim=1))  # [N, D]

        # Precompute both serialization orders
        arange = torch.arange(N, device=device)

        z_sort = z_order_key(xyz).argsort()
        z_unsort = torch.empty_like(z_sort)
        z_unsort[z_sort] = arange

        h_sort = hilbert_curve_key(xyz).argsort()
        h_unsort = torch.empty_like(h_sort)
        h_unsort[h_sort] = arange

        for block_idx, block in enumerate(self.blocks):
            # Even blocks: Z-order, Odd blocks: Hilbert
            if block_idx % 2 == 0:
                sort_order, unsort_order = z_sort, z_unsort
            else:
                sort_order, unsort_order = h_sort, h_unsort

            # Shifted windows on odd blocks for cross-window flow
            shift = (W // 2) if (block_idx % 2 == 1) else 0

            features = self._process_block(
                features, sort_order, unsort_order,
                block, shift, N, W, device,
            )

        return features
