"""Module A: Point Feature Backbone with serialized point transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    mins = xyz.min(dim=0).values
    span = (xyz.max(dim=0).values - mins).clamp(min=1e-6)
    normalized = ((xyz - mins) / span * ((1 << num_bits) - 1)).long()
    normalized = normalized.clamp(0, (1 << num_bits) - 1)

    x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    # Vectorized bit interleaving (replaces Python loop)
    bits = torch.arange(num_bits, device=xyz.device, dtype=torch.long)
    key = (
        (((x.unsqueeze(1) >> bits) & 1) << (3 * bits))
        | (((y.unsqueeze(1) >> bits) & 1) << (3 * bits + 1))
        | (((z.unsqueeze(1) >> bits) & 1) << (3 * bits + 2))
    ).sum(dim=1)

    return key


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
    """Module A: Z-order serialized point transformer backbone.

    Transforms raw point features (xyz + intensity) into per-point latent
    features via local self-attention in Z-order curve windows. Alternate
    blocks use shifted windows (by W/2) for cross-window information flow.
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

        # Z-order serialization (computed once, reused across blocks)
        z_keys = z_order_key(xyz)
        sort_order = z_keys.argsort()
        unsort_order = torch.empty_like(sort_order)
        unsort_order[sort_order] = torch.arange(N, device=device)

        sorted_feats = features[sort_order]

        for block_idx, block in enumerate(self.blocks):
            # Shifted windows at odd blocks
            shift = (W // 2) if (block_idx % 2 == 1) else 0

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
