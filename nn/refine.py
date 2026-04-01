"""Stage 3.5: Cross-Attention Refinement.

Centers refine their features by attending to local point neighborhoods
(via assign matrix) and exchanging info via self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineLayer(nn.Module):
    """Single refinement layer: cross-attn → self-attn → FFN."""

    def __init__(self, dim: int, num_heads: int = 4, local_topk: int = 64):
        super().__init__()
        self.local_topk = local_topk
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Cross-attention (centers attend to local points)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.cross_out = nn.Linear(dim, dim)

        # Self-attention (centers attend to each other)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def _local_cross_attention(
        self, center_feats, point_feats, assign,
    ):
        """Each center attends to its top-M assigned points."""
        K, D = center_feats.shape
        M = min(self.local_topk, assign.shape[0])

        # Find top-M points per center from assign matrix
        # assign.T: [K, N] — topk gives highest-assigned points per center
        _, local_idx = assign.T.topk(M, dim=-1)  # [K, M]

        # Gather local point features
        local_feats = point_feats[local_idx]  # [K, M, D]

        # Cross-attention: center (query) → local points (key, value)
        q = self.q_proj(center_feats).view(K, 1, self.num_heads, self.head_dim)
        kv = self.kv_proj(local_feats).view(K, M, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        # Reshape for attention: [K, heads, seq, head_dim]
        q = q.permute(0, 2, 1, 3)   # [K, H, 1, d]
        k = k.permute(0, 2, 1, 3)   # [K, H, M, d]
        v = v.permute(0, 2, 1, 3)   # [K, H, M, d]

        attn_out = F.scaled_dot_product_attention(q, k, v)  # [K, H, 1, d]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(K, D)  # [K, D]

        return self.cross_out(attn_out)

    def forward(
        self,
        center_feats: torch.Tensor,
        point_feats: torch.Tensor,
        assign: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            center_feats: [K, D]
            point_feats: [N, D]
            assign: [N, K] soft assignment matrix

        Returns:
            center_feats: [K, D] refined
        """
        # 1. Local cross-attention
        cross_out = self._local_cross_attention(center_feats, point_feats, assign)
        center_feats = center_feats + cross_out
        center_feats = self.norm1(center_feats)

        # 2. Self-attention (add batch dim for nn.MultiheadAttention)
        cf = center_feats.unsqueeze(0)  # [1, K, D]
        sa_out = self.self_attn(cf, cf, cf)[0].squeeze(0)  # [K, D]
        center_feats = center_feats + sa_out
        center_feats = self.norm2(center_feats)

        # 3. FFN (post-norm, consistent with sub-layers 1 and 2)
        center_feats = self.norm3(center_feats + self.ffn(center_feats))

        return center_feats


class CrossAttentionRefiner(nn.Module):
    """Multi-layer cross-attention refinement for center features."""

    def __init__(
        self,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        local_topk: int = 64,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RefineLayer(dim, num_heads, local_topk)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        center_feats: torch.Tensor,
        point_feats: torch.Tensor,
        assign: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            center_feats: [K, D]
            point_feats: [N, D]
            assign: [N, K]

        Returns:
            center_feats: [K, D] refined
        """
        for layer in self.layers:
            center_feats = layer(center_feats, point_feats, assign)
        return center_feats
