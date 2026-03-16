"""
Cross-frame feature fusion via stacked windowed cross-attention layers
with residual connections and feed-forward networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowedCrossAttention(nn.Module):
    """Cross-attention between two feature maps, computed within local windows.

    Window shape: full vertical extent (H) x ``window_w`` pixels horizontally.
    Pre-norm is applied inside; caller is responsible for residual connection.
    """

    def __init__(self, dim, n_heads=4, window_w=32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_w = window_w

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        """
        Args:
            query:     [B, C, H, W]
            key_value: [B, C, H, W]
        Returns:
            out: [B, C, H, W]  (no residual — caller adds it)
        """
        B, C, H, W = query.shape
        ww = self.window_w

        pad_w = (ww - W % ww) % ww
        if pad_w > 0:
            query = F.pad(query, (0, pad_w))
            key_value = F.pad(key_value, (0, pad_w))
        W_padded = W + pad_w
        n_windows = W_padded // ww

        # Reshape to windows: [B*n_win, H*ww, C]
        q = query.reshape(B, C, H, n_windows, ww)
        kv = key_value.reshape(B, C, H, n_windows, ww)

        q = q.permute(0, 3, 2, 4, 1).reshape(B * n_windows, H * ww, C)
        kv = kv.permute(0, 3, 2, 4, 1).reshape(B * n_windows, H * ww, C)

        # Pre-norm
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Multi-head attention
        Q = self.q_proj(q).reshape(-1, H * ww, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(-1, H * ww, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(-1, H * ww, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B * n_windows, H * ww, C)
        out = self.out_proj(out)

        # Reshape back: [B*n_win, H*ww, C] -> [B, C, H, W_padded]
        out = out.reshape(B, n_windows, H, ww, C)
        out = out.permute(0, 4, 2, 1, 3).reshape(B, C, H, W_padded)

        if pad_w > 0:
            out = out[:, :, :, :W]

        return out


class ConvFFN(nn.Module):
    """Channel-wise feed-forward network using 1x1 convolutions.
    Pre-norm (GroupNorm(1) ≈ LayerNorm) + residual built in."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fc1 = nn.Conv2d(dim, dim * expansion, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(dim * expansion, dim, 1)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class FusionLayer(nn.Module):
    """Single bidirectional cross-attention + FFN layer with residuals."""

    def __init__(self, dim, n_heads, window_w):
        super().__init__()
        self.cross_attn_01 = WindowedCrossAttention(dim, n_heads, window_w)
        self.cross_attn_10 = WindowedCrossAttention(dim, n_heads, window_w)
        self.ffn_0 = ConvFFN(dim)
        self.ffn_1 = ConvFFN(dim)

    def forward(self, feat_0, feat_1):
        # Cross-attention with residual
        feat_0 = feat_0 + self.cross_attn_01(feat_0, feat_1)
        feat_1 = feat_1 + self.cross_attn_10(feat_1, feat_0)
        # FFN with residual (built into ConvFFN)
        feat_0 = self.ffn_0(feat_0)
        feat_1 = self.ffn_1(feat_1)
        return feat_0, feat_1


class CrossFrameFusion(nn.Module):
    """Fuse time-embedded features from two frames using stacked
    bidirectional cross-attention layers."""

    def __init__(self, dim=128, n_heads=4, window_w=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            FusionLayer(dim, n_heads, window_w) for _ in range(num_layers)
        ])
        self.proj = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, feat_0_t, feat_1_t):
        """
        Args:
            feat_0_t: [B, C, H, W] — time-embedded frame 0 features.
            feat_1_t: [B, C, H, W] — time-embedded frame 1 features.
        Returns:
            fused: [B, C, H, W].
        """
        for layer in self.layers:
            feat_0_t, feat_1_t = layer(feat_0_t, feat_1_t)

        fused = torch.cat([feat_0_t, feat_1_t], dim=1)  # [B, 2C, H, W]
        fused = self.proj(fused)  # [B, C, H, W]
        return fused
