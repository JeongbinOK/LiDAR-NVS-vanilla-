
class RoPE3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 6 == 0, "dim must be divisible by 6 (2 per x,y,z)"
        self.dim = dim
        d = dim // 6
        self.register_buffer("freq", 1.0 / (10000 ** (torch.arange(0, d) / d)))  # (d,)

    def forward(self, rel_pos):
        # rel_pos: (N, K, 3)
        freq = self.freq  # (d,)
        # 각 축별로 sin/cos
        angles = rel_pos.unsqueeze(-1) * freq  # (N, K, 3, d)
        sin = angles.sin()
        cos = angles.cos()
        emb = torch.stack([sin, cos], dim=-1).reshape(*rel_pos.shape[:-1], -1)  # (N, K, dim)
        return emb


class LocalAttentionFlash(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.attn_drop  = attn_drop

        self.q_proj    = nn.Linear(dim, dim)
        self.kv_proj   = nn.Linear(dim, dim * 2)
        self.rope      = RoPE3D(dim)
        self.rope_proj = nn.Linear(dim, dim)
        self.out_proj  = nn.Linear(dim, dim)
        self.norm      = nn.LayerNorm(dim)

    def forward(self, feat, pos, k):
        N, C = feat.shape
        H, D = self.num_heads, self.head_dim
        if N <= 1:
            return feat
        k = min(k, N - 1)
        residual = feat

        # KNN
        dist    = torch.cdist(pos, pos)
        dist.fill_diagonal_(float('inf'))
        knn_idx = dist.topk(k, dim=-1, largest=False).indices  # (N, k)

        # RoPE
        neighbor_pos = pos[knn_idx]                          # (N, k, 3)
        rel_pos      = neighbor_pos - pos.unsqueeze(1)       # (N, k, 3)
        rope_emb     = self.rope(rel_pos)                    # (N, k, C)
        rope_bias    = self.rope_proj(rope_emb)              # (N, k, C)

        # Q, K, V
        q           = self.q_proj(feat)                      # (N, C)
        k_feat, v   = self.kv_proj(feat).chunk(2, dim=-1)
        k_feat_n    = k_feat[knn_idx] + rope_bias            # (N, k, C)
        v_n         = v[knn_idx]                             # (N, k, C)

        # flash_attn varlen: Q=1 per anchor, K=k per anchor
        q_flash = q.reshape(N, H, D).to(torch.float16)
        k_flash = k_feat_n.reshape(N * k, H, D).to(torch.float16)
        v_flash = v_n.reshape(N * k, H, D).to(torch.float16)

        cu_seqlens_q  = torch.arange(0, N + 1, dtype=torch.int32, device=feat.device)
        cu_seqlens_kv = torch.arange(0, (N + 1) * k, step=k, dtype=torch.int32, device=feat.device)

        out = flash_attn_varlen_func(
            q=q_flash, k=k_flash, v=v_flash,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=1,
            max_seqlen_k=k,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )  # (N, H, D)

        out = out.reshape(N, C).to(feat.dtype)
        out = self.out_proj(out)
        out = self.norm(residual + out)
        return out