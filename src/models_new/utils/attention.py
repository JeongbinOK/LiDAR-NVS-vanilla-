import torch.nn as nn
import torch
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
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
        if flash_attn_varlen_func is None:
            raise ImportError("flash_attn is required by LocalAttentionFlash.")
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


class AnchorQueryCrossAttention(nn.Module):
    """Manual (non-flash) varlen cross-attention: K learnable queries per anchor
    attend to that anchor's variable-length K/V token set, and additionally return
    a probability-weighted position ``p_init`` per query.

    Why manual instead of flash_attn: the spherical-query head needs the attention
    *probabilities* to form ``p_init = sum_l w_bar[k,l] * kv_pos[l]`` (a surface
    interpolation seed), and flash_attn does not expose them. Scores / softmax /
    weighted-position are computed in **fp32** (NaN-collapse history under bf16),
    then cast back to the input dtype.

    varlen packing: ``anchor_ids`` (ascending) gives every K/V pair's anchor; per
    anchor we derive (start, len) via bincount+cumsum and process anchors in chunks
    of ``chunk`` (padding each chunk to its own ``L_max`` with a bool mask). Padded
    slots are ``-inf`` masked so their softmax weight is exactly 0; a fully-empty
    anchor (L_a == 0, rare -- P0 guarantees >=1 in the normal flow) keeps its query
    through the residual path only and takes ``p_init = fallback_pos``.

    Cross-attention decoder-layer semantics: the K/V memory (``kv_feat``) is fixed;
    each of ``n_layers`` layers re-projects it and updates the queries via
    ``norm(query + out_proj(attn))``. ``p_init`` uses the **last** layer's
    head-averaged probabilities.
    """

    def __init__(self, dim, num_heads=8, n_layers=1, chunk=8192, mlp_ratio=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_layers = int(n_layers)
        self.chunk = int(chunk)

        self.q_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.kv_proj = nn.ModuleList([nn.Linear(dim, dim * 2) for _ in range(self.n_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])

        # Position-wise FFN sublayer per layer, completing each decoder block
        # (cross-attn sublayer -> FFN sublayer, both post-norm residual). The
        # second linear is zero-init so the FFN branch contributes 0 at init:
        # training starts from the pre-FFN behaviour and learns the delta (same
        # zero-init-residual trick as UtoniaResidualAdapter; guards the NaN-collapse
        # history from injecting random activations here).
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            for _ in range(self.n_layers)
        ])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        for ff in self.ffn:
            nn.init.zeros_(ff[-1].weight)
            nn.init.zeros_(ff[-1].bias)

    def forward(self, queries, kv_feat, kv_pos, anchor_ids, num_anchors, fallback_pos):
        """
        queries      : (A, K, D)  per-anchor query bank (learnable + anchor embed)
        kv_feat      : (P, D)      embedded K/V tokens, grouped by ascending anchor
        kv_pos       : (P, 3)      K/V token positions in the *output* coordinate frame
        anchor_ids   : (P,)  long  ascending anchor index of each K/V pair, in [0, A)
        num_anchors  : int         A
        fallback_pos : (A, 3)      anchor position used for p_init when L_a == 0
        return       : (out (A, K, D), p_init (A, K, 3))
        """
        A = int(num_anchors)
        H, dh = self.num_heads, self.head_dim
        D = self.dim
        device = queries.device
        dtype = queries.dtype
        K = queries.shape[1]

        if A == 0:
            return (queries.new_zeros((0, K, D)), queries.new_zeros((0, K, 3)))

        P = anchor_ids.shape[0]
        counts = torch.bincount(anchor_ids, minlength=A) if P > 0 else \
            torch.zeros(A, dtype=torch.long, device=device)          # (A,)
        starts = torch.zeros(A, dtype=torch.long, device=device)
        if A > 1:
            starts[1:] = torch.cumsum(counts, dim=0)[:-1]

        out_chunks = []
        pinit_chunks = []
        for a0 in range(0, A, self.chunk):
            a1 = min(a0 + self.chunk, A)
            cs = a1 - a0
            c_counts = counts[a0:a1]                                  # (cs,)
            q_chunk = queries[a0:a1]                                  # (cs, K, D)
            fb_chunk = fallback_pos[a0:a1].to(dtype)                  # (cs, 3)
            L_max = int(c_counts.max().item()) if cs > 0 else 0

            if L_max == 0:
                # No K/V anywhere in this chunk: residual path only. Still apply
                # each layer's LayerNorm so an empty anchor gets the exact same
                # treatment as an empty anchor inside a non-empty chunk (chunking
                # must never change semantics). p_init falls back to the anchor
                # position.
                q_cur = q_chunk
                for li in range(self.n_layers):
                    q_cur = self.norm[li](q_cur)
                    q_cur = self.norm_ffn[li](q_cur + self.ffn[li](q_cur))
                out_chunks.append(q_cur)
                pinit_chunks.append(fb_chunk.unsqueeze(1).expand(cs, K, 3))
                continue

            # contiguous pair range for this chunk (anchor_ids ascending)
            p_start = int(starts[a0].item())
            p_end = int(starts[a1].item()) if a1 < A else P
            cp = torch.arange(p_start, p_end, device=device)         # (Pc,)
            local_anchor = anchor_ids[cp] - a0                       # (Pc,)  in [0, cs)
            within = cp - starts[anchor_ids[cp]]                     # (Pc,)  rank in anchor

            padded_feat = torch.zeros(cs, L_max, D, device=device, dtype=dtype)
            padded_pos = torch.zeros(cs, L_max, 3, device=device, dtype=dtype)
            mask = torch.zeros(cs, L_max, device=device, dtype=torch.bool)
            padded_feat[local_anchor, within] = kv_feat[cp]
            padded_pos[local_anchor, within] = kv_pos[cp]
            mask[local_anchor, within] = True
            has_kv = c_counts > 0                                    # (cs,)
            neg_mask = ~mask.view(cs, 1, 1, L_max)                   # broadcast (cs,H,K,L)

            q_cur = q_chunk
            last_wbar = None
            for li in range(self.n_layers):
                qh = self.q_proj[li](q_cur).reshape(cs, K, H, dh)
                kf, vf = self.kv_proj[li](padded_feat).chunk(2, dim=-1)
                kf = kf.reshape(cs, L_max, H, dh)
                vf = vf.reshape(cs, L_max, H, dh)

                # fp32 scores / softmax (NaN-collapse guard)
                scores = torch.einsum('ckhd,clhd->chkl', qh.float(), kf.float()) * self.scale
                scores = scores.masked_fill(neg_mask, float('-inf'))
                probs = torch.softmax(scores, dim=-1)               # (cs,H,K,L) fp32
                probs = torch.nan_to_num(probs, nan=0.0)            # fully-masked rows -> 0

                attn = torch.einsum('chkl,clhd->ckhd', probs, vf.float())  # (cs,K,H,dh)
                attn = attn.reshape(cs, K, D).to(dtype)
                delta = self.out_proj[li](attn)
                # L_a == 0 anchors: kill the attention branch (residual path only)
                delta = torch.where(has_kv.view(cs, 1, 1), delta, torch.zeros_like(delta))
                q_cur = self.norm[li](q_cur + delta)
                # FFN sublayer (applies to every query, KV-independent; empty
                # anchors go through the same transform as in the L_max==0 path).
                q_cur = self.norm_ffn[li](q_cur + self.ffn[li](q_cur))
                last_wbar = probs.mean(dim=1)                        # (cs,K,L) fp32

            # p_init = last-layer head-averaged weights @ kv positions (fp32)
            pinit_c = torch.einsum('ckl,cld->ckd', last_wbar, padded_pos.float()).to(dtype)
            if (~has_kv).any():
                fb = fb_chunk.unsqueeze(1).expand(cs, K, 3)
                pinit_c = torch.where(has_kv.view(cs, 1, 1), pinit_c, fb)
            out_chunks.append(q_cur)
            pinit_chunks.append(pinit_c)

        out = torch.cat(out_chunks, dim=0)                          # (A, K, D)
        p_init = torch.cat(pinit_chunks, dim=0)                     # (A, K, 3)
        return out, p_init
