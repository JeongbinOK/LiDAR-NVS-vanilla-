import torch
import torch.nn as nn


class Rotary3D(nn.Module):
    """Utonia-compatible 3D RoPE with an explicit metric position scale.

    This uses the exact channel layout and frequency schedule of
    :class:`utonia.model.Point3DRoPE`: split one attention head into equal
    x/y/z chunks, split each axis chunk into two halves, and rotate matching
    half-channels with ``base ** (-2j / axis_dim)``. Therefore ``head_dim``
    must be divisible by six.

    ``position_scale`` is the only attention-specific geometry knob. For
    metric input positions, axis frequency ``j`` has wavelength
    ``2*pi / (position_scale * inv_freq[j])`` metres. Changing it preserves
    Utonia's RoPE operator while adapting its physical wavelengths to a local
    grid neighbourhood, an object-local box, or a larger spherical cell.
    """

    def __init__(self, head_dim, base=10.0, position_scale=1.0):
        super().__init__()
        head_dim = int(head_dim)
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim must be divisible by 6 for Utonia 3D RoPE, got {head_dim}"
            )
        base = float(base)
        position_scale = float(position_scale)
        if base <= 0.0:
            raise ValueError("base must be positive")
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = base
        self.position_scale = position_scale
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, self.chunk_dim, 2, dtype=torch.float32)
                / self.chunk_dim
            )
        )
        # This is deterministic geometry derived from config, not learned state.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def angles(self, pos, position_scale=None):
        """``(..., 3)`` positions -> ``(..., 3, axis_dim/2)`` fp32 angles.

        ``position_scale`` may be a scalar or match a prefix of ``pos``'s
        leading dimensions.  The latter lets heterogeneous anchor groups share
        Q/K/V weights while using coordinate-frame-specific wavelengths.
        """
        if pos.shape[-1] != 3:
            raise ValueError(f"RoPE positions must end in xyz, got {tuple(pos.shape)}")
        scale = torch.as_tensor(
            self.position_scale if position_scale is None else position_scale,
            device=pos.device, dtype=torch.float32,
        )
        leading_ndim = pos.ndim - 1
        if scale.ndim > leading_ndim:
            raise ValueError(
                "position_scale has too many dimensions for positions: "
                f"scale={tuple(scale.shape)}, pos={tuple(pos.shape)}"
            )
        scale = scale.reshape(
            *scale.shape, *([1] * (leading_ndim - scale.ndim))
        )
        return pos.float().mul(scale.unsqueeze(-1)).unsqueeze(-1) * self.inv_freq

    def rotate(self, x, angles):
        """Apply Utonia's per-axis half rotation to ``x``.

        ``angles`` must broadcast to ``x`` reshaped as
        ``(..., 3, chunk_dim/2)``; callers insert singleton head/query dims.
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"RoPE feature dim must be {self.head_dim}, got {x.shape[-1]}"
            )
        x_axis = x.reshape(*x.shape[:-1], 3, self.chunk_dim)
        half = self.chunk_dim // 2
        x1 = x_axis[..., :half]
        x2 = x_axis[..., half:]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotated = torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )
        return rotated.flatten(-2)

    def forward(self, q, k, q_pos, k_pos=None, position_scale=None):
        """Rotate q/k at possibly different positions (cross-attention safe)."""
        if k_pos is None:
            k_pos = q_pos
        q_angles = self.angles(q_pos, position_scale=position_scale)
        k_angles = self.angles(k_pos, position_scale=position_scale)
        for _ in range(q.ndim - q_pos.ndim):
            q_angles = q_angles.unsqueeze(-3)
        for _ in range(k.ndim - k_pos.ndim):
            k_angles = k_angles.unsqueeze(-3)
        return self.rotate(q, q_angles), self.rotate(k, k_angles)


class AnchorQueryCrossAttention(nn.Module):
    """Manual (non-flash) varlen cross-attention: K learnable queries per anchor
    attend to that anchor's variable-length K/V token set, and additionally return
    a probability-weighted position ``p_init`` per query.

    Why manual instead of flash_attn: the spherical-query head needs the attention
    *probabilities* to form ``p_init = sum_l w_bar[k,l] * kv_pos[l]`` (a surface
    interpolation seed), and flash_attn does not expose them. Scores / softmax /
    weighted-position are computed in **fp32** (NaN-collapse history under bf16),
    then cast back to the input dtype.

    Relative geometry enters the scores via Utonia-compatible ``Rotary3D``:
    every query is rotated by its anchor's position (``fallback_pos``) and every
    key by its ``kv_pos``, so the logits see only ``kv_pos - anchor_pos``.
    Positions are not concatenated into K/V features; ``fallback_pos`` therefore
    serves double duty (query rotation origin + empty-anchor ``p_init``).

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

    def __init__(self, dim, num_heads=8, n_layers=1, chunk=8192, mlp_ratio=4,
                 rope_base=10.0, rope_position_scale=0.5):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_layers = int(n_layers)
        self.chunk = int(chunk)
        self.rope = Rotary3D(
            self.head_dim, base=rope_base, position_scale=rope_position_scale
        )

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

    def forward(self, queries, kv_feat, kv_pos, anchor_ids, num_anchors,
                fallback_pos, position_scale=None):
        """
        queries      : (A, K, D)  per-anchor query bank (learnable + anchor embed)
        kv_feat      : (P, D)      embedded K/V tokens, grouped by ascending anchor
        kv_pos       : (P, 3)      K/V token positions in the *output* coordinate frame
        anchor_ids   : (P,)  long  ascending anchor index of each K/V pair, in [0, A)
        num_anchors  : int         A
        fallback_pos : (A, 3)      anchor position; rotary origin for the anchor's
                                   queries, and p_init when L_a == 0
        position_scale: optional scalar or (A,) tensor.  A per-anchor tensor
                        changes only RoPE wavelengths; all learned attention
                        projections remain shared.
        return       : (out (A, K, D), p_init (A, K, 3))
        """
        A = int(num_anchors)
        H, dh = self.num_heads, self.head_dim
        D = self.dim
        device = queries.device
        dtype = queries.dtype
        K = queries.shape[1]

        anchor_scale = None
        if position_scale is not None:
            anchor_scale = torch.as_tensor(
                position_scale, device=device, dtype=torch.float32
            )
            if anchor_scale.ndim > 1 or (
                anchor_scale.ndim == 1 and anchor_scale.shape[0] != A
            ):
                raise ValueError(
                    "position_scale must be a scalar or have one value per "
                    f"anchor ({A}), got {tuple(anchor_scale.shape)}"
                )
            # Do not reduce a CUDA scale tensor here: that would introduce a
            # host synchronization in every attention forward. Configuration
            # owners validate their scalar BG/FG values at construction time.

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
            # fp32 positions: they feed rotary angles and the p_init weighted
            # sum, where a low-precision dtype would cost centimeters at range.
            padded_pos = torch.zeros(cs, L_max, 3, device=device, dtype=torch.float32)
            mask = torch.zeros(cs, L_max, device=device, dtype=torch.bool)
            padded_feat[local_anchor, within] = kv_feat[cp]
            padded_pos[local_anchor, within] = kv_pos[cp].float()
            mask[local_anchor, within] = True
            has_kv = c_counts > 0                                    # (cs,)
            neg_mask = ~mask.view(cs, 1, 1, L_max)                   # broadcast (cs,H,K,L)

            # Rotary angles are per-position, shared by every layer.
            chunk_scale = (
                anchor_scale if anchor_scale is None or anchor_scale.ndim == 0
                else anchor_scale[a0:a1]
            )
            q_ang = self.rope.angles(                                # (cs, 3, axis_pairs)
                fallback_pos[a0:a1], position_scale=chunk_scale
            )
            k_ang = self.rope.angles(                                # (cs, L, 3, axis_pairs)
                padded_pos, position_scale=chunk_scale
            )

            q_cur = q_chunk
            last_wbar = None
            for li in range(self.n_layers):
                qh = self.q_proj[li](q_cur).reshape(cs, K, H, dh)
                kf, vf = self.kv_proj[li](padded_feat).chunk(2, dim=-1)
                kf = kf.reshape(cs, L_max, H, dh)
                vf = vf.reshape(cs, L_max, H, dh)

                # fp32 scores / softmax (NaN-collapse guard), rotary q/k
                q_rot = self.rope.rotate(qh.float(), q_ang[:, None, None, :, :])
                k_rot = self.rope.rotate(kf.float(), k_ang[:, :, None, :, :])
                scores = torch.einsum('ckhd,clhd->chkl', q_rot, k_rot) * self.scale
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
            pinit_c = torch.einsum('ckl,cld->ckd', last_wbar, padded_pos).to(dtype)
            if (~has_kv).any():
                fb = fb_chunk.unsqueeze(1).expand(cs, K, 3)
                pinit_c = torch.where(has_kv.view(cs, 1, 1), pinit_c, fb)
            out_chunks.append(q_cur)
            pinit_chunks.append(pinit_c)

        out = torch.cat(out_chunks, dim=0)                          # (A, K, D)
        p_init = torch.cat(pinit_chunks, dim=0)                     # (A, K, 3)
        return out, p_init
