import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None


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

class AnchorQueryCrossAttention(nn.Module):
    """Varlen cross-attention: the queries of each anchor attend to that
    anchor's variable-length K/V token set.

    The raw-seeded spherical head takes its Gaussian centres from observed
    points, so the historical probability outputs (the attention-weighted K/V
    position ``p_init`` and its per-layer head-averaged weights) are gone and
    only the attended features are returned. Flat ragged queries support two
    execution backends: ``fp32_bucket`` retains the manual fp32 score/softmax
    reference, while ``flash_varlen`` projects the complete ragged Q/K/V once
    and uses cumulative sequence lengths without an attention mask.

    Relative geometry enters the scores via Utonia-compatible ``Rotary3D``:
    every query is rotated by its own ``query_pos`` and every key by its
    ``kv_pos``, so the logits see ``kv_pos - query_pos``. Positions are not
    concatenated into K/V features.

    Varlen packing: ``anchor_ids`` (ascending) gives every K/V pair's anchor.
    Flash uses the resulting cumulative lengths directly. The fp32 reference
    processes anchors in chunks of ``chunk`` and pads each chunk to its own
    ``L_max`` with a bool mask. A fully-empty anchor (rare -- membership
    guarantees at least one row normally) keeps its query through the residual.

    Cross-attention decoder-layer semantics: the K/V memory (``kv_feat``) is fixed;
    each of ``n_layers`` layers re-projects it and updates the queries via
    ``norm(query + out_proj(attn))``.
    """

    def __init__(self, dim, num_heads=8, n_layers=1, chunk=8192, mlp_ratio=4,
                 rope_base=10.0, rope_position_scale=0.5,
                 varlen_backend="fp32_bucket"):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_layers = int(n_layers)
        self.chunk = int(chunk)
        self.varlen_backend = str(varlen_backend).lower()
        if self.varlen_backend not in ("fp32_bucket", "flash_varlen", "auto"):
            raise ValueError(
                "varlen_backend must be 'fp32_bucket', 'flash_varlen', or 'auto'"
            )
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
        # Start the residual adapter as an identity mapping.
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
                query_pos, position_scale=None, query_anchor_ids=None):
        """
        queries      : (A,K,D) fixed-count queries, or (G,D) flat ragged queries
        kv_feat      : (P, D)      embedded K/V tokens, grouped by ascending anchor
        kv_pos       : (P, 3)      K/V token positions in the *output* coordinate frame
        anchor_ids   : (P,)  long  ascending anchor index of each K/V pair, in [0, A)
        num_anchors  : int         A
        query_pos    : (A,K,3) fixed or (G,3) ragged per-query positions in the
                                   output coordinate frame. (A,3) is accepted
                                   and broadcast for fixed-count direct calls.
        query_anchor_ids: optional (G,) anchor ids required by ragged queries.
        position_scale: optional scalar or (A,) tensor.  A per-anchor tensor
                        changes only RoPE wavelengths; all learned attention
                        projections remain shared.
        return       : fixed (A,K,D) output, or ragged (G,D) output
        """
        if queries.ndim == 2:
            if query_anchor_ids is None:
                raise ValueError("flat variable queries require query_anchor_ids")
            return self.forward_variable(
                queries,
                kv_feat,
                kv_pos,
                query_anchor_ids,
                anchor_ids,
                num_anchors,
                query_pos,
                position_scale=position_scale,
            )
        if query_anchor_ids is not None:
            raise ValueError("query_anchor_ids is only valid for flat variable queries")

        A = int(num_anchors)
        H, dh = self.num_heads, self.head_dim
        D = self.dim
        device = queries.device
        dtype = queries.dtype
        K = queries.shape[1]

        if query_pos.ndim == 2 and query_pos.shape == (A, 3):
            query_pos = query_pos[:, None, :].expand(A, K, 3)
        elif query_pos.ndim != 3 or query_pos.shape != (A, K, 3):
            raise ValueError(
                "query_pos must have shape (A, 3) or (A, K, 3), "
                f"got {tuple(query_pos.shape)} for A={A}, K={K}"
            )
        query_pos_fp32 = query_pos.to(device=device, dtype=torch.float32)

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
            return queries.new_zeros((0, K, D))

        P = anchor_ids.shape[0]
        counts = torch.bincount(anchor_ids, minlength=A) if P > 0 else \
            torch.zeros(A, dtype=torch.long, device=device)          # (A,)
        starts = torch.zeros(A, dtype=torch.long, device=device)
        if A > 1:
            starts[1:] = torch.cumsum(counts, dim=0)[:-1]

        out_chunks = []
        for a0 in range(0, A, self.chunk):
            a1 = min(a0 + self.chunk, A)
            cs = a1 - a0
            c_counts = counts[a0:a1]                                  # (cs,)
            q_chunk = queries[a0:a1]                                  # (cs, K, D)
            qpos_chunk = query_pos_fp32[a0:a1]                         # (cs, K, 3)
            L_max = int(c_counts.max().item()) if cs > 0 else 0

            if L_max == 0:
                # No K/V anywhere in this chunk: residual path only. Still apply
                # each layer's LayerNorm so an empty anchor gets the exact same
                # treatment as an empty anchor inside a non-empty chunk (chunking
                # must never change semantics).
                q_cur = q_chunk
                for li in range(self.n_layers):
                    q_cur = self.norm[li](q_cur)
                    q_cur = self.norm_ffn[li](q_cur + self.ffn[li](q_cur))
                out_chunks.append(q_cur)
                continue

            # contiguous pair range for this chunk (anchor_ids ascending)
            p_start = int(starts[a0].item())
            p_end = int(starts[a1].item()) if a1 < A else P
            cp = torch.arange(p_start, p_end, device=device)         # (Pc,)
            local_anchor = anchor_ids[cp] - a0                       # (Pc,)  in [0, cs)
            within = cp - starts[anchor_ids[cp]]                     # (Pc,)  rank in anchor

            padded_feat = torch.zeros(cs, L_max, D, device=device, dtype=dtype)
            # fp32 positions: they feed rotary angles, where a low-precision
            # dtype would cost centimeters at range.
            padded_pos = torch.zeros(cs, L_max, 3, device=device, dtype=torch.float32)
            mask = torch.zeros(cs, L_max, device=device, dtype=torch.bool)
            padded_feat[local_anchor, within] = kv_feat[cp]
            padded_pos[local_anchor, within] = kv_pos[cp].float()
            mask[local_anchor, within] = True
            has_kv = c_counts > 0                                    # (cs,)
            # True = attend. Slot 0 is force-enabled so an empty anchor
            # inside a non-empty chunk attends the zero pad instead of
            # producing an all--inf softmax row (NaN); its delta is zeroed
            # below, which reproduces the residual-only semantics exactly.
            attn_mask = mask.clone()
            attn_mask[:, 0] = True
            neg_mask = ~attn_mask.view(cs, 1, 1, L_max)

            # Rotary angles are per-position, shared by every layer.
            chunk_scale = (
                anchor_scale if anchor_scale is None or anchor_scale.ndim == 0
                else anchor_scale[a0:a1]
            )
            q_ang = self.rope.angles(                                # (cs,K,3,axis_pairs)
                qpos_chunk, position_scale=chunk_scale
            )
            k_ang = self.rope.angles(                                # (cs, L, 3, axis_pairs)
                padded_pos, position_scale=chunk_scale
            )

            q_cur = q_chunk
            for li in range(self.n_layers):
                qh = self.q_proj[li](q_cur).reshape(cs, K, H, dh)
                kf, vf = self.kv_proj[li](padded_feat).chunk(2, dim=-1)
                kf = kf.reshape(cs, L_max, H, dh)
                vf = vf.reshape(cs, L_max, H, dh)

                # Evaluate rotary Q/K scores and softmax in fp32.
                q_rot = self.rope.rotate(qh.float(), q_ang[:, :, None, :, :])
                k_rot = self.rope.rotate(kf.float(), k_ang[:, :, None, :, :])
                scores = torch.einsum('ckhd,clhd->chkl', q_rot, k_rot) * self.scale
                scores = scores.masked_fill(neg_mask, float('-inf'))
                probs = torch.softmax(scores, dim=-1)               # (cs,H,K,L) fp32
                attn = torch.einsum('chkl,clhd->ckhd', probs, vf.float())
                attn = attn.reshape(cs, K, D).to(dtype)
                delta = self.out_proj[li](attn)
                # L_a == 0 anchors: kill the attention branch (residual path only)
                delta = torch.where(has_kv.view(cs, 1, 1), delta, torch.zeros_like(delta))
                q_cur = self.norm[li](q_cur + delta)
                # FFN sublayer (applies to every query, KV-independent; empty
                # anchors go through the same transform as in the L_max==0 path).
                q_cur = self.norm_ffn[li](q_cur + self.ffn[li](q_cur))

            out_chunks.append(q_cur)

        return torch.cat(out_chunks, dim=0)                         # (A, K, D)

    def forward_variable(
        self,
        queries,
        kv_feat,
        kv_pos,
        query_anchor_ids,
        kv_anchor_ids,
        num_anchors,
        query_pos,
        position_scale=None,
    ):
        """Cross-attend a variable number of flat queries per anchor.

        ``forward`` above remains the efficient fixed-``K`` kernel.  This method
        implements a ragged Q axis without padding every anchor to the global
        maximum: anchors are grouped by their *actual* query count, each group is
        passed through the shared fixed-``K`` kernel, and the results are restored
        to the original flat query order.  Learned projections are shared across
        every group; query-count grouping is only execution metadata.

        queries/query_pos       : (G,D) / (G,3), anchor-major flat queries
        query_anchor_ids        : (G,), ascending anchor id for each query
        kv_feat/kv_pos           : (P,D) / (P,3), anchor-major flat memory
        kv_anchor_ids            : (P,), ascending anchor id for each memory row
        position_scale           : optional scalar or (A,) per-anchor RoPE scale
        return                   : (G,D) output in the original flat query order
        """
        A = int(num_anchors)
        G = int(queries.shape[0])
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(
                f"variable queries must have shape (G,{self.dim}), got {tuple(queries.shape)}"
            )
        if query_pos.shape != (G, 3):
            raise ValueError(
                f"variable query_pos must have shape ({G},3), got {tuple(query_pos.shape)}"
            )
        if query_anchor_ids.shape != (G,):
            raise ValueError("query_anchor_ids must provide one anchor id per query")
        if kv_anchor_ids.shape != (kv_feat.shape[0],):
            raise ValueError("kv_anchor_ids must provide one anchor id per K/V row")
        if G == 0:
            return queries.new_zeros((0, self.dim))
        if A <= 0:
            raise ValueError("num_anchors must be positive when variable queries are non-empty")
        if (
            (query_anchor_ids < 0).any()
            or (query_anchor_ids >= A).any()
            or (query_anchor_ids[1:] < query_anchor_ids[:-1]).any()
        ):
            raise ValueError("query_anchor_ids must be ascending and lie in [0, A)")
        if kv_anchor_ids.numel() > 0 and (
            (kv_anchor_ids < 0).any()
            or (kv_anchor_ids >= A).any()
            or (kv_anchor_ids[1:] < kv_anchor_ids[:-1]).any()
        ):
            raise ValueError("kv_anchor_ids must be ascending and lie in [0, A)")

        q_counts = torch.bincount(query_anchor_ids, minlength=A)
        if (q_counts == 0).any():
            raise ValueError("every occupied anchor must own at least one evidence query")
        q_starts = torch.cumsum(q_counts, dim=0) - q_counts

        anchor_scale = None
        if position_scale is not None:
            anchor_scale = torch.as_tensor(
                position_scale, device=queries.device, dtype=torch.float32
            )
            if anchor_scale.ndim > 1 or (
                anchor_scale.ndim == 1 and anchor_scale.shape[0] != A
            ):
                raise ValueError(
                    "position_scale must be a scalar or have one value per "
                    f"anchor ({A}), got {tuple(anchor_scale.shape)}"
                )

        use_flash = (
            queries.is_cuda
            and flash_attn is not None
            and self.varlen_backend in ("flash_varlen", "auto")
        )
        if (
            self.varlen_backend == "flash_varlen"
            and queries.is_cuda
            and flash_attn is None
        ):
            raise RuntimeError(
                "flash_varlen was requested but flash_attn is unavailable"
            )
        if use_flash:
            kv_counts = torch.bincount(kv_anchor_ids, minlength=A)
            # The normal spherical membership route gives every occupied anchor
            # at least one memory row. Retain the exact residual-only behavior of
            # the reference path for synthetic/degenerate empty-memory anchors.
            if not bool((kv_counts == 0).any().item()):
                return self._forward_variable_flash(
                    queries,
                    kv_feat,
                    kv_pos,
                    query_anchor_ids,
                    kv_anchor_ids,
                    q_counts,
                    kv_counts,
                    query_pos,
                    anchor_scale,
                )

        out_parts = []
        row_parts = []
        # The observed support distribution has a small integer tail (rather
        # than one large dense bank), so grouping by count avoids both a global
        # Q_max pad and K/V duplication for every query.
        for q_len in torch.unique(q_counts, sorted=True).tolist():
            q_len = int(q_len)
            anchors = (q_counts == q_len).nonzero(as_tuple=True)[0]
            group_size = int(anchors.numel())
            if group_size == 0:
                continue

            within = torch.arange(q_len, device=queries.device).view(1, -1)
            q_rows = q_starts[anchors].view(-1, 1) + within
            q_group = queries[q_rows]
            qpos_group = query_pos[q_rows]

            anchor_remap = torch.full(
                (A,), -1, dtype=torch.long, device=queries.device
            )
            anchor_remap[anchors] = torch.arange(group_size, device=queries.device)
            if kv_anchor_ids.numel() > 0:
                local_kv_anchor = anchor_remap[kv_anchor_ids]
                keep = local_kv_anchor >= 0
                group_kv_feat = kv_feat[keep]
                group_kv_pos = kv_pos[keep]
                group_kv_anchor = local_kv_anchor[keep]
            else:
                group_kv_feat = kv_feat
                group_kv_pos = kv_pos
                group_kv_anchor = kv_anchor_ids

            group_scale = (
                anchor_scale
                if anchor_scale is None or anchor_scale.ndim == 0
                else anchor_scale[anchors]
            )
            group_out = self.forward(
                q_group,
                group_kv_feat,
                group_kv_pos,
                group_kv_anchor,
                group_size,
                qpos_group,
                position_scale=group_scale,
            )
            out_parts.append(group_out.reshape(-1, self.dim))
            row_parts.append(q_rows.reshape(-1))

        packed_rows = torch.cat(row_parts)
        restore = torch.argsort(packed_rows)
        return torch.cat(out_parts, dim=0)[restore]

    def _forward_variable_flash(
        self,
        queries,
        kv_feat,
        kv_pos,
        query_anchor_ids,
        kv_anchor_ids,
        q_counts,
        kv_counts,
        query_pos,
        anchor_scale,
    ):
        """True varlen CUDA path with one Q/K sequence pair per anchor."""
        q_scale = (
            anchor_scale
            if anchor_scale is None or anchor_scale.ndim == 0
            else anchor_scale[query_anchor_ids]
        )
        k_scale = (
            anchor_scale
            if anchor_scale is None or anchor_scale.ndim == 0
            else anchor_scale[kv_anchor_ids]
        )
        q_angles = self.rope.angles(
            query_pos.to(torch.float32), position_scale=q_scale
        )
        k_angles = self.rope.angles(
            kv_pos.to(torch.float32), position_scale=k_scale
        )
        cu_seqlens_q = nn.functional.pad(
            torch.cumsum(q_counts, dim=0, dtype=torch.int32), (1, 0)
        )
        cu_seqlens_k = nn.functional.pad(
            torch.cumsum(kv_counts, dim=0, dtype=torch.int32), (1, 0)
        )
        max_seqlen_q = int(q_counts.max().item())
        max_seqlen_k = int(kv_counts.max().item())

        H, dh = self.num_heads, self.head_dim
        q_cur = queries
        for li in range(self.n_layers):
            qh = self.q_proj[li](q_cur).reshape(-1, H, dh)
            kh, vh = self.kv_proj[li](kv_feat).chunk(2, dim=-1)
            kh = kh.reshape(-1, H, dh)
            vh = vh.reshape(-1, H, dh)
            q_rot = self.rope.rotate(
                qh.float(), q_angles[:, None, :, :]
            )
            k_rot = self.rope.rotate(
                kh.float(), k_angles[:, None, :, :]
            )
            attended = flash_attn.flash_attn_varlen_func(
                q=q_rot.to(torch.bfloat16),
                k=k_rot.to(torch.bfloat16),
                v=vh.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            attended = attended.reshape(-1, self.dim).to(q_cur.dtype)
            q_cur = self.norm[li](
                q_cur + self.out_proj[li](attended)
            )
            q_cur = self.norm_ffn[li](
                q_cur + self.ffn[li](q_cur)
            )
        return q_cur


class AnchorQuerySelfAttention(nn.Module):
    """Per-anchor ragged query self-attention after cross refinement.

    Every query attends only to the sibling evidence queries carrying the same
    ``query_anchor_id``. The block uses ordinary pre-norm residuals without a
    gate or LayerScale. CUDA can use one true varlen FlashAttention launch;
    the reference path buckets anchors by exact query count and keeps fp32
    scores/softmax.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        n_layers=1,
        mlp_ratio=4,
        rope_base=10.0,
        rope_position_scale=0.5,
        varlen_backend="fp32_bucket",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least one")
        self.varlen_backend = str(varlen_backend).lower()
        if self.varlen_backend not in ("fp32_bucket", "flash_varlen", "auto"):
            raise ValueError(
                "varlen_backend must be 'fp32_bucket', 'flash_varlen', or 'auto'"
            )
        self.rope = Rotary3D(
            self.head_dim,
            base=rope_base,
            position_scale=rope_position_scale,
        )

        hidden = int(self.dim * mlp_ratio)
        self.norm1 = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        self.qkv = nn.ModuleList([
            nn.Linear(self.dim, self.dim * 3) for _ in range(self.n_layers)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.norm2 = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.dim),
            )
            for _ in range(self.n_layers)
        ])

    def _fp32_bucket(self, q, k, v, counts):
        starts = torch.cumsum(counts, dim=0) - counts
        attended = q.new_zeros(q.shape)
        for q_len in torch.unique(counts, sorted=True).tolist():
            q_len = int(q_len)
            anchors = (counts == q_len).nonzero(as_tuple=True)[0]
            rows = starts[anchors, None] + torch.arange(
                q_len, device=q.device
            ).view(1, -1)
            q_group = q[rows]
            k_group = k[rows]
            v_group = v[rows]
            scores = torch.einsum(
                "glhd,gshd->ghls", q_group.float(), k_group.float()
            ) * self.scale
            probability = torch.softmax(scores, dim=-1)
            group_out = torch.einsum(
                "ghls,gshd->glhd", probability, v_group.float()
            ).to(q.dtype)
            attended = attended.index_copy(
                0,
                rows.reshape(-1),
                group_out.reshape(-1, self.num_heads, self.head_dim),
            )
        return attended

    def _flash_varlen(self, q, k, v, counts):
        cu_seqlens = nn.functional.pad(
            torch.cumsum(counts, dim=0, dtype=torch.int32), (1, 0)
        )
        qkv = torch.stack([q, k, v], dim=1).to(torch.bfloat16)
        return flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen=int(counts.max().item()),
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        ).to(q.dtype)

    def forward(
        self,
        queries,
        query_pos,
        query_anchor_ids,
        num_anchors,
        position_scale=None,
    ):
        A = int(num_anchors)
        G = int(queries.shape[0])
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(
                f"queries must have shape (G,{self.dim}), got {tuple(queries.shape)}"
            )
        if query_pos.shape != (G, 3):
            raise ValueError(
                f"query_pos must have shape ({G},3), got {tuple(query_pos.shape)}"
            )
        if query_anchor_ids.shape != (G,):
            raise ValueError("query_anchor_ids must provide one anchor id per query")
        if G == 0:
            return queries
        if A <= 0:
            raise ValueError("num_anchors must be positive for non-empty queries")
        if (
            (query_anchor_ids < 0).any()
            or (query_anchor_ids >= A).any()
            or (query_anchor_ids[1:] < query_anchor_ids[:-1]).any()
        ):
            raise ValueError(
                "query_anchor_ids must be ascending and lie in [0, A)"
            )

        counts = torch.bincount(query_anchor_ids, minlength=A)
        if (counts == 0).any():
            raise ValueError("every occupied anchor must own at least one query")
        anchor_scale = None
        if position_scale is not None:
            anchor_scale = torch.as_tensor(
                position_scale, device=queries.device, dtype=torch.float32
            )
            if anchor_scale.ndim > 1 or (
                anchor_scale.ndim == 1 and anchor_scale.shape[0] != A
            ):
                raise ValueError(
                    "position_scale must be a scalar or have one value per "
                    f"anchor ({A}), got {tuple(anchor_scale.shape)}"
                )
        row_scale = (
            anchor_scale
            if anchor_scale is None or anchor_scale.ndim == 0
            else anchor_scale[query_anchor_ids]
        )
        angles = self.rope.angles(
            query_pos.to(torch.float32), position_scale=row_scale
        )

        use_flash = (
            queries.is_cuda
            and flash_attn is not None
            and self.varlen_backend in ("flash_varlen", "auto")
        )
        if (
            self.varlen_backend == "flash_varlen"
            and queries.is_cuda
            and flash_attn is None
        ):
            raise RuntimeError(
                "flash_varlen was requested but flash_attn is unavailable"
            )

        x = queries
        for li in range(self.n_layers):
            normalized = self.norm1[li](x)
            q, k, v = self.qkv[li](normalized).reshape(
                G, 3, self.num_heads, self.head_dim
            ).unbind(dim=1)
            q = self.rope.rotate(q.float(), angles[:, None, :, :])
            k = self.rope.rotate(k.float(), angles[:, None, :, :])
            if use_flash:
                attended = self._flash_varlen(q, k, v, counts)
            else:
                attended = self._fp32_bucket(
                    q, k, v, counts
                )
            attended = attended.reshape(G, self.dim).to(x.dtype)
            x = x + self.out_proj[li](attended)
            x = x + self.ffn[li](self.norm2[li](x))
        return x
