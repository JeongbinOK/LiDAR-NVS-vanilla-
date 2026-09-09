"""Temporal cross-attention and correspondence matchers."""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...utils.attention import Rotary3D
from .common import SinusoidalScalarEncoder, _cfg_get

try:
    import flash_attn
except ImportError:  # CPU/unit-test fallback remains available.
    flash_attn = None

class ParallelBidirectionalCrossAttention(nn.Module):
    """Legacy v1 full other-frame refinement with synchronous endpoint updates.

    Each layer forms Q/K/V for *both* directions from one immutable input
    snapshot, executes the ragged endpoint pairs, and only then commits the
    residual/FFN update.  Attention probabilities are never interpreted as
    correspondence and never touch coordinates.

    Normalized source time conditions Q/K.  V and the residual stream remain
    feature-only so time does not become a shortcut in static GS attributes.
    Ref-frame position enters only through 3D RoPE.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(_cfg_get(cfg, "num_heads", 8))
        self.n_layers = int(_cfg_get(cfg, "layers", 1))
        self.mlp_ratio = float(_cfg_get(cfg, "mlp_ratio", 4.0))
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        if self.dim % self.num_heads != 0:
            raise ValueError("dynamic token dim must be divisible by num_heads")
        if self.n_layers < 1:
            raise ValueError("dynamic temporal layers must be at least one")
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "dynamic attention head dim must be divisible by 6 for 3D RoPE"
            )
        self.scale = self.head_dim ** -0.5

        time_dim = int(_cfg_get(cfg, "time_embedding_dim", 64))
        time_frequencies = int(_cfg_get(cfg, "time_frequencies", 8))
        self.time_encoder = SinusoidalScalarEncoder(time_dim, time_frequencies)
        rope_position_scale = _cfg_get(cfg, "rope_position_scale", None)
        if rope_position_scale is None:
            # Legacy V1 configs specified a half-period range instead of the
            # actual radians/metre multiplier. Preserve that exact semantics.
            range_m = float(_cfg_get(cfg, "rope_range_m", 110.0))
            rope_position_scale = math.pi / range_m
        self.rope = Rotary3D(
            self.head_dim,
            base=float(_cfg_get(cfg, "rope_base", 10.0)),
            position_scale=float(rope_position_scale),
        )

        # Legacy V1 is the only stack that conditions Q and K on time per layer
        # rather than adding one time projection to the token before layer one.
        self._build_transformer_layers(cfg, time_projection_dim=time_dim)

    def _build_qk_norm(self, enabled: bool) -> None:
        """Per-layer RMSNorm on Q and K, or the historical unnormalized pair.

        Nothing else bounds ``|q|`` or ``|k|``: ``input_norm`` sits before the
        projection, so the projection gain passes through unchecked and the
        logit grows as its square. RMSNorm rather than LayerNorm because
        LayerNorm's mean subtraction would mix Rotary3D's x/y/z chunks.
        """
        self.qk_norm = bool(enabled)
        if not self.qk_norm:
            self.q_norm = None
            self.k_norm = None
            return
        self.q_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])

    def _build_transformer_layers(self, cfg, *, time_projection_dim=None):
        """The per-layer stack every temporal variant shares.

        Input norm, the four attention projections, the LayerScale gate, and a
        zero-initialized FFN, all built in the one order the saved checkpoints
        expect. ``time_projection_dim`` additionally builds the legacy V1
        per-layer ``time_to_q``/``time_to_k`` pair. Variants differ only in how
        the scores are formed, never in which parameters exist here.
        """
        hidden = int(self.dim * self.mlp_ratio)
        self.input_norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        if time_projection_dim is not None:
            self.time_to_q = nn.ModuleList([
                nn.Linear(int(time_projection_dim), self.dim, bias=False)
                for _ in range(self.n_layers)
            ])
            self.time_to_k = nn.ModuleList([
                nn.Linear(int(time_projection_dim), self.dim, bias=False)
                for _ in range(self.n_layers)
            ])
        self.q_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        layer_scale_init = float(_cfg_get(cfg, "layer_scale_init", 0.1))
        if not 0.0 < layer_scale_init <= 1.0:
            raise ValueError("dynamic temporal layer_scale_init must be in (0,1]")
        self.cross_layer_scale = nn.ParameterList([
            nn.Parameter(torch.full((self.dim,), layer_scale_init))
            for _ in range(self.n_layers)
        ])
        self.ffn_norm = nn.ModuleList([
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
        # Start the position-wise refinement as an identity without suppressing
        # the cross-frame branch itself.
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    @staticmethod
    def _frame_layout(token_offset, frame_batch_idx):
        counts = torch.diff(
            token_offset,
            prepend=token_offset.new_zeros(1),
        ).long()
        batch_frames = defaultdict(list)
        for frame, batch_id in enumerate(frame_batch_idx.tolist()):
            batch_frames[int(batch_id)].append(frame)
        for batch_id, frames in batch_frames.items():
            if len(frames) != 2:
                raise ValueError(
                    "Dynamic 2DGS currently requires exactly two input frames "
                    f"per sample; batch {batch_id} has {len(frames)}"
                )
        opposite = torch.empty_like(frame_batch_idx)
        for frames in batch_frames.values():
            opposite[frames[0]] = frames[1]
            opposite[frames[1]] = frames[0]

        starts = torch.cumsum(counts, dim=0) - counts
        frame_rows = [
            torch.arange(
                int(starts[f]), int(starts[f] + counts[f]),
                device=token_offset.device,
            )
            for f in range(counts.numel())
        ]
        memory_rows = torch.cat([
            frame_rows[int(opposite[f])] for f in range(counts.numel())
        ])
        return counts, counts[opposite], memory_rows

    def _flash_attention(
        self,
        q,
        k,
        v,
        q_counts,
        k_counts,
        *,
        flash_dtype=None,
    ):
        cu_q = F.pad(torch.cumsum(q_counts, dim=0, dtype=torch.int32), (1, 0))
        cu_k = F.pad(torch.cumsum(k_counts, dim=0, dtype=torch.int32), (1, 0))
        if flash_dtype is None:
            major, _minor = torch.cuda.get_device_capability(q.device)
            flash_dtype = torch.bfloat16 if major >= 8 else torch.float16
        attended = flash_attn.flash_attn_varlen_func(
            q=q.to(flash_dtype),
            k=k.to(flash_dtype),
            v=v.to(flash_dtype),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(q_counts.max().item()),
            max_seqlen_k=int(k_counts.max().item()),
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
        )
        return attended.to(q.dtype)

    def _sdpa_attention(self, q, k, v, q_counts, k_counts):
        """Reference/fallback path; sequence pairing is identical to Flash."""
        q_starts = torch.cumsum(q_counts, dim=0) - q_counts
        k_starts = torch.cumsum(k_counts, dim=0) - k_counts
        chunks = []
        for qs, qn, ks, kn in zip(q_starts, q_counts, k_starts, k_counts):
            qs, qn, ks, kn = map(int, (qs, qn, ks, kn))
            q_i = q[qs:qs + qn].transpose(0, 1).unsqueeze(0)
            k_i = k[ks:ks + kn].transpose(0, 1).unsqueeze(0)
            v_i = v[ks:ks + kn].transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=0.0, scale=self.scale
            )
            chunks.append(out_i.squeeze(0).transpose(0, 1))
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_normalized,
    ):
        n_tokens = int(feature.shape[0])
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with dynamic token features")
        if token_time_normalized.shape != (n_tokens,):
            raise ValueError("token_time_normalized must provide one scalar per token")
        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all dynamic tokens")

        time_embedding = self.time_encoder(token_time_normalized)
        q_angles = self.rope.angles(position_ref)
        k_angles = self.rope.angles(position_ref[memory_rows])
        x = feature
        base = feature
        use_flash = (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                "dynamic temporal attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )

        for layer in range(self.n_layers):
            # All projections in this layer read the same old-state snapshot.
            snapshot = x
            normalized = self.input_norm[layer](snapshot)
            q_input = normalized + self.time_to_q[layer](time_embedding)
            k_input = normalized + self.time_to_k[layer](time_embedding)
            q = self.q_proj[layer](q_input).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            k = self.k_proj[layer](k_input)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](normalized)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            q = self.rope.rotate(q.float(), q_angles[:, None, :, :])
            k = self.rope.rotate(k.float(), k_angles[:, None, :, :])
            if use_flash:
                attended = self._flash_attention(q, k, v, q_counts, k_counts)
            else:
                attended = self._sdpa_attention(q, k, v.float(), q_counts, k_counts)
            attended = attended.reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self.cross_layer_scale[layer] * self.out_proj[layer](
                attended
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))

        temporal_delta = x - base
        return x, temporal_delta, time_embedding


class TimeConditionedParallelCrossAttention(ParallelBidirectionalCrossAttention):
    """Cross-attention with optional endpoint-time token conditioning.

    A learned projection of the caller-provided source-time coordinate is added
    once to each fused token before the Transformer layers. Q, K, and V therefore
    all carry time information. V3 and V4 supply relative seconds divided by a
    fixed reference second. V7/V7.1 disable that path and may provide a distinct
    query coordinate for RoPE while keys remain at observed reference-frame
    coordinates. V7.2 and V8 restore the V3-V5 input-time path; V8 may also
    expose final Q/K to one sparse coordinate readout. Within each layer both
    endpoint directions read the same immutable snapshot and are committed
    together.
    """

    def __init__(self, cfg, dim: int):
        nn.Module.__init__(self)
        self.dim = int(dim)
        self.num_heads = int(_cfg_get(cfg, "num_heads", 8))
        self.motion_head_count = int(_cfg_get(cfg, "motion_head_count", 0))
        self.n_layers = int(_cfg_get(cfg, "layers", 1))
        self.mlp_ratio = float(_cfg_get(cfg, "mlp_ratio", 4.0))
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        self.use_time_embedding = bool(
            _cfg_get(cfg, "use_time_embedding", True)
        )
        if self.dim % self.num_heads != 0:
            raise ValueError("dynamic token dim must be divisible by num_heads")
        if self.n_layers < 1:
            raise ValueError("dynamic temporal layers must be at least one")
        if not 0 <= self.motion_head_count <= self.num_heads:
            raise ValueError(
                "motion_head_count must be between zero and num_heads"
            )
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "dynamic attention head dim must be divisible by 6 for 3D RoPE"
            )
        self.scale = self.head_dim ** -0.5

        time_frequencies = int(_cfg_get(cfg, "time_frequencies", 8))
        time_hidden_dim = _cfg_get(cfg, "time_hidden_dim", None)
        if not self.use_time_embedding:
            # V7/V7.1 supply signed physical time only through the detached Q
            # coordinate warp. No endpoint-time parameter is constructed or
            # added to Q/K/V or the residual stream.
            self.time_encoder = None
            self.time_to_feature = None
        elif time_hidden_dim is None:
            # Preserve the exact V3-V5 checkpoint parameterization:
            # Fourier -> MLP(..., time_dim) -> bias-free Linear(time_dim, dim).
            time_dim = int(_cfg_get(cfg, "time_embedding_dim", 64))
            self.time_encoder = SinusoidalScalarEncoder(
                time_dim, time_frequencies
            )
            self.time_to_feature = nn.Linear(time_dim, self.dim, bias=False)
        else:
            # V6 has no other consumer of a standalone 64D time embedding. Let
            # one MLP map Fourier features directly through the compact hidden
            # bottleneck to the token width: 17 -> 64 -> 576 by default.
            self.time_encoder = SinusoidalScalarEncoder(
                self.dim,
                time_frequencies,
                hidden_dim=int(time_hidden_dim),
            )
            self.time_to_feature = nn.Identity()
        rope_position_scale = _cfg_get(cfg, "rope_position_scale", None)
        if rope_position_scale is None:
            # Historical configs may specify a half-period range instead of the
            # actual radians/metre multiplier. Preserve that exact semantics.
            range_m = float(_cfg_get(cfg, "rope_range_m", 110.0))
            rope_position_scale = math.pi / range_m
        self.rope = Rotary3D(
            self.head_dim,
            base=float(_cfg_get(cfg, "rope_base", 10.0)),
            position_scale=float(rope_position_scale),
        )

        # V5: the motion heads solve a correspondence at the anchor spacing
        # (~0.4 m), which the shared band above cannot resolve -- its shortest
        # wavelength is 2*pi/position_scale metres. Giving those heads their own
        # finer band leaves the feature heads' band untouched.
        motion_rope_base = _cfg_get(cfg, "motion_rope_base", None)
        motion_rope_position_scale = _cfg_get(cfg, "motion_rope_position_scale", None)
        if (motion_rope_base is None) != (motion_rope_position_scale is None):
            raise ValueError(
                "motion_rope_base and motion_rope_position_scale must be set together"
            )
        if motion_rope_base is not None and self.motion_head_count <= 0:
            raise ValueError(
                "motion RoPE requires temporal.motion_head_count > 0"
            )
        self.motion_rope = (
            Rotary3D(
                self.head_dim,
                base=float(motion_rope_base),
                position_scale=float(motion_rope_position_scale),
            )
            if motion_rope_base is not None
            else None
        )

        self._build_qk_norm(_cfg_get(cfg, "qk_norm", False))
        self._build_transformer_layers(cfg)
        self._tie_motion_qk_init(cfg)

    def _tie_motion_qk_init(self, cfg):
        """V8: give the final layer's motion heads a symmetric descriptor.

        Q and K stay distinct Parameters, so rendering and correspondence
        gradients can specialize them later; only their initial geometry is
        shared. This copies existing weights and draws no new randomness, so
        where it runs in the constructor does not affect any other parameter.
        """
        self.tie_motion_qk_init = bool(
            _cfg_get(cfg, "tie_motion_qk_init", False)
        )
        if not self.tie_motion_qk_init:
            return
        if self.motion_head_count <= 0:
            raise ValueError(
                "tie_motion_qk_init requires motion_head_count > 0"
            )
        motion_width = self.motion_head_count * self.head_dim
        with torch.no_grad():
            self.k_proj[-1].weight[:motion_width].copy_(
                self.q_proj[-1].weight[:motion_width]
            )
            self.k_proj[-1].bias[:motion_width].copy_(
                self.q_proj[-1].bias[:motion_width]
            )

    def _rotate_heads(self, x, angles, motion_angles):
        """Apply RoPE, giving the motion heads their own band when configured."""
        if motion_angles is None:
            return self.rope.rotate(x, angles[:, None, :, :])
        split = self.motion_head_count
        return torch.cat(
            [
                self.motion_rope.rotate(
                    x[:, :split], motion_angles[:, None, :, :]
                ),
                self.rope.rotate(x[:, split:], angles[:, None, :, :]),
            ],
            dim=1,
        )

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Softmax temperature diagnostics, free of any attention computation.

        Under QK-Norm ``|q| = gamma * sqrt(head_dim)`` exactly, so the norm gains
        alone bound the logit. Tracking them catches the V4 saturation collapse
        long before it shows up in a render metric.
        """
        if not self.qk_norm:
            return {}
        motion = self.motion_head_count
        stats = {}
        bound = 0.0
        for layer in range(self.n_layers):
            gq = self.q_norm[layer].weight.float()
            gk = self.k_norm[layer].weight.float()
            rms_q = float(gq.square().mean().sqrt())
            rms_k = float(gk.square().mean().sqrt())
            # |q||k| * softmax_scale with |q| = rms_q * sqrt(head_dim)
            bound = max(bound, rms_q * rms_k * self.head_dim * self.scale)
            stats[f"qk_gamma_q_layer{layer}"] = rms_q
            stats[f"qk_gamma_k_layer{layer}"] = rms_k
        stats["qk_max_logit_bound"] = bound
        stats["qk_motion_head_count"] = float(motion)
        return stats

    def _motion_displacement(
        self,
        q,
        k,
        position_ref,
        memory_rows,
        q_counts,
        k_counts,
        *,
        use_flash,
        motion_matcher=None,
    ):
        """Return source-to-opposite-frame soft displacement from selected heads.

        ``q_proj`` and ``k_proj`` are stored as one full-width Linear per layer,
        but each contiguous output block is an independent head projection. V4
        reuses the first ``motion_head_count`` Q/K blocks from the final feature
        attention layer. Supplying xyz as V computes the attention-weighted
        opposite-frame position without materializing an O(N^2) score tensor.
        """

        head_count = self.motion_head_count
        if head_count <= 0:
            raise RuntimeError(
                "motion displacement requested with motion_head_count=0"
            )
        q_motion = q[:, :head_count]
        k_motion = k[:, :head_count]
        if motion_matcher is not None:
            return motion_matcher(
                q_motion,
                k_motion,
                position_ref,
                memory_rows,
                q_counts,
                k_counts,
                scale=self.scale,
            )
        coordinate_value = F.pad(
            position_ref[memory_rows].float(),
            (0, self.head_dim - 3),
        ).unsqueeze(1).expand(-1, head_count, -1).contiguous()
        if use_flash:
            # Metric xyz benefits from fp16's finer mantissa at driving ranges;
            # the feature path retains its existing device-dependent dtype.
            matched = self._flash_attention(
                q_motion,
                k_motion,
                coordinate_value,
                q_counts,
                k_counts,
                flash_dtype=torch.float16,
            )
        else:
            matched = self._sdpa_attention(
                q_motion,
                k_motion,
                coordinate_value,
                q_counts,
                k_counts,
            )
        matched_position = matched[..., :3].mean(dim=1)
        return (matched_position - position_ref.float()).to(position_ref.dtype)

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        *,
        return_motion_displacement=False,
        query_position_ref=None,
        motion_matcher=None,
    ):
        n_tokens = int(feature.shape[0])
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with dynamic token features")
        if token_time_coordinate is None and self.use_time_embedding:
            raise ValueError("time-conditioned attention requires token time")
        if (
            token_time_coordinate is not None
            and token_time_coordinate.shape != (n_tokens,)
        ):
            raise ValueError("token_time_coordinate must provide one scalar per token")
        if query_position_ref is None:
            query_position_ref = position_ref
        if query_position_ref.shape != (n_tokens, 3):
            raise ValueError(
                "query_position_ref must align with dynamic token features"
            )
        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all dynamic tokens")
        if return_motion_displacement and self.motion_head_count <= 0:
            raise ValueError(
                "return_motion_displacement requires motion_head_count > 0"
            )

        if self.use_time_embedding:
            time_embedding = self.time_encoder(token_time_coordinate)
            x = feature + self.time_to_feature(time_embedding).to(feature.dtype)
        else:
            x = feature
        q_angles = self.rope.angles(query_position_ref)
        k_angles = self.rope.angles(position_ref[memory_rows])
        if self.motion_rope is None:
            q_motion_angles = k_motion_angles = None
        else:
            q_motion_angles = self.motion_rope.angles(position_ref)
            k_motion_angles = self.motion_rope.angles(position_ref[memory_rows])
        use_flash = (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                "dynamic temporal attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )

        motion_displacement = None
        for layer in range(self.n_layers):
            snapshot = x
            conditioned = self.input_norm[layer](snapshot)
            q = self.q_proj[layer](conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            k = self.k_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            if self.qk_norm:
                q = self.q_norm[layer](q)
                k = self.k_norm[layer](k)
            q = self._rotate_heads(q.float(), q_angles, q_motion_angles)
            k = self._rotate_heads(k.float(), k_angles, k_motion_angles)
            if use_flash:
                attended = self._flash_attention(q, k, v, q_counts, k_counts)
            else:
                attended = self._sdpa_attention(q, k, v.float(), q_counts, k_counts)
            if return_motion_displacement and layer == self.n_layers - 1:
                motion_displacement = self._motion_displacement(
                    q,
                    k,
                    position_ref,
                    memory_rows,
                    q_counts,
                    k_counts,
                    use_flash=use_flash,
                    motion_matcher=motion_matcher,
                )
            attended = attended.reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self.cross_layer_scale[layer] * self.out_proj[layer](
                attended
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))
        if return_motion_displacement:
            return x, motion_displacement
        return x

class LayerWeightedDistanceBiasCrossAttention(
    ParallelBidirectionalCrossAttention
):
    """V10 all-layer/all-head matching with a query-only global token.

    For ordinary token queries, the score in every layer and head is

    ``q @ k / sqrt(d) - ||x_q - x_k||^2 / (speed * duration)^2``.

    The distance term replaces 3D RoPE rather than being stacked with it.  It is
    represented exactly (up to a query-row constant, which softmax cancels) by
    four extra Q/K channels.  This keeps the feature update and coordinate
    expectation in one varlen FlashAttention call and avoids materializing an
    ``O(N^2)`` bias tensor.  The V tensor carries the ordinary feature channels
    plus xyz, so averaging the xyz outputs over all heads gives ``W_l @ xyz``.

    One learned token is broadcast once per source frame and appended only to
    that frame's query sequence.  It therefore pools the opposite endpoint but
    cannot consume matching probability as a coordinate-less key. It receives
    the source frame's time embedding but its four geometric Q channels are
    zero, so no fictitious xyz or distance penalty is assigned. After each
    layer's residual and FFN update, one shared two-linear/one-activation MLP
    predicts ``a_l`` from the frame token.  Softmax over layers gives ``k_l``.
    By linearity, weighting the compact per-layer coordinate expectations is
    exactly equivalent to constructing ``W = sum_l k_l W_l`` first.
    """

    _DISTANCE_CHANNELS = 4
    _FLASH_HEAD_ALIGNMENT = 8

    def __init__(self, cfg, dim: int):
        # Deliberately skip the parent constructors: both historical classes
        # instantiate a Rotary3D object and impose its head-width constraint.
        nn.Module.__init__(self)
        self.dim = int(dim)
        self.num_heads = int(_cfg_get(cfg, "num_heads", 8))
        self.motion_head_count = self.num_heads
        self.n_layers = int(_cfg_get(cfg, "layers", 1))
        self.mlp_ratio = float(_cfg_get(cfg, "mlp_ratio", 4.0))
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        self.use_time_embedding = bool(
            _cfg_get(cfg, "use_time_embedding", True)
        )
        self.position_encoding = str(
            _cfg_get(cfg, "position_encoding", "distance_bias")
        ).lower()
        self.distance_bias_speed_mps = float(
            _cfg_get(cfg, "distance_bias_speed_mps", 30.0)
        )
        self.layer_weight_hidden_dim = int(
            _cfg_get(cfg, "layer_weight_hidden_dim", self.dim)
        )

        if self.dim <= 0 or self.num_heads <= 0:
            raise ValueError("V10 attention dimensions must be positive")
        if self.dim % self.num_heads != 0:
            raise ValueError("dynamic token dim must be divisible by num_heads")
        if self.n_layers < 1:
            raise ValueError("dynamic temporal layers must be at least one")
        if self.mlp_ratio <= 0.0:
            raise ValueError("dynamic temporal mlp_ratio must be positive")
        if not self.use_time_embedding:
            raise ValueError("V10 temporal attention requires time embedding")
        if self.position_encoding != "distance_bias":
            raise ValueError(
                "V10 temporal position_encoding must be 'distance_bias'"
            )
        if self.distance_bias_speed_mps <= 0.0:
            raise ValueError("V10 distance_bias_speed_mps must be positive")
        if self.layer_weight_hidden_dim <= 0:
            raise ValueError("V10 layer_weight_hidden_dim must be positive")

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        unpadded_head_dim = self.head_dim + self._DISTANCE_CHANNELS
        self.attention_head_dim = (
            (unpadded_head_dim + self._FLASH_HEAD_ALIGNMENT - 1)
            // self._FLASH_HEAD_ALIGNMENT
            * self._FLASH_HEAD_ALIGNMENT
        )
        if self.attention_head_dim > 256:
            raise ValueError(
                "V10 padded attention head width must not exceed 256"
            )

        time_dim = int(_cfg_get(cfg, "time_embedding_dim", 64))
        time_frequencies = int(_cfg_get(cfg, "time_frequencies", 8))
        self.time_encoder = SinusoidalScalarEncoder(
            time_dim, time_frequencies
        )
        self.time_to_feature = nn.Linear(time_dim, self.dim, bias=False)

        if not bool(_cfg_get(cfg, "qk_norm", True)):
            raise ValueError("V10 requires qk_norm=true")
        self._build_qk_norm(True)
        self._build_transformer_layers(cfg)

        # A single parameter is cloned per frame/query direction.  The clones
        # become distinct states after their first cross-attention update.
        self.global_token = nn.Parameter(torch.empty(1, self.dim))
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        self.layer_score_mlp = nn.Sequential(
            nn.Linear(self.dim, self.layer_weight_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.layer_weight_hidden_dim, 1),
        )
        # Start with an exact uniform distribution over layers.  The output
        # layer immediately receives gradient from differences among W_l.
        nn.init.zeros_(self.layer_score_mlp[-1].weight)
        nn.init.zeros_(self.layer_score_mlp[-1].bias)

    @staticmethod
    def _pack_frame_queries(regular, global_state, counts):
        """Interleave one global query after each ragged frame sequence."""
        chunks = []
        start = 0
        for frame, count in enumerate(counts):
            count = int(count)
            chunks.append(torch.cat([
                regular[start:start + count], global_state[frame:frame + 1]
            ], dim=0))
            start += count
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _unpack_frame_queries(packed, counts):
        """Undo :meth:`_pack_frame_queries` without changing token order."""
        regular_chunks = []
        global_chunks = []
        packed_start = 0
        for count in counts:
            count = int(count)
            regular_chunks.append(packed[packed_start:packed_start + count])
            global_chunks.append(
                packed[packed_start + count:packed_start + count + 1]
            )
            packed_start += count + 1
        return torch.cat(regular_chunks, dim=0), torch.cat(global_chunks, dim=0)

    @staticmethod
    def _frame_first(value, counts):
        starts = torch.cumsum(counts, dim=0) - counts
        return value[starts.long()]

    @staticmethod
    def _center_pair_positions(
        position_ref, memory_rows, q_counts, k_counts
    ):
        """Center each directional pair for stable squared-distance algebra."""
        query_starts = torch.cumsum(q_counts, dim=0) - q_counts
        key_starts = torch.cumsum(k_counts, dim=0) - k_counts
        key_position = position_ref[memory_rows].float()
        query_chunks = []
        key_chunks = []
        for qs, qn, ks, kn in zip(
            query_starts, q_counts, key_starts, k_counts
        ):
            qs, qn, ks, kn = map(int, (qs, qn, ks, kn))
            if qn <= 0 or kn <= 0:
                raise ValueError("V10 requires tokens in both endpoint frames")
            query = position_ref[qs:qs + qn].float()
            key = key_position[ks:ks + kn]
            origin = 0.5 * (query.mean(dim=0) + key.mean(dim=0))
            query_chunks.append(query - origin)
            key_chunks.append(key - origin)
        return torch.cat(query_chunks, dim=0), torch.cat(key_chunks, dim=0)

    def _distance_features(self, position, radius, *, is_global=False):
        """Four channels whose scaled dot product gives the distance bias."""
        if position.shape[-1] != 3 or radius.shape != position.shape[:-1]:
            raise ValueError("V10 distance features require aligned xyz/radius")
        if is_global:
            return position.new_zeros(*position.shape[:-1], 4)
        normalized = position.float() / radius.float().unsqueeze(-1)
        inverse_sqrt_scale = math.sqrt(1.0 / self.scale)
        xyz = math.sqrt(2.0 / self.scale) * normalized
        constant = torch.full_like(radius.float().unsqueeze(-1), inverse_sqrt_scale)
        return torch.cat([xyz, constant], dim=-1)

    def _distance_key_features(self, position, radius):
        if position.shape[-1] != 3 or radius.shape != position.shape[:-1]:
            raise ValueError("V10 distance key features require aligned xyz/radius")
        normalized = position.float() / radius.float().unsqueeze(-1)
        inverse_sqrt_scale = math.sqrt(1.0 / self.scale)
        return torch.cat([
            math.sqrt(2.0 / self.scale) * normalized,
            -normalized.square().sum(dim=-1, keepdim=True)
            * inverse_sqrt_scale,
        ], dim=-1)

    def _augment_attention_tensors(
        self,
        q,
        k,
        v,
        query_position,
        key_position,
        key_value_position,
        query_radius,
        key_radius,
        q_count_values,
        frame_global_q,
    ):
        """Pad Q/K/V and carry xyz in V for one shared attention softmax."""
        query_distance = self._distance_features(
            query_position, query_radius
        )
        global_distance = self._distance_features(
            frame_global_q.new_zeros(frame_global_q.shape[0], 3),
            frame_global_q.new_ones(frame_global_q.shape[0]),
            is_global=True,
        )
        packed_distance = self._pack_frame_queries(
            query_distance, global_distance, q_count_values
        )
        key_distance = self._distance_key_features(key_position, key_radius)

        padded_q = q.new_zeros(
            q.shape[0], self.num_heads, self.attention_head_dim,
            dtype=torch.float32,
        )
        padded_k = k.new_zeros(
            k.shape[0], self.num_heads, self.attention_head_dim,
            dtype=torch.float32,
        )
        padded_v = v.new_zeros(
            v.shape[0], self.num_heads, self.attention_head_dim,
            dtype=torch.float32,
        )
        padded_q[..., :self.head_dim] = q.float()
        padded_k[..., :self.head_dim] = k.float()
        padded_v[..., :self.head_dim] = v.float()
        distance_slice = slice(
            self.head_dim, self.head_dim + self._DISTANCE_CHANNELS
        )
        padded_q[..., distance_slice] = packed_distance[:, None, :]
        padded_k[..., distance_slice] = key_distance[:, None, :]
        padded_v[..., self.head_dim:self.head_dim + 3] = (
            key_value_position.float()[:, None, :]
        )
        return padded_q, padded_k, padded_v

    @torch.no_grad()
    def attention_temperature_stats(self):
        stats = {
            "qk_motion_head_count": float(self.num_heads),
            "distance_bias_speed_mps": self.distance_bias_speed_mps,
        }
        bound = 0.0
        for layer in range(self.n_layers):
            gq = self.q_norm[layer].weight.float()
            gk = self.k_norm[layer].weight.float()
            rms_q = float(gq.square().mean().sqrt())
            rms_k = float(gk.square().mean().sqrt())
            bound = max(bound, rms_q * rms_k * self.head_dim * self.scale)
            stats[f"qk_gamma_q_layer{layer}"] = rms_q
            stats[f"qk_gamma_k_layer{layer}"] = rms_k
        stats["qk_max_logit_bound"] = bound
        return stats

    @staticmethod
    def _combine_layer_matches(
        layer_matched_positions, frame_logits, q_counts, position_ref
    ):
        """Apply the per-frame layer softmax to compact ``W_l @ xyz`` values."""
        if layer_matched_positions.ndim != 3:
            raise ValueError("V10 layer matches must have shape (N,L,3)")
        if layer_matched_positions.shape[-1] != 3:
            raise ValueError("V10 layer matches must end in xyz")
        if frame_logits.shape != (
            q_counts.numel(), layer_matched_positions.shape[1]
        ):
            raise ValueError("V10 frame logits must align with frames and layers")
        if position_ref.shape != (layer_matched_positions.shape[0], 3):
            raise ValueError("V10 positions must align with layer matches")
        frame_weights = torch.softmax(frame_logits.float(), dim=-1)
        token_weights = torch.repeat_interleave(
            frame_weights, q_counts, dim=0
        )
        token_logits = torch.repeat_interleave(frame_logits, q_counts, dim=0)
        matched_position = (
            token_weights.unsqueeze(-1) * layer_matched_positions.float()
        ).sum(dim=1)
        displacement = matched_position - position_ref.float()
        return matched_position, displacement, token_logits, token_weights

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        token_duration_sec,
    ):
        n_tokens = int(feature.shape[0])
        if feature.shape != (n_tokens, self.dim):
            raise ValueError(f"V10 feature must be (N,{self.dim})")
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with V10 token features")
        if token_time_coordinate.shape != (n_tokens,):
            raise ValueError("V10 token time must provide one scalar per token")
        if token_duration_sec.shape != (n_tokens,):
            raise ValueError("V10 duration must provide one scalar per token")
        if not bool(torch.all(
            torch.isfinite(token_duration_sec) & (token_duration_sec > 0.0)
        )):
            raise ValueError("V10 requires positive finite durations")

        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all V10 tokens")
        if bool(torch.any(q_counts <= 0)) or bool(torch.any(k_counts <= 0)):
            raise ValueError("V10 requires tokens in both endpoint frames")

        time_embedding = self.time_encoder(token_time_coordinate)
        time_feature = self.time_to_feature(time_embedding).to(feature.dtype)
        x = feature + time_feature
        frame_time_feature = self._frame_first(time_feature, q_counts)
        global_state = self.global_token.to(feature.dtype).expand(
            q_counts.numel(), -1
        ) + frame_time_feature

        query_position, key_position = self._center_pair_positions(
            position_ref, memory_rows, q_counts, k_counts
        )
        key_value_position = position_ref[memory_rows]
        frame_duration = self._frame_first(token_duration_sec, q_counts).float()
        frame_radius = self.distance_bias_speed_mps * frame_duration
        query_radius = torch.repeat_interleave(frame_radius, q_counts)
        key_radius = torch.repeat_interleave(frame_radius, k_counts)
        query_counts_with_global = q_counts + 1
        # Convert once before the layer loop. Re-reading CUDA scalar counts in
        # every pack/unpack call would otherwise introduce three synchronizing
        # host round-trips per temporal layer.
        q_count_values = [int(value) for value in q_counts.tolist()]

        use_flash = (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                "V10 temporal attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )

        layer_matched_positions = []
        frame_layer_logits = []
        for layer in range(self.n_layers):
            token_snapshot = x
            global_snapshot = global_state
            token_conditioned = self.input_norm[layer](token_snapshot)
            global_conditioned = self.input_norm[layer](global_snapshot)

            token_q = self.q_proj[layer](token_conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            global_q = self.q_proj[layer](global_conditioned).reshape(
                q_counts.numel(), self.num_heads, self.head_dim
            )
            q = self._pack_frame_queries(token_q, global_q, q_count_values)
            k = self.k_proj[layer](token_conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](token_conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            q = self.q_norm[layer](q)
            k = self.k_norm[layer](k)
            q, k, v = self._augment_attention_tensors(
                q,
                k,
                v,
                query_position,
                key_position,
                key_value_position,
                query_radius,
                key_radius,
                q_count_values,
                global_q,
            )
            if use_flash:
                # xyz at driving range needs fp16's finer mantissa than bf16;
                # QK-Norm keeps the feature channels safe in fp16 here.
                attended = self._flash_attention(
                    q,
                    k,
                    v,
                    query_counts_with_global,
                    k_counts,
                    flash_dtype=torch.float16,
                )
            else:
                attended = self._sdpa_attention(
                    q,
                    k,
                    v,
                    query_counts_with_global,
                    k_counts,
                )
            token_attended, global_attended = self._unpack_frame_queries(
                attended, q_count_values
            )

            # Every head uses the same xyz V but its own probability row.  The
            # equal head mean is precisely W_l applied to opposite-frame xyz.
            layer_matched_positions.append(
                token_attended[..., self.head_dim:self.head_dim + 3].mean(dim=1)
            )
            token_feature = token_attended[..., :self.head_dim].reshape(
                n_tokens, self.dim
            ).to(token_snapshot.dtype)
            global_feature = global_attended[..., :self.head_dim].reshape(
                q_counts.numel(), self.dim
            ).to(global_snapshot.dtype)
            x = token_snapshot + self.cross_layer_scale[layer] * self.out_proj[layer](
                token_feature
            )
            global_state = (
                global_snapshot
                + self.cross_layer_scale[layer]
                * self.out_proj[layer](global_feature)
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))
            global_state = global_state + self.ffn[layer](
                self.ffn_norm[layer](global_state)
            )
            frame_layer_logits.append(
                self.layer_score_mlp(global_state).squeeze(-1)
            )

        frame_logits = torch.stack(frame_layer_logits, dim=-1)
        layer_matched_position = torch.stack(
            layer_matched_positions, dim=1
        ).float()
        (
            matched_position,
            displacement,
            token_logits,
            token_weights,
        ) = self._combine_layer_matches(
            layer_matched_position,
            frame_logits,
            q_counts,
            position_ref,
        )
        return x, {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_layer_logits": token_logits,
            "motion_layer_weights": token_weights,
        }

class MaxSpeedBarrierLayerWeightedCrossAttention(
    ParallelBidirectionalCrossAttention
):
    """V11 split position encoding with a parameter-only layer mixture.

    Head 0 of every layer is the correspondence head, and is the only head whose
    probabilities ever touch coordinates.  It carries no 3D RoPE.  Its position
    encoding is instead one additive physical barrier on the logit,

    ``bias_ij = -a * relu(||x_i - x_j|| / (v_max * |dt|) - 1)^2``,

    which is exactly zero inside the radius a ``v_max`` object could cover over
    the endpoint interval and grows quadratically outside it.  Unlike V10's
    ``-(d/R)^2`` prior the hinge is not a bilinear form, so it cannot be folded
    into extra Q/K channels; this head therefore runs as an explicit softmax,
    chunked over queries and recomputed in backward.  Its V carries the ordinary
    feature channels plus opposite-frame xyz, so one pass returns both the
    feature update and ``W_l @ xyz``.

    Heads ``1..H-1`` keep ordinary metric 3D RoPE and stay on varlen
    FlashAttention.

    ``v_init`` mixes the ``L`` head-0 coordinate expectations under
    ``softmax(a_0..a_{L-1})`` over ``L`` plain learned scalars -- no global
    token and no per-sample scorer -- so the mixture asks only which *depth*
    resolves correspondence, starting from an exact uniform ``1/L``.
    """

    _MATCH_HEAD = 0

    @staticmethod
    def _frame_first(value, counts):
        """The first token's value in every frame, one entry per frame."""
        starts = torch.cumsum(counts, dim=0) - counts
        return value[starts.long()]

    def __init__(self, cfg, dim: int):
        # Skip both parent constructors: the base class builds one RoPE across
        # every head and a per-layer time_to_q/time_to_k pair this variant does
        # not use.
        nn.Module.__init__(self)
        self.dim = int(dim)
        self.num_heads = int(_cfg_get(cfg, "num_heads", 8))
        self.n_layers = int(_cfg_get(cfg, "layers", 1))
        self.mlp_ratio = float(_cfg_get(cfg, "mlp_ratio", 4.0))
        self.implementation = str(
            _cfg_get(cfg, "implementation", "flash_varlen")
        ).lower()
        self.use_time_embedding = bool(
            _cfg_get(cfg, "use_time_embedding", True)
        )
        self.position_encoding = str(
            _cfg_get(cfg, "position_encoding", "barrier_rope_split")
        ).lower()
        self.barrier_speed_mps = float(_cfg_get(cfg, "barrier_speed_mps", 30.0))
        self.barrier_weight = float(_cfg_get(cfg, "barrier_weight", 4.0))
        self.match_chunk_size = int(_cfg_get(cfg, "match_chunk_size", 1024))

        if self.dim <= 0 or self.num_heads <= 0:
            raise ValueError("V11 attention dimensions must be positive")
        if self.dim % self.num_heads != 0:
            raise ValueError("dynamic token dim must be divisible by num_heads")
        if self.num_heads < 2:
            raise ValueError(
                "V11 needs one barrier match head plus at least one RoPE head"
            )
        if self.n_layers < 1:
            raise ValueError("dynamic temporal layers must be at least one")
        if self.mlp_ratio <= 0.0:
            raise ValueError("dynamic temporal mlp_ratio must be positive")
        if not self.use_time_embedding:
            raise ValueError("V11 temporal attention requires time embedding")
        if self.position_encoding != "barrier_rope_split":
            raise ValueError(
                "V11 temporal position_encoding must be 'barrier_rope_split'"
            )
        if self.barrier_speed_mps <= 0.0:
            raise ValueError("V11 barrier_speed_mps must be positive")
        if self.barrier_weight < 0.0:
            raise ValueError("V11 barrier_weight must be non-negative")
        if self.match_chunk_size <= 0:
            raise ValueError("V11 match_chunk_size must be positive")

        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "dynamic attention head dim must be divisible by 6 for 3D RoPE"
            )
        self.scale = self.head_dim ** -0.5
        self.rope_head_count = self.num_heads - 1

        rope_position_scale = _cfg_get(cfg, "rope_position_scale", None)
        if rope_position_scale is None:
            range_m = float(_cfg_get(cfg, "rope_range_m", 110.0))
            rope_position_scale = math.pi / range_m
        # Heads 1..H-1 only; head 0 reads geometry from the barrier instead.
        self.rope = Rotary3D(
            self.head_dim,
            base=float(_cfg_get(cfg, "rope_base", 100.0)),
            position_scale=float(rope_position_scale),
        )

        # Preserve the V5/V9 endpoint-time parameterization exactly:
        # Fourier -> MLP(..., time_dim) -> bias-free Linear(time_dim, dim),
        # added once to the fused token before layer one.
        time_dim = int(_cfg_get(cfg, "time_embedding_dim", 64))
        time_frequencies = int(_cfg_get(cfg, "time_frequencies", 8))
        self.time_encoder = SinusoidalScalarEncoder(time_dim, time_frequencies)
        self.time_to_feature = nn.Linear(time_dim, self.dim, bias=False)

        if not bool(_cfg_get(cfg, "qk_norm", True)):
            raise ValueError("V11 requires qk_norm=true")
        self._build_qk_norm(True)
        self._build_transformer_layers(cfg)

        # One scalar per layer, shared by every token and every sample. Zero
        # init is an exact uniform 1/L mixture, and each logit's only gradient
        # is the render loss' preference among the layers' coordinate readouts.
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))

    @torch.no_grad()
    def attention_temperature_stats(self):
        """QK-Norm gain diagnostics next to the fixed barrier geometry.

        The content logit is bounded by ``gamma_q * gamma_k * head_dim * scale``
        while the barrier is a fixed constant, so watching the bound is also how
        one sees the barrier being outgrown.
        """
        stats = {
            "qk_motion_head_count": 1.0,
            "barrier_speed_mps": self.barrier_speed_mps,
            "barrier_weight": self.barrier_weight,
        }
        bound = 0.0
        for layer in range(self.n_layers):
            gq = self.q_norm[layer].weight.float()
            gk = self.k_norm[layer].weight.float()
            rms_q = float(gq.square().mean().sqrt())
            rms_k = float(gk.square().mean().sqrt())
            bound = max(bound, rms_q * rms_k * self.head_dim * self.scale)
            stats[f"qk_gamma_q_layer{layer}"] = rms_q
            stats[f"qk_gamma_k_layer{layer}"] = rms_k
        stats["qk_max_logit_bound"] = bound
        return stats

    def _barrier_bias(self, query_position, key_position, radius):
        """Reference form ``a * relu(d/R - 1)^2`` for one chunk of queries.

        This is what the barrier *means*; :meth:`_barrier_chunk` evaluates the
        algebraically identical ``(a/R^2) * relu(d - R)^2`` because that folds
        into the score GEMM. Positions are input geometry rather than
        parameters, so the barrier is a constant of the graph either way.
        """
        with torch.no_grad():
            excess = torch.cdist(query_position, key_position)
            excess = excess.div_(float(radius)).sub_(1.0).clamp_min_(0.0)
            return excess.square_().mul_(self.barrier_weight)

    def _barrier_chunk(
        self, query, key, value, query_position, key_position, radius
    ):
        """One head-0 softmax over every opposite-frame key for this chunk.

        ``radius`` is one Python scalar: a chunk never spans an endpoint pair,
        and the barrier radius is that sample's physical interval. At these
        sizes the ``N_q x N_k`` block is bandwidth-bound rather than
        FLOP-bound, so the hinge is written as ``relu(d - R)^2`` and folded
        into the score GEMM's ``beta * C`` accumulate. That leaves three
        elementwise passes over the block instead of nine.
        """
        radius = float(radius)
        with torch.no_grad():
            excess = torch.cdist(
                query_position, key_position
            ).sub_(radius).relu_().square_()
        score = torch.addmm(
            excess,
            query,
            key.transpose(0, 1),
            beta=-self.barrier_weight / (radius * radius),
            alpha=self.scale,
        )
        probability = torch.softmax(score, dim=-1)
        attended = probability @ value
        return attended[..., :self.head_dim], attended[..., self.head_dim:]

    def _barrier_attention(
        self,
        query,
        key,
        value,
        query_position,
        key_position,
        q_count_values,
        k_count_values,
        frame_radius_values,
    ):
        """Head-0 feature update and ``W_l @ xyz``, chunked over queries.

        Retaining every chunk's ``N_q x N_k`` probabilities across twelve layers
        would dominate activation memory, so each chunk is recomputed in
        backward exactly as V9's dense proposal does. The ragged layout and the
        per-frame radius arrive as Python scalars the caller resolved once, so
        no layer costs a host synchronization.
        """
        recompute = (
            self.training
            and torch.is_grad_enabled()
            and (
                query.requires_grad or key.requires_grad or value.requires_grad
            )
        )
        feature_chunks = []
        position_chunks = []
        query_start = 0
        key_start = 0
        for q_count, k_count, radius in zip(
            q_count_values, k_count_values, frame_radius_values
        ):
            if q_count <= 0 or k_count <= 0:
                raise ValueError("V11 requires tokens in both endpoint frames")
            keys = slice(key_start, key_start + k_count)
            key_xyz = key_position[keys]
            # Centering only improves the accuracy of the pairwise distance in
            # global reference coordinates. It changes no distance and no
            # displacement, because the readout adds the same origin back.
            origin = 0.5 * (
                query_position[query_start:query_start + q_count].mean(dim=0)
                + key_xyz.mean(dim=0)
            )
            key_centered = key_xyz - origin
            payload = torch.cat([value[keys], key_centered], dim=-1)
            for start in range(0, q_count, self.match_chunk_size):
                end = min(start + self.match_chunk_size, q_count)
                rows = slice(query_start + start, query_start + end)
                inputs = (
                    query[rows],
                    key[keys],
                    payload,
                    query_position[rows] - origin,
                    key_centered,
                    radius,
                )
                if recompute:
                    chunk = checkpoint(
                        self._barrier_chunk, *inputs, use_reentrant=False
                    )
                else:
                    chunk = self._barrier_chunk(*inputs)
                feature_chunks.append(chunk[0])
                position_chunks.append(chunk[1] + origin)
            query_start += q_count
            key_start += k_count
        return (
            torch.cat(feature_chunks, dim=0),
            torch.cat(position_chunks, dim=0),
        )

    def _combine_layer_matches(self, layer_matched_positions, position_ref):
        """Mix the per-layer coordinate expectations under the layer softmax."""
        if layer_matched_positions.ndim != 3:
            raise ValueError("V11 layer matches must have shape (N,L,3)")
        if layer_matched_positions.shape[1] != self.n_layers:
            raise ValueError("V11 layer matches must cover every layer")
        if layer_matched_positions.shape[-1] != 3:
            raise ValueError("V11 layer matches must end in xyz")
        if position_ref.shape != (layer_matched_positions.shape[0], 3):
            raise ValueError("V11 positions must align with layer matches")
        weights = torch.softmax(self.layer_logits.float(), dim=-1)
        matched_position = (
            weights.reshape(1, -1, 1) * layer_matched_positions.float()
        ).sum(dim=1)
        displacement = matched_position - position_ref.float()
        token_count = int(layer_matched_positions.shape[0])
        return (
            matched_position,
            displacement,
            self.layer_logits.detach().expand(token_count, -1),
            weights.detach().expand(token_count, -1),
        )

    def _prepare_forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_duration_sec,
    ):
        """Validate the batch and resolve the ragged layout and barrier radii.

        Every barrier variant needs exactly this, and needs it before any
        attention runs. Resolving the counts and the radius to Python scalars
        once matters: re-reading CUDA counts inside the layer loop would cost
        several synchronizing host round-trips per temporal layer.
        """
        n_tokens = int(feature.shape[0])
        if feature.shape != (n_tokens, self.dim):
            raise ValueError(f"V11 feature must be (N,{self.dim})")
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with V11 token features")
        if token_duration_sec.shape != (n_tokens,):
            raise ValueError("V11 duration must provide one scalar per token")
        if not bool(torch.all(
            torch.isfinite(token_duration_sec) & (token_duration_sec > 0.0)
        )):
            raise ValueError("V11 requires positive finite durations")

        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all V11 tokens")
        if bool(torch.any(q_counts <= 0)) or bool(torch.any(k_counts <= 0)):
            raise ValueError("V11 requires tokens in both endpoint frames")

        frame_duration = self._frame_first(token_duration_sec, q_counts)
        if not torch.equal(
            torch.repeat_interleave(frame_duration, q_counts),
            token_duration_sec,
        ):
            raise ValueError(
                "V11 requires one endpoint interval per frame; the barrier "
                "radius is a per-sample physical duration"
            )
        return {
            "n_tokens": n_tokens,
            "q_counts": q_counts,
            "k_counts": k_counts,
            "memory_rows": memory_rows,
            "q_count_values": [int(value) for value in q_counts.tolist()],
            "k_count_values": [int(value) for value in k_counts.tolist()],
            # What an object at the admissible top speed could cover over this
            # sample's endpoint interval.
            "frame_radius_values": [
                max(self.barrier_speed_mps * float(value), 1.0e-4)
                for value in frame_duration.tolist()
            ],
        }

    def _use_flash(self, x):
        """Whether the varlen kernel is available for this batch."""
        if (
            x.is_cuda
            and self.implementation == "flash_varlen"
            and flash_attn is None
        ):
            raise RuntimeError(
                "V11 temporal attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )
        return (
            x.is_cuda
            and flash_attn is not None
            and self.implementation in ("flash_varlen", "auto")
        )

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        token_duration_sec,
    ):
        if token_time_coordinate.shape != (int(feature.shape[0]),):
            raise ValueError("V11 token time must provide one scalar per token")
        layout = self._prepare_forward(
            feature, position_ref, token_offset, frame_batch_idx,
            token_duration_sec,
        )
        n_tokens = layout["n_tokens"]
        q_counts = layout["q_counts"]
        k_counts = layout["k_counts"]
        memory_rows = layout["memory_rows"]
        q_count_values = layout["q_count_values"]
        k_count_values = layout["k_count_values"]
        frame_radius_values = layout["frame_radius_values"]

        x = feature + self.time_to_feature(
            self.time_encoder(token_time_coordinate)
        ).to(feature.dtype)

        query_position = position_ref.float()
        key_position = position_ref[memory_rows].float()
        q_angles = self.rope.angles(query_position)
        k_angles = self.rope.angles(key_position)

        use_flash = self._use_flash(x)

        layer_matched_positions = []
        for layer in range(self.n_layers):
            snapshot = x
            conditioned = self.input_norm[layer](snapshot)
            q = self.q_proj[layer](conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            k = self.k_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            q = self.q_norm[layer](q).float()
            k = self.k_norm[layer](k).float()
            v = v.float()

            match_attended, matched_position = self._barrier_attention(
                q[:, self._MATCH_HEAD],
                k[:, self._MATCH_HEAD],
                v[:, self._MATCH_HEAD],
                query_position,
                key_position,
                q_count_values,
                k_count_values,
                frame_radius_values,
            )
            layer_matched_positions.append(matched_position)

            rope_head = slice(self._MATCH_HEAD + 1, None)
            rope_attended_inputs = (
                self.rope.rotate(q[:, rope_head], q_angles[:, None, :, :]),
                self.rope.rotate(k[:, rope_head], k_angles[:, None, :, :]),
                v[:, rope_head],
                q_counts,
                k_counts,
            )
            if use_flash:
                rope_attended = self._flash_attention(*rope_attended_inputs)
            else:
                rope_attended = self._sdpa_attention(*rope_attended_inputs)

            attended = torch.cat(
                [match_attended.unsqueeze(1), rope_attended], dim=1
            ).reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self.cross_layer_scale[layer] * self.out_proj[layer](
                attended
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))

        (
            matched_position,
            displacement,
            token_logits,
            token_weights,
        ) = self._combine_layer_matches(
            torch.stack(layer_matched_positions, dim=1), position_ref
        )
        return x, {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_layer_logits": token_logits,
            "motion_layer_weights": token_weights,
        }


class FinalFeatureBarrierCrossAttention(
    MaxSpeedBarrierLayerWeightedCrossAttention
):
    """V11.2: uniform-RoPE flash refinement with one barrier readout at the end.

    V11 splits its heads because head 0 has to double as the correspondence
    head: it drops 3D RoPE, carries the max-speed barrier instead, and reads
    opposite-frame coordinates at every layer.  That barrier is a hinge on
    Euclidean distance rather than a bilinear form, so it cannot be folded into
    Q/K channels and head 0 has to run an explicit chunked softmax while heads
    1..H-1 stay on varlen FlashAttention.

    V11.2 forms ``v_init`` *after* the stack, so nothing inside it needs to
    read coordinates.  Head 0 therefore carries ordinary 3D RoPE like every
    other head, the barrier leaves the layer loop entirely, and all ``H`` heads
    go through one FlashAttention call per layer.  On an A100 at 7.5k tokens
    per frame this is about 40% off the module's forward and backward.

    Correspondence is then read once, from the complete final feature ``f'``,
    by one dedicated head:

    ``match_input_norm(f') -> match_q_proj/match_k_proj -> qk RMSNorm``.

    That head is a structural copy of a single V11 readout head: the same
    ``LayerNorm(dim) -> Linear(dim, head_dim) -> RMSNorm(head_dim)`` chain at
    the same width.  PyTorch initializes ``Linear`` from ``fan_in``, which is
    ``dim`` for both, so it starts from the identical distribution as head 0's
    rows of V11's ``Linear(dim, dim)``.  Its dense softmax keeps V11's exact
    barrier, chunking and backward recomputation; this is the one place in the
    variant where geometry enters a probability.

    Against V11 this moves two things at once, which the experiment intends:
    where correspondence is read, and whether the layer stack still needs a
    geometric prior of its own once it no longer produces correspondence.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        # Every head is a RoPE head here. V11's layer mixture had one scalar
        # per layer to weight per-layer coordinate readouts; there are no
        # per-layer readouts left to weight, so the parameter goes rather than
        # sitting in the checkpoint collecting no gradient.
        del self.layer_logits
        self.rope_head_count = self.num_heads
        # The one readout head, appended after the layer stack.
        self.match_input_norm = nn.LayerNorm(self.dim)
        self.match_q_proj = nn.Linear(self.dim, self.head_dim)
        self.match_k_proj = nn.Linear(self.dim, self.head_dim)
        self.match_q_norm = nn.RMSNorm(self.head_dim)
        self.match_k_norm = nn.RMSNorm(self.head_dim)

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Per-layer QK gains, plus the readout head that faces the barrier.

        The inherited per-layer keys stay comparable to a V11 run even though
        head 0 now carries RoPE, because they describe the same projections.
        ``qk_motion_head_count`` is corrected to zero: no layer head reads
        coordinates any more. The readout's own bound is what races the fixed
        barrier, so it is reported separately.
        """
        stats = super().attention_temperature_stats()
        stats["qk_motion_head_count"] = 0.0
        gamma_q = float(self.match_q_norm.weight.float().square().mean().sqrt())
        gamma_k = float(self.match_k_norm.weight.float().square().mean().sqrt())
        stats["qk_match_gamma_q"] = gamma_q
        stats["qk_match_gamma_k"] = gamma_k
        stats["qk_match_logit_bound"] = (
            gamma_q * gamma_k * self.head_dim * self.scale
        )
        return stats

    def _final_match_descriptors(self, refined, memory_rows):
        """Project ``f'`` through this variant's own head-width readout."""
        conditioned = self.match_input_norm(refined)
        query = self.match_q_norm(self.match_q_proj(conditioned))
        # Gather before the norm exactly as the layer stack does; RMSNorm is
        # per row, so the order is a readability choice, not a numerical one.
        key = self.match_k_norm(self.match_k_proj(conditioned)[memory_rows])
        return query.float(), key.float()

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        token_time_coordinate,
        token_duration_sec,
    ):
        n_tokens = int(feature.shape[0])
        if token_time_coordinate.shape != (n_tokens,):
            raise ValueError("V11 token time must provide one scalar per token")
        layout = self._prepare_forward(
            feature, position_ref, token_offset, frame_batch_idx,
            token_duration_sec,
        )
        q_counts = layout["q_counts"]
        k_counts = layout["k_counts"]
        memory_rows = layout["memory_rows"]

        x = feature + self.time_to_feature(
            self.time_encoder(token_time_coordinate)
        ).to(feature.dtype)

        query_position = position_ref.float()
        key_position = position_ref[memory_rows].float()
        q_angles = self.rope.angles(query_position)
        k_angles = self.rope.angles(key_position)
        use_flash = self._use_flash(x)

        for layer in range(self.n_layers):
            snapshot = x
            conditioned = self.input_norm[layer](snapshot)
            q = self.q_proj[layer](conditioned).reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            k = self.k_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            v = self.v_proj[layer](conditioned)[memory_rows].reshape(
                n_tokens, self.num_heads, self.head_dim
            )
            # One call over every head: no head is held out of the kernel.
            attended_inputs = (
                self.rope.rotate(
                    self.q_norm[layer](q).float(), q_angles[:, None, :, :]
                ),
                self.rope.rotate(
                    self.k_norm[layer](k).float(), k_angles[:, None, :, :]
                ),
                v.float(),
                q_counts,
                k_counts,
            )
            if use_flash:
                attended = self._flash_attention(*attended_inputs)
            else:
                attended = self._sdpa_attention(*attended_inputs)

            attended = attended.reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self.cross_layer_scale[layer] * self.out_proj[layer](
                attended
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))

        query, key = self._final_match_descriptors(x, memory_rows)
        # The barrier softmax is V11's, chunked and recomputed in backward. The
        # feature payload is zero because this attention is a coordinate
        # readout only; probability still multiplies the real opposite-frame
        # key coordinates inside ``_barrier_attention``.
        _unused_feature, matched_position = self._barrier_attention(
            query,
            key,
            key.new_zeros(key.shape),
            query_position,
            key_position,
            layout["q_count_values"],
            layout["k_count_values"],
            layout["frame_radius_values"],
        )
        displacement = matched_position.float() - query_position
        return x, {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
        }


class GroupedHeadRMSNorm(nn.Module):
    """RMSNorm over the head dimension with one learned gain per head *group*.

    Under RMSNorm ``|q| = gamma_rms * sqrt(head_dim)`` holds exactly for every
    head, so a single gain broadcast over all heads pins them to one logit
    temperature by construction.  That is right when the heads share a job, and
    wrong when one of them does not: V11.1's head 0 must reach a logit gap near
    seven to make its coordinate readout metric, while the RoPE feature heads
    want to keep averaging broadly.  Splitting the gain on exactly the boundary
    the position encoding already uses lets head 0 sharpen on its own.

    ``group_of_head[h]`` names the gain row head ``h`` reads.
    """

    def __init__(self, head_dim: int, group_of_head, eps: float = 1.0e-6):
        super().__init__()
        index = torch.as_tensor(list(group_of_head), dtype=torch.long)
        if index.ndim != 1 or index.numel() < 1:
            raise ValueError("group_of_head must list one group per head")
        if int(index.min()) < 0:
            raise ValueError("group_of_head entries must be non-negative")
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        self.num_groups = int(index.max()) + 1
        # Ones is the RMSNorm identity, matching nn.RMSNorm's initialization.
        self.weight = nn.Parameter(torch.ones(self.num_groups, self.head_dim))
        self.register_buffer("group_of_head", index, persistent=False)

    def forward(self, x):
        if x.ndim < 2 or x.shape[-1] != self.head_dim:
            raise ValueError(f"grouped RMSNorm expects (..., H, {self.head_dim})")
        if x.shape[-2] != self.group_of_head.numel():
            raise ValueError("grouped RMSNorm head count does not match config")
        value = x.float()
        normalized = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight[self.group_of_head]).to(x.dtype)

    @torch.no_grad()
    def group_gain_rms(self):
        return [
            float(self.weight[g].float().square().mean().sqrt())
            for g in range(self.num_groups)
        ]

class GroupedGainBarrierCrossAttention(
    MaxSpeedBarrierLayerWeightedCrossAttention
):
    """V11.1: V11's stack with QK-Norm's gain split on the head-0 boundary.

    This is the only change to the temporal module.  Everything V11 does --
    the max-speed barrier on head 0, the chunked explicit softmax, RoPE on
    heads 1..H-1, and the parameter-only layer mixture -- is inherited
    unmodified, so the two variants differ by one tensor shape.

    Under RMSNorm ``|q| = gamma_rms * sqrt(head_dim)`` holds exactly for every
    head, so a single gain broadcast over all heads pins them to one logit
    temperature by construction.  Head 0 has a different job from the rest: its
    probabilities are read as a coordinate, and a soft-argmax is only metric
    when the mass is concentrated, which needs a logit gap near seven (measured
    on V11: ~37.7 tokens inside a 2 m ball against ~3735 admitted candidates).
    The RoPE feature heads have no such requirement and are free to keep
    averaging broadly.  Splitting the gain on exactly the boundary the position
    encoding already uses lets each side find its own temperature.

    Measured on V11.1's first three epochs, the two groups do separate and the
    direction depends on depth: head 0 runs up to 25% hotter than the RoPE
    heads in layers 1-5 and up to 7% cooler in layers 8-11, while V11's single
    shared gain sits above both.
    """

    def __init__(self, cfg, dim: int):
        super().__init__(cfg, dim)
        # Head 0 carries the barrier and the readout; heads 1..H-1 carry RoPE.
        # The gain split follows that same boundary.
        group_of_head = [1] * self.num_heads
        group_of_head[self._MATCH_HEAD] = 0
        self.q_norm = nn.ModuleList([
            GroupedHeadRMSNorm(self.head_dim, group_of_head)
            for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            GroupedHeadRMSNorm(self.head_dim, group_of_head)
            for _ in range(self.n_layers)
        ])

    @torch.no_grad()
    def attention_temperature_stats(self):
        """Per-group QK gains, so the match head can be watched on its own.

        V11 could only report one bound for all twelve heads, which is precisely
        the coupling this variant removes.
        """
        stats = {
            "qk_motion_head_count": 1.0,
            "barrier_speed_mps": self.barrier_speed_mps,
            "barrier_weight": self.barrier_weight,
        }
        names = {0: "match", 1: "rope"}
        bounds = {}
        for layer in range(self.n_layers):
            gq = self.q_norm[layer].group_gain_rms()
            gk = self.k_norm[layer].group_gain_rms()
            for group, (rms_q, rms_k) in enumerate(zip(gq, gk)):
                tag = names.get(group, str(group))
                stats[f"qk_gamma_q_{tag}_layer{layer}"] = rms_q
                stats[f"qk_gamma_k_{tag}_layer{layer}"] = rms_k
                bounds[tag] = max(
                    bounds.get(tag, 0.0),
                    rms_q * rms_k * self.head_dim * self.scale,
                )
        for tag, bound in bounds.items():
            stats[f"qk_max_logit_bound_{tag}"] = bound
        stats["qk_max_logit_bound"] = max(bounds.values()) if bounds else 0.0
        return stats

__all__ = [
    "ParallelBidirectionalCrossAttention",
    "TimeConditionedParallelCrossAttention",
    "LayerWeightedDistanceBiasCrossAttention",
    "MaxSpeedBarrierLayerWeightedCrossAttention",
    "FinalFeatureBarrierCrossAttention",
    "GroupedHeadRMSNorm",
    "GroupedGainBarrierCrossAttention",
 ]
