"""BBox-free Dynamic 2DGS modules.

The shared Point2Gaus frontend has already produced one fused feature per
occupied Utonia grid token by the time this module runs.  This backend:

1. moves each frame's token and observed medoid seed into the common ref frame;
2. refines both endpoint token sets with synchronous, bidirectional full
   cross-attention;
3. predicts time-invariant 2D Gaussian attributes directly from the refined
   token feature while retaining the observed medoid as the centre anchor;
4. predicts motion with a separate head.

Every input token emits exactly one Gaussian.  Rotation, scale, opacity, and SH
attributes remain fixed over time in these first representations; only position
is transported linearly by the renderer. ``direct_velocity_v1`` remains for
historical checkpoint reconstruction. V3/V3.1 directly regress velocity in
metres/second. V4/V5 reuse heads from the final temporal layer as a
differentiable soft-correspondence initializer. V6 instead keeps correspondence
outside feature cross-attention: one shared descriptor projection reads frozen
Utonia tokens, selects a recall-oriented global candidate pool under a soft
geometric prior, and uses reciprocal re-ranking plus a dustbin to restrict the
actual coordinate mixture to a small cluster.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.attention import FourierPositionEncoder3D, Rotary3D
from ..utils import boxes as box_utils

try:
    import flash_attn
except ImportError:  # CPU/unit-test fallback remains available.
    flash_attn = None


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


class SinusoidalScalarEncoder(nn.Module):
    """Sinusoidal scalar encoding followed by a learned MLP projection."""

    def __init__(self, out_dim: int, num_frequencies: int = 8,
                 hidden_dim: int | None = None):
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_frequencies = int(num_frequencies)
        self.hidden_dim = (
            max(self.out_dim, 32) if hidden_dim is None else int(hidden_dim)
        )
        if (
            self.out_dim <= 0
            or self.num_frequencies <= 0
            or self.hidden_dim <= 0
        ):
            raise ValueError("time embedding dimensions must be positive")
        frequencies = math.pi * (2.0 ** torch.arange(
            self.num_frequencies, dtype=torch.float32
        ))
        self.register_buffer("frequencies", frequencies, persistent=False)
        in_dim = 1 + 2 * self.num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(-1, 1)
        phase = value.float() * self.frequencies.reshape(1, -1)
        encoded = torch.cat([value.float(), phase.sin(), phase.cos()], dim=-1)
        return self.mlp(encoded.to(dtype=value.dtype))


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


        hidden = int(self.dim * self.mlp_ratio)
        self.input_norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        self.time_to_q = nn.ModuleList([
            nn.Linear(time_dim, self.dim, bias=False) for _ in range(self.n_layers)
        ])
        self.time_to_k = nn.ModuleList([
            nn.Linear(time_dim, self.dim, bias=False) for _ in range(self.n_layers)
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
    """Cross-attention over time-conditioned endpoint token features.

    A learned projection of the caller-provided source-time coordinate is added
    once to each fused token before the Transformer layers. Q, K, and V therefore
    all carry time information. V3 and V4 supply relative seconds divided by a
    fixed reference second. Within each
    layer both endpoint directions read the same immutable snapshot and are
    committed together.
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
        if time_hidden_dim is None:
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

        # V5: nothing bounds |q| or |k| otherwise -- input_norm sits before the
        # projection, so the projection gain passes through unchecked and the
        # logit grows as its square. RMSNorm rather than LayerNorm because
        # LayerNorm's mean subtraction would mix Rotary3D's x/y/z chunks.
        self.qk_norm = bool(_cfg_get(cfg, "qk_norm", False))
        if self.qk_norm:
            self.q_norm = nn.ModuleList([
                nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
            ])
            self.k_norm = nn.ModuleList([
                nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
            ])
        else:
            self.q_norm = None
            self.k_norm = None

        hidden = int(self.dim * self.mlp_ratio)
        self.input_norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
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
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

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
    ):
        n_tokens = int(feature.shape[0])
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with dynamic token features")
        if token_time_coordinate.shape != (n_tokens,):
            raise ValueError("token_time_coordinate must provide one scalar per token")
        q_counts, k_counts, memory_rows = self._frame_layout(
            token_offset, frame_batch_idx
        )
        if int(q_counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all dynamic tokens")
        if return_motion_displacement and self.motion_head_count <= 0:
            raise ValueError(
                "return_motion_displacement requires motion_head_count > 0"
            )

        time_embedding = self.time_encoder(token_time_coordinate)
        x = feature + self.time_to_feature(time_embedding).to(feature.dtype)
        q_angles = self.rope.angles(position_ref)
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
                )
            attended = attended.reshape(n_tokens, self.dim).to(snapshot.dtype)
            x = snapshot + self.cross_layer_scale[layer] * self.out_proj[layer](
                attended
            )
            x = x + self.ffn[layer](self.ffn_norm[layer](x))
        if return_motion_displacement:
            return x, motion_displacement
        return x


class GaussianAttributeHead(nn.Module):
    """V3 time-invariant 2DGS decoder driven only by the shared token feature."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__()
        self.dim = int(dim)
        self.offset_bound = float(offset_bound)
        self.output_norm = nn.LayerNorm(self.dim)

        self.sizes = {
            name: int(getattr(gs_params, name))
            for name in ("shs", "opacity", "scaling", "rotation")
        }
        self.sizes["offset"] = int(getattr(gs_params, "offset", 0) or 0)
        if self.sizes["offset"] != 3:
            raise ValueError("Dynamic 2DGS requires p2g.gs_params.offset=3")
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.dim, width)
            for name, width in self.sizes.items()
        })
        self._initialize_outputs(cfg)

    def _initialize_outputs(self, cfg):
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)
        opacity = float(_cfg_get(cfg, "initial_opacity", 0.2))
        opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(
            self.heads["opacity"].bias,
            math.log(opacity / (1.0 - opacity)),
        )
        scale = float(_cfg_get(cfg, "initial_scale_m", 0.3))
        raw_scale = math.log(math.expm1(max(scale, 1.0e-4)))
        nn.init.constant_(self.heads["scaling"].bias, raw_scale)
        if self.sizes["rotation"] != 4:
            raise ValueError("Dynamic 2DGS requires a 4D quaternion rotation")
        with torch.no_grad():
            self.heads["rotation"].bias[0] = 1.0
        nn.init.zeros_(self.heads["offset"].weight)
        nn.init.zeros_(self.heads["offset"].bias)

    def _decode(self, feature):
        decoded = self.output_norm(feature)
        raw = {name: head(decoded) for name, head in self.heads.items()}
        offset = self.offset_bound * torch.tanh(raw.pop("offset"))
        return raw, offset, decoded

    def forward(self, feature):
        return self._decode(feature)


class SeedConditionedGaussianAttributeHead(GaussianAttributeHead):
    """Legacy V1 decoder retained for its checkpoint contract."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__(cfg, gs_params, dim, offset_bound)
        self.cell_size_m = float(_cfg_get(cfg, "cell_size_m", 0.8))
        geom_dim = int(_cfg_get(cfg, "geometry_dim", 64))
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, geom_dim),
            nn.SiLU(),
            nn.Linear(geom_dim, geom_dim),
        )
        self.feature_norm = nn.LayerNorm(self.dim)
        self.refine = nn.Sequential(
            nn.Linear(self.dim + geom_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feature, seed_delta_ref):
        geom = self.geometry_encoder(seed_delta_ref / self.cell_size_m)
        decoded = feature + self.refine(torch.cat([
            self.feature_norm(feature), geom,
        ], dim=-1))
        return self._decode(decoded)


class MotionConditionedVelocityHead(nn.Module):
    """Legacy v1 direct-velocity decoder kept for checkpoint compatibility."""

    def __init__(self, cfg, dim: int, time_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.max_speed_mps = float(_cfg_get(cfg, "max_speed_mps", 25.0))
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        duration_dim = int(_cfg_get(cfg, "duration_embedding_dim", 32))
        position_frequencies = int(_cfg_get(cfg, "position_frequencies", 4))
        position_range_m = float(_cfg_get(cfg, "position_range_m", 110.0))
        self.position_encoder = FourierPositionEncoder3D(
            position_frequencies, position_range_m
        )
        self.duration_encoder = nn.Sequential(
            nn.Linear(3, duration_dim),
            nn.SiLU(),
            nn.Linear(duration_dim, duration_dim),
        )
        self.feature_norm = nn.LayerNorm(self.dim)
        self.delta_norm = nn.LayerNorm(self.dim)
        input_dim = (
            2 * self.dim + int(time_dim) + duration_dim
            + self.position_encoder.out_dim
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(input_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(
        self,
        refined_feature,
        temporal_delta,
        time_embedding,
        duration_sec,
        position_ref,
    ):
        duration = duration_sec.reshape(-1, 1).clamp_min(1.0e-4)
        duration_stat = torch.cat([
            duration - 1.0,
            duration.log(),
            duration.reciprocal() - 1.0,
        ], dim=-1)
        duration_embedding = self.duration_encoder(duration_stat)
        motion_feature = self.motion_refiner(torch.cat([
            self.feature_norm(refined_feature),
            self.delta_norm(temporal_delta),
            time_embedding,
            duration_embedding,
            self.position_encoder(position_ref),
        ], dim=-1))
        velocity = self.max_speed_mps * torch.tanh(self.velocity(motion_feature))
        return velocity


class PhysicalVelocityHead(nn.Module):
    """Predict an unbounded physical velocity term in metres/second.

    V3 deliberately consumes the exact same shared token feature as the Gaussian
    head and uses this term as the complete velocity. V4 uses it as an additive
    residual over attention-derived ``velocity_init``. The final Linear starts at
    exactly zero; unlike the historical heads, no position encoding, tanh, or
    hand-chosen speed cap changes the output semantics or saturates its gradients.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.feature_norm = nn.LayerNorm(self.dim)
        self.motion_refiner = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature):
        motion_feature = self.motion_refiner(
            self.feature_norm(refined_feature)
        )
        return self.velocity(motion_feature)


class InitConditionedVelocityHead(nn.Module):
    """Predict a V6 residual from the refined feature and proposal velocity.

    ``velocity_init`` is scaled into the normalized feature domain and
    concatenated after feature LayerNorm.  The conditioning copy is detached so
    the residual head cannot reshape the proposal through an indirect shortcut;
    the additive ``velocity_init`` path still carries the final-velocity
    gradient into the proposal itself.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.init_condition_scale_mps = float(
            _cfg_get(cfg, "init_condition_scale_mps", 10.0)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if self.init_condition_scale_mps <= 0.0:
            raise ValueError("motion.init_condition_scale_mps must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.motion_refiner = nn.Sequential(
            nn.Linear(self.dim + 3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature, velocity_init):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V6 velocity"
            )
        if velocity_init.shape != (refined_feature.shape[0], 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")

        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        normalized_feature = self.feature_norm(refined_feature)
        normalized_init = (
            condition / self.init_condition_scale_mps
        ).to(dtype=normalized_feature.dtype)
        motion_feature = self.motion_refiner(torch.cat([
            normalized_feature,
            normalized_init,
        ], dim=-1))
        return self.velocity(motion_feature)


class SparseMotionProposal(nn.Module):
    """Time-free Siamese correspondence proposal for V6.

    The branch consumes frozen Utonia token features plus reference-frame token
    coordinates. Endpoint descriptors use one shared projection, so swapping
    the frames transposes the *raw* content/geometry score rather than invoking
    separately learned Q/K maps. Directional softmaxes are still distinct:
    they represent p(j | i) and p(i | j), whose denominators differ.

    Top-K bounds the candidate support after the soft distance bias. Reverse
    probability then softly reweights (rather than hard-rejects) each forward
    candidate, suppressing many-to-one winners while tolerating density changes.
    One learned, globally shared dustbin similarity participates in the same row
    softmax. Its probability is the only abstention signal; there is no separate
    concentration confidence or coordinate-cycle gate.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.descriptor_dim = int(_cfg_get(cfg, "descriptor_dim", 96))
        self.adapter_hidden_dim = int(
            _cfg_get(cfg, "adapter_hidden_dim", self.descriptor_dim)
        )
        self.candidate_count = int(_cfg_get(cfg, "candidate_count", 16))
        self.match_count = int(_cfg_get(cfg, "match_count", 4))
        self.score_chunk_size = int(_cfg_get(cfg, "score_chunk_size", 512))
        self.temperature = float(_cfg_get(cfg, "temperature", 0.07))
        self.search_speed_min_mps = float(
            _cfg_get(cfg, "search_speed_min_mps", 2.0)
        )
        self.search_speed_max_mps = float(
            _cfg_get(cfg, "search_speed_max_mps", 30.0)
        )
        search_speed_init = float(
            _cfg_get(cfg, "search_speed_init_mps", 8.0)
        )

        if self.descriptor_dim <= 0:
            raise ValueError("motion_proposal.descriptor_dim must be positive")
        if self.adapter_hidden_dim <= 0:
            raise ValueError(
                "motion_proposal.adapter_hidden_dim must be positive"
            )
        if self.candidate_count <= 0:
            raise ValueError("motion_proposal.candidate_count must be positive")
        if not 0 < self.match_count <= self.candidate_count:
            raise ValueError(
                "motion_proposal.match_count must be in [1, candidate_count]"
            )
        if self.score_chunk_size <= 0:
            raise ValueError("motion_proposal.score_chunk_size must be positive")
        if self.temperature <= 0.0:
            raise ValueError("motion_proposal.temperature must be positive")
        if not (
            0.0 < self.search_speed_min_mps
            < search_speed_init
            < self.search_speed_max_mps
        ):
            raise ValueError(
                "require 0 < search_speed_min_mps < search_speed_init_mps "
                "< search_speed_max_mps"
            )

        # No affine gamma here: this normalizer and the metric adapter belong
        # only to the motion proposal. The Gaussian/feature heads do not share
        # their scale. The adapter's zero-initialized up-projection makes the
        # entire residual path exactly zero at initialization, so the initial
        # descriptor still reads the normalized frozen Utonia representation.
        self.input_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.descriptor_adapter = nn.Sequential(
            nn.Linear(self.dim, self.adapter_hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.adapter_hidden_dim, self.dim, bias=False),
        )
        nn.init.zeros_(self.descriptor_adapter[-1].weight)
        self.descriptor = nn.Linear(
            self.dim, self.descriptor_dim, bias=False
        )
        nn.init.orthogonal_(self.descriptor.weight)

        # Cosine descriptors make one global no-match threshold meaningful.
        # tanh keeps it in the same [-1, 1] units as cosine and prevents the
        # fallback from winning merely by sending an unconstrained logit to
        # infinity. +log(M) below makes this a per-candidate threshold rather
        # than a threshold that changes when match_count changes.
        dustbin_similarity_init = float(
            _cfg_get(cfg, "dustbin_similarity_init", 0.70)
        )
        if not -1.0 < dustbin_similarity_init < 1.0:
            raise ValueError(
                "motion_proposal.dustbin_similarity_init must be in (-1, 1)"
            )
        self.dustbin_similarity_raw = nn.Parameter(torch.tensor(
            math.atanh(dustbin_similarity_init), dtype=torch.float32
        ))

        # One feature-conditioned scalar is enough to express the essential
        # static/dynamic distinction in the geometric prior. It is a search
        # speed, so multiplying by the physical endpoint interval gives metres.
        self.search_speed = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.search_speed.weight)
        fraction = (
            (search_speed_init - self.search_speed_min_mps)
            / (self.search_speed_max_mps - self.search_speed_min_mps)
        )
        nn.init.constant_(
            self.search_speed.bias,
            math.log(fraction / (1.0 - fraction)),
        )

    def _encode(self, feature, duration_sec):
        normalized = self.input_norm(feature)
        adapted = normalized + self.descriptor_adapter(normalized)
        descriptor = F.normalize(
            self.descriptor(adapted).float(), dim=-1, eps=1.0e-6
        )
        speed_span = self.search_speed_max_mps - self.search_speed_min_mps
        speed = self.search_speed_min_mps + speed_span * torch.sigmoid(
            self.search_speed(normalized).squeeze(-1).float()
        )
        radius = speed * duration_sec.float().clamp_min(1.0e-4)
        return descriptor, speed, radius

    def _pair_score(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        query_radius,
        key_radius,
    ):
        """Symmetric content score plus a soft, adaptive distance prior."""
        content = (
            query_descriptor @ key_descriptor.transpose(0, 1)
        ) / self.temperature

        # Centering changes neither pairwise displacement nor its gradient, but
        # avoids cancellation when ref-frame coordinates are large.
        origin = 0.5 * (
            query_position.float().mean(dim=0)
            + key_position.float().mean(dim=0)
        )
        query = query_position.float() - origin
        key = key_position.float() - origin
        squared_distance = (
            query.square().sum(dim=-1, keepdim=True)
            + key.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (query @ key.transpose(0, 1))
        ).clamp_min(0.0)
        variance = (
            query_radius.float().square().unsqueeze(1)
            + key_radius.float().square().unsqueeze(0)
        ).clamp_min(1.0e-6)
        return content - squared_distance / variance

    def _topk_direction(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        query_radius,
        key_radius,
    ):
        key_count = int(key_descriptor.shape[0])
        if key_count == 0:
            raise ValueError("motion proposal requires tokens in both endpoints")
        support = min(self.candidate_count, key_count)
        value_chunks = []
        index_chunks = []
        for start in range(
            0, int(query_descriptor.shape[0]), self.score_chunk_size
        ):
            end = min(start + self.score_chunk_size, query_descriptor.shape[0])
            score = self._pair_score(
                query_descriptor[start:end],
                key_descriptor,
                query_position[start:end],
                key_position,
                query_radius[start:end],
                key_radius,
            )
            values, indices = torch.topk(
                score, k=support, dim=-1, largest=True, sorted=True
            )
            value_chunks.append(values)
            index_chunks.append(indices)

        top_score = torch.cat(value_chunks, dim=0)
        candidate_index = torch.cat(index_chunks, dim=0)
        return {
            "candidate_index": candidate_index,
            "score": top_score,
            "conditional_probability": torch.softmax(top_score, dim=-1),
            "support": support,
        }

    def _finish_direction(self, direction, reverse, position, key_position):
        """Apply a soft reciprocal boost and form one dustbin-aware proposal.

        Missing reverse top-K membership receives a uniform 1/K prior instead
        of zero. Therefore reciprocal evidence can suppress a many-to-one
        winner, but a valid match is never hard-cut solely because sampling
        density or occlusion made the reverse list asymmetric.
        """
        candidate_index = direction["candidate_index"]
        reverse_candidate = reverse["candidate_index"][candidate_index]
        reverse_probability = reverse["conditional_probability"][candidate_index]
        query_index = torch.arange(
            candidate_index.shape[0], device=candidate_index.device
        )[:, None, None]
        is_reciprocal = reverse_candidate == query_index
        reciprocal_probability = (
            reverse_probability * is_reciprocal.to(reverse_probability.dtype)
        ).sum(dim=-1)

        reciprocal_prior = 1.0 / float(reverse["support"])
        mutual_logit = direction["score"] + (
            reciprocal_probability + reciprocal_prior
        ).log()

        # K is a recall-oriented candidate pool. Only M <= K candidates may
        # contribute coordinates, so a flat but semantically homogeneous pool
        # cannot average eleven or sixteen distant surfaces into one centroid.
        match_support = min(self.match_count, direction["support"])
        selected_logit, selected_slot = torch.topk(
            mutual_logit, k=match_support, dim=-1, largest=True, sorted=True
        )
        selected_candidate = candidate_index.gather(1, selected_slot)
        selected_reciprocal = reciprocal_probability.gather(1, selected_slot)

        dustbin_similarity = torch.tanh(self.dustbin_similarity_raw)
        dustbin_logit = (
            dustbin_similarity / self.temperature
            + math.log(float(match_support))
        )
        augmented_probability = torch.softmax(torch.cat([
            selected_logit,
            dustbin_logit.expand(selected_logit.shape[0], 1),
        ], dim=-1), dim=-1)
        real_weight = augmented_probability[:, :match_support]
        p_unmatched = augmented_probability[:, match_support]
        match_probability = real_weight.sum(dim=-1)
        conditional_weight = real_weight / match_probability.clamp_min(
            1.0e-12
        ).unsqueeze(-1)
        candidate_position = key_position.float()[selected_candidate]
        displacement = candidate_position - position.float().unsqueeze(1)
        delta_p_init = (real_weight.unsqueeze(-1) * displacement).sum(dim=1)
        delta_p_match = (
            conditional_weight.unsqueeze(-1) * displacement
        ).sum(dim=1)
        entropy = -(
            conditional_weight
            * conditional_weight.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "delta_p_match": delta_p_match,
            "delta_p_init": delta_p_init,
            "matched_position": position.float() + delta_p_match,
            "motion_top1_probability": conditional_weight.max(dim=-1).values,
            "motion_effective_support": entropy.exp(),
            "motion_reciprocal_probability": (
                conditional_weight * selected_reciprocal
            ).sum(dim=-1),
            "match_probability": match_probability,
            "p_unmatched": p_unmatched,
            "motion_dustbin_similarity": dustbin_similarity.expand(
                position.shape[0]
            ),
        }

    def _match_pair(
        self,
        descriptor0,
        descriptor1,
        position0,
        position1,
        radius0,
        radius1,
    ):
        forward = self._topk_direction(
            descriptor0, descriptor1, position0, position1, radius0, radius1
        )
        backward = self._topk_direction(
            descriptor1, descriptor0, position1, position0, radius1, radius0
        )

        return (
            self._finish_direction(forward, backward, position0, position1),
            self._finish_direction(backward, forward, position1, position0),
        )

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        duration_sec,
    ):
        n_tokens = int(feature.shape[0])
        if feature.shape != (n_tokens, self.dim):
            raise ValueError(f"motion proposal feature must be (N,{self.dim})")
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("motion proposal positions must align with features")
        if duration_sec.shape != (n_tokens,):
            raise ValueError(
                "motion proposal duration must provide one value per token"
            )
        valid_duration = torch.isfinite(duration_sec) & (duration_sec > 0.0)
        if not bool(torch.all(valid_duration)):
            raise ValueError("motion proposal requires positive finite durations")

        counts = torch.diff(token_offset, prepend=token_offset.new_zeros(1)).long()
        if int(counts.sum()) != n_tokens:
            raise ValueError("token_offset does not cover all motion proposal tokens")
        starts = torch.cumsum(counts, dim=0) - counts
        batch_frames = defaultdict(list)
        for frame, batch_id in enumerate(frame_batch_idx.tolist()):
            batch_frames[int(batch_id)].append(frame)
        for batch_id, frames in batch_frames.items():
            if len(frames) != 2:
                raise ValueError(
                    "motion proposal requires exactly two frames per sample; "
                    f"batch {batch_id} has {len(frames)}"
                )

        descriptor, search_speed, radius = self._encode(feature, duration_sec)
        by_frame = [None] * int(counts.numel())
        for frames in batch_frames.values():
            frame0, frame1 = frames
            start0, count0 = int(starts[frame0]), int(counts[frame0])
            start1, count1 = int(starts[frame1]), int(counts[frame1])
            rows0 = slice(start0, start0 + count0)
            rows1 = slice(start1, start1 + count1)
            result0, result1 = self._match_pair(
                descriptor[rows0], descriptor[rows1],
                position_ref[rows0], position_ref[rows1],
                radius[rows0], radius[rows1],
            )
            by_frame[frame0] = result0
            by_frame[frame1] = result1

        result = {
            name: torch.cat([frame[name] for frame in by_frame], dim=0)
            for name in by_frame[0]
        }
        result["motion_search_speed_mps"] = search_speed
        return result


class DynamicGaussianBackend(nn.Module):
    """Legacy direct-velocity v1 backend."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__()
        self.cfg = cfg
        self.dim = int(dim)
        self.temporal = ParallelBidirectionalCrossAttention(cfg.temporal, self.dim)
        gaussian_cfg = _cfg_get(cfg, "gaussian_head", None)
        self.gaussian_head = SeedConditionedGaussianAttributeHead(
            gaussian_cfg, gs_params, self.dim, offset_bound
        )
        self.velocity_head = MotionConditionedVelocityHead(
            cfg.motion,
            self.dim,
            self.temporal.time_encoder.out_dim,
        )

    @staticmethod
    def _frame_bounds(token_offset, frame):
        start = int(token_offset[frame - 1]) if frame > 0 else 0
        return start, int(token_offset[frame])

    def _reference_geometry_and_time(
        self,
        token_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
    ):
        device, dtype = token_sensor.device, token_sensor.dtype
        token_ref = torch.empty_like(token_sensor)
        seed_ref = torch.empty_like(seed_sensor)
        token_time_norm = torch.empty(token_sensor.shape[0], device=device, dtype=dtype)
        token_time_sec = torch.empty_like(token_time_norm)
        token_duration = torch.empty_like(token_time_norm)
        token_batch = torch.empty(
            token_sensor.shape[0], device=device, dtype=torch.long
        )
        local_frame_index = torch.empty_like(token_batch)
        seen = defaultdict(int)
        for global_frame, batch_id_value in enumerate(frame_batch_idx.tolist()):
            batch_id = int(batch_id_value)
            local_frame = seen[batch_id]
            seen[batch_id] += 1
            start, end = self._frame_bounds(token_offset, global_frame)
            pose = pose_list[batch_id][local_frame].to(device=device, dtype=dtype)
            token_ref[start:end] = box_utils.apply_pose(
                token_sensor[start:end], pose
            )
            seed_ref[start:end] = box_utils.apply_pose(
                seed_sensor[start:end], pose
            )
            t_norm = timestamps_normalized[batch_id].to(device=device, dtype=dtype)
            t_sec = timestamps_sec[batch_id].to(device=device, dtype=dtype)
            if local_frame >= t_norm.numel() or local_frame >= t_sec.numel():
                raise ValueError("source timestamp count does not match input frames")
            token_time_norm[start:end] = t_norm[local_frame]
            token_time_sec[start:end] = t_sec[local_frame]
            duration = torch.as_tensor(
                window_duration_sec[batch_id], device=device, dtype=dtype
            )
            token_duration[start:end] = duration
            token_batch[start:end] = batch_id
            local_frame_index[start:end] = local_frame
        return {
            "token_ref": token_ref,
            "seed_ref": seed_ref,
            "time_normalized": token_time_norm,
            "time_sec": token_time_sec,
            "duration_sec": token_duration,
            "batch": token_batch,
            "local_frame": local_frame_index,
        }

    def forward(
        self,
        fused_feature,
        token_position_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
    ):
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(
                f"dynamic fused_feature must be (N,{self.dim})"
            )
        if seed_sensor.ndim == 3:
            if seed_sensor.shape[1:] != (1, 3):
                raise ValueError("Dynamic 2DGS requires exactly one seed per token")
            seed_sensor = seed_sensor[:, 0]
        if seed_sensor.shape != token_position_sensor.shape:
            raise ValueError("dynamic seed and token positions must align")

        geometry = self._reference_geometry_and_time(
            token_position_sensor,
            seed_sensor,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
        )
        refined, temporal_delta, time_embedding = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["time_normalized"],
        )
        raw, position_offset, _gaussian_feature = self.gaussian_head(
            refined,
            geometry["seed_ref"] - geometry["token_ref"],
        )
        position = geometry["seed_ref"] + position_offset
        velocity = self.velocity_head(
            refined,
            temporal_delta,
            time_embedding,
            geometry["duration_sec"],
            position,
        )

        batch_gaussians = []
        for batch_id in range(len(pose_list)):
            rows = (geometry["batch"] == batch_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                batch_gaussians.append(None)
                continue
            n = int(rows.numel())
            motion_mask = torch.ones(n, dtype=torch.bool, device=rows.device)
            batch_gaussians.append({
                "position": position[rows],
                "coord": position[rows],
                "coord_ref": position[rows],
                "shs": raw["shs"][rows],
                "opacity": raw["opacity"][rows],
                "scaling": raw["scaling"][rows],
                "rotation": raw["rotation"][rows],
                "velocity": velocity[rows],
                "source_time_sec": geometry["time_sec"][rows],
                "source_time_normalized": geometry["time_normalized"][rows],
                "window_duration_sec": geometry["duration_sec"][rows],
                # Diagnostics only: this is predicted-speed thresholding, not
                # bbox assignment and not a rendering gate.
                "is_dynamic": motion_mask,
                "instance_id": torch.full(
                    (n,), -1, dtype=torch.long, device=rows.device
                ),
                "bg_mask": ~motion_mask,
                "fg_masks": {},
                "frame_bboxes": [],
            })

        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": geometry["batch"],
        }


class PhysicalVelocityGaussianBackend(DynamicGaussianBackend):
    """Current bbox-free backend with physical-time conditioning and direct m/s."""

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.dim = int(dim)
        self.temporal = TimeConditionedParallelCrossAttention(
            cfg.temporal, self.dim
        )
        self.time_reference_sec = float(
            _cfg_get(cfg.temporal, "time_reference_sec", 1.0)
        )
        if self.time_reference_sec <= 0.0:
            raise ValueError("time_reference_sec must be positive")
        self.gaussian_head = GaussianAttributeHead(
            _cfg_get(cfg, "gaussian_head", None),
            gs_params,
            self.dim,
            offset_bound,
        )
        self.velocity_head = PhysicalVelocityHead(cfg.motion, self.dim)

    def _temporal_coordinate(self, geometry):
        # A fixed reference unit preserves cadence: endpoints 0.8 s apart are
        # encoded as [0, 0.8], not collapsed to [0, 1].
        return geometry["time_sec"] / self.time_reference_sec

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
        )
        return refined, {}

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del geometry, position, temporal_fields
        return self.velocity_head(refined), {}

    def _predict_gaussian(self, refined, geometry):
        del geometry
        return self.gaussian_head(refined)

    def forward(
        self,
        fused_feature,
        token_position_sensor,
        seed_sensor,
        token_offset,
        frame_batch_idx,
        pose_list,
        timestamps_normalized,
        timestamps_sec,
        window_duration_sec,
        motion_proposal_feature=None,
    ):
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        if seed_sensor.ndim == 3:
            if seed_sensor.shape[1:] != (1, 3):
                raise ValueError("Dynamic 2DGS requires exactly one seed per token")
            seed_sensor = seed_sensor[:, 0]
        if seed_sensor.shape != token_position_sensor.shape:
            raise ValueError("dynamic seed and token positions must align")

        geometry = self._reference_geometry_and_time(
            token_position_sensor,
            seed_sensor,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
            motion_proposal_feature,
        )
        raw, position_offset, _gaussian_feature = self._predict_gaussian(
            refined, geometry
        )
        position = geometry["seed_ref"] + position_offset
        velocity, motion_fields = self._predict_motion(
            refined, geometry, position, temporal_fields
        )

        batch_gaussians = []
        for batch_id in range(len(pose_list)):
            rows = (geometry["batch"] == batch_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                batch_gaussians.append(None)
                continue
            n = int(rows.numel())
            motion_mask = torch.ones(n, dtype=torch.bool, device=rows.device)
            batch_gaussians.append({
                "position": position[rows],
                "coord": position[rows],
                "coord_ref": position[rows],
                "shs": raw["shs"][rows],
                "opacity": raw["opacity"][rows],
                "scaling": raw["scaling"][rows],
                "rotation": raw["rotation"][rows],
                "velocity": velocity[rows],
                **{name: value[rows] for name, value in motion_fields.items()},
                "source_time_sec": geometry["time_sec"][rows],
                "source_time_normalized": geometry["time_normalized"][rows],
                "window_duration_sec": geometry["duration_sec"][rows],
                "is_dynamic": motion_mask,
                "instance_id": torch.full(
                    (n,), -1, dtype=torch.long, device=rows.device
                ),
                "bg_mask": ~motion_mask,
                "fg_masks": {},
                "frame_bboxes": [],
            })
        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": geometry["batch"],
        }


class AttentionInitializedVelocityGaussianBackend(PhysicalVelocityGaussianBackend):
    """V4 physical velocity with a four-head correspondence initialization.

    The selected heads remain part of the ordinary all-head feature update.
    Their final-layer Q/K probabilities are additionally applied to ref-frame
    token xyz to produce a source-to-opposite-frame displacement. Dividing by
    the signed source-to-opposite timestamp interval converts both endpoint
    directions to one forward-time velocity convention. The V3 velocity head
    is retained as a zero-initialized residual in metres/second.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__(cfg, gs_params, dim, offset_bound)
        if self.temporal.motion_head_count <= 0:
            raise ValueError(
                "V4 attention velocity requires temporal.motion_head_count > 0"
            )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        refined, delta_p_init = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            return_motion_displacement=True,
        )
        return refined, {"delta_p_init": delta_p_init}

    @staticmethod
    def _signed_pair_delta_t(geometry):
        # Dynamic inputs contain exactly two endpoint frames per sample. Their
        # physical timestamps are relative to the same window start, so local
        # frame 0 travels forward by +duration and frame 1 backward by -duration.
        local_frame = geometry["local_frame"]
        if not bool(torch.all((local_frame == 0) | (local_frame == 1))):
            raise ValueError("V4 attention velocity requires endpoint indices 0/1")
        duration = geometry["duration_sec"]
        if not bool(torch.all(torch.isfinite(duration) & (duration > 0.0))):
            raise ValueError("V4 attention velocity requires positive finite duration")
        return torch.where(local_frame == 0, duration, -duration)

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        delta_p_init = temporal_fields["delta_p_init"]
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_residual = self.velocity_head(refined)
        velocity = velocity_init + velocity_residual
        return velocity, {
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_init": velocity_init,
            "velocity_residual": velocity_residual,
        }


class ProposalInitializedVelocityGaussianBackend(PhysicalVelocityGaussianBackend):
    """V6 velocity from a sparse proposal plus an init-aware residual.

    Correspondence is computed from frozen Utonia tokens before the trainable
    fusion/refiner and before the temporal module adds time. The ordinary
    temporal stack refines attributes and supplies the feature for a velocity
    residual conditioned on the dustbin-aware proposal velocity. Dustbin
    probability remains a soft proposal-confidence gate; it no longer gates the
    residual itself.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 proposal_dim: int | None = None):
        super().__init__(cfg, gs_params, dim, offset_bound)
        if self.temporal.motion_head_count != 0:
            raise ValueError(
                "V6 motion proposal is independent; temporal.motion_head_count "
                "must be omitted or zero"
            )
        self.motion_proposal = SparseMotionProposal(
            cfg.motion_proposal,
            self.dim if proposal_dim is None else int(proposal_dim),
        )
        # Old V6 checkpoints do not carry this flag and retain their original
        # feature-only head and dustbin-gated fallback exactly. Fresh configs
        # opt into the init-conditioned additive residual explicitly.
        self.init_conditioned_residual = bool(
            _cfg_get(cfg.motion, "init_conditioned_residual", False)
        )
        if self.init_conditioned_residual:
            self.velocity_head = InitConditionedVelocityHead(
                cfg.motion, self.dim
            )

    @staticmethod
    def _signed_pair_delta_t(geometry):
        local_frame = geometry["local_frame"]
        if not bool(torch.all((local_frame == 0) | (local_frame == 1))):
            raise ValueError("V6 motion proposal requires endpoint indices 0/1")
        duration = geometry["duration_sec"]
        if not bool(torch.all(torch.isfinite(duration) & (duration > 0.0))):
            raise ValueError("V6 motion proposal requires positive finite duration")
        return torch.where(local_frame == 0, duration, -duration)

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        if motion_proposal_feature is None:
            raise ValueError("V6 requires frozen Utonia motion proposal features")
        # This call is intentionally before self.temporal: descriptors never see
        # trainable feature fusion, endpoint time embedding, or cross-frame
        # feature attention.
        proposal_fields = self.motion_proposal(
            motion_proposal_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["duration_sec"],
        )
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
        )
        return refined, proposal_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_match = temporal_fields["delta_p_match"]
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_match = delta_p_match / pair_delta_t_sec.unsqueeze(-1)
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        if self.init_conditioned_residual:
            velocity_offset = self.velocity_head(refined, velocity_init)
            velocity = velocity_init + velocity_offset
        else:
            # Checkpoint compatibility for the original V6 formulation.
            velocity_offset = self.velocity_head(refined)
            velocity = (
                velocity_init
                + temporal_fields["p_unmatched"].unsqueeze(-1) * velocity_offset
            )

        fields = {
            "delta_p_match": delta_p_match,
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": velocity_match,
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        fields.update({
            name: temporal_fields[name]
            for name in (
                "matched_position",
                "motion_top1_probability",
                "motion_effective_support",
                "motion_reciprocal_probability",
                "motion_dustbin_similarity",
                "motion_search_speed_mps",
                "match_probability",
                "p_unmatched",
            )
        })
        return velocity, fields


class DynamicGausTemp(nn.Module):
    """Validate and expose the physical-time Gaussian trajectory contract."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, timestamps_sec=None, window_duration_sec=None):
        batch_gaussians = x.get("batch_gaussians", x.get("gaussians"))
        if batch_gaussians is None:
            raise KeyError("DynamicGausTemp expected batch_gaussians")
        results = []
        for batch_id, item in enumerate(batch_gaussians):
            if item is None:
                results.append(None)
                continue
            for key in ("position", "velocity", "source_time_sec"):
                if key not in item:
                    raise KeyError(f"Dynamic Gaussian batch is missing {key!r}")
            if item["velocity"].shape != item["position"].shape:
                raise ValueError("velocity must align one-to-one with Gaussian position")
            result = {**item}
            if timestamps_sec is not None:
                result["segment_timestamps_sec"] = timestamps_sec[batch_id].to(
                    device=item["position"].device,
                    dtype=item["position"].dtype,
                )
            if window_duration_sec is not None:
                result["segment_duration_sec"] = torch.as_tensor(
                    window_duration_sec[batch_id],
                    device=item["position"].device,
                    dtype=item["position"].dtype,
                )
            results.append(result)
        return results


__all__ = [
    "AttentionInitializedVelocityGaussianBackend",
    "DynamicGaussianBackend",
    "DynamicGausTemp",
    "GaussianAttributeHead",
    "MotionConditionedVelocityHead",
    "PhysicalVelocityGaussianBackend",
    "PhysicalVelocityHead",
    "InitConditionedVelocityHead",
    "ParallelBidirectionalCrossAttention",
    "ProposalInitializedVelocityGaussianBackend",
    "SeedConditionedGaussianAttributeHead",
    "SinusoidalScalarEncoder",
    "SparseMotionProposal",
    "TimeConditionedParallelCrossAttention",
]
