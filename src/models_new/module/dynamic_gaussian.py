"""BBox-free Dynamic 2DGS modules.

The shared Point2Gaus frontend has already produced one fused feature per
occupied Utonia grid token by the time this module runs.  This backend:

1. moves each frame's token and observed medoid seed into the common ref frame;
2. refines both endpoint token sets with synchronous, bidirectional full
   cross-attention;
3. predicts time-invariant 2D Gaussian attributes from the refined token
   feature and observed own-frame seed geometry;
4. predicts motion with a separate head.

V1-V9 emit exactly one Gaussian per token; V10 adaptively emits one to three.
Rotation, scale, opacity, and SH attributes remain fixed over time; only
position is transported linearly by the renderer. ``direct_velocity_v1`` remains for
historical checkpoint reconstruction. V3/V3.1 directly regress velocity in
metres/second. V4/V5 reuse heads from the final temporal layer as a
differentiable soft-correspondence initializer. V6 instead keeps correspondence
outside feature cross-attention: one shared descriptor projection reads frozen
Utonia tokens, selects a recall-oriented global candidate pool under a soft
geometric prior, and uses reciprocal re-ranking plus a dustbin to restrict the
actual coordinate mixture to a small cluster. V7 removes the learned descriptor
adapter/projection, predicts unmatched probability from bidirectional matching
evidence outside the candidate softmax, and warps temporal RoPE queries with the
detached hard-gated initializer before predicting an additive velocity offset.
V7.2 keeps V7's pre-attention direct Utonia descriptors, replaces the learned
tokenwise spatial prior with one fixed 30 m/s envelope, and removes reciprocal
re-ranking and the unmatched branch. It reads coordinates from one direct Top-4
support in the forward pass while a full-key softmax supplies the backward
surrogate. V8 instead removes the pre-attention proposal and
reuses the last temporal layer's first four dense attention heads: their
probabilities are averaged before one reciprocal K-to-M coordinate readout,
followed by the same additive offset form. V9 keeps the proposal a separate
branch like V6/V7 but moves it behind temporal refinement: a learned
LayerNorm+projection descriptor reads the refined feature and every key
contributes to one plain dense coordinate expectation, with no Top-K support,
straight-through surrogate, or query warp. V10 replaces 3D RoPE with the fixed
quadratic distance bias, averages all heads at every temporal layer, and uses a
query-only global token to predict a per-frame softmax over those layer maps.
Its refined token then drives the legacy hard-Gumbel-STE K={1,2,3} Gaussian
router, while one token velocity is shared by every selected child Gaussian.
V11 keeps that router and offset head but splits the position encoding across
heads: head 0 alone carries correspondence and is biased by a max-speed soft
barrier instead of RoPE, heads 1-11 keep metric 3D RoPE, and the twelve head-0
coordinate readouts are mixed by a softmax over twelve plain learned scalars.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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


class ConsensusAttentionMotionMatcher(nn.Module):
    """Sparse coordinate readout from final-layer feature-attention heads.

    The ordinary feature update remains a dense all-head softmax.  This module
    only re-reads the already RoPE-rotated Q/K tensors from the selected heads:

    1. compute each head's dense directional row-softmax;
    2. average those probabilities into one consensus distribution;
    3. retain a recall-oriented Top-K pool from that one distribution;
    4. re-rank with the reverse directional probability and retain Top-M;
    5. form one soft coordinate expectation over those M tokens.

    Top-K and Top-M therefore select support only once per token, rather than
    once per head.  With four heads and M=4, at most four (not sixteen)
    opposite-frame coordinates contribute to the initializer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.candidate_count = int(_cfg_get(cfg, "candidate_count", 16))
        self.match_count = int(_cfg_get(cfg, "match_count", 4))
        self.score_chunk_size = int(_cfg_get(cfg, "score_chunk_size", 128))
        if self.candidate_count <= 0:
            raise ValueError("motion_matching.candidate_count must be positive")
        if not 0 < self.match_count <= self.candidate_count:
            raise ValueError(
                "motion_matching.match_count must be in [1, candidate_count]"
            )
        if self.score_chunk_size <= 0:
            raise ValueError("motion_matching.score_chunk_size must be positive")

    def _topk_direction(self, query, key, scale):
        """Return Top-K of the equal-head-mean dense attention probability."""
        if query.ndim != 3 or key.ndim != 3:
            raise ValueError("consensus matching Q/K must have shape (N,H,D)")
        if query.shape[1:] != key.shape[1:]:
            raise ValueError("consensus matching Q/K head shapes must agree")
        key_count = int(key.shape[0])
        if key_count == 0:
            raise ValueError("consensus matching requires tokens in both frames")
        support = min(self.candidate_count, key_count)
        probability_chunks = []
        index_chunks = []
        entropy_chunks = []
        disagreement_chunks = []
        candidate_mass_chunks = []
        key_float = key.float()

        def consensus_topk(query_chunk, all_keys):
            logits = torch.einsum(
                "qhd,khd->qhk", query_chunk.float(), all_keys
            ) * float(scale)
            # Average probabilities, not logits: this is an equal vote over the
            # actual dense attention distributions used by the four heads.
            head_probability = torch.softmax(logits, dim=-1)
            consensus_probability = head_probability.mean(dim=1)
            probability, index = torch.topk(
                consensus_probability,
                k=support,
                dim=-1,
                largest=True,
                sorted=True,
            )
            consensus_entropy = -(
                consensus_probability
                * consensus_probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            mean_head_entropy = -(
                head_probability
                * head_probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1).mean(dim=-1)
            disagreement = (
                consensus_entropy - mean_head_entropy
            ).clamp_min(0.0)
            return (
                probability,
                index,
                consensus_entropy,
                disagreement,
                probability.sum(dim=-1),
            )

        for start in range(0, int(query.shape[0]), self.score_chunk_size):
            end = min(start + self.score_chunk_size, int(query.shape[0]))
            query_chunk = query[start:end]
            if torch.is_grad_enabled() and (
                query_chunk.requires_grad or key_float.requires_grad
            ):
                # The dense feature path is FlashAttention-memory-efficient.
                # Do not reintroduce O(N^2) saved softmax activations merely to
                # expose coordinate probabilities: retain only these compact
                # outputs and recompute the dense chunk during backward.
                outputs = checkpoint(
                    consensus_topk,
                    query_chunk,
                    key_float,
                    use_reentrant=False,
                )
            else:
                outputs = consensus_topk(query_chunk, key_float)
            (
                probability,
                index,
                consensus_entropy,
                disagreement,
                candidate_mass,
            ) = outputs
            probability_chunks.append(probability)
            index_chunks.append(index)
            entropy_chunks.append(consensus_entropy)
            # H(mean p_h) - mean H(p_h) is the Jensen-Shannon divergence
            # between heads. It is zero only when their distributions agree.
            disagreement_chunks.append(disagreement)
            candidate_mass_chunks.append(candidate_mass)
        probability = torch.cat(probability_chunks, dim=0)
        candidate_probability_mass = torch.cat(
            candidate_mass_chunks, dim=0
        )
        # Preserve the full-key Top-K mass as a concentration diagnostic, but
        # use a K-conditional distribution for reciprocal matching. This puts
        # reverse probabilities on the same 1/K scale as V7 without discarding
        # how much global attention mass the candidate pool captured.
        conditional_probability = probability / (
            candidate_probability_mass.clamp_min(1.0e-12).unsqueeze(-1)
        )
        return {
            "candidate_index": torch.cat(index_chunks, dim=0),
            "probability": probability,
            "conditional_probability": conditional_probability,
            "consensus_entropy": torch.cat(entropy_chunks, dim=0),
            "head_js_divergence": torch.cat(disagreement_chunks, dim=0),
            "candidate_probability_mass": candidate_probability_mass,
            "support": support,
        }

    def _finish_direction(self, direction, reverse, position, key_position):
        candidate_index = direction["candidate_index"]

        # candidate_index[q, k] is a query row in the reverse direction.  Look
        # for the original q among that row's reverse Top-K candidates and use
        # its K-conditional consensus probability when it is present.
        reverse_candidate = reverse["candidate_index"][candidate_index]
        reverse_probability = reverse[
            "conditional_probability"
        ][candidate_index]
        query_index = torch.arange(
            candidate_index.shape[0], device=candidate_index.device
        )[:, None, None]
        # Membership is a hard, non-differentiable Top-K decision. The gathered
        # conditional probabilities are deliberately not detached: reciprocal
        # pairs train reverse Q/K, while absent pairs receive a zero mask.
        reciprocal_probability = (
            reverse_probability
            * (reverse_candidate == query_index).to(reverse_probability.dtype)
        ).sum(dim=-1)

        # p_fwd * (p_reverse + 1/K) in log space.  The uniform floor prevents
        # density/occlusion from making reverse Top-K membership a hard reject;
        # it does not add a coordinate hypothesis or a dustbin.
        reciprocal_prior = 1.0 / float(reverse["support"])
        mutual_log_probability = (
            direction["conditional_probability"].clamp_min(1.0e-12).log()
            + (reciprocal_probability + reciprocal_prior).log()
        )
        match_support = min(self.match_count, direction["support"])
        ranked_count = min(match_support + 1, direction["support"])
        ranked_log_probability, ranked_slot = torch.topk(
            mutual_log_probability,
            k=ranked_count,
            dim=-1,
            largest=True,
            sorted=True,
        )
        selected_log_probability = ranked_log_probability[:, :match_support]
        selected_slot = ranked_slot[:, :match_support]
        selected_candidate = candidate_index.gather(1, selected_slot)
        selected_reciprocal = reciprocal_probability.gather(1, selected_slot)
        selected_forward = direction["conditional_probability"].gather(
            1, selected_slot
        )
        weight = torch.softmax(selected_log_probability, dim=-1)
        candidate_position = key_position.float()[selected_candidate]
        matched_position = (
            weight.unsqueeze(-1) * candidate_position
        ).sum(dim=1)
        displacement = matched_position - position.float()
        entropy = -(
            weight * weight.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_top1_probability": weight.max(dim=-1).values,
            "motion_effective_support": entropy.exp(),
            "motion_reciprocal_probability": (
                weight * selected_reciprocal
            ).sum(dim=-1),
            "motion_selected_forward_probability": (
                weight * selected_forward
            ).sum(dim=-1),
            "motion_consensus_entropy": direction["consensus_entropy"],
            "motion_head_js_divergence": direction["head_js_divergence"],
            "motion_candidate_probability_mass": (
                direction["candidate_probability_mass"]
            ),
            "motion_selection_log_margin": (
                selected_log_probability[:, -1]
                - ranked_log_probability[:, match_support]
                if ranked_count > match_support
                else torch.zeros_like(selected_log_probability[:, -1])
            ),
            "motion_candidate_support": torch.full(
                (position.shape[0],),
                float(direction["support"]),
                device=position.device,
                dtype=weight.dtype,
            ),
            "motion_match_support": torch.full(
                (position.shape[0],),
                float(match_support),
                device=position.device,
                dtype=weight.dtype,
            ),
        }

    def forward(
        self,
        query,
        key,
        position_ref,
        memory_rows,
        query_counts,
        key_counts,
        *,
        scale,
    ):
        frame_count = int(query_counts.numel())
        if frame_count == 0 or frame_count != int(key_counts.numel()):
            raise ValueError("consensus matching requires aligned frame counts")
        if bool(torch.any(query_counts <= 0)) or bool(torch.any(key_counts <= 0)):
            raise ValueError("consensus matching requires non-empty endpoint frames")

        query_starts = torch.cumsum(query_counts, dim=0) - query_counts
        key_starts = torch.cumsum(key_counts, dim=0) - key_counts
        frame_for_start = {
            int(query_starts[frame]): frame for frame in range(frame_count)
        }
        opposite = []
        directions = []
        key_positions = []
        for frame in range(frame_count):
            q_start = int(query_starts[frame])
            q_end = q_start + int(query_counts[frame])
            k_start = int(key_starts[frame])
            k_end = k_start + int(key_counts[frame])
            memory = memory_rows[k_start:k_end]
            opposite_start = int(memory[0])
            if opposite_start not in frame_for_start:
                raise ValueError("invalid opposite-frame row layout")
            opposite.append(frame_for_start[opposite_start])
            directions.append(self._topk_direction(
                query[q_start:q_end], key[k_start:k_end], scale
            ))
            key_positions.append(position_ref[memory])

        by_frame = []
        for frame in range(frame_count):
            q_start = int(query_starts[frame])
            q_end = q_start + int(query_counts[frame])
            reverse_frame = opposite[frame]
            if opposite[reverse_frame] != frame:
                raise ValueError("opposite-frame mapping must be reciprocal")
            by_frame.append(self._finish_direction(
                directions[frame],
                directions[reverse_frame],
                position_ref[q_start:q_end],
                key_positions[frame],
            ))
        return {
            name: torch.cat([frame[name] for frame in by_frame], dim=0)
            for name in by_frame[0]
        }


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
        self.tie_motion_qk_init = bool(
            _cfg_get(cfg, "tie_motion_qk_init", False)
        )
        if self.tie_motion_qk_init:
            if self.motion_head_count <= 0:
                raise ValueError(
                    "tie_motion_qk_init requires motion_head_count > 0"
                )
            # V8 reads correspondence only from the final layer's leading
            # motion heads. Give those heads a symmetric descriptor geometry
            # at initialization, but keep Q and K as distinct Parameters so
            # rendering/correspondence gradients can specialize them later.
            motion_width = self.motion_head_count * self.head_dim
            with torch.no_grad():
                self.k_proj[-1].weight[:motion_width].copy_(
                    self.q_proj[-1].weight[:motion_width]
                )
                self.k_proj[-1].bias[:motion_width].copy_(
                    self.q_proj[-1].bias[:motion_width]
                )
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

        self.qk_norm = bool(_cfg_get(cfg, "qk_norm", True))
        if not self.qk_norm:
            raise ValueError("V10 requires qk_norm=true")
        self.q_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])

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

        self.qk_norm = bool(_cfg_get(cfg, "qk_norm", True))
        if not self.qk_norm:
            raise ValueError("V11 requires qk_norm=true")
        self.q_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])
        self.k_norm = nn.ModuleList([
            nn.RMSNorm(self.head_dim) for _ in range(self.n_layers)
        ])

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
            raise ValueError(f"V11 feature must be (N,{self.dim})")
        if position_ref.shape != (n_tokens, 3):
            raise ValueError("position_ref must align with V11 token features")
        if token_time_coordinate.shape != (n_tokens,):
            raise ValueError("V11 token time must provide one scalar per token")
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

        x = feature + self.time_to_feature(
            self.time_encoder(token_time_coordinate)
        ).to(feature.dtype)

        query_position = position_ref.float()
        key_position = position_ref[memory_rows].float()
        q_angles = self.rope.angles(query_position)
        k_angles = self.rope.angles(key_position)
        # Resolve the ragged layout and the barrier radius to Python scalars
        # once. Re-reading CUDA counts inside the layer loop would otherwise
        # cost several synchronizing host round-trips per temporal layer.
        q_count_values = [int(value) for value in q_counts.tolist()]
        k_count_values = [int(value) for value in k_counts.tolist()]
        frame_duration = self._frame_first(token_duration_sec, q_counts)
        if not torch.equal(
            torch.repeat_interleave(frame_duration, q_counts),
            token_duration_sec,
        ):
            raise ValueError(
                "V11 requires one endpoint interval per frame; the barrier "
                "radius is a per-sample physical duration"
            )
        # What an object at the admissible top speed could cover over this
        # sample's endpoint interval.
        frame_radius_values = [
            max(self.barrier_speed_mps * float(value), 1.0e-4)
            for value in frame_duration.tolist()
        ]

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
                "V11 temporal attention requested flash_varlen, but "
                "flash_attn is unavailable"
            )

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

    ``residual_hidden_dim`` picks the refiner width. Omitting it keeps the
    historical two ``dim``-wide layers so existing V6 checkpoints rebuild
    unchanged; a positive value builds one bottleneck layer instead, i.e.
    ``Linear(dim+3, h) -> SiLU -> Linear(h, 3)``. The offset is a 3-vector
    residual over a proposal that already carries the motion, so the wide
    variant spends ~9x the parameters on a target the regularizer holds near
    zero anyway.
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

        hidden_dim = _cfg_get(cfg, "residual_hidden_dim", None)
        if hidden_dim is not None and int(hidden_dim) <= 0:
            raise ValueError(
                "motion.residual_hidden_dim must be positive when set"
            )
        self.residual_hidden_dim = (
            None if hidden_dim is None else int(hidden_dim)
        )

        self.feature_norm = nn.LayerNorm(self.dim)
        if self.residual_hidden_dim is None:
            self.motion_refiner = nn.Sequential(
                nn.Linear(self.dim + 3, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.dim),
                nn.SiLU(),
            )
            output_dim = self.dim
        else:
            self.motion_refiner = nn.Sequential(
                nn.Linear(self.dim + 3, self.residual_hidden_dim),
                nn.SiLU(),
            )
            output_dim = self.residual_hidden_dim
        self.velocity = nn.Linear(output_dim, 3)
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


class EmbeddedInitVelocityHead(nn.Module):
    """V8 offset from the refined token and a detached velocity initializer.

    The current estimate is embedded explicitly because the dense feature
    value path does not contain the metric coordinate expectation itself.  The
    conditioning copy is detached; the additive ``v_init + v_offset`` path
    outside this module remains the only offset-loss route into matching.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.velocity_embedding_dim = int(
            _cfg_get(cfg, "velocity_embedding_dim", 32)
        )
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if self.velocity_embedding_dim <= 0:
            raise ValueError("V8 velocity_embedding_dim must be positive")
        if self.residual_hidden_dim <= 0:
            raise ValueError("V8 residual_hidden_dim must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, self.velocity_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.velocity_embedding_dim,
                self.velocity_embedding_dim,
            ),
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(
                self.dim + self.velocity_embedding_dim,
                self.residual_hidden_dim,
            ),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature, velocity_init):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V8 velocity"
            )
        if velocity_init.shape != (refined_feature.shape[0], 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")
        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        feature = self.feature_norm(refined_feature)
        velocity_embedding = self.velocity_encoder(
            condition.to(dtype=feature.dtype)
        )
        motion_feature = self.motion_refiner(torch.cat([
            feature,
            velocity_embedding,
        ], dim=-1))
        return self.velocity(motion_feature)


class FeatureOnlyVelocityOffsetHead(nn.Module):
    """V7.2 compact offset from only the time-refined token feature.

    The complete path is ``LN(feature) -> Linear -> SiLU -> Linear(3)``:
    exactly one nonlinear activation and no explicit initializer or duration
    conditioning. V7.2 exposes time to the preceding cross-attention instead,
    while the detached initializer Q warp makes the refined feature depend on
    the current correspondence lookup. The final Linear is zero initialized,
    so a fresh model still starts exactly at ``v_final = v_init``.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        if self.dim <= 0 or self.residual_hidden_dim <= 0:
            raise ValueError("V7.2 feature-only offset dimensions must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.motion_refiner = nn.Sequential(
            nn.Linear(self.dim, self.residual_hidden_dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def forward(self, refined_feature):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V7.2 velocity"
            )
        motion_feature = self.motion_refiner(
            self.feature_norm(refined_feature)
        )
        return self.velocity(motion_feature)


class EmbeddedInitDurationVelocityHead(nn.Module):
    """V7 additive offset head with explicit velocity and duration embeddings.

    The input contract is exactly ``LN(refined_feature)`` concatenated with
    ``MLP(v_init): 3 -> Dv -> SiLU -> Dv`` and
    ``MLP(Fourier4(duration)): 8 -> Dt -> SiLU -> Dt``. Duration Fourier
    features contain sin/cos only (no raw scalar), and physical seconds are
    kept raw when ``duration_reference_sec=1``. The conditioning copy of
    ``v_init`` is detached by default; its additive path outside this module
    still trains the proposal.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.velocity_embedding_dim = int(
            _cfg_get(cfg, "velocity_embedding_dim", 32)
        )
        self.duration_embedding_dim = int(
            _cfg_get(cfg, "duration_embedding_dim", 16)
        )
        self.duration_frequency_count = int(
            _cfg_get(cfg, "duration_frequencies", 4)
        )
        self.duration_reference_sec = float(
            _cfg_get(cfg, "duration_reference_sec", 1.0)
        )
        self.residual_hidden_dim = int(
            _cfg_get(cfg, "residual_hidden_dim", 96)
        )
        self.detach_init_condition = bool(
            _cfg_get(cfg, "detach_init_condition", True)
        )
        if min(
            self.velocity_embedding_dim,
            self.duration_embedding_dim,
            self.duration_frequency_count,
            self.residual_hidden_dim,
        ) <= 0:
            raise ValueError("V7 motion embedding dimensions must be positive")
        if self.duration_reference_sec <= 0.0:
            raise ValueError("motion.duration_reference_sec must be positive")

        self.feature_norm = nn.LayerNorm(self.dim)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, self.velocity_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.velocity_embedding_dim,
                self.velocity_embedding_dim,
            ),
        )
        duration_input_dim = 2 * self.duration_frequency_count
        self.duration_encoder = nn.Sequential(
            nn.Linear(duration_input_dim, self.duration_embedding_dim),
            nn.SiLU(),
            nn.Linear(
                self.duration_embedding_dim,
                self.duration_embedding_dim,
            ),
        )
        self.register_buffer(
            "duration_frequency_bands",
            math.pi * 2.0 ** torch.arange(
                self.duration_frequency_count, dtype=torch.float32
            ),
            persistent=False,
        )
        head_input_dim = (
            self.dim
            + self.velocity_embedding_dim
            + self.duration_embedding_dim
        )
        self.motion_refiner = nn.Sequential(
            nn.Linear(head_input_dim, self.residual_hidden_dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(self.residual_hidden_dim, 3)
        if bool(_cfg_get(cfg, "zero_init", True)):
            nn.init.zeros_(self.velocity.weight)
            nn.init.zeros_(self.velocity.bias)

    def _duration_fourier(self, duration_sec):
        normalized = (
            duration_sec.reshape(-1, 1).float()
            / self.duration_reference_sec
        )
        phase = normalized * self.duration_frequency_bands.reshape(1, -1)
        # Keep the stated 8D layout for four bands: all sin, then all cos.
        return torch.cat([phase.sin(), phase.cos()], dim=-1)

    def forward(self, refined_feature, velocity_init, duration_sec):
        if refined_feature.ndim != 2 or refined_feature.shape[1] != self.dim:
            raise ValueError(
                f"refined_feature must be (N,{self.dim}) for V7 velocity"
            )
        n_tokens = refined_feature.shape[0]
        if velocity_init.shape != (n_tokens, 3):
            raise ValueError("velocity_init must be (N,3) and align with feature")
        if duration_sec.shape != (n_tokens,):
            raise ValueError("duration_sec must provide one scalar per token")
        if not bool(torch.all(
            torch.isfinite(duration_sec) & (duration_sec > 0.0)
        )):
            raise ValueError("V7 velocity head requires positive finite duration")

        condition = (
            velocity_init.detach()
            if self.detach_init_condition
            else velocity_init
        )
        feature = self.feature_norm(refined_feature)
        velocity_embedding = self.velocity_encoder(
            condition.to(dtype=feature.dtype)
        )
        duration_embedding = self.duration_encoder(
            self._duration_fourier(duration_sec).to(dtype=feature.dtype)
        )
        motion_feature = self.motion_refiner(torch.cat([
            feature,
            velocity_embedding,
            duration_embedding,
        ], dim=-1))
        return self.velocity(motion_feature)


class SparseMotionProposal(nn.Module):
    """Time-free Siamese sparse correspondence proposal for V6/V7.

    The branch consumes Utonia token features plus reference-frame token
    coordinates. The endpoint descriptor transform is shared (a projection in
    V6 and identity in V7), so swapping the frames transposes the *raw*
    content/geometry score rather than invoking separately learned Q/K maps.
    Directional softmaxes are still distinct:
    they represent p(j | i) and p(i | j), whose denominators differ.

    Top-K bounds the candidate support after the soft distance bias. Reverse
    probability then softly reweights (rather than hard-rejects) each forward
    candidate, suppressing many-to-one winners while tolerating density changes.
    V6 uses a learned global dustbin similarity inside the final row softmax;
    fresh V6 configs can add a token-conditioned residual. V7 instead uses the
    raw L2-normalized Utonia feature as its descriptor and predicts unmatched
    probability separately from three normalized cross-frame evidence scalars.
    V7 uses Top-K entropy, the best mutual displacement magnitude, and global
    reciprocal mass. V7.1 instead uses the selected M=4 hypotheses only:
    weighted mean-displacement magnitude, weighted spatial spread, and weighted
    reciprocal probability.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.descriptor_mode = str(
            _cfg_get(cfg, "descriptor_mode", "adapter_projection")
        ).lower()
        self.dustbin_mode = str(
            _cfg_get(cfg, "dustbin_mode", "in_softmax")
        ).lower()
        self.dustbin_evidence_mode = str(_cfg_get(
            cfg,
            "dustbin_evidence_mode",
            "entropy_best_reciprocal_sum",
        )).lower()
        self.mean_displacement_scale_m = float(_cfg_get(
            cfg, "mean_displacement_scale_m", 4.0
        ))
        self.spread_scale_m = float(_cfg_get(
            cfg, "spread_scale_m", 0.5
        ))
        if self.descriptor_mode not in ("adapter_projection", "direct_l2"):
            raise ValueError(
                "motion_proposal.descriptor_mode must be "
                "'adapter_projection' or 'direct_l2'"
            )
        if self.dustbin_mode not in ("in_softmax", "evidence_mlp"):
            raise ValueError(
                "motion_proposal.dustbin_mode must be "
                "'in_softmax' or 'evidence_mlp'"
            )
        if self.dustbin_evidence_mode not in (
            "entropy_best_reciprocal_sum",
            "mean_spread_reciprocal_m4",
        ):
            raise ValueError(
                "motion_proposal.dustbin_evidence_mode must be "
                "'entropy_best_reciprocal_sum' or "
                "'mean_spread_reciprocal_m4'"
            )
        if self.mean_displacement_scale_m <= 0.0:
            raise ValueError(
                "motion_proposal.mean_displacement_scale_m must be positive"
            )
        if self.spread_scale_m <= 0.0:
            raise ValueError(
                "motion_proposal.spread_scale_m must be positive"
            )
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

        # This normalizer serves only the feature-conditioned search-radius
        # side branch (and the legacy V6 descriptor/dustbin paths). In V7 the
        # matching descriptor bypasses it exactly: f_utonia -> L2Norm.
        self.input_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        if self.descriptor_mode == "adapter_projection":
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
        else:
            self.descriptor_dim = self.dim
            self.descriptor_adapter = None
            self.descriptor = None

        # Cosine descriptors make one global no-match threshold meaningful.
        # tanh keeps it in the same [-1, 1] units as cosine and prevents the
        # fallback from winning merely by sending an unconstrained logit to
        # infinity. +log(M) below makes this a per-candidate threshold rather
        # than a threshold that changes when match_count changes.
        if self.dustbin_mode == "in_softmax":
            dustbin_similarity_init = float(
                _cfg_get(cfg, "dustbin_similarity_init", 0.50)
            )
            if not -1.0 < dustbin_similarity_init < 1.0:
                raise ValueError(
                    "motion_proposal.dustbin_similarity_init must be in (-1, 1)"
                )
            self.dustbin_similarity_raw = nn.Parameter(torch.tensor(
                math.atanh(dustbin_similarity_init), dtype=torch.float32
            ))
            self.dustbin_token_conditioned = bool(
                _cfg_get(cfg, "dustbin_token_conditioned", False)
            )
            if self.dustbin_token_conditioned:
                # No bias: the learned scalar above is the global intercept.
                self.dustbin_token_residual = nn.Linear(
                    self.dim, 1, bias=False
                )
                nn.init.zeros_(self.dustbin_token_residual.weight)
            else:
                # Do not add a state-dict key for historical V6 checkpoints.
                self.dustbin_token_residual = None
            self.dustbin_predictor = None
        else:
            self.register_parameter("dustbin_similarity_raw", None)
            self.dustbin_token_conditioned = False
            self.dustbin_token_residual = None
            dustbin_hidden_dim = int(
                _cfg_get(cfg, "dustbin_hidden_dim", 16)
            )
            dustbin_prior = float(
                _cfg_get(cfg, "dustbin_prior_probability", 0.05)
            )
            self.unmatched_gate_mode = str(
                _cfg_get(cfg, "unmatched_gate_mode", "soft")
            ).lower()
            self.unmatched_hard_threshold = float(
                _cfg_get(cfg, "unmatched_hard_threshold", 0.9)
            )
            if dustbin_hidden_dim <= 0:
                raise ValueError(
                    "motion_proposal.dustbin_hidden_dim must be positive"
                )
            if not 0.0 < dustbin_prior < 1.0:
                raise ValueError(
                    "motion_proposal.dustbin_prior_probability must be in (0, 1)"
                )
            if self.unmatched_gate_mode not in ("soft", "ste_hard"):
                raise ValueError(
                    "motion_proposal.unmatched_gate_mode must be "
                    "'soft' or 'ste_hard'"
                )
            if not 0.0 < self.unmatched_hard_threshold < 1.0:
                raise ValueError(
                    "motion_proposal.unmatched_hard_threshold must be in (0, 1)"
                )
            self.dustbin_predictor = nn.Sequential(
                nn.Linear(3, dustbin_hidden_dim),
                nn.SiLU(),
                nn.Linear(dustbin_hidden_dim, 1),
            )
            # A conservative, evidence-independent 5% start. The output can
            # learn immediately while the hidden layer starts receiving signal
            # after the zero output weights take their first optimizer step.
            nn.init.zeros_(self.dustbin_predictor[-1].weight)
            nn.init.constant_(
                self.dustbin_predictor[-1].bias,
                math.log(dustbin_prior / (1.0 - dustbin_prior)),
            )

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
        if self.descriptor_mode == "direct_l2":
            descriptor = F.normalize(
                feature.float(), dim=-1, eps=1.0e-6
            )
        else:
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

    def _token_dustbin_similarity(self, feature):
        """Return one bounded dustbin threshold per time-free input token."""
        if self.dustbin_mode != "in_softmax":
            raise RuntimeError(
                "token dustbin similarity is only defined for in_softmax mode"
            )
        base = self.dustbin_similarity_raw
        if self.dustbin_token_residual is None:
            raw = base.expand(feature.shape[0])
        else:
            residual = self.dustbin_token_residual(
                self.input_norm(feature)
            ).squeeze(-1)
            raw = base + residual
        return torch.tanh(raw.float())

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

    def _finish_direction(
        self,
        direction,
        reverse,
        position,
        key_position,
        dustbin_similarity=None,
        duration_sec=None,
    ):
        """Apply a soft reciprocal boost and form one dustbin-aware proposal.

        Missing reverse top-K membership contributes exactly zero reciprocal
        probability. A uniform 1/K floor is used only inside the logarithmic
        reciprocal re-ranking term, so a valid match is not hard-cut solely
        because sampling density or occlusion made the reverse list asymmetric.
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
        candidate_position = key_position.float()[selected_candidate]
        displacement = candidate_position - position.float().unsqueeze(1)

        extra_fields = {}
        if self.dustbin_mode == "in_softmax":
            if dustbin_similarity is None:
                dustbin_similarity = torch.tanh(
                    self.dustbin_similarity_raw
                ).expand(selected_logit.shape[0])
            if dustbin_similarity.shape != (selected_logit.shape[0],):
                raise ValueError(
                    "dustbin similarity must provide one scalar per query token"
                )
            dustbin_logit = (
                dustbin_similarity / self.temperature
                + math.log(float(match_support))
            )
            augmented_probability = torch.softmax(torch.cat([
                selected_logit,
                dustbin_logit.unsqueeze(-1),
            ], dim=-1), dim=-1)
            real_weight = augmented_probability[:, :match_support]
            p_unmatched = augmented_probability[:, match_support]
            match_probability = real_weight.sum(dim=-1)
            conditional_weight = real_weight / match_probability.clamp_min(
                1.0e-12
            ).unsqueeze(-1)
            extra_fields["motion_dustbin_similarity"] = dustbin_similarity
        else:
            if dustbin_similarity is not None:
                raise ValueError(
                    "evidence_mlp dustbin does not accept a similarity token"
                )
            conditional_weight = torch.softmax(selected_logit, dim=-1)
            if self.dustbin_evidence_mode == "mean_spread_reciprocal_m4":
                # All geometry is evaluated in float32 metres. Centering before
                # squaring avoids the cancellation of E[||d||^2]-||E[d]||^2.
                # The 1e-12 floor only makes sqrt's derivative finite at an
                # exactly collapsed hypothesis set (a 1 micrometre floor).
                mean_displacement = (
                    conditional_weight.unsqueeze(-1) * displacement
                ).sum(dim=1)
                centered_displacement = (
                    displacement - mean_displacement.unsqueeze(1)
                )
                spread_squared_m2 = (
                    conditional_weight
                    * centered_displacement.square().sum(dim=-1)
                ).sum(dim=-1).clamp_min(0.0)
                candidate_spread_m = torch.sqrt(
                    spread_squared_m2.clamp_min(1.0e-12)
                )
                mean_displacement_magnitude = mean_displacement.norm(dim=-1)
                reciprocal_probability_m4 = (
                    conditional_weight * selected_reciprocal
                ).sum(dim=-1).clamp(0.0, 1.0)

                # x/(x+s) is monotone, bounded, and keeps the physical midpoint
                # explicit: 4 m mean displacement and 0.5 m spread map to 0.5.
                normalized_mean_displacement = (
                    mean_displacement_magnitude
                    / (mean_displacement_magnitude
                       + self.mean_displacement_scale_m)
                )
                normalized_candidate_spread = (
                    candidate_spread_m
                    / (candidate_spread_m + self.spread_scale_m)
                )
                evidence = torch.stack([
                    normalized_mean_displacement,
                    normalized_candidate_spread,
                    reciprocal_probability_m4,
                ], dim=-1)
                extra_fields.update({
                    "motion_mean_displacement_m": mean_displacement,
                    "motion_candidate_spread_m": candidate_spread_m,
                    "motion_normalized_mean_displacement": (
                        normalized_mean_displacement
                    ),
                    "motion_normalized_candidate_spread": (
                        normalized_candidate_spread
                    ),
                })
            else:
                candidate_probability = direction["conditional_probability"]
                candidate_entropy = -(
                    candidate_probability
                    * candidate_probability.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
                best_candidate_displacement = displacement[:, 0]
                best_candidate_displacement_magnitude = (
                    best_candidate_displacement.norm(dim=-1)
                )
                reciprocal_probability_sum = reciprocal_probability.sum(dim=-1)
                if duration_sec is None:
                    raise ValueError(
                        "legacy evidence_mlp dustbin requires query duration_sec"
                    )
                if duration_sec.shape != (position.shape[0],):
                    raise ValueError(
                        "dustbin duration_sec must provide one value per query"
                    )
                if not bool(torch.all(
                    torch.isfinite(duration_sec) & (duration_sec > 0.0)
                )):
                    raise ValueError(
                        "dustbin duration_sec must be positive and finite"
                    )

                if direction["support"] > 1:
                    normalized_entropy = (
                        candidate_entropy
                        / math.log(float(direction["support"]))
                    ).clamp(0.0, 1.0)
                else:
                    normalized_entropy = torch.zeros_like(candidate_entropy)
                max_displacement = (
                    duration_sec.float() * self.search_speed_max_mps
                ).clamp_min(1.0e-6)
                normalized_displacement = torch.tanh(
                    best_candidate_displacement_magnitude / max_displacement
                )
                normalized_reciprocal_sum = (
                    torch.log1p(reciprocal_probability_sum.clamp(
                        min=0.0, max=float(direction["support"])
                    ))
                    / math.log1p(float(direction["support"]))
                )
                evidence = torch.stack([
                    normalized_entropy,
                    normalized_displacement,
                    normalized_reciprocal_sum,
                ], dim=-1)
                extra_fields.update({
                    "motion_candidate_entropy": candidate_entropy,
                    "motion_best_candidate_displacement_m": (
                        best_candidate_displacement
                    ),
                    "motion_reciprocal_probability_sum": (
                        reciprocal_probability_sum
                    ),
                })
            dustbin_logit = self.dustbin_predictor(evidence).squeeze(-1)
            p_unmatched = torch.sigmoid(dustbin_logit)
            soft_match_probability = 1.0 - p_unmatched
            if self.unmatched_gate_mode == "ste_hard":
                hard_match = (
                    p_unmatched < self.unmatched_hard_threshold
                ).to(soft_match_probability.dtype)
                # Forward value is binary. Backward is the soft match
                # probability, i.e. d(match_gate)/d(p_unmatched) = -1.
                match_probability = (
                    hard_match
                    + soft_match_probability
                    - soft_match_probability.detach()
                )
            else:
                hard_match = torch.ones_like(soft_match_probability)
                match_probability = soft_match_probability
            real_weight = (
                match_probability.unsqueeze(-1) * conditional_weight
            )
            extra_fields.update({
                "motion_dustbin_logit": dustbin_logit,
                "motion_soft_match_probability": soft_match_probability,
                "motion_hard_reject": 1.0 - hard_match,
            })

        delta_p_init = (real_weight.unsqueeze(-1) * displacement).sum(dim=1)
        delta_p_match = (
            conditional_weight.unsqueeze(-1) * displacement
        ).sum(dim=1)
        entropy = -(
            conditional_weight
            * conditional_weight.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        result = {
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
        }
        result.update(extra_fields)
        return result

    def _match_pair(
        self,
        descriptor0,
        descriptor1,
        position0,
        position1,
        radius0,
        radius1,
        dustbin_similarity0=None,
        dustbin_similarity1=None,
        duration0=None,
        duration1=None,
    ):
        forward = self._topk_direction(
            descriptor0, descriptor1, position0, position1, radius0, radius1
        )
        backward = self._topk_direction(
            descriptor1, descriptor0, position1, position0, radius1, radius0
        )

        return (
            self._finish_direction(
                forward, backward, position0, position1,
                dustbin_similarity0,
                duration0,
            ),
            self._finish_direction(
                backward, forward, position1, position0,
                dustbin_similarity1,
                duration1,
            ),
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
        dustbin_similarity = (
            self._token_dustbin_similarity(feature)
            if self.dustbin_mode == "in_softmax"
            else None
        )
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
                (
                    dustbin_similarity[rows0]
                    if dustbin_similarity is not None else None
                ),
                (
                    dustbin_similarity[rows1]
                    if dustbin_similarity is not None else None
                ),
                duration_sec[rows0],
                duration_sec[rows1],
            )
            by_frame[frame0] = result0
            by_frame[frame1] = result1

        result = {
            name: torch.cat([frame[name] for frame in by_frame], dim=0)
            for name in by_frame[0]
        }
        result["motion_search_speed_mps"] = search_speed
        return result


class StraightThroughTop4MotionProposal(nn.Module):
    """Direct cosine Top-4 coordinates with a dense-all-key STE surrogate.

    Forward weights are the full-softmax probabilities restricted to the four
    largest entries and renormalized to sum to one. Backward weights are the
    original full-softmax probabilities, so every key receives correspondence
    gradient even though only four coordinates contribute to the rendered
    forward result::

        w = p_dense + stopgrad(w_top4 - p_dense)

    The implementation materializes these weights only for one query chunk at
    a time and checkpoints train-time chunks. It has no learned descriptor,
    reciprocal term, or unmatched/dustbin branch. A fixed physical distance
    prior only suppresses displacement beyond one shared speed envelope; unlike
    V7, its radius is neither token-conditioned nor learned.
    """

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.descriptor_mode = str(_cfg_get(cfg, "descriptor_mode", "direct_l2"))
        self.match_count = int(_cfg_get(cfg, "match_count", 4))
        self.temperature = float(_cfg_get(cfg, "temperature", 0.07))
        self.score_chunk_size = int(_cfg_get(cfg, "score_chunk_size", 128))
        self.distance_prior_speed_mps = float(
            _cfg_get(cfg, "distance_prior_speed_mps", 30.0)
        )
        self.ste_surrogate = str(
            _cfg_get(cfg, "ste_surrogate", "dense_softmax")
        )
        if self.dim <= 0:
            raise ValueError("motion proposal dim must be positive")
        if self.descriptor_mode != "direct_l2":
            raise ValueError("V7.2 requires direct_l2 motion descriptors")
        if self.match_count != 4:
            raise ValueError("V7.2 requires exactly four forward matches")
        if self.temperature <= 0.0:
            raise ValueError("motion proposal temperature must be positive")
        if self.score_chunk_size <= 0:
            raise ValueError("motion proposal score_chunk_size must be positive")
        if self.distance_prior_speed_mps <= 0.0:
            raise ValueError(
                "motion proposal distance_prior_speed_mps must be positive"
            )
        if self.ste_surrogate != "dense_softmax":
            raise ValueError("V7.2 requires ste_surrogate='dense_softmax'")

    def _direction_chunk(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        query_duration_sec,
    ):
        content_score = (
            query_descriptor @ key_descriptor.transpose(0, 1)
        ) / self.temperature
        # One fixed physical envelope for the entire pair:
        #   penalty_ij = (||x_i-x_j|| / (v_max * delta_t))^2.
        # The radius varies only with the sample duration, never with i/j or a
        # learned token prediction. Centering improves squared-distance accuracy
        # in global reference coordinates without changing the displacement.
        origin = 0.5 * (
            query_position.float().mean(dim=0)
            + key_position.float().mean(dim=0)
        )
        query_centered = query_position.float() - origin
        key_centered = key_position.float() - origin
        squared_distance = (
            query_centered.square().sum(dim=-1, keepdim=True)
            + key_centered.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (query_centered @ key_centered.transpose(0, 1))
        ).clamp_min(0.0)
        radius = (
            self.distance_prior_speed_mps
            * query_duration_sec.float().clamp_min(1.0e-4)
        )
        distance_penalty = squared_distance / radius.square().unsqueeze(-1)
        score = content_score - distance_penalty
        dense_probability = torch.softmax(score, dim=-1)
        key_count = int(key_descriptor.shape[0])
        support = min(self.match_count, key_count)
        ranked_support = min(support + 1, key_count)
        ranked_probability, ranked_index = torch.topk(
            dense_probability,
            k=ranked_support,
            dim=-1,
            largest=True,
            sorted=True,
        )
        top_probability = ranked_probability[:, :support]
        selected_index = ranked_index[:, :support]
        selected_mass = top_probability.sum(dim=-1, keepdim=True)
        conditional_weight = (
            top_probability / selected_mass.clamp_min(1.0e-12)
        )

        hard_weight = torch.zeros_like(dense_probability).scatter(
            1, selected_index, conditional_weight
        )
        # Exact normalized Top-4 forward; exact dense-all-key softmax backward.
        # The stop-gradient covers both hard support membership and its
        # within-support renormalization. The surrogate therefore does not
        # pretend that torch.topk itself is differentiable.
        ste_weight = dense_probability + (
            hard_weight - dense_probability
        ).detach()
        matched_position = ste_weight @ key_position.float()
        displacement = matched_position - query_position.float()

        with torch.no_grad():
            entropy = -(
                conditional_weight
                * conditional_weight.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            if ranked_support > support:
                selection_log_margin = (
                    ranked_probability[:, support - 1].clamp_min(1.0e-12).log()
                    - ranked_probability[:, support].clamp_min(1.0e-12).log()
                )
            else:
                selection_log_margin = score.new_zeros(score.shape[0])
            dense_matched_position = (
                dense_probability.detach() @ key_position.float()
            )
            dense_displacement = dense_matched_position - query_position.float()
            hard_displacement = displacement.detach()
            hard_soft_cosine = F.cosine_similarity(
                hard_displacement, dense_displacement, dim=-1, eps=1.0e-8
            )
            hard_soft_norm_ratio = (
                hard_displacement.norm(dim=-1)
                / dense_displacement.norm(dim=-1).clamp_min(1.0e-8)
            )
            selected_distance_penalty = distance_penalty.gather(
                1, selected_index
            )
            mean_selected_distance_penalty = (
                conditional_weight * selected_distance_penalty
            ).sum(dim=-1)

        return (
            displacement,
            matched_position,
            selected_index,
            conditional_weight[:, 0].detach(),
            entropy.exp().detach(),
            selected_mass.squeeze(-1).detach(),
            selection_log_margin.detach(),
            hard_soft_cosine,
            hard_soft_norm_ratio,
            mean_selected_distance_penalty.detach(),
            score.new_full((score.shape[0],), float(support)).detach(),
        )

    def _match_direction(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        duration_sec=None,
    ):
        if int(key_descriptor.shape[0]) == 0:
            raise ValueError("motion proposal requires tokens in both endpoints")
        if duration_sec is None:
            duration_sec = query_position.new_ones(query_position.shape[0])
        if duration_sec.shape != (query_position.shape[0],):
            raise ValueError(
                "motion proposal direction duration must align with queries"
            )
        chunks = []
        for start in range(
            0, int(query_descriptor.shape[0]), self.score_chunk_size
        ):
            end = min(
                start + self.score_chunk_size, query_descriptor.shape[0]
            )
            inputs = (
                query_descriptor[start:end],
                key_descriptor,
                query_position[start:end],
                key_position,
                duration_sec[start:end],
            )
            if (
                self.training
                and torch.is_grad_enabled()
                and (query_descriptor.requires_grad or key_descriptor.requires_grad)
            ):
                output = checkpoint(
                    self._direction_chunk, *inputs, use_reentrant=False
                )
            else:
                output = self._direction_chunk(*inputs)
            chunks.append(output)

        fields = tuple(
            torch.cat([chunk[index] for chunk in chunks], dim=0)
            for index in range(len(chunks[0]))
        )
        (
            displacement,
            matched_position,
            selected_index,
            top1_probability,
            effective_support,
            candidate_probability_mass,
            selection_log_margin,
            hard_soft_cosine,
            hard_soft_norm_ratio,
            selected_distance_penalty,
            match_support,
        ) = fields
        return {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_selected_index": selected_index,
            "motion_top1_probability": top1_probability,
            "motion_effective_support": effective_support,
            "motion_candidate_probability_mass": candidate_probability_mass,
            "motion_selection_log_margin": selection_log_margin,
            "motion_hard_soft_displacement_cosine": hard_soft_cosine,
            "motion_hard_soft_displacement_norm_ratio": hard_soft_norm_ratio,
            "motion_selected_distance_prior_penalty": (
                selected_distance_penalty
            ),
            "motion_match_support": match_support,
        }

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
        if not bool(torch.all(
            torch.isfinite(duration_sec) & (duration_sec > 0.0)
        )):
            raise ValueError("motion proposal requires positive finite durations")

        counts = torch.diff(
            token_offset, prepend=token_offset.new_zeros(1)
        ).long()
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

        descriptor = F.normalize(feature.float(), dim=-1, eps=1.0e-6)
        by_frame = [None] * int(counts.numel())
        for frames in batch_frames.values():
            frame0, frame1 = frames
            start0, count0 = int(starts[frame0]), int(counts[frame0])
            start1, count1 = int(starts[frame1]), int(counts[frame1])
            rows0 = slice(start0, start0 + count0)
            rows1 = slice(start1, start1 + count1)
            by_frame[frame0] = self._match_direction(
                descriptor[rows0], descriptor[rows1],
                position_ref[rows0], position_ref[rows1],
                duration_sec[rows0],
            )
            by_frame[frame1] = self._match_direction(
                descriptor[rows1], descriptor[rows0],
                position_ref[rows1], position_ref[rows0],
                duration_sec[rows1],
            )

        return {
            name: torch.cat([frame[name] for frame in by_frame], dim=0)
            for name in by_frame[0]
        }


class ProjectedDenseMotionProposal(nn.Module):
    """V9 dense correspondence from a learned post-attention descriptor.

    Unlike V6/V7/V7.2, this matcher runs *after* temporal cross-attention and
    consumes the refined token feature, so its descriptors already carry
    endpoint time and cross-frame context.  The descriptor path is
    ``LN(f') -> Linear(dim, descriptor_dim) -> L2Norm``: V7.2's parameter-free
    ``direct_l2`` normalization is replaced by one learned projection whose
    width is decoupled from the trunk.

    Readout is the plain dense expectation.  Every key contributes to both the
    forward coordinate and the gradient::

        p   = softmax(cosine / temperature - (distance / (v_max * dt))^2)
        x_m = p @ x_key

    There is no Top-K truncation, straight-through surrogate, reciprocal
    re-ranking, dustbin, or unmatched gate.  V7.2's ``StraightThroughTop4``
    matcher is left untouched for A/B runs; the only change to the shared
    distance prior is its denominator, which V9 tightens from 30 m/s to
    5 m/s.  Scores are materialized one query chunk at a time and train-time
    chunks are recomputed during backward.
    """

    # Fixed purely so the concentration diagnostics below stay numerically
    # comparable with V7.2's Top-4 forward. It never truncates the readout.
    _DIAGNOSTIC_TOP_K = 4

    def __init__(self, cfg, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.descriptor_mode = str(
            _cfg_get(cfg, "descriptor_mode", "projected_l2")
        )
        self.descriptor_dim = int(_cfg_get(cfg, "descriptor_dim", 128))
        self.temperature = float(_cfg_get(cfg, "temperature", 0.07))
        self.score_chunk_size = int(_cfg_get(cfg, "score_chunk_size", 128))
        self.distance_prior_speed_mps = float(
            _cfg_get(cfg, "distance_prior_speed_mps", 5.0)
        )
        self.readout = str(_cfg_get(cfg, "readout", "dense_expectation"))
        if self.dim <= 0:
            raise ValueError("motion proposal dim must be positive")
        if self.descriptor_dim <= 0:
            raise ValueError("motion proposal descriptor_dim must be positive")
        if self.descriptor_mode != "projected_l2":
            raise ValueError("V9 requires projected_l2 motion descriptors")
        if self.readout != "dense_expectation":
            raise ValueError("V9 requires readout='dense_expectation'")
        if self.temperature <= 0.0:
            raise ValueError("motion proposal temperature must be positive")
        if self.score_chunk_size <= 0:
            raise ValueError("motion proposal score_chunk_size must be positive")
        if self.distance_prior_speed_mps <= 0.0:
            raise ValueError(
                "motion proposal distance_prior_speed_mps must be positive"
            )

        # One endpoint-shared descriptor branch. Both frames are encoded by the
        # same parameters, so the score stays symmetric under a frame swap.
        self.descriptor_norm = nn.LayerNorm(self.dim)
        self.descriptor_proj = nn.Linear(self.dim, self.descriptor_dim)

        # Concentration diagnostics need a Top-K over the full dense score
        # matrix, which the dense readout itself never computes -- unlike V7.2,
        # where the same Top-K produces the rendered coordinate. Measured at
        # 432D/128D over ~7.3k tokens per frame that is ~4% of a training step,
        # so ModelWrapper switches it on only when it is about to log. The flag
        # is read, never learned, and is identical on every DDP rank.
        self.collect_diagnostics = True

    def _encode(self, feature):
        return F.normalize(
            self.descriptor_proj(self.descriptor_norm(feature)).float(),
            dim=-1,
            eps=1.0e-6,
        )

    def _direction_chunk(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        query_duration_sec,
    ):
        content_score = (
            query_descriptor @ key_descriptor.transpose(0, 1)
        ) / self.temperature
        # One fixed physical envelope for the entire pair:
        #   penalty_ij = (||x_i-x_j|| / (v_max * delta_t))^2.
        # Identical in form to V7.2; only v_max changes (30 -> 5 m/s), so at a
        # one-second interval 5, 10, and 15 m now cost 1.0, 4.0, and 9.0 logits
        # where V7.2 charged 0.028, 0.111, and 0.25. Centering improves
        # squared-distance accuracy in global reference coordinates without
        # changing the displacement.
        origin = 0.5 * (
            query_position.float().mean(dim=0)
            + key_position.float().mean(dim=0)
        )
        query_centered = query_position.float() - origin
        key_centered = key_position.float() - origin
        squared_distance = (
            query_centered.square().sum(dim=-1, keepdim=True)
            + key_centered.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (query_centered @ key_centered.transpose(0, 1))
        ).clamp_min(0.0)
        radius = (
            self.distance_prior_speed_mps
            * query_duration_sec.float().clamp_min(1.0e-4)
        )
        distance_penalty = squared_distance / radius.square().unsqueeze(-1)
        score = content_score - distance_penalty
        probability = torch.softmax(score, dim=-1)
        matched_position = probability @ key_position.float()
        displacement = matched_position - query_position.float()
        if not self.collect_diagnostics:
            return (displacement, matched_position)

        with torch.no_grad():
            key_count = int(key_descriptor.shape[0])
            support = min(self._DIAGNOSTIC_TOP_K, key_count)
            ranked_support = min(support + 1, key_count)
            ranked_probability, ranked_index = torch.topk(
                probability,
                k=ranked_support,
                dim=-1,
                largest=True,
                sorted=True,
            )
            top_probability = ranked_probability[:, :support]
            top_mass = top_probability.sum(dim=-1)
            # exp(entropy) over *all* keys: a dense readout fails by spreading
            # mass over the scene, and only the full-support number shows it.
            effective_support = (
                -(probability * probability.clamp_min(1.0e-12).log())
                .sum(dim=-1)
            ).exp()
            if ranked_support > support:
                selection_log_margin = (
                    ranked_probability[:, support - 1].clamp_min(1.0e-12).log()
                    - ranked_probability[:, support].clamp_min(1.0e-12).log()
                )
            else:
                selection_log_margin = score.new_zeros(score.shape[0])
            # Same two vectors V7.2 compares, so the pair stays directly
            # readable across the two matchers: here the dense expectation is
            # the rendered forward and the Top-4 mixture is the counterfactual.
            top_weight = top_probability / top_mass.clamp_min(1.0e-12).unsqueeze(-1)
            top_displacement = (
                (top_weight.unsqueeze(-1) * key_position.float()[ranked_index[:, :support]])
                .sum(dim=1)
                - query_position.float()
            )
            dense_displacement = displacement.detach()
            hard_soft_cosine = F.cosine_similarity(
                top_displacement, dense_displacement, dim=-1, eps=1.0e-8
            )
            hard_soft_norm_ratio = (
                top_displacement.norm(dim=-1)
                / dense_displacement.norm(dim=-1).clamp_min(1.0e-8)
            )
            # The penalty the rendered readout actually pays, weighted by the
            # same dense probabilities that produced the coordinate.
            mean_distance_penalty = (probability * distance_penalty).sum(dim=-1)

        return (
            displacement,
            matched_position,
            ranked_probability[:, 0].detach(),
            effective_support,
            top_mass,
            selection_log_margin,
            hard_soft_cosine,
            hard_soft_norm_ratio,
            mean_distance_penalty,
        )

    def _match_direction(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        duration_sec=None,
    ):
        if int(key_descriptor.shape[0]) == 0:
            raise ValueError("motion proposal requires tokens in both endpoints")
        # An empty query frame would otherwise produce no chunks at all and
        # surface as an opaque IndexError below.
        if int(query_descriptor.shape[0]) == 0:
            raise ValueError("motion proposal requires tokens in both endpoints")
        if duration_sec is None:
            duration_sec = query_position.new_ones(query_position.shape[0])
        if duration_sec.shape != (query_position.shape[0],):
            raise ValueError(
                "motion proposal direction duration must align with queries"
            )
        chunks = []
        for start in range(
            0, int(query_descriptor.shape[0]), self.score_chunk_size
        ):
            end = min(
                start + self.score_chunk_size, query_descriptor.shape[0]
            )
            inputs = (
                query_descriptor[start:end],
                key_descriptor,
                query_position[start:end],
                key_position,
                duration_sec[start:end],
            )
            if (
                self.training
                and torch.is_grad_enabled()
                and (query_descriptor.requires_grad or key_descriptor.requires_grad)
            ):
                output = checkpoint(
                    self._direction_chunk, *inputs, use_reentrant=False
                )
            else:
                output = self._direction_chunk(*inputs)
            chunks.append(output)

        fields = tuple(
            torch.cat([chunk[index] for chunk in chunks], dim=0)
            for index in range(len(chunks[0]))
        )
        if not self.collect_diagnostics:
            displacement, matched_position = fields
            return {
                "delta_p_match": displacement,
                "delta_p_init": displacement,
                "matched_position": matched_position,
            }
        (
            displacement,
            matched_position,
            top1_probability,
            effective_support,
            candidate_probability_mass,
            selection_log_margin,
            hard_soft_cosine,
            hard_soft_norm_ratio,
            distance_penalty,
        ) = fields
        return {
            "delta_p_match": displacement,
            "delta_p_init": displacement,
            "matched_position": matched_position,
            "motion_top1_probability": top1_probability,
            "motion_effective_support": effective_support,
            "motion_candidate_probability_mass": candidate_probability_mass,
            "motion_selection_log_margin": selection_log_margin,
            "motion_hard_soft_displacement_cosine": hard_soft_cosine,
            "motion_hard_soft_displacement_norm_ratio": hard_soft_norm_ratio,
            "motion_selected_distance_prior_penalty": distance_penalty,
        }

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
        if not bool(torch.all(
            torch.isfinite(duration_sec) & (duration_sec > 0.0)
        )):
            raise ValueError("motion proposal requires positive finite durations")

        counts = torch.diff(
            token_offset, prepend=token_offset.new_zeros(1)
        ).long()
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

        descriptor = self._encode(feature)
        by_frame = [None] * int(counts.numel())
        for frames in batch_frames.values():
            frame0, frame1 = frames
            start0, count0 = int(starts[frame0]), int(counts[frame0])
            start1, count1 = int(starts[frame1]), int(counts[frame1])
            rows0 = slice(start0, start0 + count0)
            rows1 = slice(start1, start1 + count1)
            by_frame[frame0] = self._match_direction(
                descriptor[rows0], descriptor[rows1],
                position_ref[rows0], position_ref[rows1],
                duration_sec[rows0],
            )
            by_frame[frame1] = self._match_direction(
                descriptor[rows1], descriptor[rows0],
                position_ref[rows1], position_ref[rows0],
                duration_sec[rows1],
            )

        return {
            name: torch.cat([frame[name] for frame in by_frame], dim=0)
            for name in by_frame[0]
        }


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
        seed_delta_sensor=None,
    ):
        device, dtype = token_sensor.device, token_sensor.dtype
        token_ref = torch.empty_like(token_sensor)
        seed_ref = torch.empty_like(seed_sensor)
        seed_delta_ref = (
            torch.empty_like(seed_delta_sensor)
            if seed_delta_sensor is not None else None
        )
        if seed_delta_sensor is not None and seed_delta_sensor.shape != seed_sensor.shape:
            raise ValueError("seed_delta_sensor must have the same shape as seeds")
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
            seed_frame = seed_sensor[start:end]
            seed_ref[start:end] = box_utils.apply_pose(
                seed_frame.reshape(-1, 3), pose
            ).reshape_as(seed_frame)
            if seed_delta_ref is not None:
                delta_frame = seed_delta_sensor[start:end]
                center_frame = seed_frame - delta_frame
                center_ref = box_utils.apply_pose(
                    center_frame.reshape(-1, 3), pose
                ).reshape_as(center_frame)
                seed_delta_ref[start:end] = seed_ref[start:end] - center_ref
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
            "seed_delta_ref": seed_delta_ref,
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


class LayerWeightedAttentionVelocityGaussianBackend(
    PhysicalVelocityGaussianBackend
):
    """V10 all-layer initializer plus adaptive-K Gaussians and shared motion.

    The temporal module returns the layer-weighted coordinate expectation.
    Dividing its signed displacement by the physical endpoint interval produces
    one token-level ``v_init``.  The normalized refined token then enters the
    legacy grid ``learned_gumbel`` router and only its selected K-specific head
    predicts K Gaussians. Every packed child Gaussian indexes the same token
    velocity, while the additive correction retains
    ``LN -> Linear -> SiLU -> Linear(3)`` at token resolution.
    """

    def __init__(
        self,
        cfg,
        gs_params,
        dim: int,
        offset_bound: float,
        gaussian_count_cfg=None,
    ):
        # Build the physical backend directly so no unused RoPE temporal module
        # is ever instantiated for a distance-bias-only V10 run.
        nn.Module.__init__(self)
        self.cfg = cfg
        self.dim = int(dim)
        self.temporal = self._build_temporal(cfg)
        self.time_reference_sec = float(
            _cfg_get(cfg.temporal, "time_reference_sec", 1.0)
        )
        if self.time_reference_sec <= 0.0:
            raise ValueError("time_reference_sec must be positive")
        if gaussian_count_cfg is None:
            raise ValueError("adaptive-K dynamic backend requires adaptive Gaussian-count config")
        from .grid_query_head import GridSlotHead

        self.gaussian_output_norm = nn.LayerNorm(self.dim)
        self.gaussian_head = GridSlotHead(
            gaussian_count_cfg, gs_params, dim=self.dim
        )
        if self.gaussian_head.count_mode != "learned_gumbel":
            raise ValueError("adaptive-K Gaussian count must use learned_gumbel")
        if self.gaussian_head.k_max != 3:
            raise ValueError("adaptive-K Gaussian count requires K_max=3")
        self.offset_bound = float(offset_bound)
        self.gaussian_param_names = [
            "shs", "opacity", "scaling", "rotation", "offset",
        ]
        self.gaussian_param_sizes = [
            int(getattr(gs_params, name)) for name in self.gaussian_param_names
        ]
        if self.gaussian_param_sizes[-2:] != [4, 3]:
            raise ValueError(
                "adaptive-K Gaussian head requires rotation=4 and offset=3"
            )
        if sum(self.gaussian_param_sizes) != self.gaussian_head.param_dim:
            raise ValueError("adaptive-K Gaussian parameter widths disagree")
        self._initialize_adaptive_gaussian_heads(
            _cfg_get(cfg, "gaussian_head", None)
        )
        self.velocity_head = self._build_velocity_head(cfg)

    def _build_temporal(self, cfg):
        """The temporal module whose head probabilities become ``v_init``."""
        return LayerWeightedDistanceBiasCrossAttention(cfg.temporal, self.dim)

    def _build_velocity_head(self, cfg):
        """The additive correction applied on top of ``v_init``."""
        return FeatureOnlyVelocityOffsetHead(cfg.motion, self.dim)

    def _velocity_offset(self, refined, velocity_init):
        """V10 reads only the refined token; the initializer is not an input."""
        del velocity_init
        return self.velocity_head(refined)

    def _initialize_adaptive_gaussian_heads(self, cfg):
        """Give every K expert the existing dynamic-head output priors."""
        opacity = float(_cfg_get(cfg, "initial_opacity", 0.2))
        opacity = min(max(opacity, 1.0e-4), 1.0 - 1.0e-4)
        opacity_bias = math.log(opacity / (1.0 - opacity))
        scale = float(_cfg_get(cfg, "initial_scale_m", 0.3))
        scale_bias = math.log(math.expm1(max(scale, 1.0e-4)))

        sh_size, opacity_size, scale_size, rotation_size, offset_size = (
            self.gaussian_param_sizes
        )
        opacity_start = sh_size
        scale_start = opacity_start + opacity_size
        rotation_start = scale_start + scale_size
        offset_start = rotation_start + rotation_size
        block_bias = self.gaussian_head.k_heads[0].bias.new_zeros(
            self.gaussian_head.param_dim
        )
        block_bias[opacity_start:scale_start] = opacity_bias
        block_bias[scale_start:rotation_start] = scale_bias
        block_bias[rotation_start] = 1.0

        for k, head in enumerate(self.gaussian_head.k_heads, start=1):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                head.bias.copy_(block_bias.repeat(k))
                for slot in range(k):
                    row_start = slot * self.gaussian_head.param_dim + offset_start
                    row_end = row_start + offset_size
                    head.weight[row_start:row_end].zero_()

    def _predict_adaptive_gaussians(self, refined, geometry, token_offset):
        seed_ref = geometry["seed_ref"]
        expected_seed_shape = (
            refined.shape[0],
            self.gaussian_head.k_max,
            self.gaussian_head.k_max,
            3,
        )
        if tuple(seed_ref.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K seed bank must have shape "
                f"{expected_seed_shape}, got {tuple(seed_ref.shape)}"
            )
        gaussian_feature = self.gaussian_output_norm(refined)
        delta_ref = geometry.get("seed_delta_ref")
        if delta_ref is None or tuple(delta_ref.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires a transformed K-specific seed-delta bank"
            )
        raw_params, packing = self.gaussian_head(
            gaussian_feature,
            None,
            delta_ref,
            token_offset,
        )
        selected_index = packing["anchor_k"] - 1
        token_rows = torch.arange(
            refined.shape[0], device=refined.device
        )
        selected_seed = seed_ref[token_rows, selected_index]
        anchor_index = packing["anchor_index"]
        slot_index = packing["slot_index"]
        packed_seed = selected_seed[anchor_index, slot_index]

        parts = torch.split(raw_params, self.gaussian_param_sizes, dim=-1)
        raw = dict(zip(self.gaussian_param_names, parts))
        position = packed_seed + self.offset_bound * torch.tanh(
            raw.pop("offset")
        )
        gradient_weight = self.gaussian_head.gradient_weight(
            packing["slot_k"], position.dtype
        )
        if (
            self.gaussian_head.grad_balance == "sqrt_k"
            and self.gaussian_head.grad_balance_scope == "output"
        ):
            from .gaussian_assembly import gradient_scale_identity

            position = gradient_scale_identity(position, gradient_weight)
            raw = {
                name: gradient_scale_identity(value, gradient_weight)
                for name, value in raw.items()
            }
        return raw, position, packing, gradient_weight

    @staticmethod
    def _signed_pair_delta_t(geometry):
        local_frame = geometry["local_frame"]
        if not bool(torch.all((local_frame == 0) | (local_frame == 1))):
            raise ValueError("adaptive-K attention velocity requires endpoint indices 0/1")
        duration = geometry["duration_sec"]
        if not bool(torch.all(torch.isfinite(duration) & (duration > 0.0))):
            raise ValueError(
                "adaptive-K attention velocity requires positive finite duration"
            )
        return torch.where(local_frame == 0, duration, -duration)

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        del motion_proposal_feature
        return self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            geometry["duration_sec"],
        )

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self._velocity_offset(refined, velocity_init)
        velocity = velocity_init + velocity_offset
        return velocity, {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "matched_position": temporal_fields["matched_position"],
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": velocity_init,
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
            "motion_layer_logits": temporal_fields["motion_layer_logits"],
            "motion_layer_weights": temporal_fields["motion_layer_weights"],
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
        motion_proposal_feature=None,
        seed_delta_sensor=None,
    ):
        del motion_proposal_feature
        if fused_feature.ndim != 2 or fused_feature.shape[1] != self.dim:
            raise ValueError(f"dynamic fused_feature must be (N,{self.dim})")
        expected_seed_shape = (
            fused_feature.shape[0],
            self.gaussian_head.k_max,
            self.gaussian_head.k_max,
            3,
        )
        if tuple(seed_sensor.shape) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires the learned-count seed bank with shape "
                f"{expected_seed_shape}, got {tuple(seed_sensor.shape)}"
            )
        if seed_delta_sensor is None or tuple(
            seed_delta_sensor.shape
        ) != expected_seed_shape:
            raise ValueError(
                "adaptive-K dynamic backend requires a learned-count seed-delta bank with shape "
                f"{expected_seed_shape}"
            )
        if token_position_sensor.shape != (fused_feature.shape[0], 3):
            raise ValueError("adaptive-K token positions must align with features")

        geometry = self._reference_geometry_and_time(
            token_position_sensor,
            seed_sensor,
            token_offset,
            frame_batch_idx,
            pose_list,
            timestamps_normalized,
            timestamps_sec,
            window_duration_sec,
            seed_delta_sensor=seed_delta_sensor,
        )
        refined, temporal_fields = self._temporal_refine(
            fused_feature,
            geometry,
            token_offset,
            frame_batch_idx,
        )
        raw, position, packing, gradient_weight = (
            self._predict_adaptive_gaussians(
                refined, geometry, token_offset
            )
        )

        token_velocity, token_motion_fields = self._predict_motion(
            refined, geometry, None, temporal_fields
        )
        anchor_index = packing["anchor_index"]
        velocity = token_velocity[anchor_index]
        motion_fields = {
            name: value[anchor_index]
            for name, value in token_motion_fields.items()
        }
        if (
            self.gaussian_head.grad_balance == "sqrt_k"
            and self.gaussian_head.grad_balance_scope == "output"
        ):
            from .gaussian_assembly import gradient_scale_identity

            velocity = gradient_scale_identity(velocity, gradient_weight)

        packed_batch = geometry["batch"][anchor_index]
        packed_time_sec = geometry["time_sec"][anchor_index]
        packed_time_normalized = geometry["time_normalized"][anchor_index]
        packed_duration = geometry["duration_sec"][anchor_index]
        packed_selected_k = packing["anchor_k"][anchor_index]

        batch_gaussians = []
        for batch_id in range(len(pose_list)):
            rows = (packed_batch == batch_id).nonzero(as_tuple=True)[0]
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
                **{
                    name: value[rows]
                    for name, value in motion_fields.items()
                },
                "source_time_sec": packed_time_sec[rows],
                "source_time_normalized": packed_time_normalized[rows],
                "window_duration_sec": packed_duration[rows],
                "source_token_index": anchor_index[rows],
                "selected_k": packed_selected_k[rows],
                "gaussian_slot_index": packing["slot_index"][rows],
                "is_dynamic": motion_mask,
                "instance_id": torch.full(
                    (n,), -1, dtype=torch.long, device=rows.device
                ),
                "bg_mask": ~motion_mask,
                "fg_masks": {},
                "frame_bboxes": [],
            })

        routing_stats = {
            "k_logits": packing["k_logits"].detach(),
            "selected_k": packing["anchor_k"].detach(),
            "token_position_sensor": token_position_sensor.detach(),
            "is_dynamic": torch.ones(
                fused_feature.shape[0],
                dtype=torch.bool,
                device=fused_feature.device,
            ),
        }
        return {
            "batch_gaussians": batch_gaussians,
            "gaussians": batch_gaussians,
            "batch": packed_batch,
            "routing_stats": routing_stats,
        }


class MaxSpeedBarrierVelocityGaussianBackend(
    LayerWeightedAttentionVelocityGaussianBackend
):
    """V11: V10's adaptive-K Gaussians behind a split-position-encoding stack.

    Everything downstream of temporal refinement is V10's: the same
    ``learned_gumbel`` K={1,2,3} router over ``LN(f')``, the same feature-only
    ``v_offset``, and one token velocity index-expanded to every child Gaussian.
    Only the temporal module changes. Correspondence now comes from head 0 of
    each layer alone, under the max-speed soft barrier instead of RoPE, and the
    twelve readouts are mixed by ``L`` plain learned scalars rather than by a
    per-frame global token.
    """

    def _build_temporal(self, cfg):
        return MaxSpeedBarrierLayerWeightedCrossAttention(
            cfg.temporal, self.dim
        )

    def _build_velocity_head(self, cfg):
        """V8's head: ``[LN(f'), MLP(v_init; 3->32->32)] -> 96 -> 3``.

        The refined feature carries head 0's attended *feature* channels, never
        the metric coordinate expectation itself, so the offset head cannot see
        what it is correcting unless the initializer is embedded explicitly.
        """
        return EmbeddedInitVelocityHead(cfg.motion, self.dim)

    def _velocity_offset(self, refined, velocity_init):
        return self.velocity_head(refined, velocity_init)


class ConsensusAttentionVelocityGaussianBackend(
    AttentionInitializedVelocityGaussianBackend
):
    """V8 final-attention consensus matching plus an additive offset.

    Heads ``[0, motion_head_count)`` remain ordinary dense feature-attention
    heads.  After the final layer has computed that dense update, their Q/K
    probabilities are averaged and one shared K->M reciprocal support is used
    only to read opposite-frame coordinates.  There is no independent LoRA
    descriptor branch, distance bias, dustbin, or query-coordinate warp.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__(cfg, gs_params, dim, offset_bound)
        self.motion_matcher = ConsensusAttentionMotionMatcher(
            cfg.motion_matching
        )
        self.velocity_head = EmbeddedInitVelocityHead(
            cfg.motion, self.dim
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
        refined, matching_fields = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
            return_motion_displacement=True,
            motion_matcher=self.motion_matcher,
        )
        return refined, matching_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self.velocity_head(
            refined,
            velocity_init,
        )
        velocity = velocity_init + velocity_offset
        fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": (
                temporal_fields["delta_p_match"]
                / pair_delta_t_sec.unsqueeze(-1)
            ),
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        for name in (
            "matched_position",
            "motion_top1_probability",
            "motion_effective_support",
            "motion_reciprocal_probability",
            "motion_selected_forward_probability",
            "motion_consensus_entropy",
            "motion_head_js_divergence",
            "motion_candidate_probability_mass",
            "motion_selection_log_margin",
            "motion_candidate_support",
            "motion_match_support",
        ):
            fields[name] = temporal_fields[name]
        return velocity, fields


class ProposalInitializedVelocityGaussianBackend(PhysicalVelocityGaussianBackend):
    """V6 velocity from a sparse proposal plus an init-aware residual.

    Correspondence is computed from the same Utonia tokens used by the temporal
    trunk and before the temporal module adds time. The ordinary
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
            raise ValueError("V6 requires Utonia motion proposal features")
        # This call is intentionally before self.temporal: descriptors never see
        # endpoint time embedding or cross-frame
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


class WarpedProposalVelocityGaussianBackend(
    ProposalInitializedVelocityGaussianBackend
):
    """V7 proposal-warped attention and duration-aware velocity offset.

    The hard-gated initializer moves only the RoPE query coordinate to
    ``x_query + stopgrad(v_init) * signed_dt``. Keys stay at their observed
    reference-frame coordinates. There is no multiplicative confidence gate on
    cross-attention and V7 constructs no temporal time embedding. A confidently
    unmatched token therefore keeps its query at the observed position.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 proposal_dim: int | None = None):
        super().__init__(
            cfg,
            gs_params,
            dim,
            offset_bound,
            proposal_dim=proposal_dim,
        )
        if self.temporal.use_time_embedding:
            raise ValueError("V7 temporal attention must disable time embedding")
        if self.motion_proposal.descriptor_mode != "direct_l2":
            raise ValueError("V7 requires direct_l2 motion descriptors")
        if self.motion_proposal.dustbin_mode != "evidence_mlp":
            raise ValueError("V7 requires evidence_mlp unmatched prediction")
        self.velocity_head = EmbeddedInitDurationVelocityHead(
            cfg.motion, self.dim
        )

    def _temporal_attention_time(self, geometry):
        # Historical V7/V7.1 intentionally keep temporal feature conditioning
        # disabled. V7.2 overrides this hook with raw relative seconds.
        return None

    def _velocity_offset(self, refined, velocity_init, geometry):
        return self.velocity_head(
            refined,
            velocity_init,
            geometry["duration_sec"],
        )

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        if motion_proposal_feature is None:
            raise ValueError("V7 requires Utonia motion proposal features")
        proposal_fields = self.motion_proposal(
            motion_proposal_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["duration_sec"],
        )
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        velocity_match = (
            proposal_fields["delta_p_match"]
            / pair_delta_t_sec.unsqueeze(-1)
        )
        velocity_init = (
            proposal_fields["delta_p_init"]
            / pair_delta_t_sec.unsqueeze(-1)
        )
        # V7/V7.1 use their binary dustbin gate here; V7.2 supplies its exact
        # normalized Top-4 forward displacement. Detaching keeps attention loss
        # from reshaping the proposal through its own lookup coordinate, while
        # final-velocity loss still trains the proposal through the STE below.
        query_position_ref = (
            geometry["token_ref"]
            + velocity_init.detach() * pair_delta_t_sec.unsqueeze(-1)
        )
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_attention_time(geometry),
            query_position_ref=query_position_ref,
        )
        proposal_fields["pair_delta_t_sec"] = pair_delta_t_sec
        proposal_fields["velocity_match"] = velocity_match
        return refined, proposal_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del position
        pair_delta_t_sec = temporal_fields["pair_delta_t_sec"]
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self._velocity_offset(
            refined, velocity_init, geometry
        )
        velocity = velocity_init + velocity_offset

        fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": temporal_fields["velocity_match"],
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        for name in (
            "matched_position",
            "motion_top1_probability",
            "motion_effective_support",
            "motion_reciprocal_probability",
            "motion_candidate_entropy",
            "motion_best_candidate_displacement_m",
            "motion_reciprocal_probability_sum",
            "motion_mean_displacement_m",
            "motion_candidate_spread_m",
            "motion_normalized_mean_displacement",
            "motion_normalized_candidate_spread",
            "motion_dustbin_logit",
            "motion_soft_match_probability",
            "motion_hard_reject",
            "motion_search_speed_mps",
            "motion_candidate_probability_mass",
            "motion_selection_log_margin",
            "motion_hard_soft_displacement_cosine",
            "motion_hard_soft_displacement_norm_ratio",
            "motion_selected_distance_prior_penalty",
            "motion_match_support",
            "match_probability",
            "p_unmatched",
        ):
            if name in temporal_fields:
                fields[name] = temporal_fields[name]
        return velocity, fields


class StraightThroughProposalVelocityGaussianBackend(
    WarpedProposalVelocityGaussianBackend
):
    """V7.2: direct Top-4 STE, time-aware Q warp, and feature-only offset.

    The matcher remains time-free and consumes raw LoRA-adapted Utonia tokens.
    Temporal refinement then adds the historical V5 endpoint-time embedding
    before all four cross-attention layers while retaining the detached hard
    initializer Q-coordinate warp. The additive offset sees only the refined
    feature. Correspondence uses direct Utonia cosine, one fixed distance
    envelope, no reciprocal or unmatched terms, and a full-key softmax backward
    surrogate for the normalized Top-4 forward.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float,
                 proposal_dim: int | None = None):
        # Skip the V6/V7 SparseMotionProposal constructor and its dustbin/search
        # parameters while retaining the common physical Gaussian trunk.
        PhysicalVelocityGaussianBackend.__init__(
            self, cfg, gs_params, dim, offset_bound
        )
        if self.temporal.motion_head_count != 0:
            raise ValueError(
                "V7.2 motion proposal is independent; "
                "temporal.motion_head_count must be omitted or zero"
            )
        if not self.temporal.use_time_embedding:
            raise ValueError("V7.2 temporal attention must enable time embedding")
        self.motion_proposal = StraightThroughTop4MotionProposal(
            cfg.motion_proposal,
            self.dim if proposal_dim is None else int(proposal_dim),
        )
        self.velocity_head = FeatureOnlyVelocityOffsetHead(
            cfg.motion, self.dim
        )

    def _temporal_attention_time(self, geometry):
        return self._temporal_coordinate(geometry)

    def _velocity_offset(self, refined, velocity_init, geometry):
        # Q-warped, time-conditioned attention has already made the refined
        # feature depend on correspondence and cadence. Keep the final residual
        # decoder feature-only; v_init retains its separate additive path.
        del velocity_init, geometry
        return self.velocity_head(refined)



class PostAttentionProposalVelocityGaussianBackend(
    PhysicalVelocityGaussianBackend
):
    """V9: correspondence after temporal refinement, dense and offset-free.

    The temporal stack is the plain V3/V5 form -- endpoint time added once as
    ``f = f + E(t)``, twelve bidirectional cross-attention layers, and no
    query-coordinate warp, motion head, or matcher hook.  Its output ``f'``
    then feeds two independent consumers: the dense projected-descriptor
    proposal that produces ``v_init``, and the feature-only offset head that
    produces ``v_offset``.  Final motion is ``v_init + v_offset``.

    This is the structural difference from V6/V7/V7.2, whose proposals run
    *before* attention on raw LoRA Utonia tokens and therefore need a detached
    warp to feed correspondence back into the trunk.  Here the trunk feeds the
    matcher directly, so correspondence gradient reaches the same twelve layers
    that render the Gaussians and no detached path is required.
    """

    def __init__(self, cfg, gs_params, dim: int, offset_bound: float):
        super().__init__(cfg, gs_params, dim, offset_bound)
        if self.temporal.motion_head_count != 0:
            raise ValueError(
                "V9 motion proposal is a separate descriptor branch; "
                "temporal.motion_head_count must be omitted or zero"
            )
        if not self.temporal.use_time_embedding:
            raise ValueError("V9 temporal attention must enable time embedding")
        self.motion_proposal = ProjectedDenseMotionProposal(
            cfg.motion_proposal, self.dim
        )
        self.velocity_head = FeatureOnlyVelocityOffsetHead(
            cfg.motion, self.dim
        )

    @staticmethod
    def _signed_pair_delta_t(geometry):
        local_frame = geometry["local_frame"]
        if not bool(torch.all((local_frame == 0) | (local_frame == 1))):
            raise ValueError("V9 motion proposal requires endpoint indices 0/1")
        duration = geometry["duration_sec"]
        if not bool(torch.all(torch.isfinite(duration) & (duration > 0.0))):
            raise ValueError("V9 motion proposal requires positive finite duration")
        return torch.where(local_frame == 0, duration, -duration)

    def _temporal_refine(
        self,
        fused_feature,
        geometry,
        token_offset,
        frame_batch_idx,
        motion_proposal_feature=None,
    ):
        # V9 matches on the refined feature, so the raw pre-attention Utonia
        # tokens V6/V7 need are deliberately unused here.
        del motion_proposal_feature
        refined = self.temporal(
            fused_feature,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            self._temporal_coordinate(geometry),
        )
        proposal_fields = self.motion_proposal(
            refined,
            geometry["token_ref"],
            token_offset,
            frame_batch_idx,
            geometry["duration_sec"],
        )
        pair_delta_t_sec = self._signed_pair_delta_t(geometry)
        proposal_fields["pair_delta_t_sec"] = pair_delta_t_sec
        proposal_fields["velocity_match"] = (
            proposal_fields["delta_p_match"] / pair_delta_t_sec.unsqueeze(-1)
        )
        return refined, proposal_fields

    def _predict_motion(self, refined, geometry, position, temporal_fields):
        del geometry, position
        pair_delta_t_sec = temporal_fields["pair_delta_t_sec"]
        delta_p_init = temporal_fields["delta_p_init"]
        velocity_init = delta_p_init / pair_delta_t_sec.unsqueeze(-1)
        velocity_offset = self.velocity_head(refined)
        velocity = velocity_init + velocity_offset

        fields = {
            "delta_p_match": temporal_fields["delta_p_match"],
            "delta_p_init": delta_p_init,
            "pair_delta_t_sec": pair_delta_t_sec,
            "velocity_match": temporal_fields["velocity_match"],
            "velocity_init": velocity_init,
            "velocity_offset": velocity_offset,
        }
        for name in (
            "matched_position",
            "motion_top1_probability",
            "motion_effective_support",
            "motion_candidate_probability_mass",
            "motion_selection_log_margin",
            "motion_hard_soft_displacement_cosine",
            "motion_hard_soft_displacement_norm_ratio",
            "motion_selected_distance_prior_penalty",
        ):
            if name in temporal_fields:
                fields[name] = temporal_fields[name]
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
    "PostAttentionProposalVelocityGaussianBackend",
    "InitConditionedVelocityHead",
    "LayerWeightedAttentionVelocityGaussianBackend",
    "LayerWeightedDistanceBiasCrossAttention",
    "MaxSpeedBarrierLayerWeightedCrossAttention",
    "MaxSpeedBarrierVelocityGaussianBackend",
    "EmbeddedInitDurationVelocityHead",
    "FeatureOnlyVelocityOffsetHead",
    "ParallelBidirectionalCrossAttention",
    "ProjectedDenseMotionProposal",
    "ProposalInitializedVelocityGaussianBackend",
    "SeedConditionedGaussianAttributeHead",
    "SinusoidalScalarEncoder",
    "SparseMotionProposal",
    "StraightThroughProposalVelocityGaussianBackend",
    "StraightThroughTop4MotionProposal",
    "TimeConditionedParallelCrossAttention",
    "WarpedProposalVelocityGaussianBackend",
]
