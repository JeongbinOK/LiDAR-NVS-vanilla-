"""Attention-derived correspondence matching."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .common import _cfg_get

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


__all__ = ["ConsensusAttentionMotionMatcher"]
