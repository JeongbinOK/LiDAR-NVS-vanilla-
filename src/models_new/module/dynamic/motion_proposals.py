"""Paired-frame motion proposal implementations."""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import _cfg_get

class PairedFrameMotionProposal(nn.Module):
    """Endpoint pairing shared by every motion-proposal matcher.

    A dynamic sample is exactly two endpoint frames of ragged token counts.
    Every matcher therefore does the same three things: check that the caller's
    per-token tensors agree, cut the packed rows into frames, and run its
    matcher once per direction so the result is symmetric under a frame swap.
    Only the middle step -- how one direction is actually matched -- differs,
    and that is the single hook subclasses implement.
    """

    def _prepare_descriptors(self, feature, duration_sec):
        """Whatever the matcher needs per token before any pair is formed."""
        raise NotImplementedError

    def _match_frame_pair(
        self, context, position_ref, duration_sec, rows0, rows1
    ):
        """Match both directions of one endpoint pair, frame 0 first."""
        raise NotImplementedError

    def _finalize(self, result, context):
        """Per-token fields that are not produced per direction."""
        del context
        return result

    def _validate_tokens(self, feature, position_ref, duration_sec):
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
        return n_tokens

    @staticmethod
    def _endpoint_frames(token_offset, frame_batch_idx, n_tokens):
        counts = torch.diff(
            token_offset, prepend=token_offset.new_zeros(1)
        ).long()
        if int(counts.sum()) != n_tokens:
            raise ValueError(
                "token_offset does not cover all motion proposal tokens"
            )
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
        return counts, starts, batch_frames

    def forward(
        self,
        feature,
        position_ref,
        token_offset,
        frame_batch_idx,
        duration_sec,
    ):
        n_tokens = self._validate_tokens(feature, position_ref, duration_sec)
        counts, starts, batch_frames = self._endpoint_frames(
            token_offset, frame_batch_idx, n_tokens
        )
        context = self._prepare_descriptors(feature, duration_sec)
        by_frame = [None] * int(counts.numel())
        for frames in batch_frames.values():
            frame0, frame1 = frames
            start0, count0 = int(starts[frame0]), int(counts[frame0])
            start1, count1 = int(starts[frame1]), int(counts[frame1])
            by_frame[frame0], by_frame[frame1] = self._match_frame_pair(
                context,
                position_ref,
                duration_sec,
                slice(start0, start0 + count0),
                slice(start1, start1 + count1),
            )

        result = {
            name: torch.cat([frame[name] for frame in by_frame], dim=0)
            for name in by_frame[0]
        }
        return self._finalize(result, context)

class ChunkedDenseMotionProposal(PairedFrameMotionProposal):
    """Full-softmax correspondence over one endpoint pair, chunked by query.

    Both concrete matchers score every query token against every token in the
    opposite endpoint under

        ``score_ij = cos(d_i, d_j) / tau - (||x_i - x_j|| / (v_max * dt_i))^2``

    and materialize that matrix one query chunk at a time, recomputing
    train-time chunks in backward.  They differ in exactly two places: how the
    descriptor is produced (``_encode``), and how one chunk's scores become a
    coordinate plus its diagnostics (``_readout``).  The distance prior,
    chunking, checkpointing, and field naming are shared.
    """

    #: Names for the entries ``_readout`` returns after displacement/position.
    diagnostic_fields: tuple = ()

    def _validate_shared_config(self):
        if self.dim <= 0:
            raise ValueError("motion proposal dim must be positive")
        if self.temperature <= 0.0:
            raise ValueError("motion proposal temperature must be positive")
        if self.score_chunk_size <= 0:
            raise ValueError("motion proposal score_chunk_size must be positive")
        if self.distance_prior_speed_mps <= 0.0:
            raise ValueError(
                "motion proposal distance_prior_speed_mps must be positive"
            )

    def _encode(self, feature):
        """Token descriptor. The default is parameter-free L2 normalization."""
        return F.normalize(feature.float(), dim=-1, eps=1.0e-6)

    def _readout(
        self, score, distance_penalty, query_position, key_position,
        key_descriptor,
    ):
        """(displacement, matched_position, *diagnostics) for one chunk."""
        raise NotImplementedError

    def _pair_score(
        self, query_descriptor, key_descriptor, query_position, key_position,
        query_duration_sec,
    ):
        """Cosine content score under one fixed physical distance envelope.

        ``penalty_ij = (||x_i - x_j|| / (v_max * delta_t))^2``. The radius
        varies only with the sample duration, never with i/j or a learned
        token prediction. Centering improves squared-distance accuracy in
        global reference coordinates without changing the displacement.
        """
        content_score = (
            query_descriptor @ key_descriptor.transpose(0, 1)
        ) / self.temperature
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
        return content_score - distance_penalty, distance_penalty

    def _direction_chunk(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        query_duration_sec,
    ):
        score, distance_penalty = self._pair_score(
            query_descriptor, key_descriptor, query_position, key_position,
            query_duration_sec,
        )
        return self._readout(
            score, distance_penalty, query_position, key_position,
            key_descriptor,
        )

    def _match_direction(
        self,
        query_descriptor,
        key_descriptor,
        query_position,
        key_position,
        duration_sec=None,
    ):
        if int(query_descriptor.shape[0]) == 0 or int(
            key_descriptor.shape[0]
        ) == 0:
            # An empty frame would otherwise surface as an opaque IndexError
            # on the first chunk rather than as a stated precondition.
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
                and (
                    query_descriptor.requires_grad
                    or key_descriptor.requires_grad
                )
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
        displacement, matched_position = fields[0], fields[1]
        named = {
            "delta_p_match": displacement,
            # The proposal is the whole initializer; there is no separate
            # refinement stage between the match and v_init.
            "delta_p_init": displacement,
            "matched_position": matched_position,
        }
        names = self._active_diagnostic_fields()
        if len(names) != len(fields) - 2:
            raise RuntimeError(
                "motion proposal diagnostics disagree with the readout: "
                f"{len(names)} names for {len(fields) - 2} tensors"
            )
        named.update(zip(names, fields[2:]))
        return named

    def _active_diagnostic_fields(self):
        return self.diagnostic_fields

    def _prepare_descriptors(self, feature, duration_sec):
        del duration_sec
        return self._encode(feature)

    def _match_frame_pair(
        self, context, position_ref, duration_sec, rows0, rows1
    ):
        return (
            self._match_direction(
                context[rows0], context[rows1],
                position_ref[rows0], position_ref[rows1],
                duration_sec[rows0],
            ),
            self._match_direction(
                context[rows1], context[rows0],
                position_ref[rows1], position_ref[rows0],
                duration_sec[rows1],
            ),
        )

class SparseMotionProposal(PairedFrameMotionProposal):
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

    def _prepare_descriptors(self, feature, duration_sec):
        descriptor, search_speed, radius = self._encode(feature, duration_sec)
        dustbin_similarity = (
            self._token_dustbin_similarity(feature)
            if self.dustbin_mode == "in_softmax"
            else None
        )
        return descriptor, search_speed, radius, dustbin_similarity

    def _match_frame_pair(
        self, context, position_ref, duration_sec, rows0, rows1
    ):
        descriptor, _search_speed, radius, dustbin_similarity = context
        return self._match_pair(
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

    def _finalize(self, result, context):
        # The learned search speed is one value per token, not per direction.
        result["motion_search_speed_mps"] = context[1]
        return result

class StraightThroughTop4MotionProposal(ChunkedDenseMotionProposal):
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

    diagnostic_fields = (
        "motion_selected_index",
        "motion_top1_probability",
        "motion_effective_support",
        "motion_candidate_probability_mass",
        "motion_selection_log_margin",
        "motion_hard_soft_displacement_cosine",
        "motion_hard_soft_displacement_norm_ratio",
        "motion_selected_distance_prior_penalty",
        "motion_match_support",
    )

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
        self._validate_shared_config()
        if self.descriptor_mode != "direct_l2":
            raise ValueError("V7.2 requires direct_l2 motion descriptors")
        if self.match_count != 4:
            raise ValueError("V7.2 requires exactly four forward matches")
        if self.ste_surrogate != "dense_softmax":
            raise ValueError("V7.2 requires ste_surrogate='dense_softmax'")

    def _readout(
        self, score, distance_penalty, query_position, key_position,
        key_descriptor,
    ):
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

class ProjectedDenseMotionProposal(ChunkedDenseMotionProposal):
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

    diagnostic_fields = (
        "motion_top1_probability",
        "motion_effective_support",
        "motion_candidate_probability_mass",
        "motion_selection_log_margin",
        "motion_hard_soft_displacement_cosine",
        "motion_hard_soft_displacement_norm_ratio",
        "motion_selected_distance_prior_penalty",
    )

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
        self._validate_shared_config()
        if self.descriptor_dim <= 0:
            raise ValueError("motion proposal descriptor_dim must be positive")
        if self.descriptor_mode != "projected_l2":
            raise ValueError("V9 requires projected_l2 motion descriptors")
        if self.readout != "dense_expectation":
            raise ValueError("V9 requires readout='dense_expectation'")

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

    def _active_diagnostic_fields(self):
        return self.diagnostic_fields if self.collect_diagnostics else ()

    def _readout(
        self, score, distance_penalty, query_position, key_position,
        key_descriptor,
    ):
        # The distance prior is identical in form to V7.2's; only v_max changes
        # (30 -> 5 m/s), so at a one-second interval 5, 10, and 15 m now cost
        # 1.0, 4.0, and 9.0 logits where V7.2 charged 0.028, 0.111, and 0.25.
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

__all__ = [
    "PairedFrameMotionProposal",
    "ChunkedDenseMotionProposal",
    "SparseMotionProposal",
    "StraightThroughTop4MotionProposal",
    "ProjectedDenseMotionProposal",
 ]
