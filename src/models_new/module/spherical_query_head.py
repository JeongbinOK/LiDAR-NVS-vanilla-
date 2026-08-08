"""Spherical query-anchor head (anchor_mode="spherical").

For spherical mode, each frame's own raw points are labelled bg / dynamic
instance by that frame's bboxes (dynamic = instance shared by both endpoint
frames) and then binned into (theta, phi, log r) spherical cells about the
sensor origin. Every occupied **(cell, label) group** becomes an anchor (its
own 1x1x1 cell only -- no neighbour stencil): a boundary cell holding car and
road points splits into one fg anchor per instance plus one bg anchor instead
of voting, so no gaussian is ever seeded on the other label's points and no
raw point is dropped. This equals running the same spherical binning on the
bg-only and per-instance point subsets separately.

``squery.count_mode`` selects one of two checkpoint-isolated output paths:

* ``legacy`` keeps the historical behaviour: every unique
  ``(anchor, supporting token)`` pair is one evidence query and one Gaussian.
  Its coarse centre is the observed point nearest that evidence set's Cartesian
  mean; the source token query cross-attends to own-frame token memory.
* ``learned_gumbel`` makes the labelled spherical group the output unit. Its
  connected fused tokens follow Utonia GridPooling (Linear -> max -> LN ->
  GELU), then same-frame/same-label 3x3x3 local self-attention. A separate
  float64 raw-geometry statistic encoder is concatenated only for K routing;
  K-specific observed range-quantile points and the content feature feed the
  same joint K-head used by grid learned routing.

Cross-frame temporal context is NOT assembled here. The caller first runs the
shared grid temporal aggregator (background 0.8 m radius cross-frame attention
+ per-instance box-local self-attention) over the fused tokens, so this head
receives temporally fused features and every K/V route stays within the
anchor's OWN frame. Tokens carry no fg/bg labels anywhere -- they are only
ever reached through raw-point membership:
- bg anchors live in the batch's ref frame (frame 0). K/V = the tokens of the
  group's own raw points (exactly the anchor-embedding set, dedup'd).
- fg anchors live in bbox-local coords. K/V = the tokens reached by the SAME
  instance's raw points in the anchor's own frame (dedup'd, instance-wide
  rather than cell-member-only -- objects hold few tokens); each token's RoPE
  position is the own-frame instance-box-local position, matching the query
  seed frame.
- K/V features are the fused token features themselves (LayerNorm + the
  attention's W_k/W_v only, no extra embedding); K/V RoPE positions are the
  Utonia token positions (features['coord']-derived, not grid_coord).
- Both are capped to `max_kv` nearest tokens (|token - cell centre|).
- BG/FG share learned query/cross-attention weights, but use independent RoPE
  metric scales selected per anchor to respect their different spatial support.

Outputs remain frame-major. Legacy feeds the shared ``gs_predictor``;
learned routing returns an anchor-level K-specific seed bank for ``GridSlotHead``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..utils import boxes as box_utils
from ..utils.attention import (
    AnchorQueryCrossAttention,
    AnchorQuerySelfAttention,
    Rotary3D,
)
from .spherical_bins import (
    SphericalBins,
    build_cells,
)


# Fixed geometry constants, not experiment knobs.  The operator/frequency
# schedule matches Utonia Point3DRoPE; metric scales follow scale =
# pi / p95(max_axis |kv_pos - seed_pos|) measured over 100 train windows with
# the raw-cell label-group pair assembly
# (scripts/attention_rope_pair_stats_train_100_rawsplit: bg p95 1.772 m ->
# pi/p95 = 1.77 -> 1.75; fg p95 2.325 m -> 1.35 -- both land on the pre-split
# values, so the constants carry over measured, not assumed). The own-frame
# K/V restriction only removes pairs from that measured assembly (bg drops
# the same-cell cross-frame rebin, fg drops the other frame's box-local
# tokens), so the p95 support can only shrink and the scales stay valid
# upper-bound picks.
_SPHERICAL_ROPE_BASE = 10.0
_SPHERICAL_BG_ROPE_POSITION_SCALE = 1.75
_SPHERICAL_FG_ROPE_POSITION_SCALE = 1.35

# --- adaptive per-anchor metric scale (default) ------------------------------
# The constants above are single numbers fitted to one p95 measured under an
# EARLIER bin config. Re-measured 2026-07-21 on the current config
# (dr_m=0.8, r_max=110, instance-wide fg K/V), 5 windows:
#   bg: support p95 1.77 -> 0.85..1.13 m  => 1.75 spends only 47..63% of the
#       usable pi range (wasted resolution, not a correctness bug)
#   fg: support p95 1.77..3.96 m, max 7.8 m => 1.35 puts 1.0 / 4.6 / 31.4% of
#       fg pairs past the wrap threshold pi/1.35 = 2.33 m (ALIASING: two
#       different displacements collapse onto the same phase)
# `Rotary3D.inv_freq[0] == 1`, so the lowest band's relative phase is exactly
# |delta_axis| * scale and it wraps beyond pi/scale metres.
#
# A per-object bound (pi / max(w,l,h)) does NOT fix fg: the fg K/V is
# instance-wide and deliberately keeps tokens whose box-local centroid drifts
# OUTSIDE the box, which is where the 7.8 m tail comes from. So the bound is
# taken from each anchor's REALIZED support -- the axis-aligned extent of that
# anchor's own {K/V positions} U {query seeds} -- which upper-bounds every
# pairwise |delta_axis| in that attention group by construction. RoPE phases
# are only ever compared within one anchor, so a per-anchor metric scale is the
# natural normalisation (and `position_scale` already accepts an (A,) tensor
# without splitting any learned weights).
_SPHERICAL_ROPE_SCALE_MIN = 0.25   # support <= 12.3 m
_SPHERICAL_ROPE_SCALE_MAX = 8.0    # support >= 0.38 m
# The pair that DEFINES the support would otherwise land exactly on pi, where
# +pi and -pi are the same phase and fp32 rounding puts it on either side.
_SPHERICAL_ROPE_PHASE_MARGIN = 0.98


class _SphericalLocalSelfAttention(nn.Module):
    """Sparse 3x3x3 anchor self-attention with metric 3D RoPE.

    ``neighbor_index`` is a padded ``(A, <=27)`` lookup built from spherical
    cell indices.  Each layer re-projects the *updated* features of those
    neighbours, so stacking layers genuinely grows the receptive field rather
    than repeatedly cross-attending to a frozen one-hop memory.
    """

    def __init__(self, dim, num_heads=8, n_layers=2, mlp_ratio=4):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError("local attention dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 6 != 0:
            raise ValueError(
                "local attention head_dim must be divisible by 6 for 3D RoPE"
            )
        self.n_layers = int(n_layers)
        self.scale = self.head_dim ** -0.5
        self.rope = Rotary3D(
            self.head_dim, base=_SPHERICAL_ROPE_BASE, position_scale=1.0
        )
        self.q_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.kv_proj = nn.ModuleList([
            nn.Linear(self.dim, 2 * self.dim) for _ in range(self.n_layers)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(self.dim, self.dim) for _ in range(self.n_layers)
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        hidden = int(self.dim * mlp_ratio)
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.dim),
            )
            for _ in range(self.n_layers)
        ])
        self.norm_ffn = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.n_layers)
        ])
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    def forward(self, feature, position, neighbor_index, neighbor_valid):
        A = int(feature.shape[0])
        if A == 0 or self.n_layers == 0:
            return feature
        if neighbor_index.ndim != 2 or neighbor_index.shape[0] != A:
            raise ValueError("neighbor_index must have shape (A, L)")
        if neighbor_valid.shape != neighbor_index.shape:
            raise ValueError("neighbor_valid must align with neighbor_index")
        if not bool(neighbor_valid.any(dim=1).all()):
            raise RuntimeError("every occupied anchor must attend to at least itself")

        safe_index = neighbor_index.clamp(0, A - 1)
        neighbor_position = position[safe_index].float()
        query_position = position.float()
        relative = (neighbor_position - query_position[:, None, :]).abs()
        relative = torch.where(
            neighbor_valid[..., None], relative, torch.zeros_like(relative)
        )
        support = relative.amax(dim=(1, 2))
        position_scale = (
            _SPHERICAL_ROPE_PHASE_MARGIN * torch.pi
            / support.clamp_min(1.0e-3)
        ).clamp(_SPHERICAL_ROPE_SCALE_MIN, _SPHERICAL_ROPE_SCALE_MAX)
        q_angle = self.rope.angles(
            query_position, position_scale=position_scale
        )
        k_angle = self.rope.angles(
            neighbor_position, position_scale=position_scale
        )
        negative_mask = ~neighbor_valid[:, None, :]

        x = feature
        H, dh = self.num_heads, self.head_dim
        for layer in range(self.n_layers):
            q = self.q_proj[layer](x).reshape(A, H, dh)
            key_all, value_all = self.kv_proj[layer](x).chunk(2, dim=-1)
            key = key_all[safe_index].reshape(A, safe_index.shape[1], H, dh)
            value = value_all[safe_index].reshape(
                A, safe_index.shape[1], H, dh
            )
            q_rot = self.rope.rotate(q.float(), q_angle[:, None, :, :])
            key_rot = self.rope.rotate(
                key.float(), k_angle[:, :, None, :, :]
            )
            score = torch.einsum("ahd,alhd->ahl", q_rot, key_rot) * self.scale
            score = score.masked_fill(negative_mask, float("-inf"))
            probability = torch.softmax(score, dim=-1)
            attended = torch.einsum(
                "ahl,alhd->ahd", probability, value.float()
            ).reshape(A, self.dim).to(x.dtype)
            x = self.norm[layer](x + self.out_proj[layer](attended))
            x = self.norm_ffn[layer](x + self.ffn[layer](x))
        return x


def _range_quantile_seed_rows(points, anchor_index, num_anchors, k_max):
    """Grid-mode K-specific range-quantile rule, returning raw-point rows.

    For candidate K and slot s, select the observed point at

        floor(((2s + 1) * N_raw) / (2K)).

    after sorting each anchor's points by sensor range (then x/y/z for stable
    ties). Thus every active seed lies on an actually observed surface. When
    ``N_raw < K``, repeated raw points are intentional and match Grid mode.
    """
    A, K = int(num_anchors), int(k_max)
    if points.shape[0] == 0 or A == 0:
        return torch.zeros(
            (A, K, K), dtype=torch.long, device=points.device
        )
    counts = torch.bincount(anchor_index, minlength=A)
    if bool((counts <= 0).any()):
        raise RuntimeError("every spherical anchor needs at least one raw point")

    point_range = torch.linalg.vector_norm(points, dim=-1)
    # Stable least-to-most-significant sorting gives (anchor, r, x, y, z),
    # exactly as `_r_quantile_seed_bank` in builders/common.py.
    sorted_rows = torch.arange(points.shape[0], device=points.device)
    for value in (
        points[:, 2], points[:, 1], points[:, 0], point_range, anchor_index,
    ):
        sorted_rows = sorted_rows[
            torch.argsort(value[sorted_rows], stable=True)
        ]
    starts = torch.cumsum(counts, dim=0) - counts

    seed_rows = torch.zeros((A, K, K), dtype=torch.long, device=points.device)
    for k in range(1, K + 1):
        slot = torch.arange(k, device=points.device).view(1, -1)
        rank = torch.div(
            (2 * slot + 1) * counts[:, None],
            2 * k,
            rounding_mode="floor",
        )
        seed_rows[:, k - 1, :k] = sorted_rows[starts[:, None] + rank]
    return seed_rows


def _segment_rank(sorted_ids, num_segments):
    """Ranks 0,1,2,... within each contiguous id block of an ascending id tensor."""
    counts = torch.bincount(sorted_ids, minlength=num_segments)
    starts = torch.zeros(num_segments, dtype=torch.long, device=sorted_ids.device)
    if num_segments > 1:
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
    return torch.arange(sorted_ids.numel(), device=sorted_ids.device) - starts[sorted_ids]


def _select_evidence_seeds(raw_points, raw_anchor, raw_token, num_anchors, num_tokens):
    """One observed medoid-like seed per unique ``(anchor, token)`` evidence.

    For each non-empty pair, compute its raw-point Cartesian mean and select the
    actually observed point nearest that mean. Ties use deterministic x/y/z
    ordering. The discrete grouping and argmin operate only on observation
    geometry; gathered token features remain fully differentiable downstream.
    """
    A = int(num_anchors)
    N = int(num_tokens)
    if raw_points.ndim != 2 or raw_points.shape[-1] != 3:
        raise ValueError(f"raw_points must have shape (R,3), got {tuple(raw_points.shape)}")
    if raw_anchor.ndim != 1 or raw_anchor.shape[0] != raw_points.shape[0]:
        raise ValueError("raw_anchor must provide one anchor index per raw point")
    if raw_token.ndim != 1 or raw_token.shape[0] != raw_points.shape[0]:
        raise ValueError("raw_token must provide one token index per raw point")
    if raw_points.shape[0] == 0:
        raise RuntimeError("occupied spherical anchors require non-empty raw membership")
    if (
        (raw_anchor < 0).any() or (raw_anchor >= A).any()
        or (raw_token < 0).any() or (raw_token >= N).any()
    ):
        raise ValueError("raw anchor/token membership contains an out-of-range index")

    anchor_counts = torch.bincount(raw_anchor, minlength=A)
    missing = anchor_counts == 0
    if missing.any():
        missing_ids = missing.nonzero(as_tuple=True)[0]
        raise RuntimeError(
            "occupied spherical anchors must have at least one own-frame raw point; "
            f"missing {int(missing_ids.numel())} / {A} anchors"
        )

    pair_key = raw_anchor * N + raw_token
    unique_key, point_to_evidence = torch.unique(
        pair_key, sorted=True, return_inverse=True
    )
    query_anchor = torch.div(unique_key, N, rounding_mode="floor")
    query_token = unique_key - query_anchor * N
    E = int(unique_key.numel())
    evidence_counts = torch.bincount(point_to_evidence, minlength=E)
    point_sum = raw_points.new_zeros(E, 3)
    point_sum.index_add_(0, point_to_evidence, raw_points)
    point_mean = point_sum / evidence_counts.to(raw_points.dtype).unsqueeze(-1)
    distance2 = ((raw_points - point_mean[point_to_evidence]) ** 2).sum(dim=-1)
    # Raw coordinates are fp32, so analytically symmetric points can differ by
    # about 1e-7 m^2 after the mean/subtraction (especially at long range).
    # Quantize only the ordering key at 1e-6 m^2 so such numerical ties obey the
    # documented x/y/z rule; selected coordinates remain the original raw row.
    distance_key = torch.round(distance2 * 1.0e6)

    # Stable least-to-most-significant sorting gives
    # (evidence id, distance-to-mean, x, y, z). The first row of every evidence
    # block is therefore the deterministic observed seed.
    order = torch.arange(raw_points.shape[0], device=raw_points.device)
    for value in (
        raw_points[:, 2], raw_points[:, 1], raw_points[:, 0],
        distance_key, point_to_evidence,
    ):
        order = order[torch.argsort(value[order], stable=True)]
    starts = torch.cumsum(evidence_counts, dim=0) - evidence_counts
    seed_row = order[starts]
    return (
        raw_points[seed_row], query_anchor, query_token, seed_row,
        anchor_counts, evidence_counts,
    )


class SphericalQueryHead(nn.Module):
    def __init__(
        self, squery_cfg, dim, r_far, *, ring_to_elevation_deg=None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.count_mode = str(
            getattr(squery_cfg, "count_mode", "legacy")
        ).lower()
        if self.count_mode not in ("legacy", "learned_gumbel"):
            raise ValueError(
                "p2g.squery.count_mode must be 'legacy' or 'learned_gumbel'"
            )
        self.max_kv = int(getattr(squery_cfg, "max_kv", 64))
        self.token_stride_m = float(getattr(squery_cfg, "token_stride_m", 0.4))
        if self.token_stride_m <= 0.0:
            raise ValueError("p2g.squery.token_stride_m must be positive")
        dr_m = getattr(squery_cfg, "dr_m", None)
        self.bins = SphericalBins(
            float(squery_cfg.dtheta_deg), float(squery_cfg.dphi_deg),
            getattr(squery_cfg, "dlogr", None),
            float(squery_cfg.r_min), float(squery_cfg.r_max),
            dr_m=float(dr_m) if dr_m is not None else None,
        )

        if self.count_mode == "learned_gumbel":
            learned = getattr(squery_cfg, "learned_count", None)
            if learned is None:
                raise ValueError(
                    "p2g.squery.learned_count is required for learned_gumbel"
                )
            self.k_max = int(getattr(learned, "K_max", 0) or 0)
            if self.k_max <= 0:
                raise ValueError("p2g.squery.learned_count.K_max must be positive")
            self.stat_embedding_dim = int(
                getattr(learned, "stat_embedding_dim", 64)
            )
            if self.stat_embedding_dim <= 0:
                raise ValueError("stat_embedding_dim must be positive")
            self.router_dim = self.dim + self.stat_embedding_dim
            self.stat_min_raw = int(getattr(learned, "stat_min_raw", 6))
            self.stat_min_valid_loo = int(
                getattr(learned, "stat_min_valid_loo", 3)
            )
            self.stat_loo_eps = float(
                getattr(learned, "stat_loo_eps", 1.0e-3)
            )
            self.stat_range_min_m = float(
                getattr(learned, "stat_range_min_m", 2.5)
            )
            self.stat_range_max_m = float(
                getattr(learned, "stat_range_max_m", self.bins.r_max)
            )
            self.stat_loo_knee_m = float(
                getattr(learned, "stat_loo_knee_m", 0.01)
            )
            self.stat_loo_cap_m = float(
                getattr(
                    learned,
                    "stat_loo_cap_m",
                    float(dr_m) if dr_m is not None else 0.8,
                )
            )
            if self.stat_min_raw < 1 or self.stat_min_valid_loo < 1:
                raise ValueError("statistic support thresholds must be positive")
            if not 0.0 < self.stat_loo_eps < 1.0:
                raise ValueError("stat_loo_eps must lie in (0, 1)")
            if not self.stat_range_min_m < self.stat_range_max_m:
                raise ValueError("statistic range bounds must be ordered")
            if self.stat_loo_knee_m <= 0.0 or self.stat_loo_cap_m <= 0.0:
                raise ValueError("LOO statistic knee/cap must be positive")
            self.stat_loo_log_denominator = math.log1p(
                self.stat_loo_cap_m / self.stat_loo_knee_m
            )

            # Utonia GridPooling order: Linear -> segment max -> LN -> GELU.
            self.anchor_pool_linear = nn.Linear(self.dim, self.dim)
            self.anchor_pool_norm = nn.LayerNorm(self.dim)
            self.anchor_pool_act = nn.GELU()
            local_heads = int(getattr(learned, "local_attn_heads", 8))
            local_layers = int(getattr(learned, "local_attn_layers", 2))
            self.local_attention = _SphericalLocalSelfAttention(
                self.dim,
                num_heads=local_heads,
                n_layers=local_layers,
                mlp_ratio=int(getattr(learned, "local_ffn_ratio", 4)),
            )
            # Two Linear layers as requested; LayerNorm keeps the six input
            # channels from relying on a dataset-wide mean/std calibration.
            self.stat_encoder = nn.Sequential(
                nn.Linear(6, self.stat_embedding_dim),
                nn.LayerNorm(self.stat_embedding_dim),
                nn.GELU(),
                nn.Linear(self.stat_embedding_dim, self.stat_embedding_dim),
                nn.LayerNorm(self.stat_embedding_dim),
                nn.GELU(),
            )

            if ring_to_elevation_deg is None:
                raise ValueError(
                    "learned spherical routing requires p2g.ring_to_elevation_deg"
                )
            elevation = torch.as_tensor(
                list(ring_to_elevation_deg), dtype=torch.float64
            )
            theta_index = torch.floor(
                (elevation + 90.0) / float(squery_cfg.dtheta_deg)
            ).to(torch.long).clamp(0, self.bins.n_theta - 1)
            ring_count = torch.bincount(
                theta_index, minlength=self.bins.n_theta
            ).to(torch.float32)

            # Expected emitted-ray capacity, not the integer capacity of one
            # arbitrarily phase-aligned panorama. Every phi cell receives the
            # same 1085 / n_phi expectation; theta capacity still follows the
            # exact number of physical LiDAR rings assigned to that theta bin.
            ray_azimuth_count = int(
                getattr(learned, "ray_azimuth_count", 1085)
            )
            if ray_azimuth_count <= 0:
                raise ValueError("ray_azimuth_count must be positive")
            expected_phi_rays = float(ray_azimuth_count) / self.bins.n_phi
            ray_capacity = (
                ring_count[:, None]
                .expand(-1, self.bins.n_phi)
                .contiguous()
                * expected_phi_rays
            )
            self.register_buffer(
                "ray_capacity", ray_capacity, persistent=False
            )
            return

        # Keep the checkpoint-facing normalization name, but normalize each
        # evidence's own source-token feature rather than an anchor-pooled P0.
        self.p0_norm = nn.LayerNorm(self.dim)

        # The fused token trunk has no final normalization. Normalize raw token
        # features once, then let the attention's W_k/W_v form K and V.
        self.kv_norm = nn.LayerNorm(self.dim)

        self.bg_rope_position_scale = _SPHERICAL_BG_ROPE_POSITION_SCALE
        self.fg_rope_position_scale = _SPHERICAL_FG_ROPE_POSITION_SCALE
        # Separate switches so a from-scratch A/B can attribute the two effects
        # (fg is a real aliasing bug, bg is only under-utilisation).
        self.bg_rope_adaptive = str(
            getattr(squery_cfg, "rope_scale_bg", "adaptive")) == "adaptive"
        self.fg_rope_adaptive = str(
            getattr(squery_cfg, "rope_scale_fg", "adaptive")) == "adaptive"

        self.attn = AnchorQueryCrossAttention(
            self.dim, num_heads=8,
            n_layers=int(getattr(squery_cfg, "n_layers", 1) or 1),
            chunk=int(getattr(squery_cfg, "attn_chunk", 8192) or 8192),
            mlp_ratio=int(getattr(squery_cfg, "ffn_ratio", 4) or 4),
            rope_base=_SPHERICAL_ROPE_BASE,
            # Direct calls default to BG scale. Normal spherical forward passes
            # override it per anchor below without splitting learned weights.
            rope_position_scale=self.bg_rope_position_scale,
            varlen_backend=str(
                getattr(squery_cfg, "cross_attn_backend", "fp32_bucket")
            ),
        )
        self.query_self_attn = AnchorQuerySelfAttention(
            self.dim,
            num_heads=8,
            n_layers=int(
                getattr(squery_cfg, "self_attn_layers", 1) or 1
            ),
            mlp_ratio=int(getattr(squery_cfg, "ffn_ratio", 4) or 4),
            rope_base=_SPHERICAL_ROPE_BASE,
            rope_position_scale=self.bg_rope_position_scale,
            varlen_backend=str(
                getattr(squery_cfg, "self_attn_backend", "fp32_bucket")
            ),
        )

    # ------------------------------------------------------------------ stages
    def _token_meta(self, tok_pos, raw_sensor, raw_token_index, new_offset,
                    frame_batch_idx, pose_list, bbox_list, bbox_iids_list,
                    token_ref=None, bbox_ref_by_frame=None):
        """Per-token AND per-raw-point metadata over ALL frames (computed once,
        carried everywhere). Tokens get only their frame and ref coordinates --
        they carry no fg/bg label and are reached exclusively through raw-point
        membership. Raw points get ref coordinates, bbox labels, and their
        global frame (they drive anchor formation and every K/V route).

        ``token_ref`` (per-token ref coords) and ``bbox_ref_by_frame`` are
        bit-identical to what the shared GridTemporalAggregator already produced
        upstream (same pure ``apply_pose`` / ``transform_boxes_to_ref`` on the
        same token positions, poses and boxes). The caller threads them in so
        this head reuses them instead of recomputing; only the raw-point
        labelling in the frame loop below is head-specific. When absent (e.g.
        direct unit-test calls) both fall back to being recomputed here."""
        device = tok_pos.device
        N = tok_pos.shape[0]
        R = raw_sensor.shape[0]
        n_frames = len(frame_batch_idx)

        reuse_ref = token_ref is not None
        tok_ref = token_ref if reuse_ref else torch.zeros_like(tok_pos)
        reuse_bbox = bbox_ref_by_frame is not None
        bbox_ref_by_frame = (
            list(bbox_ref_by_frame) if reuse_bbox else [None] * n_frames
        )

        tok_frame = torch.zeros(N, dtype=torch.long, device=device)
        raw_frame = torch.zeros(R, dtype=torch.long, device=device)
        raw_ref = torch.zeros_like(raw_sensor)
        raw_label = torch.full((R,), -1, dtype=torch.long, device=device)

        frame_slices = [None] * n_frames             # (start, end) per global frame
        iids_by_frame = [None] * n_frames

        batch_frame_map = {}
        for global_f, b in enumerate(frame_batch_idx.tolist()):
            batch_frame_map.setdefault(int(b), []).append(global_f)

        for b, frame_indices in batch_frame_map.items():
            pose_b = pose_list[b]
            bbox_b = bbox_list[b]
            iids_b = bbox_iids_list[b] if bbox_iids_list is not None else None
            if iids_b is not None and len(iids_b) >= 2:
                common_ids = box_utils.common_instance_ids(iids_b[0], iids_b[-1])
            else:
                common_ids = set()

            for local_f, global_f in enumerate(frame_indices):
                prev = int(new_offset[global_f - 1]) if global_f > 0 else 0
                end = int(new_offset[global_f])
                sl = slice(prev, end)
                frame_slices[global_f] = (prev, end)

                pose_f = pose_b[local_f].to(device)
                tok_frame[sl] = global_f
                if not reuse_ref:
                    tok_ref[sl] = box_utils.apply_pose(tok_pos[sl], pose_f)

                bbox_sensor_f = bbox_b[local_f].to(device)
                if not reuse_bbox:
                    bbox_ref_by_frame[global_f] = box_utils.transform_boxes_to_ref(
                        bbox_sensor_f, pose_f
                    )
                if iids_b is not None:
                    iids_f = iids_b[local_f].to(device)
                else:
                    iids_f = torch.arange(bbox_sensor_f.shape[0], device=device, dtype=torch.long)
                iids_by_frame[global_f] = iids_f

                # Raw points of this frame: the membership rows are frame-
                # contiguous, so the frame's token-row range identifies them.
                rmask = (raw_token_index >= prev) & (raw_token_index < end)
                if rmask.any():
                    rxyz = raw_sensor[rmask]
                    raw_frame[rmask] = global_f
                    raw_ref[rmask] = box_utils.apply_pose(rxyz, pose_f)
                    rbox = box_utils.point_in_box(rxyz, bbox_sensor_f)
                    rinst = torch.full_like(rbox, -1)
                    rin = rbox >= 0
                    if rin.any() and iids_f.numel() > 0:
                        rinst[rin] = iids_f[rbox[rin]]
                    rdyn = torch.zeros_like(rin)
                    for inst_id in common_ids:
                        rdyn |= rinst == int(inst_id)
                    raw_label[rmask] = torch.where(
                        rdyn, rinst, torch.full_like(rinst, -1)
                    )

        return {
            "frame": tok_frame, "ref": tok_ref,
            "raw_frame": raw_frame, "raw_ref": raw_ref, "raw_label": raw_label,
            "frame_slices": frame_slices, "bbox_ref_by_frame": bbox_ref_by_frame,
            "iids_by_frame": iids_by_frame, "batch_frame_map": batch_frame_map,
        }

    def _frame_anchors(self, tm, raw_sensor, pose_f, global_f):
        """Anchors of one frame: occupied (spherical cell, raw label) groups.

        Returns None when the frame has no in-range raw point, else a dict
        with the anchor arrays. A mixed boundary cell splits into one anchor
        per label (bg first, then instances in ascending id order) instead of
        voting, so every group is label-pure and no raw point is dropped. This
        equals binning the bg-only and per-instance point subsets separately.
        All split anchors of one cell share the geometric cell centre as their
        reference position."""
        device = raw_sensor.device
        pi = (tm["raw_frame"] == global_f).nonzero(as_tuple=True)[0]
        if pi.numel() == 0:
            return None
        idx3, valid = self.bins.bin_coords(raw_sensor[pi])
        pi = pi[valid]
        if pi.numel() == 0:
            return None

        cell_of_point = self.bins.hash(idx3[valid])
        labels_p = tm["raw_label"][pi]
        uniq_lab, lab_idx = torch.unique(labels_p, return_inverse=True)
        L = uniq_lab.numel()
        group_key, pt2anchor = build_cells(cell_of_point * L + lab_idx)
        U = group_key.numel()
        anchor_cell = torch.div(group_key, L, rounding_mode="floor")
        anchor_label_index = group_key - anchor_cell * L
        anchor_label = uniq_lab[anchor_label_index]
        is_dyn = anchor_label >= 0

        # Anchor reference position = geometric spherical-cell centre; delta_p
        # and the max_kv capping distance are measured from it downstream.
        anchor_idx3 = self.bins.unhash(anchor_cell)
        anchor_sensor = self.bins.cell_center_xyz(anchor_idx3).to(
            dtype=raw_sensor.dtype
        )
        anchor_ref = box_utils.apply_pose(anchor_sensor, pose_f)
        anchor_out = anchor_ref.clone()

        # frame-local box index of the group instance (fg anchors only). A
        # dynamic raw label is produced by indexing this frame's own iids, so
        # a lookup miss is impossible -- fail loudly rather than emit anchors
        # with a bogus box id.
        anchor_box = torch.full((U,), -1, dtype=torch.long, device=device)
        iids_f = tm["iids_by_frame"][global_f]
        bbox_ref_f = tm["bbox_ref_by_frame"][global_f]
        if is_dyn.any():
            dyn_idx = is_dyn.nonzero(as_tuple=True)[0]
            eq = anchor_label[dyn_idx].unsqueeze(1) == iids_f.unsqueeze(0)  # (Ud, B_f)
            if not bool(eq.any(dim=1).all()):
                raise RuntimeError(
                    "dynamic raw label without a matching own-frame box; "
                    "raw labelling and bbox inputs are inconsistent"
                )
            bidx = eq.long().argmax(dim=1)
            anchor_box[dyn_idx] = bidx
            # One batched box-local transform (each anchor gathers its own box)
            # instead of a launch per unique box.
            anchor_out[dyn_idx] = box_utils.points_to_box_local_batched(
                anchor_ref[dyn_idx], bbox_ref_f[bidx]
            )

        return {
            "pt2anchor": pt2anchor, "pi": pi, "num_anchors": U,
            "label": anchor_label, "box": anchor_box, "is_dyn": is_dyn,
            "sensor": anchor_sensor, "ref": anchor_ref, "out": anchor_out,
            "idx3": anchor_idx3, "group_key": group_key,
            "label_index": anchor_label_index, "num_labels": int(L),
        }

    def _local_neighbor_table(self, frame_anchor, anchor_base, num_anchors):
        """Exact same-frame/same-label 3x3x3 spherical neighbourhood."""
        A = int(num_anchors)
        device = next(
            fa["idx3"].device for fa in frame_anchor if fa is not None
        )
        offsets = torch.tensor(
            [[dt, dp, dr] for dt in (-1, 0, 1)
             for dp in (-1, 0, 1) for dr in (-1, 0, 1)],
            dtype=torch.long,
            device=device,
        )
        neighbor_index = torch.arange(device=device, end=A)[:, None].expand(
            A, offsets.shape[0]
        ).clone()
        neighbor_valid = torch.zeros_like(neighbor_index, dtype=torch.bool)
        for global_f, fa in enumerate(frame_anchor):
            if fa is None:
                continue
            U = int(fa["num_anchors"])
            base = int(anchor_base[global_f])
            candidate = fa["idx3"][:, None, :] + offsets[None, :, :]
            in_bounds = (
                (candidate[..., 0] >= 0)
                & (candidate[..., 0] < self.bins.n_theta)
                & (candidate[..., 2] >= 0)
                & (candidate[..., 2] < self.bins.n_lr)
            )
            candidate = candidate.clone()
            candidate[..., 0] = candidate[..., 0].clamp(
                0, self.bins.n_theta - 1
            )
            candidate[..., 1] = torch.remainder(
                candidate[..., 1], self.bins.n_phi
            )
            candidate[..., 2] = candidate[..., 2].clamp(
                0, self.bins.n_lr - 1
            )
            neighbor_key = (
                self.bins.hash(candidate) * fa["num_labels"]
                + fa["label_index"][:, None]
            )
            position = torch.searchsorted(fa["group_key"], neighbor_key)
            safe_position = position.clamp(max=U - 1)
            found = (
                in_bounds
                & (position < U)
                & (fa["group_key"][safe_position] == neighbor_key)
            )
            neighbor_index[base:base + U] = base + safe_position
            neighbor_valid[base:base + U] = found
        if not bool(neighbor_valid.any(dim=1).all()):
            raise RuntimeError("spherical local neighbourhood lost its self cell")
        return neighbor_index, neighbor_valid

    @staticmethod
    def _segmented_quantiles(
        value, anchor_index, counts, quantiles, num_anchors,
    ):
        """Linear-interpolated quantiles for a flat ragged value set."""
        A = int(num_anchors)
        result = value.new_zeros((A, len(quantiles)))
        present = counts > 0
        if value.numel() == 0 or not bool(present.any()):
            return result
        order = torch.argsort(value, stable=True)
        order = order[
            torch.argsort(anchor_index[order], stable=True)
        ]
        sorted_value = value[order]
        starts = torch.cumsum(counts, dim=0) - counts
        rows = present.nonzero(as_tuple=True)[0]
        for column, quantile in enumerate(quantiles):
            position = (counts.to(value.dtype) - 1.0) * float(quantile)
            lower = torch.floor(position).to(torch.long).clamp_min(0)
            upper = torch.ceil(position).to(torch.long).clamp_min(0)
            lower_value = sorted_value[starts[rows] + lower[rows]]
            upper_value = sorted_value[starts[rows] + upper[rows]]
            weight = position[rows] - lower[rows].to(value.dtype)
            result[rows, column] = torch.lerp(
                lower_value, upper_value, weight
            )
        return result

    @torch.no_grad()
    def _normalize_loo_statistic(self, value):
        """Map a non-negative metre residual to [0,1] on a physical log scale."""
        clipped = value.clamp(0.0, self.stat_loo_cap_m)
        return torch.log1p(clipped / self.stat_loo_knee_m) / (
            self.stat_loo_log_denominator
        )

    @torch.no_grad()
    def _anchor_statistics(self, raw_points, raw_anchor, anchor_idx3, num_anchors):
        """Six requested channels, with float64 exact inverse-range LOO."""
        A = int(num_anchors)
        point = raw_points.detach().to(dtype=torch.float64)
        anchor = raw_anchor.to(dtype=torch.long)
        radius = torch.linalg.vector_norm(point, dim=-1)
        ray = point / radius[:, None].clamp_min(torch.finfo(torch.float64).tiny)
        inverse_range = radius.reciprocal()
        count = torch.bincount(anchor, minlength=A)

        range_sum = radius.new_zeros(A)
        range_sum.index_add_(0, anchor, radius)
        mean_range = range_sum / count.clamp_min(1).to(radius.dtype)

        gram = radius.new_zeros((A, 3, 3))
        cross = radius.new_zeros((A, 3))
        gram.index_add_(0, anchor, ray[:, :, None] * ray[:, None, :])
        cross.index_add_(0, anchor, ray * inverse_range[:, None])

        support = count >= self.stat_min_raw
        support_rows = support.nonzero(as_tuple=True)[0]
        solve_success = torch.zeros(A, dtype=torch.bool, device=point.device)
        inverse_gram = torch.zeros_like(gram)
        beta = radius.new_zeros((A, 3))
        if support_rows.numel() > 0:
            identity = torch.eye(
                3, dtype=torch.float64, device=point.device
            ).expand(support_rows.numel(), -1, -1)
            solution, info = torch.linalg.solve_ex(
                gram[support_rows], identity, check_errors=False
            )
            finite = torch.isfinite(solution).all(dim=(1, 2))
            good_local = (info == 0) & finite
            good_rows = support_rows[good_local]
            solve_success[good_rows] = True
            inverse_gram[good_rows] = solution[good_local]
            beta[good_rows] = torch.einsum(
                "nij,nj->ni", solution[good_local], cross[good_rows]
            )

        solve_point_rows = solve_success[anchor].nonzero(as_tuple=True)[0]
        valid_residual = radius.new_zeros(0)
        valid_residual_anchor = anchor.new_zeros(0)
        if solve_point_rows.numel() > 0:
            solve_anchor = anchor[solve_point_rows]
            solve_ray = ray[solve_point_rows]
            solve_inverse_range = inverse_range[solve_point_rows]
            inverse_times_ray = torch.einsum(
                "nij,nj->ni", inverse_gram[solve_anchor], solve_ray
            )
            full_prediction = (
                solve_ray * beta[solve_anchor]
            ).sum(dim=-1)
            leverage = (solve_ray * inverse_times_ray).sum(dim=-1)
            residual = solve_inverse_range - full_prediction
            denominator = 1.0 - leverage
            denominator_valid = (
                torch.isfinite(denominator)
                & torch.isfinite(residual)
                & (denominator >= self.stat_loo_eps)
            )
            if denominator_valid.any():
                member_anchor = solve_anchor[denominator_valid]
                loo_beta = (
                    beta[member_anchor]
                    - inverse_times_ray[denominator_valid]
                    * (
                        residual[denominator_valid]
                        / denominator[denominator_valid]
                    )[:, None]
                )
                predicted_inverse_range = (
                    solve_ray[denominator_valid] * loo_beta
                ).sum(dim=-1)
                predicted_range = predicted_inverse_range.reciprocal()
                loo_residual = torch.abs(
                    radius[solve_point_rows[denominator_valid]] - predicted_range
                )
                member_valid = (
                    torch.isfinite(loo_beta).all(dim=-1)
                    & torch.isfinite(predicted_inverse_range)
                    & (predicted_inverse_range > 0.0)
                    & torch.isfinite(loo_residual)
                )
                valid_residual = loo_residual[member_valid]
                valid_residual_anchor = member_anchor[member_valid]

        valid_loo_count = torch.bincount(
            valid_residual_anchor, minlength=A
        )
        # n>=6 is the policy criterion. solve success and >=3 usable residuals
        # are definition/safety guards: without them Q10/Q50/Q90 do not exist.
        valid = (
            support
            & solve_success
            & (valid_loo_count >= self.stat_min_valid_loo)
        )
        quantiles = self._segmented_quantiles(
            valid_residual,
            valid_residual_anchor,
            valid_loo_count,
            (0.10, 0.50, 0.90),
            A,
        )
        q10, q50, q90 = quantiles.unbind(dim=-1)

        capacity = self.ray_capacity[
            anchor_idx3[:, 0], anchor_idx3[:, 1]
        ].to(dtype=torch.float64)
        capacity_known = capacity > 0
        count_ratio_unclipped = (
            count.to(torch.float64) / capacity.clamp_min(1.0)
        )
        count_ratio = torch.where(
            capacity_known, count_ratio_unclipped.clamp(0.0, 1.0),
            torch.zeros_like(count_ratio_unclipped),
        )
        range_normalized = (
            (mean_range - self.stat_range_min_m)
            / (self.stat_range_max_m - self.stat_range_min_m)
        ).clamp(0.0, 1.0)
        zero = torch.zeros_like(q50)
        q50_normalized = self._normalize_loo_statistic(q50)
        upper_spread_normalized = self._normalize_loo_statistic(
            (q90 - q50).clamp_min(0.0)
        )
        lower_spread_normalized = self._normalize_loo_statistic(
            (q50 - q10).clamp_min(0.0)
        )
        statistic = torch.stack([
            count_ratio,
            range_normalized,
            torch.where(valid, q50_normalized, zero),
            torch.where(valid, upper_spread_normalized, zero),
            torch.where(valid, lower_spread_normalized, zero),
            valid.to(torch.float64),
        ], dim=-1).to(dtype=raw_points.dtype)
        diagnostics = {
            "anchor_raw_count": count,
            "anchor_mean_range": mean_range.to(dtype=raw_points.dtype),
            "anchor_ray_capacity": capacity.to(dtype=raw_points.dtype),
            "anchor_ray_capacity_known": capacity_known,
            "anchor_ray_fill_ratio_unclipped": count_ratio_unclipped.to(
                dtype=raw_points.dtype
            ),
            "stat_valid": valid,
            "valid_loo_count": valid_loo_count,
            "solve_success": solve_success,
            "loo_q10_m": q10.to(dtype=raw_points.dtype),
            "loo_q50_m": q50.to(dtype=raw_points.dtype),
            "loo_q90_m": q90.to(dtype=raw_points.dtype),
            "router_statistics": statistic,
        }
        return statistic, diagnostics

    def _forward_learned(
        self,
        feat,
        tok_pos,
        raw_point_sensor,
        raw_rows,
        raw_anchor,
        raw_token,
        tm,
        frame_anchor,
        anchor_base,
        anchor_out,
        anchor_label,
        anchor_box,
        anchor_is_dyn,
        anchor_frame,
        num_anchors,
    ):
        A = int(num_anchors)
        projected_member = self.anchor_pool_linear(feat[raw_token])
        pooled = feat.new_full((A, self.dim), float("-inf"))
        pooled.index_reduce_(
            0, raw_anchor, projected_member, "amax", include_self=True
        )
        if not bool(torch.isfinite(pooled).all()):
            raise RuntimeError("occupied spherical anchor has no token feature")
        anchor_feature = self.anchor_pool_act(self.anchor_pool_norm(pooled))

        neighbor_index, neighbor_valid = self._local_neighbor_table(
            frame_anchor, anchor_base, A
        )
        anchor_feature = self.local_attention(
            anchor_feature, anchor_out, neighbor_index, neighbor_valid
        )

        anchor_idx3 = torch.cat([
            fa["idx3"] for fa in frame_anchor if fa is not None
        ])
        statistic, statistic_meta = self._anchor_statistics(
            raw_point_sensor[raw_rows], raw_anchor, anchor_idx3, A
        )
        statistic_embedding = self.stat_encoder(
            statistic.to(dtype=anchor_feature.dtype)
        )
        router_feature = torch.cat(
            [anchor_feature, statistic_embedding], dim=-1
        )

        local_seed_rows = _range_quantile_seed_rows(
            raw_point_sensor[raw_rows], raw_anchor, A, self.k_max
        )
        k_row = torch.arange(
            1, self.k_max + 1, device=feat.device
        ).view(1, self.k_max, 1)
        slot = torch.arange(
            self.k_max, device=feat.device
        ).view(1, 1, self.k_max)
        seed_valid = (slot < k_row).expand(A, -1, -1)
        seed_anchor = torch.arange(A, device=feat.device)[:, None, None].expand(
            A, self.k_max, self.k_max
        )[seed_valid]
        seed_raw_row = raw_rows[local_seed_rows[seed_valid]]

        seed_ref = tok_pos.new_zeros((A, self.k_max, self.k_max, 3))
        seed_out = tok_pos.new_zeros((A, self.k_max, self.k_max, 3))
        flat_seed_ref = tm["raw_ref"][seed_raw_row]
        flat_seed_out = flat_seed_ref.clone()
        for global_f, fa in enumerate(frame_anchor):
            if fa is None or not bool(fa["is_dyn"].any()):
                continue
            rows = (
                (anchor_frame[seed_anchor] == global_f)
                & anchor_is_dyn[seed_anchor]
            ).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            boxes = tm["bbox_ref_by_frame"][global_f][
                anchor_box[seed_anchor[rows]]
            ]
            flat_seed_out[rows] = box_utils.points_to_box_local_batched(
                flat_seed_ref[rows], boxes
            )
        seed_ref[seed_valid] = flat_seed_ref
        seed_out[seed_valid] = flat_seed_out
        seed_delta = tok_pos.new_zeros(seed_out.shape)
        seed_delta[seed_valid] = (
            flat_seed_out - anchor_out[seed_anchor]
        ) / self.token_stride_m

        anchor_sensor = torch.cat([
            fa["sensor"] for fa in frame_anchor if fa is not None
        ])
        anchor_ref = torch.cat([
            fa["ref"] for fa in frame_anchor if fa is not None
        ])
        frame_anchor_count = torch.tensor(
            [0 if fa is None else int(fa["num_anchors"])
             for fa in frame_anchor],
            dtype=torch.long,
            device=feat.device,
        )
        anchor_offset = torch.cumsum(frame_anchor_count, dim=0)
        metadata = {
            "box_assign": torch.where(
                anchor_is_dyn, anchor_box, torch.full_like(anchor_box, -1)
            ),
            "instance_id": torch.where(
                anchor_is_dyn, anchor_label, torch.full_like(anchor_label, -1)
            ),
            "is_dynamic": anchor_is_dyn,
            "coord_ref": anchor_ref,
            "coord_sensor": anchor_sensor,
            "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
            "seed_ref": seed_ref,
            "router_feature": router_feature,
            "statistic_embedding": statistic_embedding,
            "anchor_idx3": anchor_idx3,
            "neighbor_count": neighbor_valid.sum(dim=-1),
            **statistic_meta,
        }
        return anchor_feature, seed_out, seed_delta, anchor_offset, metadata

    @staticmethod
    def _adaptive_rope_scale(num_anchors, kv_anchor, kv_pos, query_anchor,
                             query_pos):
        """Per-anchor metric RoPE scale = pi / (that anchor's realized support).

        Support = the largest axis-aligned extent of the anchor's own
        {K/V positions} U {query seeds}, which upper-bounds |delta_axis| for
        every (query, K/V) pair inside that anchor. Setting scale = pi / support
        therefore keeps the lowest RoPE band (inv_freq[0] == 1) strictly below a
        half turn -- no two displacements in the group can share a phase -- while
        spending the full usable range instead of a fraction of it.

        Purely geometric: no learned state, no gradient path (positions are
        detached bookkeeping), and clamped so a degenerate group cannot produce
        an extreme wavelength.
        """
        A = int(num_anchors)
        pos = torch.cat([kv_pos, query_pos], dim=0).detach().float()
        idx = torch.cat([kv_anchor, query_anchor], dim=0)
        hi = pos.new_full((A, 3), float("-inf")).index_reduce_(
            0, idx, pos, "amax", include_self=False)
        lo = pos.new_full((A, 3), float("inf")).index_reduce_(
            0, idx, pos, "amin", include_self=False)
        support = (hi - lo).max(dim=-1).values                      # (A,)
        support = torch.where(torch.isfinite(support), support,
                              torch.zeros_like(support))
        scale = (_SPHERICAL_ROPE_PHASE_MARGIN * torch.pi) / support.clamp_min(1e-3)
        return scale.clamp(_SPHERICAL_ROPE_SCALE_MIN, _SPHERICAL_ROPE_SCALE_MAX)

    # ------------------------------------------------------------------ forward
    def forward(self, feat, tok_pos, raw_point_sensor, raw_token_index,
                new_offset, frame_batch_idx, pose_list, bbox_list,
                bbox_instance_ids_list=None, token_ref=None,
                bbox_ref_by_frame=None):
        """Convert fused grid tokens + raw points into spherical Gaussian seeds.

        feat (N,D) fused token features; tok_pos (N,3) per-frame sensor coords
        (Utonia token positions); raw_point_sensor (R,3) own-frame raw points;
        raw_token_index (R,) maps each raw point to a row of feat/tok_pos;
        new_offset (n_frames,) global cumsum; frame_batch_idx (n_frames,);
        pose_list[b] (V,4,4) frame->frame0; bbox_list[b][local_f] (B_f,7).

        Returns (out_feat (G,D), seed (G,3), delta_p (G,3),
        gauss_offset (n_frames,) long, meta) with a variable G, ordered
        frame-major, anchor-major, source-token-major. seed lives in the output
        coordinate frame (bg: ref frame, fg: box-local); delta_p is the
        normalized seed-minus-source-token position in that same frame; meta
        follows the shared seed contract.
        """
        device = feat.device
        n_frames = len(frame_batch_idx)
        N = tok_pos.shape[0]

        if raw_point_sensor.shape[0] != raw_token_index.shape[0]:
            raise ValueError("raw_point_sensor and raw_token_index must have equal length")
        raw_point_sensor = raw_point_sensor.to(device=device, dtype=tok_pos.dtype)
        raw_token_index = raw_token_index.to(device=device, dtype=torch.long)
        if raw_token_index.numel() > 0 and (
            (raw_token_index < 0).any() or (raw_token_index >= N).any()
        ):
            raise ValueError("raw_token_index contains an out-of-range occupied-token row")

        tm = self._token_meta(tok_pos, raw_point_sensor, raw_token_index,
                              new_offset, frame_batch_idx,
                              pose_list, bbox_list, bbox_instance_ids_list,
                              token_ref=token_ref,
                              bbox_ref_by_frame=bbox_ref_by_frame)

        # ---- Stage B: per-frame anchors from raw-occupied spherical cells ----
        frame_anchor = [None] * n_frames
        anchor_base = [0] * n_frames
        for b, frame_indices in tm["batch_frame_map"].items():
            pose_b = pose_list[b]
            for local_f, global_f in enumerate(frame_indices):
                fa = self._frame_anchors(
                    tm, raw_point_sensor, pose_b[local_f].to(device), global_f
                )
                frame_anchor[global_f] = fa
        # Bases follow the same global-frame order used by every later concat,
        # even if a future collate function interleaves batch frame indices.
        A = 0
        for global_f, fa in enumerate(frame_anchor):
            anchor_base[global_f] = A
            A += 0 if fa is None else fa["num_anchors"]

        if A == 0:
            D = feat.shape[1]
            if self.count_mode == "learned_gumbel":
                zero_off = torch.zeros(
                    n_frames, dtype=torch.long, device=device
                )
                empty_long = torch.zeros(0, dtype=torch.long, device=device)
                empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
                empty_position = tok_pos.new_zeros(
                    0, self.k_max, self.k_max, 3
                )
                empty_meta = {
                    "box_assign": empty_long,
                    "instance_id": empty_long,
                    "is_dynamic": empty_bool,
                    "coord_ref": tok_pos.new_zeros(0, 3),
                    "coord_sensor": tok_pos.new_zeros(0, 3),
                    "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
                    "seed_ref": empty_position.clone(),
                    "router_feature": feat.new_zeros(0, self.router_dim),
                    "statistic_embedding": feat.new_zeros(
                        0, self.stat_embedding_dim
                    ),
                    "router_statistics": feat.new_zeros(0, 6),
                    "anchor_raw_count": empty_long,
                    "anchor_mean_range": tok_pos.new_zeros(0),
                    "anchor_ray_capacity": tok_pos.new_zeros(0),
                    "anchor_ray_capacity_known": empty_bool,
                    "anchor_ray_fill_ratio_unclipped": tok_pos.new_zeros(0),
                    "stat_valid": empty_bool,
                    "valid_loo_count": empty_long,
                    "solve_success": empty_bool,
                    "loo_q10_m": tok_pos.new_zeros(0),
                    "loo_q50_m": tok_pos.new_zeros(0),
                    "loo_q90_m": tok_pos.new_zeros(0),
                    "anchor_idx3": tok_pos.new_zeros(0, 3).long(),
                    "neighbor_count": empty_long,
                }
                return (
                    feat.new_zeros(0, D), empty_position,
                    empty_position.clone(), zero_off, empty_meta,
                )
            empty_meta = {
                "box_assign": torch.zeros(0, dtype=torch.long, device=device),
                "instance_id": torch.zeros(0, dtype=torch.long, device=device),
                "is_dynamic": torch.zeros(0, dtype=torch.bool, device=device),
                "coord_ref": tok_pos.new_zeros(0, 3),
                "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
                "slot_index": torch.zeros(0, dtype=torch.long, device=device),
                "anchor_index": torch.zeros(0, dtype=torch.long, device=device),
                "source_token_index": torch.zeros(0, dtype=torch.long, device=device),
                "anchor_raw_count": torch.zeros(0, dtype=torch.long, device=device),
                "evidence_raw_count": torch.zeros(0, dtype=torch.long, device=device),
            }
            zero_off = torch.zeros(n_frames, dtype=torch.long, device=device)
            return (
                feat.new_zeros(0, D), tok_pos.new_zeros(0, 3),
                tok_pos.new_zeros(0, 3), zero_off, empty_meta,
            )

        anchor_out = torch.cat([fa["out"] for fa in frame_anchor if fa is not None])
        anchor_label = torch.cat([fa["label"] for fa in frame_anchor if fa is not None])
        anchor_box = torch.cat([fa["box"] for fa in frame_anchor if fa is not None])
        anchor_is_dyn = anchor_label >= 0
        anchor_frame = torch.cat([
            torch.full((fa["num_anchors"],), g, dtype=torch.long, device=device)
            for g, fa in enumerate(frame_anchor) if fa is not None
        ])

        # ---- Stage B1: one evidence query per unique (anchor, source token) ----
        raw_rows = []       # retained (in-range) raw-point rows, frame-major
        raw_anchor = []     # matching global anchor ids
        for g, fa in enumerate(frame_anchor):
            if fa is None:
                continue
            raw_rows.append(fa["pi"])
            raw_anchor.append(anchor_base[g] + fa["pt2anchor"])
        raw_rows = torch.cat(raw_rows)
        raw_anchor = torch.cat(raw_anchor)
        raw_token = raw_token_index[raw_rows]

        if self.count_mode == "learned_gumbel":
            return self._forward_learned(
                feat,
                tok_pos,
                raw_point_sensor,
                raw_rows,
                raw_anchor,
                raw_token,
                tm,
                frame_anchor,
                anchor_base,
                anchor_out,
                anchor_label,
                anchor_box,
                anchor_is_dyn,
                anchor_frame,
                A,
            )

        (
            _seed_sensor, query_anchor, query_token, seed_local_row,
            anchor_raw_count, evidence_raw_count,
        ) = _select_evidence_seeds(
            raw_point_sensor[raw_rows], raw_anchor, raw_token, A, N,
        )
        seed_raw_row = raw_rows[seed_local_row]
        seed_ref = tm["raw_ref"][seed_raw_row]
        source_token_ref = tm["ref"][query_token]
        seed_out = seed_ref.clone()
        source_token_out = source_token_ref.clone()
        for global_f, fa in enumerate(frame_anchor):
            if fa is None or not fa["is_dyn"].any():
                continue
            bbox_ref_f = tm["bbox_ref_by_frame"][global_f]
            # All dynamic queries of this frame at once: each row gathers its own
            # anchor's box, one batched transform instead of a launch per box.
            rows = (
                (anchor_frame[query_anchor] == global_f)
                & anchor_is_dyn[query_anchor]
            ).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            boxes_row = bbox_ref_f[anchor_box[query_anchor[rows]]]
            seed_out[rows] = box_utils.points_to_box_local_batched(
                seed_ref[rows], boxes_row
            )
            source_token_out[rows] = box_utils.points_to_box_local_batched(
                source_token_ref[rows], boxes_row
            )

        # The head sees only local position within the 0.4m source-token support.
        # Absolute metric positions stay on the seed/base-centre and RoPE paths.
        delta_p = (seed_out - source_token_out) / self.token_stride_m

        # ---- Stage C: source-token feature is the query identity ----
        # Gather/indexing preserves gradient to feat[query_token]. No anchor P0,
        # raw-count feature, fixed slot embedding, or range metadata is injected.
        queries = self.p0_norm(feat[query_token])                    # (G, D)

        # ---- Stage D: K/V pairs, each carrying its own RoPE position ----
        # Every route stays within the anchor's own frame: cross-frame context
        # already lives in the features via the shared temporal aggregator.
        pair_anchor_parts, pair_token_parts, pair_pos_parts = [], [], []

        # bg anchors: the group's member tokens = exactly the anchor-embedding
        # set, dedup'd to one K/V entry per (anchor, token). Token positions
        # live in the ref frame.
        own_bg = ~anchor_is_dyn[raw_anchor]
        if own_bg.any():
            key = torch.unique(raw_anchor[own_bg] * N + raw_token[own_bg])
            own_anchor = torch.div(key, N, rounding_mode="floor")
            own_token = key - own_anchor * N
            pair_anchor_parts.append(own_anchor)
            pair_token_parts.append(own_token)
            pair_pos_parts.append(tm["ref"][own_token])

        # fg anchors: the tokens reached by the SAME instance's raw points in
        # the anchor's OWN frame (dedup'd per instance, instance-wide rather
        # than cell-member-only -- objects hold few tokens). Each token's RoPE
        # position is the own-frame instance-box-local position, the same
        # frame the anchor's query seeds live in. Tokens carry no labels here
        # -- the membership is raw-point-defined, so instance tokens survive
        # even when a token's centroid drifts outside the box.
        for global_f, fa in enumerate(frame_anchor):
            if fa is None or not bool(fa["is_dyn"].any()):
                continue
            fg_raw = (
                (tm["raw_frame"] == global_f) & (tm["raw_label"] >= 0)
            ).nonzero(as_tuple=True)[0]
            if fg_raw.numel() == 0:
                continue
            # (instance, token) membership, dedup'd across the frame's points.
            uniq_inst, inst_c = torch.unique(
                tm["raw_label"][fg_raw], return_inverse=True
            )
            pair_key = torch.unique(inst_c * N + raw_token_index[fg_raw])
            it_inst = torch.div(pair_key, N, rounding_mode="floor")
            it_tok = pair_key - it_inst * N
            # Box-local position w.r.t. this frame's box of the pair's
            # instance. A dynamic raw label is produced by indexing this
            # frame's own iids, so a lookup miss is impossible -- fail loudly
            # rather than route K/V through a bogus box. Positions stay in
            # the position dtype (fp32), never the feature dtype: under a
            # bf16 trunk the box-local write would crash, and bf16 positions
            # cost centimeters at range.
            iids_f = tm["iids_by_frame"][global_f]
            bbox_ref_f = tm["bbox_ref_by_frame"][global_f]
            eq = uniq_inst[it_inst].unsqueeze(1) == iids_f.unsqueeze(0)
            if not bool(eq.any(dim=1).all()):
                raise RuntimeError(
                    "dynamic raw label without a matching own-frame box; "
                    "raw labelling and bbox inputs are inconsistent"
                )
            bidx = eq.long().argmax(dim=1)
            # One batched transform: each (instance, token) pair gathers its box.
            it_pos = box_utils.points_to_box_local_batched(
                tm["ref"][it_tok], bbox_ref_f[bidx]
            )
            # per-instance CSR over tokens, then expand each fg anchor of this
            # frame to its instance block. Anchor labels are built from the
            # frame's own in-range raw points, a subset of fg_raw, so every
            # anchor label must resolve -- fail loudly otherwise.
            du = fa["is_dyn"].nonzero(as_tuple=True)[0]
            fg_a = anchor_base[global_f] + du
            fg_lab = fa["label"][du]
            counts = torch.bincount(it_inst, minlength=uniq_inst.numel())
            starts = torch.cumsum(counts, dim=0) - counts
            a_lc = torch.searchsorted(uniq_inst, fg_lab)
            a_lc = a_lc.clamp(max=uniq_inst.numel() - 1)
            if not bool((uniq_inst[a_lc] == fg_lab).all()):
                raise RuntimeError(
                    "fg anchor instance without own-frame raw membership; "
                    "anchor formation and raw labelling are inconsistent"
                )
            a_cnt = counts[a_lc]
            rep = torch.repeat_interleave(torch.arange(fg_a.numel(), device=device), a_cnt)
            blk = torch.cumsum(a_cnt, dim=0) - a_cnt
            within = torch.arange(int(a_cnt.sum().item()), device=device) - blk[rep]
            sel = starts[a_lc][rep] + within
            pair_anchor_parts.append(fg_a[rep])
            pair_token_parts.append(it_tok[sel])
            pair_pos_parts.append(it_pos[sel])

        if pair_anchor_parts:
            pair_anchor = torch.cat(pair_anchor_parts)
            pair_token = torch.cat(pair_token_parts)
            pair_pos = torch.cat(pair_pos_parts)
        else:
            pair_anchor = torch.zeros(0, dtype=torch.long, device=device)
            pair_token = torch.zeros(0, dtype=torch.long, device=device)
            pair_pos = tok_pos.new_zeros(0, 3)

        # ---- Stage E1: cap to max_kv nearest tokens per anchor ----
        # Distance is |K/V position - cell centre| in the output frame.
        # (two stable sorts: |delta| asc, then anchor asc -> per-anchor blocks
        # already distance-ordered; keep rank < max_kv. Result stays
        # anchor-ascending, as AnchorQueryCrossAttention requires.)
        if pair_anchor.numel() > 0:
            dist = (pair_pos - anchor_out[pair_anchor]).norm(dim=-1)
            o1 = torch.argsort(dist, stable=True)
            o2 = torch.argsort(pair_anchor[o1], stable=True)
            perm = o1[o2]
            pair_anchor = pair_anchor[perm]
            pair_token = pair_token[perm]
            pair_pos = pair_pos[perm]
            rank = _segment_rank(pair_anchor, A)
            keep = rank < self.max_kv
            pair_anchor = pair_anchor[keep]
            pair_token = pair_token[keep]
            pair_pos = pair_pos[keep]

        # ---- Stage E3: K/V (normalized fused features) + cross attention ----
        if pair_anchor.numel() > 0:
            kv_feat = self.kv_norm(feat[pair_token])
            kv_pos = pair_pos
        else:
            kv_feat = feat.new_zeros(0, self.dim)
            kv_pos = tok_pos.new_zeros(0, 3)
        anchor_rope_scale = torch.where(
            anchor_is_dyn,
            torch.full((A,), self.fg_rope_position_scale, device=device),
            torch.full((A,), self.bg_rope_position_scale, device=device),
        )
        adaptive = self._adaptive_rope_scale(
            A, pair_anchor, pair_pos, query_anchor, seed_out)
        use_adaptive = torch.where(
            anchor_is_dyn,
            torch.tensor(self.fg_rope_adaptive, device=device),
            torch.tensor(self.bg_rope_adaptive, device=device),
        )
        anchor_rope_scale = torch.where(use_adaptive, adaptive, anchor_rope_scale)
        out_feat = self.attn(
            queries, kv_feat, kv_pos, pair_anchor, A, seed_out,
            position_scale=anchor_rope_scale,
            query_anchor_ids=query_anchor,
        )
        # Cross-attention refines each evidence query from token memory. This
        # separate set block then exposes every refined query to the sibling
        # queries of the same (frame, spherical cell, bg/instance label) anchor.
        # It uses an ordinary residual block; no gate or LayerScale suppresses
        # the newly introduced query-query path.
        out_feat = self.query_self_attn(
            out_feat,
            seed_out,
            query_anchor,
            A,
            position_scale=anchor_rope_scale,
        )

        # ---- Stage E4: queries are already flat, frame/anchor/token-major ----
        query_rank = _segment_rank(query_anchor, A)
        p_flat = seed_out
        delta_flat = delta_p

        is_dyn_g = anchor_is_dyn[query_anchor]
        label_g = anchor_label[query_anchor]
        box_g = torch.where(
            is_dyn_g, anchor_box[query_anchor], torch.full_like(label_g, -1)
        )
        frame_g = anchor_frame[query_anchor]
        coord_ref = seed_ref

        counts_f = torch.bincount(frame_g, minlength=n_frames)
        gauss_offset = torch.cumsum(counts_f, dim=0)

        meta = {
            "box_assign": box_g,
            "instance_id": torch.where(is_dyn_g, label_g, torch.full_like(label_g, -1)),
            "is_dynamic": is_dyn_g,
            "coord_ref": coord_ref,
            "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
            # Retained name for downstream diagnostics; it is now the rank of
            # the source-token evidence inside its anchor, not a learned slot id.
            "slot_index": query_rank,
            "anchor_index": query_anchor,
            "source_token_index": query_token,
            "anchor_raw_count": anchor_raw_count[query_anchor],
            "evidence_raw_count": evidence_raw_count,
        }
        return out_feat, p_flat, delta_flat, gauss_offset, meta
