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

Each anchor emits up to three gaussians. Their coarse centres are
deterministic range-quantile raw points of the group; one raw point activates
the middle query, two activate the outer queries, and three or more activate
all three. The queries cross-attend to the token K/V set with their raw seed
positions as Q-side 3D RoPE coordinates. A bounded learned offset is applied
by the caller. The anchor reference position is the geometric spherical-cell
centre (shared by the split anchors of one cell), so both the head's delta_p
(seed - centre) and the max_kv capping distance are measured from the cell
centre in the output frame.

Anchor embedding: raw-count weighted sum of the group tokens' fused features
(a group whose 10 raw points map 3/6/1 onto three tokens pools 0.3/0.6/0.1 of
their features), followed by LayerNorm. Deliberately no per-token MLP: K/V
consumes the same fused features directly (below), so the query-side summary
stays in the same representation space; the count weights are data-fixed, and
adaptive per-query selection belongs to the cross-attention.

Tokens carry no fg/bg labels anywhere -- they are only ever reached through
raw-point membership:
- bg anchors live in the batch's ref frame (frame 0). K/V = the tokens of the
  group's own raw points (exactly the anchor-embedding set, dedup'd) + other
  frames' bg raw points ego-compensated into this frame's sensor coords,
  re-binned into the same cell, each contributing the token it is assigned to.
- fg anchors live in bbox-local coords. K/V = the tokens reached by the SAME
  instance's raw points across both frames (dedup'd); each token's RoPE
  position is its own frame's instance-box-local position, so the two frames'
  tokens concatenate in one motion-compensated frame (spherical reprojection
  breaks for movers).
- K/V features are the fused token features themselves (LayerNorm + the
  attention's W_k/W_v only, no extra embedding); K/V RoPE positions are the
  Utonia token positions (features['coord']-derived, not grid_coord).
- Both are capped to `max_kv` nearest tokens (|token - cell centre|).
- BG/FG share learned query/cross-attention weights, but use independent RoPE
  metric scales selected per anchor to respect their different spatial support.

Outputs are packed per active gaussian (anchor-major, slot fastest) and feed
the shared `gs_predictor` in m1_p2g.py together with the seed-minus-cell-centre
delta.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import boxes as box_utils
from ..utils.attention import AnchorQueryCrossAttention
from .spherical_bins import (
    SphericalBins,
    build_cells,
    lookup_cells,
)


# Fixed geometry constants, not experiment knobs.  The operator/frequency
# schedule matches Utonia Point3DRoPE; metric scales follow scale =
# pi / p95(max_axis |kv_pos - seed_pos|) measured over 100 train windows with
# the raw-cell label-group pair assembly
# (scripts/attention_rope_pair_stats_train_100_rawsplit: bg p95 1.772 m ->
# pi/p95 = 1.77 -> 1.75; fg p95 2.325 m -> 1.35 -- both land on the pre-split
# values, so the constants carry over measured, not assumed).
_SPHERICAL_ROPE_BASE = 10.0
_SPHERICAL_BG_ROPE_POSITION_SCALE = 1.75
_SPHERICAL_FG_ROPE_POSITION_SCALE = 1.35


def _segment_rank(sorted_ids, num_segments):
    """Ranks 0,1,2,... within each contiguous id block of an ascending id tensor."""
    counts = torch.bincount(sorted_ids, minlength=num_segments)
    starts = torch.zeros(num_segments, dtype=torch.long, device=sorted_ids.device)
    if num_segments > 1:
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
    return torch.arange(sorted_ids.numel(), device=sorted_ids.device) - starts[sorted_ids]


def _select_raw_seed_slots(raw_points, raw_anchor, anchor_sensor):
    """Select canonical 1/2/3 raw-point slots for every spherical anchor.

    Points are sorted deterministically by ``(anchor, range, x, y, z)``. For
    three or more points, slots use midpoint quantiles 1/6, 1/2, and 5/6.
    """
    A = anchor_sensor.shape[0]
    if raw_points.ndim != 2 or raw_points.shape[-1] != 3:
        raise ValueError(f"raw_points must have shape (R,3), got {tuple(raw_points.shape)}")
    if raw_anchor.ndim != 1 or raw_anchor.shape[0] != raw_points.shape[0]:
        raise ValueError("raw_anchor must provide one anchor index per raw point")

    counts = torch.bincount(raw_anchor, minlength=A)
    missing = counts == 0
    if missing.any():
        missing_ids = missing.nonzero(as_tuple=True)[0]
        raise RuntimeError(
            "occupied spherical anchors must have at least one own-frame raw point; "
            f"missing {int(missing_ids.numel())} / {A} anchors"
        )

    point_range = raw_points.norm(dim=-1)
    order = torch.arange(raw_points.shape[0], device=raw_points.device)
    # Stable least-to-most-significant sorting gives (anchor, r, x, y, z).
    for value in (
        raw_points[:, 2], raw_points[:, 1], raw_points[:, 0], point_range, raw_anchor,
    ):
        order = order[torch.argsort(value[order], stable=True)]
    sorted_points = raw_points[order]
    starts = torch.cumsum(counts, dim=0) - counts

    seeds = anchor_sensor[:, None, :].expand(A, 3, 3).clone()
    active = torch.zeros((A, 3), dtype=torch.bool, device=raw_points.device)

    one = counts == 1
    seeds[one, 1] = sorted_points[starts[one]]
    active[one, 1] = True

    two = counts == 2
    seeds[two, 0] = sorted_points[starts[two]]
    seeds[two, 2] = sorted_points[starts[two] + 1]
    active[two, 0] = True
    active[two, 2] = True

    many_ids = (counts >= 3).nonzero(as_tuple=True)[0]
    if many_ids.numel() > 0:
        slot = torch.arange(3, device=raw_points.device).view(1, 3)
        count_many = counts[many_ids].view(-1, 1)
        rank = torch.div(
            (2 * slot + 1) * count_many, 6, rounding_mode="floor"
        )
        rows = starts[many_ids].view(-1, 1) + rank
        seeds[many_ids] = sorted_points[rows]
        active[many_ids] = True

    return seeds, active, counts


class SphericalQueryHead(nn.Module):
    def __init__(self, squery_cfg, dim, r_far):
        super().__init__()
        self.dim = int(dim)
        self.K = int(squery_cfg.K)
        if self.K != 3:
            raise ValueError(
                "raw-seeded spherical queries require p2g.squery.K == 3 "
                f"for canonical middle/outer slots, got {self.K}"
            )
        self.max_kv = int(squery_cfg.max_kv)
        self.bins = SphericalBins(
            float(squery_cfg.dtheta_deg), float(squery_cfg.dphi_deg),
            float(squery_cfg.dlogr), float(squery_cfg.r_min), float(squery_cfg.r_max),
        )

        # K learnable queries, shared by every anchor; each anchor differentiates
        # them through its pooled anchor embedding (query_k = learnable_k + emb).
        self.query_embed = nn.Parameter(torch.randn(self.K, self.dim) * 0.02)

        # Anchor embedding: raw-count weighted token pooling -> LN. There is no
        # token MLP on purpose -- K/V uses the same un-embedded fused features,
        # so the query-side summary stays in the same space and the following
        # cross-attention owns all adaptive/query-specific selection.
        self.p0_norm = nn.LayerNorm(self.dim)

        # The fused token trunk has no final normalization. Normalize raw token
        # features once, then let the attention's W_k/W_v form K and V.
        self.kv_norm = nn.LayerNorm(self.dim)

        self.bg_rope_position_scale = _SPHERICAL_BG_ROPE_POSITION_SCALE
        self.fg_rope_position_scale = _SPHERICAL_FG_ROPE_POSITION_SCALE

        self.attn = AnchorQueryCrossAttention(
            self.dim, num_heads=8,
            n_layers=int(getattr(squery_cfg, "n_layers", 1) or 1),
            chunk=int(getattr(squery_cfg, "attn_chunk", 8192) or 8192),
            mlp_ratio=int(getattr(squery_cfg, "ffn_ratio", 4) or 4),
            rope_base=_SPHERICAL_ROPE_BASE,
            # Direct calls default to BG scale. Normal spherical forward passes
            # override it per anchor below without splitting learned weights.
            rope_position_scale=self.bg_rope_position_scale,
        )

    # ------------------------------------------------------------------ stages
    def _token_meta(self, tok_pos, raw_sensor, raw_token_index, new_offset,
                    frame_batch_idx, pose_list, bbox_list, bbox_iids_list):
        """Per-token AND per-raw-point metadata over ALL frames (computed once,
        carried everywhere). Tokens get only their frame and ref coordinates --
        they carry no fg/bg label and are reached exclusively through raw-point
        membership. Raw points get ref coordinates, bbox labels, and their
        global frame (they drive anchor formation and every K/V route)."""
        device = tok_pos.device
        N = tok_pos.shape[0]
        R = raw_sensor.shape[0]
        n_frames = len(frame_batch_idx)

        tok_frame = torch.zeros(N, dtype=torch.long, device=device)
        tok_ref = torch.zeros_like(tok_pos)

        raw_frame = torch.zeros(R, dtype=torch.long, device=device)
        raw_ref = torch.zeros_like(raw_sensor)
        raw_label = torch.full((R,), -1, dtype=torch.long, device=device)

        frame_slices = [None] * n_frames             # (start, end) per global frame
        bbox_ref_by_frame = [None] * n_frames
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
                tok_ref[sl] = box_utils.apply_pose(tok_pos[sl], pose_f)

                bbox_sensor_f = bbox_b[local_f].to(device)
                bbox_ref_f = box_utils.transform_boxes_to_ref(bbox_sensor_f, pose_f)
                bbox_ref_by_frame[global_f] = bbox_ref_f
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
        with the per-frame bg cell table (for cross-frame lookups) and the
        anchor arrays. A mixed boundary cell splits into one anchor per label
        (bg first, then instances in ascending id order) instead of voting, so
        every group is label-pure and no raw point is dropped. This equals
        binning the bg-only and per-instance point subsets separately. All
        split anchors of one cell share the geometric cell centre as their
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
        anchor_label = uniq_lab[group_key - anchor_cell * L]
        is_dyn = anchor_label >= 0

        # Anchor reference position = geometric spherical-cell centre; delta_p
        # and the max_kv capping distance are measured from it downstream.
        anchor_sensor = self.bins.cell_center_xyz(
            self.bins.unhash(anchor_cell)
        ).to(dtype=raw_sensor.dtype)
        anchor_ref = box_utils.apply_pose(anchor_sensor, pose_f)
        anchor_out = anchor_ref.clone()
        anchor_r = anchor_sensor.norm(dim=-1)

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
            for box_i in bidx.unique().tolist():
                box_i = int(box_i)
                sel = torch.zeros(U, dtype=torch.bool, device=device)
                sel[dyn_idx[bidx == box_i]] = True
                anchor_out[sel] = box_utils.points_to_box_local(anchor_ref[sel], bbox_ref_f[box_i])

        # bg cell table for cross-frame lookups: one bg anchor per cell at
        # most, and group keys ascend by (cell, label idx), so the bg subset
        # stays sorted and unique in cell hash.
        bg_rows = (~is_dyn).nonzero(as_tuple=True)[0]

        return {
            "bg_cell_hash": anchor_cell[bg_rows], "bg_rows": bg_rows,
            "pt2anchor": pt2anchor, "pi": pi, "num_anchors": U,
            "label": anchor_label, "box": anchor_box, "is_dyn": is_dyn,
            "sensor": anchor_sensor, "ref": anchor_ref, "out": anchor_out, "r": anchor_r,
        }

    # ------------------------------------------------------------------ forward
    def forward(self, feat, tok_pos, raw_point_sensor, raw_token_index,
                new_offset, frame_batch_idx, pose_list, bbox_list,
                bbox_instance_ids_list=None):
        """Convert fused grid tokens + raw points into spherical Gaussian seeds.

        feat (N,D) fused token features; tok_pos (N,3) per-frame sensor coords
        (Utonia token positions); raw_point_sensor (R,3) own-frame raw points;
        raw_token_index (R,) maps each raw point to a row of feat/tok_pos;
        new_offset (n_frames,) global cumsum; frame_batch_idx (n_frames,);
        pose_list[b] (V,4,4) frame->frame0; bbox_list[b][local_f] (B_f,7).

        Returns (out_feat (G,D), seed (G,3), delta_p (G,3), r_anchor (G,),
        gauss_offset (n_frames,) long, meta) with a variable G, ordered
        frame-major, anchor-major, active-slot fastest. seed lives in the
        output coordinate frame (bg: ref frame, fg: box-local); delta_p is the
        seed minus the anchor's spherical-cell centre in that same frame; meta
        follows the shared seed contract (box_assign / instance_id /
        is_dynamic / coord_ref / bbox_ref_by_frame).
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
                              pose_list, bbox_list, bbox_instance_ids_list)

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
            empty_meta = {
                "box_assign": torch.zeros(0, dtype=torch.long, device=device),
                "instance_id": torch.zeros(0, dtype=torch.long, device=device),
                "is_dynamic": torch.zeros(0, dtype=torch.bool, device=device),
                "coord_ref": tok_pos.new_zeros(0, 3),
                "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
            }
            zero_off = torch.zeros(n_frames, dtype=torch.long, device=device)
            return (
                feat.new_zeros(0, D), tok_pos.new_zeros(0, 3),
                tok_pos.new_zeros(0, 3), tok_pos.new_zeros(0), zero_off, empty_meta,
            )

        anchor_sensor = torch.cat([fa["sensor"] for fa in frame_anchor if fa is not None])
        anchor_out = torch.cat([fa["out"] for fa in frame_anchor if fa is not None])
        anchor_r = torch.cat([fa["r"] for fa in frame_anchor if fa is not None])
        anchor_label = torch.cat([fa["label"] for fa in frame_anchor if fa is not None])
        anchor_box = torch.cat([fa["box"] for fa in frame_anchor if fa is not None])
        anchor_is_dyn = anchor_label >= 0
        anchor_frame = torch.cat([
            torch.full((fa["num_anchors"],), g, dtype=torch.long, device=device)
            for g, fa in enumerate(frame_anchor) if fa is not None
        ])

        # ---- Stage B1: flat raw membership -> range-quantile seed slots ----
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

        seed_sensor, active_slot, anchor_raw_count = _select_raw_seed_slots(
            raw_point_sensor[raw_rows], raw_anchor, anchor_sensor,
        )

        # Seeds and cell centres are transformed with the same own-frame pose.
        # Dynamic anchors then use the same own-frame bbox-local coordinate
        # system as their K/V.
        seed_ref = torch.zeros_like(seed_sensor)
        for b, frame_indices in tm["batch_frame_map"].items():
            pose_b = pose_list[b]
            for local_f, global_f in enumerate(frame_indices):
                fa = frame_anchor[global_f]
                if fa is None:
                    continue
                start = anchor_base[global_f]
                end = start + fa["num_anchors"]
                pose_f = pose_b[local_f].to(device=device, dtype=seed_sensor.dtype)
                transformed = box_utils.apply_pose(
                    seed_sensor[start:end].reshape(-1, 3), pose_f
                )
                seed_ref[start:end] = transformed.reshape(-1, self.K, 3)

        seed_out = seed_ref.clone()
        for global_f, fa in enumerate(frame_anchor):
            if fa is None or not fa["is_dyn"].any():
                continue
            bbox_ref_f = tm["bbox_ref_by_frame"][global_f]
            for box_i in fa["box"][fa["is_dyn"]].unique().tolist():
                box_i = int(box_i)
                local_anchor = fa["is_dyn"] & (fa["box"] == box_i)
                rows = local_anchor.nonzero(as_tuple=True)[0] + anchor_base[global_f]
                local = box_utils.points_to_box_local(
                    seed_ref[rows].reshape(-1, 3), bbox_ref_f[box_i]
                )
                seed_out[rows] = local.reshape(-1, self.K, 3)
        # Precomputed head conditioning: seed minus the spherical cell centre,
        # both expressed in the anchor's output frame (bg: ref, fg: box-local).
        delta_p = seed_out - anchor_out[:, None, :]

        # ---- Stage C: anchor embedding = raw-count weighted token pooling ----
        # index_add over raw points realises the count weighting exactly: a
        # token holding k of the cell's n raw points contributes k/n of its
        # feature. No MLP (see module docstring); LayerNorm keeps the pooled
        # scale stable before the learnable queries are added.
        emb = feat.new_zeros(A, self.dim)
        emb.index_add_(0, raw_anchor, feat[raw_token])
        emb = emb / anchor_raw_count.to(feat.dtype).clamp_min(1.0).unsqueeze(-1)
        emb = self.p0_norm(emb)
        queries = self.query_embed.unsqueeze(0) + emb.unsqueeze(1)   # (A, K, D)

        # ---- Stage D: K/V pairs, each carrying its own RoPE position ----
        pair_anchor_parts, pair_token_parts, pair_pos_parts = [], [], []

        # bg anchors, own frame: the group's member tokens = exactly the
        # anchor-embedding set, dedup'd to one K/V entry per (anchor, token).
        # Token positions live in the ref frame.
        own_bg = ~anchor_is_dyn[raw_anchor]
        if own_bg.any():
            key = torch.unique(raw_anchor[own_bg] * N + raw_token[own_bg])
            own_anchor = torch.div(key, N, rounding_mode="floor")
            own_token = key - own_anchor * N
            pair_anchor_parts.append(own_anchor)
            pair_token_parts.append(own_token)
            pair_pos_parts.append(tm["ref"][own_token])

        # bg anchors, other frames: bg-labelled raw points are ego-compensated
        # into this frame's sensor coords, re-binned into the same 1x1x1 cell,
        # and contribute the token they are assigned to (dedup'd). fg raw
        # points are excluded: without motion compensation their re-binned
        # location is stale.
        for b, frame_indices in tm["batch_frame_map"].items():
            pose_b = pose_list[b]
            for local_f, global_f in enumerate(frame_indices):
                fa = frame_anchor[global_f]
                if fa is None or fa["bg_rows"].numel() == 0:
                    continue
                pose_f = pose_b[local_f].to(device)
                inv_pose_f = torch.linalg.inv(pose_f)
                for global_g in frame_indices:
                    if global_g == global_f:
                        continue
                    cand = (
                        (tm["raw_frame"] == global_g) & (tm["raw_label"] == -1)
                    ).nonzero(as_tuple=True)[0]
                    if cand.numel() == 0:
                        continue
                    x_f = box_utils.apply_pose(tm["raw_ref"][cand], inv_pose_f)
                    idx3_2, valid_2 = self.bins.bin_coords(x_f)
                    cand = cand[valid_2]
                    if cand.numel() == 0:
                        continue
                    cell_idx2, found2 = lookup_cells(
                        self.bins.hash(idx3_2[valid_2]), fa["bg_cell_hash"]
                    )
                    cand = cand[found2]
                    if cand.numel() == 0:
                        continue
                    a_ids = anchor_base[global_f] + fa["bg_rows"][cell_idx2[found2]]
                    key = torch.unique(a_ids * N + raw_token_index[cand])
                    a_u = torch.div(key, N, rounding_mode="floor")
                    t_u = key - a_u * N
                    pair_anchor_parts.append(a_u)
                    pair_token_parts.append(t_u)
                    pair_pos_parts.append(tm["ref"][t_u])

        # fg anchors: the tokens reached by the SAME instance's raw points
        # across both frames (dedup'd per instance), each expressed in its own
        # frame's instance-box-local coordinates so the two frames concatenate
        # in one motion-compensated frame. Tokens carry no labels here -- the
        # membership is raw-point-defined, so instance tokens survive even
        # when a token's centroid drifts outside the box.
        for b, frame_indices in tm["batch_frame_map"].items():
            fg_a, fg_lab = [], []
            for global_f in frame_indices:
                fa = frame_anchor[global_f]
                if fa is None:
                    continue
                du = fa["is_dyn"].nonzero(as_tuple=True)[0]
                fg_a.append(anchor_base[global_f] + du)
                fg_lab.append(fa["label"][du])
            if not fg_a:
                continue
            fg_a = torch.cat(fg_a)
            fg_lab = torch.cat(fg_lab)
            if fg_a.numel() == 0:
                continue
            frames_b = torch.zeros_like(tm["raw_frame"], dtype=torch.bool)
            for global_f in frame_indices:
                frames_b |= tm["raw_frame"] == global_f
            fg_raw = (frames_b & (tm["raw_label"] >= 0)).nonzero(as_tuple=True)[0]
            if fg_raw.numel() == 0:
                continue
            # (instance, token) membership, dedup'd across every raw point.
            uniq_inst, inst_c = torch.unique(
                tm["raw_label"][fg_raw], return_inverse=True
            )
            pair_key = torch.unique(inst_c * N + raw_token_index[fg_raw])
            it_inst = torch.div(pair_key, N, rounding_mode="floor")
            it_tok = pair_key - it_inst * N
            # Box-local position of each token w.r.t. its OWN frame's box of
            # that instance (a dynamic instance has a box in both endpoint
            # frames by definition; drop the pair defensively otherwise).
            # Positions stay in the position dtype (fp32), never the feature
            # dtype: under a bf16 trunk the box-local write would crash, and
            # bf16 positions cost centimeters at range.
            it_pos = tm["ref"].new_zeros(it_tok.shape[0], 3)
            it_valid = torch.zeros(it_tok.shape[0], dtype=torch.bool, device=device)
            tok_frame_it = tm["frame"][it_tok]
            for global_f in frame_indices:
                m_rows = (tok_frame_it == global_f).nonzero(as_tuple=True)[0]
                iids_f = tm["iids_by_frame"][global_f]
                if m_rows.numel() == 0 or iids_f.numel() == 0:
                    continue
                bbox_ref_f = tm["bbox_ref_by_frame"][global_f]
                eq = uniq_inst[it_inst[m_rows]].unsqueeze(1) == iids_f.unsqueeze(0)
                has = eq.any(dim=1)
                bidx = eq.long().argmax(dim=1)
                for box_i in bidx[has].unique().tolist():
                    box_i = int(box_i)
                    sel = m_rows[has & (bidx == box_i)]
                    it_pos[sel] = box_utils.points_to_box_local(
                        tm["ref"][it_tok[sel]], bbox_ref_f[box_i]
                    )
                it_valid[m_rows[has]] = True
            keep = it_valid.nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            it_inst = it_inst[keep]
            it_tok = it_tok[keep]
            it_pos = it_pos[keep]
            # per-instance CSR over tokens, then expand each anchor to its block
            counts = torch.bincount(it_inst, minlength=uniq_inst.numel())
            starts = torch.zeros(uniq_inst.numel(), dtype=torch.long, device=device)
            if uniq_inst.numel() > 1:
                starts[1:] = torch.cumsum(counts, dim=0)[:-1]
            a_lc = torch.searchsorted(uniq_inst, fg_lab)
            a_lc_c = a_lc.clamp(max=uniq_inst.numel() - 1)
            matched = (a_lc < uniq_inst.numel()) & (uniq_inst[a_lc_c] == fg_lab)
            fg_a = fg_a[matched]
            a_lc = a_lc_c[matched]
            if fg_a.numel() == 0:
                continue
            a_cnt = counts[a_lc]
            rep = torch.repeat_interleave(torch.arange(fg_a.numel(), device=device), a_cnt)
            blk = torch.zeros(fg_a.numel(), dtype=torch.long, device=device)
            if fg_a.numel() > 1:
                blk[1:] = torch.cumsum(a_cnt, dim=0)[:-1]
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
        out, _weighted_kv_pos = self.attn(
            queries, kv_feat, kv_pos, pair_anchor, A, seed_out,
            position_scale=anchor_rope_scale,
        )

        # ---- Stage E4: pack only active slots (anchor-major, slot fastest) ----
        anchor_index, slot_index = active_slot.nonzero(as_tuple=True)
        out_feat = out[anchor_index, slot_index]
        p_flat = seed_out[anchor_index, slot_index]
        delta_flat = delta_p[anchor_index, slot_index]
        r_g = anchor_r[anchor_index]

        is_dyn_g = anchor_is_dyn[anchor_index]
        label_g = anchor_label[anchor_index]
        box_g = torch.where(
            is_dyn_g, anchor_box[anchor_index], torch.full_like(label_g, -1)
        )
        frame_g = anchor_frame[anchor_index]
        coord_ref = seed_ref[anchor_index, slot_index]

        counts_f = torch.bincount(frame_g, minlength=n_frames)
        gauss_offset = torch.cumsum(counts_f, dim=0)

        meta = {
            "box_assign": box_g,
            "instance_id": torch.where(is_dyn_g, label_g, torch.full_like(label_g, -1)),
            "is_dynamic": is_dyn_g,
            "coord_ref": coord_ref,
            "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
            "slot_index": slot_index,
            "anchor_raw_count": anchor_raw_count[anchor_index],
        }
        return out_feat, p_flat, delta_flat, r_g, gauss_offset, meta
