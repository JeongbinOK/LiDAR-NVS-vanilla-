"""Spherical query-anchor head (anchor_mode="spherical").

For spherical mode, the fused Utonia tokens
(0.4 m cells, sensor-frame positions) are binned into (theta, phi, log r)
spherical cells about that frame's sensor origin; every occupied cell becomes an
anchor. Near-range anchors tend to hold few tokens while far-range anchors can
pool several tokens. Each anchor then predicts K gaussians: K learnable queries
(+ an anchor embedding pooled from the anchor's own-frame cell tokens)
cross-attend to the anchor's K/V token set, and each query's initial position is
the attention-probability-weighted mean of the K/V token positions (surface
interpolation seed; a bounded offset head on top is applied by the caller).

Background/foreground coordinate routing:
- bg anchors/tokens live in the batch's ref frame (frame 0). P0 uses only the
  anchor's own-frame cell tokens. K/V = P1 bg context: same-frame 1x1x1
  spherical neighbourhood + other-frame bg tokens ego-compensated into this
  frame's sensor coords and re-binned into the same 1x1x1 cell.
- fg (dynamic, instance shared by both endpoint frames) anchors/tokens live in
  bbox-local coords. K/V = ALL tokens of the same instance across frames
  (spherical reprojection breaks for movers; box-local pooling is
  motion-compensated by construction).
- Both are capped to `max_kv` nearest-|delta_p| tokens per anchor.

Anchor cell labels are decided by majority vote over the cell's token labels
(ties -> bg), and the anchor position is the mean of the *label-matching*
tokens, so a cell straddling a bbox boundary stays stable.

Outputs are per-gaussian (anchor-major, k fastest) and feed the shared
`gs_predictor` in m1_p2g.py and the shared gaussian assembly module.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import boxes as box_utils
from ..utils.attention import AnchorQueryCrossAttention
from .builders.common import encode_ray_meta, xyz_to_theta_phi_r
from .spherical_bins import (
    SphericalBins,
    build_cells,
    gather_cell_members,
    lookup_cells,
)


def _segment_rank(sorted_ids, num_segments):
    """Ranks 0,1,2,... within each contiguous id block of an ascending id tensor."""
    counts = torch.bincount(sorted_ids, minlength=num_segments)
    starts = torch.zeros(num_segments, dtype=torch.long, device=sorted_ids.device)
    if num_segments > 1:
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
    return torch.arange(sorted_ids.numel(), device=sorted_ids.device) - starts[sorted_ids]


class SphericalQueryHead(nn.Module):
    def __init__(self, squery_cfg, dim, r_far):
        super().__init__()
        self.dim = int(dim)
        self.K = int(squery_cfg.K)
        self.max_kv = int(squery_cfg.max_kv)
        self.r_far = float(r_far)
        self.bins = SphericalBins(
            float(squery_cfg.dtheta_deg), float(squery_cfg.dphi_deg),
            float(squery_cfg.dlogr), float(squery_cfg.r_min), float(squery_cfg.r_max),
        )

        # K learnable queries, shared by every anchor; each anchor differentiates
        # them through its pooled anchor embedding (query_k = learnable_k + emb).
        self.query_embed = nn.Parameter(torch.randn(self.K, self.dim) * 0.02)

        # P0 anchor embedding: proj([token feat, delta_p]) -> mean pool -> LN.
        self.p0_proj = nn.Linear(self.dim + 3, self.dim)
        self.p0_norm = nn.LayerNorm(self.dim)

        # K/V input embedding: [feat(D), delta_p(3), dframe(1), ray_meta(4)] -> D.
        self.kv_embed = nn.Linear(self.dim + 8, self.dim)
        self.kv_norm = nn.LayerNorm(self.dim)

        self.attn = AnchorQueryCrossAttention(
            self.dim, num_heads=8,
            n_layers=int(getattr(squery_cfg, "n_layers", 1) or 1),
            chunk=int(getattr(squery_cfg, "attn_chunk", 8192) or 8192),
            mlp_ratio=int(getattr(squery_cfg, "ffn_ratio", 4) or 4),
        )

    # ------------------------------------------------------------------ stages
    def _token_meta(self, tok_pos, new_offset, frame_batch_idx,
                    pose_list, bbox_list, bbox_iids_list):
        """Per-token metadata over ALL frames (spec item 7: computed once,
        carried everywhere). Returns a dict of (N,*) tensors + per-frame lists."""
        device = tok_pos.device
        N = tok_pos.shape[0]
        n_frames = len(frame_batch_idx)

        tok_frame = torch.zeros(N, dtype=torch.long, device=device)
        tok_ref = torch.zeros_like(tok_pos)
        tok_out = torch.zeros_like(tok_pos)          # bg: ref frame, fg: box-local
        tok_label = torch.full((N,), -1, dtype=torch.long, device=device)
        tok_box = torch.full((N,), -1, dtype=torch.long, device=device)  # frame-local box idx (dynamic only)
        tpr_own = xyz_to_theta_phi_r(tok_pos)        # own-sensor-origin ray (theta, phi, r)
        tok_ray4 = encode_ray_meta(tpr_own, self.r_far)
        idx3, tok_valid = self.bins.bin_coords(tok_pos)

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

                pos_f = tok_pos[sl]
                pose_f = pose_b[local_f].to(device)
                ref_f = box_utils.apply_pose(pos_f, pose_f)
                tok_frame[sl] = global_f
                tok_ref[sl] = ref_f
                tok_out[sl] = ref_f

                bbox_sensor_f = bbox_b[local_f].to(device)
                bbox_ref_f = box_utils.transform_boxes_to_ref(bbox_sensor_f, pose_f)
                bbox_ref_by_frame[global_f] = bbox_ref_f
                if iids_b is not None:
                    iids_f = iids_b[local_f].to(device)
                else:
                    iids_f = torch.arange(bbox_sensor_f.shape[0], device=device, dtype=torch.long)
                iids_by_frame[global_f] = iids_f

                box_assign_f = box_utils.point_in_box(pos_f, bbox_sensor_f)
                instance_f = torch.full_like(box_assign_f, -1)
                in_box = box_assign_f >= 0
                if in_box.any() and iids_f.numel() > 0:
                    instance_f[in_box] = iids_f[box_assign_f[in_box]]
                is_dyn_f = torch.zeros_like(in_box)
                for inst_id in common_ids:
                    is_dyn_f |= instance_f == int(inst_id)

                # Dynamic tokens use box-local output coordinates.
                label_f = torch.where(is_dyn_f, instance_f, torch.full_like(instance_f, -1))
                tok_label[sl] = label_f
                tok_box[sl] = torch.where(is_dyn_f, box_assign_f,
                                          torch.full_like(box_assign_f, -1))
                for box_i in box_assign_f[is_dyn_f].unique().tolist():
                    box_i = int(box_i)
                    m = is_dyn_f & (box_assign_f == box_i)
                    if m.any():
                        out_sl = tok_out[sl]
                        out_sl[m] = box_utils.points_to_box_local(ref_f[m], bbox_ref_f[box_i])
                        tok_out[sl] = out_sl

        return {
            "frame": tok_frame, "ref": tok_ref, "out": tok_out, "label": tok_label,
            "box": tok_box, "ray4": tok_ray4, "idx3": idx3, "valid": tok_valid,
            "frame_slices": frame_slices, "bbox_ref_by_frame": bbox_ref_by_frame,
            "iids_by_frame": iids_by_frame, "batch_frame_map": batch_frame_map,
        }

    def _frame_anchors(self, tm, tok_pos, pose_f, global_f):
        """Anchors of one frame: occupied spherical cells over its valid tokens.

        Returns None when the frame has no valid token, else a dict with the
        per-frame cell table (for P1/P2 lookups) and the anchor arrays. Anchor
        label = majority vote of the cell's token labels (tie -> bg); anchor
        position = mean of the label-matching tokens (>=1 by construction)."""
        device = tok_pos.device
        start, end = tm["frame_slices"][global_f]
        g_all = torch.arange(start, end, device=device)
        vmask = tm["valid"][start:end]
        vi = g_all[vmask]                                   # global token ids, valid only
        if vi.numel() == 0:
            return None

        hashes = self.bins.hash(tm["idx3"][vi])
        cell_hash, tok2cell = build_cells(hashes)
        U = cell_hash.numel()
        labels_v = tm["label"][vi]

        # majority label per cell (bg wins ties): compact labels, count per
        # (cell, label), then take the best-scored label of each cell via a
        # single composite sort.
        uniq_lab, lab_c = torch.unique(labels_v, return_inverse=True)
        L = uniq_lab.numel()
        key = tok2cell * L + lab_c
        uk, kcounts = torch.unique(key, return_counts=True)
        kcell = torch.div(uk, L, rounding_mode="floor")
        klab = uk - kcell * L
        score = kcounts * 2 + (uniq_lab[klab] == -1).long()   # bg tie-break
        smax = int(score.max().item()) + 1
        order = torch.argsort(kcell * smax + (smax - 1 - score))
        kcell_s = kcell[order]
        first = torch.ones_like(kcell_s, dtype=torch.bool)
        first[1:] = kcell_s[1:] != kcell_s[:-1]
        anchor_label = uniq_lab[klab[order][first]]           # (U,) aligned to cell 0..U-1

        # anchor position = mean over label-matching cell tokens (sensor frame)
        match = labels_v == anchor_label[tok2cell]            # (Nv,)
        cnt = torch.bincount(tok2cell[match], minlength=U).to(tok_pos.dtype)
        sum_pos = torch.zeros(U, 3, device=device, dtype=tok_pos.dtype)
        sum_pos.index_add_(0, tok2cell[match], tok_pos[vi[match]])
        anchor_sensor = sum_pos / cnt.clamp_min(1.0).unsqueeze(-1)

        anchor_ref = box_utils.apply_pose(anchor_sensor, pose_f)
        anchor_out = anchor_ref.clone()
        anchor_r = anchor_sensor.norm(dim=-1)

        # frame-local box index of the majority instance (fg anchors only).
        # Defensive: a dynamic label always comes from this frame's own boxes, so
        # a lookup miss should be impossible -- if it ever happens, demote the
        # anchor to bg instead of indexing with a bogus box id downstream.
        anchor_box = torch.full((U,), -1, dtype=torch.long, device=device)
        is_dyn = anchor_label >= 0
        iids_f = tm["iids_by_frame"][global_f]
        bbox_ref_f = tm["bbox_ref_by_frame"][global_f]
        if is_dyn.any() and iids_f.numel() > 0:
            dyn_idx = is_dyn.nonzero(as_tuple=True)[0]
            eq = anchor_label[dyn_idx].unsqueeze(1) == iids_f.unsqueeze(0)  # (Ud, B_f)
            has = eq.any(dim=1)
            bidx = torch.where(has, eq.long().argmax(dim=1), torch.full_like(has.long(), -1))
            anchor_box[dyn_idx] = bidx
            if (~has).any():
                anchor_label[dyn_idx[~has]] = -1
                anchor_box[dyn_idx[~has]] = -1
                is_dyn = anchor_label >= 0
            for box_i in bidx[has].unique().tolist():
                box_i = int(box_i)
                sel = torch.zeros(U, dtype=torch.bool, device=device)
                sel[dyn_idx[bidx == box_i]] = True
                anchor_out[sel] = box_utils.points_to_box_local(anchor_ref[sel], bbox_ref_f[box_i])
        elif is_dyn.any():
            # dynamic labels but no boxes in this frame: cannot happen by
            # construction; demote defensively.
            anchor_label[is_dyn] = -1
            is_dyn = anchor_label >= 0

        return {
            "cell_hash": cell_hash, "tok2cell": tok2cell, "vi": vi, "match": match,
            "label": anchor_label, "box": anchor_box, "is_dyn": is_dyn,
            "sensor": anchor_sensor, "ref": anchor_ref, "out": anchor_out, "r": anchor_r,
        }

    # ------------------------------------------------------------------ forward
    def forward(self, feat, tok_pos, new_offset, frame_batch_idx, pose_list,
                bbox_list, bbox_instance_ids_list=None):
        """Convert fused grid tokens into spherical Gaussian query seeds.

        feat (N,D) fused token features; tok_pos (N,3) per-frame sensor coords;
        new_offset (n_frames,) global cumsum; frame_batch_idx (n_frames,);
        pose_list[b] (V,4,4) frame->frame0; bbox_list[b][local_f] (B_f,7).

        Returns (out_feat (G,D), p_init (G,3), r_anchor (G,), gauss_offset
        (n_frames,) long, meta) with G = total_anchors * K, ordered frame-major,
        anchor-major, k fastest. p_init lives in the output coordinate frame
        (bg: ref frame, fg: box-local); meta follows the shared seed contract
        (box_assign / instance_id / is_dynamic / coord_ref / bbox_ref_by_frame).
        """
        device = feat.device
        n_frames = len(frame_batch_idx)
        tm = self._token_meta(tok_pos, new_offset, frame_batch_idx,
                              pose_list, bbox_list, bbox_instance_ids_list)

        # ---- Stage B: per-frame anchors (global frame order = concat order) ----
        frame_anchor = [None] * n_frames
        anchor_base = [0] * n_frames
        A = 0
        for b, frame_indices in tm["batch_frame_map"].items():
            pose_b = pose_list[b]
            for local_f, global_f in enumerate(frame_indices):
                fa = self._frame_anchors(tm, tok_pos, pose_b[local_f].to(device), global_f)
                frame_anchor[global_f] = fa
                anchor_base[global_f] = A
                A += 0 if fa is None else fa["cell_hash"].numel()

        if A == 0:
            D = feat.shape[1]
            empty_meta = {
                "box_assign": torch.zeros(0, dtype=torch.long, device=device),
                "instance_id": torch.zeros(0, dtype=torch.long, device=device),
                "is_dynamic": torch.zeros(0, dtype=torch.bool, device=device),
                "coord_ref": feat.new_zeros(0, 3),
                "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
            }
            zero_off = torch.zeros(n_frames, dtype=torch.long, device=device)
            return (feat.new_zeros(0, D), feat.new_zeros(0, 3),
                    feat.new_zeros(0), zero_off, empty_meta)

        anchor_out = torch.cat([fa["out"] for fa in frame_anchor if fa is not None])
        anchor_ref = torch.cat([fa["ref"] for fa in frame_anchor if fa is not None])
        anchor_r = torch.cat([fa["r"] for fa in frame_anchor if fa is not None])
        anchor_label = torch.cat([fa["label"] for fa in frame_anchor if fa is not None])
        anchor_box = torch.cat([fa["box"] for fa in frame_anchor if fa is not None])
        anchor_is_dyn = anchor_label >= 0
        anchor_frame = torch.cat([
            torch.full((fa["cell_hash"].numel(),), g, dtype=torch.long, device=device)
            for g, fa in enumerate(frame_anchor) if fa is not None
        ])

        # ---- Stage B2: P0-own pairs (label-matching cell tokens, every anchor) ----
        p0_anchor = []
        p0_token = []
        for g, fa in enumerate(frame_anchor):
            if fa is None:
                continue
            p0_anchor.append(anchor_base[g] + fa["tok2cell"][fa["match"]])
            p0_token.append(fa["vi"][fa["match"]])

        # ---- Stage C: bg P1 K/V pairs (same spherical cell only) ----
        p1_anchor = []
        p1_token = []
        for b, frame_indices in tm["batch_frame_map"].items():
            pose_b = pose_list[b]
            for local_f, global_f in enumerate(frame_indices):
                fa = frame_anchor[global_f]
                if fa is None:
                    continue
                bg_u = (~fa["is_dyn"]).nonzero(as_tuple=True)[0]     # frame-local anchor ids
                if bg_u.numel() == 0:
                    continue
                # Background attention is intentionally restricted to the
                # anchor's own (theta, phi, log-r) cell. Other-frame tokens are
                # ego-compensated and re-binned below, then must match this same
                # hash exactly. Foreground uses its separate same-instance path.
                nbr_hash = fa["cell_hash"][bg_u].unsqueeze(1)  # (U_bg, 1)
                nbr_valid = torch.ones_like(nbr_hash, dtype=torch.bool)
                n_slots = 1

                # P1: same-frame tokens via the frame's own cell table
                cell_idx, found = lookup_cells(nbr_hash.reshape(-1), fa["cell_hash"])
                acell = torch.where(found & nbr_valid.reshape(-1), cell_idx,
                                    torch.full_like(cell_idx, -1)).reshape(-1, n_slots)
                a_ids, t_ids, _ = gather_cell_members(
                    acell, fa["tok2cell"], fa["cell_hash"].numel())
                tok_g = fa["vi"][t_ids]
                keep = tm["label"][tok_g] == -1                       # bg tokens only
                p1_anchor.append(anchor_base[global_f] + bg_u[a_ids[keep]])
                p1_token.append(tok_g[keep])

                # Other frames' bg tokens are ego-compensated into this frame's
                # sensor coords, re-binned, and used only as K/V context.
                pose_f = pose_b[local_f].to(device)
                inv_pose_f = torch.linalg.inv(pose_f)
                for local_g, global_g in enumerate(frame_indices):
                    if global_g == global_f:
                        continue
                    fo = frame_anchor[global_g]
                    if fo is None:
                        continue
                    cand = fo["vi"][tm["label"][fo["vi"]] == -1]      # other-frame bg tokens
                    if cand.numel() == 0:
                        continue
                    x_f = box_utils.apply_pose(tm["ref"][cand], inv_pose_f)
                    idx3_2, valid_2 = self.bins.bin_coords(x_f)
                    cand = cand[valid_2]
                    if cand.numel() == 0:
                        continue
                    cell_hash_2, tok2cell_2 = build_cells(self.bins.hash(idx3_2[valid_2]))
                    cell_idx2, found2 = lookup_cells(nbr_hash.reshape(-1), cell_hash_2)
                    acell2 = torch.where(found2 & nbr_valid.reshape(-1), cell_idx2,
                                         torch.full_like(cell_idx2, -1)).reshape(-1, n_slots)
                    a2, t2, _ = gather_cell_members(
                        acell2, tok2cell_2, cell_hash_2.numel())
                    p1_anchor.append(anchor_base[global_f] + bg_u[a2])
                    p1_token.append(cand[t2])

        # ---- Stage D: fg pairs = all same-(batch, instance) tokens, both frames ----
        fg_anchor = []
        fg_token = []
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
            tok_ids = []
            for global_f in frame_indices:
                fa = frame_anchor[global_f]
                if fa is None:
                    continue
                vi = fa["vi"]
                tok_ids.append(vi[tm["label"][vi] >= 0])
            tok_ids = torch.cat(tok_ids) if tok_ids else fg_a.new_zeros(0)
            if tok_ids.numel() == 0:
                continue
            tok_lab = tm["label"][tok_ids]
            # per-instance CSR over tokens, then expand each anchor to its block
            uniq, tok_lc = torch.unique(tok_lab, return_inverse=True)
            order = torch.argsort(tok_lc)
            tok_sorted = tok_ids[order]
            counts = torch.bincount(tok_lc, minlength=uniq.numel())
            starts = torch.zeros(uniq.numel(), dtype=torch.long, device=device)
            if uniq.numel() > 1:
                starts[1:] = torch.cumsum(counts, dim=0)[:-1]
            a_lc = torch.searchsorted(uniq, fg_lab)              # anchor label -> compact
            a_cnt = counts[a_lc]
            rep = torch.repeat_interleave(torch.arange(fg_a.numel(), device=device), a_cnt)
            blk = torch.zeros(fg_a.numel(), dtype=torch.long, device=device)
            if fg_a.numel() > 1:
                blk[1:] = torch.cumsum(a_cnt, dim=0)[:-1]
            within = torch.arange(int(a_cnt.sum().item()), device=device) - blk[rep]
            fg_anchor.append(fg_a[rep])
            fg_token.append(tok_sorted[starts[a_lc][rep] + within])

        p1_anchor = torch.cat(p1_anchor) if p1_anchor else \
            torch.zeros(0, dtype=torch.long, device=device)
        p1_token = torch.cat(p1_token) if p1_token else \
            torch.zeros(0, dtype=torch.long, device=device)
        fg_anchor = torch.cat(fg_anchor) if fg_anchor else \
            torch.zeros(0, dtype=torch.long, device=device)
        fg_token = torch.cat(fg_token) if fg_token else \
            torch.zeros(0, dtype=torch.long, device=device)

        pair_anchor = torch.cat([p1_anchor, fg_anchor]) \
            if (p1_anchor.numel() > 0 or fg_anchor.numel() > 0) else \
            torch.zeros(0, dtype=torch.long, device=device)
        pair_token = torch.cat([p1_token, fg_token]) \
            if (p1_token.numel() > 0 or fg_token.numel() > 0) else \
            torch.zeros(0, dtype=torch.long, device=device)

        # ---- Stage E1: cap to max_kv nearest-|delta_p| per anchor ----
        # (two stable sorts: |delta| asc, then anchor asc -> per-anchor blocks
        # already distance-ordered; keep rank < max_kv. Result stays
        # anchor-ascending, as AnchorQueryCrossAttention requires.)
        if pair_anchor.numel() > 0:
            delta = tm["out"][pair_token] - anchor_out[pair_anchor]
            dist = delta.norm(dim=-1)
            o1 = torch.argsort(dist, stable=True)
            o2 = torch.argsort(pair_anchor[o1], stable=True)
            perm = o1[o2]
            pair_anchor = pair_anchor[perm]
            pair_token = pair_token[perm]
            rank = _segment_rank(pair_anchor, A)
            keep = rank < self.max_kv
            pair_anchor = pair_anchor[keep]
            pair_token = pair_token[keep]

        # ---- Stage E2: anchor embedding (P0) + queries ----
        p0_anchor = torch.cat(p0_anchor) if p0_anchor else \
            torch.zeros(0, dtype=torch.long, device=device)
        p0_token = torch.cat(p0_token) if p0_token else \
            torch.zeros(0, dtype=torch.long, device=device)
        emb = feat.new_zeros(A, self.dim)
        if p0_anchor.numel() > 0:
            p0_delta = tm["out"][p0_token] - anchor_out[p0_anchor]
            p0_feat = self.p0_proj(torch.cat([feat[p0_token], p0_delta], dim=-1))
            emb.index_add_(0, p0_anchor, p0_feat)
            p0_cnt = torch.bincount(p0_anchor, minlength=A).to(feat.dtype)
            emb = emb / p0_cnt.clamp_min(1.0).unsqueeze(-1)
        emb = self.p0_norm(emb)
        queries = self.query_embed.unsqueeze(0) + emb.unsqueeze(1)   # (A, K, D)

        # ---- Stage E3: K/V embedding + cross attention ----
        if pair_anchor.numel() > 0:
            kv_delta = tm["out"][pair_token] - anchor_out[pair_anchor]
            dframe = (tm["frame"][pair_token] - anchor_frame[pair_anchor]).to(feat.dtype)
            kv_in = torch.cat([
                feat[pair_token], kv_delta, dframe.unsqueeze(-1),
                tm["ray4"][pair_token],
            ], dim=-1)
            kv_feat = self.kv_norm(self.kv_embed(kv_in))
            kv_pos = tm["out"][pair_token]
        else:
            kv_feat = feat.new_zeros(0, self.dim)
            kv_pos = feat.new_zeros(0, 3)
        out, p_init = self.attn(queries, kv_feat, kv_pos, pair_anchor, A, anchor_out)

        # ---- Stage E4: flatten to gaussians (anchor-major, k fastest) + meta ----
        G = A * self.K
        out_feat = out.reshape(G, self.dim)
        p_flat = p_init.reshape(G, 3)
        r_g = torch.repeat_interleave(anchor_r, self.K)

        rep = lambda t: torch.repeat_interleave(t, self.K, dim=0)
        is_dyn_g = rep(anchor_is_dyn)
        label_g = rep(anchor_label)
        box_g = torch.where(is_dyn_g, rep(anchor_box), torch.full_like(label_g, -1))
        frame_g = rep(anchor_frame)

        # coord_ref: bg = p_init (already ref frame); fg = box-local -> ref via the
        # anchor's own-frame bbox (m3 falls back to this when no trajectory).
        coord_ref = p_flat.clone()
        if is_dyn_g.any():
            fg_rows = is_dyn_g.nonzero(as_tuple=True)[0]
            fg_frames = frame_g[fg_rows]
            fg_boxes = box_g[fg_rows]
            fb_key = fg_frames * 100000 + fg_boxes
            for key in fb_key.unique().tolist():
                g = int(key) // 100000
                box_i = int(key) % 100000
                rows = fg_rows[fb_key == key]
                bbox_ref_f = tm["bbox_ref_by_frame"][g]
                coord_ref[rows] = box_utils.box_local_to_ref(p_flat[rows], bbox_ref_f[box_i])

        counts_f = torch.tensor(
            [0 if frame_anchor[g] is None else frame_anchor[g]["cell_hash"].numel() * self.K
             for g in range(n_frames)], dtype=torch.long, device=device)
        gauss_offset = torch.cumsum(counts_f, dim=0)

        meta = {
            "box_assign": box_g,
            "instance_id": torch.where(is_dyn_g, label_g, torch.full_like(label_g, -1)),
            "is_dynamic": is_dyn_g,
            "coord_ref": coord_ref,
            "bbox_ref_by_frame": tm["bbox_ref_by_frame"],
        }
        return out_feat, p_flat, r_g, gauss_offset, meta
