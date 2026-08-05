"""Variable-count Gaussian generation for ``anchor_mode='grid'``.

Each occupied Utonia token keeps its temporal context while own-frame raw points
provide range-quantile position seeds. Background seeds live in the reference
frame and persistent foreground seeds live in box-local coordinates. In legacy
mode, deterministic metadata routes each token to an independent joint K head.
In learned mode, a post-temporal-fusion MLP predicts K logits and hard
Gumbel-Softmax straight-through routing selects among padded K-specific seed/head
candidates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import boxes as box_utils
from ..utils.attention import Rotary3D
from .gaussian_assembly import gradient_scale_identity, opacity_gate_st


# Fixed query-head geometry constants.  They are intentionally kept out of the
# training config because changing them alters the physical RoPE wavelengths,
# not an ordinary optimization hyperparameter.
_GRID_ROPE_BASE = 10.0
_GRID_BG_ROPE_POSITION_SCALE = 4.0
_GRID_FG_ROPE_POSITION_SCALE = 1.0


def _radius_pairs(query_pos, source_pos, radius: float, max_neighbors: int):
    """Radius lookup with a Cartesian spatial hash, followed by exact filtering.

    Returns local ``(query_index, source_index)`` pairs ordered by query and then
    distance.  The cap is applied independently to this source frame, which is
    what makes ``bg_max_kv_per_frame`` independent of the number of frames.
    No ``cdist`` or exact-cell-only matching is used.
    """
    device = query_pos.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if query_pos.shape[0] == 0 or source_pos.shape[0] == 0:
        return empty, empty
    radius = float(radius)
    max_neighbors = int(max_neighbors)
    if radius <= 0.0:
        raise ValueError("bg_radius_m must be positive")
    if max_neighbors <= 0:
        return empty, empty

    q_cell = torch.floor(query_pos / radius).to(torch.long)
    s_cell = torch.floor(source_pos / radius).to(torch.long)
    s_min = s_cell.amin(dim=0)
    s_max = s_cell.amax(dim=0)
    dims = (s_max - s_min + 1).clamp_min(1)
    stride_x = dims[1] * dims[2]
    stride_y = dims[2]

    s_local = s_cell - s_min
    s_key = s_local[:, 0] * stride_x + s_local[:, 1] * stride_y + s_local[:, 2]
    sorted_key, sorted_order = torch.sort(s_key)

    offsets = torch.tensor(
        [[x, y, z] for x in (-1, 0, 1)
         for y in (-1, 0, 1) for z in (-1, 0, 1)],
        dtype=torch.long, device=device,
    )
    neighbor_cell = q_cell[:, None, :] + offsets[None, :, :]
    neighbor_local = neighbor_cell - s_min.view(1, 1, 3)
    in_bounds = (
        (neighbor_local >= 0) & (neighbor_local < dims.view(1, 1, 3))
    ).all(dim=-1)
    safe = neighbor_local.clamp(min=0)
    safe = torch.minimum(safe, (dims - 1).view(1, 1, 3))
    neighbor_key = (
        safe[..., 0] * stride_x + safe[..., 1] * stride_y + safe[..., 2]
    ).reshape(-1)

    left = torch.searchsorted(sorted_key, neighbor_key, right=False)
    right = torch.searchsorted(sorted_key, neighbor_key, right=True)
    cell_count = (right - left) * in_bounds.reshape(-1).to(torch.long)
    total = int(cell_count.sum().item())
    if total == 0:
        return empty, empty

    neighbor_row = torch.repeat_interleave(
        torch.arange(cell_count.numel(), device=device), cell_count
    )
    cell_start = torch.cumsum(cell_count, dim=0) - cell_count
    rank = torch.arange(total, device=device) - torch.repeat_interleave(
        cell_start, cell_count
    )
    source_sorted_row = left[neighbor_row] + rank
    source_index = sorted_order[source_sorted_row]
    query_index = torch.div(neighbor_row, 27, rounding_mode="floor")

    delta = source_pos[source_index] - query_pos[query_index]
    distance2 = (delta * delta).sum(dim=-1)
    inside = distance2 <= radius * radius
    query_index = query_index[inside]
    source_index = source_index[inside]
    distance2 = distance2[inside]
    if query_index.numel() == 0:
        return empty, empty

    # Stable secondary sort by distance, then primary sort by query id.
    order = torch.argsort(distance2, stable=True)
    order = order[torch.argsort(query_index[order], stable=True)]
    query_index = query_index[order]
    source_index = source_index[order]

    counts = torch.bincount(query_index, minlength=query_pos.shape[0])
    starts = torch.cumsum(counts, dim=0) - counts
    within = torch.arange(query_index.numel(), device=device) - torch.repeat_interleave(
        starts, counts
    )
    keep = within < max_neighbors
    return query_index[keep], source_index[keep]


class _RaggedCrossAttention(nn.Module):
    """One-query-per-anchor cross attention over a ragged, fixed K/V memory.

    Relative geometry enters the scores via Utonia-compatible ``Rotary3D``:
    queries are rotated by their own token position and keys by their source-token
    position, so the logits depend only on the source-query displacement. Pair
    memory is the source token feature itself; no metadata is concatenated.
    """

    def __init__(self, dim, num_heads, n_layers, mlp_ratio=4,
                 rope_base=10.0, rope_position_scale=4.0):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_layers = int(n_layers)
        self.rope = Rotary3D(
            self.head_dim, base=rope_base, position_scale=rope_position_scale
        )

        # Normalize the feature-only Q/K/V embeddings once before their learned
        # projections. The attention/FFN blocks themselves then use the same
        # post-norm residual topology as AnchorQueryCrossAttention.
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.q_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.kv_proj = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(self.n_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            for _ in range(self.n_layers)
        ])
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    def forward(self, query_feat, pair_memory, pair_query, query_pos, pair_pos):
        query_feat = self.query_norm(query_feat)
        if pair_query.numel() == 0 or self.n_layers == 0:
            return query_feat

        order = torch.argsort(pair_query, stable=True)
        pair_query = pair_query[order]
        pair_memory = self.memory_norm(pair_memory[order])
        pair_pos = pair_pos[order]
        active, counts = torch.unique_consecutive(pair_query, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        local_row = torch.repeat_interleave(
            torch.arange(active.numel(), device=query_feat.device), counts
        )
        within = torch.arange(pair_query.numel(), device=query_feat.device) - \
            torch.repeat_interleave(starts, counts)
        max_len = int(counts.max().item())

        memory = pair_memory.new_zeros((active.numel(), max_len, self.dim))
        mask = torch.zeros(
            (active.numel(), max_len), dtype=torch.bool, device=query_feat.device
        )
        memory[local_row, within] = pair_memory
        mask[local_row, within] = True
        # fp32 positions for the rotary angles (bf16 costs centimeters at range)
        pos_pad = torch.zeros(
            (active.numel(), max_len, 3), dtype=torch.float32, device=query_feat.device
        )
        pos_pad[local_row, within] = pair_pos.float()

        x = query_feat[active]
        H, dh = self.num_heads, self.head_dim
        neg_mask = ~mask[:, None, None, :]
        # Rotary angles are per-position, shared by every layer.
        q_ang = self.rope.angles(query_pos[active])          # (A, 3, axis_pairs)
        k_ang = self.rope.angles(pos_pad)                    # (A, L, 3, axis_pairs)
        for layer in range(self.n_layers):
            q = self.q_proj[layer](x).reshape(-1, H, dh)
            key, value = self.kv_proj[layer](memory).chunk(2, dim=-1)
            key = key.reshape(active.numel(), max_len, H, dh)
            value = value.reshape(active.numel(), max_len, H, dh)
            q_rot = self.rope.rotate(q.float(), q_ang[:, None, :, :])
            key_rot = self.rope.rotate(key.float(), k_ang[:, :, None, :, :])
            scores = torch.einsum("ahd,alhd->ahl", q_rot, key_rot) * self.scale
            scores = scores.unsqueeze(2).masked_fill(neg_mask, float("-inf"))
            prob = torch.softmax(scores, dim=-1).squeeze(2)
            attended = torch.einsum("ahl,alhd->ahd", prob, value.float())
            attended = attended.reshape(active.numel(), self.dim).to(x.dtype)
            x = self.norm[layer](x + self.out_proj[layer](attended))
            x = self.norm_ffn[layer](x + self.ffn[layer](x))

        # Unmatched rows retain the normalized query embedding. Only active rows
        # receive cross-frame attention updates.
        return query_feat.index_copy(0, active, x)


class _FullSelfAttention(nn.Module):
    """Full self attention for one persistent object instance.

    The token content is feature-only; bbox-local position reaches Q/K scores
    through Utonia-compatible RoPE and is not added to V or the residual stream.
    """

    def __init__(self, dim, num_heads, n_layers, mlp_ratio=4,
                 rope_base=10.0, rope_position_scale=1.0):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_layers = int(n_layers)
        self.rope = Rotary3D(
            self.head_dim, base=rope_base, position_scale=rope_position_scale
        )
        self.input_norm = nn.LayerNorm(dim)
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(self.n_layers)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            for _ in range(self.n_layers)
        ])
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    def forward(self, feat, position, attn_mask=None):
        x = self.input_norm(feat)
        if feat.shape[0] == 0 or self.n_layers == 0:
            return x
        H, dh = self.num_heads, self.head_dim
        rope_angles = self.rope.angles(position)
        # Optional (N, N) bool mask (True = attend). It lets several disjoint
        # instances share ONE call through a block-diagonal same-instance mask
        # instead of one attention launch per instance; None keeps the original
        # full self-attention (each row attends every row).
        neg_mask = None if attn_mask is None else ~attn_mask[None]   # (1, N, N)
        for layer in range(self.n_layers):
            qkv = self.qkv[layer](x)
            q, key, value = qkv.chunk(3, dim=-1)
            q = q.reshape(-1, H, dh)
            key = key.reshape(-1, H, dh)
            value = value.reshape(-1, H, dh)
            q_rot = self.rope.rotate(q.float(), rope_angles[:, None, :, :])
            key_rot = self.rope.rotate(key.float(), rope_angles[:, None, :, :])
            scores = torch.einsum(
                "ihd,jhd->hij", q_rot, key_rot
            ) * self.scale
            if neg_mask is not None:
                scores = scores.masked_fill(neg_mask, float("-inf"))
            prob = torch.softmax(scores, dim=-1)
            attended = torch.einsum("hij,jhd->ihd", prob, value.float())
            attended = attended.reshape(-1, self.dim).to(x.dtype)
            x = self.norm[layer](x + self.out[layer](attended))
            x = self.norm_ffn[layer](x + self.ffn[layer](x))
        return x


class GridTemporalAggregator(nn.Module):
    """Background radius cross-attention and foreground self-attention.

    This is the token temporal-fusion stage shared by BOTH anchor modes:
    grid runs it with its padded seed tensors (which it additionally
    transforms into ref/box-local frames), while spherical passes
    ``seed_sensor=None`` and consumes only the fused features -- its own
    raw-point seed geometry is built downstream by the spherical head.
    """

    def __init__(self, cfg, dim, r_far):
        super().__init__()
        required = (
            "bg_radius_m", "bg_max_kv_per_frame", "num_heads",
            "bg_layers", "fg_layers",
        )
        missing = [name for name in required if getattr(cfg, name, None) is None]
        if missing:
            raise ValueError(
                "p2g.grid_query is missing required keys: " + ", ".join(missing)
            )
        self.dim = int(dim)
        self.radius = float(cfg.bg_radius_m)
        self.max_kv_per_frame = int(cfg.bg_max_kv_per_frame)
        num_heads = int(cfg.num_heads)
        self.bg_attention = _RaggedCrossAttention(
            self.dim, num_heads, int(cfg.bg_layers),
            rope_base=_GRID_ROPE_BASE,
            rope_position_scale=_GRID_BG_ROPE_POSITION_SCALE,
        )
        self.fg_attention = _FullSelfAttention(
            self.dim, num_heads, int(cfg.fg_layers),
            rope_base=_GRID_ROPE_BASE,
            rope_position_scale=_GRID_FG_ROPE_POSITION_SCALE,
        )
        # Foreground instances are packed into one block-diagonal masked
        # self-attention (O(T^2) scores). Above this many fg tokens the scores
        # tensor gets large, so fall back to the per-instance loop instead.
        self._fg_dense_cap = int(getattr(cfg, "fg_dense_cap", 2048))

    @staticmethod
    def _frame_bounds(new_offset, global_frame):
        start = int(new_offset[global_frame - 1]) if global_frame > 0 else 0
        return start, int(new_offset[global_frame])

    def _token_metadata(self, anchor, seed_sensor, delta_sensor, new_offset,
                        frame_batch_idx, pose_list, bbox_list,
                        bbox_instance_ids_list, timestamps_list):
        device, dtype = anchor.device, anchor.dtype
        N = anchor.shape[0]
        has_seeds = seed_sensor is not None
        if has_seeds:
            if delta_sensor is None or seed_sensor.shape != delta_sensor.shape:
                raise ValueError("seed_sensor and delta_sensor must have identical shapes")
            if (
                seed_sensor.ndim not in (3, 4)
                or seed_sensor.shape[0] != N
                or seed_sensor.shape[-1] != 3
            ):
                raise ValueError(
                    "grid seeds must have shape (N, K_max, 3) or "
                    "(N, K_choices, K_max, 3)"
                )
        elif delta_sensor is not None:
            raise ValueError("delta_sensor requires seed_sensor")
        n_frames = int(frame_batch_idx.numel())
        token_frame = torch.zeros(N, dtype=torch.long, device=device)
        coord_ref = torch.zeros_like(anchor)
        coord_out = torch.zeros_like(anchor)
        seed_ref = torch.zeros_like(seed_sensor) if has_seeds else None
        seed_out = torch.zeros_like(seed_sensor) if has_seeds else None
        center_out = torch.zeros_like(seed_sensor) if has_seeds else None
        is_dynamic = torch.zeros(N, dtype=torch.bool, device=device)
        instance_id = torch.full((N,), -1, dtype=torch.long, device=device)
        box_assign = torch.full((N,), -1, dtype=torch.long, device=device)
        bbox_ref_by_frame = [None for _ in range(n_frames)]
        frame_indices_out = [None for _ in range(n_frames)]

        batch_frame_map = {}
        for global_f, batch_id in enumerate(frame_batch_idx.tolist()):
            batch_frame_map.setdefault(int(batch_id), []).append(global_f)

        for batch_id, frame_indices in batch_frame_map.items():
            pose_b = pose_list[batch_id]
            bbox_b = bbox_list[batch_id]
            iids_b = (
                bbox_instance_ids_list[batch_id]
                if bbox_instance_ids_list is not None else None
            )
            if iids_b is not None and len(iids_b) >= 2:
                common_ids = box_utils.common_instance_ids(iids_b[0], iids_b[-1])
            else:
                common_ids = set()
            for local_f, global_f in enumerate(frame_indices):
                start, end = self._frame_bounds(new_offset, global_f)
                rows = torch.arange(start, end, device=device)
                frame_indices_out[global_f] = rows
                token_frame[start:end] = global_f

                anchor_f = anchor[start:end]
                pose_f = pose_b[local_f].to(device=device, dtype=dtype)
                ref_f = box_utils.apply_pose(anchor_f, pose_f)
                coord_ref[start:end] = ref_f
                coord_out[start:end] = ref_f

                if has_seeds:
                    seed_sensor_f = seed_sensor[start:end]
                    center_sensor_f = seed_sensor_f - delta_sensor[start:end]
                    seed_shape = seed_sensor_f.shape
                    seed_ref_f = box_utils.apply_pose(
                        seed_sensor_f.reshape(-1, 3), pose_f
                    ).reshape(seed_shape)
                    center_ref_f = box_utils.apply_pose(
                        center_sensor_f.reshape(-1, 3), pose_f
                    ).reshape(seed_shape)
                    seed_ref[start:end] = seed_ref_f
                    seed_out[start:end] = seed_ref_f
                    center_out[start:end] = center_ref_f

                bbox_sensor_f = bbox_b[local_f].to(device=device, dtype=dtype)
                bbox_ref_f = box_utils.transform_boxes_to_ref(bbox_sensor_f, pose_f)
                bbox_ref_by_frame[global_f] = bbox_ref_f
                if iids_b is not None:
                    iids_f = iids_b[local_f].to(device=device, dtype=torch.long)
                else:
                    iids_f = torch.arange(
                        bbox_sensor_f.shape[0], device=device, dtype=torch.long
                    )

                assigned_f = box_utils.point_in_box(anchor_f, bbox_sensor_f)
                instance_f = torch.full_like(assigned_f, -1)
                valid = assigned_f >= 0
                if valid.any() and iids_f.numel() > 0:
                    instance_f[valid] = iids_f[assigned_f[valid]]
                dynamic_f = torch.zeros_like(valid)
                for item_id in common_ids:
                    dynamic_f |= instance_f == int(item_id)

                is_dynamic[start:end] = dynamic_f
                instance_id[start:end] = torch.where(
                    dynamic_f, instance_f, torch.full_like(instance_f, -1)
                )
                box_assign[start:end] = torch.where(
                    dynamic_f, assigned_f, torch.full_like(assigned_f, -1)
                )
                for box_index in assigned_f[dynamic_f].unique().tolist():
                    box_index = int(box_index)
                    selected = dynamic_f & (assigned_f == box_index)
                    if selected.any():
                        out_f = coord_out[start:end]
                        out_f[selected] = box_utils.points_to_box_local(
                            ref_f[selected], bbox_ref_f[box_index]
                        )
                        coord_out[start:end] = out_f
                        if has_seeds:
                            seed_out_f = seed_out[start:end]
                            center_out_f = center_out[start:end]
                            selected_shape = seed_out_f[selected].shape
                            seed_out_f[selected] = box_utils.points_to_box_local(
                                seed_ref_f[selected].reshape(-1, 3), bbox_ref_f[box_index]
                            ).reshape(selected_shape)
                            center_out_f[selected] = box_utils.points_to_box_local(
                                center_ref_f[selected].reshape(-1, 3), bbox_ref_f[box_index]
                            ).reshape(selected_shape)
                            seed_out[start:end] = seed_out_f
                            center_out[start:end] = center_out_f

        return {
            "frame": token_frame,
            "ref": coord_ref,
            "out": coord_out,
            "seed_ref": seed_ref,
            "seed_out": seed_out,
            "seed_delta": (seed_out - center_out) if has_seeds else None,
            "is_dynamic": is_dynamic,
            "instance_id": instance_id,
            "box_assign": box_assign,
            "bbox_ref_by_frame": bbox_ref_by_frame,
            "frame_indices": frame_indices_out,
            "batch_frame_map": batch_frame_map,
        }

    def _background_pairs(self, metadata):
        pair_query, pair_source = [], []
        background = ~metadata["is_dynamic"]
        for frame_indices in metadata["batch_frame_map"].values():
            for target_frame in frame_indices:
                target_rows = metadata["frame_indices"][target_frame]
                target_rows = target_rows[background[target_rows]]
                if target_rows.numel() == 0:
                    continue
                for source_frame in frame_indices:
                    if source_frame == target_frame:
                        continue
                    source_rows = metadata["frame_indices"][source_frame]
                    source_rows = source_rows[background[source_rows]]
                    local_q, local_s = _radius_pairs(
                        metadata["out"][target_rows], metadata["out"][source_rows],
                        self.radius, self.max_kv_per_frame,
                    )
                    if local_q.numel() > 0:
                        pair_query.append(target_rows[local_q])
                        pair_source.append(source_rows[local_s])
        device = metadata["out"].device
        empty = torch.zeros(0, dtype=torch.long, device=device)
        if not pair_query:
            return empty, empty
        return torch.cat(pair_query), torch.cat(pair_source)

    def forward(self, feat, anchor, seed_sensor, delta_sensor, new_offset,
                frame_batch_idx, pose_list, bbox_list,
                bbox_instance_ids_list=None, timestamps_list=None):
        metadata = self._token_metadata(
            anchor, seed_sensor, delta_sensor, new_offset, frame_batch_idx,
            pose_list, bbox_list, bbox_instance_ids_list, timestamps_list,
        )
        pair_query, pair_source = self._background_pairs(metadata)
        out_feat = self.bg_attention(
            feat, feat[pair_source], pair_query,
            metadata["out"], metadata["out"][pair_source],
        )

        # Each dynamic instance's tokens fuse only among themselves. They are
        # disjoint and background attention never touches them, so instead of one
        # attention launch per instance (measured launch-bound) pack every
        # instance into a SINGLE block-diagonal masked self-attention and scatter
        # once. A rare very dense frame exceeds the O(T^2) score budget and falls
        # back to the per-instance path.
        fg_rows_list = []
        for frame_indices in metadata["batch_frame_map"].values():
            frame_rows = torch.cat([metadata["frame_indices"][f] for f in frame_indices])
            dynamic_rows = frame_rows[metadata["is_dynamic"][frame_rows]]
            if dynamic_rows.numel() == 0:
                continue
            instance_ids = metadata["instance_id"][dynamic_rows]
            for item_id in instance_ids.unique().tolist():
                rows = dynamic_rows[instance_ids == int(item_id)]
                if rows.numel() > 0:
                    fg_rows_list.append(rows)
        if fg_rows_list:
            all_rows = torch.cat(fg_rows_list)
            if int(all_rows.numel()) <= self._fg_dense_cap:
                lengths = torch.tensor(
                    [int(r.numel()) for r in fg_rows_list], device=out_feat.device
                )
                seg = torch.repeat_interleave(
                    torch.arange(lengths.numel(), device=out_feat.device), lengths
                )
                block_mask = seg[:, None] == seg[None, :]     # True = same instance
                updated = self.fg_attention(
                    feat[all_rows], metadata["out"][all_rows], attn_mask=block_mask
                )
            else:
                updated = torch.cat([
                    self.fg_attention(feat[rows], metadata["out"][rows])
                    for rows in fg_rows_list
                ])
            out_feat = out_feat.index_copy(0, all_rows, updated)

        agg_meta = {
            "box_assign": metadata["box_assign"],
            "instance_id": metadata["instance_id"],
            "is_dynamic": metadata["is_dynamic"],
            "coord_ref": metadata["ref"],
            "seed_ref": metadata["seed_ref"],
            "bbox_ref_by_frame": metadata["bbox_ref_by_frame"],
        }
        return (
            out_feat, metadata["out"], metadata["seed_out"],
            metadata["seed_delta"], agg_meta,
        )


class ViewTokenPositionEncoder(nn.Module):
    """Fourier-encode each token's position in the target view's sensor frame.

    ``pose`` is ``T_{0<-t}``, so a ref-frame token center lands at
    ``R^T (p_ref - t)`` in that target sensor's axes. Encoding this instead of a
    token-independent pose descriptor makes the router input differ per (token,
    view) pair, which is what a per-token count decision actually depends on:
    the same ego motion changes a 5 m token's viewing geometry far more than a
    100 m one's.
    """

    def __init__(self, cfg):
        super().__init__()
        self.position_frequencies = int(
            getattr(cfg, "position_frequencies", 8)
        )
        if self.position_frequencies <= 0:
            raise ValueError("viewpoint position frequency count must be positive")
        # Half-period of the lowest band. It must cover the farthest returns
        # (110 m here), otherwise band 0 wraps inside the sensing range and two
        # tokens at very different distances collide in the embedding.
        position_scale = float(getattr(cfg, "position_scale_m", 110.0))
        if position_scale <= 0.0:
            raise ValueError("viewpoint position_scale_m must be positive")
        self.register_buffer(
            "position_scale",
            torch.tensor(position_scale, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "position_bands",
            2.0 ** torch.arange(self.position_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.out_dim = 3 * 2 * self.position_frequencies

    @staticmethod
    def to_view_frame(position_ref, pose):
        """Map ref-frame token centers into their target view's sensor frame."""
        if pose.ndim < 2 or tuple(pose.shape[-2:]) != (4, 4):
            raise ValueError(
                f"viewpoint pose must end in (4, 4), got {tuple(pose.shape)}"
            )
        if position_ref.shape[-1] != 3:
            raise ValueError(
                "viewpoint token positions must end in 3 channels, got "
                f"{position_ref.shape[-1]}"
            )
        rotation = pose[..., :3, :3]
        translation = pose[..., :3, 3]
        centered = position_ref - translation
        # pose maps target sensor -> ref, so the inverse rotation is R^T.
        return torch.einsum("...ji,...j->...i", rotation, centered)

    def forward(self, position_ref, pose):
        position_view = self.to_view_frame(position_ref.float(), pose.float())
        normalized = position_view / self.position_scale
        angles = torch.pi * normalized.unsqueeze(-1) * self.position_bands
        return torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        ).flatten(-2)


class GridSlotHead(nn.Module):
    """Predict variable-count Gaussian parameters for config-selected routing.

    ``count_mode="legacy"`` preserves the original module shapes and state-dict
    keys exactly: external ``anchor_k`` metadata selects one head.

    ``count_mode="learned_gumbel"`` predicts logits from the temporally fused
    anchor feature and executes only the selected K-specific head for each
    token. The selected hard-ST scalar gates the activated opacity of that
    expert's K outputs: forward rendering is unchanged, while backward connects
    the rendering loss to all count logits through the softmax Jacobian.

    ``count_mode="learned_gumbel_viewpt"`` predicts one view-independent Common
    Gaussian from each fused token, then predicts total K separately for every
    target ``gt["pose"]``. The 192D view feature is a *router-only* input: it is
    built from the fused token plus that token's position in the target sensor
    frame and feeds nothing but the count logits. Both the Common and the K-1
    Additional Gaussians are predicted from the fused token itself, so Gaussian
    parameters stay view-independent and only their count adapts. The Common
    output is stored once with ``view_index = -1``; its per-view router gate is
    deferred until the renderer constructs ``Common union Additional_view``.
    """

    def __init__(self, cfg, gs_params, dim):
        super().__init__()
        self.dim = int(dim)
        self.count_mode = str(getattr(cfg, "count_mode", "legacy")).lower()
        learned_count = None
        if self.count_mode == "legacy":
            k_max = getattr(cfg, "K_max", None)
            if k_max is None:
                raise ValueError("p2g.grid_query.K_max is required")
            self.gumbel_tau = None
        elif self.count_mode in ("learned_gumbel", "learned_gumbel_viewpt"):
            learned_count = getattr(cfg, "learned_count", None)
            if learned_count is None:
                raise ValueError(
                    "p2g.grid_query.learned_count is required for "
                    f"count_mode={self.count_mode!r}"
                )
            k_max = getattr(learned_count, "K_max", None)
            if k_max is None:
                raise ValueError(
                    "p2g.grid_query.learned_count.K_max is required"
                )
            self.gumbel_tau = float(getattr(learned_count, "tau", 1.0))
            if not self.gumbel_tau > 0.0:
                raise ValueError(
                    "p2g.grid_query.learned_count.tau must be positive"
                )
        else:
            raise ValueError(
                "p2g.grid_query.count_mode must be 'legacy', "
                "'learned_gumbel', or 'learned_gumbel_viewpt'"
            )
        self.k_max = int(k_max)
        self.grad_balance = str(getattr(cfg, "grad_balance", "sqrt_k"))
        configured_scope = getattr(cfg, "grad_balance_scope", "output")
        if learned_count is not None:
            configured_scope = getattr(
                learned_count, "grad_balance_scope", configured_scope
            )
        # Missing scope preserves the historical output-level backward. The
        # active learned experiment nests ``token`` under learned_count, so
        # switching count_mode back to legacy also restores legacy gradients.
        self.grad_balance_scope = str(
            configured_scope
        ).lower()
        if self.k_max <= 0:
            raise ValueError("grid_query.K_max must be positive")
        if self.grad_balance not in ("sqrt_k", "none"):
            raise ValueError("grid_query.grad_balance must be 'sqrt_k' or 'none'")
        if self.grad_balance_scope not in ("output", "token"):
            raise ValueError(
                "grid_query.grad_balance_scope must be 'output' or 'token'"
            )

        param_names = ["shs", "opacity", "scaling", "rotation"]
        param_sizes = []
        for name in param_names:
            size = getattr(gs_params, name, None)
            if size is None:
                raise ValueError(f"p2g.gs_params.{name} is required for grid mode")
            param_sizes.append(int(size))
        offset_size = int(getattr(gs_params, "offset", 0) or 0)
        if offset_size > 0:
            param_sizes.append(offset_size)
        self.param_dim = sum(param_sizes)
        if self.param_dim <= 0:
            raise ValueError("p2g.gs_params must define a positive output width")
        self.opacity_start = param_sizes[0]
        self.opacity_end = self.opacity_start + param_sizes[1]

        if self.count_mode != "learned_gumbel_viewpt":
            # Preserve the exact legacy/learned_gumbel module names and shapes.
            self.trunk = nn.Sequential(
                nn.Linear(self.dim + 3 * self.k_max, self.dim),
                nn.SiLU(),
            )
            self.k_heads = nn.ModuleList([
                nn.Linear(self.dim, k * self.param_dim)
                for k in range(1, self.k_max + 1)
            ])

        if self.count_mode == "learned_gumbel":
            self.count_predictor = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.k_max),
            )
            # Uniform initial categorical probabilities avoid imposing an
            # arbitrary K preference before the rendering losses provide signal.
            nn.init.zeros_(self.count_predictor[-1].weight)
            nn.init.zeros_(self.count_predictor[-1].bias)
        elif self.count_mode == "learned_gumbel_viewpt":
            viewpoint_cfg = getattr(learned_count, "viewpoint", learned_count)
            self.view_position_encoder = ViewTokenPositionEncoder(viewpoint_cfg)
            if self.view_position_encoder.out_dim != 48:
                raise ValueError(
                    "learned_gumbel_viewpt requires a 48D view-token position "
                    "embedding; configured frequencies produce "
                    f"{self.view_position_encoder.out_dim}D"
                )
            self.view_mlp = nn.Sequential(
                nn.LayerNorm(self.dim + 48),
                nn.Linear(self.dim + 48, 384),
                nn.SiLU(),
                nn.Linear(384, 384),
                nn.SiLU(),
                nn.Linear(384, 192),
            )
            self.common_predictor = nn.Sequential(
                nn.Linear(self.dim + 3, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.param_dim),
            )
            self.count_predictor = nn.Sequential(
                nn.Linear(192, 192),
                nn.SiLU(),
                nn.Linear(192, self.k_max),
            )
            additional_width = 3 * max(self.k_max - 1, 0)
            # Additional Gaussians read the same fused token as Common, not the
            # router feature, so their parameters never depend on the target
            # view. With dim=192 this keeps the historical 201 -> 192 shapes.
            self.additional_trunk = nn.Sequential(
                nn.Linear(self.dim + additional_width, self.dim),
                nn.SiLU(),
            )
            self.additional_heads = nn.ModuleList([
                nn.Linear(self.dim, additional_k * self.param_dim)
                for additional_k in range(1, self.k_max)
            ])
            # Start with a uniform total-K policy.
            nn.init.zeros_(self.count_predictor[-1].weight)
            nn.init.zeros_(self.count_predictor[-1].bias)

    def _pack_selected_prefix(self, selected_k, anchor_offset):
        num_anchors = selected_k.shape[0]
        anchor_index = torch.repeat_interleave(
            torch.arange(num_anchors, device=selected_k.device),
            selected_k,
        )
        gaussian_offset = self._gaussian_offset(selected_k, anchor_offset)
        if anchor_index.numel() == 0:
            empty_long = torch.zeros(
                0, dtype=torch.long, device=selected_k.device
            )
            return anchor_index, empty_long, empty_long, gaussian_offset

        slot_start = torch.cumsum(selected_k, dim=0) - selected_k
        slot_index = (
            torch.arange(anchor_index.numel(), device=selected_k.device)
            - torch.repeat_interleave(slot_start, selected_k)
        )
        slot_k = selected_k[anchor_index]
        return anchor_index, slot_index, slot_k, gaussian_offset

    def _opacity_gate_st(self, raw_params, gate):
        """Forward-identical raw opacity with activated-opacity ST backward.

        The renderer applies sigmoid internally.  We therefore build the exact
        activated opacity ``sigmoid(raw) * gate``, map it back to a raw logit,
        and use a straight-through value replacement so the selected expert's
        opacities are bit-identical in the forward. Only packed selected slots
        enter this function, hence their hard gate value is one and the renderer
        never receives zero-opacity placeholders.
        """
        if raw_params.shape[0] == 0:
            return raw_params
        raw_opacity = raw_params[:, self.opacity_start:self.opacity_end]
        opacity_st = opacity_gate_st(raw_opacity, gate)
        return torch.cat([
            raw_params[:, :self.opacity_start],
            opacity_st,
            raw_params[:, self.opacity_end:],
        ], dim=-1)

    def _gaussian_offset(self, anchor_k, anchor_offset):
        frame_gaussian_counts = []
        start = 0
        for end_tensor in anchor_offset:
            end = int(end_tensor)
            frame_gaussian_counts.append(anchor_k[start:end].sum())
            start = end
        if not frame_gaussian_counts:
            return torch.zeros(0, dtype=torch.long, device=anchor_k.device)
        return torch.cumsum(torch.stack(frame_gaussian_counts), dim=0).long()

    def _balance_token_feature(self, feature, k):
        """Scale only the gradient returning to the shared token/trunk path."""
        if (
            self.grad_balance != "sqrt_k"
            or self.grad_balance_scope != "token"
        ):
            return feature
        return gradient_scale_identity(feature, float(k) ** -0.5)

    def _forward_legacy(self, anchor_feature, anchor_k, delta_p, anchor_offset):
        if anchor_k is None:
            raise ValueError("legacy grid count routing requires anchor_k")
        if anchor_k.ndim != 1 or anchor_k.shape[0] != anchor_feature.shape[0]:
            raise ValueError("anchor_k must be one-dimensional and aligned with anchor_feature")
        expected_delta_shape = (anchor_feature.shape[0], self.k_max, 3)
        if tuple(delta_p.shape) != expected_delta_shape:
            raise ValueError(
                f"delta_p must have shape {expected_delta_shape}, got {tuple(delta_p.shape)}"
            )
        anchor_k = anchor_k.to(device=anchor_feature.device, dtype=torch.long)
        invalid_k = (anchor_k < 1) | (anchor_k > self.k_max)
        if invalid_k.any():
            raise ValueError(f"anchor_k values must be in [1, {self.k_max}]")
        anchor_index = torch.repeat_interleave(
            torch.arange(anchor_feature.shape[0], device=anchor_feature.device), anchor_k
        )
        gaussian_offset = self._gaussian_offset(anchor_k, anchor_offset)
        if anchor_index.numel() == 0:
            empty_long = torch.zeros(0, dtype=torch.long, device=anchor_feature.device)
            return anchor_feature.new_zeros((0, self.param_dim)), {
                "anchor_index": empty_long,
                "slot_index": empty_long,
                "slot_k": empty_long,
                "anchor_k": anchor_k,
                "gaussian_offset": gaussian_offset,
            }

        slot_k = anchor_k[anchor_index]
        slot_start = torch.cumsum(anchor_k, dim=0) - anchor_k
        slot_index = torch.arange(anchor_index.numel(), device=anchor_feature.device) - \
            torch.repeat_interleave(slot_start, anchor_k)
        trunk_input = torch.cat([
            anchor_feature,
            delta_p.to(dtype=anchor_feature.dtype).reshape(anchor_feature.shape[0], -1),
        ], dim=-1)
        trunk_feature = self.trunk(trunk_input)
        raw_params = anchor_feature.new_zeros((anchor_index.numel(), self.param_dim))
        anchor_start = torch.cumsum(anchor_k, dim=0) - anchor_k
        for k, head in enumerate(self.k_heads, start=1):
            anchor_rows = (anchor_k == k).nonzero(as_tuple=True)[0]
            if anchor_rows.numel() == 0:
                continue
            head_input = self._balance_token_feature(
                trunk_feature[anchor_rows], k
            )
            predicted = head(head_input).reshape(-1, k, self.param_dim)
            output_rows = anchor_start[anchor_rows, None] + torch.arange(
                k, device=anchor_feature.device
            ).view(1, -1)
            raw_params = raw_params.index_copy(
                0, output_rows.reshape(-1), predicted.reshape(-1, self.param_dim)
            )
        return raw_params, {
            "anchor_index": anchor_index,
            "slot_index": slot_index,
            "slot_k": slot_k,
            "anchor_k": anchor_k,
            "gaussian_offset": gaussian_offset,
        }

    def _gumbel_selection(self, logits):
        """Hard one-hot forward; soft categorical derivative backward."""
        if self.training:
            return F.gumbel_softmax(
                logits.float(),
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            ).to(dtype=logits.dtype)
        selected = logits.argmax(dim=-1)
        return F.one_hot(selected, num_classes=self.k_max).to(dtype=logits.dtype)

    def _forward_learned(self, anchor_feature, anchor_k, delta_p, anchor_offset):
        if anchor_k is not None:
            raise ValueError(
                "learned_gumbel predicts anchor_k after temporal fusion; "
                "external anchor_k must be None"
            )
        expected_delta_shape = (
            anchor_feature.shape[0], self.k_max, self.k_max, 3,
        )
        if tuple(delta_p.shape) != expected_delta_shape:
            raise ValueError(
                "learned_gumbel delta_p must have shape "
                f"{expected_delta_shape}, got {tuple(delta_p.shape)}"
            )

        logits = self.count_predictor(anchor_feature)
        selection = self._gumbel_selection(logits)
        selected_index = selection.argmax(dim=-1).to(torch.long)
        selected_k = selected_index + 1
        num_anchors = anchor_feature.shape[0]

        # K is a hard routing decision in the forward: each token contributes
        # only its selected candidate geometry and runs only its selected
        # K-specific output head. The discrete gather deliberately carries no
        # count-router gradient; that gradient is supplied below by the selected
        # ST scalar applied to activated opacity.
        anchor_rows = torch.arange(
            num_anchors, device=anchor_feature.device
        )
        selected_delta = delta_p[anchor_rows, selected_index]
        trunk_input = torch.cat([
            anchor_feature,
            selected_delta.to(dtype=anchor_feature.dtype).reshape(
                num_anchors, 3 * self.k_max
            ),
        ], dim=-1)
        trunk_feature = self.trunk(trunk_input)

        (
            anchor_index,
            slot_index,
            slot_k,
            gaussian_offset,
        ) = self._pack_selected_prefix(
            selected_k, anchor_offset
        )
        raw_params = anchor_feature.new_zeros(
            (anchor_index.numel(), self.param_dim)
        )
        anchor_start = torch.cumsum(selected_k, dim=0) - selected_k
        for k, head in enumerate(self.k_heads, start=1):
            selected_rows = (selected_k == k).nonzero(as_tuple=True)[0]
            if selected_rows.numel() == 0:
                continue
            head_input = self._balance_token_feature(
                trunk_feature[selected_rows], k
            )
            predicted = head(head_input).reshape(
                -1, k, self.param_dim
            )
            output_rows = anchor_start[selected_rows, None] + torch.arange(
                k, device=anchor_feature.device
            ).view(1, -1)
            raw_params = raw_params.index_copy(
                0,
                output_rows.reshape(-1),
                predicted.reshape(-1, self.param_dim),
            )

        # For token n, gather only y_ST[n, k*_n]. Its forward value is exactly
        # one. Since that single probability still depends on every logit via
        # softmax normalization, the opacity gate sends gradients to all K logits
        # without executing or padding the unselected expert outputs.
        selected_gate = selection.gather(
            1, selected_index.unsqueeze(-1)
        ).squeeze(-1)
        packed_gate = selected_gate[anchor_index]
        raw_params = self._opacity_gate_st(raw_params, packed_gate)

        return raw_params, {
            "anchor_index": anchor_index,
            "slot_index": slot_index,
            "slot_k": slot_k,
            "anchor_k": selected_k,
            "gaussian_offset": gaussian_offset,
            "k_logits": logits,
            "k_selection": selection,
            "packed_selected_gate": packed_gate,
            "selected_only": True,
        }

    def _target_view_decisions(self, anchor_batch, target_pose, *, device, dtype):
        """Expand each source token into the target views of its batch item."""
        if not isinstance(target_pose, (list, tuple)):
            raise ValueError(
                "learned_gumbel_viewpt requires gt['pose'] as a list of "
                "(V, 4, 4) tensors"
            )
        if anchor_batch.ndim != 1:
            raise ValueError("anchor_batch must be one-dimensional")
        view_counts = []
        pose_parts = []
        for batch_id, pose_b in enumerate(target_pose):
            if not torch.is_tensor(pose_b) or pose_b.ndim != 3 or \
                    tuple(pose_b.shape[-2:]) != (4, 4):
                raise ValueError(
                    f"target_pose[{batch_id}] must have shape (V, 4, 4)"
                )
            view_counts.append(int(pose_b.shape[0]))
            pose_parts.append(pose_b.to(device=device, dtype=dtype))
        view_counts = torch.tensor(view_counts, device=device, dtype=torch.long)

        if anchor_batch.numel() > 0:
            if view_counts.numel() == 0:
                raise ValueError("target_pose cannot be empty when anchors exist")
            invalid = (anchor_batch < 0) | (anchor_batch >= view_counts.numel())
            if invalid.any():
                raise ValueError("anchor_batch contains an invalid batch index")
            if (view_counts[anchor_batch] <= 0).any():
                raise ValueError("every batch item containing anchors needs a target view")

        per_anchor = (
            view_counts[anchor_batch]
            if anchor_batch.numel() > 0
            else anchor_batch.new_zeros((0,))
        )
        decision_anchor = torch.repeat_interleave(
            torch.arange(anchor_batch.numel(), device=device), per_anchor
        )
        if decision_anchor.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            empty_pose = torch.zeros((0, 4, 4), device=device, dtype=dtype)
            return decision_anchor, empty, empty_pose

        decision_start = torch.cumsum(per_anchor, dim=0) - per_anchor
        decision_view = (
            torch.arange(decision_anchor.numel(), device=device)
            - torch.repeat_interleave(decision_start, per_anchor)
        )
        pose_start = torch.cumsum(view_counts, dim=0) - view_counts
        flat_pose = torch.cat(pose_parts, dim=0)
        pose_index = (
            pose_start[anchor_batch[decision_anchor]] + decision_view
        )
        return decision_anchor, decision_view, flat_pose[pose_index]

    def _forward_viewpoint(
        self, anchor_feature, anchor_k, delta_p, anchor_offset,
        target_pose, anchor_batch, anchor_position_ref,
    ):
        if anchor_k is not None:
            raise ValueError(
                "learned_gumbel_viewpt predicts total K after temporal fusion; "
                "external anchor_k must be None"
            )
        expected_delta_shape = (
            anchor_feature.shape[0], self.k_max, self.k_max, 3,
        )
        if tuple(delta_p.shape) != expected_delta_shape:
            raise ValueError(
                "learned_gumbel_viewpt delta_p must have shape "
                f"{expected_delta_shape}, got {tuple(delta_p.shape)}"
            )
        if anchor_batch is None:
            raise ValueError("learned_gumbel_viewpt requires anchor_batch")
        anchor_batch = anchor_batch.to(
            device=anchor_feature.device, dtype=torch.long
        )
        if anchor_batch.shape != (anchor_feature.shape[0],):
            raise ValueError("anchor_batch must align one-to-one with anchor_feature")
        if anchor_position_ref is None:
            raise ValueError(
                "learned_gumbel_viewpt requires anchor_position_ref: the router "
                "conditions on each token's position in the target sensor frame"
            )
        # One row per (token, target view): a dynamic token sits somewhere else
        # in each view because its box has moved by then.
        if (
            anchor_position_ref.ndim != 3
            or anchor_position_ref.shape[0] != anchor_feature.shape[0]
            or anchor_position_ref.shape[-1] != 3
        ):
            raise ValueError(
                "anchor_position_ref must have shape "
                f"({anchor_feature.shape[0]}, V, 3), got "
                f"{tuple(anchor_position_ref.shape)}"
            )
        anchor_position_ref = anchor_position_ref.to(
            device=anchor_feature.device
        )

        decision_anchor, decision_view, decision_pose = self._target_view_decisions(
            anchor_batch,
            target_pose,
            device=anchor_feature.device,
            dtype=torch.float32,
        )
        decision_feature = anchor_feature[decision_anchor]
        if decision_view.numel() > 0 and (
            int(decision_view.max()) >= anchor_position_ref.shape[1]
        ):
            raise ValueError(
                "anchor_position_ref is narrower than the target view count"
            )
        # Geometry only: token centers and target poses are data, so this branch
        # adds no gradient path of its own. The router still reaches the fused
        # token through decision_feature below.
        position_embedding = self.view_position_encoder(
            anchor_position_ref[decision_anchor, decision_view].detach(),
            decision_pose,
        ).to(dtype=anchor_feature.dtype)
        # The 192D view feature is consumed by the count router alone. Nothing
        # downstream of it produces a Gaussian parameter, so the only gradient
        # reaching view_mlp is the straight-through opacity gate (plus the
        # routing-budget loss on the same logits).
        view_feature = self.view_mlp(torch.cat([
            decision_feature, position_embedding,
        ], dim=-1))

        logits = self.count_predictor(view_feature)
        selection = self._gumbel_selection(logits)
        selected_index = selection.argmax(dim=-1).to(torch.long)
        selected_k = selected_index + 1

        # Common is predicted once per source token without any viewpoint input.
        # Its base position delta is included exactly like the Additional branch.
        common_delta = delta_p[:, 0, 0].to(dtype=anchor_feature.dtype)
        common_params = self.common_predictor(torch.cat([
            anchor_feature, common_delta,
        ], dim=-1))

        # Store Common physically once per source token. The remaining rows are
        # the selected K-1 Additional Gaussians from every target-view decision.
        # Rows remain anchor-major, hence frame offsets keep their historical
        # contiguous-frame contract.
        additional_count = selected_k - 1
        per_anchor_additional = torch.zeros(
            anchor_feature.shape[0], dtype=torch.long,
            device=anchor_feature.device,
        )
        if additional_count.numel() > 0:
            per_anchor_additional.index_add_(
                0, decision_anchor, additional_count
            )
        per_anchor_gaussians = per_anchor_additional + 1
        gaussian_anchor = torch.repeat_interleave(
            torch.arange(anchor_feature.shape[0], device=anchor_feature.device),
            per_anchor_gaussians,
        )
        anchor_start = torch.cumsum(
            per_anchor_gaussians, dim=0
        ) - per_anchor_gaussians
        local_row = (
            torch.arange(gaussian_anchor.numel(), device=anchor_feature.device)
            - torch.repeat_interleave(anchor_start, per_anchor_gaussians)
        )
        is_common = local_row == 0
        common_rows = is_common.nonzero(as_tuple=True)[0]
        additional_rows = (~is_common).nonzero(as_tuple=True)[0]

        decision_rows = torch.arange(
            selected_k.numel(), device=anchor_feature.device
        )
        additional_decision = torch.repeat_interleave(
            decision_rows, additional_count
        )
        if additional_rows.numel() != additional_decision.numel():
            raise RuntimeError("Additional Gaussian packing lost decision alignment")

        decision_index = torch.full(
            (gaussian_anchor.numel(),), -1, dtype=torch.long,
            device=anchor_feature.device,
        )
        gaussian_view = torch.full_like(decision_index, -1)
        slot_index = torch.zeros_like(decision_index)
        slot_k = torch.ones_like(decision_index)
        additional_start = torch.cumsum(
            additional_count, dim=0
        ) - additional_count
        additional_slot = (
            torch.arange(additional_decision.numel(), device=anchor_feature.device)
            - torch.repeat_interleave(additional_start, additional_count)
            + 1
        )
        if additional_rows.numel() > 0:
            decision_index = decision_index.index_copy(
                0, additional_rows, additional_decision
            )
            gaussian_view = gaussian_view.index_copy(
                0, additional_rows, decision_view[additional_decision]
            )
            slot_index = slot_index.index_copy(
                0, additional_rows, additional_slot
            )
            slot_k = slot_k.index_copy(
                0, additional_rows, selected_k[additional_decision]
            )

        raw_params = anchor_feature.new_zeros(
            (gaussian_anchor.numel(), self.param_dim)
        )
        raw_params = raw_params.index_copy(0, common_rows, common_params)

        # Candidate K already contains [Common, Additional...]. Only the selected
        # K row enters its predictor, and K=1 executes no Additional head.
        selected_delta = delta_p[
            decision_anchor, selected_index
        ].to(dtype=anchor_feature.dtype)
        additional_delta = selected_delta[:, 1:].reshape(
            selected_k.numel(), 3 * max(self.k_max - 1, 0)
        )
        for total_k in range(2, self.k_max + 1):
            selected_rows = decision_rows[selected_k == total_k]
            if selected_rows.numel() == 0:
                continue
            additional_k = total_k - 1
            # Selected-only execution: K=1 decisions do not enter either the
            # Additional trunk or an Additional output head. The input is the
            # fused token that also produced Common, so two target views that
            # route the same token to the same K get identical Additional
            # parameters; only the count is view-dependent.
            additional_feature = self.additional_trunk(torch.cat([
                anchor_feature[decision_anchor[selected_rows]],
                additional_delta[selected_rows],
            ], dim=-1))
            head_input = self._balance_token_feature(
                additional_feature, total_k
            )
            predicted = self.additional_heads[additional_k - 1](head_input).reshape(
                -1, additional_k, self.param_dim
            )
            flat_additional = (
                additional_start[selected_rows, None]
                + torch.arange(
                    additional_k, device=anchor_feature.device
                ).view(1, -1)
            )
            output_rows = additional_rows[flat_additional]
            raw_params = raw_params.index_copy(
                0, output_rows.reshape(-1), predicted.reshape(-1, self.param_dim)
            )

        # Additional rows can be gated now because each belongs to exactly one
        # view. Common is shared, so retain one scalar gate per (token, view) and
        # apply it only when the renderer constructs that view's union. Thus K=1
        # still trains its router without duplicating Common Gaussian tensors.
        selected_gate = selection.gather(
            1, selected_index.unsqueeze(-1)
        ).squeeze(-1)
        packed_gate = raw_params.new_ones((raw_params.shape[0],))
        if additional_rows.numel() > 0:
            additional_gate = selected_gate[additional_decision]
            gated_additional = self._opacity_gate_st(
                raw_params[additional_rows], additional_gate
            )
            raw_params = raw_params.index_copy(
                0, additional_rows, gated_additional
            )
            packed_gate = packed_gate.index_copy(
                0, additional_rows, additional_gate
            )

        max_views = max(
            (int(pose_b.shape[0]) for pose_b in target_pose), default=0
        )
        common_view_gate = raw_params.new_zeros(
            (anchor_feature.shape[0] * max_views,)
        )
        if selected_gate.numel() > 0:
            common_gate_index = decision_anchor * max_views + decision_view
            common_view_gate = common_view_gate.index_copy(
                0, common_gate_index, selected_gate
            )
        common_view_gate = common_view_gate.reshape(
            anchor_feature.shape[0], max_views
        )

        gaussian_offset = self._gaussian_offset(
            per_anchor_gaussians, anchor_offset
        )
        return raw_params, {
            "anchor_index": gaussian_anchor,
            "decision_index": decision_index,
            "decision_anchor_index": decision_anchor,
            "decision_view_index": decision_view,
            "view_index": gaussian_view,
            "is_common": is_common,
            "common_view_gate": common_view_gate,
            "slot_index": slot_index,
            "slot_k": slot_k,
            "anchor_k": selected_k,
            "gaussian_offset": gaussian_offset,
            "k_logits": logits,
            "k_selection": selection,
            "packed_selected_gate": packed_gate,
            "selected_only": True,
            "view_dependent": True,
        }

    def forward(
        self, anchor_feature, anchor_k, delta_p, anchor_offset,
        target_pose=None, anchor_batch=None, anchor_position_ref=None,
    ):
        if self.count_mode == "learned_gumbel_viewpt":
            return self._forward_viewpoint(
                anchor_feature, anchor_k, delta_p, anchor_offset,
                target_pose, anchor_batch, anchor_position_ref,
            )
        if self.count_mode == "learned_gumbel":
            return self._forward_learned(
                anchor_feature, anchor_k, delta_p, anchor_offset
            )
        return self._forward_legacy(
            anchor_feature, anchor_k, delta_p, anchor_offset
        )

    def gradient_weight(self, slot_k, dtype):
        if (
            self.grad_balance == "none"
            or self.grad_balance_scope == "token"
        ):
            return torch.ones_like(slot_k, dtype=dtype)
        return slot_k.to(dtype=dtype).rsqrt()


__all__ = [
    "GridTemporalAggregator",
    "GridSlotHead",
    "ViewTokenPositionEncoder",
]
