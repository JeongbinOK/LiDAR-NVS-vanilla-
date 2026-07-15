"""Variable-count Gaussian generation for ``anchor_mode='grid'``.

Each occupied Utonia token keeps its temporal context while own-frame raw points
provide range-quantile position seeds. Background seeds live in the reference
frame and persistent foreground seeds live in box-local coordinates. A shared
anchor trunk consumes the padded seed offsets, then an independent joint head for
each possible ``K_i`` predicts all Gaussian parameters for that anchor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import boxes as box_utils
from ..utils.attention import Rotary3D


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

    def forward(self, feat, position):
        x = self.input_norm(feat)
        if feat.shape[0] == 0 or self.n_layers == 0:
            return x
        H, dh = self.num_heads, self.head_dim
        rope_angles = self.rope.angles(position)
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
            prob = torch.softmax(scores, dim=-1)
            attended = torch.einsum("hij,jhd->ihd", prob, value.float())
            attended = attended.reshape(-1, self.dim).to(x.dtype)
            x = self.norm[layer](x + self.out[layer](attended))
            x = self.norm_ffn[layer](x + self.ffn[layer](x))
        return x


class GridTemporalAggregator(nn.Module):
    """Grid-only background radius cross-attention and foreground self-attention."""

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

    @staticmethod
    def _frame_bounds(new_offset, global_frame):
        start = int(new_offset[global_frame - 1]) if global_frame > 0 else 0
        return start, int(new_offset[global_frame])

    def _token_metadata(self, anchor, seed_sensor, delta_sensor, new_offset,
                        frame_batch_idx, pose_list, bbox_list,
                        bbox_instance_ids_list, timestamps_list):
        device, dtype = anchor.device, anchor.dtype
        N = anchor.shape[0]
        if seed_sensor.shape != delta_sensor.shape:
            raise ValueError("seed_sensor and delta_sensor must have identical shapes")
        if seed_sensor.ndim != 3 or seed_sensor.shape[0] != N or seed_sensor.shape[2] != 3:
            raise ValueError("grid seeds must have shape (num_anchors, K_max, 3)")
        n_frames = int(frame_batch_idx.numel())
        token_frame = torch.zeros(N, dtype=torch.long, device=device)
        coord_ref = torch.zeros_like(anchor)
        coord_out = torch.zeros_like(anchor)
        seed_ref = torch.zeros_like(seed_sensor)
        seed_out = torch.zeros_like(seed_sensor)
        center_out = torch.zeros_like(seed_sensor)
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
            "seed_delta": seed_out - center_out,
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

        fg_rows_to_update = []
        fg_feature_updates = []
        for frame_indices in metadata["batch_frame_map"].values():
            frame_rows = torch.cat([metadata["frame_indices"][f] for f in frame_indices])
            dynamic_rows = frame_rows[metadata["is_dynamic"][frame_rows]]
            for item_id in metadata["instance_id"][dynamic_rows].unique().tolist():
                item_id = int(item_id)
                rows = dynamic_rows[metadata["instance_id"][dynamic_rows] == item_id]
                if rows.numel() == 0:
                    continue
                # Instances are disjoint and background attention never touches
                # these rows, so accumulate and scatter once instead of copying
                # the full N-token tensor once per object.
                updated = self.fg_attention(feat[rows], metadata["out"][rows])
                fg_rows_to_update.append(rows)
                fg_feature_updates.append(updated)
        if fg_rows_to_update:
            out_feat = out_feat.index_copy(
                0, torch.cat(fg_rows_to_update), torch.cat(fg_feature_updates)
            )

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


class GridSlotHead(nn.Module):
    """Predict variable-count Gaussian parameters with independent joint K heads."""

    def __init__(self, cfg, gs_params, dim):
        super().__init__()
        self.dim = int(dim)
        k_max = getattr(cfg, "K_max", None)
        if k_max is None:
            raise ValueError("p2g.grid_query.K_max is required")
        self.k_max = int(k_max)
        self.grad_balance = str(getattr(cfg, "grad_balance", "sqrt_k"))
        if self.k_max <= 0:
            raise ValueError("grid_query.K_max must be positive")
        if self.grad_balance not in ("sqrt_k", "none"):
            raise ValueError("grid_query.grad_balance must be 'sqrt_k' or 'none'")

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

        self.trunk = nn.Sequential(
            nn.Linear(self.dim + 3 * self.k_max, self.dim),
            nn.SiLU(),
        )
        self.k_heads = nn.ModuleList([
            nn.Linear(self.dim, k * self.param_dim)
            for k in range(1, self.k_max + 1)
        ])

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

    def forward(self, anchor_feature, anchor_k, delta_p, anchor_offset):
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
            predicted = head(trunk_feature[anchor_rows]).reshape(-1, k, self.param_dim)
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

    def gradient_weight(self, slot_k, dtype):
        if self.grad_balance == "none":
            return torch.ones_like(slot_k, dtype=dtype)
        return slot_k.to(dtype=dtype).rsqrt()


__all__ = [
    "GridTemporalAggregator",
    "GridSlotHead",
]
