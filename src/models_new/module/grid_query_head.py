"""Variable-count Gaussian queries for ``anchor_mode='grid'``.

The grid path keeps every occupied Utonia token as an anchor.  Temporal
aggregation updates only its feature; the anchor coordinate itself remains the
Gaussian seed (reference-frame for background, box-local for persistent
foreground).  A shared FiLM-conditioned Gaussian head then expands anchor ``i``
into ``K_i`` slots, where ``K_i`` depends only on the token's own-frame raw point
count.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..utils import boxes as box_utils
from .builders.common import encode_ray_meta, xyz_to_theta_phi_r


def counts_to_variable_k(raw_count, points_per_gaussian: int, k_max: int):
    """Map positive own-frame point counts to ``ceil(count / ppg)`` in [1, Kmax]."""
    points_per_gaussian = int(points_per_gaussian)
    k_max = int(k_max)
    if points_per_gaussian <= 0:
        raise ValueError("points_per_gaussian must be positive")
    if k_max <= 0:
        raise ValueError("K_max must be positive")
    count = raw_count.to(dtype=torch.long).clamp_min(1)
    k = torch.div(
        count + points_per_gaussian - 1,
        points_per_gaussian,
        rounding_mode="floor",
    )
    return k.clamp_(min=1, max=k_max)


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
    """One-query-per-anchor cross attention over a ragged, fixed K/V memory."""

    def __init__(self, dim, num_heads, n_layers, layer_scale, mlp_ratio=4):
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

        self.q_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        self.mem_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        self.q_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.kv_proj = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(self.n_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            for _ in range(self.n_layers)
        ])
        self.attn_scale = nn.ParameterList([
            nn.Parameter(torch.full((dim,), float(layer_scale)))
            for _ in range(self.n_layers)
        ])
        self.ffn_scale = nn.ParameterList([
            nn.Parameter(torch.full((dim,), float(layer_scale)))
            for _ in range(self.n_layers)
        ])
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    def forward(self, query_feat, pair_memory, pair_query):
        if pair_query.numel() == 0 or self.n_layers == 0:
            return query_feat

        order = torch.argsort(pair_query, stable=True)
        pair_query = pair_query[order]
        pair_memory = pair_memory[order]
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

        x = query_feat[active]
        H, dh = self.num_heads, self.head_dim
        neg_mask = ~mask[:, None, None, :]
        for layer in range(self.n_layers):
            q = self.q_proj[layer](self.q_norm[layer](x)).reshape(-1, H, dh)
            mem = self.mem_norm[layer](memory)
            key, value = self.kv_proj[layer](mem).chunk(2, dim=-1)
            key = key.reshape(active.numel(), max_len, H, dh)
            value = value.reshape(active.numel(), max_len, H, dh)
            scores = torch.einsum("ahd,alhd->ahl", q.float(), key.float()) * self.scale
            scores = scores.unsqueeze(2).masked_fill(neg_mask, float("-inf"))
            prob = torch.softmax(scores, dim=-1).squeeze(2)
            attended = torch.einsum("ahl,alhd->ahd", prob, value.float())
            attended = attended.reshape(active.numel(), self.dim).to(x.dtype)
            x = x + self.attn_scale[layer] * self.out_proj[layer](attended)
            x = x + self.ffn_scale[layer] * self.ffn[layer](self.ffn_norm[layer](x))

        # Unmatched rows are copied verbatim: exact identity in forward and backward.
        return query_feat.index_copy(0, active, x)


class _FullSelfAttention(nn.Module):
    """Full self attention for one persistent object instance."""

    def __init__(self, dim, num_heads, n_layers, layer_scale, mlp_ratio=4):
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
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(self.n_layers)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_layers)])
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
            for _ in range(self.n_layers)
        ])
        self.attn_scale = nn.ParameterList([
            nn.Parameter(torch.full((dim,), float(layer_scale)))
            for _ in range(self.n_layers)
        ])
        self.ffn_scale = nn.ParameterList([
            nn.Parameter(torch.full((dim,), float(layer_scale)))
            for _ in range(self.n_layers)
        ])
        for ffn in self.ffn:
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)

    def forward(self, feat, context):
        if feat.shape[0] == 0 or self.n_layers == 0:
            return feat
        x = feat
        H, dh = self.num_heads, self.head_dim
        for layer in range(self.n_layers):
            qkv = self.qkv[layer](self.norm[layer](x + context))
            q, key, value = qkv.chunk(3, dim=-1)
            q = q.reshape(-1, H, dh)
            key = key.reshape(-1, H, dh)
            value = value.reshape(-1, H, dh)
            scores = torch.einsum("ihd,jhd->hij", q.float(), key.float()) * self.scale
            prob = torch.softmax(scores, dim=-1)
            attended = torch.einsum("hij,jhd->ihd", prob, value.float())
            attended = attended.reshape(-1, self.dim).to(x.dtype)
            x = x + self.attn_scale[layer] * self.out[layer](attended)
            x = x + self.ffn_scale[layer] * self.ffn[layer](self.ffn_norm[layer](x))
        return x


class GridTemporalAggregator(nn.Module):
    """Grid-only background radius cross-attention and foreground self-attention."""

    def __init__(self, cfg, dim, r_far):
        super().__init__()
        self.dim = int(dim)
        self.radius = float(cfg.bg_radius_m)
        self.max_kv_per_frame = int(cfg.bg_max_kv_per_frame)
        self.r_far = float(r_far)
        num_heads = int(cfg.num_heads)
        layer_scale = float(cfg.layer_scale)

        # [refined feature, source-query xyz, source-target time, source own-ray]
        self.bg_kv_embed = nn.Sequential(
            nn.Linear(self.dim + 8, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.bg_attention = _RaggedCrossAttention(
            self.dim, num_heads, int(cfg.bg_layers), layer_scale
        )
        # Foreground positional context: [box-local xyz, timestamp, own ray4].
        self.fg_context_embed = nn.Sequential(
            nn.Linear(8, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.fg_attention = _FullSelfAttention(
            self.dim, num_heads, int(cfg.fg_layers), layer_scale
        )

    @staticmethod
    def _frame_bounds(new_offset, global_frame):
        start = int(new_offset[global_frame - 1]) if global_frame > 0 else 0
        return start, int(new_offset[global_frame])

    def _token_metadata(self, anchor, new_offset, frame_batch_idx, pose_list,
                        bbox_list, bbox_instance_ids_list, timestamps_list):
        device, dtype = anchor.device, anchor.dtype
        N = anchor.shape[0]
        n_frames = int(frame_batch_idx.numel())
        token_frame = torch.zeros(N, dtype=torch.long, device=device)
        token_time = torch.zeros(N, dtype=dtype, device=device)
        coord_ref = torch.zeros_like(anchor)
        coord_out = torch.zeros_like(anchor)
        is_dynamic = torch.zeros(N, dtype=torch.bool, device=device)
        instance_id = torch.full((N,), -1, dtype=torch.long, device=device)
        box_assign = torch.full((N,), -1, dtype=torch.long, device=device)
        ray4 = encode_ray_meta(xyz_to_theta_phi_r(anchor), self.r_far)
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
            if timestamps_list is not None:
                timestamps_b = timestamps_list[batch_id].to(device=device, dtype=dtype)
            else:
                denom = max(len(frame_indices) - 1, 1)
                timestamps_b = torch.arange(
                    len(frame_indices), device=device, dtype=dtype
                ) / float(denom)

            for local_f, global_f in enumerate(frame_indices):
                start, end = self._frame_bounds(new_offset, global_f)
                rows = torch.arange(start, end, device=device)
                frame_indices_out[global_f] = rows
                token_frame[start:end] = global_f
                token_time[start:end] = timestamps_b[local_f]

                anchor_f = anchor[start:end]
                pose_f = pose_b[local_f].to(device=device, dtype=dtype)
                ref_f = box_utils.apply_pose(anchor_f, pose_f)
                coord_ref[start:end] = ref_f
                coord_out[start:end] = ref_f

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

        return {
            "frame": token_frame,
            "time": token_time,
            "ref": coord_ref,
            "out": coord_out,
            "is_dynamic": is_dynamic,
            "instance_id": instance_id,
            "box_assign": box_assign,
            "ray4": ray4,
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

    def forward(self, feat, anchor, new_offset, frame_batch_idx, pose_list,
                bbox_list, bbox_instance_ids_list=None, timestamps_list=None):
        metadata = self._token_metadata(
            anchor, new_offset, frame_batch_idx, pose_list, bbox_list,
            bbox_instance_ids_list, timestamps_list,
        )
        out_feat = feat
        pair_query, pair_source = self._background_pairs(metadata)
        if pair_query.numel() > 0:
            delta = metadata["out"][pair_source] - metadata["out"][pair_query]
            delta_t = metadata["time"][pair_source] - metadata["time"][pair_query]
            kv_input = torch.cat([
                feat[pair_source], delta, delta_t.unsqueeze(-1),
                metadata["ray4"][pair_source],
            ], dim=-1)
            pair_memory = self.bg_kv_embed(kv_input)
            out_feat = self.bg_attention(out_feat, pair_memory, pair_query)

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
                context_input = torch.cat([
                    metadata["out"][rows], metadata["time"][rows, None],
                    metadata["ray4"][rows],
                ], dim=-1)
                context = self.fg_context_embed(context_input)
                # Instances are disjoint and background attention never touches
                # these rows, so accumulate and scatter once instead of copying
                # the full N-token tensor once per object.
                updated = self.fg_attention(feat[rows], context)
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
            "bbox_ref_by_frame": metadata["bbox_ref_by_frame"],
        }
        return out_feat, metadata["out"], agg_meta


class GridSlotHead(nn.Module):
    """Expand anchors into variable slots and FiLM-condition one shared GS head."""

    def __init__(self, cfg, dim, r_far):
        super().__init__()
        self.dim = int(dim)
        self.k_max = int(cfg.K_max)
        self.points_per_gaussian = int(cfg.points_per_gaussian)
        self.film_scale = float(cfg.film_scale)
        self.count_cap = float(cfg.count_condition_cap)
        self.r_far = float(r_far)
        self.grad_balance = str(getattr(cfg, "grad_balance", "sqrt_k"))
        if self.grad_balance not in ("sqrt_k", "none"):
            raise ValueError("grid_query.grad_balance must be 'sqrt_k' or 'none'")
        if self.count_cap <= 0:
            raise ValueError("grid_query.count_condition_cap must be positive")

        self.anchor_norm = nn.LayerNorm(self.dim)
        # Fourier(u): sin/cos at 1x and 2x frequencies + log-count + range + K/Kmax.
        self.conditioner = nn.Sequential(
            nn.Linear(7, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, 2 * self.dim),
        )
        nn.init.normal_(self.conditioner[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.conditioner[-1].bias)

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

    def forward(self, anchor_feature, raw_count, sensor_range, anchor_offset):
        if raw_count.shape[0] != anchor_feature.shape[0]:
            raise ValueError("raw_count and anchor_feature must have the same first dimension")
        anchor_k = counts_to_variable_k(
            raw_count, self.points_per_gaussian, self.k_max
        )
        anchor_index = torch.repeat_interleave(
            torch.arange(anchor_feature.shape[0], device=anchor_feature.device), anchor_k
        )
        gaussian_offset = self._gaussian_offset(anchor_k, anchor_offset)
        if anchor_index.numel() == 0:
            empty_long = torch.zeros(0, dtype=torch.long, device=anchor_feature.device)
            return anchor_feature.new_zeros((0, self.dim)), {
                "anchor_index": empty_long,
                "slot_index": empty_long,
                "slot_k": empty_long,
                "anchor_k": anchor_k,
                "slot_u": anchor_feature.new_zeros(0),
                "conditioner_input": anchor_feature.new_zeros((0, 7)),
                "gaussian_offset": gaussian_offset,
            }

        slot_k = anchor_k[anchor_index]
        slot_start = torch.cumsum(anchor_k, dim=0) - anchor_k
        slot_index = torch.arange(anchor_index.numel(), device=anchor_feature.device) - \
            torch.repeat_interleave(slot_start, anchor_k)
        slot_u = (slot_index.to(anchor_feature.dtype) + 0.5) / slot_k.to(anchor_feature.dtype)

        two_pi_u = 2.0 * math.pi * slot_u
        four_pi_u = 2.0 * two_pi_u
        raw_slot_count = raw_count[anchor_index].to(anchor_feature.dtype)
        count_condition = torch.log1p(raw_slot_count.clamp(max=self.count_cap)) / \
            math.log1p(self.count_cap)
        range_condition = (
            sensor_range[anchor_index].to(anchor_feature.dtype).clamp(min=0.0, max=self.r_far)
            / self.r_far
        )
        k_condition = slot_k.to(anchor_feature.dtype) / float(self.k_max)
        conditioner_input = torch.stack([
            torch.sin(two_pi_u), torch.cos(two_pi_u),
            torch.sin(four_pi_u), torch.cos(four_pi_u),
            count_condition, range_condition, k_condition,
        ], dim=-1)
        gamma_beta = self.conditioner(conditioner_input)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = self.film_scale * torch.tanh(gamma)
        beta = self.film_scale * torch.tanh(beta)
        normalized = self.anchor_norm(anchor_feature)[anchor_index]
        slot_feature = (1.0 + gamma) * normalized + beta
        return slot_feature, {
            "anchor_index": anchor_index,
            "slot_index": slot_index,
            "slot_k": slot_k,
            "anchor_k": anchor_k,
            "slot_u": slot_u,
            "conditioner_input": conditioner_input,
            "gaussian_offset": gaussian_offset,
        }

    def gradient_weight(self, slot_k, dtype):
        if self.grad_balance == "none":
            return torch.ones_like(slot_k, dtype=dtype)
        return slot_k.to(dtype=dtype).rsqrt()


__all__ = [
    "GridTemporalAggregator",
    "GridSlotHead",
    "counts_to_variable_k",
]
