"""Legacy grid/anchor temporal aggregation with dynamic-object routing."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import boxes as box_utils
from ..utils.attention import LocalAttentionFlash


class TimeAgg(nn.Module):
    """Aggregate background globally and foreground per persistent instance."""

    def __init__(self, dim, num_heads=8, k_bg=8, k_fg=16, attn_drop=0.0):
        super().__init__()
        self.k_bg = k_bg
        self.k_fg = k_fg
        self.bg_attn = LocalAttentionFlash(dim, num_heads, attn_drop)
        self.fg_attn = LocalAttentionFlash(dim, num_heads, attn_drop)

    def forward(self, feat, anchor, new_offset, frame_batch_idx,
                pose_list, bbox_list, bbox_instance_ids_list=None):
        """Aggregate per-frame sensor coordinates in reference/object frames."""
        device = feat.device
        num_tokens = feat.shape[0]
        out_feat = torch.zeros_like(feat)
        out_coord = torch.zeros_like(anchor)
        box_assign_out = torch.full(
            (num_tokens,), -1, dtype=torch.long, device=device
        )
        instance_id_out = torch.full(
            (num_tokens,), -1, dtype=torch.long, device=device
        )
        is_dynamic_out = torch.zeros(
            (num_tokens,), dtype=torch.bool, device=device
        )
        coord_ref_out = torch.zeros_like(anchor)
        bbox_ref_by_frame = [None for _ in range(len(frame_batch_idx))]

        batch_frame_map = {}
        for global_f, batch_id in enumerate(frame_batch_idx.tolist()):
            batch_frame_map.setdefault(int(batch_id), []).append(global_f)

        for batch_id, frame_indices in batch_frame_map.items():
            pose_b = pose_list[batch_id]
            bbox_b = bbox_list[batch_id]
            bbox_iids_b = (
                bbox_instance_ids_list[batch_id]
                if bbox_instance_ids_list is not None else None
            )
            if bbox_iids_b is not None and len(bbox_iids_b) >= 2:
                common_ids = box_utils.common_instance_ids(
                    bbox_iids_b[0], bbox_iids_b[-1]
                )
            else:
                common_ids = set()

            frame_feats = []
            frame_coords = []
            frame_attn_coords = []
            frame_global_indices = []
            frame_is_dynamic = []
            frame_box_assign = []
            frame_instance_id = []

            for local_f, global_f in enumerate(frame_indices):
                start = int(new_offset[global_f - 1]) if global_f > 0 else 0
                end = int(new_offset[global_f])
                global_indices = torch.arange(start, end, device=device)

                feat_f = feat[start:end]
                anchor_f = anchor[start:end]
                pose_f = pose_b[local_f].to(device)
                coord_ref_f = box_utils.apply_pose(anchor_f, pose_f)
                bbox_sensor_f = bbox_b[local_f].to(device)
                bbox_ref_f = box_utils.transform_boxes_to_ref(
                    bbox_sensor_f, pose_f
                )
                bbox_ref_by_frame[global_f] = bbox_ref_f

                if bbox_iids_b is not None:
                    bbox_iids_f = bbox_iids_b[local_f].to(device)
                else:
                    bbox_iids_f = torch.arange(
                        bbox_sensor_f.shape[0], device=device, dtype=torch.long
                    )

                box_assign_f = box_utils.point_in_box(anchor_f, bbox_sensor_f)
                instance_id_f = torch.full(
                    (anchor_f.shape[0],), -1, dtype=torch.long, device=device
                )
                valid_box = box_assign_f >= 0
                if valid_box.any() and bbox_iids_f.numel() > 0:
                    instance_id_f[valid_box] = bbox_iids_f[
                        box_assign_f[valid_box]
                    ]

                is_dynamic_f = torch.zeros_like(valid_box)
                for instance_id in common_ids:
                    is_dynamic_f |= instance_id_f == int(instance_id)

                attn_coord_f = coord_ref_f.clone()
                dynamic_boxes = box_assign_f[
                    is_dynamic_f & (box_assign_f >= 0)
                ].unique()
                for box_idx in dynamic_boxes.tolist():
                    box_idx = int(box_idx)
                    mask = is_dynamic_f & (box_assign_f == box_idx)
                    if mask.any():
                        attn_coord_f[mask] = box_utils.points_to_box_local(
                            coord_ref_f[mask], bbox_ref_f[box_idx]
                        )

                frame_feats.append(feat_f)
                frame_coords.append(coord_ref_f)
                frame_attn_coords.append(attn_coord_f)
                frame_global_indices.append(global_indices)
                frame_is_dynamic.append(is_dynamic_f)
                frame_box_assign.append(box_assign_f)
                frame_instance_id.append(instance_id_f)

            all_feat = torch.cat(frame_feats, dim=0)
            all_coord_ref = torch.cat(frame_coords, dim=0)
            all_coord_attn = torch.cat(frame_attn_coords, dim=0)
            all_global_indices = torch.cat(frame_global_indices, dim=0)
            all_dynamic = torch.cat(frame_is_dynamic, dim=0)
            all_box_assign = torch.cat(frame_box_assign, dim=0)
            all_instance_id = torch.cat(frame_instance_id, dim=0)

            coord_ref_out[all_global_indices] = all_coord_ref
            is_dynamic_out[all_global_indices] = all_dynamic
            instance_id_out[all_global_indices] = all_instance_id
            box_assign_out[all_global_indices] = torch.where(
                all_dynamic,
                all_box_assign,
                torch.full_like(all_box_assign, -1),
            )

            background = ~all_dynamic
            if background.any():
                background_indices = all_global_indices[background]
                background_coord = all_coord_attn[background]
                out_feat[background_indices] = self.bg_attn(
                    all_feat[background], background_coord, k=self.k_bg
                )
                out_coord[background_indices] = background_coord

            dynamic_instance_ids = all_instance_id[all_dynamic].unique()
            for instance_id in dynamic_instance_ids.tolist():
                instance_id = int(instance_id)
                foreground = all_dynamic & (all_instance_id == instance_id)
                foreground_indices = all_global_indices[foreground]
                foreground_coord = all_coord_attn[foreground]
                out_feat[foreground_indices] = self.fg_attn(
                    all_feat[foreground], foreground_coord, k=self.k_fg
                )
                out_coord[foreground_indices] = foreground_coord

        meta = {
            "box_assign": box_assign_out,
            "instance_id": instance_id_out,
            "is_dynamic": is_dynamic_out,
            "coord_ref": coord_ref_out,
            "bbox_ref_by_frame": bbox_ref_by_frame,
        }
        return out_feat, out_coord, meta


__all__ = ["TimeAgg"]
