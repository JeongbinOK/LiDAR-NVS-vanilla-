"""Shared conversion from flat Gaussian predictions to downstream batch data."""
from __future__ import annotations

import torch

from ..utils import boxes as box_utils


def gradient_scale_identity(x, scale):
    """Preserve forward values while scaling each row's upstream gradient."""
    scale = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
    while scale.ndim < x.ndim:
        scale = scale.unsqueeze(-1)
    detached = x.detach()
    return detached + scale * (x - detached)


def opacity_gate_st(raw_opacity, gate):
    """Apply a hard-ST gate in activated-opacity space.

    The rasterizer applies ``sigmoid`` to raw opacity logits.  A selected hard
    gate therefore has forward value one, while its soft backward derivative
    must be applied after that sigmoid.  Keeping this helper shared by the grid
    head and renderer makes the deferred Common gate exactly match the
    Additional gate applied during Gaussian prediction.
    """
    if raw_opacity.shape[0] == 0:
        return raw_opacity
    work_opacity = raw_opacity.float()
    work_gate = gate.to(device=raw_opacity.device, dtype=torch.float32)
    while work_gate.ndim < work_opacity.ndim:
        work_gate = work_gate.unsqueeze(-1)
    activated = torch.sigmoid(work_opacity)
    gated_activated = (activated * work_gate).clamp(
        min=1.0e-6, max=1.0 - 1.0e-6
    )
    gated_logit = torch.logit(gated_activated).to(raw_opacity.dtype)
    return raw_opacity.detach() + gated_logit - gated_logit.detach()


def assemble_batch_gaussians(
    gs_raw, out_coord, agg_meta, frame_offset, frame_batch_ids,
    frame_bboxes, frame_bbox_instance_ids, has_bbox_instance_ids, device,
):
    """Split flat Gaussian tensors into the per-batch m2/m3 contract."""
    num_batches = max(frame_batch_ids) + 1
    batch_frame_map = {}
    for global_frame, batch_id in enumerate(frame_batch_ids):
        batch_frame_map.setdefault(batch_id, []).append(global_frame)

    batch_gaussians = []
    for batch_id in range(num_batches):
        frame_indices = batch_frame_map[batch_id]
        start = int(frame_offset[frame_indices[0] - 1]) if frame_indices[0] > 0 else 0
        end = int(frame_offset[frame_indices[-1]])
        batch_indices = torch.arange(start, end, device=device)

        box_assign = agg_meta["box_assign"][batch_indices]
        instance_id = agg_meta["instance_id"][batch_indices]
        is_dynamic = agg_meta["is_dynamic"][batch_indices]
        batch_item = {key: value[batch_indices] for key, value in gs_raw.items()}
        batch_item.update({
            "position": out_coord[batch_indices],
            "box_assign": box_assign,
            "coord": out_coord[batch_indices],
            "coord_ref": agg_meta["coord_ref"][batch_indices],
            "instance_id": instance_id,
            "is_dynamic": is_dynamic,
            "bg_mask": ~is_dynamic,
            "fg_masks": {
                int(item_id): (instance_id == item_id) & is_dynamic
                for item_id in instance_id[is_dynamic].unique().tolist()
            },
            "frame_bboxes": [],
        })
        if "view_index" in agg_meta:
            batch_item["view_index"] = agg_meta["view_index"][batch_indices]
        if "common_view_gate" in agg_meta:
            if "anchor_index" not in agg_meta or "view_index" not in agg_meta:
                raise KeyError(
                    "common_view_gate requires Gaussian-aligned anchor_index "
                    "and view_index metadata"
                )
            common_mask = agg_meta["view_index"][batch_indices] == -1
            common_anchor = agg_meta["anchor_index"][batch_indices][common_mask]
            batch_item["common_view_gate"] = agg_meta[
                "common_view_gate"
            ][common_anchor]

        for global_frame in frame_indices:
            bbox = frame_bboxes[global_frame]
            bbox_instance_ids = (
                frame_bbox_instance_ids[global_frame]
                if has_bbox_instance_ids else torch.arange(
                    bbox.shape[0], device=device, dtype=torch.long
                )
            )
            batch_item["frame_bboxes"].append({
                "center": bbox[:, :3],
                "size": bbox[:, 3:6],
                "yaw": bbox[:, 6],
                "bbox": bbox,
                "bbox_ref": agg_meta["bbox_ref_by_frame"][global_frame],
                "instance_id": bbox_instance_ids.to(device=device),
            })
        batch_gaussians.append(batch_item)
    return batch_gaussians


def refresh_coord_ref_after_offset(out_coord, agg_meta, frame_offset):
    """Rebuild reference-frame coordinates after the learned position offset."""
    device = out_coord.device
    dtype = out_coord.dtype
    old_coord_ref = agg_meta.get("coord_ref")
    coord_ref = (
        old_coord_ref.to(device=device, dtype=dtype).clone()
        if old_coord_ref is not None else out_coord.clone()
    )
    is_dynamic = agg_meta["is_dynamic"].to(device=device)
    box_assign = agg_meta["box_assign"].to(device=device)
    bbox_ref_by_frame = agg_meta["bbox_ref_by_frame"]
    coord_ref[~is_dynamic] = out_coord[~is_dynamic]

    for global_frame in range(int(frame_offset.numel())):
        start = int(frame_offset[global_frame - 1]) if global_frame > 0 else 0
        end = int(frame_offset[global_frame])
        if end <= start:
            continue
        dynamic_local = is_dynamic[start:end] & (box_assign[start:end] >= 0)
        bbox_ref = bbox_ref_by_frame[global_frame]
        if not dynamic_local.any() or bbox_ref is None or bbox_ref.shape[0] == 0:
            continue
        bbox_ref = bbox_ref.to(device=device, dtype=dtype)
        rows = dynamic_local.nonzero(as_tuple=True)[0] + start
        row_boxes = box_assign[rows]
        for box_index_tensor in row_boxes.unique():
            box_index = int(box_index_tensor)
            if box_index < 0 or box_index >= bbox_ref.shape[0]:
                continue
            selected = rows[row_boxes == box_index]
            coord_ref[selected] = box_utils.box_local_to_ref(
                out_coord[selected], bbox_ref[box_index]
            )

    refreshed = dict(agg_meta)
    refreshed["coord_ref"] = coord_ref
    return refreshed


__all__ = [
    "assemble_batch_gaussians",
    "gradient_scale_identity",
    "opacity_gate_st",
    "refresh_coord_ref_after_offset",
]
