"""Pure torch box/pose geometry helpers, shared across anchor_mode paths.

Extracted verbatim (same math) from two call sites so the logic lives in one
place instead of being duplicated by the upcoming spherical-query anchor path
(Step 3 of the spherical-query redesign):
  - `apply_pose`, `yaw_from_pose`, `transform_boxes_to_ref`, `points_to_box_local`,
    `point_in_box`, `common_instance_ids` were TimeAgg methods in
    `module/m1_p2g.py`; TimeAgg now delegates to these (thin wrappers), so its
    method names/signatures are unchanged and behavior is bit-identical.
  - `box_local_to_ref` mirrors `GausRender.box_local_to_ref` in
    `module/m3_g2p.py` (read-only reference; m3 itself still defines its own
    copy and is not modified).

No torch.nn.Module state anywhere here -- every function is stateless so it
can be called directly by the new spherical-query head without going through
TimeAgg.
"""
from __future__ import annotations

import torch
from torch import Tensor


def apply_pose(points: Tensor, pose: Tensor) -> Tensor:
    """Apply a 4x4 homogeneous pose to (N, 3) points.

    points : (N, 3)
    pose   : (4, 4)  e.g. frame_i -> frame_0
    return : (N, 3) = (pose @ [points; 1])[:, :3]
    """
    N = points.shape[0]
    ones = torch.ones(N, 1, device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)   # (N, 4)
    return (pose @ pts_h.T).T[:, :3]            # (N, 3)


def yaw_from_pose(pose: Tensor) -> Tensor:
    """Yaw (rotation about z) implied by a 4x4 pose's rotation block."""
    return torch.atan2(pose[1, 0], pose[0, 0])


def transform_boxes_to_ref(boxes: Tensor, pose: Tensor) -> Tensor:
    """Transform (B_f, 7) [x,y,z,w,l,h,yaw] boxes into the pose's target frame.

    boxes : (B_f, 7) sensor-frame boxes (empty B_f=0 is a no-op passthrough).
    pose  : (4, 4) e.g. frame_i -> frame_0.
    """
    if boxes.shape[0] == 0:
        return boxes
    out = boxes.clone()
    out[:, :3] = apply_pose(boxes[:, :3], pose)
    out[:, 6] = boxes[:, 6] + yaw_from_pose(pose)
    return out


def points_to_box_local(points_ref: Tensor, box_ref: Tensor) -> Tensor:
    """Rotate/translate (N, 3) points into a single box's yaw-aligned local frame.

    points_ref : (N, 3), same frame as box_ref.
    box_ref    : (7,) [x,y,z,w,l,h,yaw].
    """
    center = box_ref[:3]
    yaw = box_ref[6]
    shifted = points_ref - center.unsqueeze(0)
    cos_y = torch.cos(-yaw)
    sin_y = torch.sin(-yaw)
    local_x = cos_y * shifted[:, 0] - sin_y * shifted[:, 1]
    local_y = sin_y * shifted[:, 0] + cos_y * shifted[:, 1]
    return torch.stack([local_x, local_y, shifted[:, 2]], dim=-1)


def box_local_to_ref(points_local: Tensor, box_ref: Tensor) -> Tensor:
    """Inverse of `points_to_box_local`: box-local (N, 3) points -> box_ref's frame.

    Same math as `GausRender.box_local_to_ref` (module/m3_g2p.py); m3's copy is
    left untouched, this is the shared implementation Step 3 will also call.

    points_local : (N, 3), in box_ref's local (yaw-aligned) frame.
    box_ref      : (7,) [x,y,z,w,l,h,yaw].
    """
    yaw = box_ref[6]
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    x = cos_y * points_local[:, 0] - sin_y * points_local[:, 1]
    y = sin_y * points_local[:, 0] + cos_y * points_local[:, 1]
    z = points_local[:, 2]
    return torch.stack([x, y, z], dim=-1) + box_ref[:3].unsqueeze(0)


def point_in_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Assign each of (N, 3) points to the box it falls inside (or -1).

    points : (N, 3)
    boxes  : (B_f, 7) x,y,z,w,l,h,yaw, same frame as points.
    return : (N,) long, -1 = background, 0..B_f-1 = box index. Ties (point
             inside more than one box) resolve to the lowest box index.
    """
    N, B_f = points.shape[0], boxes.shape[0]
    if B_f == 0:
        return torch.full((N,), -1, dtype=torch.long, device=points.device)

    cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    w, l, h    = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    yaw        = boxes[:, 6]

    dx = points[:, 0].unsqueeze(1) - cx.unsqueeze(0)  # (N, B_f)
    dy = points[:, 1].unsqueeze(1) - cy.unsqueeze(0)
    dz = points[:, 2].unsqueeze(1) - cz.unsqueeze(0)

    cos_y  = torch.cos(-yaw).unsqueeze(0)
    sin_y  = torch.sin(-yaw).unsqueeze(0)
    local_x = cos_y * dx - sin_y * dy
    local_y = sin_y * dx + cos_y * dy

    inside = (
        (local_x.abs() <= w.unsqueeze(0) / 2) &
        (local_y.abs() <= l.unsqueeze(0) / 2) &
        (dz.abs()      <= h.unsqueeze(0) / 2)
    )  # (N, B_f)

    box_idx = torch.full((N,), -1, dtype=torch.long, device=points.device)
    for b in range(B_f):
        mask = inside[:, b] & (box_idx == -1)
        box_idx[mask] = b
    return box_idx


def common_instance_ids(first_ids, last_ids) -> set:
    """Intersection of instance ids present at both trajectory endpoints.

    first_ids, last_ids : 1D long Tensor or None.
    return : python set[int], empty if either side is None. An instance only
             counts as "dynamic" if it appears at both endpoints (matches the
             TimeAgg is_dynamic definition).
    """
    if first_ids is None or last_ids is None:
        return set()
    first = {int(x) for x in first_ids.tolist()}
    last = {int(x) for x in last_ids.tolist()}
    return first & last
