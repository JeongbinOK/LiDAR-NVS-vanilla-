"""Stateless torch box/pose geometry shared by both anchor modes.

``box_local_to_ref`` mirrors the transform in ``GausRender``.  Keeping these
operations here gives spherical and grid routing one coordinate convention.
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


def points_to_box_local_batched(points_ref: Tensor, box_ref: Tensor) -> Tensor:
    """Vectorized ``points_to_box_local`` with one box PER point.

    Same yaw-aligned local transform as ``points_to_box_local`` but each point
    carries its own box, so a caller can transform points assigned to many
    different boxes in a single launch instead of looping one call per box.

    points_ref : (N, 3)
    box_ref    : (N, 7) [x,y,z,w,l,h,yaw], the box each point belongs to.
    return     : (N, 3), elementwise-identical to looping ``points_to_box_local``
                 over per-box groups.
    """
    center = box_ref[:, :3]
    yaw = box_ref[:, 6]
    shifted = points_ref - center
    cos_y = torch.cos(-yaw)
    sin_y = torch.sin(-yaw)
    local_x = cos_y * shifted[:, 0] - sin_y * shifted[:, 1]
    local_y = sin_y * shifted[:, 0] + cos_y * shifted[:, 1]
    return torch.stack([local_x, local_y, shifted[:, 2]], dim=-1)


def box_local_to_ref(points_local: Tensor, box_ref: Tensor) -> Tensor:
    """Inverse of `points_to_box_local`: box-local (N, 3) points -> box_ref's frame.

    Same math as `GausRender.box_local_to_ref` (module/m3_g2p.py); m3's copy is
    left untouched; this is the shared inverse transform used by anchor heads.

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


def box_local_to_ref_multi(points_local: Tensor, boxes_ref: Tensor) -> Tensor:
    """Replay one set of box-local points through V poses of the same box.

    points_local : (N, 3) in the box's yaw-aligned local frame.
    boxes_ref    : (V, 7), the same instance's box at V different times.
    return       : (N, V, 3), row-wise identical to looping ``box_local_to_ref``
                   once per box.
    """
    yaw = boxes_ref[:, 6]
    cos_y = torch.cos(yaw).unsqueeze(0)
    sin_y = torch.sin(yaw).unsqueeze(0)
    local_x = points_local[:, 0].unsqueeze(-1)
    local_y = points_local[:, 1].unsqueeze(-1)
    x = cos_y * local_x - sin_y * local_y
    y = sin_y * local_x + cos_y * local_y
    z = points_local[:, 2].unsqueeze(-1).expand_as(x)
    return torch.stack([x, y, z], dim=-1) + boxes_ref[:, :3].unsqueeze(0)


def interpolate_boxes_ref(times: Tensor, boxes: Tensor, query: Tensor) -> Tensor:
    """Sample a ref-frame box trajectory at V query times.

    Elementwise identical to ``GausRender.interpolate_box_ref`` -- clamped at
    both ends, linear in center/size and shortest-arc in yaw -- but evaluated
    for every query time at once. Anything that needs to predict where a dynamic
    Gaussian will be rendered must use the renderer's own interpolation, so this
    equivalence is asserted in the tests rather than merely intended.

    times : (F,) ascending; boxes : (F, 7); query : (V,)
    """
    if times.ndim != 1 or boxes.ndim != 2 or boxes.shape[-1] != 7:
        raise ValueError("interpolate_boxes_ref expects (F,) times and (F, 7) boxes")
    if times.shape[0] != boxes.shape[0] or times.shape[0] < 2:
        raise ValueError("a box trajectory needs at least two aligned samples")
    hi = torch.searchsorted(times, query).clamp(1, times.shape[0] - 1)
    lo = hi - 1
    span = (times[hi] - times[lo]).clamp_min(1e-6)
    # Clamping alpha reproduces the renderer's two out-of-range early returns.
    alpha = ((query - times[lo]) / span).clamp(0.0, 1.0).unsqueeze(-1)
    center_size = boxes[lo, :6] * (1.0 - alpha) + boxes[hi, :6] * alpha
    yaw0, yaw1 = boxes[lo, 6], boxes[hi, 6]
    delta = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))
    yaw = yaw0 + alpha.squeeze(-1) * delta
    return torch.cat([center_size, yaw.unsqueeze(-1)], dim=-1)


def point_in_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Assign each of (N, 3) points to the box it falls inside (or -1).

    points : (N, 3)
    boxes  : (B_f, 7) x,y,z,w,l,h,yaw, same frame as points.
    return : (N,) long, -1 = background, 0..B_f-1 = box index. Ties (point
             inside more than one box) resolve to the lowest box index.

    Size convention: columns 3:6 are nuScenes `size` = (width, length, height)
    and yaw rotates the box's LENGTH axis onto local +x (nuScenes `Box.corners`
    builds x from l and y from w). So local x is tested against l/2 and local y
    against w/2 -- testing x against w/2 rotates every non-square box by 90 deg
    and was measured to capture 2.2x fewer points on seq_1250_1300.
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
        (local_x.abs() <= l.unsqueeze(0) / 2) &   # local +x spans the LENGTH
        (local_y.abs() <= w.unsqueeze(0) / 2) &   # local +y spans the WIDTH
        (dz.abs()      <= h.unsqueeze(0) / 2)
    )  # (N, B_f)

    # ``argmax`` returns the first maximum, preserving the existing lowest-box
    # tie break without launching one masked-assignment kernel per box.  Rows
    # with no match need an explicit guard because argmax would return 0 there.
    has_match = inside.any(dim=1)
    first_match = inside.to(dtype=torch.uint8).argmax(dim=1).long()
    return torch.where(has_match, first_match, torch.full_like(first_match, -1))


def common_instance_ids(first_ids, last_ids) -> set:
    """Intersection of instance ids present at both trajectory endpoints.

    first_ids, last_ids : 1D long Tensor or None.
    return : python set[int], empty if either side is None. An instance only
             counts as "dynamic" if it appears at both endpoints.
    """
    if first_ids is None or last_ids is None:
        return set()
    first = {int(x) for x in first_ids.tolist()}
    last = {int(x) for x in last_ids.tolist()}
    return first & last
