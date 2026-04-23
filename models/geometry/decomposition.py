"""
Scene decomposition: static / dynamic split + canonical local transform.

Strategy (from CLAUDE.md and plan):
- Tracked instances (present in BOTH frames) -> dynamic pipeline
- Untracked / single-frame instances -> folded into static (Q3)
- Static: all points NOT in any bbox, merged in frame-0 via rel_input_1_pose
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _yaw_to_rotmat(yaw: Tensor) -> Tensor:
    """
    Convert scalar yaw angles (rotation around Z axis) to 3x3 rotation matrices.

    Args:
        yaw: [...] float tensor of yaw angles (radians).

    Returns:
        [..., 3, 3] rotation matrices R_z(yaw).
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros_like(c)
    one  = torch.ones_like(c)
    # Row-major layout: R[i,j] means row i, col j
    # R_z = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    R = torch.stack([
        torch.stack([ c, -s, zero], dim=-1),
        torch.stack([ s,  c, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)   # [..., 3, 3]
    return R


def _points_in_box(pts: Tensor, box: Tensor) -> Tensor:
    """
    Test which points lie inside a 3-D oriented bounding box.

    Args:
        pts: [N, 3]   points in the same frame as the box.
        box: [7]      (x, y, z, w, l, h, yaw) in nuScenes wlh convention.
                      - center: (x, y, z)
                      - w = width (y-extent in vehicle frame)
                      - l = length (x-extent in vehicle frame)
                      - h = height (z-extent)
                      - yaw: rotation around Z axis (radians)

    Returns:
        [N] bool mask, True if point is inside the box.
    """
    cx, cy, cz = box[0], box[1], box[2]
    w, l, h    = box[3], box[4], box[5]
    yaw        = box[6]

    # Translate to box centre
    d = pts - torch.stack([cx, cy, cz]).to(pts)  # [N, 3]

    # Rotate into box-local frame (undo yaw)
    R_inv = _yaw_to_rotmat(-yaw)   # [3, 3]
    d_local = (R_inv @ d.T).T      # [N, 3]

    # Half-extents: nuScenes wlh -> l=x, w=y, h=z in vehicle frame
    half_l = l * 0.5
    half_w = w * 0.5
    half_h = h * 0.5

    inside = (
        (d_local[:, 0].abs() <= half_l) &
        (d_local[:, 1].abs() <= half_w) &
        (d_local[:, 2].abs() <= half_h)
    )
    return inside


def _transform_points(pts: Tensor, T: Tensor) -> Tensor:
    """
    Apply 4x4 homogeneous transform to points.

    Args:
        pts: [N, 3]
        T:   [4, 4]

    Returns:
        [N, 3]
    """
    N = pts.shape[0]
    pts_h = torch.cat([pts, torch.ones(N, 1, dtype=pts.dtype, device=pts.device)], dim=1)  # [N, 4]
    out = (T @ pts_h.T).T   # [N, 4]
    return out[:, :3]


def _transform_box_to_frame0(box: Tensor, T: Tensor) -> Tensor:
    """
    Transform a bbox (in frame-1 LiDAR) into frame-0 LiDAR using T = T_{1->0}.

    The centre is transformed with the full SE(3); the yaw is updated by the
    rotation component's yaw (z-rotation extracted from the 3x3 block).

    Args:
        box: [7]  (x,y,z, w,l,h, yaw) in frame-1 coords.
        T:   [4,4] frame-1 -> frame-0 transform.

    Returns:
        [7]  box in frame-0 coords.
    """
    # Transform centre
    centre = box[:3].unsqueeze(0)   # [1, 3]
    centre_0 = _transform_points(centre, T)[0]   # [3]

    # Extract yaw delta from rotation matrix (yaw around Z)
    R = T[:3, :3]
    yaw_delta = torch.atan2(R[1, 0], R[0, 0])
    new_yaw = box[6] + yaw_delta

    return torch.cat([centre_0, box[3:6], new_yaw.unsqueeze(0)], dim=0)


def _to_canonical(pts: Tensor, centre: Tensor, yaw: Tensor) -> Tensor:
    """
    Transform points from LiDAR frame to canonical (object-centred, forward-x).

    p_canon = R(-yaw) @ (p - centre)

    Args:
        pts:    [N, 3]
        centre: [3]
        yaw:    scalar Tensor

    Returns:
        [N, 3]
    """
    p_centred = pts - centre.to(pts)
    R_inv = _yaw_to_rotmat(-yaw)   # [3, 3]
    return (R_inv @ p_centred.T).T


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decompose_scene(
    p0: Tensor,
    p1: Tensor,
    intensity_0: Tensor,
    intensity_1: Tensor,
    boxes_0: Tensor,
    boxes_1: Tensor,
    instance_ids_0: Tensor,
    instance_ids_1: Tensor,
    rel_input_1_pose: Tensor,
) -> dict:
    """
    Split scene into static background and tracked dynamic instances.

    See module docstring and CLAUDE.md for full design rationale.

    Args:
        p0:                [N0, 3]  frame-0 LiDAR points (in frame-0 sensor frame).
        p1:                [N1, 3]  frame-1 LiDAR points (in frame-1 sensor frame).
        intensity_0:       [N0]
        intensity_1:       [N1]
        boxes_0:           [B0, 7]  (x,y,z,w,l,h,yaw) in frame-0 LiDAR.
        boxes_1:           [B1, 7]  in frame-1 LiDAR.
        instance_ids_0:    [B0] long — instance IDs for each box in frame-0.
        instance_ids_1:    [B1] long — instance IDs for each box in frame-1.
        rel_input_1_pose:  [4, 4]   frame-1 -> frame-0 transform.

    Returns dict:
        'static_xyz':        [N_s, 3]
        'static_intensity':  [N_s]
        'static_time':       [N_s]  in {0, 1}
        'dynamic':           list of per-instance dicts (see below)
        'untracked_stats':   dict with logging info
    """
    device = p0.device
    dtype = p0.dtype

    B0 = boxes_0.shape[0] if boxes_0.numel() > 0 else 0
    B1 = boxes_1.shape[0] if boxes_1.numel() > 0 else 0

    # -----------------------------------------------------------------------
    # Transform frame-1 points and boxes into frame-0 sensor frame
    # -----------------------------------------------------------------------
    if p1.shape[0] > 0:
        p1_in_frame0 = _transform_points(p1.float(), rel_input_1_pose.float()).to(dtype)
    else:
        p1_in_frame0 = p1

    # Transform frame-1 boxes into frame-0 LiDAR
    boxes_1_in_frame0: list[Tensor] = []
    for bi in range(B1):
        boxes_1_in_frame0.append(
            _transform_box_to_frame0(boxes_1[bi].float(), rel_input_1_pose.float()).to(dtype)
        )
    if len(boxes_1_in_frame0) > 0:
        boxes_1_f0 = torch.stack(boxes_1_in_frame0, dim=0)   # [B1, 7]
    else:
        boxes_1_f0 = boxes_1.clone()

    # -----------------------------------------------------------------------
    # Identify tracked instances (present in BOTH frames)
    # -----------------------------------------------------------------------
    ids_0_set = set(instance_ids_0.tolist()) if B0 > 0 else set()
    ids_1_set = set(instance_ids_1.tolist()) if B1 > 0 else set()
    tracked_ids = ids_0_set & ids_1_set
    untracked_ids = ids_0_set.symmetric_difference(ids_1_set)

    # -----------------------------------------------------------------------
    # Build point-level masks: is each point inside a tracked bbox?
    # -----------------------------------------------------------------------
    N0 = p0.shape[0]
    N1 = p1.shape[0]

    in_tracked_0 = torch.zeros(N0, dtype=torch.bool, device=device)
    in_tracked_1 = torch.zeros(N1, dtype=torch.bool, device=device)
    # Also track untracked-instance points separately for logging
    in_untracked_0 = torch.zeros(N0, dtype=torch.bool, device=device)
    in_untracked_1 = torch.zeros(N1, dtype=torch.bool, device=device)

    # Per-tracked-instance point sets (for dynamic pipeline)
    per_instance_mask_0: dict[int, Tensor] = {}
    per_instance_mask_1: dict[int, Tensor] = {}

    # Process frame-0 boxes
    for bi in range(B0):
        iid = int(instance_ids_0[bi].item())
        mask = _points_in_box(p0.float(), boxes_0[bi].float())
        if iid in tracked_ids:
            in_tracked_0 |= mask
            per_instance_mask_0[iid] = per_instance_mask_0.get(iid, torch.zeros(N0, dtype=torch.bool, device=device)) | mask
        else:
            in_untracked_0 |= mask

    # Process frame-1 boxes
    for bi in range(B1):
        iid = int(instance_ids_1[bi].item())
        mask = _points_in_box(p1_in_frame0.float(), boxes_1_f0[bi].float())
        if iid in tracked_ids:
            in_tracked_1 |= mask
            per_instance_mask_1[iid] = per_instance_mask_1.get(iid, torch.zeros(N1, dtype=torch.bool, device=device)) | mask
        else:
            in_untracked_1 |= mask

    # -----------------------------------------------------------------------
    # Static pool: points NOT in any tracked or untracked bbox
    # Untracked instance points are INCLUDED in static (Q3 decision)
    # -----------------------------------------------------------------------
    static_mask_0 = ~in_tracked_0   # includes untracked + background
    static_mask_1 = ~in_tracked_1

    static_xyz_0 = p0[static_mask_0]
    static_int_0 = intensity_0[static_mask_0]
    static_time_0 = torch.zeros(static_xyz_0.shape[0], dtype=dtype, device=device)

    static_xyz_1 = p1_in_frame0[static_mask_1]
    static_int_1 = intensity_1[static_mask_1]
    static_time_1 = torch.ones(static_xyz_1.shape[0], dtype=dtype, device=device)

    static_xyz = torch.cat([static_xyz_0, static_xyz_1], dim=0)
    static_intensity = torch.cat([static_int_0, static_int_1], dim=0)
    static_time = torch.cat([static_time_0, static_time_1], dim=0)

    # -----------------------------------------------------------------------
    # Dynamic instances
    # -----------------------------------------------------------------------
    dynamic_list: list[dict] = []

    for iid in tracked_ids:
        # --- frame-0 points in canonical space ---
        present_0 = iid in per_instance_mask_0 and per_instance_mask_0[iid].any()
        present_1 = iid in per_instance_mask_1 and per_instance_mask_1[iid].any()

        # Retrieve frame-0 bbox
        box0_idx = (instance_ids_0 == iid).nonzero(as_tuple=True)[0]
        box0 = boxes_0[box0_idx[0]] if len(box0_idx) > 0 else None

        # Retrieve frame-1 bbox (in frame-0 LiDAR)
        box1_idx = (instance_ids_1 == iid).nonzero(as_tuple=True)[0]
        box1_f0 = boxes_1_f0[box1_idx[0]] if len(box1_idx) > 0 and boxes_1_f0.numel() > 0 else None

        canon_pts_list: list[Tensor] = []
        canon_int_list: list[Tensor] = []
        canon_time_list: list[Tensor] = []

        if present_0 and box0 is not None:
            pts0 = p0[per_instance_mask_0[iid]]
            int0 = intensity_0[per_instance_mask_0[iid]]
            pts0_canon = _to_canonical(pts0.float(), box0[:3].float(), box0[6].float()).to(dtype)
            t0 = torch.zeros(pts0_canon.shape[0], dtype=dtype, device=device)
            canon_pts_list.append(pts0_canon)
            canon_int_list.append(int0)
            canon_time_list.append(t0)

        if present_1 and box0 is not None:
            # Canonical space is anchored to the frame-0 box so both frames
            # share one object-centric coordinate system.
            pts1_f0 = p1_in_frame0[per_instance_mask_1[iid]]
            int1 = intensity_1[per_instance_mask_1[iid]]
            pts1_canon = _to_canonical(
                pts1_f0.float(), box0[:3].float(), box0[6].float()
            ).to(dtype)
            t1 = torch.ones(pts1_canon.shape[0], dtype=dtype, device=device)
            canon_pts_list.append(pts1_canon)
            canon_int_list.append(int1)
            canon_time_list.append(t1)

        if len(canon_pts_list) == 0:
            # No points extracted for this instance — skip
            continue

        canonical_xyz = torch.cat(canon_pts_list, dim=0)
        canonical_intensity = torch.cat(canon_int_list, dim=0)
        canonical_time = torch.cat(canon_time_list, dim=0)

        dynamic_list.append({
            "instance_id":          iid,
            "canonical_xyz":        canonical_xyz,
            "canonical_intensity":  canonical_intensity,
            "canonical_time":       canonical_time,
            "box_0":                box0,           # [7] frame-0 bbox in frame-0 LiDAR
            "box_1":                box1_f0,        # [7] frame-1 bbox in frame-0 LiDAR
            "present_in_0":         bool(present_0),
            "present_in_1":         bool(present_1),
        })

    # -----------------------------------------------------------------------
    # Untracked stats (for logging / paper reporting)
    # -----------------------------------------------------------------------
    n_untracked_pts_0 = int(in_untracked_0.sum().item())
    n_untracked_pts_1 = int(in_untracked_1.sum().item())
    untracked_stats = {
        "n_untracked_instances": len(untracked_ids),
        "n_untracked_points_frame0": n_untracked_pts_0,
        "n_untracked_points_frame1": n_untracked_pts_1,
        "n_untracked_points_total": n_untracked_pts_0 + n_untracked_pts_1,
        "n_tracked_instances": len(tracked_ids),
        "n_dynamic_instances_kept": len(dynamic_list),
    }

    return {
        "static_xyz":       static_xyz,
        "static_intensity": static_intensity,
        "static_time":      static_time,
        "dynamic":          dynamic_list,
        "untracked_stats":  untracked_stats,
    }
