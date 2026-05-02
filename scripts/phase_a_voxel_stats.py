"""Phase A — Spherical voxel size sweep statistics.

Random 10 train samples × 3 voxel candidates → reports:
  1) points-per-valid-voxel distribution (Q1/median/Q3/mean/max)
  2) query-voxel ratio (count >= query_min) and occupancy pass ratio
  3) range-stratified src_ratio (frame1 share) — verifies LiDAR_1 far-range coverage
  4) (optional) dynamic instance count for context

Static-only voxelization (per §-0 decision in plan):
  - Dynamic = boxes whose instance_id appears in BOTH frames.
  - Static  = all points NOT inside any dynamic box.
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, '/data1/nuScenes/loader')
from dataset import NuScenesNVSDataset  # noqa: E402


CANDIDATES = [
    ('A', 2.0, 2.0, 2.0),
    ('B', 3.0, 4.0, 3.0),
    ('C', 1.5, 1.5, 1.5),
]
K_MIN = 8
QUERY_MIN = 1
N_SAMPLES = 10
SEED = 42
RANGE_BANDS = [(0, 10), (10, 25), (25, 50), (50, 80), (80, 200)]


def transform_xyz(xyz: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    return (T @ h.T).T[:, :3]


def points_in_box(xyz: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """xyz: (N,3). box: (7,) [x,y,z,w,l,h,yaw] (nuScenes wlh)."""
    cx, cy, cz, w, l, h, yaw = [float(x) for x in box.tolist()]
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    dx = xyz[:, 0] - cx
    dy = xyz[:, 1] - cy
    dz = xyz[:, 2] - cz
    lx = cos_y * dx - sin_y * dy
    ly = sin_y * dx + cos_y * dy
    return (lx.abs() <= l / 2) & (ly.abs() <= w / 2) & (dz.abs() <= h / 2)


def static_mask(xyz: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.shape[0] == 0:
        return torch.ones(xyz.shape[0], dtype=torch.bool)
    inside_any = torch.zeros(xyz.shape[0], dtype=torch.bool)
    for b in boxes:
        inside_any |= points_in_box(xyz, b)
    return ~inside_any


def transform_box_to_lidar0(box: torch.Tensor, T_1_to_0: torch.Tensor) -> torch.Tensor:
    """Convert box [x,y,z,w,l,h,yaw] from LiDAR_1 frame to LiDAR_0 frame."""
    center_h = torch.tensor([box[0], box[1], box[2], 1.0], dtype=T_1_to_0.dtype)
    center_0 = (T_1_to_0 @ center_h)[:3]
    R = T_1_to_0[:3, :3]
    yaw0 = box[6].item()
    fwd = torch.tensor([np.cos(yaw0), np.sin(yaw0), 0.0], dtype=T_1_to_0.dtype)
    fwd_0 = (R @ fwd)[:3]
    new_yaw = float(np.arctan2(float(fwd_0[1]), float(fwd_0[0])))
    return torch.tensor([center_0[0], center_0[1], center_0[2],
                         box[3], box[4], box[5], new_yaw], dtype=box.dtype)


def spherical_voxelize(xyz: torch.Tensor, dphi_deg: float, dtheta_deg: float, dr_m: float):
    r = torch.norm(xyz, dim=-1)
    safe_r = r.clamp(min=1e-6)
    phi = torch.atan2(xyz[:, 1], xyz[:, 0])               # [-pi, pi]
    theta = torch.asin((xyz[:, 2] / safe_r).clamp(-1, 1)) # [-pi/2, pi/2]
    dphi = np.deg2rad(dphi_deg)
    dtheta = np.deg2rad(dtheta_deg)
    iphi = torch.floor((phi + np.pi) / dphi).long()
    itheta = torch.floor((theta + np.pi / 2) / dtheta).long()
    ir = torch.floor(r / dr_m).long()
    n_phi = int(np.ceil(2 * np.pi / dphi)) + 1
    n_theta = int(np.ceil(np.pi / dtheta)) + 1
    voxel_id = ir * (n_phi * n_theta) + itheta * n_phi + iphi
    return voxel_id, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset (train, frame_gap=2, mode=bbox)...")
    ds = NuScenesNVSDataset(
        dataroot='/data1/nuScenes',
        version='v1.0-trainval',
        split='train',
        frame_gap=2,
        mode='bbox',
        bbox_json_path=None,  # use GT annotations
    )
    print(f"Dataset size: {len(ds)}\n")

    indices = random.sample(range(len(ds)), args.n_samples)

    results = {
        name: {
            'counts_per_valid_voxel': [],
            'valid_voxel_ratio': [],
            'query_voxel_count': [],
            'src_ratio_per_range': {b: [] for b in RANGE_BANDS},
            'voxel_count_per_band': {b: 0 for b in RANGE_BANDS},
        }
        for name, *_ in CANDIDATES
    }
    total_pts_log = []
    dyn_count_log = []

    for i, idx in enumerate(indices):
        item = ds[idx]
        pts0 = item['input_0']                 # (N0, 4) — LiDAR_0 frame
        pts1 = item['input_1']                 # (N1, 4) — LiDAR_1 frame
        T_1to0 = item['input_1_pose']          # (4, 4)

        boxes_0 = item.get('boxes_0', torch.zeros((0, 7)))
        boxes_1 = item.get('boxes_1', torch.zeros((0, 7)))
        ids_0 = item.get('instance_ids_0', torch.zeros((0,), dtype=torch.int64)).tolist()
        ids_1 = item.get('instance_ids_1', torch.zeros((0,), dtype=torch.int64)).tolist()
        common_ids = set(ids_0) & set(ids_1)

        # Filter dynamic boxes per locked design — dynamic = appears in BOTH frames
        mask0 = torch.tensor([_id in common_ids for _id in ids_0], dtype=torch.bool)
        mask1 = torch.tensor([_id in common_ids for _id in ids_1], dtype=torch.bool)
        dyn_boxes_0 = boxes_0[mask0] if boxes_0.shape[0] > 0 else boxes_0
        dyn_boxes_1 = boxes_1[mask1] if boxes_1.shape[0] > 0 else boxes_1

        # Drop dynamic points from each frame in its own sensor frame
        keep0 = static_mask(pts0[:, :3], dyn_boxes_0)
        keep1 = static_mask(pts1[:, :3], dyn_boxes_1)
        pts0_static = pts0[keep0]
        pts1_static = pts1[keep1]
        # Bring frame-1 static points to LiDAR_0 frame
        xyz1_in0 = transform_xyz(pts1_static[:, :3], T_1to0)

        n0 = pts0_static.shape[0]
        n1 = pts1_static.shape[0]
        all_xyz = torch.cat([pts0_static[:, :3], xyz1_in0], dim=0)
        src = torch.cat([torch.zeros(n0), torch.ones(n1)])  # 0 = LiDAR_0, 1 = LiDAR_1

        total_pts_log.append((n0, n1))
        dyn_count_log.append(len(common_ids))

        for name, dphi, dtheta, dr in CANDIDATES:
            voxel_id, r_pts = spherical_voxelize(all_xyz, dphi, dtheta, dr)
            unique, counts = torch.unique(voxel_id, return_counts=True)

            valid_mask = counts >= K_MIN
            query_mask = counts >= QUERY_MIN
            valid_voxel_ratio = valid_mask.float().mean().item()
            results[name]['query_voxel_count'].append(int(query_mask.sum().item()))
            results[name]['counts_per_valid_voxel'].extend(counts[valid_mask].tolist())
            results[name]['valid_voxel_ratio'].append(valid_voxel_ratio)

            # Per-voxel mean range — for range-band src_ratio
            sort_id = torch.argsort(voxel_id)
            sorted_vid = voxel_id[sort_id]
            sorted_src = src[sort_id]
            sorted_r = r_pts[sort_id]

            # group-wise mean range and src_ratio per voxel
            change = torch.cat([
                torch.tensor([True]),
                sorted_vid[1:] != sorted_vid[:-1]
            ])
            group_starts = torch.where(change)[0]
            group_ends = torch.cat([group_starts[1:], torch.tensor([sorted_vid.shape[0]])])

            for s_idx, e_idx, c in zip(group_starts.tolist(), group_ends.tolist(), counts.tolist()):
                if c < QUERY_MIN:
                    continue
                vox_r_mean = sorted_r[s_idx:e_idx].mean().item()
                vox_src_ratio = sorted_src[s_idx:e_idx].mean().item()
                for band in RANGE_BANDS:
                    if band[0] <= vox_r_mean < band[1]:
                        results[name]['src_ratio_per_range'][band].append(vox_src_ratio)
                        results[name]['voxel_count_per_band'][band] += 1
                        break

        print(f"[{i+1:2d}/{args.n_samples}] sample {idx}: "
              f"pts0={n0}, pts1={n1}, dyn_inst={len(common_ids)}")

    # ----------------------------------------------------------------- Report
    print("\n" + "=" * 88)
    print(f"Phase A — Spherical voxel statistics ({args.n_samples} train samples, query_min={QUERY_MIN}, k_min={K_MIN})")
    print("=" * 88)
    print(f"\nMean static-points per sample: "
          f"frame0={np.mean([n[0] for n in total_pts_log]):.0f}, "
          f"frame1={np.mean([n[1] for n in total_pts_log]):.0f}")
    print(f"Mean common (dynamic) instance count: {np.mean(dyn_count_log):.1f}")

    for name, dphi, dtheta, dr in CANDIDATES:
        rd = results[name]
        cnt = np.array(rd['counts_per_valid_voxel'], dtype=np.float64)
        vvr = np.array(rd['valid_voxel_ratio'])
        qvc = np.array(rd['query_voxel_count'])
        print("\n" + "-" * 88)
        print(f"Candidate ({name})  Δφ={dphi}°, Δθ={dtheta}°, Δr={dr}m")
        print("-" * 88)
        print(f"  Query voxels (count≥{QUERY_MIN}, mean±std over samples): "
              f"{qvc.mean():.1f} ± {qvc.std():.1f}")
        print(f"  Occupancy pass ratio (count≥{K_MIN}, mean±std over samples): "
              f"{vvr.mean():.3f} ± {vvr.std():.3f}")
        if cnt.size:
            print(f"  Points per VALID voxel:  Q1={np.percentile(cnt,25):.1f}  "
                  f"median={np.median(cnt):.1f}  Q3={np.percentile(cnt,75):.1f}  "
                  f"mean={cnt.mean():.1f}  max={cnt.max():.0f}  total_voxels={cnt.size}")
        print(f"  Range-stratified src_ratio (mean frame-1 share per voxel within band):")
        for band in RANGE_BANDS:
            ratios = rd['src_ratio_per_range'][band]
            n_vox = rd['voxel_count_per_band'][band]
            if ratios:
                arr = np.array(ratios)
                print(f"    r ∈ [{band[0]:>3}, {band[1]:>3}) m  "
                      f"n_vox={n_vox:>5}  src_ratio={arr.mean():.3f} ± {arr.std():.3f}")
            else:
                print(f"    r ∈ [{band[0]:>3}, {band[1]:>3}) m  n_vox=0")


if __name__ == '__main__':
    main()
