"""Phase B — voxel-anchored quad_fit smoke + fit_quality distribution.

Pipeline per sample:
  1) Static / dynamic split (instance_id 두 frame 공통 → dynamic, 그 외 static).
  2) SphericalVoxelizer (Δφ=3°, Δθ=4°, Δr=3m, query_min=1) on static cloud.
  3) hybrid_radius_knn(query=voxel_mean, candidates=all_static_pts).
  4) fit_local_quadrics → residual, planarity, support, reach.

Aggregate fit_quality across 10 samples → report percentiles.
Recommended τ_r, τ_p = 75 percentile (plan §−0).
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, "/data1/nuScenes/loader")

from dataset import NuScenesNVSDataset  # noqa: E402

from models.geometry.decomposition import decompose_scene  # noqa: E402
from models.geometry.spherical_voxel import SphericalVoxelizer  # noqa: E402
from models.geometry.knn_radius import hybrid_radius_knn  # noqa: E402
from models.geometry.quadric_fit import fit_local_quadrics  # noqa: E402


# Plan §-0: voxel B from Phase A
DPHI_DEG = 3.0
DTHETA_DEG = 4.0
DR_M = 3.0
K_MIN = 8
QUERY_MIN = 1
K_TARGET = 16
N_SAMPLES = 10
SEED = 42


def voxel_adaptive_r_max(query_xyz: torch.Tensor) -> torch.Tensor:
    """Spherical-voxel-aware k-NN radius.

    Larger than the default per-point r_max because voxel diameter at far range
    grows with r (Δφ·r, Δθ·r). Use r_max = max(voxel_diag, default_floor).
    """
    r = query_xyz.norm(dim=-1)
    dphi_r = math.radians(DPHI_DEG) * r
    dtheta_r = math.radians(DTHETA_DEG) * r
    dr = torch.full_like(r, DR_M)
    voxel_diag = torch.sqrt(dphi_r * dphi_r + dtheta_r * dtheta_r + dr * dr)
    return torch.clamp(voxel_diag, min=0.5, max=8.0)


def main():
    import math as _math
    global math
    math = _math

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading dataset (train, frame_gap=2, mode=bbox), device={device}")
    ds = NuScenesNVSDataset(
        dataroot="/data1/nuScenes",
        version="v1.0-trainval",
        split="train",
        frame_gap=2,
        mode="bbox",
        bbox_json_path=None,
    )
    print(f"Dataset size: {len(ds)}\n")

    indices = random.sample(range(len(ds)), args.n_samples)

    voxelizer = SphericalVoxelizer(
        DPHI_DEG, DTHETA_DEG, DR_M, query_voxel_min_points=QUERY_MIN
    )

    all_residual: list[float] = []
    all_planarity: list[float] = []
    all_support: list[float] = []
    all_reach: list[float] = []
    all_use_geom: list[bool] = []
    n_voxels_total = 0
    n_static_total = 0

    for i, idx in enumerate(indices):
        item = ds[idx]
        pts0 = item["input_0"]               # (N0, 4)
        pts1 = item["input_1"]               # (N1, 4)
        T_1to0 = item["input_1_pose"]

        boxes_0 = item.get("boxes_0", torch.zeros((0, 7)))
        boxes_1 = item.get("boxes_1", torch.zeros((0, 7)))
        ids_0 = item.get("instance_ids_0", torch.zeros((0,), dtype=torch.int64))
        ids_1 = item.get("instance_ids_1", torch.zeros((0,), dtype=torch.int64))
        scene = decompose_scene(
            p0=pts0[:, :3].to(device),
            p1=pts1[:, :3].to(device),
            intensity_0=(pts0[:, 3].clamp(0.0, 255.0) / 255.0).to(device),
            intensity_1=(pts1[:, 3].clamp(0.0, 255.0) / 255.0).to(device),
            boxes_0=boxes_0.to(device),
            boxes_1=boxes_1.to(device),
            instance_ids_0=ids_0.to(device),
            instance_ids_1=ids_1.to(device),
            rel_input_1_pose=T_1to0.to(device),
        )
        xyz_all = scene["static_xyz"]
        i_all = scene["static_intensity"]
        src_all = scene["static_time"]

        vox = voxelizer(xyz_all, intensity=i_all, src=src_all)
        M = vox.query_xyz.shape[0]
        N_total = xyz_all.shape[0]
        n_static_total += N_total
        n_voxels_total += M

        # k-NN: query = voxel mean, candidates = all static points
        knn = hybrid_radius_knn(
            points=vox.query_xyz,
            candidates=xyz_all,
            k_target=K_TARGET,
            r_max_fn=voxel_adaptive_r_max,
            k_min=K_MIN,
            chunk_size=256,
        )
        idx_knn = knn["idx"]            # [M, K]
        k_eff = knn["k_eff"]            # [M]

        # Gather neighbors. -1 indices (padding) replaced with 0; valid mask separately
        idx_safe = idx_knn.clamp(min=0)
        nbrs = xyz_all[idx_safe]        # [M, K, 3]

        # quad_fit expects [B, N, 3] / [B, N, K, 3] / [B, N]
        geom = fit_local_quadrics(
            points=vox.query_xyz.unsqueeze(0),
            neighbors=nbrs.unsqueeze(0),
            k_eff=k_eff.unsqueeze(0),
            k_min=K_MIN,
            k_target=K_TARGET,
        )
        fit_q = geom["fit_quality"][0].cpu()  # [M, 4]
        use_geom = geom["use_geom_init"][0].cpu()  # [M]

        all_residual.extend(fit_q[:, 0].tolist())
        all_planarity.extend(fit_q[:, 1].tolist())
        all_support.extend(fit_q[:, 2].tolist())
        all_reach.extend(fit_q[:, 3].tolist())
        all_use_geom.extend(use_geom.tolist())

        print(f"[{i+1:2d}/{args.n_samples}] sample {idx}: "
              f"static_pts={N_total}, query_voxels={M}, "
              f"k_eff_pass={int((k_eff >= K_MIN).sum())}/{M}, "
              f"use_geom={int(use_geom.sum())}/{M}")

    # ----------------------------------------------------------------- Report
    print("\n" + "=" * 88)
    print(f"Phase B — fit_quality distribution over {args.n_samples} samples "
          f"(voxel B, query_min={QUERY_MIN}, k_min={K_MIN}, k_target={K_TARGET})")
    print("=" * 88)

    res = np.array(all_residual)
    plan = np.array(all_planarity)
    sup = np.array(all_support)
    rch = np.array(all_reach)
    use_geom_arr = np.array(all_use_geom)

    print(f"\nVoxels considered: {res.size} (across all samples)")
    print(f"Static points considered: {n_static_total}")
    print(f"use_geom_init=True ratio: {use_geom_arr.mean():.3f}  "
          f"(k_eff>=k_min after k-NN)")

    def report(name, arr):
        print(f"  {name:<12s}  "
              f"min={arr.min():.4f}  Q1={np.percentile(arr,25):.4f}  "
              f"med={np.median(arr):.4f}  Q3={np.percentile(arr,75):.4f}  "
              f"P95={np.percentile(arr,95):.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}")

    print("\n[All voxels] fit_quality components:")
    report("residual", res)
    report("planarity", plan)
    report("support", sup)
    report("reach", rch)

    if use_geom_arr.any():
        idx_ok = use_geom_arr.astype(bool)
        print("\n[use_geom_init=True only] fit_quality components:")
        report("residual", res[idx_ok])
        report("planarity", plan[idx_ok])

    print("\n" + "-" * 88)
    print("Recommended hard thresholds (75 percentile, use_geom only):")
    if use_geom_arr.any():
        idx_ok = use_geom_arr.astype(bool)
        tau_r = float(np.percentile(res[idx_ok], 75))
        tau_p = float(np.percentile(plan[idx_ok], 75))
        n_pass = int(((res[idx_ok] < tau_r) & (plan[idx_ok] < tau_p)).sum())
        n_total_geom = int(idx_ok.sum())
        print(f"  τ_r (residual)  ≈ {tau_r:.4f}")
        print(f"  τ_p (planarity) ≈ {tau_p:.4f}")
        print(f"  Anchor pass rate (residual<τ_r AND planarity<τ_p): "
              f"{n_pass}/{n_total_geom} = {n_pass/n_total_geom:.3f}")
    print("-" * 88)


if __name__ == "__main__":
    main()
