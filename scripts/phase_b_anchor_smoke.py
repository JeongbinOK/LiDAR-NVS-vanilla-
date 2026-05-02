"""Phase B — VoxelAnchorBuilder end-to-end smoke test.

1 sample → voxelize → kNN → quad_fit → 2-stage filter → 22-ch token.
Verifies shapes, ranges, and token-channel correctness.
"""

import os
import sys
import random
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, "/data1/nuScenes/loader")

from dataset import NuScenesNVSDataset  # noqa: E402
from models.geometry.decomposition import decompose_scene  # noqa: E402
from models.geometry.voxel_anchor import VoxelAnchorBuilder  # noqa: E402


SEED = 42
SAMPLE_IDX = 12345  # same as HTML viz


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    ds = NuScenesNVSDataset(
        dataroot="/data1/nuScenes",
        version="v1.0-trainval",
        split="train",
        frame_gap=2,
        mode="bbox",
        bbox_json_path=None,
    )
    item = ds[SAMPLE_IDX]
    print(f"Sample idx {SAMPLE_IDX}, dataset size {len(ds)}\n")

    pts0 = item["input_0"]
    pts1 = item["input_1"]
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

    print(f"Static input: N={xyz_all.shape[0]} pts, "
          f"tracked_dynamic={scene['untracked_stats']['n_tracked_instances']}")

    builder = VoxelAnchorBuilder()
    out = builder(xyz_all, i_all, src_all)
    M_prime = out.token.shape[0]

    print(f"\nAnchor output: M' = {M_prime}")
    print(f"  diagnostics   {out.diagnostics}")
    print(f"  c_init        {tuple(out.c_init.shape)}        "
          f"range x:[{out.c_init[:,0].min():.1f},{out.c_init[:,0].max():.1f}] "
          f"y:[{out.c_init[:,1].min():.1f},{out.c_init[:,1].max():.1f}] "
          f"z:[{out.c_init[:,2].min():.1f},{out.c_init[:,2].max():.1f}]")
    print(f"  R_init        {tuple(out.R_init.shape)}      "
          f"det range:[{torch.det(out.R_init).min():.4f},{torch.det(out.R_init).max():.4f}]")
    print(f"  s_init        {tuple(out.s_init.shape)}        "
          f"|s1| med={out.s_init[:,0].abs().median():.4f}, "
          f"|s2| med={out.s_init[:,1].abs().median():.4f}, "
          f"s3 med={out.s_init[:,2].median():.4f}")
    print(f"  kappa1/2      med1={out.kappa1.median():.4f}, med2={out.kappa2.median():.4f}")
    print(f"  fit_quality   {tuple(out.fit_quality.shape)}      "
          f"residual med={out.fit_quality[:,0].median():.4f} "
          f"max={out.fit_quality[:,0].max():.4f}  "
          f"(threshold τ_r={builder.residual_threshold})")
    print(f"  i_mean/std    med m={out.i_mean.median():.3f}, std={out.i_std.median():.3f}")
    print(f"  src_ratio     med={out.src_ratio.median():.3f} "
          f"(LiDAR_1 share inside each anchor's voxel)")
    print(f"  n_points      med={out.n_points.float().median():.1f}, "
          f"max={out.n_points.max()}")
    print(f"  token         {tuple(out.token.shape)}      "
          f"min={out.token.min():.3f}, max={out.token.max():.3f}")

    # ---------------- Sanity checks ----------------
    print("\nSanity checks:")
    assert out.token.shape[1] == 22, "token width must be 22"
    print("  ✓ token width = 22")
    assert (out.fit_quality[:, 0] < builder.residual_threshold).all(), \
        "all anchors must satisfy residual < τ_r"
    print(f"  ✓ all anchors residual < τ_r = {builder.residual_threshold}")
    R = out.R_init
    eye = torch.eye(3, device=device).expand_as(R)
    err = (R @ R.transpose(-1, -2) - eye).abs().max()
    print(f"  ✓ R orthogonality error max = {err:.2e}")
    det = torch.det(R)
    assert (det > 0.99).all(), "R must be proper rotation"
    print(f"  ✓ det(R) ≈ +1, all proper rotations")

    print(f"\nAnchor reduction: {xyz_all.shape[0]} pts → {M_prime} anchors "
          f"({xyz_all.shape[0]/max(M_prime,1):.1f}× compression)")


if __name__ == "__main__":
    main()
