"""Phase C — full static + dynamic anchor pipeline smoke.

decompose_scene → VoxelAnchorBuilder (static, spherical)
                → DynamicVoxelAnchorBuilder (dynamic, bbox-local cartesian)

Verifies both branches end-to-end on one nuScenes sample, plus aggregate stats
across a few samples to gauge dynamic-branch behaviour.
"""

import os
import random
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, "/data1/nuScenes/loader")

from dataset import NuScenesNVSDataset  # noqa: E402

from models.geometry.decomposition import decompose_scene  # noqa: E402
from models.geometry.voxel_anchor import (  # noqa: E402
    DynamicVoxelAnchorBuilder,
    VoxelAnchorBuilder,
)


SEED = 42
SAMPLE_INDICES = [12345, 20952, 7314]


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
    print(f"Dataset size: {len(ds)}\n")

    static_builder = VoxelAnchorBuilder()
    dynamic_builder = DynamicVoxelAnchorBuilder()

    for idx in SAMPLE_INDICES:
        print("=" * 80)
        print(f"Sample idx {idx}")
        print("=" * 80)
        item = ds[idx]
        pts0 = item["input_0"]
        pts1 = item["input_1"]
        T_1to0 = item["input_1_pose"]
        boxes_0 = item.get("boxes_0", torch.zeros((0, 7)))
        boxes_1 = item.get("boxes_1", torch.zeros((0, 7)))
        ids_0 = item.get("instance_ids_0", torch.zeros((0,), dtype=torch.int64))
        ids_1 = item.get("instance_ids_1", torch.zeros((0,), dtype=torch.int64))

        decomp = decompose_scene(
            p0=pts0[:, :3].to(device),
            p1=pts1[:, :3].to(device),
            intensity_0=(pts0[:, 3] / 255.0).clamp(0.0, 1.0).to(device),
            intensity_1=(pts1[:, 3] / 255.0).clamp(0.0, 1.0).to(device),
            boxes_0=boxes_0.to(device),
            boxes_1=boxes_1.to(device),
            instance_ids_0=ids_0.to(device),
            instance_ids_1=ids_1.to(device),
            rel_input_1_pose=T_1to0.to(device),
        )

        # ---------- Dynamic branch first ----------
        dynamic_results = dynamic_builder(decomp["dynamic"])
        n_inst = len(dynamic_results)
        n_inst_kept = sum(1 for _, o in dynamic_results if o.token.shape[0] > 0)
        total_dyn_anchors = sum(o.token.shape[0] for _, o in dynamic_results)
        per_inst_pts = [d["canonical_xyz"].shape[0] for d in decomp["dynamic"]]
        fallback_instances = [
            inst for inst, (_, out) in zip(decomp["dynamic"], dynamic_results)
            if out.token.shape[0] == 0
        ]
        print(f"  Dynamic: {n_inst} tracked instances "
              f"({n_inst_kept} produced anchors, {len(fallback_instances)} fall back to static)")
        print(f"           total dynamic anchors: {total_dyn_anchors}")
        print(f"           query_voxels={sum(o.diagnostics['query_voxels'] for _, o in dynamic_results)}  "
              f"k_eff_pass={sum(o.diagnostics['k_eff_pass'] for _, o in dynamic_results)}  "
              f"residual_pass={sum(o.diagnostics['residual_pass'] for _, o in dynamic_results)}")
        if per_inst_pts:
            print(f"           pts per instance — "
                  f"min={min(per_inst_pts)}, "
                  f"med={sorted(per_inst_pts)[len(per_inst_pts)//2]}, "
                  f"max={max(per_inst_pts)}")

        # ---------- Static branch after dynamic fallback merge ----------
        static_xyz = decomp["static_xyz"]
        static_int = decomp["static_intensity"]
        static_time = decomp["static_time"]
        base_static_pts = static_xyz.shape[0]
        if fallback_instances:
            static_xyz = torch.cat([static_xyz] + [d["fallback_xyz"] for d in fallback_instances], dim=0)
            static_int = torch.cat([static_int] + [d["fallback_intensity"] for d in fallback_instances], dim=0)
            static_time = torch.cat([static_time] + [d["fallback_time"] for d in fallback_instances], dim=0)
        static_out = static_builder(static_xyz, static_int, static_time)
        Ms = static_out.token.shape[0]
        fallback_pts = static_xyz.shape[0] - base_static_pts
        print(f"  Static: base={base_static_pts} pts + fallback={fallback_pts} pts "
              f"→ {static_xyz.shape[0]} pts → {Ms} anchors  "
              f"(token {tuple(static_out.token.shape)})")
        print(f"          diagnostics {static_out.diagnostics}")

        # Per-instance breakdown of first 3
        print("  Per-instance (first 3):")
        for k, (iid, out) in enumerate(dynamic_results[:3]):
            inst = decomp["dynamic"][k]
            box0 = inst["box_0"]
            max_dim = float(box0[3:6].max())
            voxel_size = max(max_dim / 8.0, 0.05)
            print(f"    iid={iid}  bbox_max_dim={max_dim:.2f}m  "
                  f"voxel_size={voxel_size:.3f}m  "
                  f"pts={inst['canonical_xyz'].shape[0]}  "
                  f"query={out.diagnostics['query_voxels']}  "
                  f"k_eff={out.diagnostics['k_eff_pass']}  "
                  f"anchors={out.token.shape[0]}  "
                  f"fallback={out.diagnostics['fallback_reason']}")

        # Sanity checks
        all_outputs = [static_out] + [o for _, o in dynamic_results if o.token.shape[0] > 0]
        for o in all_outputs:
            assert o.token.shape[1] == 22, f"token width {o.token.shape[1]} != 22"
            R = o.R_init
            if R.shape[0] > 0:
                eye = torch.eye(3, device=R.device).expand_as(R)
                err = (R @ R.transpose(-1, -2) - eye).abs().max()
                det = torch.det(R)
                assert err < 1e-4, f"R orth err {err:.2e}"
                assert (det > 0.99).all(), f"det(R) min {det.min():.4f}"
        print("  ✓ all branches: token=22 ch, R orthogonal, det≈+1")

    print("\nAll samples passed end-to-end.")


if __name__ == "__main__":
    main()
