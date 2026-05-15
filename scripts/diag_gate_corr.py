"""Init-summary / residual diagnostic after gate removal.

Checks whether residual magnitudes vary with the init summary consumed by the
geometry head.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/diag_gate_corr.py \
        --ckpt outputs/train_068/ckpt/best_model.pt --n_batches 20
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as _train_module  # noqa: triggers sys.path for dataset

import torch
import numpy as np
from config import QGSConfig, resolve_lidar_latent_dim
from nn.lidar_geometry import make_lidar_ray_grid

DropHead = None  # legacy diagnostic; per-Gaussian raydrop replaces DropHead.
from nn.model import QGSModel
from dataset import NuScenesNVSDataset, nvs_collate_fn
from train import (
    _make_static_anchor_builder,
    _batch_item, _filter_frame_points, _ensure_2d_pose,
    decompose_scene,
)


def collect_per_anchor(model, batch, device, cfg):
    """Run one sample through anchor-build + head, return init/residual tensors."""
    p0 = _batch_item(batch, "input_0", 0).to(device)
    p1 = _batch_item(batch, "input_1", 0).to(device)
    rel_pose = _ensure_2d_pose(_batch_item(batch, "input_1_pose", 0)).to(device)

    xyz0 = p0[:, :3].float();  i0 = (p0[:, 3].float() / 255.0).clamp(0, 1)
    xyz1 = p1[:, :3].float();  i1 = (p1[:, 3].float() / 255.0).clamp(0, 1)
    xyz0, i0 = _filter_frame_points(xyz0, i0, cfg)
    xyz1, i1 = _filter_frame_points(xyz1, i1, cfg)
    if xyz0.shape[0] < cfg.knn_k_min or xyz1.shape[0] < cfg.knn_k_min:
        return None

    boxes_0 = _batch_item(batch, "boxes_0", 0).to(device) if "boxes_0" in batch else torch.empty(0, 7, device=device)
    boxes_1 = _batch_item(batch, "boxes_1", 0).to(device) if "boxes_1" in batch else torch.empty(0, 7, device=device)
    instance_ids_0 = _batch_item(batch, "instance_ids_0", 0).to(device) if "instance_ids_0" in batch else torch.empty(0, dtype=torch.long, device=device)
    instance_ids_1 = _batch_item(batch, "instance_ids_1", 0).to(device) if "instance_ids_1" in batch else torch.empty(0, dtype=torch.long, device=device)

    if boxes_0.numel() and boxes_1.numel():
        scene = decompose_scene(xyz0, xyz1, i0, i1, boxes_0, boxes_1, instance_ids_0, instance_ids_1, rel_pose)
    else:
        p1_in_0 = (rel_pose[:3, :3] @ xyz1.T).T + rel_pose[:3, 3]
        scene = {
            "static_xyz": torch.cat([xyz0, p1_in_0], dim=0),
            "static_intensity": torch.cat([i0, i1], dim=0),
            "static_time": torch.cat([
                torch.zeros(xyz0.shape[0], device=device),
                torch.ones(xyz1.shape[0], device=device),
            ], dim=0),
            "dynamic_list": [],
        }

    static_anchor_out = _make_static_anchor_builder(cfg)(
        scene["static_xyz"],
        scene["static_intensity"],
        scene["static_time"],
        pose_frame1_in_frame0=rel_pose,
    )
    if static_anchor_out.c_init.shape[0] == 0:
        return None

    static_prims = model.forward_anchor_context(static_anchor_out, context_type="static")
    if static_prims is None:
        return None

    aux = static_prims["aux"]
    init_summary = aux["init_summary"].squeeze(0).float().cpu()
    omega_norm = aux["omega_local"].squeeze(0).float().norm(dim=-1).cpu()
    delta_c_norm = aux["delta_c"].squeeze(0).float().norm(dim=-1).cpu()
    delta_scale = (
        aux["delta_mu"].squeeze(0).float().abs()
        + aux["delta_gap"].squeeze(0).float().abs()
        + aux["delta_log_abs_s3"].squeeze(0).float().abs()
    ).cpu()
    return {
        "fit_residual": init_summary[:, 0].numpy(),
        "planarity": init_summary[:, 1].numpy(),
        "support": init_summary[:, 2].numpy(),
        "reach": init_summary[:, 3].numpy(),
        "tangent_aniso": init_summary[:, 4].numpy(),
        "curvature_aniso": init_summary[:, 5].numpy(),
        "log_abs_s1": init_summary[:, 6].numpy(),
        "log_abs_s2": init_summary[:, 7].numpy(),
        "log_abs_s3": init_summary[:, 8].numpy(),
        "kappa1": init_summary[:, 9].numpy(),
        "kappa2": init_summary[:, 10].numpy(),
        "use_geom_init": init_summary[:, 11].numpy(),
        "k_eff_norm": init_summary[:, 12].numpy(),
        "omega_norm": omega_norm.numpy(),
        "delta_c_norm": delta_c_norm.numpy(),
        "delta_scale": delta_scale.numpy(),
    }


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).mean()) * np.sqrt((b**2).mean())
    return float((a * b).mean() / (denom + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/train_068/ckpt/best_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = QGSConfig()

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg_json = os.path.join(ckpt_dir, "..", "configs", "config.json")
    if os.path.exists(cfg_json):
        import json, dataclasses
        saved = json.load(open(cfg_json))
        field_names = {f.name for f in dataclasses.fields(cfg)}
        for k, v in saved.items():
            if k in field_names:
                try:
                    cur = getattr(cfg, k)
                    if isinstance(cur, tuple) and isinstance(v, list):
                        v = tuple(v)
                    setattr(cfg, k, v)
                except Exception:
                    pass

    cfg.lidar_latent_dim = resolve_lidar_latent_dim(cfg.lidar_latent_dim)

    model = QGSModel(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    print(f"Loaded {args.ckpt}  epoch={ckpt.get('epoch','?')}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bbox_json = getattr(cfg, "bbox_json_path", "bbox/tracking_{split}.json")
    if bbox_json:
        bbox_json = bbox_json.format(split=getattr(cfg, "train_split", "train"))
        if not os.path.isabs(bbox_json):
            bbox_json = os.path.join(repo_root, bbox_json)
    bbox_json = bbox_json if os.path.isfile(bbox_json) else None

    ds = NuScenesNVSDataset(
        dataroot=cfg.data_root,
        version=getattr(cfg, "nuscenes_version", "v1.0-trainval"),
        split=getattr(cfg, "train_split", "train"),
        frame_gap=getattr(cfg, "frame_gap", 2),
        mode=getattr(cfg, "dataset_mode", "bbox"),
        bbox_json_path=bbox_json,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=nvs_collate_fn, num_workers=2)

    feature_names = [
        "fit_residual", "planarity", "support", "reach",
        "tangent_aniso", "curvature_aniso",
        "log_abs_s1", "log_abs_s2", "log_abs_s3",
        "kappa1", "kappa2", "use_geom_init", "k_eff_norm",
    ]
    target_names = ["omega_norm", "delta_c_norm", "delta_scale"]
    buckets = {k: [] for k in feature_names + target_names}
    n_ok = 0
    print(f"Running {args.n_batches} batches ...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if n_ok >= args.n_batches:
                break
            result = collect_per_anchor(model, batch, device, cfg)
            if result is None:
                continue
            for k, v in result.items():
                buckets[k].append(v)
            n = len(result["delta_scale"])
            if n_ok % 5 == 0:
                ds = result["delta_scale"]
                print(
                    f"  sample {n_ok}: n_anchors={n} "
                    f"delta_scale mean={ds.mean():.4f} std={ds.std():.4f} "
                    f"min={ds.min():.4f} max={ds.max():.4f}"
                )
            n_ok += 1

    if n_ok == 0:
        print("No valid samples.")
        return

    data = {k: np.concatenate(v) for k, v in buckets.items()}
    N = len(data["delta_scale"])
    print(f"\nTotal anchors: {N}")
    print()

    for target in target_names:
        arr = data[target]
        print(
            f"{target:12s} mean={arr.mean():.4f} std={arr.std():.4f} "
            f"min={arr.min():.4f} max={arr.max():.4f}"
        )
        for pct in [5, 25, 50, 75, 95]:
            print(f"  P{pct:2d}: {np.percentile(arr, pct):.4f}")

    for target in target_names:
        print(f"\n--- Pearson corr(init_summary, {target}) ---")
        for feat in feature_names:
            r = pearson(data[feat], data[target])
            print(f"  corr({feat:20s}, {target:12s}) = {r:+.4f}")


if __name__ == "__main__":
    main()
