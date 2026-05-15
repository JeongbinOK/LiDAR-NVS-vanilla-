"""M1/M2 signed-residual diagnostic: load a checkpoint, run 1 sample forward.

Prints:
  - [M1] signed means of delta_mu / delta_gap / delta_s3 → Hypothesis A vs B
  - [M2] tilt vs spin split of omega → R_init quality vs gauge drift

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/diag_m1m2.py \
        --ckpt outputs/train_067/ckpt/best_model.pt
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# train.py already sets up sys.path for the dataset loader at import time
import train as _train_module  # noqa: triggers sys.path for dataset

import torch
from config import QGSConfig, resolve_lidar_latent_dim
from nn.lidar_geometry import make_lidar_ray_grid

DropHead = None  # legacy diagnostic; per-Gaussian raydrop replaces DropHead.
from nn.model import QGSModel
from nn.qgs_loss import QGSLoss
from dataset import NuScenesNVSDataset, nvs_collate_fn  # injected by train import


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/train_067/ckpt/best_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = QGSConfig()

    # Load the saved config from the checkpoint's run directory
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
        print(f"Loaded config from {cfg_json}")
    else:
        print(f"WARNING: no config.json found at {cfg_json}, using defaults")

    cfg.lidar_latent_dim = resolve_lidar_latent_dim(cfg.lidar_latent_dim)

    # ---- model + helpers ----
    model = QGSModel(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    print(f"Loaded {args.ckpt}  (epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss','?'):.4f})")

    loss_fn = QGSLoss(
        w_depth=cfg.loss_w_range,
        w_intensity=cfg.loss_w_intensity,
        w_raydrop=getattr(cfg, "loss_w_raydrop", 0.1),
        w_distortion=getattr(cfg, "loss_w_distortion", 0.05),
        w_normal=getattr(cfg, "loss_w_normal", 0.05),
        alpha_eps=cfg.loss_alpha_eps,
    ).to(device)

    drop_head = DropHead(latent_dim=cfg.lidar_latent_dim).to(device)
    if ckpt.get("drop_head_state_dict"):
        drop_head.load_state_dict(ckpt["drop_head_state_dict"], strict=True)
    drop_head.eval()

    ray_grid = make_lidar_ray_grid(cfg, device=device)

    # ---- dataset (mirror train.py construction) ----
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
    print(f"Dataset size: {len(ds)}, using sample idx={args.sample_idx}")

    sample = ds[args.sample_idx]
    batch = nvs_collate_fn([sample])

    # ---- forward ----
    print("Running forward pass ...")
    with torch.no_grad():
        out = _train_module.process_pair(
            model, loss_fn, drop_head, ray_grid,
            batch, idx=0, device=device, cfg=cfg,
        )

    if out is None:
        print("ERROR: process_pair returned None (not enough points?)")
        return

    # ---- report ----
    mu_abs  = float(out.get("diag_delta_mu", 0.0))
    mu_sign = float(out.get("diag_delta_mu_sign", float("nan")))
    gap_abs  = float(out.get("diag_delta_gap", 0.0))
    gap_sign = float(out.get("diag_delta_gap_sign", float("nan")))
    s3_abs   = float(out.get("diag_delta_s3", 0.0))
    s3_sign  = float(out.get("diag_delta_s3_sign", float("nan")))
    tilt_deg = float(out.get("diag_tilt_deg", float("nan")))
    spin_deg = float(out.get("diag_spin_deg", float("nan")))
    omega_deg = float(out.get("diag_omega_deg", 0.0))

    ln2  = math.log(2.0)
    ln15 = math.log(1.5)

    print()
    print("=" * 68)
    print("  QGSHead residual diagnostics (no gate module)")
    print()
    print("  [M2] Rotation split  (tilt cap=10°, spin cap=30°)")
    print(f"    dOmega (‖ω‖) : {omega_deg:.2f}°")
    print(f"    tilt  (ω_x,y): {tilt_deg:.2f}°  ({tilt_deg/10*100:.0f}% of cap)  ← rendering-sensitive")
    print(f"    spin  (ω_z)  : {spin_deg:.2f}°  ({spin_deg/30*100:.0f}% of cap)  ← gauge-free on flat surfaces")
    if math.isnan(tilt_deg):
        print("    (diag_tilt_deg not in output — check train.py edit)")
    elif tilt_deg < 4.0 and spin_deg > 15.0:
        print("    >> Pure gauge drift — R_init OK, only spin is saturated")
    elif tilt_deg > 6.0:
        print("    >> R_init inaccurate — normal direction noisy from sparse PCA")
    else:
        print(f"    >> Borderline: tilt={tilt_deg:.1f}°")

    print()
    print("  [M1] Signed scale residuals  (A=systematic bias, B=capacity blow-up)")
    print(f"    dMu  |{mu_abs:.4f}|  sgn={mu_sign:+.4f}  (cap=ln2={ln2:.4f})")
    print(f"    dGap |{gap_abs:.4f}|  sgn={gap_sign:+.4f}  (cap=ln1.5={ln15:.4f})")
    print(f"    dS3  |{s3_abs:.4f}|  sgn={s3_sign:+.4f}  (cap=ln1.5={ln15:.4f})")
    if math.isnan(mu_sign):
        print("    (diag_delta_mu_sign not in output — check train.py edit)")
    elif abs(mu_sign) > 0.35:
        direction = "TOO SMALL (head scales up)" if mu_sign > 0 else "TOO LARGE (head scales down)"
        print(f"\n    >> HYPOTHESIS A: quadric init scale systematically {direction}")
        print(f"       |sgn|={abs(mu_sign):.4f} > 0.35 threshold → fix quadric_fit.py first")
    else:
        print(f"\n    >> HYPOTHESIS B: capacity blow-up (mixed dirs, |sgn|={abs(mu_sign):.4f} < 0.35)")
        print("       → residual/loss contract, not gate regularization, should be checked next")
    print("=" * 68)


if __name__ == "__main__":
    main()
