"""LiDAR4D / GS-LiDAR-style temporal NVS evaluation on the 5 Table S6 nuScenes
sequences (v1.0-trainval).

Protocol (1-second temporal novel-view synthesis):
  per sequence, for each target second T in {1,2,3,4}:
      input  = LiDAR at (T-0.5)s and (T+0.5)s
      render = predict the panorama at T s and compare to the GT frame at T s.

Metrics follow the official GS-LiDAR code (utils/metrics_utils.py):
  depth / intensity : RMSE, MedAE, LPIPS(alex), SSIM, PSNR  (raydrop-masked, full map)
  raydrop           : RMSE, Acc, F1
  points            : Chamfer distance, F-score@0.05
Our depth is already in meters (g2p.scale_factor == 1.0), so GS-LiDAR's
`scale_factor` rescale is NOT applied (scale=1.0).

Usage:
  python test_lidar4d_nvs.py test.ckpt_path=/path/to/epoch=XX.ckpt \
      [data.mode=bbox] [out_dir=outputs/lidar4d_eval] [device='[4]'] \
      [test.velocity_source=total|init|match|zero]
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.model_wrapper import ModelWrapper
from src.config_loader import (
    DYNAMIC_VARIANTS,
    assert_model_variant_implemented,
    resolve_eval_config,
)
from src.dataloader.nuscene import multiframe_collate_fn
from src.dataloader.nuscene_lidar4d_test import LiDAR4DNuScenesTestDataset
from src.eval.gslidar_metrics import (
    MetricBackends, depth_errors, intensity_errors, raydrop_errors, point_metrics,
)
from src.eval.gaussian_viz import (
    gaussians_from_output, save_sequence_html,
    surfels_from_output, save_sequence_surfel_html,
    velocity_transport_from_output,
)
from src.eval.gaussian_stats import collect_window_stats, analyze_gaussian_sizes
from src.models_new.utils.graphics_utils import lidar4d_range_image_to_points
from src.models_new.utils.render import visualize_depth

RAYDROP_THRESHOLD = 0.5
EVAL_MAX_DEPTH_M = 80.0


def to_device(obj, device):
    """Move all tensors (leave Camera/other objects untouched -- render moves
    their tensors internally)."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def load_model(cfg, ckpt_path, device, *, allow_checkpoint_mismatch=False):
    model = ModelWrapper(cfg)
    if ckpt_path:
        blob = torch.load(ckpt_path, map_location="cpu")
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] {ckpt_path}\n       loaded "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if (missing or unexpected) and not allow_checkpoint_mismatch:
            raise RuntimeError(
                "Checkpoint/config mismatch. Refusing to evaluate partial weights; "
                "pass allow_checkpoint_mismatch=true only for an intentional "
                f"ablation. missing={list(missing)}, unexpected={list(unexpected)}"
            )
    else:
        print("[ckpt] WARNING: no checkpoint -> randomly initialized weights")
    model.eval().to(device)
    return model


def _viz_depth_black_holes(depth_1hw, near=2, far=50):
    """Colorize depth (turbo), but paint no-return pixels (depth==0) BLACK so
    they are not confused with near-range red (the -log curve maps both 0 and
    near depth to turbo's red end)."""
    vis = visualize_depth(depth_1hw, near=near, far=far)      # [3,H,W]
    keep = (depth_1hw[0] > 0).to(vis.dtype).to(vis.device)   # [H,W]
    return vis * keep


def _binary_raydrop(raydrop, threshold=RAYDROP_THRESHOLD):
    """Return 1 for predicted/GT drop and 0 for non-drop (keep)."""
    return (raydrop > float(threshold)).to(raydrop.dtype)


def prediction_points_ref(depth, raydrop, camera):
    """Back-project raydrop-kept prediction ranges into the Gaussian ref frame.

    ``lidar4d_range_image_to_points`` returns points in the target panorama's
    swapped camera frame.  The surfel meshes and raw LiDAR overlays live in the
    window's frame-0 reference coordinates, so apply the camera-to-ref matrix
    before serializing the point trace.
    """
    points_camera = lidar4d_range_image_to_points(
        depth,
        camera.vfov,
        camera.hfov,
        row_to_theta=camera.row_to_theta,
        raydrop=raydrop,
        raydrop_threshold=RAYDROP_THRESHOLD,
        max_range=EVAL_MAX_DEPTH_M,
    )
    c2w = camera.c2w.to(device=points_camera.device, dtype=points_camera.dtype)
    return points_camera @ c2w[:3, :3].T + c2w[:3, 3]


def save_viz(
    out_png,
    depth_keep,
    gt_depth,
    intensity_keep,
    gt_intensity,
    raydrop,
    gt_raydrop,
):
    pred_drop = _binary_raydrop(raydrop)
    gt_drop = _binary_raydrop(gt_raydrop)
    rows = [
        _viz_depth_black_holes(depth_keep),                   # pred depth  [3,H,W]
        _viz_depth_black_holes(gt_depth),                     # gt depth (holes=black)
        intensity_keep.clamp(0, 1).repeat(3, 1, 1),           # pred intensity
        gt_intensity.clamp(0, 1).repeat(3, 1, 1),             # gt intensity
        pred_drop.repeat(3, 1, 1),                            # pred drop, >0.5 = white
        gt_drop.repeat(3, 1, 1),                              # GT drop, 1 = white
    ]
    grid = make_grid(torch.stack([r.detach().cpu() for r in rows], dim=0), nrow=1)
    save_image(grid, str(out_png))


def _agg(window_metrics, key, sub):
    vals = [w[key][sub] for w in window_metrics if not np.isnan(w[key][sub])]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(window_metrics):
    out = {}
    for key, subs in (("depth", ["rmse", "medae", "lpips", "ssim", "psnr"]),
                      ("intensity", ["rmse", "medae", "lpips", "ssim", "psnr"]),
                      ("raydrop", ["rmse", "acc", "f1"]),
                      ("points", ["cd", "fscore"])):
        out[key] = {s: _agg(window_metrics, key, s) for s in subs}
    return out


def select_render_velocity(batch_gaussians, velocity_source: str):
    """Select the checkpoint velocity component used by the renderer.

    ``total`` preserves the trained model output. ``init`` uses the dustbin-aware
    proposal, while ``match`` uses its conditional matched velocity before the
    dustbin probability scale. Both are inference-only component ablations.
    ``zero`` freezes every Gaussian at its source position, measuring how much
    the velocity field contributes to reconstruction at all.
    """
    velocity_source = str(velocity_source).lower()
    if velocity_source not in {"total", "init", "match", "zero"}:
        raise ValueError(
            "test.velocity_source must be one of "
            "{'total', 'init', 'match', 'zero'}, got "
            f"{velocity_source!r}"
        )
    if velocity_source == "total":
        return batch_gaussians

    if velocity_source == "zero":
        selected = []
        for item in batch_gaussians:
            if item is None:
                selected.append(None)
                continue
            velocity = item.get("velocity")
            if not torch.is_tensor(velocity):
                raise KeyError("test.velocity_source=zero requires velocity")
            selected.append({**item, "velocity": torch.zeros_like(velocity)})
        return selected

    selected = []
    for batch_index, item in enumerate(batch_gaussians):
        if item is None:
            selected.append(None)
            continue
        velocity = item.get("velocity")
        source_key = f"velocity_{velocity_source}"
        selected_velocity = item.get(source_key)
        if not torch.is_tensor(selected_velocity):
            raise KeyError(
                f"test.velocity_source={velocity_source} requires {source_key} "
                "in every "
                f"Gaussian batch item; missing at batch index {batch_index}"
            )
        if (
            not torch.is_tensor(velocity)
            or selected_velocity.shape != velocity.shape
        ):
            raise ValueError(
                f"{source_key} must be a tensor aligned one-to-one with velocity"
            )
        selected.append({**item, "velocity": selected_velocity})
    return selected


# ---------------------------------------------------------------------------
@torch.no_grad()
def main(cfg, config_source="unspecified"):
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = f"cuda:{int(cfg.device[0])}" if len(cfg.device) else "cuda:0"
    torch.cuda.set_device(device)
    out_dir = Path(str(cfg.get("out_dir", os.path.join(cfg.logger.dir, "lidar4d_eval"))))
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = out_dir / "effective_config.yaml"
    OmegaConf.save(config=cfg, f=str(effective_config_path), resolve=True)

    print(
        f"[setup] device={device} seed={seed} "
        f"mode={cfg.data.mode} out_dir={out_dir}"
    )
    print(f"[config] source={config_source}")
    print(f"[config] effective={effective_config_path}")
    print("[setup] scale_factor=1.0 (depth already in meters; GS-LiDAR rescale N/A)")
    depth_statistic = str(cfg.get("depth_statistic", "median")).lower()
    depth_key = {"mean": "depth", "median": "depth_median"}.get(depth_statistic)
    if depth_key is None:
        raise ValueError(
            "depth_statistic must be one of {'mean', 'median'}, got "
            f"{depth_statistic!r}"
        )
    print(
        f"[setup] depth_statistic={depth_statistic} "
        f"metric_clamp=[1e-6,{EVAL_MAX_DEPTH_M:g}]m"
    )
    velocity_source = str(
        OmegaConf.select(cfg, "test.velocity_source", default="total")
    ).lower()
    if velocity_source not in {"total", "init", "match", "zero"}:
        raise ValueError(
            "test.velocity_source must be one of "
            "{'total', 'init', 'match', 'zero'}, got "
            f"{velocity_source!r}"
        )

    model = load_model(
        cfg,
        cfg.test.ckpt_path,
        device,
        allow_checkpoint_mismatch=bool(cfg.get("allow_checkpoint_mismatch", False)),
    )
    transport_mode = getattr(model.g2p_model, "transport_mode", None)
    if transport_mode not in {"bbox", "velocity"}:
        raise RuntimeError(
            "Gaussian renderer must declare transport_mode as bbox or velocity"
        )
    show_velocity_vectors = transport_mode == "velocity"
    if velocity_source != "total" and not show_velocity_vectors:
        raise ValueError(
            "Non-total test.velocity_source ablations are only available for "
            "velocity-transport models"
        )
    print(
        f"[setup] gaussian_transport={transport_mode} "
        f"velocity_arrows={'on' if show_velocity_vectors else 'off'}"
        + (
            " (source + velocity * actual signed delta_t -> target)"
            if show_velocity_vectors else ""
        )
    )
    print(
        f"[setup] render_velocity={velocity_source}"
        + (
            " (inference-only velocity ablation; weights unchanged)"
            if velocity_source != "total" else ""
        )
    )
    backends = MetricBackends(lpips_net="alex")

    dataset = LiDAR4DNuScenesTestDataset(cfg.data, split=cfg.data.test_split)
    target_cam = dataset.target_cam_index
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=multiframe_collate_fn)

    vfov = tuple(cfg.data.vfov)
    window_metrics = []  # list of dicts (with seq_name / target_s)
    seq_gauss = defaultdict(list)   # seq_name -> center-point HTML frames
    seq_surfel = defaultdict(list)  # seq_name -> [{label, static, dynamic, boxes}] 1σ-surfel HTML
    gauss_dir = viz_dir / "gaussians"
    gauss_dir.mkdir(parents=True, exist_ok=True)
    size_records = []                     # per-window Gaussian-size stats (all)
    size_by_split = defaultdict(list)     # per nuScenes split

    limit = int(cfg.get("limit", 0)) or len(dataset)
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        meta = dataset.index_meta[i]
        seq_name, target_s = meta["seq_name"], meta["target_s"]
        scene, nuscenes_split = meta["scene"], meta["nuscenes_split"]
        batch = to_device(batch, device)
        _input, gt = batch["input"], batch["gt"]

        out = model.p2g_model(
            _input,
            target_pose=gt.get("pose"),
            target_timestamps=gt.get("timestamps"),
        )
        if model.model_variant in DYNAMIC_VARIANTS:
            out = model.g2g_model(
                out,
                _input["timestamps_sec"],
                _input["window_duration_sec"],
            )
        else:
            out = model.g2g_model(out, _input["timestamps"])
        out = select_render_velocity(out, velocity_source)
        renders = model.g2p_model(out, gt)

        b = 0
        # Keep the image metrics, point metrics, PNGs, and predicted LiDAR HTML
        # on the same renderer statistic. This evaluator defaults to median
        # depth; pass depth_statistic=mean for an explicit ablation.
        depth = renders[depth_key][b, target_cam]           # [1,H,W]
        intensity = renders["intensity_sh"][b, target_cam]
        raydrop = renders["raydrop"][b, target_cam]
        gt_depth = renders["gt_depth"][b, target_cam]
        gt_intensity = renders["gt_intensity_sh"][b, target_cam]
        gt_raydrop = renders["gt_raydrop"][b, target_cam]

        # GS-LiDAR: zero out predicted-drop pixels before depth/intensity/points.
        keep = 1.0 - _binary_raydrop(raydrop).to(depth.dtype)
        depth_keep = depth * keep
        intensity_keep = intensity * keep

        gt_cam = gt["cameras"][b][target_cam]
        row_to_theta = gt_cam.row_to_theta
        pred_pts = prediction_points_ref(depth, raydrop, gt_cam).detach().cpu().numpy()

        wm = {
            "seq_name": seq_name,
            "target_s": target_s,
            "scene": scene,
            "nuscenes_split": nuscenes_split,
            # GS-LiDAR-style capped range-image metric: no-return stays zero
            # until depth_errors maps it to epsilon, while positive Pred/GT
            # returns above 80 m saturate at 80 m.
            "depth": depth_errors(
                depth_keep, gt_depth, backends,
                max_depth=EVAL_MAX_DEPTH_M,
            ),
            "intensity": intensity_errors(intensity_keep, gt_intensity, backends),
            "raydrop": raydrop_errors(
                raydrop, gt_raydrop, ratio=RAYDROP_THRESHOLD
            ),
            "points": point_metrics(
                depth_keep, gt_depth, row_to_theta, backends,
                vfov=vfov, far=EVAL_MAX_DEPTH_M,
            ),
        }
        window_metrics.append(wm)
        save_viz(viz_dir / f"{seq_name}_T{target_s}s.png",
                 depth_keep, gt_depth, intensity_keep, gt_intensity,
                 raydrop, gt_raydrop)

        # Raw LiDAR in the same (ref) frame as the Gaussian means, so the HTML
        # can show whether a Gaussian actually sits on a measured surface: the
        # two input endpoints (what the model saw) and the GT sweep at the
        # target time (what it must reproduce).
        in_pts = _input["lidar_points"][:, :3].detach().cpu().numpy()
        gt_off = gt["offset"].tolist()
        gt_start = ([0] + gt_off[:-1])[target_cam]
        gt_pts = gt["lidar_points"][gt_start:gt_off[target_cam], :3].detach().cpu().numpy()

        # Gaussian centers + 1σ surfels at this window's target time (per-seq HTML).
        # Viewpoint mode stores Common once and Additional per target view;
        # diagnostics must build the same transient union as the renderer.
        target_gaussians = model.g2p_model.select_target_view(out[b], target_cam)
        render_t = float(model.g2p_model.render_timestamp(gt_cam))
        if show_velocity_vectors:
            transport = velocity_transport_from_output(
                model.g2p_model, target_gaussians, render_t
            )
            source_times = transport["source_times_sec"]
            source_delta_t = render_t - source_times
            delta_label = ", ".join(
                f"{float(value):+.6f}" for value in source_delta_t
            )
            center_frame = {
                "label": (
                    f"T={target_s}s | target={render_t:.6f}s | "
                    f"source→target Δt=[{delta_label}]s"
                ),
                "source_frame_centers": transport[
                    "target_centers_by_source_frame"
                ],
                "velocity_origins": transport["source_centers"],
                "velocity_mps": transport["velocity_mps"],
                "velocity_delta_t_sec": transport["delta_t_sec"],
                "velocity_source_frame_index": transport[
                    "source_frame_index"
                ],
                "input_points": in_pts,
                "gt_points": gt_pts,
            }
        else:
            static_xyz, dynamic_xyz = gaussians_from_output(
                model.g2p_model, target_gaussians, render_t
            )
            center_frame = {
                "label": f"T={target_s}s",
                "static": static_xyz,
                "dynamic": dynamic_xyz,
                "input_points": in_pts,
                "gt_points": gt_pts,
            }
        seq_gauss[seq_name].append(center_frame)
        surf = surfels_from_output(
            model.g2p_model, target_gaussians, render_t
        )
        seq_surfel[seq_name].append({
            "label": f"T={target_s}s", "static": surf["static"],
            "dynamic": surf["dynamic"], "boxes": surf["boxes"],
            "input_points": in_pts, "gt_points": gt_pts,
            "pred_points": pred_pts})

        # per-window Gaussian-size statistics (effective radius vs geometry)
        st = collect_window_stats(
            model.g2p_model, target_gaussians, render_t
        )
        size_records.append(st)
        size_by_split[nuscenes_split].append(st)

        print(f"[{i+1:02d}/{len(dataset)}] {seq_name} T={target_s}s | "
              f"depth RMSE={wm['depth']['rmse']:.3f} PSNR={wm['depth']['psnr']:.2f} | "
              f"int RMSE={wm['intensity']['rmse']:.3f} | "
              f"CD={wm['points']['cd']:.4f} F={wm['points']['fscore']:.3f}")

    # ── per-sequence 3D Gaussian HTML (buttons 1..4 = target seconds) ───────
    for seq_name, frames in seq_gauss.items():
        sp = next((w["nuscenes_split"] for w in window_metrics
                   if w["seq_name"] == seq_name), "")
        save_sequence_html(gauss_dir / f"{seq_name}.html",
                           title=(
                               f"{seq_name}  ({sp})  — Gaussian centers "
                               f"[velocity={velocity_source}]"
                           ), frames=frames)
        save_sequence_surfel_html(gauss_dir / f"{seq_name}_surfel.html",
                                  title=f"{seq_name}  ({sp})", frames=seq_surfel[seq_name])
    print(f"[gaussians] wrote {len(seq_gauss)} center + {len(seq_surfel)} surfel HTML -> {gauss_dir}")

    # ── Gaussian-size statistical analysis (overall + per nuScenes split) ────
    print("\n" + "-" * 70)
    print(analyze_gaussian_sizes(size_records, out_dir, tag="all"))
    for sp in sorted(size_by_split):
        print(analyze_gaussian_sizes(size_by_split[sp], out_dir, tag=sp))

    # ── aggregate (overall + per nuScenes split + per sequence) ─────────────
    overall = summarize(window_metrics)
    per_seq = {}
    by_seq = defaultdict(list)
    for wm in window_metrics:
        by_seq[wm["seq_name"]].append(wm)
    for name, wms in by_seq.items():
        per_seq[name] = summarize(wms)

    # val = clean held-out; train = seen during training (leakage) for a
    # feed-forward model trained on the nuScenes train split.
    per_split = {}
    by_split = defaultdict(list)
    for wm in window_metrics:
        by_split[wm["nuscenes_split"]].append(wm)
    for sp, wms in by_split.items():
        per_split[sp] = summarize(wms)
        per_split[sp]["sequences"] = sorted({w["seq_name"] for w in wms})
        per_split[sp]["num_windows"] = len(wms)

    payload = {
        "config": {"ckpt": cfg.test.ckpt_path, "mode": cfg.data.mode,
                   "version": cfg.data.version, "scale_factor": 1.0,
                   "seed": seed,
                   "source": config_source,
                   "effective_config": str(effective_config_path),
                   "depth_statistic": depth_statistic,
                   "max_depth_m": EVAL_MAX_DEPTH_M,
                   "gt_range_policy": "clamp_positive_returns",
                   "velocity_source": velocity_source,
                   "vfov": list(vfov), "target_seconds": [1, 2, 3, 4]},
        "overall": overall,
        "per_nuscenes_split": per_split,
        "per_sequence": per_seq,
        "windows": window_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ── print table ─────────────────────────────────────────────────────────
    def fmt(s):
        d, it, rd, pt = s["depth"], s["intensity"], s["raydrop"], s["points"]
        return (f"  depth : RMSE={d['rmse']:.3f} MedAE={d['medae']:.3f} "
                f"LPIPS={d['lpips']:.3f} SSIM={d['ssim']:.3f} PSNR={d['psnr']:.2f}\n"
                f"  inten : RMSE={it['rmse']:.3f} MedAE={it['medae']:.3f} "
                f"LPIPS={it['lpips']:.3f} SSIM={it['ssim']:.3f} PSNR={it['psnr']:.2f}\n"
                f"  raydp : RMSE={rd['rmse']:.3f} Acc={rd['acc']:.3f} F1={rd['f1']:.3f}\n"
                f"  point : CD={pt['cd']:.4f} F-score@0.05={pt['fscore']:.3f}")

    seq_split = {wm["seq_name"]: wm["nuscenes_split"] for wm in window_metrics}
    print("\n" + "=" * 70)
    print("PER-SEQUENCE")
    for name in sorted(per_seq):
        print(f"[{name}  ({seq_split[name]})]\n{fmt(per_seq[name])}")
    print("-" * 70)
    print("PER nuScenes SPLIT  (val = clean held-out; train = SEEN during training)")
    for sp in sorted(per_split):
        print(f"<{sp}>  seqs={per_split[sp]['sequences']}  "
              f"n={per_split[sp]['num_windows']}\n{fmt(per_split[sp])}")
    print("-" * 70)
    print(f"OVERALL (mean over {len(window_metrics)} windows)")
    print(fmt(overall))
    print("=" * 70)
    print(f"\nsaved: {out_dir/'metrics.json'}  |  viz: {viz_dir}")


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    cfg, config_source = resolve_eval_config(cli)
    assert_model_variant_implemented(cfg)
    cfg.mode = "test"
    main(cfg, config_source=config_source)
