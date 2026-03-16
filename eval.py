"""
Evaluation pipeline for feed-forward LiDAR NVS.
Computes RMSE, MedAE, Chamfer Distance, F-Score per timestep.
"""
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
sys.path.insert(0, '/data1/nuScenes/loader')
# ─────────────────────────────────────────────────────────────────────────────

import json
import argparse
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import NuScenesNVSDataset, nvs_collate_fn

from model.model import FeedForwardGaussianModel
from renderer.render import render_full_pano
from utils.misc import seed_everything, points_to_pano, ego_mask
from utils.graphics import pano_to_lidar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party'))
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist


def prepare_gt_pano(gt_pts, gt_pose, vfov, hfov, H, W, ego_radius, device):
    """Transform GT points and create panoramic maps."""
    pts = gt_pts.to(device)
    pose = gt_pose.to(device)
    pts = ego_mask(pts, ego_radius)

    pts_homo = torch.cat([pts[:, :3], torch.ones(pts.shape[0], 1, device=device)], dim=1)
    pts_xyz = (pts_homo @ pose.T)[:, :3]
    pts_trans = torch.cat([pts_xyz, pts[:, 3:4]], dim=1)

    ri, _, _ = points_to_pano(pts_trans, vfov, hfov, H, W)
    return ri[0:1], ri[4:5]  # depth, intensity


def compute_metrics(pred_depth, gt_depth, vfov, hfov):
    """Compute depth and point cloud metrics for a single panorama pair."""
    mask = gt_depth > 0
    if mask.sum() == 0:
        return {'rmse': 0.0, 'medae': 0.0, 'chamfer': 0.0, 'f_score': 0.0}

    pred_vals = pred_depth[mask]
    gt_vals = gt_depth[mask]

    rmse = torch.sqrt(((pred_vals - gt_vals) ** 2).mean()).item()
    medae = torch.median(torch.abs(pred_vals - gt_vals)).item()

    mask_f = (gt_depth > 0).float()
    pred_pts = pano_to_lidar(pred_depth * mask_f, vfov, hfov)
    gt_pts = pano_to_lidar(gt_depth, vfov, hfov)

    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return {'rmse': rmse, 'medae': medae, 'chamfer': 0.0, 'f_score': 0.0}

    cham_fn = chamfer_3DDist()
    d1, d2, _, _ = cham_fn(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0))
    cd = (d1.mean() + d2.mean()).item()

    threshold = 0.05
    precision = (d1 < threshold ** 2).float().mean().item()
    recall = (d2 < threshold ** 2).float().mean().item()
    f_score = 2 * precision * recall / max(precision + recall, 1e-8)

    return {'rmse': rmse, 'medae': medae, 'chamfer': cd, 'f_score': f_score}


@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(args.seed)

    dataset = NuScenesNVSDataset(
        dataroot=args.data.dataroot,
        version=args.data.version,
        split='val',
        frame_gap=args.data.frame_gap,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=nvs_collate_fn,
    )

    model = FeedForwardGaussianModel(args).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    vfov = tuple(args.vfov)
    hfov = tuple(args.hfov)
    H, W = args.H, args.W
    W_half = args.W_half
    ego_radius = args.ego_mask_radius

    per_step_metrics = defaultdict(lambda: defaultdict(list))
    overall_metrics = defaultdict(list)

    eval_dir = os.path.join(args.output_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc="Evaluating"):
        gaussian_params, aux = model(
            batch['input_0'], batch['input_1'],
            batch['input_1_pose'].to(device),
        )

        b = 0
        gts = batch['gts'][b]
        gt_poses = batch['gts_poses'][b].to(device)
        gt_timestamps = batch['gts_timestamps'][b]

        gp_b = {k: v[b] for k, v in gaussian_params.items()}

        for idx in range(len(gts)):
            t_q = gt_timestamps[idx].item()

            pred_d, pred_i, _, _ = render_full_pano(gp_b, t_q, vfov, H, W_half)
            gt_d, gt_i = prepare_gt_pano(
                gts[idx], gt_poses[idx], vfov, hfov, H, W, ego_radius, device)

            m = compute_metrics(pred_d, gt_d, vfov, hfov)

            t_bin = round(t_q / 0.05) * 0.05
            for k, v in m.items():
                per_step_metrics[f'{t_bin:.2f}'][k].append(v)
                overall_metrics[k].append(v)

    results = {'overall': {}, 'per_timestep': {}}
    for k, vals in overall_metrics.items():
        results['overall'][k] = float(np.mean(vals))

    for t_key in sorted(per_step_metrics.keys()):
        results['per_timestep'][t_key] = {}
        for k, vals in per_step_metrics[t_key].items():
            results['per_timestep'][t_key][k] = float(np.mean(vals))

    results_path = os.path.join(eval_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Results ===")
    print(f"Overall RMSE: {results['overall']['rmse']:.4f}")
    print(f"Overall MedAE: {results['overall']['medae']:.4f}")
    print(f"Overall Chamfer: {results['overall']['chamfer']:.6f}")
    print(f"Overall F-Score@0.05m: {results['overall']['f_score']:.4f}")
    print(f"\nPer-timestep CD:")
    for t_key in sorted(results['per_timestep'].keys()):
        cd = results['per_timestep'][t_key]['chamfer']
        print(f"  t={t_key}: CD={cd:.6f}")
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    cli_args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(cli_args.config)
    cli_overrides = OmegaConf.from_cli(unknown)
    args = OmegaConf.merge(cfg, cli_overrides)
    args.checkpoint = cli_args.checkpoint

    evaluate(args)
