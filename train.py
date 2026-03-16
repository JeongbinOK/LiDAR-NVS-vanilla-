"""
Training loop for feed-forward LiDAR NVS with 2D Gaussians.
"""
import os
import sys

# ── Path setup (must be before any project imports) ──────────────────────────
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
sys.path.insert(0, '/data1/nuScenes/loader')
# ─────────────────────────────────────────────────────────────────────────────

import json
import time
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

from dataset import NuScenesNVSDataset, nvs_collate_fn

from model.model import FeedForwardGaussianModel
from renderer.render import render_full_pano
from losses.losses import compute_total_loss
from utils.misc import seed_everything, points_to_pano, ego_mask


def visualize_depth(depth, vmin=0.0, vmax=50.0):
    """Normalize depth to [0,1] for WandB image logging."""
    d = depth.float().clamp(vmin, vmax)
    d = (d - vmin) / (vmax - vmin + 1e-8)
    return d


def prepare_gt_pano(gt_pts, gt_pose, vfov, hfov, H, W, ego_radius, device):
    """Transform GT points to input_0 frame and create panoramic depth/intensity maps."""
    pts = gt_pts.to(device)
    pose = gt_pose.to(device)
    pts = ego_mask(pts, ego_radius)
    pts_homo = torch.cat([pts[:, :3], torch.ones(pts.shape[0], 1, device=device)], dim=1)
    pts_xyz = (pts_homo @ pose.T)[:, :3]
    pts_trans = torch.cat([pts_xyz, pts[:, 3:4]], dim=1)
    ri, _, _ = points_to_pano(pts_trans, vfov, hfov, H, W)
    return ri[0:1], ri[4:5]  # gt_depth, gt_intensity


def get_curriculum_config(epoch, total_epochs, curriculum_cfg):
    """Determine training parameters based on curriculum phase."""
    frac = epoch / total_epochs
    if frac < curriculum_cfg['phase1_end']:
        return {'num_steps': curriculum_cfg['phase1_num_steps'],
                'front_only': curriculum_cfg['phase1_front_only'],
                'use_chamfer': False, 'use_smooth': False,
                'use_intensity': False, 'use_reg': False,
                'use_normal': False, 'use_dvar': False,
                'use_temporal': False, 'phase': 1}
    elif frac < curriculum_cfg['phase2_end']:
        return {'num_steps': curriculum_cfg['phase2_num_steps'],
                'front_only': False,
                'use_chamfer': True, 'use_smooth': True,
                'use_intensity': False, 'use_reg': False,
                'use_normal': False, 'use_dvar': False,
                'use_temporal': False, 'phase': 2}
    else:
        return {'num_steps': 9, 'front_only': False,
                'use_chamfer': True, 'use_smooth': True,
                'use_intensity': True, 'use_reg': True,
                'use_normal': True, 'use_dvar': True,
                'use_temporal': True, 'phase': 3}


def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ckpt'), exist_ok=True)

    # WandB init
    run = wandb.init(
        project=getattr(args, 'wandb_project', 'lidar-nvs'),
        name=getattr(args, 'wandb_run_name', os.path.basename(output_dir)),
        config=OmegaConf.to_container(args, resolve=True),
        dir=output_dir,
        resume='allow',
    )
    print(f"WandB run: {run.url}")

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(args, f)

    seed_everything(args.seed)

    # Dataset
    dataset = NuScenesNVSDataset(
        dataroot=args.data.dataroot,
        version=args.data.version,
        split='train',
        frame_gap=args.data.frame_gap,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=nvs_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = FeedForwardGaussianModel(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params*4/1e6:.1f} MB)")

    # Optimizer
    encoder_params = (list(model.encoder.parameters()) +
                      list(model.time_embed.parameters()) +
                      list(model.fusion.parameters()))
    decoder_params = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr.encoder},
        {'params': decoder_params, 'lr': args.lr.decoder},
    ], weight_decay=args.optimizer.weight_decay)

    total_steps = args.epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)
    warmup_steps = args.lr.warmup_steps

    vfov = tuple(args.vfov)
    hfov = tuple(args.hfov)
    H, W, W_half = args.H, args.W, args.W_half
    ego_radius = args.ego_mask_radius

    global_step = 0
    ema_loss = 0.0
    start_epoch = 0
    t_train_start = time.time()

    if args.get('resume'):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['global_step']
        ema_loss = ckpt['ema_loss']
        start_epoch = ckpt['epoch']
        print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    print(f"Training on {device} | {args.epochs} epochs x {len(dataloader)} iters/epoch")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"VRAM used before training: "
          f"{torch.cuda.memory_allocated(device)/1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(device).total_mem/1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(device), 'total_mem')
          else f"{torch.cuda.memory_allocated(device)/1e9:.2f} GB / "
               f"{torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        cur_cfg = get_curriculum_config(epoch, args.epochs, args.curriculum)

        # Adjust loss weights for curriculum phase
        loss_weights = dict(args.loss)
        if not cur_cfg['use_chamfer']:
            loss_weights['chamfer'] = 0.0
        if not cur_cfg['use_smooth']:
            loss_weights['smooth'] = 0.0
        if not cur_cfg['use_intensity']:
            loss_weights['intensity_l1'] = 0.0
        if not cur_cfg['use_reg']:
            loss_weights['velocity_reg'] = 0.0
            loss_weights['opacity_entropy'] = 0.0
        if not cur_cfg['use_normal']:
            loss_weights['normal_consistency'] = 0.0
        if not cur_cfg['use_dvar']:
            loss_weights['depth_var'] = 0.0
        if not cur_cfg['use_temporal']:
            loss_weights['temporal_smooth'] = 0.0

        # Log curriculum phase change
        wandb.log({'train/curriculum_phase': cur_cfg['phase']}, step=global_step)

        epoch_loss_accum = defaultdict(float)
        epoch_batches = 0
        t_epoch_start = time.time()

        pbar = tqdm(dataloader,
                    desc=f"Ep {epoch+1:03d}/{args.epochs} [P{cur_cfg['phase']}]",
                    dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            t_iter_start = time.time()
            global_step += 1

            # Linear warmup
            if global_step <= warmup_steps:
                warmup_factor = global_step / warmup_steps
                for i, pg in enumerate(optimizer.param_groups):
                    base_lr = args.lr.encoder if i == 0 else args.lr.decoder
                    pg['lr'] = base_lr * warmup_factor

            # ── Forward pass ──────────────────────────────────────────────
            gaussian_params, aux = model(
                batch['input_0'], batch['input_1'],
                batch['input_1_pose'].to(device),
            )

            B = len(batch['input_0'])
            total_loss = torch.tensor(0.0, device=device)
            total_loss_dict = defaultdict(float)

            for b in range(B):
                gts = batch['gts'][b]
                gt_poses = batch['gts_poses'][b].to(device)
                gt_timestamps = batch['gts_timestamps'][b]
                n_gt = len(gts)
                if n_gt == 0:
                    continue

                n_use = min(cur_cfg['num_steps'], n_gt)
                indices = sorted(torch.randperm(n_gt)[:n_use].tolist()) \
                    if n_use < n_gt else list(range(n_gt))

                gp_b = {k: v[b] for k, v in gaussian_params.items()}

                pred_depths, pred_intensities = [], []
                gt_depths, gt_intensities = [], []
                depth_sqs, rendered_normals = [], []

                for idx in indices:
                    t_q = gt_timestamps[idx].item()
                    pred_d, pred_i, dsq, norm = render_full_pano(
                        gp_b, t_q, vfov, H, W_half)
                    pred_depths.append(pred_d)
                    pred_intensities.append(pred_i)
                    depth_sqs.append(dsq)
                    rendered_normals.append(norm)
                    gt_d, gt_i = prepare_gt_pano(
                        gts[idx], gt_poses[idx], vfov, hfov, H, W,
                        ego_radius, device)
                    gt_depths.append(gt_d)
                    gt_intensities.append(gt_i)

                boundary_pred, boundary_gt = [], []
                if loss_weights['boundary_depth'] > 0:
                    bp0, _, _, _ = render_full_pano(gp_b, 0.0, vfov, H, W_half)
                    boundary_pred.append(bp0)
                    boundary_gt.append(aux['range_img_0'][b, 0:1])
                    bp1, _, _, _ = render_full_pano(gp_b, 0.5, vfov, H, W_half)
                    boundary_pred.append(bp1)
                    boundary_gt.append(aux['range_img_1'][b, 0:1])

                gp_b_list = {k: [v] for k, v in gp_b.items()}
                loss, ld = compute_total_loss(
                    pred_depths, pred_intensities,
                    gt_depths, gt_intensities,
                    gp_b_list, loss_weights, vfov, hfov,
                    boundary_pred or None, boundary_gt or None,
                    depth_sqs if loss_weights.get('depth_var', 0) > 0 else None,
                    rendered_normals if loss_weights.get('normal_consistency', 0) > 0 else None,
                )
                total_loss = total_loss + loss / B
                for k, v in ld.items():
                    total_loss_dict[k] += v / B

            # ── Backward + optimizer step ─────────────────────────────────
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if global_step > warmup_steps:
                scheduler.step()
            wandb.log({'train/grad_norm': grad_norm.item()}, step=global_step)

            # ── EMA loss & tqdm ───────────────────────────────────────────
            loss_val = total_loss.item()
            ema_loss = 0.9 * ema_loss + 0.1 * loss_val
            for k, v in total_loss_dict.items():
                epoch_loss_accum[k] += v
            epoch_batches += 1

            iter_ms = (time.time() - t_iter_start) * 1000
            vram_gb = torch.cuda.memory_allocated(device) / 1e9
            vram_peak = torch.cuda.max_memory_allocated(device) / 1e9

            if global_step % args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{ema_loss:.4f}',
                    'depth': f'{total_loss_dict.get("depth_l1", 0):.4f}',
                    'lr_enc': f'{optimizer.param_groups[0]["lr"]:.1e}',
                    'vram': f'{vram_gb:.1f}G',
                    'ms': f'{iter_ms:.0f}',
                }, refresh=False)

                log_dict = {
                    'train/loss_ema':   ema_loss,
                    'train/loss_total': loss_val,
                    'train/lr_encoder': optimizer.param_groups[0]['lr'],
                    'train/lr_decoder': optimizer.param_groups[1]['lr'],
                    'monitor/vram_gb':       vram_gb,
                    'monitor/vram_peak_gb':  vram_peak,
                    'monitor/iter_ms':       iter_ms,
                }
                for k, v in total_loss_dict.items():
                    log_dict[f'loss/{k}'] = v
                wandb.log(log_dict, step=global_step)

            # ── WandB: images + histograms every vis_interval steps ──────
            if global_step % args.vis_interval == 0:
                with torch.no_grad():
                    gp_vis = {k: v[0] for k, v in gaussian_params.items()}
                    pd_vis, pi_vis, _, _ = render_full_pano(
                        gp_vis, 0.25, vfov, H, W_half)

                    if len(batch['gts'][0]) > 0:
                        mid_idx = len(batch['gts'][0]) // 2
                        gd_vis, gi_vis = prepare_gt_pano(
                            batch['gts'][0][mid_idx],
                            batch['gts_poses'][0][mid_idx].to(device),
                            vfov, hfov, H, W, ego_radius, device)
                    else:
                        gd_vis = torch.zeros_like(pd_vis)
                        gi_vis = torch.zeros_like(pi_vis)

                    def to_wandb_img(t):
                        arr = t.squeeze(0).cpu().numpy()
                        return wandb.Image(arr)

                    gp0 = {k: v[0] for k, v in gaussian_params.items()}
                    wandb.log({
                        'vis/depth_pred_t0.25':  to_wandb_img(visualize_depth(pd_vis)),
                        'vis/depth_gt_t0.25':    to_wandb_img(visualize_depth(gd_vis)),
                        'vis/depth_input0':      to_wandb_img(visualize_depth(aux['range_img_0'][0, 0:1])),
                        'vis/intensity_pred':    to_wandb_img(pi_vis.clamp(0, 1)),
                        'vis/intensity_gt':      to_wandb_img(gi_vis.clamp(0, 1)),
                        'gaussians/n_points':    gp0['xyz'].shape[0],
                        'gaussians/opacity_mean':     gp0['opacity'].mean().item(),
                        'gaussians/velocity_norm_mean': gp0['velocity'].norm(dim=-1).mean().item(),
                    }, step=global_step)

        # ── End of epoch ─────────────────────────────────────────────────
        epoch_time = time.time() - t_epoch_start
        elapsed_total = time.time() - t_train_start

        epoch_log = {'epoch/time_sec': epoch_time}
        for k, v in epoch_loss_accum.items():
            epoch_log[f'epoch/{k}'] = v / max(epoch_batches, 1)
        wandb.log(epoch_log, step=global_step)

        eta_sec = (epoch_time * (args.epochs - epoch - 1))
        print(f"\n[Epoch {epoch+1}/{args.epochs}]  "
              f"loss={ema_loss:.4f}  "
              f"time={epoch_time/60:.1f}min  "
              f"elapsed={elapsed_total/3600:.1f}h  "
              f"ETA={eta_sec/3600:.1f}h  "
              f"VRAM={torch.cuda.memory_allocated(device)/1e9:.1f}/"
              f"{torch.cuda.get_device_properties(device).total_memory/1e9:.0f}GB")

        # Checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(output_dir, 'ckpt', f'epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_loss': ema_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    cli_args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(cli_args.config)
    cli_overrides = OmegaConf.from_cli(unknown)
    args = OmegaConf.merge(cfg, cli_overrides)

    # Default GPU field if not in config
    if not hasattr(args, 'gpu'):
        args.gpu = 0

    train(args)
