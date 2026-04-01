"""Training loop for neural 2D Gaussian clustering."""

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add external dataloader path
from config import NeuralClusteringConfig
sys.path.insert(0, os.path.join(os.path.expanduser(NeuralClusteringConfig.data_root), "loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn


from nn.model import NeuralClusteringModel
from nn.losses import ClusteringLoss


def _make_run_dir(base: str = "outputs") -> str:
    """Create outputs/train_NNN/ with configs/ and ckpt/ subdirs."""
    os.makedirs(base, exist_ok=True)
    existing = [
        d for d in os.listdir(base)
        if d.startswith("train_") and os.path.isdir(os.path.join(base, d))
    ]
    indices = []
    for d in existing:
        try:
            indices.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_idx = max(indices, default=0) + 1
    run_dir = os.path.join(base, f"train_{next_idx:03d}")
    os.makedirs(os.path.join(run_dir, "configs"))
    os.makedirs(os.path.join(run_dir, "ckpt"))
    return run_dir


def compute_tau(epoch: int, cfg: NeuralClusteringConfig) -> float:
    """Linear temperature annealing from tau_start to tau_end."""
    if cfg.num_epochs <= 1:
        return cfg.cluster_tau_end
    t = epoch / (cfg.num_epochs - 1)
    return cfg.cluster_tau_start + t * (cfg.cluster_tau_end - cfg.cluster_tau_start)


def evaluate_geometric(xyz: torch.Tensor, output: dict) -> dict:
    """Geometric self-evaluation metrics (no GT needed)."""
    gaussians = output["gaussians"]
    assign = output["assign"]  # [N, K] dense

    mu = gaussians["mu"]  # [K, 3]
    hard = assign.argmax(dim=1)  # [N]

    # Distance to assigned center
    dist = (xyz - mu[hard]).norm(dim=1)
    dist_rms = dist.pow(2).mean().sqrt().item()

    # Off-plane distance for 2D (normal exists)
    if "n" in gaussians:
        n = gaussians["n"]
        gamma = ((xyz - mu[hard]) * n[hard]).sum(dim=1)
        gamma_rms = gamma.pow(2).mean().sqrt().item()
    else:
        gamma_rms = dist_rms

    coverage = (assign.max(dim=1).values > 0.1).float().mean().item()

    # Issue 6 monitor: large offsets indicate assign-mu position mismatch
    offset_norm = output["offset"].norm(dim=1).mean().item()

    return {
        "gamma_rms": gamma_rms,
        "dist_rms": dist_rms,
        "coverage": coverage,
        "num_gaussians": mu.shape[0],
        "offset_norm": offset_norm,
    }


def process_frame(model, loss_fn, pts, device, tau, ego_radius):
    """Run model on a single frame and return loss dict + output."""
    xyz = pts[:, :3].to(device)
    intensity = pts[:, 3:4].to(device)

    # Ego-vehicle mask
    mask = torch.norm(xyz, dim=1) > ego_radius
    xyz, intensity = xyz[mask], intensity[mask]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(xyz, intensity, tau=tau)
        loss_dict = loss_fn(xyz, output)
    return loss_dict, output, xyz


def train(cfg: NeuralClusteringConfig, overfit_frames: int = 0, resume: str = ""):
    """Main training loop."""
    device = cfg.device
    
    # Create per-run output directory
    run_dir = _make_run_dir(os.path.join(os.path.dirname(__file__), "outputs"))
    ckpt_dir = os.path.join(run_dir, "ckpt")
    cfg_dir = os.path.join(run_dir, "configs")

    # Save config
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    print(f"Run dir: {run_dir}")

    data_root = os.path.expanduser(cfg.data_root)
    dataset = NuScenesNVSDataset(
        dataroot=data_root, version='v1.0-trainval', split='train',
    )

    if overfit_frames > 0:
        dataset.data_infos = dataset.data_infos[:overfit_frames]
        print(f"Overfit mode: {overfit_frames} pair(s)")

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=nvs_collate_fn, num_workers=4, persistent_workers=True,
    )

    model = NeuralClusteringModel(cfg).to(device)
    loss_fn = ClusteringLoss(
        primitive=cfg.primitive_type,
        top_k_assign=cfg.top_k_assign,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr * 1e-2,
    )
    
    print("Model Parameters per Module:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:<20}: {params:>12,}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,}")
    print(f"Dataset: {len(dataset)} pairs")
    print(f"Epochs: {cfg.num_epochs}, batch_size: {cfg.batch_size}")
    print("-" * 60)

    best_loss = float("inf")
    start_epoch = 0

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from {resume} (epoch {ckpt['epoch']+1}, loss={best_loss:.4f})")
        print("-" * 60)

    for epoch in range(start_epoch, cfg.num_epochs):
        tau = compute_tau(epoch, cfg)
        epoch_losses = []
        epoch_metrics = []
        t0 = time.time()

        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # batch['input_0']: List[Tensor(N, 4)], batch['input_1']: List[Tensor(M, 4)]
            B = len(batch['input_0'])
            batch_loss = 0.0
            batch_loss_dicts = []
            
            optimizer.zero_grad()
            valid_frames = 0
            eval_metric = None

            for b in range(B):
                for frame_pts in [batch['input_0'][b], batch['input_1'][b]]:
                    ld, output, xyz = process_frame(
                        model, loss_fn, frame_pts, device, tau, cfg.ego_radius,
                    )

                    if not torch.isfinite(ld["total"]):
                        pbar.write(f"  NaN/inf loss at batch {batch_idx} frame {valid_frames}, skipping frame")
                        continue

                    # Backward without scaling; gradients will be rescaled after the
                    # loop so that skipped frames don't under-weight valid ones.
                    ld["total"].backward()
                    valid_frames += 1
                    batch_loss += ld["total"].item()
                    batch_loss_dicts.append({k: v.item() for k, v in ld.items()})

                    if overfit_frames > 0 or (batch_idx + 1) % 50 == 0:
                        with torch.no_grad():
                            eval_metric = evaluate_geometric(xyz, output)

            if valid_frames == 0:
                pbar.write(f"  All frames NaN at batch {batch_idx}, skipping batch")
                optimizer.zero_grad()
                continue

            batch_loss = batch_loss / valid_frames

            # Normalize accumulated gradients by the actual number of valid frames.
            # When no frames are skipped valid_frames == B*2, so this is a no-op
            # compared to the old num_frames divisor.
            if valid_frames > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= valid_frames

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            has_bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if has_bad_grad:
                pbar.write(f"  Bad grad at batch {batch_idx}, skipping step")
                optimizer.zero_grad()
                continue

            optimizer.step()

            avg_batch = {k: sum(d[k] for d in batch_loss_dicts) / len(batch_loss_dicts)
                         for k in batch_loss_dicts[0]}
            epoch_losses.append(avg_batch)
            pbar.set_postfix(loss=f"{avg_batch['total']:.4f}", tau=f"{tau:.2f}")

            if eval_metric is not None:
                epoch_metrics.append(eval_metric)
                eval_metric = None

        dt = time.time() - t0
        if not epoch_losses:
            continue

        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
               for k in epoch_losses[0]}

        log = (f"[{epoch+1:3d}/{cfg.num_epochs}] "
               f"loss={avg['total']:.4f} "
               f"tau={tau:.2f} {dt:.1f}s"
               f" | s={avg['s_mean']:.3f} K={avg['K']:.0f}")

        if epoch_metrics:
            mg = {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                  for k in epoch_metrics[0]}
            log += f" | gamma={mg['gamma_rms']:.4f} cov={mg['coverage']:.3f} K={mg['num_gaussians']:.0f} off={mg['offset_norm']:.3f}m"

        tqdm.write(log)

        model_has_nan = any(
            torch.isnan(p).any() or torch.isinf(p).any()
            for p in model.parameters()
        )
        if avg["total"] < best_loss and not model_has_nan:
            best_loss = avg["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_loss,
            }, os.path.join(ckpt_dir, "best_model.pt"))

        scheduler.step()

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg["total"],
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"))

    print(f"\nDone. Best loss: {best_loss:.4f}")
    return model


def main():
    _defaults = NeuralClusteringConfig()
    parser = argparse.ArgumentParser(description="Train neural 2D Gaussian clustering")
    parser.add_argument("--data-root", default=_defaults.data_root)
    parser.add_argument("--epochs", type=int, default=_defaults.num_epochs)
    parser.add_argument("--lr", type=float, default=_defaults.lr)
    parser.add_argument("--batch-size", type=int, default=_defaults.batch_size)
    parser.add_argument("--overfit", type=int, default=0,
                        help="Overfit on N pairs (0=full training)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", default=_defaults.device)
    parser.add_argument("--backbone", default=_defaults.backbone_type,
                        choices=["ptv3", "custom"])
    parser.add_argument("--primitive", default=_defaults.primitive_type,
                        choices=["2d", "3d"])
    args = parser.parse_args()

    cfg = NeuralClusteringConfig(
        data_root=args.data_root,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        backbone_type=args.backbone,
        primitive_type=args.primitive,
    )

    train(cfg, overfit_frames=args.overfit, resume=args.resume)


if __name__ == "__main__":
    main()
