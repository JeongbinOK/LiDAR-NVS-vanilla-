"""Training loop for neural 2D Gaussian clustering."""

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")

# assign_full.T.topk() 등 비연속 텐서 topk에서 임시 contiguous copy가 생성되어
# 메모리 단편화가 심해질 수 있음. expandable_segments는 CUDA VMM API로 단편화 완화.
# import torch 이전에 설정해야 allocator 초기화에 반영됨.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add external dataloader path
from config import QGSConfig
sys.path.insert(0, os.path.join(os.path.expanduser(QGSConfig.data_root), "loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn


from nn.model import QGSModel
# TODO: Import your QGSLoss here
# from nn.qgs_loss import QGSLoss


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


def evaluate_geometric(xyz: torch.Tensor, output: dict) -> dict:
    """Evaluate QGS predictions."""
    # TODO: Implement geometry evaluation (e.g., PSNR, depth metrics)
    return {
        "dummy_metric": 0.0
    }


def process_frame(model, loss_fn, pts, device, ego_radius):
    """Run model on a single frame and return loss dict + output."""
    xyz = pts[:, :3].to(device)
    intensity = pts[:, 3:4].to(device)

    # Ego-vehicle mask
    mask = torch.norm(xyz, dim=1) > ego_radius
    xyz, intensity = xyz[mask], intensity[mask]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(xyz, intensity)
        
        # TODO: compute actual loss
        # loss_dict = loss_fn(xyz, output)
        
        # Skeleton dummy loss for now to keep train loop running
        dummy_loss = output["features"].sum() * 0.0
        loss_dict = {"total": dummy_loss}
        
    return loss_dict, output, xyz


def train(cfg: QGSConfig, overfit_frames: int = 0, resume: str = ""):
    """Main training loop."""
    device = cfg.device
    
    # Create per-run output directory
    run_dir = _make_run_dir(os.path.join(os.path.dirname(__file__), "outputs"))
    ckpt_dir = os.path.join(run_dir, "ckpt")
    cfg_dir = os.path.join(run_dir, "configs")

    # Save config
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
        
    # Save git info for reproducibility
    try:
        import subprocess
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        git_diff = subprocess.check_output(["git", "diff", "--stat"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        
        git_info = {
            "commit_hash": commit_hash,
            "branch": branch_name,
            "is_dirty": bool(git_diff),
            "diff_stat": git_diff,
        }
        with open(os.path.join(cfg_dir, "git_info.json"), "w") as f:
            json.dump(git_info, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save git info: {e}")

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

    model = QGSModel(cfg).to(device)
    # TODO: Instantiate QGSLoss
    loss_fn = None
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
        # Setup metrics or schedulers specific to epoch here
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
                        model, loss_fn, frame_pts, device, cfg.ego_radius,
                    )

                    # Memory constraint checks can go here if needed

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
            pbar.set_postfix(loss=f"{avg_batch['total']:.4f}")

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
               f"{dt:.1f}s")

        if epoch_metrics:
            mg = {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                  for k in epoch_metrics[0]}
            log += f" | gamma={mg['gamma_rms']:.4f} cov={mg['coverage']:.3f} K={mg['num_gaussians']:.0f} vote_off={mg['vote_offset']:.3f}m mu_off={mg['mu_offset']:.3f}m"

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
    parser.add_argument("--seed-voxel-size", type=float, default=_defaults.seed_voxel_size,
                        help="Voxel size for seed center generation (WSL2 3090: use 4.0)")
    args = parser.parse_args()

    cfg = NeuralClusteringConfig(
        data_root=args.data_root,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        backbone_type=args.backbone,
        primitive_type=args.primitive,
        seed_voxel_size=args.seed_voxel_size,
    )

    train(cfg, overfit_frames=args.overfit, resume=args.resume)


if __name__ == "__main__":
    main()
