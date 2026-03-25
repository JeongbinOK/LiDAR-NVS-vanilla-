"""Training loop for neural 2D Gaussian clustering."""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add external dataloader path
sys.path.insert(0, os.path.expanduser("~/data/nuScenes/loader"))
from dataset import NuScenesNVSDataset, nvs_collate_fn

from config import NeuralClusteringConfig
from nn.model import NeuralClusteringModel
from nn.losses import ClusteringLoss


def compute_tau(epoch: int, cfg: NeuralClusteringConfig) -> float:
    """Linear temperature annealing from tau_start to tau_end."""
    if cfg.num_epochs <= 1:
        return cfg.gumbel_tau_end
    t = epoch / (cfg.num_epochs - 1)
    return cfg.gumbel_tau_start + t * (cfg.gumbel_tau_end - cfg.gumbel_tau_start)


def evaluate_geometric(xyz: torch.Tensor, output: dict) -> dict:
    """Geometric self-evaluation metrics (no GT needed)."""
    gaussians = output["gaussians"]
    assign_indices = output["assign_indices"]
    assign_weights = output["assign_weights"]

    mu = gaussians["mu"]
    n = gaussians["n"]

    hard = assign_indices.gather(
        1, assign_weights.argmax(dim=1, keepdim=True)
    ).squeeze(1)

    gamma = ((xyz - mu[hard]) * n[hard]).sum(dim=1)
    gamma_rms = gamma.pow(2).mean().sqrt().item()
    coverage = (assign_weights.max(dim=1).values > 0.1).float().mean().item()

    return {
        "gamma_rms": gamma_rms,
        "coverage": coverage,
        "num_gaussians": mu.shape[0],
    }


def process_frame(model, loss_fn, pts, device, tau, ego_radius):
    """Run model on a single frame and return loss dict + output."""
    xyz = pts[:, :3].to(device)
    intensity = pts[:, 3:4].to(device)

    # Ego-vehicle mask
    mask = torch.norm(xyz, dim=1) > ego_radius
    xyz, intensity = xyz[mask], intensity[mask]

    output = model(xyz, intensity, tau=tau)
    loss_dict = loss_fn(xyz, output)
    return loss_dict, output, xyz


def train(cfg: NeuralClusteringConfig, overfit_frames: int = 0):
    """Main training loop."""
    device = cfg.device

    data_root = os.path.expanduser(cfg.data_root)
    dataset = NuScenesNVSDataset(
        dataroot=data_root, version='v1.0-trainval', split='train',
    )

    if overfit_frames > 0:
        dataset.data_infos = dataset.data_infos[:overfit_frames]
        print(f"Overfit mode: {overfit_frames} pair(s)")

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=nvs_collate_fn, num_workers=0,
    )

    model = NeuralClusteringModel(cfg).to(device)
    loss_fn = ClusteringLoss(
        w_surface=cfg.w_surface,
        w_assign=cfg.w_assign,
        w_scale=cfg.w_scale,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")
    print(f"Dataset: {len(dataset)} pairs")
    print(f"Epochs: {cfg.num_epochs}, batch_size: {cfg.batch_size}")
    print("-" * 60)

    best_loss = float("inf")

    for epoch in range(cfg.num_epochs):
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

            for b in range(B):
                for frame_pts in [batch['input_0'][b], batch['input_1'][b]]:
                    ld, output, xyz = process_frame(
                        model, loss_fn, frame_pts, device, tau, cfg.ego_radius,
                    )
                    batch_loss = batch_loss + ld["total"]
                    batch_loss_dicts.append({k: v.item() for k, v in ld.items()})

            # Average over all frames in the batch (B * 2 frames)
            num_frames = B * 2
            batch_loss = batch_loss / num_frames

            if torch.isnan(batch_loss):
                pbar.write(f"  NaN loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                batch_loss = 0.0
                continue

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_batch = {k: sum(d[k] for d in batch_loss_dicts) / len(batch_loss_dicts)
                         for k in batch_loss_dicts[0]}
            epoch_losses.append(avg_batch)
            pbar.set_postfix(loss=f"{avg_batch['total']:.4f}", tau=f"{tau:.2f}")

            if overfit_frames > 0 or (batch_idx + 1) % 50 == 0:
                with torch.no_grad():
                    epoch_metrics.append(evaluate_geometric(xyz, output))

        dt = time.time() - t0
        if not epoch_losses:
            continue

        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
               for k in epoch_losses[0]}

        log = (f"[{epoch+1:3d}/{cfg.num_epochs}] "
               f"loss={avg['total']:.4f} "
               f"(S={avg['surface']:.4f} cmp={avg['compact']:.4f}) "
               f"tau={tau:.2f} {dt:.1f}s")

        if epoch_metrics:
            mg = {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                  for k in epoch_metrics[0]}
            log += f" | gamma={mg['gamma_rms']:.4f} cov={mg['coverage']:.3f} K={mg['num_gaussians']:.0f}"

        tqdm.write(log)

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, os.path.join(os.path.dirname(__file__), "outputs", "best_model.pt"))

    print(f"\nDone. Best loss: {best_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train neural 2D Gaussian clustering")
    parser.add_argument("--data-root", default=os.path.expanduser("~/data/nuScenes"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overfit", type=int, default=0,
                        help="Overfit on N pairs (0=full training)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = NeuralClusteringConfig(
        data_root=args.data_root,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    train(cfg, overfit_frames=args.overfit)


if __name__ == "__main__":
    main()
