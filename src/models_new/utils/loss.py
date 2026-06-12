import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # loss weights
        self.w_chamfer   = cfg.w_chamfer    # chamfer distance weight
        self.w_depth     = cfg.w_depth      # depth mse weight
        self.w_intensity = cfg.w_intensity  # intensity mse weight
        self.w_raydrop   = cfg.w_raydrop    # raydrop mse weight

    def forward(self, batch, depth, points_position, intensity, raydrop_map):
        """
        batch            : dataloader batch
            "lidar_points"  : (N, 4) xyz + intensity, ref frame
            "offset"        : (n_frames,) cumsum
        depth            : (H, W) predicted depth map
        points_position  : (N, 3) predicted GS positions
        intensity        : (H, W) predicted intensity map
        raydrop_map      : (H, W) predicted raydrop map
        """
        losses = {}

        # ── 1. Chamfer Distance (predicted positions vs gt lidar points) ──
        gt_xyz = batch["lidar_points"][:, :3]   # (N, 3)

        # pytorch3d chamfer는 (B, N, 3) 형태 필요
        chamfer_loss, _ = chamfer_distance(
            points_position.unsqueeze(0),   # (1, N_pred, 3)
            gt_xyz.unsqueeze(0),            # (1, N_gt, 3)
        )
        losses["chamfer"] = chamfer_loss

        # ── 2. Depth MSE ──────────────────────────────────────────────────
        gt_depth = batch["depth"]           # (H, W)
        valid    = gt_depth > 0             # valid depth mask
        depth_loss = F.mse_loss(
            depth[valid], gt_depth[valid]
        )
        losses["depth"] = depth_loss

        # ── 3. Intensity MSE ──────────────────────────────────────────────
        gt_intensity = batch["intensity"]   # (H, W)
        intensity_loss = F.mse_loss(
            intensity[valid], gt_intensity[valid]
        )
        losses["intensity"] = intensity_loss

        # ── 4. Raydrop MSE ───────────────────────────────────────────────
        gt_raydrop = batch["raydrop"]       # (H, W)
        raydrop_loss = F.mse_loss(
            raydrop_map, gt_raydrop
        )
        losses["raydrop"] = raydrop_loss

        # ── 5. Total loss ─────────────────────────────────────────────────
        total = (
            self.w_chamfer   * losses["chamfer"]   +
            self.w_depth     * losses["depth"]     +
            self.w_intensity * losses["intensity"] +
            self.w_raydrop   * losses["raydrop"]
        )
        losses["total"] = total

        return losses