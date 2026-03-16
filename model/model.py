"""
Full feed-forward LiDAR NVS model with viewpoint-centric rendering.
pts -> natural range_imgs -> encode -> warp -> motion -> FiLM -> fuse -> decode
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import RangeViewEncoder, FiLMTimeEmbedding
from model.fusion import CrossFrameFusion
from model.decoder import GaussianHead
from utils.misc import points_to_pano, ego_mask, compute_beam_dirs, compute_warp_grid


class FeedForwardGaussianModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        enc_cfg = cfg['encoder']
        te_cfg = cfg['time_embed']
        fus_cfg = cfg['fusion']
        dec_cfg = cfg['decoder']

        self.H = cfg['H']
        self.W = cfg['W']
        self.vfov = cfg['vfov']
        self.hfov = cfg['hfov']
        self.ego_radius = cfg['ego_mask_radius']

        feat_dim = enc_cfg['feat_dim']
        motion_dim = dec_cfg.get('motion_dim', 32)

        # Shared encoder for both frames
        self.encoder = RangeViewEncoder(
            in_channels=enc_cfg['in_channels'],
            feat_dim=feat_dim,
        )
        self.time_embed = FiLMTimeEmbedding(
            feat_dim=feat_dim,
            time_embed_dim=te_cfg['dim'],
            max_freq=te_cfg['max_freq'],
        )
        self.fusion = CrossFrameFusion(
            dim=feat_dim,
            n_heads=fus_cfg['n_heads'],
            window_w=fus_cfg['window_w'],
            num_layers=fus_cfg.get('num_layers', 2),
        )
        self.decoder = GaussianHead(
            feat_dim=feat_dim,
            hidden_dim=dec_cfg['hidden_dim'],
            delta_xyz_scale=dec_cfg['delta_xyz_scale'],
            motion_dim=motion_dim,
        )

        # Motion projection: cat([feat_0, feat_1_warped]) -> motion features
        self.motion_proj = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, motion_dim, 1),
        )

        # Precompute beam directions for warping (registered as buffer)
        beam_dirs = compute_beam_dirs(self.H, self.W, self.vfov, self.hfov)
        self.register_buffer('beam_dirs', beam_dirs)

    def forward(self, input_0_pts, input_1_pts, input_1_pose, t0=None, t1=None):
        """
        Args:
            input_0_pts: list of [N0, 4] tensors (one per batch).
            input_1_pts: list of [N1, 4] tensors (in Frame 1 native coords).
            input_1_pose: [B, 4, 4] transform from input_1 to input_0 frame.
            t0: [B] timestamps for frame 0 (default 0.0).
            t1: [B] timestamps for frame 1 (default 0.5).

        Returns:
            gaussian_params: dict of lists, each element [N0+N1, ...] per batch.
            auxiliary: dict with range images and uv coords for loss computation.
        """
        B = len(input_0_pts)
        device = input_1_pose.device

        if t0 is None:
            t0 = torch.zeros(B, device=device)
        if t1 is None:
            t1 = torch.full((B,), 0.5, device=device)

        # Pre-process per sample (variable point counts)
        range_imgs_0 = []
        range_imgs_1 = []
        pts_0_clean = []
        pts_1_trans = []
        uv_0_list = []
        uv_1_f0_list = []
        uv_1_native_list = []

        for b in range(B):
            # Ego-mask
            p0 = ego_mask(input_0_pts[b].to(device), self.ego_radius)
            p1_raw = ego_mask(input_1_pts[b].to(device), self.ego_radius)

            # Step 1: Natural Range Images (each from its own sensor origin)
            ri_0, _, uv_0 = points_to_pano(p0, self.vfov, self.hfov,
                                           self.H, self.W)
            ri_1, _, uv_1_native = points_to_pano(p1_raw, self.vfov, self.hfov,
                                                   self.H, self.W)

            # Step 2: Transform p1 to Frame 0 for decoder UV + point coords
            p1_homo = torch.cat([p1_raw[:, :3],
                                 torch.ones(p1_raw.shape[0], 1, device=device)],
                                dim=1)
            p1_xyz = (p1_homo @ input_1_pose[b].T)[:, :3]
            p1_trans = torch.cat([p1_xyz, p1_raw[:, 3:4]], dim=1)  # [N1, 4]

            # UV for Frame 1 points in Frame 0 pixel space (for fused feature lookup)
            _, _, uv_1_f0 = points_to_pano(p1_trans, self.vfov, self.hfov,
                                           self.H, self.W)

            range_imgs_0.append(ri_0)
            range_imgs_1.append(ri_1)
            pts_0_clean.append(p0)
            pts_1_trans.append(p1_trans)
            uv_0_list.append(uv_0)
            uv_1_f0_list.append(uv_1_f0)
            uv_1_native_list.append(uv_1_native)

        # Stack range images -> [B, 5, H, W]
        ri_0_batch = torch.stack(range_imgs_0)
        ri_1_batch = torch.stack(range_imgs_1)

        # Step 3: Encode (both with natural RIs)
        feat_0 = self.encoder(ri_0_batch)  # [B, C, H, W]
        feat_1 = self.encoder(ri_1_batch)  # [B, C, H, W]

        # Step 4: Feature Warping
        T_0_to_1 = torch.inverse(input_1_pose)  # [B, 4, 4]
        warp_grid = compute_warp_grid(
            ri_0_batch[:, 0:1], self.beam_dirs, T_0_to_1,
            self.H, self.W, self.vfov, self.hfov
        )
        feat_1_warped = F.grid_sample(
            feat_1, warp_grid, mode='bilinear',
            padding_mode='zeros', align_corners=True
        )  # [B, C, H, W]

        # Step 5: Motion Signal
        motion_feat = self.motion_proj(
            torch.cat([feat_0, feat_1_warped], dim=1)
        )  # [B, motion_dim, H, W]

        # Step 6: FiLM Time Embedding
        feat_0_t = self.time_embed(t0, feat_0)
        feat_1_t = self.time_embed(t1, feat_1_warped)

        # Step 7: Fusion
        fused = self.fusion(feat_0_t, feat_1_t)  # [B, C, H, W]

        # Step 8: Decode -> per-point Gaussian params
        gaussian_params = self.decoder(
            fused, motion_feat, feat_1,
            pts_0_clean, pts_1_trans,
            uv_0_list, uv_1_f0_list, uv_1_native_list,
            self.H, self.W,
        )

        auxiliary = {
            'range_img_0': ri_0_batch,
            'range_img_1': ri_1_batch,
            'pts_0': pts_0_clean,
            'pts_1_transformed': pts_1_trans,
        }

        return gaussian_params, auxiliary
