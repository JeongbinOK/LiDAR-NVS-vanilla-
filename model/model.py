"""
Full feed-forward LiDAR NVS model.
pts -> range_img -> encode -> time_embed -> fuse -> decode -> Gaussian params
"""
import torch
import torch.nn as nn

from model.encoder import RangeViewEncoder, TimeEmbedding
from model.fusion import CrossFrameFusion
from model.decoder import GaussianHead
from utils.misc import points_to_pano, ego_mask


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

        # Shared encoder for both frames
        self.encoder = RangeViewEncoder(
            in_channels=enc_cfg['in_channels'],
            feat_dim=feat_dim,
        )
        self.time_embed = TimeEmbedding(
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
        )

    def forward(self, input_0_pts, input_1_pts, input_1_pose, t0=None, t1=None):
        """
        Args:
            input_0_pts: list of [N0, 4] tensors (one per batch).
            input_1_pts: list of [N1, 4] tensors.
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
        uv_1_list = []

        for b in range(B):
            # Ego-mask
            p0 = ego_mask(input_0_pts[b].to(device), self.ego_radius)

            # Transform input_1 to input_0 coordinate frame
            p1_raw = input_1_pts[b].to(device)
            p1_raw = ego_mask(p1_raw, self.ego_radius)
            p1_homo = torch.cat([p1_raw[:, :3],
                                 torch.ones(p1_raw.shape[0], 1, device=device)],
                                dim=1)
            p1_xyz = (p1_homo @ input_1_pose[b].T)[:, :3]
            p1_trans = torch.cat([p1_xyz, p1_raw[:, 3:4]], dim=1)  # [N1, 4]

            # Range images
            ri_0, _, uv_0 = points_to_pano(p0, self.vfov, self.hfov, self.H, self.W)
            ri_1, _, uv_1 = points_to_pano(p1_trans, self.vfov, self.hfov,
                                           self.H, self.W)

            range_imgs_0.append(ri_0)
            range_imgs_1.append(ri_1)
            pts_0_clean.append(p0)
            pts_1_trans.append(p1_trans)
            uv_0_list.append(uv_0)
            uv_1_list.append(uv_1)

        # Stack range images -> [B, 5, H, W]
        ri_0_batch = torch.stack(range_imgs_0)
        ri_1_batch = torch.stack(range_imgs_1)

        # Encode
        feat_0 = self.encoder(ri_0_batch)  # [B, C, H, W]
        feat_1 = self.encoder(ri_1_batch)

        # Time embed
        feat_0_t = self.time_embed(t0, feat_0)
        feat_1_t = self.time_embed(t1, feat_1)

        # Fuse
        fused = self.fusion(feat_0_t, feat_1_t)  # [B, C, H, W]

        # Decode -> per-point Gaussian params
        gaussian_params = self.decoder(
            fused, pts_0_clean, pts_1_trans,
            uv_0_list, uv_1_list, self.H, self.W
        )

        auxiliary = {
            'range_img_0': ri_0_batch,
            'range_img_1': ri_1_batch,
            'pts_0': pts_0_clean,
            'pts_1_transformed': pts_1_trans,
        }

        return gaussian_params, auxiliary
