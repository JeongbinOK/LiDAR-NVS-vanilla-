"""
Per-point Gaussian parameter prediction heads with split spatial/temporal paths
and dual-path feature lookup for Frame 1 points.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """Predict 2D Gaussian primitive parameters for each input point.

    Spatial heads (xyz, scaling, rotation, opacity, intensity) use:
        input = cat([per_point_feat, pts]) = [N, feat_dim + 4]

    Temporal heads (velocity, t_center, scaling_t) use:
        input = cat([per_point_feat, per_point_motion, pts]) = [N, feat_dim + motion_dim + 4]

    Frame 1 points use dual-path feature lookup:
        - fused feature via uv_1_f0 (cross-frame info)
        - native feature via uv_1_native (Frame 1 encoder feature)
        - combined via learned projection
    """

    def __init__(self, feat_dim=128, hidden_dim=128, delta_xyz_scale=0.5,
                 motion_dim=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.delta_xyz_scale = delta_xyz_scale

        spatial_in = feat_dim + 4
        temporal_in = feat_dim + motion_dim + 4

        def _make_head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        # Spatial heads
        self.xyz_head = _make_head(spatial_in, 3)
        self.scaling_head = _make_head(spatial_in, 2)
        self.rotation_head = _make_head(spatial_in, 4)
        self.opacity_head = _make_head(spatial_in, 1)
        self.intensity_head = _make_head(spatial_in, 1)

        # Temporal heads (wider input: + motion_dim)
        self.velocity_head = _make_head(temporal_in, 3)
        self.t_center_head = _make_head(temporal_in, 1)
        self.scaling_t_head = _make_head(temporal_in, 1)

        # Frame 1 dual-path projection: cat([fused, native]) -> feat_dim
        self.f1_proj = nn.Linear(feat_dim * 2, feat_dim)

        # Initialize opacity bias so sigmoid ~ 0.5
        nn.init.zeros_(self.opacity_head[-1].weight)
        nn.init.zeros_(self.opacity_head[-1].bias)

        # Initialize scaling bias for small initial Gaussians: exp(-3) ~ 0.05m
        nn.init.constant_(self.scaling_head[-1].bias, -3.0)

        # Initialize scaling_t bias for ~100ms temporal width: softplus(-2) ~ 0.13
        nn.init.constant_(self.scaling_t_head[-1].bias, -2.0)

        # Initialize velocity near zero
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(self, fused_feat, motion_feat, feat_1,
                pts_0, pts_1_transformed,
                uv_0, uv_1_f0, uv_1_native, H, W):
        """
        Args:
            fused_feat: [B, C, H, W] -- fused feature map (Frame 0 pixel space).
            motion_feat: [B, motion_dim, H, W] -- per-pixel motion signal.
            feat_1: [B, C, H, W] -- Frame 1 encoder features (native pixel space).
            pts_0: list of [N0, 4] -- frame 0 points (xyz + intensity).
            pts_1_transformed: list of [N1, 4] -- frame 1 points in frame 0 coords.
            uv_0: list of [N0, 2] -- pixel coords of pts_0 in Frame 0 RI.
            uv_1_f0: list of [N1, 2] -- pixel coords of pts_1 in Frame 0 pixel space.
            uv_1_native: list of [N1, 2] -- pixel coords of pts_1 in Frame 1 RI.
            H, W: range image dimensions.

        Returns:
            dict of Gaussian parameters, each list of [N0+N1, ...] per batch.
        """
        B = fused_feat.shape[0]
        device = fused_feat.device

        results = {
            'xyz': [], 'scaling': [], 'rotation': [], 'opacity': [],
            'velocity': [], 't_center': [], 'scaling_t': [], 'intensity': [],
        }

        for b in range(B):
            fused_b = fused_feat[b:b+1]    # [1, C, H, W]
            motion_b = motion_feat[b:b+1]  # [1, motion_dim, H, W]
            feat_1_b = feat_1[b:b+1]       # [1, C, H, W]

            N0 = pts_0[b].shape[0]
            N1 = pts_1_transformed[b].shape[0]

            # --- Frame 0 points: sample from fused + motion ---
            grid_0 = self._uv_to_grid(uv_0[b], H, W, device)  # [1, 1, N0, 2]
            pf_0 = F.grid_sample(fused_b, grid_0, mode='bilinear',
                                 align_corners=True)
            pf_0 = pf_0.squeeze(2).squeeze(0).T  # [N0, C]

            pm_0 = F.grid_sample(motion_b, grid_0, mode='bilinear',
                                 align_corners=True)
            pm_0 = pm_0.squeeze(2).squeeze(0).T  # [N0, motion_dim]

            # --- Frame 1 points: dual-path feature lookup ---
            grid_1_f0 = self._uv_to_grid(uv_1_f0[b], H, W, device)
            pf_1_fused = F.grid_sample(fused_b, grid_1_f0, mode='bilinear',
                                       align_corners=True)
            pf_1_fused = pf_1_fused.squeeze(2).squeeze(0).T  # [N1, C]

            grid_1_native = self._uv_to_grid(uv_1_native[b], H, W, device)
            pf_1_native = F.grid_sample(feat_1_b, grid_1_native, mode='bilinear',
                                        align_corners=True)
            pf_1_native = pf_1_native.squeeze(2).squeeze(0).T  # [N1, C]

            # Combine dual-path
            pf_1 = self.f1_proj(torch.cat([pf_1_fused, pf_1_native], dim=1))  # [N1, C]

            pm_1 = F.grid_sample(motion_b, grid_1_f0, mode='bilinear',
                                 align_corners=True)
            pm_1 = pm_1.squeeze(2).squeeze(0).T  # [N1, motion_dim]

            # --- Concatenate Frame 0 + Frame 1 ---
            per_point_feat = torch.cat([pf_0, pf_1], dim=0)      # [N0+N1, C]
            per_point_motion = torch.cat([pm_0, pm_1], dim=0)     # [N0+N1, motion_dim]
            pts_cat = torch.cat([pts_0[b], pts_1_transformed[b]], dim=0)  # [N0+N1, 4]

            # --- Spatial input ---
            spatial_enriched = torch.cat([per_point_feat, pts_cat], dim=1)  # [N, C+4]

            # --- Temporal input ---
            temporal_enriched = torch.cat([per_point_feat, per_point_motion,
                                           pts_cat], dim=1)  # [N, C+motion_dim+4]

            # Predict Gaussian parameters
            xyz_base = pts_cat[:, :3]
            delta_xyz = torch.tanh(self.xyz_head(spatial_enriched)) * self.delta_xyz_scale
            xyz = xyz_base + delta_xyz

            scaling = torch.exp(self.scaling_head(spatial_enriched).clamp(-7, 2))
            rotation = F.normalize(self.rotation_head(spatial_enriched), dim=-1)
            opacity = torch.sigmoid(self.opacity_head(spatial_enriched))
            intensity = torch.sigmoid(self.intensity_head(spatial_enriched))

            velocity = self.velocity_head(temporal_enriched).clamp(-10, 10)
            t_center = torch.sigmoid(self.t_center_head(temporal_enriched)) * 0.5
            scaling_t = F.softplus(self.scaling_t_head(temporal_enriched)) + 1e-4

            results['xyz'].append(xyz)
            results['scaling'].append(scaling)
            results['rotation'].append(rotation)
            results['opacity'].append(opacity)
            results['velocity'].append(velocity)
            results['t_center'].append(t_center)
            results['scaling_t'].append(scaling_t)
            results['intensity'].append(intensity)

        return results

    @staticmethod
    def _uv_to_grid(uv, H, W, device):
        """Convert pixel coords (u=col, v=row) to normalized [-1,1] grid."""
        grid = torch.zeros(uv.shape[0], 2, device=device, dtype=torch.float32)
        grid[:, 0] = 2.0 * uv[:, 0].float() / max(W - 1, 1) - 1.0
        grid[:, 1] = 2.0 * uv[:, 1].float() / max(H - 1, 1) - 1.0
        return grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
