"""
Per-point Gaussian parameter prediction heads.

Improvements over v1:
- Concatenates raw point geometry (xyz + intensity) with learned features.
- 3-layer MLPs (was 2-layer) with larger hidden_dim.
- Better initialization for scaling and scaling_t.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """Predict 2D Gaussian primitive parameters for each input point.

    For every LiDAR point (from both input frames), we look up its per-point
    feature from the fused range-view feature map via bilinear sampling,
    concatenate raw geometric context (xyz + intensity), then run 3-layer
    MLP heads to predict Gaussian attributes.
    """

    def __init__(self, feat_dim=128, hidden_dim=128, delta_xyz_scale=0.5):
        super().__init__()
        self.delta_xyz_scale = delta_xyz_scale

        # Input: learned feature + raw point info (x, y, z, intensity)
        in_dim = feat_dim + 4

        def _make_head(out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        self.xyz_head = _make_head(3)       # delta position
        self.scaling_head = _make_head(2)   # 2D disk scales (log-space)
        self.rotation_head = _make_head(4)  # quaternion
        self.opacity_head = _make_head(1)   # logit -> sigmoid
        self.velocity_head = _make_head(3)  # 3D velocity (m/s)
        self.t_center_head = _make_head(1)  # temporal center (logit -> sigmoid * 0.5)
        self.scaling_t_head = _make_head(1) # temporal scale (softplus)
        self.intensity_head = _make_head(1) # intensity (sigmoid)

        # Initialize opacity bias so sigmoid ≈ 0.5
        nn.init.zeros_(self.opacity_head[-1].weight)
        nn.init.zeros_(self.opacity_head[-1].bias)

        # Initialize scaling bias for small initial Gaussians: exp(-3) ≈ 0.05m
        nn.init.constant_(self.scaling_head[-1].bias, -3.0)

        # Initialize scaling_t bias for ~100ms temporal width: softplus(-2) ≈ 0.13
        nn.init.constant_(self.scaling_t_head[-1].bias, -2.0)

        # Initialize velocity near zero
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(self, fused_feat, pts_0, pts_1_transformed, uv_0, uv_1, H, W):
        """
        Args:
            fused_feat: [B, C, H, W] — fused feature map.
            pts_0: [B, N0, 4] — frame 0 points (xyz + intensity).
            pts_1_transformed: [B, N1, 4] — frame 1 points in frame 0 coords.
            uv_0: [B, N0, 2] — pixel coords of pts_0 in range image.
            uv_1: [B, N1, 2] — pixel coords of pts_1 in range image.
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
            feat_b = fused_feat[b:b+1]  # [1, C, H, W]

            # Concatenate UV coords and sample features
            uv_cat = torch.cat([uv_0[b], uv_1[b]], dim=0)  # [N0+N1, 2]
            grid = self._uv_to_grid(uv_cat, H, W, device)  # [1, 1, N0+N1, 2]

            per_point_feat = F.grid_sample(
                feat_b, grid, mode='bilinear', align_corners=True
            )  # [1, C, 1, N0+N1]
            per_point_feat = per_point_feat.squeeze(2).squeeze(0).T  # [N0+N1, C]

            # Raw point geometry: xyz + intensity
            pts_cat = torch.cat([pts_0[b], pts_1_transformed[b]], dim=0)  # [N0+N1, 4]

            # Enrich features with geometric context
            enriched = torch.cat([per_point_feat, pts_cat], dim=1)  # [N0+N1, C+4]

            # Predict Gaussian parameters
            xyz_base = pts_cat[:, :3]
            delta_xyz = torch.tanh(self.xyz_head(enriched)) * self.delta_xyz_scale
            xyz = xyz_base + delta_xyz

            scaling = torch.exp(self.scaling_head(enriched).clamp(-7, 2))  # [N, 2] range [0.001m, 7.4m]
            rotation = F.normalize(self.rotation_head(enriched), dim=-1)  # [N, 4]
            opacity = torch.sigmoid(self.opacity_head(enriched))   # [N, 1]
            velocity = self.velocity_head(enriched).clamp(-10, 10) # [N, 3] cap at ±10 m/s
            t_center = torch.sigmoid(self.t_center_head(enriched)) * 0.5  # [N, 1]
            scaling_t = F.softplus(self.scaling_t_head(enriched)) + 1e-4  # [N, 1]
            intensity = torch.sigmoid(self.intensity_head(enriched))  # [N, 1]

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
