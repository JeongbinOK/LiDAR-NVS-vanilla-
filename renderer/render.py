"""
Temporal Gaussian evolution and panoramic rendering via local 2D Gaussian rasterizer.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Ensure local third_party is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party'))

from diff_gaussian_rasterization_2d import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def evolve_gaussians(xyz, velocity, t_center, scaling_t, opacity, t_query):
    """Apply linear motion and temporal marginal weighting.

    Args:
        xyz: [N, 3] Gaussian centres.
        velocity: [N, 3] velocity in m/s.
        t_center: [N, 1] temporal Gaussian centre (0~0.5 s).
        scaling_t: [N, 1] temporal Gaussian width.
        opacity: [N, 1] base opacity.
        t_query: scalar — query timestamp (seconds).

    Returns:
        xyz_t: [N, 3] evolved positions.
        eff_opacity: [N, 1] temporally-weighted opacity.
    """
    xyz_t = xyz + velocity * t_query
    marginal_t = torch.exp(-0.5 * (t_center - t_query) ** 2 / scaling_t ** 2)
    eff_opacity = opacity * marginal_t
    return xyz_t, eff_opacity


def _build_view_matrix(forward=True, device='cuda'):
    """Build the world-to-LiDAR view matrix following GS-LiDAR's convention.

    The w2l matrix swaps axes from nuScenes (x-right, y-fwd, z-up) to the
    rasteriser's camera convention (x-right, y-down, z-fwd):
        x_cam =  x_lidar
        y_cam = -z_lidar
        z_cam =  y_lidar

    For backward (rear 180 deg), apply an extra 180-deg yaw rotation.
    """
    w2l = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)

    if not forward:
        rot180 = np.array([
            [-1, 0,  0, 0],
            [ 0, 1,  0, 0],
            [ 0, 0, -1, 0],
            [ 0, 0,  0, 1],
        ], dtype=np.float32)
        w2l = rot180 @ w2l

    return torch.from_numpy(w2l).to(device)


def _rasterize_half(xyz, opacity, scaling, rotation, intensity,
                    viewmat, vfov, H, W_half, scale_factor=1.0):
    """Rasterise Gaussians for a 180-deg half-panorama.

    Returns:
        depth: [1, H, W_half]
        depth_sq: [1, H, W_half] — depth squared (for depth variance loss)
        normal: [3, H, W_half] — rendered surface normals
        intensity_map: [1, H, W_half]
        alpha: [1, H, W_half]
    """
    N = xyz.shape[0]
    device = xyz.device

    bg = torch.zeros(4, device=device)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W_half,
        tanfovx=-1.0,
        tanfovy=-1.0,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmat.contiguous(),
        projmatrix=viewmat.contiguous(),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        vfov=vfov,
        hfov=(-90, 90),
        scale_factor=scale_factor,
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)

    screenspace = torch.zeros(N, 4, device=device, requires_grad=True)

    # Colors precomp: [N, 4] — (unused, unused, intensity, zeros)
    colors_precomp = torch.zeros(N, 4, device=device)
    colors_precomp[:, 2] = intensity.squeeze(-1)

    # 2D Gaussian: scaling [N, 2] → pad to [N, 3] with dummy z=0
    scales_3d = torch.cat([scaling, torch.zeros(N, 1, device=device)], dim=1)

    # Features: empty → rasterizer returns [3, H, W] surface normals
    features = torch.zeros(N, 0, device=device)

    mask = opacity.squeeze(-1) > (1.0 / 255.0)

    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace,
        shs=None,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales_3d,
        rotations=rotation,
        cov3D_precomp=None,
        mask=mask,
    )

    # rendered_depth: [4, H, W_half] — [mean, median, distortion, depth_square]
    depth = rendered_depth[0:1]        # [1, H, W_half]
    depth_sq = rendered_depth[3:4]     # [1, H, W_half]
    alpha = rendered_opacity           # [1, H, W_half]
    intensity_map = rendered_image[2:3]  # [1, H, W_half]
    # rendered_feature: [3, H, W_half] — surface normals
    normal = rendered_feature           # [3, H, W_half]

    return depth, depth_sq, normal, intensity_map, alpha


def _stitch_pano(front, back, H, W_half):
    """Stitch front/back half-panorama into full 360-deg panorama.

    Works for any leading channel dimension: [C, H, W_half] → [C, H, 2*W_half].
    """
    W_full = 2 * W_half
    C = front.shape[0]
    device = front.device

    pano = torch.zeros(C, H, W_full, device=device)

    b1 = W_half // 2
    b2 = W_half + W_half // 2
    b3 = W_full

    # Front → centre
    pano[:, :, b1:b2] = front
    # Back → left and right edges
    pano[:, :, b2:b3] = back[:, :, :b3 - b2]
    pano[:, :, 0:b1] = back[:, :, W_half - b1:]

    return pano


def render_full_pano(gaussian_params_b, t_query, vfov, H, W_half,
                     scale_factor=1.0):
    """Render a full 360-deg panoramic depth/intensity/normal map for one sample.

    Args:
        gaussian_params_b: dict with keys xyz, scaling, rotation, opacity,
                           velocity, t_center, scaling_t, intensity — each [N, ...].
        t_query: scalar timestamp (seconds, 0~0.5).
        vfov: (min_deg, max_deg).
        H: panorama height.
        W_half: half-panorama width.
        scale_factor: depth scale factor.

    Returns:
        depth_pano: [1, H, 2*W_half] full-360 depth map.
        intensity_pano: [1, H, 2*W_half] full-360 intensity map.
        depth_sq_pano: [1, H, 2*W_half] depth squared map (for depth var loss).
        normal_pano: [3, H, 2*W_half] surface normal map.
    """
    device = gaussian_params_b['xyz'].device

    # Temporal evolution
    xyz_t, eff_opacity = evolve_gaussians(
        gaussian_params_b['xyz'],
        gaussian_params_b['velocity'],
        gaussian_params_b['t_center'],
        gaussian_params_b['scaling_t'],
        gaussian_params_b['opacity'],
        t_query,
    )

    scaling = gaussian_params_b['scaling']
    rotation = gaussian_params_b['rotation']
    intensity = gaussian_params_b['intensity']

    # Front 180 deg
    viewmat_fwd = _build_view_matrix(forward=True, device=device)
    depth_f, dsq_f, norm_f, int_f, alpha_f = _rasterize_half(
        xyz_t, eff_opacity, scaling, rotation, intensity,
        viewmat_fwd, vfov, H, W_half, scale_factor)

    # Back 180 deg
    viewmat_bwd = _build_view_matrix(forward=False, device=device)
    depth_b, dsq_b, norm_b, int_b, alpha_b = _rasterize_half(
        xyz_t, eff_opacity, scaling, rotation, intensity,
        viewmat_bwd, vfov, H, W_half, scale_factor)

    # Stitch all outputs
    depth_pano = _stitch_pano(depth_f, depth_b, H, W_half)
    intensity_pano = _stitch_pano(int_f, int_b, H, W_half)
    depth_sq_pano = _stitch_pano(dsq_f, dsq_b, H, W_half)
    normal_pano = _stitch_pano(norm_f, norm_b, H, W_half)

    return depth_pano, intensity_pano, depth_sq_pano, normal_pano
