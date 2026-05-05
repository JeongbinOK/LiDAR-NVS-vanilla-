#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple, Optional
import math
import torch.nn as nn
import torch
from . import _C

# LiDAR channel layout constants (mirrored from cuda_rasterizer/channel_layout.h
# via ext.cpp). Importable as `from diff_quadratic_rasterization import LIDAR_*`.
# Current values (A3.4): NUM_CHANNELS=17 → OUTPUT_CHANNELS=27, LIDAR_LATENT_DIM=16.
OUTPUT_CHANNELS        = int(_C.OUTPUT_CHANNELS)
NUM_CHANNELS           = int(_C.NUM_CHANNELS)
NORMAL_OFFSET          = int(_C.NORMAL_OFFSET)
DEPTH_OFFSET           = int(_C.DEPTH_OFFSET)
ALPHA_OFFSET           = int(_C.ALPHA_OFFSET)
CURVATURE_OFFSET       = int(_C.CURVATURE_OFFSET)
MIDDEPTH_OFFSET        = int(getattr(_C, "MIDDEPTH_OFFSET", NUM_CHANNELS + 6))
LIDAR_INTENSITY_OFFSET = int(_C.LIDAR_INTENSITY_OFFSET)
LIDAR_LATENT_OFFSET    = int(_C.LIDAR_LATENT_OFFSET)
LIDAR_LATENT_DIM       = int(_C.LIDAR_LATENT_DIM)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    view2gaussian_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        view2gaussian_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        view2gaussian_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            raster_settings.sigma,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.cam_intr, # cpu
            raster_settings.prefiltered,
            raster_settings.return_depth,
            raster_settings.return_normal,
            raster_settings.debug,
            raster_settings.lidar_mode,
            raster_settings.r_near,
            raster_settings.r_far,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, n_touched, aabb, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, n_touched, aabb, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(color, colors_precomp, means3D, scales, rotations, view2gaussian_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, aabb, n_touched

    @staticmethod
    def backward(ctx, grad_out_color, _radii, _aabb, _n_touched):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        color, colors_precomp, means3D, scales, rotations, view2gaussian_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                color,
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                raster_settings.sigma,
                view2gaussian_precomp,
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                raster_settings.kernel_size,
                raster_settings.subpixel_offset,
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                raster_settings.cam_intr, # On cpu
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.return_depth,
                raster_settings.return_normal,
                raster_settings.debug,
                raster_settings.stop_z_gradient,
                raster_settings.reciprocal_z,
                raster_settings.lidar_mode,
                raster_settings.r_near,
                raster_settings.r_far)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_rotations, grad_view2gaussian_precomp = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_sh, grad_scales, grad_rotations, grad_view2gaussian_precomp = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_view2gaussian_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    kernel_size : float
    subpixel_offset: torch.Tensor
    bg : torch.Tensor
    scale_modifier : float
    sigma : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    cam_intr : torch.Tensor
    stop_z_gradient : bool
    reciprocal_z : bool
    return_depth : bool
    return_normal : bool
    # A3.2.c: panoramic LiDAR mode (spherical projection). If True, cam_intr is
    # repurposed as [el_min_rad, el_max_rad, w_per_rad_az, h_per_rad_el] and
    # projmatrix is ignored. Camera mode (default) is unchanged.
    lidar_mode : bool = False
    r_near : float = 0.2
    r_far : float = 100.0

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, view2gaussian_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and view2gaussian_precomp is None) or ((scales is not None or rotations is not None) and view2gaussian_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed view2gaussian !')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])

        if view2gaussian_precomp is None:
            view2gaussian_precomp = torch.Tensor([])
        elif view2gaussian_precomp.numel() > 0:
            N = means3D.shape[0]
            shp = tuple(view2gaussian_precomp.shape)
            if shp != (N, 4, 4) and shp != (N, 16):
                raise ValueError(
                    f"view2gaussian_precomp must be [N,4,4] or [N,16] "
                    f"(row-major), got shape {shp} with N={N}"
                )
            
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            view2gaussian_precomp,
            raster_settings,
        )


# ---------------------------------------------------------------------------
# A3.3 — LiDAR rasterizer wrapper.
#
# Thin layer over GaussianRasterizer that:
#   1. Builds a panoramic-LiDAR `GaussianRasterizationSettings` from sensor specs
#      (image size, elevation FOV) instead of pinhole intrinsics.
#   2. Packs per-Gaussian (intensity, latent[…]) into the colors_precomp slot
#      that the CUDA kernel alpha-blends.
#   3. Unpacks the flat [OUTPUT_CHANNELS, H, W] output tensor into a named
#      `LiDARRasterOutput` so downstream code (drop MLP, losses) can be agnostic
#      to the storage layout.
#
# The CUDA kernel itself is unchanged — see auxiliary.h for the layout.
# ---------------------------------------------------------------------------

class LiDARRasterOutput(NamedTuple):
    """Named view over the flat rasterizer output. All tensors live on the same
    device as the rasterizer call. Shapes are stated in the field comments.

    Channels not yet produced by the CUDA kernel (drop_logit, full feat_agg
    when LIDAR_LATENT_DIM is too small) are exposed as `None`; the consumer
    (drop MLP in A3.4) is responsible for synthesising them.
    """
    range:       torch.Tensor              # [H, W]      intersection root r (m)
    middepth:    torch.Tensor              # [H, W]      median depth root r (m)
    intensity:   torch.Tensor              # [H, W]      alpha-blended scalar in [0,1]
    alpha_accum: torch.Tensor              # [H, W]      ∑ T_i α_i in [0,1]
    normal:      torch.Tensor              # [3, H, W]   sensor-frame xyz
    curvature:   torch.Tensor              # [H, W]      alpha-blended signed Gaussian curvature κ
    latent:      torch.Tensor              # [L, H, W]   alpha-blended latent (L=LIDAR_LATENT_DIM)
    drop_logit:  Optional[torch.Tensor]    # [H, W] or None — None until A3.4
    radii:       torch.Tensor              # [N]         per-Gaussian image radius
    aabb:        torch.Tensor              # rect bbox per Gaussian (passthrough)
    n_touched:   torch.Tensor              # [N]         tile-touch count (passthrough)
    raw:         torch.Tensor              # [OUTPUT_CHANNELS, H, W] full storage view


def make_lidar_settings(
    *,
    image_height: int,
    image_width: int,
    el_min_rad: float,
    el_max_rad: float,
    viewmatrix: torch.Tensor,
    campos: torch.Tensor,
    sigma: float = 3.0,
    scale_modifier: float = 1.0,
    kernel_size: float = 0.0,
    bg: Optional[torch.Tensor] = None,
    r_near: float = 0.2,
    r_far: float = 100.0,
    debug: bool = False,
    return_depth: bool = True,
    return_normal: bool = True,
) -> GaussianRasterizationSettings:
    """Build a GaussianRasterizationSettings configured for panoramic LiDAR.

    `viewmatrix` brings world points into sensor frame (x=right, y=forward, z=up).
    `cam_intr` is repurposed as [el_min_rad, el_max_rad, w_per_rad_az, h_per_rad_el]
    on the CPU (the upstream convention, see rasterizer_impl.cu host-side unpack).
    """
    if el_max_rad <= el_min_rad:
        raise ValueError(
            f"el_max_rad ({el_max_rad}) must exceed el_min_rad ({el_min_rad})"
        )

    device = viewmatrix.device
    cam_intr = torch.tensor(
        [
            float(el_min_rad),
            float(el_max_rad),
            image_width / (2.0 * math.pi),
            image_height / (el_max_rad - el_min_rad),
        ],
        device="cpu",
        dtype=torch.float32,
    )
    if bg is None:
        bg = torch.zeros(NUM_CHANNELS, device=device, dtype=torch.float32)

    # projmatrix is unused in LiDAR mode but the NamedTuple requires a tensor.
    eye4 = torch.eye(4, device=device, dtype=torch.float32)
    return GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=1.0,                       # unused
        tanfovy=1.0,                       # unused
        kernel_size=kernel_size,
        subpixel_offset=torch.zeros(image_height, image_width, 2, device=device,
                                    dtype=torch.float32),
        bg=bg,
        scale_modifier=scale_modifier,
        sigma=sigma,
        viewmatrix=viewmatrix,
        projmatrix=eye4,                   # unused
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=debug,
        cam_intr=cam_intr,
        stop_z_gradient=False,
        reciprocal_z=False,
        return_depth=return_depth,
        return_normal=return_normal,
        lidar_mode=True,
        r_near=r_near,
        r_far=r_far,
    )


class LiDARRasterizer(nn.Module):
    """Panoramic LiDAR rasterizer with named output channels.

    Args to `forward`:
        means3D:    [N, 3]    primitive centres in world frame
        means2D:    [N, 3]    placeholder for image-space gradients (autograd only)
        opacities:  [N, 1]    α_i in (0, 1)
        scales:     [N, 3]    signed QGS surface scales; s1/s2 set tangent
                              curvature signs and s3 is the height scale
        rotations:  [N, 4]    quaternions (w, x, y, z)
        intensity:  [N]       per-Gaussian scalar in [0, 1]
        latent:     [N, L]    per-Gaussian latent. L must equal LIDAR_LATENT_DIM.

    Returns a `LiDARRasterOutput`.
    """

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        if not raster_settings.lidar_mode:
            raise ValueError(
                "LiDARRasterizer requires raster_settings.lidar_mode=True. "
                "Use diff_quadratic_rasterization.make_lidar_settings(...)."
            )
        self.raster_settings = raster_settings

    @staticmethod
    def _pack_features(intensity: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Pack (intensity[N], latent[N, L]) into colors_precomp[N, NUM_CHANNELS].

        Layout in the colors slot must match auxiliary.h:
            channel 0      : intensity
            channel 1..L   : latent[..., 0..L-1]
        Where L = LIDAR_LATENT_DIM = NUM_CHANNELS - 1.
        """
        if intensity.dim() != 1:
            raise ValueError(f"intensity must be [N], got {tuple(intensity.shape)}")
        if latent.dim() != 2 or latent.shape[0] != intensity.shape[0]:
            raise ValueError(
                f"latent must be [N, L], got {tuple(latent.shape)} "
                f"with N={intensity.shape[0]}"
            )
        if latent.shape[1] != LIDAR_LATENT_DIM:
            raise ValueError(
                f"latent dim {latent.shape[1]} != LIDAR_LATENT_DIM={LIDAR_LATENT_DIM} "
                f"(current build). Either truncate/zero-pad upstream, or rebuild "
                f"the rasterizer with the desired dim:\n"
                f"    LIDAR_LATENT_DIM={latent.shape[1]} pip install -e "
                f"renderer/lidar_qgs_rasterizer --force-reinstall --no-deps"
            )
        return torch.cat([intensity.unsqueeze(-1), latent], dim=-1).contiguous()

    @staticmethod
    def _compute_view2gaussian(
        means3D: torch.Tensor,
        rotations: torch.Tensor,
        viewmatrix: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable view-to-Gaussian transform matching CUDA layout."""
        q = rotations / rotations.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r, x, y, z = q.unbind(dim=-1)
        R = torch.stack(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - r * z),
                2.0 * (x * z + r * y),
                2.0 * (x * y + r * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - r * x),
                2.0 * (x * z - r * y),
                2.0 * (y * z + r * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)

        N = means3D.shape[0]
        G2W = torch.eye(4, device=means3D.device, dtype=means3D.dtype).expand(N, 4, 4).clone()
        G2W[:, :3, :3] = R
        G2W[:, 3, :3] = means3D

        G2V = viewmatrix.to(device=means3D.device, dtype=means3D.dtype).unsqueeze(0) @ G2W
        R_g2v = G2V[:, :3, :3]
        R_t = R_g2v.transpose(1, 2)
        t = G2V[:, 3, :3]

        out = torch.zeros((N, 4, 4), device=means3D.device, dtype=means3D.dtype)
        # CUDA transform helpers index the flat buffer as a column-major 3x4
        # matrix. Row-major [N,4,4] storage therefore needs R, not R^T, here.
        out[:, :3, :3] = R_g2v
        out[:, 3, :3] = -(R_t @ t.unsqueeze(-1)).squeeze(-1)
        out[:, 3, 3] = 1.0
        return out.contiguous()

    @staticmethod
    def _unpack(
        rendered: torch.Tensor,
        radii: torch.Tensor,
        aabb: torch.Tensor,
        n_touched: torch.Tensor,
    ) -> LiDARRasterOutput:
        """Slice the [OUTPUT_CHANNELS, H, W] tensor into named fields."""
        if rendered.dim() != 3 or rendered.shape[0] != OUTPUT_CHANNELS:
            raise RuntimeError(
                f"unexpected rasterizer output shape {tuple(rendered.shape)}; "
                f"expected first dim = {OUTPUT_CHANNELS}"
            )
        latent_end = LIDAR_LATENT_OFFSET + LIDAR_LATENT_DIM
        return LiDARRasterOutput(
            range       = rendered[DEPTH_OFFSET],
            middepth    = rendered[MIDDEPTH_OFFSET],
            intensity   = rendered[LIDAR_INTENSITY_OFFSET],
            alpha_accum = rendered[ALPHA_OFFSET],
            normal      = rendered[NORMAL_OFFSET:NORMAL_OFFSET + 3],
            curvature   = rendered[CURVATURE_OFFSET],
            latent      = rendered[LIDAR_LATENT_OFFSET:latent_end],
            drop_logit  = None,
            radii       = radii,
            aabb        = aabb,
            n_touched   = n_touched,
            raw         = rendered,
        )

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        intensity: torch.Tensor,
        latent: torch.Tensor,
    ) -> LiDARRasterOutput:
        colors_precomp = self._pack_features(intensity, latent)
        view2gaussian_precomp = self._compute_view2gaussian(
            means3D,
            rotations,
            self.raster_settings.viewmatrix,
        )
        rendered, radii, aabb, n_touched = rasterize_gaussians(
            means3D=means3D.detach(),
            means2D=means2D,
            sh=torch.empty(0, device=means3D.device),
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations.detach(),
            view2gaussian_precomp=view2gaussian_precomp,
            raster_settings=self.raster_settings,
        )
        return self._unpack(rendered, radii, aabb, n_touched)
