import torch
import math
from .diff_gaussian_rasterization_2d import GaussianRasterizationSettings, GaussianRasterizer

from ..utils.camera import Camera
#from utils.sh_utils import eval_sh
from ..utils.render import Gaussianutil

 

gu = Gaussianutil(None)





def render(viewpoint_camera, pc, cfg, bg_color, input_timestamp, scaling_modifier=1.0,
           override_color=None, env_map=None, other=[], mask=None, is_training=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    render_device = pc["position"].device
    render_dtype = pc["position"].dtype

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc["position"].shape[0], 4), dtype=render_dtype, requires_grad=True, device=render_device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if cfg.neg_fov:
        # we find that set fov as -1 slightly improves the results
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color.to(device=render_device, dtype=render_dtype),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.to(device=render_device, dtype=render_dtype),
        projmatrix=viewpoint_camera.full_proj_transform.to(device=render_device, dtype=render_dtype),
        sh_degree=cfg.sh_degree,
        campos=viewpoint_camera.camera_center.to(device=render_device, dtype=render_dtype),
        prefiltered=False,
        debug=cfg.debug,
        vfov=viewpoint_camera.vfov,
        hfov=viewpoint_camera.hfov,
        row_to_theta=viewpoint_camera.row_to_theta.to(device=render_device, dtype=render_dtype),
        scale_factor=float(1.0 if cfg.scale_factor is None else cfg.scale_factor)
    )

    assert raster_settings.bg.shape[0] == 4

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points


    #params
    means3D = pc["position"]
    opacity = gu.get_opacity(pc["opacity"])
    scales = gu.get_scaling(pc["scales"])
    rotations = gu.get_rotation(pc["rotations"])
    cov3D_precomp = None
    shs = pc["shs"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # colors_precomp = None
    # if override_color is None:
    #     if cfg.convert_SHs_python:
    #         shs_view = pc["shs"]
    #         dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc["shs"].shape[0], 1)).detach()
    #         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(cfg.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
            
    # else:
    #     colors_precomp = override_color

    feature_list = other

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0

    # Prefilter
    mask = (opacity[:, 0] > 1 / 255) if mask is None else mask & (opacity[:, 0] > 1 / 255)
    if cfg.dynamic:
        mask = mask & (marginal_t[:, 0] > 0.05)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask)

    _, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)
    rendered_other, rendered_normal = rendered_feature.split([S_other, 3], dim=0)
    rendered_normal = rendered_normal / (rendered_normal.norm(dim=0, keepdim=True) + 1e-8)

    if env_map is not None:
        lidar_raydrop_prior_from_envmap = env_map(viewpoint_camera.towards)
        rendered_raydrop = lidar_raydrop_prior_from_envmap + (1 - lidar_raydrop_prior_from_envmap) * rendered_raydrop

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {
    #     "viewspace_points": screenspace_points,
    #     "visibility_filter": radii > 0,
    #     "radii": radii,
    #     "contrib": contrib,
    #     "depth": rendered_depth[[1]] if cfg.median_depth else rendered_depth[[0]],
    #     "depth_mean": rendered_depth[[0]],
    #     "depth_median": rendered_depth[[1]],
    #     "distortion": rendered_depth[[2]],
    #     "depth_square": rendered_depth[[3]],
    #     "alpha": rendered_opacity,
    #     "feature": rendered_other,
    #     "normal": rendered_normal,
    #     "intensity_sh": rendered_intensity_sh,
    #     "raydrop": rendered_raydrop.clamp(0, 1)
    # }
    return {
        "depth": rendered_depth[[0]],          # mean: 알파 가중 기대 depth (depth loss + metric)
        "depth_median": rendered_depth[[1]],   # median: T가 0.5를 넘는 contributor의 depth (chamfer + median loss)
        "normal": rendered_normal,
        "intensity_sh": rendered_intensity_sh,
        "raydrop": rendered_raydrop.clamp(0, 1)
    }
