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

def render_range_map(args, cam_front: Camera, cam_back: Camera, gaussians, renderFunc, renderArgs, env_map, hw):
    # assert cam_front.towards == "forward" and cam_back.towards == "backward"
    # assert cam_front.colmap_id + args.frames == cam_back.colmap_id

    EPS = 1e-5
    h, w = hw
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    depth_pano = torch.zeros([3, h, w * 2]).cuda()
    intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
    raydrop_pano = torch.zeros([1, h, w * 2]).cuda()
    gt_depth_pano = torch.zeros([1, h, w * 2]).cuda()
    gt_intensity_pano = torch.zeros([1, h, w * 2]).cuda()

    for idx, viewpoint in enumerate([cam_front, cam_back]):
        depth_gt = viewpoint.pts_depth
        intensity_gt = viewpoint.pts_intensity
        render_pkg = renderFunc(viewpoint, gaussians, *renderArgs, env_map=env_map)

        depth = render_pkg['depth']
        alpha = render_pkg['alpha']
        raydrop_render = render_pkg['raydrop']

        depth_var = render_pkg['depth_square'] - depth ** 2
        depth_median = render_pkg["depth_median"]
        var_quantile = depth_var.median() * 10

        depth_mix = torch.zeros_like(depth)
        depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
        depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]

        depth = torch.cat([depth_mix, depth, depth_median])

        if args.sky_depth:
            sky_depth = 900
            depth = depth / alpha.clamp_min(EPS)
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        intensity_sh = render_pkg['intensity_sh']

        if idx % 2 == 0:  # 前180度
            depth_pano[:, :, breaks[1]:breaks[2]] = depth
            gt_depth_pano[:, :, breaks[1]:breaks[2]] = depth_gt

            intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
            gt_intensity_pano[:, :, breaks[1]:breaks[2]] = intensity_gt

            raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render

            continue
        else:
            depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
            depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]

            gt_depth_pano[:, :, breaks[2]:breaks[3]] = depth_gt[:, :, 0:(breaks[3] - breaks[2])]
            gt_depth_pano[:, :, breaks[0]:breaks[1]] = depth_gt[:, :, (w - breaks[1] + breaks[0]):w]

            intensity_sh_pano[:, :, breaks[2]:breaks[3]] = intensity_sh[:, :, 0:(breaks[3] - breaks[2])]
            intensity_sh_pano[:, :, breaks[0]:breaks[1]] = intensity_sh[:, :, (w - breaks[1] + breaks[0]):w]

            gt_intensity_pano[:, :, breaks[2]:breaks[3]] = intensity_gt[:, :, 0:(breaks[3] - breaks[2])]
            gt_intensity_pano[:, :, breaks[0]:breaks[1]] = intensity_gt[:, :, (w - breaks[1] + breaks[0]):w]

            raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop_render[:, :, 0:(breaks[3] - breaks[2])]
            raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop_render[:, :, (w - breaks[1] + breaks[0]):w]

    return depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano


def get_position_at_t(b_result, t_val):
    """
    b_result  : GausTemp output의 배치 b 결과
    t_val     : float, 0~1 timestamp
    """
    position          = b_result["position"]           # (Nb, 3)
    fg_masks          = b_result["fg_masks"]           # {box_id: (Nb,) bool}
    velocity_segments = b_result["velocity_segments"]  # List[dict]
    ts                = b_result["input_timestamps"]   # (n_input,)
    device            = position.device

    pos_t = position.clone()

    for box_id, fg_mask in fg_masks.items():
        if fg_mask.sum() == 0:
            continue

        displacement = torch.zeros(3, device=device)
        for seg in range(len(velocity_segments)):
            t_start = ts[seg].item()
            t_end   = ts[seg + 1].item()

            if t_val <= t_start:
                break
            elif t_val >= t_end:
                dt_seg = t_end - t_start
            else:
                dt_seg = t_val - t_start

            if box_id in velocity_segments[seg]:
                displacement += velocity_segments[seg][box_id] * dt_seg

        pos_t[fg_mask] = position[fg_mask] + displacement


def get_means3D(pc, t):
    """
    pc["position"]          : (N, 3)
    pc["velocity_segments"] : List[Tensor(N, 3)]  구간별 velocity
    pc["segment_ts"]        : Tensor(n_input,)    구간 경계 timestamps
    t                       : float, 0~1
    """
    position     = pc["position"]           # (N, 3)
    vel_segs     = pc["velocity_segments"]  # List[Tensor(N, 3)]
    seg_ts       = pc["segment_ts"]         # (n_input,)
    device       = position.device

    displacement = torch.zeros_like(position)

    for seg in range(len(vel_segs)):
        t_start = seg_ts[seg].item()
        t_end   = seg_ts[seg + 1].item()

        if t <= t_start:
            break
        elif t >= t_end:
            dt_seg = t_end - t_start
        else:
            dt_seg = t - t_start

        displacement += vel_segs[seg] * dt_seg

    return position + displacement   # (N, 3)
