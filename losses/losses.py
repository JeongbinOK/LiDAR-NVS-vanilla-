"""
Loss functions for feed-forward LiDAR NVS training.
Adapted from GS-LiDAR (ICLR 2025) train.py loss computations.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party'))

import torch
import torch.nn.functional as F
from utils.graphics import pano_to_lidar, depth_to_normal
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist


def depth_l1_loss(pred_depth, gt_depth):
    """L1 loss on valid (non-zero) depth pixels."""
    mask = gt_depth > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    return F.l1_loss(pred_depth[mask], gt_depth[mask])


def chamfer_loss(pred_depth, gt_depth, vfov, hfov):
    """Chamfer distance between predicted and GT point clouds."""
    mask = (gt_depth > 0).float()
    pred_pts = pano_to_lidar(pred_depth * mask, vfov, hfov)
    gt_pts = pano_to_lidar(gt_depth, vfov, hfov)

    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return torch.tensor(0.0, device=pred_depth.device)

    cham_fn = chamfer_3DDist()
    d1, d2, _, _ = cham_fn(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0))
    return d1.mean() + d2.mean()


def depth_smoothness_loss(pred_depth, gt_depth, grad_clip=0.01):
    """Gradient-matching smoothness loss on locally-smooth GT regions."""
    gt_gx = gt_depth[:, :, :-1] - gt_depth[:, :, 1:]
    gt_gy = gt_depth[:, :-1, :] - gt_depth[:, 1:, :]

    mask_x = (gt_depth[:, :, :-1] > 0) & (gt_depth[:, :, 1:] > 0)
    mask_y = (gt_depth[:, :-1, :] > 0) & (gt_depth[:, 1:, :] > 0)

    smooth_x = (torch.abs(gt_gx) < grad_clip) & mask_x
    smooth_y = (torch.abs(gt_gy) < grad_clip) & mask_y

    pred_gx = pred_depth[:, :, :-1] - pred_depth[:, :, 1:]
    pred_gy = pred_depth[:, :-1, :] - pred_depth[:, 1:, :]

    loss = torch.tensor(0.0, device=pred_depth.device)
    if smooth_x.sum() > 0:
        loss = loss + F.l1_loss(pred_gx[smooth_x], gt_gx[smooth_x])
    if smooth_y.sum() > 0:
        loss = loss + F.l1_loss(pred_gy[smooth_y], gt_gy[smooth_y])
    return loss


def boundary_depth_loss(pred_depth, gt_depth):
    """Depth L1 at boundary timesteps (t=0, t=0.5)."""
    return depth_l1_loss(pred_depth, gt_depth)


def velocity_reg_loss(velocity_list):
    """Regularize velocity magnitude."""
    total = torch.tensor(0.0, device=velocity_list[0].device)
    for v in velocity_list:
        total = total + torch.abs(v).mean()
    return total / len(velocity_list)


def opacity_entropy_loss(opacity_list):
    """Encourage binary opacity (0 or 1)."""
    total = torch.tensor(0.0, device=opacity_list[0].device)
    for o in opacity_list:
        o_c = o.clamp(1e-6, 1 - 1e-6)
        total = total - (o_c * torch.log(o_c)).mean()
    return total / len(opacity_list)


def intensity_l1_loss(pred_intensity, gt_intensity):
    """L1 loss on intensity map where GT is valid."""
    mask = gt_intensity > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_intensity.device)
    return F.l1_loss(pred_intensity[mask], gt_intensity[mask])


def normal_consistency_loss(pred_depth, rendered_normal, vfov, hfov):
    """Consistency between depth-derived normals and rasterized surface normals.

    Args:
        pred_depth: [1, H, W] predicted depth panorama.
        rendered_normal: [3, H, W] surface normals from rasterizer.
        vfov, hfov: FOV tuples in degrees.
    Returns:
        Scalar loss.
    """
    depth_normal = depth_to_normal(pred_depth, vfov, hfov)  # [3, H, W]
    mask = (pred_depth.squeeze(0) > 0) & (rendered_normal.norm(dim=0) > 0.5)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    # 1 - cosine similarity
    cos = (depth_normal * rendered_normal).sum(dim=0)
    return (1.0 - cos[mask]).mean()


def depth_var_loss(pred_depth, depth_sq):
    """Minimize rendered depth variance (encourages sharp depth).

    Args:
        pred_depth: [1, H, W] rendered mean depth.
        depth_sq: [1, H, W] rendered depth squared (E[d^2]).
    Returns:
        Scalar loss — mean variance over valid pixels.
    """
    mask = pred_depth > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    var = (depth_sq - pred_depth ** 2).clamp(min=0.0)
    return var[mask].mean()


def temporal_smooth_loss(pred_depths):
    """L1 difference between consecutive predicted depth maps.

    Args:
        pred_depths: list of [1, H, W] depth maps sorted by timestamp.
    Returns:
        Scalar loss.
    """
    if len(pred_depths) < 2:
        return torch.tensor(0.0, device=pred_depths[0].device)
    total = torch.tensor(0.0, device=pred_depths[0].device)
    count = 0
    for i in range(len(pred_depths) - 1):
        mask = (pred_depths[i] > 0) & (pred_depths[i + 1] > 0)
        if mask.sum() > 0:
            total = total + F.l1_loss(pred_depths[i][mask], pred_depths[i + 1][mask])
            count += 1
    return total / max(count, 1)


def compute_total_loss(pred_depths, pred_intensities, gt_depths, gt_intensities,
                       gaussian_params, cfg_loss, vfov, hfov,
                       boundary_pred_depths=None, boundary_gt_depths=None,
                       depth_sqs=None, rendered_normals=None):
    """Compute weighted sum of all losses over multiple GT timesteps.

    Args:
        pred_depths: list of [1, H, W] predicted depth panoramas.
        pred_intensities: list of [1, H, W] predicted intensity panoramas.
        gt_depths: list of [1, H, W] ground-truth depth panoramas.
        gt_intensities: list of [1, H, W] ground-truth intensity panoramas.
        gaussian_params: dict with 'velocity', 'opacity' etc — lists of [N, ...].
        cfg_loss: loss weight config dict.
        vfov, hfov: FOV tuples.
        boundary_pred_depths: list of [1, H, W] for t=0, t=0.5.
        boundary_gt_depths: list of [1, H, W] for t=0, t=0.5.
        depth_sqs: list of [1, H, W] rendered depth squared maps.
        rendered_normals: list of [3, H, W] rendered surface normals.

    Returns:
        total_loss: scalar.
        loss_dict: dict of individual loss values for logging.
    """
    device = pred_depths[0].device
    loss_dict = {}
    total = torch.tensor(0.0, device=device)
    n_steps = len(pred_depths)

    # === Per-timestep losses ===
    l_depth = torch.tensor(0.0, device=device)
    l_chamfer = torch.tensor(0.0, device=device)
    l_smooth = torch.tensor(0.0, device=device)
    l_intensity = torch.tensor(0.0, device=device)
    l_normal = torch.tensor(0.0, device=device)
    l_dvar = torch.tensor(0.0, device=device)

    for i in range(n_steps):
        l_depth = l_depth + depth_l1_loss(pred_depths[i], gt_depths[i])
        if cfg_loss['chamfer'] > 0:
            l_chamfer = l_chamfer + chamfer_loss(pred_depths[i], gt_depths[i],
                                                 vfov, hfov)
        if cfg_loss['smooth'] > 0:
            l_smooth = l_smooth + depth_smoothness_loss(pred_depths[i], gt_depths[i])
        if cfg_loss['intensity_l1'] > 0:
            l_intensity = l_intensity + intensity_l1_loss(
                pred_intensities[i], gt_intensities[i])
        if cfg_loss.get('normal_consistency', 0) > 0 and rendered_normals is not None:
            l_normal = l_normal + normal_consistency_loss(
                pred_depths[i], rendered_normals[i], vfov, hfov)
        if cfg_loss.get('depth_var', 0) > 0 and depth_sqs is not None:
            l_dvar = l_dvar + depth_var_loss(pred_depths[i], depth_sqs[i])

    l_depth = l_depth / max(n_steps, 1)
    l_chamfer = l_chamfer / max(n_steps, 1)
    l_smooth = l_smooth / max(n_steps, 1)
    l_intensity = l_intensity / max(n_steps, 1)
    l_normal = l_normal / max(n_steps, 1)
    l_dvar = l_dvar / max(n_steps, 1)

    total = total + cfg_loss['depth_l1'] * l_depth
    loss_dict['depth_l1'] = l_depth.item()

    if cfg_loss['chamfer'] > 0:
        total = total + cfg_loss['chamfer'] * l_chamfer
        loss_dict['chamfer'] = l_chamfer.item()

    if cfg_loss['smooth'] > 0:
        total = total + cfg_loss['smooth'] * l_smooth
        loss_dict['smooth'] = l_smooth.item()

    if cfg_loss['intensity_l1'] > 0:
        total = total + cfg_loss['intensity_l1'] * l_intensity
        loss_dict['intensity_l1'] = l_intensity.item()

    if cfg_loss.get('normal_consistency', 0) > 0 and rendered_normals is not None:
        total = total + cfg_loss['normal_consistency'] * l_normal
        loss_dict['normal_consistency'] = l_normal.item()

    if cfg_loss.get('depth_var', 0) > 0 and depth_sqs is not None:
        total = total + cfg_loss['depth_var'] * l_dvar
        loss_dict['depth_var'] = l_dvar.item()

    # === Temporal smoothness ===
    if cfg_loss.get('temporal_smooth', 0) > 0:
        l_tsmooth = temporal_smooth_loss(pred_depths)
        total = total + cfg_loss['temporal_smooth'] * l_tsmooth
        loss_dict['temporal_smooth'] = l_tsmooth.item()

    # === Boundary depth loss ===
    if boundary_pred_depths is not None and cfg_loss['boundary_depth'] > 0:
        l_bnd = torch.tensor(0.0, device=device)
        for bp, bg in zip(boundary_pred_depths, boundary_gt_depths):
            l_bnd = l_bnd + boundary_depth_loss(bp, bg)
        l_bnd = l_bnd / len(boundary_pred_depths)
        total = total + cfg_loss['boundary_depth'] * l_bnd
        loss_dict['boundary_depth'] = l_bnd.item()

    # === Regularization ===
    if cfg_loss['velocity_reg'] > 0:
        l_vreg = velocity_reg_loss(gaussian_params['velocity'])
        total = total + cfg_loss['velocity_reg'] * l_vreg
        loss_dict['velocity_reg'] = l_vreg.item()

    if cfg_loss['opacity_entropy'] > 0:
        l_opa = opacity_entropy_loss(gaussian_params['opacity'])
        total = total + cfg_loss['opacity_entropy'] * l_opa
        loss_dict['opacity_entropy'] = l_opa.item()

    loss_dict['total'] = total.item()
    return total, loss_dict
