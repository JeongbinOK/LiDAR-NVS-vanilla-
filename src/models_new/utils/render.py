import os
import torch
import numpy as np
import random
from matplotlib import cm


def visualize_depth(depth, near=2, far=50, linear=False, scale_factor=None):
    if scale_factor is not None:
        depth = depth / scale_factor

    depth = depth[0].clone().detach().cpu().numpy()
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    out_depth = torch.from_numpy(out_depth).permute(2, 0, 1).float().cuda() / 255
    return out_depth


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L




class Gaussianutil: 
    def __init__(self, cfg):
        # softplus (not exp): its derivative is sigmoid (<=1) and growth is linear, not
        # explosive, so a large raw scale logit can no longer overflow the rasterizer
        # backward into a NaN gradient (the epoch-9 collapse trigger). c=1, plain softplus.
        self.scaling_activation = torch.nn.functional.softplus
        self.scaling_inverse_activation = lambda x: torch.log(torch.expm1(x))

        self.scaling_t_activation = torch.exp
        self.scaling_t_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.intensity_activation = torch.sigmoid

    
    def get_scaling(self, scale, max_scale_m=None):
        """Apply softplus, then optionally hard-clip each scale axis in metres.

        Values above the cap receive zero direct gradient through this output,
        matching the intended emergency-bound behavior of a regular clamp.
        """
        scaling = self.scaling_activation(scale)
        if max_scale_m is None:
            return scaling
        max_scale_m = float(max_scale_m)
        if not max_scale_m > 0.0:
            raise ValueError("max_scale_m must be positive")
        return scaling.clamp_max(max_scale_m)
    
    def get_covariance(self, scale, scaling_modifier, rotation,
                       max_scale_m=None):
        return self.covariance_activation(
            self.get_scaling(scale, max_scale_m=max_scale_m),
            scaling_modifier,
            self.get_rotation(rotation),
        )
    
    def get_opacity(self, opacity):
        return self.opacity_activation(opacity)
    
    def get_rotation(self, rotation):
        return self.rotation_activation(rotation)
    
    def get_intensity(self, intensity):
        return self.intensity_activation(intensity)

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm
