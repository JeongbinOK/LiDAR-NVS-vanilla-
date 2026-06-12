import torch
import torch.nn as nn
from torch import Tensor
from gaussian_renderer import render, render_range_map
class GausRender(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.renderer = Renderer(sh_degree=3, white_background=True, radius=1, render="2dgs")
        self.background = torch.tensor(bg_color, dtype=torch.float32)
    def forward(self, x, time, view):
        
        render_pkg = render(view, x["gaussians"], self.cfg.mode, self.background, time_shift=time, is_training=(self.cfg.mode==True))
        
        depth = render_pkg["depth"]
        depth_median = render_pkg["depth_median"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        log_dict = {}
        intensity_sh_map = render_pkg['intensity_sh']
        raydrop_map = render_pkg['raydrop']
        points_position = render_pkg['points_position']
        return depth, points_position, intensity, raydrop_map