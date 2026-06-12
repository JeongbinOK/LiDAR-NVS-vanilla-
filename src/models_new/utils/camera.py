import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
import kornia
from torchvision.utils import save_image
# from ..utils.render_utils import (
#     build_gt_normal_map,
#     quat_to_rotmat,
#     render_primitives,
#     rotmat_to_quat,
# )



def get_batch_idx(offset):
    sizes = torch.diff(
        torch.cat([torch.zeros(1, dtype=offset.dtype, device=offset.device), offset])
    )
    return torch.repeat_interleave(
        torch.arange(len(offset), device=offset.device), sizes
    )


def f2w_pose(points, poses, offset):
    """
    points: [N, 3+]   (offset 기반 batch)
    poses:  [B, 4, 4] world←frame transform
    offset: [B]
    """
    batch_idx = get_batch_idx(offset)         # [N]
    R = poses[batch_idx, :3, :3]              # [N, 3, 3]
    t = poses[batch_idx, :3, 3]               # [N, 3]

    xyz = points[:, :3]
    xyz_world = (R @ xyz.unsqueeze(-1)).squeeze(-1) + t  # R @ p + t

    return torch.cat([xyz_world, points[:, 3:]], dim=-1)


def w2f_pose(points, poses, offset):
    """
    points: [N, 3+]   (offset 기반 batch)
    poses:  [B, 4, 4] world←frame transform
    offset: [B]
    """
    batch_idx = get_batch_idx(offset)         # [N]
    R = poses[batch_idx, :3, :3]              # [N, 3, 3]
    t = poses[batch_idx, :3, 3]               # [N, 3]

    xyz = points[:, :3]
    xyz_local = (R.transpose(-1, -2) @ (xyz - t).unsqueeze(-1)).squeeze(-1)  # R^T @ (p - t)

    return torch.cat([xyz_local, points[:, 3:]], dim=-1)


def intensity_dir(coords, keep_mask):
    # direction: origin → anchor 방향 단위벡터
    anchor_valid = coords[keep_mask]                               # (Ui_valid, 3)
    # norm         = anchor_valid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # direction    = anchor_valid / norm    
    return anchor_valid 


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    orig_h, orig_w = args.hw

    if args.resolution == -1:
        global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    vfov = args.vfov
    hfov = args.hfov
    if cam_info.pointcloud_camera is not None:
        intensity = cam_info.intensity
        if intensity is None:
            intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

        w = resolution[0]
        h = resolution[1]

        pts_depth = np.zeros([1, h, w])
        pts_intensity = np.zeros([1, h, w])
        point_camera = cam_info.pointcloud_camera
        x = point_camera[:, 0]
        y = point_camera[:, 1]
        z = point_camera[:, 2]
        phi = np.arctan2(x, z)
        theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
        VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
        HFOV_max = hfov[1] * np.pi / 180
        HFOV_min = hfov[0] * np.pi / 180

        theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
        phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
        uvz = np.stack((theta, phi, r, intensity), 1)

        uvz = uvz[uvz[:, 0] >= -0.5]
        uvz = uvz[uvz[:, 0] < h - 0.5]
        uvz = uvz[uvz[:, 1] >= -0.5]
        uvz = uvz[uvz[:, 1] < w - 0.5]
        uv = uvz[:, :2]
        uv = np.around(uv).astype(int)

        for i in range(uv.shape[0]):
            x, y = uv[i]
            if pts_depth[0, x, y] == 0:
                pts_depth[0, x, y] = uvz[i, 2]
                pts_intensity[0, x, y] = uvz[i, 3]
            elif uvz[i, 2] < pts_depth[0, x, y]:
                pts_depth[0, x, y] = uvz[i, 2]
                pts_intensity[0, x, y] = uvz[i, 3]

        pts_depth = torch.from_numpy(pts_depth).float().cuda()
        pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
    else:
        pts_depth = None
        pts_intensity = None

    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        vfov=vfov,
        hfov=hfov,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        pts_depth=pts_depth,
        pts_intensity=pts_intensity,
        towards=cam_info.towards
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list





class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, vfov=None, hfov=None, uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", timestamp=0.0,
                 resolution=None, image_path=None,
                 pts_depth=None, pts_intensity=None, towards=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.vfov = vfov
        self.hfov = hfov
        self.resolution = resolution
        self.image_path = image_path
        self.towards = towards

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.pts_depth = pts_depth.to(self.data_device) if pts_depth is not None else pts_depth
        self.pts_intensity = pts_intensity.to(self.data_device) if pts_intensity is not None else pts_intensity

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = torch.eye(4).cuda()  # no use
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device='cuda')[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u - self.cx + torch.rand_like(u)) / self.fx,
                                      (v - self.cy + torch.rand_like(v)) / self.fy,
                                      torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u - self.cx + 0.5) / self.fx,
                                      (v - self.cy + 0.5) / self.fy,
                                      torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_world_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        if train:
            theta = theta + torch.rand_like(theta.float()) - 0.5
            phi = phi + torch.rand_like(phi.float()) - 0.5

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height,
                                                                            self.image_width)
        return directions

    def get_local_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        assert self.towards is not None
        if self.towards == 'forward':
            directions = torch.stack([dx, dy, dz], dim=0)
        else:
            directions = torch.stack([-dx, dy, -dz], dim=0)
        directions = F.normalize(directions, dim=0)
        return directions