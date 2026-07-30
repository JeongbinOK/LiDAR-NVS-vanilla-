import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
import numpy as np
from .graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
import kornia
from torchvision.utils import save_image
from pyquaternion import Quaternion
# from ..utils.render_utils import (
#     build_gt_normal_map,
#     quat_to_rotmat,
#     render_primitives,
#     rotmat_to_quat,
# )



# def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
#     orig_h, orig_w = args.hw

#     if args.resolution == -1:
#         global_down = 1
#     else:
#         global_down = orig_w / args.resolution

#     scale = float(global_down) * float(resolution_scale)
#     resolution = (int(orig_w / scale), int(orig_h / scale))

#     vfov = args.vfov
#     hfov = args.hfov
#     if cam_info.pointcloud_camera is not None:
#         intensity = cam_info.intensity
#         if intensity is None:
#             intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

#         w = resolution[0]
#         h = resolution[1]

#         pts_depth = np.zeros([1, h, w])
#         pts_intensity = np.zeros([1, h, w])
#         point_camera = cam_info.pointcloud_camera
#         x = point_camera[:, 0]
#         y = point_camera[:, 1]
#         z = point_camera[:, 2]
#         phi = np.arctan2(x, z)
#         theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
#         r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

#         VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
#         VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
#         HFOV_max = hfov[1] * np.pi / 180
#         HFOV_min = hfov[0] * np.pi / 180

#         theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
#         phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
#         uvz = np.stack((theta, phi, r, intensity), 1)

#         uvz = uvz[uvz[:, 0] >= -0.5]
#         uvz = uvz[uvz[:, 0] < h - 0.5]
#         uvz = uvz[uvz[:, 1] >= -0.5]
#         uvz = uvz[uvz[:, 1] < w - 0.5]
#         uv = uvz[:, :2]
#         uv = np.around(uv).astype(int)

#         for i in range(uv.shape[0]):
#             x, y = uv[i]
#             if pts_depth[0, x, y] == 0:
#                 pts_depth[0, x, y] = uvz[i, 2]
#                 pts_intensity[0, x, y] = uvz[i, 3]
#             elif uvz[i, 2] < pts_depth[0, x, y]:
#                 pts_depth[0, x, y] = uvz[i, 2]
#                 pts_intensity[0, x, y] = uvz[i, 3]

#         pts_depth = torch.from_numpy(pts_depth).float().cuda()
#         pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
#     else:
#         pts_depth = None
#         pts_intensity = None

#     return Camera(
#         colmap_id=cam_info.uid,
#         uid=id,
#         R=cam_info.R,
#         T=cam_info.T,
#         vfov=vfov,
#         hfov=hfov,
#         data_device=args.data_device,
#         timestamp=cam_info.timestamp,
#         resolution=resolution,
#         pts_depth=pts_depth,
#         pts_intensity=pts_intensity,
#         towards=cam_info.towards
#     )


# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
#     camera_list = []

#     for id, c in enumerate(tqdm(cam_infos)):
#         camera_list.append(loadCam(args, id, c, resolution_scale))

#     return camera_list





class Camera(nn.Module):
    def __init__(self, R, T, vfov=None, hfov=None, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cpu", timestamp=0.0,
                 resolution=None, 
                 pts_depth=None, pts_intensity=None, points=None,
                 viewmatrix=None,
                 ring_to_elevation_deg=None,
                 ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.vfov = vfov
        self.hfov = hfov
        self.resolution = resolution
        

        # try:
        #     self.data_device = torch.device(data_device)
        # except Exception as e:
        #     print(e)
        #     print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
        #     self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.ring_to_elevation_deg = (
            list(ring_to_elevation_deg) if ring_to_elevation_deg is not None else None
        )
        self.row_to_theta = self._build_row_to_theta(
            self.image_height,
            vfov,
            self.ring_to_elevation_deg,
        )

        self.pts_depth = pts_depth if pts_depth is not None else pts_depth
        self.pts_intensity = pts_intensity if pts_intensity is not None else pts_intensity
        self.points = points

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if viewmatrix is None:
            view = getWorld2View2(R, T, trans, scale)
        else:
            view = np.asarray(viewmatrix, dtype=np.float32)
        self.world_view_transform = torch.tensor(view).transpose(0, 1)
        self.projection_matrix = torch.eye(4) # no use
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(
            self.image_height,
            self.image_width,
            normalized_coordinates=False,
            device="cpu",
        )[0]

    @staticmethod
    def _build_row_to_theta(image_height, vfov, ring_to_elevation_deg=None):
        if ring_to_elevation_deg is not None:
            row_to_el = np.sort(np.asarray(list(ring_to_elevation_deg), dtype=np.float32))[::-1]
            if row_to_el.shape[0] != int(image_height):
                raise ValueError(
                    f"ring_to_elevation_deg length {row_to_el.shape[0]} "
                    f"does not match image_height {image_height}"
                )
        else:
            row = np.arange(int(image_height), dtype=np.float32)
            row_to_el = float(vfov[1]) - row / float(image_height) * (float(vfov[1]) - float(vfov[0]))
        row_to_theta = np.pi / 2.0 - np.radians(row_to_el)
        return torch.from_numpy(row_to_theta.astype(np.float32))


    @classmethod
    def from_nuscenes(cls, nusc, lidar_token, timestamp_normalized,
                      pts_sensor, cfg, uid=0, lidar_ring=None, ref_to_sensor=None):
        """
        NuScenes lidar_token으로 Camera 객체 생성
        pts_sensor : (N, 4) sensor frame xyz+intensity
        """
        sd = nusc.get('sample_data', lidar_token)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', sd['ego_pose_token'])

        s2e = np.eye(4, dtype=np.float32)
        s2e[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
        s2e[:3,  3] = cs['translation']
        e2w = np.eye(4, dtype=np.float32)
        e2w[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
        e2w[:3,  3] = ep['translation']
        s2w = e2w @ s2e

        R = s2w[:3, :3]
        T = s2w[:3,  3]

        vfov = cfg.vfov
        hfov = cfg.hfov
        H    = cfg.image_height
        W    = cfg.image_width
        ring_to_elevation_deg = getattr(cfg, "ring_to_elevation_deg", None)

        # range image 생성
        #####
        #현재는 균등 상태. GT intensity, depth 만들고 싶으면 참고. 
        #####
        # nuScenes LiDAR(z-up) → 카메라(y-down) 축 swap: (x,y,z)->(x,-z,y).
        # range image/rasterizer가 y를 수직축으로 쓰므로 z-up 점을 그대로 넣으면 elevation(행)이
        # 스크램블된다(검증: corr(ring, rasterizer행)≈0). GS-LiDAR nuscenes_loader.py:73-82와 동일.
        x = pts_sensor[:, 0].numpy()
        y = -pts_sensor[:, 2].numpy()
        z = pts_sensor[:, 1].numpy()

        intensity = pts_sensor[:, 3].numpy()

        phi   = np.arctan2(x, z)
        theta = np.arctan2(np.sqrt(x**2 + z**2), -y)
        r     = np.sqrt(x**2 + y**2 + z**2)

        VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
        VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
        HFOV_max = hfov[1] * np.pi / 180
        HFOV_min = hfov[0] * np.pi / 180

        theta_px = (theta - VFOV_min) * H / (VFOV_max - VFOV_min)
        phi_px   = (phi   - HFOV_min) * W / (HFOV_max - HFOV_min)


        uvz = np.stack([theta_px, phi_px, r, intensity], axis=1)
        # ring으로 각 점을 정확한 beam 행에 snap → 기하 elevation 노이즈/한 elevation 충돌 방지(GT 정확도↑).
        # render/Camera convention은 row0=최고 elevation이므로, 실제 beam table을 높은 elevation부터
        # 정렬한 순서로 panorama row를 만든다.
        if lidar_ring is not None:
            ring_np = lidar_ring.numpy() if torch.is_tensor(lidar_ring) else np.asarray(lidar_ring)
            ring_to_el = ring_to_elevation_deg
            if ring_to_el is not None:
                ring_to_el = np.asarray(list(ring_to_el), dtype=np.float32)
                if ring_to_el.shape[0] != H:
                    raise ValueError(
                        f"ring_to_elevation_deg length {ring_to_el.shape[0]} "
                        f"does not match image_height {H}"
                    )
                ring_order_top_down = np.argsort(-ring_to_el)
                ring_to_row = np.empty_like(ring_order_top_down)
                ring_to_row[ring_order_top_down] = np.arange(H)
                valid_ring = (ring_np >= 0) & (ring_np < H)
                uvz[valid_ring, 0] = ring_to_row[ring_np[valid_ring]].astype(np.float32)
            else:
                valid_ring = (ring_np >= 0) & (ring_np < H)
                el_ring = vfov[0] + ring_np[valid_ring] / (H - 1) * (vfov[1] - vfov[0])
                theta_ring = np.pi / 2 - np.radians(el_ring)
                uvz[valid_ring, 0] = (theta_ring - VFOV_min) * H / (VFOV_max - VFOV_min)
        uvz = uvz[(uvz[:, 0] >= -0.5) & (uvz[:, 0] < H - 0.5)]
        uvz = uvz[(uvz[:, 1] >= -0.5) & (uvz[:, 1] < W - 0.5)]
        uv  = np.around(uvz[:, :2]).astype(int)

        pts_depth_np     = np.zeros([1, H, W], dtype=np.float32)
        pts_intensity_np = np.zeros([1, H, W], dtype=np.float32)
        for i in range(uv.shape[0]):
            row, col = uv[i]
            if pts_depth_np[0, row, col] == 0 or uvz[i, 2] < pts_depth_np[0, row, col]:
                pts_depth_np[0, row, col]     = uvz[i, 2]
                pts_intensity_np[0, row, col] = uvz[i, 3]

        # viewmatrix(ref→sensor)에도 동일 swap을 곱해 gaussian 렌더를 swapped 카메라 프레임으로 맞춘다.
        # M4: sensor(z-up) → camera(y-down). render와 GT range image가 같은 프레임을 쓰게 됨.
        if ref_to_sensor is not None:
            M4 = np.array([[1, 0, 0, 0],
                           [0, 0, -1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
            ref_to_sensor = (M4 @ np.asarray(ref_to_sensor, dtype=np.float32)).astype(np.float32)

        return cls(
            R=R, T=T,
            vfov=vfov, hfov=hfov,
            data_device="cpu",
            timestamp=timestamp_normalized,
            resolution=(W, H),
            points = pts_sensor[:,:3],
            pts_depth=torch.from_numpy(pts_depth_np).float(),
            pts_intensity=torch.from_numpy(pts_intensity_np).float(),
            viewmatrix=ref_to_sensor,
            ring_to_elevation_deg=ring_to_elevation_deg,
        )

