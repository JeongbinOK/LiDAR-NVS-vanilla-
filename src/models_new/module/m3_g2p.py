import torch
import torch.nn as nn
from torch import Tensor
from ..gaussian_renderer import render, render_range_map
from ..utils.graphics_utils import pano_to_lidar 
class GausRender(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.background = torch.tensor([1, 1, 1, 1], dtype=torch.float32)


    def get_means3D(self, b_gs, t):
        position     = b_gs["position"]            # (N, 3)
        vel_segs     = b_gs["velocity_segments"]   # List[Tensor(N, 3)]
        seg_ts       = b_gs["segment_ts"]          # (n_input,)
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



def forward(self, gaussians, gt_cameras_batch):
        # 텐서로 묶을 데이터용 리스트
        all_depths, all_intensity_shs, all_raydrops = [], [], []
        all_gt_depths, all_gt_intensity_shs, all_gt_raydrops = [], [], []
        
        # 포인트 개수가 유동적일 수 있으므로 파이썬 리스트로 유지
        all_render_points = []
        all_gt_points = []
        
        for b, (b_gs, gt_cameras) in enumerate(zip(gaussians, gt_cameras_batch)):
            if b_gs is None:
                raise ValueError(f"Batch {b}의 Gaussian 데이터가 None입니다.")

            b_depths, b_intensity_shs, b_raydrops = [], [], []
            b_gt_depths, b_gt_intensity_shs, b_gt_raydrops = [], [], []
            b_render_points, b_gt_points = [], []

            for gt_cam in gt_cameras:
                t = gt_cam.timestamp

                # gt timestamp 기준 position 계산
                means3D = self.get_means3D(b_gs, t)

                pc = {
                    "position": means3D,
                    "opacity":  b_gs["opacity"],
                    "scale":    b_gs["scale"],
                    "rotation": b_gs["rotation"],
                    "shs":      b_gs["shs"],
                }

                # 랜더링
                render_pkg = render(
                    viewpoint_camera=gt_cam,
                    pc=pc,
                    cfg=self.cfg,
                    bg_color=self.background,
                    is_training=(self.cfg.mode == "train"),
                )
                
                gt_depth = gt_cam.pts_depth
                gt_intensity_sh = gt_cam.pts_intensity
                gt_raydrop = 1.0 - (gt_depth > 0).float()


                render_points = pano_to_lidar(gt_depth, gt_cam.vfov, gt_cam.hfov)

                # 픽셀 기반 맵 모으기
                b_depths.append(render_pkg["depth"])
                b_intensity_shs.append(render_pkg["intensity_sh"])
                b_raydrops.append(render_pkg["raydrop"])

                b_gt_depths.append(gt_depth)
                b_gt_intensity_shs.append(gt_intensity_sh)
                b_gt_raydrops.append(gt_raydrop)

                # 포인트 데이터 모으기 (개수가 다를 수 있으므로 리스트 유지)
                b_render_points.append(render_points)
                b_gt_points.append(gt_cam.points)

            # 카메라 차원 stack
            all_depths.append(torch.stack(b_depths, dim=0))
            all_intensity_shs.append(torch.stack(b_intensity_shs, dim=0))
            all_raydrops.append(torch.stack(b_raydrops, dim=0))
            all_gt_depths.append(torch.stack(b_gt_depths, dim=0))
            all_gt_intensity_shs.append(torch.stack(b_gt_intensity_shs, dim=0))
            all_gt_raydrops.append(torch.stack(b_gt_raydrops, dim=0))

            # 포인트 리스트 저장
            all_render_points.append(b_render_points)
            all_gt_points.append(b_gt_points)

        # 최종 딕셔너리 반환
        return {
            # 고정 차원 텐서 (Batch, Cam, H, W)
            "depth": torch.stack(all_depths, dim=0),
            "intensity_sh": torch.stack(all_intensity_shs, dim=0),
            "raydrop": torch.stack(all_raydrops, dim=0),
            "gt_depth": torch.stack(all_gt_depths, dim=0),
            "gt_intensity_sh": torch.stack(all_gt_intensity_shs, dim=0),
            "gt_raydrop": torch.stack(all_gt_raydrops, dim=0),
            
            # 가변 크기 가능 리스트 (중첩 리스트 형태: [Batch][Cam])
            "render_points": all_render_points,
            "gt_points": all_gt_points
        }