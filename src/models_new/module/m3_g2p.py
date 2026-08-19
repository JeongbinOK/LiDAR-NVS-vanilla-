import torch
import torch.nn as nn
from torch import Tensor
from ..gaussian_renderer import render
from .gaussian_assembly import opacity_gate_st
from ..utils.graphics_utils import lidar4d_range_image_to_points


class GausRender(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Channels are [unused, unused, intensity, raydrop]. No-return background
        # should carry zero intensity/depth support and raydrop probability 1.
        self.background = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

    @staticmethod
    def render_timestamp(camera):
        """Legacy bbox transport uses the normalized window coordinate."""
        return camera.timestamp

    def pack_lidar_shs(self, shs):
        if shs.dim() == 3:
            return shs
        if shs.shape[-1] == 64:
            return shs.reshape(shs.shape[0], 16, 4)
        if shs.shape[-1] != 32:
            raise ValueError(f"Expected SH tensor with 32 or 64 channels, got {shs.shape[-1]}.")

        coeff = shs.reshape(shs.shape[0], 16, 2)
        unused = torch.zeros(shs.shape[0], 16, 2, device=shs.device, dtype=shs.dtype)
        return torch.cat([unused, coeff], dim=-1)

    @staticmethod
    def pack_scales(scales):
        if scales.shape[-1] == 3:
            return scales
        if scales.shape[-1] != 2:
            raise ValueError(f"Expected scale tensor with 2 or 3 channels, got {scales.shape[-1]}.")
        z_log_scale = torch.zeros(scales.shape[0], 1, device=scales.device, dtype=scales.dtype)
        return torch.cat([scales, z_log_scale], dim=-1)

    @staticmethod
    def group_target_views(b_gs, num_views):
        """Build one Common+Additional index vector per view with one sort.

        Common rows use the reserved ``view_index=-1`` and are prepended to
        every transient render union. Additional rows are sorted once by their
        non-negative target-view tag, avoiding a full-union boolean scan for
        every rasterizer call.
        """
        gaussian_view = b_gs.get("view_index")
        if gaussian_view is None:
            return [None] * int(num_views)
        num_views = int(num_views)
        if num_views <= 0:
            return []
        if gaussian_view.ndim != 1:
            raise ValueError("view_index must be one-dimensional")

        common_rows = (gaussian_view == -1).nonzero(as_tuple=True)[0]
        additional_rows = (gaussian_view >= 0).nonzero(as_tuple=True)[0]
        additional_view = gaussian_view[additional_rows].long()
        if additional_view.numel() > 0:
            order = torch.argsort(additional_view, stable=True)
            sorted_rows = additional_rows[order]
            counts = torch.bincount(
                additional_view, minlength=num_views
            ).tolist()
            if len(counts) != num_views:
                raise ValueError(
                    "Additional Gaussian view_index falls outside target cameras"
                )
        else:
            sorted_rows = additional_rows
            counts = [0] * num_views

        grouped = []
        start = 0
        for count in counts:
            end = start + int(count)
            grouped.append(torch.cat([common_rows, sorted_rows[start:end]]))
            start = end
        return grouped

    @staticmethod
    def select_target_view(b_gs, view_index, gaussian_indices=None):
        """Construct the transient ``Common union Additional_view`` tensors."""
        gaussian_view = b_gs.get("view_index")
        if gaussian_view is None:
            return b_gs
        if gaussian_indices is None:
            common_indices = (gaussian_view == -1).nonzero(as_tuple=True)[0]
            additional_indices = (
                gaussian_view == int(view_index)
            ).nonzero(as_tuple=True)[0]
            gaussian_indices = torch.cat([
                common_indices, additional_indices,
            ])
        if gaussian_indices.numel() == 0:
            raise RuntimeError(
                f"No Gaussians were generated for target view {view_index}"
            )
        gaussian_keys = {
            "position", "coord", "coord_ref", "box_assign", "instance_id",
            "is_dynamic", "bg_mask", "shs", "opacity", "scaling", "scale",
            "scales", "rotation", "rotations", "view_index",
        }
        selected = {}
        for key, value in b_gs.items():
            if key in gaussian_keys and torch.is_tensor(value):
                selected[key] = value.index_select(0, gaussian_indices)
            elif key == "common_view_gate":
                # This compact [num_common, num_views] tensor is consumed below,
                # not expanded into a Gaussian-aligned render attribute.
                continue
            elif key == "fg_masks" and isinstance(value, dict):
                selected[key] = {
                    item_id: item_mask.index_select(0, gaussian_indices)
                    for item_id, item_mask in value.items()
                }
            else:
                selected[key] = value

        common_view_gate = b_gs.get("common_view_gate")
        if common_view_gate is not None:
            if common_view_gate.ndim != 2:
                raise ValueError("common_view_gate must have shape (N_common, V)")
            common_count = common_view_gate.shape[0]
            if common_count == 0:
                return selected
            if common_count > selected["opacity"].shape[0]:
                raise ValueError(
                    "common_view_gate rows must align with stored Common Gaussians"
                )
            if int(view_index) >= common_view_gate.shape[1]:
                raise IndexError("target view exceeds common_view_gate width")
            common_opacity = opacity_gate_st(
                selected["opacity"][:common_count],
                common_view_gate[:, int(view_index)],
            )
            selected["opacity"] = torch.cat([
                common_opacity, selected["opacity"][common_count:],
            ])
        return selected


    def get_means3D(self, b_gs, t):
        position = b_gs["position"]
        device = position.device
        means = b_gs.get("coord_ref", position).to(device).clone()
        is_dynamic = b_gs.get(
            "is_dynamic",
            torch.zeros((position.shape[0],), dtype=torch.bool, device=device),
        ).to(device)
        instance_id = b_gs.get(
            "instance_id",
            torch.full((position.shape[0],), -1, dtype=torch.long, device=device),
        ).to(device)

        if (~is_dynamic).any():
            means[~is_dynamic] = position[~is_dynamic]

        for inst_id, trajectory in b_gs.get("object_trajectories", {}).items():
            mask = is_dynamic & (instance_id == int(inst_id))
            if not mask.any():
                continue
            box_t = self.interpolate_box_ref(trajectory, t, device, position.dtype)
            means[mask] = self.box_local_to_ref(position[mask], box_t)

        return means

    @staticmethod
    def _angle_lerp(yaw0, yaw1, alpha):
        delta = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))
        return yaw0 + alpha * delta

    def interpolate_box_ref(self, trajectory, t, device, dtype):
        times = trajectory["timestamps"].to(device=device, dtype=dtype)
        boxes = trajectory["bbox_ref"].to(device=device, dtype=dtype)
        t_tensor = torch.as_tensor(float(t), device=device, dtype=dtype)

        if t_tensor <= times[0]:
            return boxes[0]
        if t_tensor >= times[-1]:
            return boxes[-1]

        hi = int(torch.searchsorted(times, t_tensor).item())
        lo = max(hi - 1, 0)
        alpha = (t_tensor - times[lo]) / (times[hi] - times[lo]).clamp_min(1e-6)
        box = boxes[lo].clone()
        box[:3] = boxes[lo, :3] * (1.0 - alpha) + boxes[hi, :3] * alpha
        box[3:6] = boxes[lo, 3:6] * (1.0 - alpha) + boxes[hi, 3:6] * alpha
        box[6] = self._angle_lerp(boxes[lo, 6], boxes[hi, 6], alpha)
        return box

    @staticmethod
    def box_local_to_ref(local_points, box_ref):
        yaw = box_ref[6]
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        x = cos_y * local_points[:, 0] - sin_y * local_points[:, 1]
        y = sin_y * local_points[:, 0] + cos_y * local_points[:, 1]
        z = local_points[:, 2]
        return torch.stack([x, y, z], dim=-1) + box_ref[:3].unsqueeze(0)

    @staticmethod
    def yaw_to_quat(yaw):
        half = 0.5 * yaw
        return torch.stack([
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ], dim=-1)

    @staticmethod
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def get_rotations(self, b_gs, t):
        rotations = b_gs["rotation"].clone()
        device = rotations.device
        is_dynamic = b_gs.get(
            "is_dynamic",
            torch.zeros((rotations.shape[0],), dtype=torch.bool, device=device),
        ).to(device)
        instance_id = b_gs.get(
            "instance_id",
            torch.full((rotations.shape[0],), -1, dtype=torch.long, device=device),
        ).to(device)

        for inst_id, trajectory in b_gs.get("object_trajectories", {}).items():
            mask = is_dynamic & (instance_id == int(inst_id))
            if not mask.any():
                continue
            box_t = self.interpolate_box_ref(trajectory, t, device, rotations.dtype)
            yaw_q = self.yaw_to_quat(box_t[6]).expand(mask.sum(), -1)
            rotations[mask] = self.quat_mul(yaw_q, rotations[mask])
        return rotations


    def forward(self, gaussians, gt):
        if isinstance(gaussians, dict):
            gaussians = gaussians.get("batch_gaussians", gaussians.get("gaussians"))
        if isinstance(gt, dict):
            gt_cameras_batch = gt["cameras"]
        else:
            gt_cameras_batch = gt

        # 텐서로 묶을 데이터용 리스트
        all_depths, all_depths_median, all_intensity_shs, all_raydrops = [], [], [], []
        # 진단 전용: eval에서만 수집해 학습 메모리/그래프를 그대로 둔다.
        all_alphas = [] if not self.training else None
        all_gt_depths, all_gt_intensity_shs, all_gt_raydrops = [], [], []
        
        # 포인트 개수가 유동적일 수 있으므로 파이썬 리스트로 유지
        all_render_points = []
        all_gt_points = []
        
        for b, (b_gs, gt_cameras) in enumerate(zip(gaussians, gt_cameras_batch)):
            if b_gs is None:
                raise ValueError(f"Batch {b}의 Gaussian 데이터가 None입니다.")

            b_depths, b_depths_median, b_intensity_shs, b_raydrops = [], [], [], []
            b_alphas = [] if all_alphas is not None else None
            b_gt_depths, b_gt_intensity_shs, b_gt_raydrops = [], [], []
            b_render_points, b_gt_points = [], []
            target_view_groups = self.group_target_views(
                b_gs, len(gt_cameras)
            )

            for view_index, gt_cam in enumerate(gt_cameras):
                t = self.render_timestamp(gt_cam)
                view_gs = self.select_target_view(
                    b_gs, view_index, target_view_groups[view_index]
                )

                # gt timestamp 기준 position 계산
                means3D = self.get_means3D(view_gs, t)
                rotations = self.get_rotations(view_gs, t)

                pc = {
                    "position": means3D,
                    "opacity":  view_gs["opacity"],
                    "scales":   self.pack_scales(view_gs.get("scales", view_gs.get("scale", view_gs["scaling"]))),
                    "rotation": rotations,
                    "rotations": rotations,
                    "shs":      self.pack_lidar_shs(view_gs["shs"]),
                }

                # 랜더링
                render_pkg = render(
                    viewpoint_camera=gt_cam,
                    pc=pc,
                    cfg=self.cfg,
                    bg_color=self.background.to(means3D.device),
                    input_timestamp=t,
                    is_training=self.training,
                )
                
                gt_depth = gt_cam.pts_depth.to(device=means3D.device, dtype=means3D.dtype)
                gt_intensity_sh = gt_cam.pts_intensity.to(device=means3D.device, dtype=means3D.dtype)
                gt_raydrop = 1.0 - (gt_depth > 0).float()


                # Chamfer supervision: hard-mask predicted no-return rays at
                # 0.5, discard target-sensor ranges >=80 m, and back-project
                # the remaining median depths. The hard masks are detached
                # inside the helper, while gradients still flow to retained
                # depths.
                render_points = lidar4d_range_image_to_points(
                    render_pkg["depth_median"],
                    gt_cam.vfov,
                    gt_cam.hfov,
                    row_to_theta=gt_cam.row_to_theta,
                    raydrop=render_pkg["raydrop"],
                )

                # 픽셀 기반 맵 모으기
                b_depths.append(render_pkg["depth"])
                b_depths_median.append(render_pkg["depth_median"])
                b_intensity_shs.append(render_pkg["intensity_sh"])
                b_raydrops.append(render_pkg["raydrop"])
                if b_alphas is not None:
                    b_alphas.append(render_pkg["alpha"])

                b_gt_depths.append(gt_depth)
                b_gt_intensity_shs.append(gt_intensity_sh)
                b_gt_raydrops.append(gt_raydrop)

                # 포인트 데이터 모으기 (개수가 다를 수 있으므로 리스트 유지)
                # gt도 pred와 동일한 LiDAR4D range/mapping 규칙으로 복원한다.
                # raw gt_cam.points(z-up 센서)는 pano(y-up view) pred와 프레임이 달라 chamfer 폭증 →
                # round-trip 검증상 pano는 rasterizer를 정확히 역변환하므로 양쪽을 pano로 맞춘다.
                b_render_points.append(render_points)
                b_gt_points.append(
                    lidar4d_range_image_to_points(
                        gt_depth,
                        gt_cam.vfov,
                        gt_cam.hfov,
                        row_to_theta=gt_cam.row_to_theta,
                    )
                )

            # 카메라 차원 stack
            all_depths.append(torch.stack(b_depths, dim=0))
            all_depths_median.append(torch.stack(b_depths_median, dim=0))
            all_intensity_shs.append(torch.stack(b_intensity_shs, dim=0))
            all_raydrops.append(torch.stack(b_raydrops, dim=0))
            if all_alphas is not None:
                all_alphas.append(torch.stack(b_alphas, dim=0))
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
            "depth_median": torch.stack(all_depths_median, dim=0),
            "intensity_sh": torch.stack(all_intensity_shs, dim=0),
            "raydrop": torch.stack(all_raydrops, dim=0),
            **({"alpha": torch.stack(all_alphas, dim=0)}
               if all_alphas is not None else {}),
            "gt_depth": torch.stack(all_gt_depths, dim=0),
            "gt_intensity_sh": torch.stack(all_gt_intensity_shs, dim=0),
            "gt_raydrop": torch.stack(all_gt_raydrops, dim=0),
            
            # 가변 크기 가능 리스트 (중첩 리스트 형태: [Batch][Cam])
            "render_points": all_render_points,
            "gt_points": all_gt_points
        }


class DynamicGausRender(GausRender):
    """Render linearly transported Dynamic 2D Gaussians in physical seconds."""

    @staticmethod
    def render_timestamp(camera):
        if getattr(camera, "timestamp_sec", None) is None:
            raise AttributeError(
                "Dynamic rendering requires Camera.timestamp_sec from the dataloader"
            )
        return camera.timestamp_sec

    def get_means3D(self, b_gs, t):
        position = b_gs["position"]
        velocity = b_gs.get("velocity")
        source_time = b_gs.get("source_time_sec")
        if velocity is None or source_time is None:
            raise KeyError(
                "Dynamic Gaussian render input needs velocity and source_time_sec"
            )
        if velocity.shape != position.shape:
            raise ValueError("Dynamic velocity must align with Gaussian positions")
        if source_time.shape != (position.shape[0],):
            raise ValueError("source_time_sec must provide one value per Gaussian")
        target_time = torch.as_tensor(
            t, device=position.device, dtype=position.dtype
        )
        delta_t = target_time - source_time.to(
            device=position.device, dtype=position.dtype
        )
        return position + velocity * delta_t.unsqueeze(-1)

    def get_rotations(self, b_gs, t):
        # The first Dynamic 2DGS representation transports only position.
        return b_gs["rotation"]


__all__ = ["GausRender", "DynamicGausRender"]
