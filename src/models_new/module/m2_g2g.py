import torch.nn as nn


class GausTemp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, timestamps):
        batch_gaussians = x["batch_gaussians"]
        results = []

        for b, b_gs in enumerate(batch_gaussians):
            if b_gs is None:
                results.append(None)
                continue

            frame_bboxes = b_gs["frame_bboxes"]
            fg_masks     = b_gs["fg_masks"]
            position     = b_gs["position"]        # (Nb, 3)
            ts           = timestamps[b]           # (n_input,) 0~1
            pose_b       = x["pose"][b]            # (n_input, 4, 4)
            device       = position.device
            n_input      = len(frame_bboxes)

            # 구간별 velocity: (n_input-1)개, 각 (Nb, 3)
            velocity_segments = []

            for seg in range(n_input - 1):
                vel_seg = torch.zeros_like(position)   # (Nb, 3)

                bbox_fs  = frame_bboxes[seg]["bbox"]       # (B_f, 7) sensor
                bbox_fe  = frame_bboxes[seg + 1]["bbox"]   # (B_f, 7) sensor
                pose_fs  = pose_b[seg].to(device)
                pose_fe  = pose_b[seg + 1].to(device)
                dt       = (ts[seg + 1] - ts[seg]).item()
                dt       = max(dt, 1e-6)

                for box_id, fg_mask in fg_masks.items():
                    if fg_mask.sum() == 0:
                        continue
                    if box_id >= bbox_fs.shape[0] or box_id >= bbox_fe.shape[0]:
                        continue

                    c_start = apply_pose(
                        bbox_fs[box_id, :3].unsqueeze(0), pose_fs
                    ).squeeze(0)   # (3,)

                    c_end = apply_pose(
                        bbox_fe[box_id, :3].unsqueeze(0), pose_fe
                    ).squeeze(0)   # (3,)

                    v = (c_end - c_start) / dt   # (3,)
                    vel_seg[fg_mask] = v

                velocity_segments.append(vel_seg)

            b_result = {**b_gs}
            b_result["velocity_segments"] = velocity_segments  # List[Tensor(Nb, 3)]
            b_result["segment_ts"]        = ts                 # (n_input,)
            results.append(b_result)

        return results