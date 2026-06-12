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

            frame_bboxes = b_gs["frame_bboxes"]   # List[dict]
            fg_masks     = b_gs["fg_masks"]        # {box_id: (Nb,) bool}
            position     = b_gs["position"]        # (Nb, 3) frame_0 좌표계
            ts           = timestamps[b]           # (V,) 0~1
            pose_b       = x["pose"][b]            # (V, 4, 4)
            bbox_f0      = frame_bboxes[0]["bbox"] # (B_f0, 7) sensor frame
            bbox_fn      = frame_bboxes[-1]["bbox"] # (B_fn, 7) sensor frame
            device       = position.device

            # velocity: (Nb, 3) 기본 0 (bg는 정지)
            velocity = torch.zeros_like(position)

            for box_id, fg_mask in fg_masks.items():
                if fg_mask.sum() == 0:
                    continue
                if box_id >= bbox_f0.shape[0] or box_id >= bbox_fn.shape[0]:
                    continue


                center_f0 = apply_pose(
                    bbox_f0[box_id, :3].unsqueeze(0),
                    pose_b[0].to(device)
                ).squeeze(0)  # (3,)

                center_fn = apply_pose(
                    bbox_fn[box_id, :3].unsqueeze(0),
                    pose_b[-1].to(device)
                ).squeeze(0)  # (3,)

                # t=0 ~ t=1 사이의 velocity
                # timestamps[-1] - timestamps[0] = 1 (normalize되어 있으므로)
                v = center_fn - center_f0   # (3,) per unit time

                velocity[fg_mask] = v

            b_result = {**b_gs}
            b_result["velocity"] = velocity   # (Nb, 3)
            # rendering 시: position + velocity * t
            results.append(b_result)

        return results