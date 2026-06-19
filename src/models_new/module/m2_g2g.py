import torch
import torch.nn as nn


class GausTemp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, timestamps):
        batch_gaussians = x.get("batch_gaussians", x.get("gaussians"))
        if batch_gaussians is None:
            raise KeyError("GausTemp expected 'batch_gaussians' or 'gaussians' in Point2Gaus output.")
        results = []

        for b, b_gs in enumerate(batch_gaussians):
            if b_gs is None:
                results.append(None)
                continue

            frame_bboxes = b_gs["frame_bboxes"]
            device       = b_gs["position"].device
            ts           = timestamps[b].to(device)           # (n_input,) 0~1
            dynamic_ids = b_gs["instance_id"][b_gs["is_dynamic"]].unique().tolist()
            object_trajectories = {}

            for inst_id in dynamic_ids:
                inst_id = int(inst_id)
                boxes_for_inst = []
                times_for_inst = []
                for frame_idx, frame_box in enumerate(frame_bboxes):
                    iids = frame_box["instance_id"].to(device)
                    matches = (iids == inst_id).nonzero(as_tuple=True)[0]
                    if matches.numel() == 0:
                        continue
                    box_idx = int(matches[0])
                    boxes_for_inst.append(frame_box["bbox_ref"][box_idx].to(device))
                    times_for_inst.append(ts[frame_idx])

                if len(boxes_for_inst) >= 2:
                    object_trajectories[inst_id] = {
                        "timestamps": torch.stack(times_for_inst),
                        "bbox_ref": torch.stack(boxes_for_inst),
                    }

            b_result = {**b_gs}
            b_result["object_trajectories"] = object_trajectories
            b_result["segment_ts"] = ts
            results.append(b_result)

        return results