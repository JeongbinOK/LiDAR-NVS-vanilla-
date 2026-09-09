"""Validation for the physical-time Gaussian trajectory payload."""
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicGausTemp(nn.Module):
    """Validate and expose the physical-time Gaussian trajectory contract."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, timestamps_sec=None, window_duration_sec=None):
        batch_gaussians = x.get("batch_gaussians", x.get("gaussians"))
        if batch_gaussians is None:
            raise KeyError("DynamicGausTemp expected batch_gaussians")
        results = []
        for batch_id, item in enumerate(batch_gaussians):
            if item is None:
                results.append(None)
                continue
            for key in ("position", "velocity", "source_time_sec"):
                if key not in item:
                    raise KeyError(f"Dynamic Gaussian batch is missing {key!r}")
            if item["velocity"].shape != item["position"].shape:
                raise ValueError("velocity must align one-to-one with Gaussian position")
            result = {**item}
            if timestamps_sec is not None:
                result["segment_timestamps_sec"] = timestamps_sec[batch_id].to(
                    device=item["position"].device,
                    dtype=item["position"].dtype,
                )
            if window_duration_sec is not None:
                result["segment_duration_sec"] = torch.as_tensor(
                    window_duration_sec[batch_id],
                    device=item["position"].device,
                    dtype=item["position"].dtype,
                )
            results.append(result)
        return results


__all__ = ["DynamicGausTemp"]
