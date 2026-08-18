"""Detached, fixed-size statistics for learned grid/spherical count routing."""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.distributed as dist


def routing_sufficient_statistics(
    routing: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Summarize learned-count routing without retaining its autograd graph.

    Only the global K distribution is produced, from ``k_logits``/``selected_k``.
    Both anchor modes keep the same small W&B surface.
    """
    with torch.no_grad():
        logits = routing["k_logits"].detach().float()
        selected_k = routing["selected_k"].detach().long()

        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("routing k_logits must have shape (N, K) with K >= 2")
        token_count, k_max = logits.shape
        expected_shape = (token_count,)
        if tuple(selected_k.shape) != expected_shape:
            raise ValueError("routing selected_k must align with k_logits")

        dtype = logits.dtype
        argmax_k = logits.argmax(dim=-1) + 1
        selected_index = selected_k - 1
        if token_count > 0 and (
            (selected_index < 0).any() or (selected_index >= k_max).any()
        ):
            raise ValueError("routing selected_k contains an invalid count")

        selected_counts = torch.bincount(
            selected_index, minlength=k_max
        ).to(dtype)
        argmax_counts = torch.bincount(
            argmax_k - 1, minlength=k_max
        ).to(dtype)
        statistics = {
            # Keep the historical key for compatibility; in spherical mode
            # the counted units are labelled-cell anchors rather than tokens.
            "token_count": logits.new_tensor(float(token_count)),
            "selected_counts": selected_counts,
            "argmax_counts": argmax_counts,
            "sampled_k_sum": selected_k.to(dtype).sum(),
            "argmax_k_sum": argmax_k.to(dtype).sum(),
        }
        return statistics


def distributed_sum_statistics(
    statistics: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Sum a statistics mapping across DDP ranks with one collective."""
    keys = tuple(statistics.keys())
    shapes = {key: statistics[key].shape for key in keys}
    sizes = {key: statistics[key].numel() for key in keys}
    flat = torch.cat([statistics[key].reshape(-1) for key in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)

    reduced = {}
    start = 0
    for key in keys:
        end = start + sizes[key]
        reduced[key] = flat[start:end].reshape(shapes[key])
        start = end
    return reduced


__all__ = [
    "distributed_sum_statistics",
    "routing_sufficient_statistics",
]
