"""Detached, fixed-size statistics for learned grid-count routing."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist


def _validated_range_edges(range_edges_m: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(value) for value in range_edges_m)
    if not edges or edges[0] != 0.0:
        raise ValueError("routing range_edges_m must start at 0")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("routing range_edges_m must be strictly increasing")
    return edges


def routing_sufficient_statistics(
    routing: Mapping[str, torch.Tensor],
    range_edges_m: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Summarize token routing without retaining or changing its autograd graph.

    ``range_edges_m`` are lower bounds. For example ``[0, 10, 20]`` creates
    ``[0, 10)``, ``[10, 20)``, and ``[20, inf)`` bins.
    """
    edges = _validated_range_edges(range_edges_m)

    with torch.no_grad():
        logits = routing["k_logits"].detach().float()
        selected_k = routing["selected_k"].detach().long()
        token_position_sensor = routing["token_position_sensor"].detach().float()
        is_dynamic = routing["is_dynamic"].detach().bool()

        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("routing k_logits must have shape (N, K) with K >= 2")
        token_count, k_max = logits.shape
        expected_shape = (token_count,)
        if tuple(selected_k.shape) != expected_shape:
            raise ValueError("routing selected_k must align with k_logits")
        if tuple(token_position_sensor.shape) != (token_count, 3):
            raise ValueError(
                "routing token_position_sensor must have shape (N, 3)"
            )
        if tuple(is_dynamic.shape) != expected_shape:
            raise ValueError("routing is_dynamic must align with k_logits")

        device = logits.device
        dtype = logits.dtype
        k_values = torch.arange(
            1, k_max + 1, device=device, dtype=dtype
        )
        policy = torch.softmax(logits, dim=-1)
        expected_k = (policy * k_values).sum(dim=-1)
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
        policy_prob_sums = policy.sum(dim=0)
        entropy = -(
            policy.clamp_min(torch.finfo(dtype).tiny).log() * policy
        ).sum(dim=-1) / math.log(k_max)
        top2_logits = logits.topk(k=2, dim=-1).values
        top1_prob = policy.amax(dim=-1)
        logit_margin = top2_logits[:, 0] - top2_logits[:, 1]

        ranges = token_position_sensor.norm(dim=-1)
        boundaries = torch.tensor(
            edges[1:], device=device, dtype=ranges.dtype
        )
        range_index = torch.bucketize(ranges, boundaries, right=True)
        num_range_bins = len(edges)

        def scatter_sum(values, index, size):
            output = torch.zeros(size, device=device, dtype=dtype)
            if values.numel() > 0:
                output.scatter_add_(0, index, values.to(dtype))
            return output

        ones = torch.ones(token_count, device=device, dtype=dtype)
        range_token_counts = scatter_sum(
            ones, range_index, num_range_bins
        )
        range_sampled_k_sums = scatter_sum(
            selected_k, range_index, num_range_bins
        )
        range_expected_k_sums = scatter_sum(
            expected_k, range_index, num_range_bins
        )
        range_argmax_k_sums = scatter_sum(
            argmax_k, range_index, num_range_bins
        )

        # group 0 = background, group 1 = dynamic foreground.
        group_index = is_dynamic.long()
        group_token_counts = scatter_sum(ones, group_index, 2)
        group_sampled_k_sums = scatter_sum(selected_k, group_index, 2)
        group_expected_k_sums = scatter_sum(expected_k, group_index, 2)
        group_argmax_k_sums = scatter_sum(argmax_k, group_index, 2)
        group_selected_flat = group_index * k_max + selected_index
        group_argmax_flat = group_index * k_max + (argmax_k - 1)
        group_selected_counts = torch.bincount(
            group_selected_flat, minlength=2 * k_max
        ).to(dtype).reshape(2, k_max)
        group_argmax_counts = torch.bincount(
            group_argmax_flat, minlength=2 * k_max
        ).to(dtype).reshape(2, k_max)
        group_policy_prob_sums = torch.zeros(
            2, k_max, device=device, dtype=dtype
        )
        if token_count > 0:
            group_policy_prob_sums.index_add_(0, group_index, policy)

        return {
            "token_count": logits.new_tensor(float(token_count)),
            "selected_counts": selected_counts,
            "argmax_counts": argmax_counts,
            "policy_prob_sums": policy_prob_sums,
            "sampled_k_sum": selected_k.to(dtype).sum(),
            "expected_k_sum": expected_k.sum(),
            "argmax_k_sum": argmax_k.to(dtype).sum(),
            "entropy_sum": entropy.sum(),
            "top1_prob_sum": top1_prob.sum(),
            "logit_margin_sum": logit_margin.sum(),
            "range_token_counts": range_token_counts,
            "range_sampled_k_sums": range_sampled_k_sums,
            "range_expected_k_sums": range_expected_k_sums,
            "range_argmax_k_sums": range_argmax_k_sums,
            "group_token_counts": group_token_counts,
            "group_sampled_k_sums": group_sampled_k_sums,
            "group_expected_k_sums": group_expected_k_sums,
            "group_argmax_k_sums": group_argmax_k_sums,
            "group_selected_counts": group_selected_counts,
            "group_argmax_counts": group_argmax_counts,
            "group_policy_prob_sums": group_policy_prob_sums,
        }


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


def range_bin_labels(range_edges_m: Sequence[float]) -> tuple[str, ...]:
    edges = _validated_range_edges(range_edges_m)
    labels = []
    for index, lower in enumerate(edges):
        if index + 1 < len(edges):
            upper = edges[index + 1]
            labels.append(f"{lower:g}_{upper:g}m")
        else:
            labels.append(f"{lower:g}_infm")
    return tuple(labels)


__all__ = [
    "distributed_sum_statistics",
    "range_bin_labels",
    "routing_sufficient_statistics",
]
