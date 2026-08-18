"""Geometry-only pseudo targets for learned Grid count routing.

The input membership is the production occupied-token membership produced by
``aggregate_points_to_cells_with_membership``.  No second Cartesian
``floor(x / cell_size)`` voxelization is performed here.

For every occupied token we fit the inverse-range plane

    1 / r = beta^T q,  q = x / ||x||,

then evaluate each raw point with the exact leave-one-out plane.  The token
score is Q75 of the valid held-out ray-depth errors in metres.  Thresholds map
that score to the ordinal K target.  Low-support or numerically undefined
geometry is conservatively supervised as K=1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class GridRouterPseudoGT:
    """Per-token pseudo-GT fields in compact occupied-token row order."""

    target_k: torch.Tensor
    ray_error_q_m: torch.Tensor
    geometry_valid: torch.Tensor
    raw_count: torch.Tensor
    valid_loo_count: torch.Tensor


def _segment_quantile(
    values: torch.Tensor,
    segment: torch.Tensor,
    segment_count: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Quantile of ``values`` for every integer segment.

    Missing segments are returned as NaN.  Sorting is lexicographic by
    ``(segment, value)`` via two stable passes, avoiding fused numeric keys that
    can lose integer precision.
    """
    num_segments = int(segment_count.numel())
    result = torch.full(
        (num_segments,),
        float("nan"),
        dtype=values.dtype,
        device=values.device,
    )
    if values.numel() == 0:
        return result

    order = torch.argsort(values, stable=True)
    order = order[torch.argsort(segment[order], stable=True)]
    sorted_values = values[order]
    starts = torch.cumsum(segment_count, dim=0) - segment_count
    present = (segment_count > 0).nonzero(as_tuple=True)[0]
    position = probability * (
        segment_count[present].to(values.dtype) - 1.0
    )
    lower = position.floor().to(torch.long)
    upper = position.ceil().to(torch.long)
    weight = position - lower.to(values.dtype)
    low_value = sorted_values[starts[present] + lower]
    high_value = sorted_values[starts[present] + upper]
    result[present] = low_value + weight * (high_value - low_value)
    return result


@torch.no_grad()
def grid_inverse_range_pseudo_gt(
    points_sensor: torch.Tensor,
    token_index: torch.Tensor,
    num_tokens: int,
    *,
    thresholds_m: Sequence[float],
    min_raw_points: int = 3,
    loo_denominator_min: float = 1.0e-3,
    min_valid_loo_count: int = 3,
    min_valid_loo_fraction: float = 0.75,
    residual_quantile: float = 0.75,
) -> GridRouterPseudoGT:
    """Build ordinal K pseudo targets from exact raw-token membership.

    ``thresholds_m`` contains ``K_max - 1`` strictly increasing ray-depth
    thresholds.  A valid token receives

        K = 1 + sum(error_q >= threshold).

    Tokens with fewer than ``min_raw_points`` raw members, singular full fits,
    or insufficient valid LOO members receive K=1 and ``geometry_valid=False``.
    Membership/shape errors are invariants and raise instead of silently
    becoming K=1.
    """
    num_tokens = int(num_tokens)
    min_raw_points = int(min_raw_points)
    min_valid_loo_count = int(min_valid_loo_count)
    loo_denominator_min = float(loo_denominator_min)
    min_valid_loo_fraction = float(min_valid_loo_fraction)
    residual_quantile = float(residual_quantile)
    thresholds = tuple(float(value) for value in thresholds_m)

    if points_sensor.ndim != 2 or points_sensor.shape[1] != 3:
        raise ValueError("points_sensor must have shape (N, 3)")
    if token_index.ndim != 1 or token_index.shape[0] != points_sensor.shape[0]:
        raise ValueError(
            "token_index must be one-dimensional and aligned with points_sensor"
        )
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if min_raw_points < 3:
        raise ValueError("min_raw_points must be at least 3 for a 3D plane")
    if min_valid_loo_count < 1:
        raise ValueError("min_valid_loo_count must be positive")
    if not 0.0 < min_valid_loo_fraction <= 1.0:
        raise ValueError("min_valid_loo_fraction must be in (0, 1]")
    if not 0.0 <= residual_quantile <= 1.0:
        raise ValueError("residual_quantile must be in [0, 1]")
    if not loo_denominator_min > 0.0:
        raise ValueError("loo_denominator_min must be positive")
    if len(thresholds) < 1:
        raise ValueError("thresholds_m must contain at least one threshold")
    if not all(math.isfinite(value) and value >= 0.0 for value in thresholds):
        raise ValueError("thresholds_m must be finite and non-negative")
    if not all(left < right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("thresholds_m must be strictly increasing")

    device = points_sensor.device
    token_index = token_index.to(device=device, dtype=torch.long)
    if token_index.numel() > 0 and (
        bool((token_index < 0).any())
        or bool((token_index >= num_tokens).any())
    ):
        raise ValueError("token_index contains an out-of-range occupied-token row")

    raw_count = torch.zeros(num_tokens, dtype=torch.long, device=device)
    if token_index.numel() > 0:
        raw_count.index_add_(0, token_index, torch.ones_like(token_index))
    target_k = torch.ones(num_tokens, dtype=torch.long, device=device)
    valid_loo_count = torch.zeros_like(raw_count)
    geometry_valid = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    score64 = torch.full(
        (num_tokens,), float("nan"), dtype=torch.float64, device=device
    )
    if num_tokens == 0 or points_sensor.shape[0] == 0:
        return GridRouterPseudoGT(
            target_k=target_k,
            ray_error_q_m=score64.to(dtype=points_sensor.dtype),
            geometry_valid=geometry_valid,
            raw_count=raw_count,
            valid_loo_count=valid_loo_count,
        )
    if bool((raw_count == 0).any()):
        raise RuntimeError(
            "pseudo-GT received an occupied-token row without raw membership"
        )

    points = points_sensor.to(torch.float64)
    radius = torch.linalg.vector_norm(points, dim=-1)
    if not bool(torch.isfinite(points).all()) or not bool(
        (torch.isfinite(radius) & (radius > 0.0)).all()
    ):
        raise ValueError("points_sensor must contain finite non-zero-range points")
    direction = points / radius.unsqueeze(-1)
    inverse_range = radius.reciprocal()

    gram = torch.zeros(
        (num_tokens, 3, 3), dtype=torch.float64, device=device
    )
    gram.index_add_(
        0,
        token_index,
        direction.unsqueeze(-1) * direction.unsqueeze(-2),
    )
    rhs = torch.zeros((num_tokens, 3), dtype=torch.float64, device=device)
    rhs.index_add_(
        0,
        token_index,
        direction * inverse_range.unsqueeze(-1),
    )

    support = raw_count >= min_raw_points
    supported_rows = support.nonzero(as_tuple=True)[0]
    if supported_rows.numel() == 0:
        return GridRouterPseudoGT(
            target_k=target_k,
            ray_error_q_m=score64.to(dtype=points_sensor.dtype),
            geometry_valid=geometry_valid,
            raw_count=raw_count,
            valid_loo_count=valid_loo_count,
        )

    gram_supported = gram[supported_rows]
    inverse_gram, info = torch.linalg.inv_ex(gram_supported)
    eigenvalues = torch.linalg.eigvalsh(gram_supported)
    rank_tolerance = (
        3.0
        * torch.finfo(torch.float64).eps
        * eigenvalues.abs().amax(dim=-1, keepdim=True)
    )
    rank_three = (eigenvalues.abs() > rank_tolerance).sum(dim=-1) == 3
    solve_ok = (
        (info == 0)
        & rank_three
        & torch.isfinite(inverse_gram).all(dim=(-2, -1))
    )
    beta = torch.einsum("aij,aj->ai", inverse_gram, rhs[supported_rows])
    fit_ok = solve_ok & torch.isfinite(beta).all(dim=-1)

    # Remap global token rows to the compact supported solve batch.
    supported_index = torch.full(
        (num_tokens,), -1, dtype=torch.long, device=device
    )
    supported_index[supported_rows] = torch.arange(
        supported_rows.numel(), device=device
    )
    member_supported = support[token_index]
    member_token = token_index[member_supported]
    member_solve = supported_index[member_token]
    member_direction = direction[member_supported]
    member_inverse_range = inverse_range[member_supported]
    member_radius = radius[member_supported]

    inverse_gram_q = torch.einsum(
        "aij,aj->ai", inverse_gram[member_solve], member_direction
    )
    leverage = (member_direction * inverse_gram_q).sum(dim=-1)
    full_residual = member_inverse_range - (
        member_direction * beta[member_solve]
    ).sum(dim=-1)
    denominator = 1.0 - leverage
    denominator_ok = torch.isfinite(denominator) & (
        denominator >= loo_denominator_min
    )
    safe_denominator = torch.where(
        denominator_ok, denominator, torch.ones_like(denominator)
    )
    beta_loo = beta[member_solve] - inverse_gram_q * (
        full_residual / safe_denominator
    ).unsqueeze(-1)
    inverse_range_loo = (member_direction * beta_loo).sum(dim=-1)
    member_valid = (
        fit_ok[member_solve]
        & denominator_ok
        & torch.isfinite(inverse_range_loo)
        & (inverse_range_loo > 0.0)
    )
    safe_inverse_range = torch.where(
        member_valid, inverse_range_loo, torch.ones_like(inverse_range_loo)
    )
    ray_error = (member_radius - safe_inverse_range.reciprocal()).abs()
    member_valid = member_valid & torch.isfinite(ray_error)

    valid_token = member_token[member_valid]
    valid_error = ray_error[member_valid]
    if valid_token.numel() > 0:
        valid_loo_count.index_add_(
            0, valid_token, torch.ones_like(valid_token)
        )
        score64 = _segment_quantile(
            valid_error,
            valid_token,
            valid_loo_count,
            residual_quantile,
        )

    required_fraction = torch.ceil(
        min_valid_loo_fraction * raw_count.to(torch.float64)
    ).to(torch.long)
    required_count = torch.maximum(
        torch.full_like(raw_count, min_valid_loo_count),
        required_fraction,
    )
    full_fit_ok = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    full_fit_ok[supported_rows] = fit_ok
    geometry_valid = (
        support
        & full_fit_ok
        & (valid_loo_count >= required_count)
        & torch.isfinite(score64)
    )

    for threshold in thresholds:
        target_k = target_k + (
            geometry_valid & (score64 >= threshold)
        ).to(torch.long)

    return GridRouterPseudoGT(
        target_k=target_k,
        ray_error_q_m=score64.to(dtype=points_sensor.dtype),
        geometry_valid=geometry_valid,
        raw_count=raw_count,
        valid_loo_count=valid_loo_count,
    )


__all__ = ["GridRouterPseudoGT", "grid_inverse_range_pseudo_gt"]
