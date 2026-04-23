"""
Hybrid radius-k k-NN with chunked GPU computation.

Instance separation is handled by the caller: pass separate (points, candidates)
pairs per instance / static pool (Q11 — cross-instance k-NN is forbidden).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Default radius function
# ---------------------------------------------------------------------------

def default_r_max(p: Tensor) -> Tensor:
    """
    Adaptive radius: clip(0.3 + 0.01 * ||p||, 0.3, 2.0) per point.

    Args:
        p: [N, 3] point positions.

    Returns:
        [N] per-point radius upper bound.
    """
    dist = p.norm(dim=-1)  # [N]
    return (0.3 + 0.01 * dist).clamp(0.3, 2.0)


# ---------------------------------------------------------------------------
# Core k-NN primitive
# ---------------------------------------------------------------------------

def hybrid_radius_knn(
    points: Tensor,
    candidates: Tensor,
    k_target: int = 16,
    r_max_fn: Optional[Callable[[Tensor], Tensor]] = None,
    k_min: int = 8,
    chunk_size: int = 1024,
) -> dict:
    """
    Hybrid radius-k k-NN search (chunked pairwise, no CUDA kernels needed).

    For each query point in `points`, finds up to `k_target` nearest neighbors
    from `candidates` that lie within per-point radius `r_max(p)`.

    Instance separation (Q11) is the caller's responsibility: pass only same-
    instance / same-context candidates.

    Args:
        points:     [N, 3]  query positions.
        candidates: [M, 3]  search pool positions.
        k_target:   maximum number of neighbors to return.
        r_max_fn:   callable [N,3]->[N] giving per-point radius cap.
                    Defaults to default_r_max.
        k_min:      not used for filtering here; caller uses k_eff to check.
        chunk_size: number of query points processed per GPU chunk to avoid OOM.

    Returns dict with:
        'idx':   [N, k_target] long  — neighbor indices into `candidates`
                                       (padded with -1 if k_eff < k_target)
        'dist':  [N, k_target] float — distances (padded with inf)
        'k_eff': [N] long            — effective neighbor count (≤ k_target)
        'mask':  [N, k_target] bool  — True for valid entries
    """
    if r_max_fn is None:
        r_max_fn = default_r_max

    N = points.shape[0]
    M = candidates.shape[0]
    device = points.device
    dtype = points.dtype

    r_max = r_max_fn(points)  # [N]

    # Output buffers — initialised as "no neighbor"
    out_idx = torch.full((N, k_target), -1, dtype=torch.long, device=device)
    out_dist = torch.full((N, k_target), float("inf"), dtype=dtype, device=device)

    n_chunks = (N + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, N)
        q_chunk = points[start:end]          # [C, 3]
        r_chunk = r_max[start:end]           # [C]
        C = q_chunk.shape[0]

        # Pairwise distances: [C, M]
        # Use squared distances for efficiency, take sqrt only for top-k
        diff = q_chunk.unsqueeze(1) - candidates.unsqueeze(0)  # [C, M, 3]
        sq_dist = (diff * diff).sum(dim=-1)                    # [C, M]

        # Radius mask
        r_sq = (r_chunk ** 2).unsqueeze(1)   # [C, 1]
        in_radius = sq_dist <= r_sq          # [C, M]  bool

        # Replace out-of-radius with inf so top-k picks in-radius first
        sq_dist_masked = sq_dist.clone()
        sq_dist_masked[~in_radius] = float("inf")

        # Take top-k_target smallest (nearest) per query
        k_actual = min(k_target, M)
        topk_sq, topk_idx = torch.topk(sq_dist_masked, k=k_actual, dim=1, largest=False)
        # topk_sq: [C, k_actual], topk_idx: [C, k_actual]

        topk_dist = topk_sq.sqrt()

        if k_actual < k_target:
            # Pad to k_target
            pad_d = torch.full(
                (C, k_target - k_actual), float("inf"), dtype=dtype, device=device
            )
            pad_i = torch.full(
                (C, k_target - k_actual), -1, dtype=torch.long, device=device
            )
            topk_dist = torch.cat([topk_dist, pad_d], dim=1)
            topk_idx = torch.cat([topk_idx, pad_i], dim=1)

        out_dist[start:end] = topk_dist
        out_idx[start:end] = topk_idx

    # Valid entry = index is non-negative AND distance is finite (within radius)
    mask = (out_idx >= 0) & out_dist.isfinite()   # [N, k_target]
    k_eff = mask.long().sum(dim=1)                # [N]

    # For invalid entries, also set idx to -1 so gather_neighbors works cleanly
    out_idx = out_idx.masked_fill(~mask, -1)

    return {
        "idx": out_idx,
        "dist": out_dist,
        "k_eff": k_eff,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Neighbor gathering utility
# ---------------------------------------------------------------------------

def gather_neighbors(candidates: Tensor, idx: Tensor, mask: Tensor) -> Tensor:
    """
    Gather neighbor xyz coordinates given neighbor indices.

    Padded entries (idx == -1) are filled with the *zero vector* (origin).
    The caller is responsible for centering relative to the query point if
    needed — padding at origin contributes zero offset after centering.

    Args:
        candidates: [M, 3]
        idx:        [N, K] long  (padded entries = -1)
        mask:       [N, K] bool  (True = valid)

    Returns:
        [N, K, 3]  neighbor xyz; invalid slots = (0, 0, 0).
    """
    N, K = idx.shape
    device = candidates.device

    # Replace -1 with 0 for safe indexing
    safe_idx = idx.clamp(min=0)           # [N, K]
    gathered = candidates[safe_idx]       # [N, K, 3]

    # Zero out invalid entries
    gathered = gathered * mask.unsqueeze(-1).to(gathered.dtype)

    return gathered
