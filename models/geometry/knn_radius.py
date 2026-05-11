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
    method: str = "bruteforce",
    voxel_size: float | None = None,
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
        method: "bruteforce" for the original all-candidate search,
                "voxel_chunk" for GPU chunked search over raw points in the
                union of nearby spatial voxels, or "voxel_local" for an exact
                CPU per-query reference implementation.
        voxel_size: optional cartesian spatial-hash cell size for voxel methods.
                    Defaults to median r_max.

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
    if method == "voxel_chunk":
        return _hybrid_radius_knn_voxel_chunk(
            points=points,
            candidates=candidates,
            r_max=r_max,
            k_target=k_target,
            chunk_size=chunk_size,
            voxel_size=voxel_size,
        )
    if method == "voxel_local":
        return _hybrid_radius_knn_voxel_local(
            points=points,
            candidates=candidates,
            r_max=r_max,
            k_target=k_target,
            voxel_size=voxel_size,
        )
    if method != "bruteforce":
        raise ValueError(f"unsupported k-NN method {method!r}")

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


def _default_spatial_cell_size(r_max: Tensor, voxel_size: float | None) -> float:
    if voxel_size is not None:
        return max(float(voxel_size), 1e-6)
    r_cpu = r_max.detach().cpu().float()
    positive = r_cpu[r_cpu > 0]
    median_r = float(positive.median().item()) if positive.numel() else 1.0
    return max(0.5 * median_r, 0.25)


def _hybrid_radius_knn_voxel_chunk(
    *,
    points: Tensor,
    candidates: Tensor,
    r_max: Tensor,
    k_target: int,
    chunk_size: int,
    voxel_size: float | None,
) -> dict:
    """GPU radius k-NN over raw candidate points from nearby spatial voxels."""
    N = points.shape[0]
    M = candidates.shape[0]
    device = points.device
    dtype = points.dtype
    out_idx = torch.full((N, k_target), -1, dtype=torch.long, device=device)
    out_dist = torch.full((N, k_target), float("inf"), dtype=dtype, device=device)
    if N == 0 or M == 0:
        mask = torch.zeros((N, k_target), dtype=torch.bool, device=device)
        return {
            "idx": out_idx,
            "dist": out_dist,
            "k_eff": torch.zeros((N,), dtype=torch.long, device=device),
            "mask": mask,
        }

    cell_size = _default_spatial_cell_size(r_max, voxel_size)
    point_cell = torch.floor(points / cell_size).to(torch.long)
    cand_cell = torch.floor(candidates / cell_size).to(torch.long)

    all_cell = torch.cat([point_cell, cand_cell], dim=0)
    min_cell = all_cell.min(dim=0).values
    max_cell = all_cell.max(dim=0).values
    extent = (max_cell - min_cell + 1).clamp(min=1)
    point_off = point_cell - min_cell
    stride_z = torch.ones((), dtype=torch.long, device=device)
    stride_y = extent[2]
    stride_x = extent[1] * extent[2]
    point_hash = point_off[:, 0] * stride_x + point_off[:, 1] * stride_y + point_off[:, 2] * stride_z
    query_order = torch.argsort(point_hash)

    n_chunks = (N + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, N)
        q_idx = query_order[start:end]
        q_chunk = points.index_select(0, q_idx)
        r_chunk = r_max.index_select(0, q_idx).clamp(min=0.0)
        if bool((r_chunk <= 0.0).all()):
            continue

        lo = torch.floor((q_chunk - r_chunk.unsqueeze(1)) / cell_size).to(torch.long).min(dim=0).values
        hi = torch.floor((q_chunk + r_chunk.unsqueeze(1)) / cell_size).to(torch.long).max(dim=0).values
        local_mask = ((cand_cell >= lo.unsqueeze(0)) & (cand_cell <= hi.unsqueeze(0))).all(dim=1)
        local_ids = local_mask.nonzero(as_tuple=False).squeeze(-1)
        if local_ids.numel() == 0:
            continue

        local_candidates = candidates.index_select(0, local_ids)
        diff = q_chunk.unsqueeze(1) - local_candidates.unsqueeze(0)
        sq_dist = (diff * diff).sum(dim=-1)
        sq_dist = sq_dist.masked_fill(sq_dist > r_chunk.square().unsqueeze(1), float("inf"))
        k_actual = min(k_target, int(local_ids.numel()))
        topk_sq, topk_local = torch.topk(sq_dist, k=k_actual, dim=1, largest=False)
        gathered_idx = local_ids[topk_local]
        valid = topk_sq.isfinite()
        gathered_idx = gathered_idx.masked_fill(~valid, -1)
        out_idx[q_idx, :k_actual] = gathered_idx
        out_dist[q_idx, :k_actual] = topk_sq.sqrt()

    mask = (out_idx >= 0) & out_dist.isfinite()
    k_eff = mask.long().sum(dim=1)
    out_idx = out_idx.masked_fill(~mask, -1)
    return {
        "idx": out_idx,
        "dist": out_dist,
        "k_eff": k_eff,
        "mask": mask,
    }


def _hybrid_radius_knn_voxel_local(
    *,
    points: Tensor,
    candidates: Tensor,
    r_max: Tensor,
    k_target: int,
    voxel_size: float | None,
) -> dict:
    """Exact radius k-NN using nearby spatial voxels as raw-point candidate bins.

    The spatial hash only decides which candidate *points* to examine. Distances
    and top-k are still computed against the original points, not voxel centres.
    """
    N = points.shape[0]
    M = candidates.shape[0]
    device = points.device
    dtype = points.dtype
    out_idx_cpu = torch.full((N, k_target), -1, dtype=torch.long)
    out_dist_cpu = torch.full((N, k_target), float("inf"), dtype=torch.float32)
    if N == 0 or M == 0:
        out_idx = out_idx_cpu.to(device=device)
        out_dist = out_dist_cpu.to(device=device, dtype=dtype)
        mask = torch.zeros((N, k_target), dtype=torch.bool, device=device)
        return {
            "idx": out_idx,
            "dist": out_dist,
            "k_eff": torch.zeros((N,), dtype=torch.long, device=device),
            "mask": mask,
        }

    points_cpu = points.detach().cpu().float()
    candidates_cpu = candidates.detach().cpu().float()
    r_max_cpu = r_max.detach().cpu().float().clamp(min=0.0)
    if voxel_size is None:
        positive = r_max_cpu[r_max_cpu > 0]
        cell_size = float(positive.median().item()) if positive.numel() else 1.0
    else:
        cell_size = float(voxel_size)
    cell_size = max(cell_size, 1e-6)

    cand_cell = torch.floor(candidates_cpu / cell_size).to(torch.long)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, cell in enumerate(cand_cell.tolist()):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        buckets.setdefault(key, []).append(idx)

    for qi in range(N):
        q = points_cpu[qi]
        radius = float(r_max_cpu[qi].item())
        if radius <= 0.0:
            continue
        lo = torch.floor((q - radius) / cell_size).to(torch.long)
        hi = torch.floor((q + radius) / cell_size).to(torch.long)
        candidate_ids: list[int] = []
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    candidate_ids.extend(buckets.get((ix, iy, iz), ()))
        if not candidate_ids:
            continue

        local_idx = torch.tensor(candidate_ids, dtype=torch.long)
        local_pts = candidates_cpu.index_select(0, local_idx)
        sq_dist = ((local_pts - q.unsqueeze(0)) ** 2).sum(dim=-1)
        valid = sq_dist <= radius * radius
        if not bool(valid.any()):
            continue
        valid_idx = local_idx[valid]
        valid_sq = sq_dist[valid]
        k_actual = min(k_target, int(valid_sq.numel()))
        topk_sq, order = torch.topk(valid_sq, k=k_actual, largest=False)
        out_idx_cpu[qi, :k_actual] = valid_idx[order]
        out_dist_cpu[qi, :k_actual] = topk_sq.sqrt()

    out_idx = out_idx_cpu.to(device=device)
    out_dist = out_dist_cpu.to(device=device, dtype=dtype)
    mask = (out_idx >= 0) & out_dist.isfinite()
    k_eff = mask.long().sum(dim=1)
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
