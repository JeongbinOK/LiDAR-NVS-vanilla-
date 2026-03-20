"""Stage 5: Adaptive split/merge refinement."""

import torch

from .config import ClusteringConfig
from .geometry import _knn_points


def _cluster_stats(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    assignments: torch.Tensor,
    K: int,
) -> dict:
    """Compute per-cluster statistics needed for split/merge decisions."""
    device = xyz.device
    counts = torch.zeros(K, device=device)
    counts.scatter_add_(0, assignments, torch.ones(xyz.shape[0], device=device))

    mu = torch.zeros(K, 3, device=device)
    mu.scatter_add_(0, assignments.unsqueeze(1).expand(-1, 3), xyz)
    mu /= counts.unsqueeze(1).clamp(min=1)

    # Cluster normals (mean of per-point normals, renormalized)
    cluster_normals = torch.zeros(K, 3, device=device)
    cluster_normals.scatter_add_(0, assignments.unsqueeze(1).expand(-1, 3), normals)
    cluster_normals = cluster_normals / cluster_normals.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Gamma RMS
    centered = xyz - mu[assignments]
    cn = cluster_normals[assignments]
    gammas = (cn * centered).sum(dim=1)
    gamma_sq = torch.zeros(K, device=device)
    gamma_sq.scatter_add_(0, assignments, gammas.pow(2))
    gamma_rms = (gamma_sq / counts.clamp(min=1)).sqrt()

    # Condition number (per-cluster PCA)
    cov = torch.zeros(K, 3, 3, device=device)
    outer = centered.unsqueeze(2) * centered.unsqueeze(1)
    cov.scatter_add_(0, assignments.view(-1, 1, 1).expand(-1, 3, 3), outer)
    cov /= (counts.view(-1, 1, 1) - 1).clamp(min=1)

    S = torch.linalg.svdvals(cov)  # [K, 3]
    cond_number = S[:, 0] / S[:, 2].clamp(min=1e-8)

    return {
        "counts": counts,
        "mu": mu,
        "normals": cluster_normals,
        "gamma_rms": gamma_rms,
        "cond_number": cond_number,
    }


def _split_cluster(
    xyz: torch.Tensor,
    mask: torch.Tensor,
    n_iters: int = 10,
) -> torch.Tensor:
    """2-means split of points within a cluster (in 3D).

    Args:
        xyz: [N, 3] all points
        mask: [N] boolean mask for this cluster
        n_iters: k-means iterations

    Returns:
        labels: [M] 0 or 1 for each point in the cluster
    """
    pts = xyz[mask]  # [M, 3]
    M = pts.shape[0]
    if M < 2:
        return torch.zeros(M, dtype=torch.long, device=xyz.device)

    # Initialize: two farthest points
    idx0 = 0
    d = (pts - pts[idx0]).pow(2).sum(dim=1)
    idx1 = d.argmax().item()
    centers = torch.stack([pts[idx0], pts[idx1]])  # [2, 3]

    for _ in range(n_iters):
        dists = torch.cdist(pts, centers)  # [M, 2]
        labels = dists.argmin(dim=1)
        for c in range(2):
            cmask = labels == c
            if cmask.any():
                centers[c] = pts[cmask].mean(dim=0)

    return labels


def refine_clusters(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    assignments: torch.Tensor,
    cfg: ClusteringConfig,
) -> torch.Tensor:
    """Stage 5: Iterative split/merge refinement.

    Even iterations: split
    Odd iterations: merge

    Args:
        xyz: [N, 3]
        normals: [N, 3]
        assignments: [N] cluster IDs (0..K-1)
        cfg: configuration

    Returns:
        assignments: [N] refined cluster IDs (0..K'-1), contiguous
    """
    device = xyz.device
    K = assignments.max().item() + 1
    merge_cooldown = torch.zeros(K, dtype=torch.long, device=device)

    for iteration in range(cfg.max_refine_iters):
        K = assignments.max().item() + 1
        stats = _cluster_stats(xyz, normals, assignments, K)

        if iteration % 2 == 0:
            # ---- SPLIT (even iterations) ----
            split_mask = (
                (stats["gamma_rms"] > cfg.split_gamma_rms)
                | (stats["counts"] > cfg.split_max_size)
                | (stats["cond_number"] > cfg.split_cond_number)
            ) & (stats["counts"] >= cfg.min_cluster_size * 2)

            clusters_to_split = split_mask.nonzero(as_tuple=True)[0]
            next_id = K

            for cid in clusters_to_split:
                cmask = assignments == cid.item()
                if cmask.sum() < cfg.min_cluster_size * 2:
                    continue
                labels = _split_cluster(xyz, cmask)
                # Assign one half to new cluster
                pts_in_cluster = cmask.nonzero(as_tuple=True)[0]
                new_mask = labels == 1
                assignments[pts_in_cluster[new_mask]] = next_id
                next_id += 1

        else:
            # ---- MERGE (odd iterations) ----
            K_current = assignments.max().item() + 1
            stats = _cluster_stats(xyz, normals, assignments, K_current)

            # Extend cooldown if needed
            if merge_cooldown.shape[0] < K_current:
                merge_cooldown = torch.cat([
                    merge_cooldown,
                    torch.zeros(K_current - merge_cooldown.shape[0],
                                dtype=torch.long, device=device)
                ])

            # Find neighboring clusters via centroid KNN
            valid_clusters = (stats["counts"] > 0).nonzero(as_tuple=True)[0]
            if valid_clusters.shape[0] < 2:
                continue

            centroids = stats["mu"][valid_clusters]
            n_neighbors = min(6, valid_clusters.shape[0])
            _, neighbor_ids = _knn_points(
                centroids.unsqueeze(0), centroids.unsqueeze(0), k=n_neighbors
            )
            neighbor_ids = neighbor_ids.squeeze(0)  # [V, n_neighbors]

            merged = set()
            for i_local in range(valid_clusters.shape[0]):
                cid = valid_clusters[i_local].item()
                if cid in merged or merge_cooldown[cid] > 0:
                    continue

                ni = stats["normals"][cid]
                for j_local_idx in range(1, n_neighbors):
                    j_local = neighbor_ids[i_local, j_local_idx].item()
                    cjd = valid_clusters[j_local].item()
                    if cjd in merged or cjd == cid or merge_cooldown[cjd] > 0:
                        continue

                    nj = stats["normals"][cjd]
                    angle = torch.acos((ni * nj).sum().clamp(-1, 1))
                    angle_deg = angle.item() * 180.0 / 3.14159265

                    if angle_deg < cfg.merge_angle_deg:
                        # Check merged gamma_rms
                        merged_mask = (assignments == cid) | (assignments == cjd)
                        merged_pts = xyz[merged_mask]
                        merged_mu = merged_pts.mean(dim=0)
                        merged_n = (ni + nj)
                        merged_n = merged_n / merged_n.norm().clamp(min=1e-8)
                        merged_gamma = ((merged_pts - merged_mu) * merged_n).sum(dim=1)
                        merged_gamma_rms = merged_gamma.pow(2).mean().sqrt()

                        if merged_gamma_rms < cfg.split_gamma_rms:
                            # Merge j into i
                            assignments[assignments == cjd] = cid
                            merged.add(cjd)
                            # Cooldown
                            if cid < merge_cooldown.shape[0]:
                                merge_cooldown[cid] = 1

            # Decrement cooldowns
            merge_cooldown = (merge_cooldown - 1).clamp(min=0)

        # ---- Check convergence ----
        # Recompute and check if < threshold% of points changed
        # (For simplicity, we always run all iterations)

    # ---- Compactify cluster IDs ----
    unique_ids = assignments.unique()
    remap = torch.zeros(unique_ids.max().item() + 1, dtype=torch.long, device=device)
    remap[unique_ids] = torch.arange(unique_ids.shape[0], device=device)
    assignments = remap[assignments]

    return assignments
