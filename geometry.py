"""Stage 0-1: Preprocessing, GPU batched PCA, and normal estimation."""

import torch

from .config import ClusteringConfig


def _knn_points(src: torch.Tensor, dst: torch.Tensor, k: int):
    """GPU KNN via pytorch3d, with torch_cluster and brute-force fallbacks.

    Args:
        src: [B, N, 3] query points
        dst: [B, M, 3] reference points
        k: number of neighbors

    Returns:
        dists: [B, N, K] squared distances
        idx: [B, N, K] neighbor indices into dst
    """
    try:
        from pytorch3d.ops import knn_points
        result = knn_points(src, dst, K=k)
        return result.dists, result.idx
    except ImportError:
        pass

    try:
        from torch_cluster import knn
        # torch_cluster works on [N, 3] unbatched
        s = src.squeeze(0)  # [N, 3]
        d = dst.squeeze(0)  # [M, 3]
        edge_index = knn(d, s, k=k)
        # edge_index[0] = query indices (repeated k times), edge_index[1] = ref indices
        idx = edge_index[1].reshape(-1, k).unsqueeze(0)  # [1, N, K]
        neighbors = d[idx.squeeze(0)]  # [N, K, 3]
        dists = (s.unsqueeze(1) - neighbors).pow(2).sum(-1).unsqueeze(0)
        return dists, idx
    except ImportError:
        pass

    # Brute-force fallback (memory-heavy for large N, but works)
    # Process in chunks to avoid OOM
    s = src.squeeze(0)  # [N, 3]
    d = dst.squeeze(0)  # [M, 3]
    N = s.shape[0]
    chunk_size = 4096
    idx_list = []
    dist_list = []
    for i in range(0, N, chunk_size):
        chunk = s[i:i + chunk_size]  # [C, 3]
        dmat = torch.cdist(chunk, d)  # [C, M]
        topk = dmat.topk(k, dim=1, largest=False)
        dist_list.append(topk.values.pow(2))
        idx_list.append(topk.indices)
    dists = torch.cat(dist_list, dim=0).unsqueeze(0)  # [1, N, K]
    idx = torch.cat(idx_list, dim=0).unsqueeze(0)      # [1, N, K]
    return dists, idx


def preprocess(xyz: torch.Tensor, cfg: ClusteringConfig) -> torch.Tensor:
    """Stage 0: Ego-vehicle mask + statistical outlier removal.

    Args:
        xyz: [N, 3] raw point cloud
        cfg: configuration

    Returns:
        mask: [N] boolean mask of inlier points
    """
    N = xyz.shape[0]
    mask = torch.ones(N, dtype=torch.bool, device=xyz.device)

    # Ego-vehicle mask
    dist_to_origin = torch.norm(xyz, dim=1)
    mask &= dist_to_origin > cfg.ego_radius

    # Statistical outlier removal via KNN
    xyz_masked = xyz[mask]
    dists, _ = _knn_points(
        xyz_masked.unsqueeze(0), xyz_masked.unsqueeze(0), k=cfg.sor_k + 1
    )
    # Exclude self (index 0), take mean of k neighbors
    mean_dists = dists.squeeze(0)[:, 1:].mean(dim=1).sqrt()  # [M]
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    sor_mask = mean_dists < (global_mean + cfg.sor_std_mul * global_std)

    # Apply SOR mask back to full mask
    inlier_indices = mask.nonzero(as_tuple=True)[0]
    mask[inlier_indices[~sor_mask]] = False

    return mask


def estimate_local_geometry(xyz: torch.Tensor, cfg: ClusteringConfig) -> dict:
    """Stage 1: GPU batched PCA for normals, curvature, planarity.

    Args:
        xyz: [N, 3] preprocessed point cloud
        cfg: configuration

    Returns:
        dict with keys: normals [N,3], tangent_u [N,3], tangent_v [N,3],
                         curvature [N], planarity [N], knn_idx [N,K]
    """
    N = xyz.shape[0]
    K = cfg.knn_k

    # KNN
    dists, indices = _knn_points(xyz.unsqueeze(0), xyz.unsqueeze(0), k=K)
    indices = indices.squeeze(0)  # [N, K]

    # Batched covariance
    neighbors = xyz[indices]  # [N, K, 3]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)  # [N, K, 3]
    cov = torch.bmm(centered.transpose(1, 2), centered) / (K - 1)  # [N, 3, 3]

    # Batched SVD
    U, S, Vh = torch.linalg.svd(cov)  # S: [N, 3] descending, Vh: [N, 3, 3]

    # Extract features
    normals = Vh[:, 2, :]     # [N, 3] — smallest eigenvalue direction
    tangent_u = Vh[:, 0, :]   # [N, 3]
    tangent_v = Vh[:, 1, :]   # [N, 3]

    S_sum = S.sum(dim=1).clamp(min=1e-8)
    S0 = S[:, 0].clamp(min=1e-8)
    curvature = S[:, 2] / S_sum              # [N]
    planarity = (S[:, 1] - S[:, 2]) / S0     # [N]

    # Orient normals toward sensor (origin = [0,0,0])
    flip_mask = (normals * (-xyz)).sum(dim=1) < 0
    normals[flip_mask] *= -1
    # Flip tangent frame to maintain right-handedness
    tangent_v[flip_mask] *= -1

    return {
        "normals": normals,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "curvature": curvature,
        "planarity": planarity,
        "knn_idx": indices,
    }
