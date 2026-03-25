"""Stage 5: 2D Gaussian fitting per cluster (batched)."""

import torch

from config import ClusteringConfig


def fit_2d_gaussians(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    assignments: torch.Tensor,
    intensity: torch.Tensor,
    cfg: ClusteringConfig,
) -> dict:
    """Fit a 2D Gaussian surfel to each cluster.

    Args:
        xyz: [N, 3] point positions
        normals: [N, 3] per-point normals
        assignments: [N] cluster ID for each point (values in 0..K-1)
        intensity: [N] per-point intensity
        cfg: configuration

    Returns:
        dict with Gaussian parameters and reparameterization data
    """
    device = xyz.device
    K = assignments.max().item() + 1

    # ---- Cluster centroids via scatter ----
    counts = torch.zeros(K, device=device)
    counts.scatter_add_(0, assignments, torch.ones(xyz.shape[0], device=device))

    mu = torch.zeros(K, 3, device=device)
    mu.scatter_add_(0, assignments.unsqueeze(1).expand(-1, 3), xyz)
    mu /= counts.unsqueeze(1).clamp(min=1)

    # ---- Per-cluster mean intensity ----
    mean_intensity = torch.zeros(K, device=device)
    mean_intensity.scatter_add_(0, assignments, intensity)
    mean_intensity /= counts.clamp(min=1)

    # ---- Per-cluster PCA for tangent frame ----
    centered = xyz - mu[assignments]  # [N, 3]

    cov = torch.zeros(K, 3, 3, device=device)
    outer = centered.unsqueeze(2) * centered.unsqueeze(1)  # [N, 3, 3]
    cov.scatter_add_(
        0, assignments.view(-1, 1, 1).expand(-1, 3, 3), outer
    )
    cov /= (counts.view(-1, 1, 1) - 1).clamp(min=1)

    # SVD per cluster
    U, S, Vh = torch.linalg.svd(cov)  # S: [K, 3], Vh: [K, 3, 3]

    cluster_normals = Vh[:, 2, :]   # [K, 3]
    tangent_u = Vh[:, 0, :]         # [K, 3]
    tangent_v = Vh[:, 1, :]         # [K, 3]

    # Orient normals (toward sensor origin)
    flip = (cluster_normals * (-mu)).sum(dim=1) < 0
    cluster_normals[flip] *= -1
    tangent_v[flip] *= -1

    # ---- Reparameterization ----
    cu = tangent_u[assignments]
    cv = tangent_v[assignments]
    cn = cluster_normals[assignments]

    alphas = (cu * centered).sum(dim=1)
    betas = (cv * centered).sum(dim=1)
    gammas = (cn * centered).sum(dim=1)

    # ---- 2D covariance in tangent plane ----
    ab = torch.stack([alphas, betas], dim=1)  # [N, 2]
    cov_2d = torch.zeros(K, 2, 2, device=device)
    outer_2d = ab.unsqueeze(2) * ab.unsqueeze(1)  # [N, 2, 2]
    cov_2d.scatter_add_(
        0, assignments.view(-1, 1, 1).expand(-1, 2, 2), outer_2d
    )
    cov_2d /= (counts.view(-1, 1, 1) - 1).clamp(min=1)

    # 2D eigenvalues -> scaling
    eig_2d = torch.linalg.eigvalsh(cov_2d)  # [K, 2] ascending
    scaling_2d = eig_2d.clamp(min=1e-8).sqrt().flip(dims=[1])  # [K, 2] descending

    # ---- Tangent frame -> quaternion ----
    rot = torch.stack([tangent_u, tangent_v, cluster_normals], dim=1)  # [K, 3, 3]
    quaternion = _rotation_matrix_to_quaternion(rot)  # [K, 4]

    # ---- Opacity based on cluster quality ----
    gamma_sq = torch.zeros(K, device=device)
    gamma_sq.scatter_add_(0, assignments, gammas.pow(2))
    gamma_rms = (gamma_sq / counts.clamp(min=1)).sqrt()

    opacity = torch.exp(-gamma_rms * 10.0).unsqueeze(1).clamp(min=0.01, max=0.99)

    # ---- Filter small clusters ----
    valid = counts >= cfg.min_cluster_size
    valid_idx = valid.nonzero(as_tuple=True)[0]

    remap = torch.full((K,), -1, dtype=torch.long, device=device)
    remap[valid_idx] = torch.arange(valid_idx.shape[0], device=device)
    new_assignments = remap[assignments]

    point_mask = new_assignments >= 0

    return {
        # Gaussian parameters (valid clusters only)
        "xyz": mu[valid_idx],
        "scaling": scaling_2d[valid_idx],
        "rotation": quaternion[valid_idx],
        "normal": cluster_normals[valid_idx],
        "opacity": opacity[valid_idx],
        "intensity": mean_intensity[valid_idx].unsqueeze(1),
        "tangent_u": tangent_u[valid_idx],
        "tangent_v": tangent_v[valid_idx],
        # Temporal placeholders
        "velocity": torch.zeros(valid_idx.shape[0], 3, device=device),
        "t_center": torch.zeros(valid_idx.shape[0], 1, device=device),
        "scaling_t": torch.ones(valid_idx.shape[0], 1, device=device),
        # Reparameterization
        "point_assignments": new_assignments,
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
        "point_mask": point_mask,
        # Internal
        "_counts": counts[valid_idx],
        "_gamma_rms": gamma_rms[valid_idx],
        "_cov_2d": cov_2d[valid_idx],
    }


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert batched 3x3 rotation matrices to quaternions [w, x, y, z].

    Uses kornia when available for numerical stability,
    with a manual fallback.

    Args:
        R: [B, 3, 3] rotation matrices

    Returns:
        q: [B, 4] quaternions (w, x, y, z)
    """
    try:
        from kornia.geometry.conversions import rotation_matrix_to_quaternion
        # kornia returns (x, y, z, w) — convert to (w, x, y, z)
        q_xyzw = rotation_matrix_to_quaternion(R)
        q = torch.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], dim=1)
        return q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    except ImportError:
        pass

    # Fallback: Shepperd's method
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask1 = trace > 0
    if mask1.any():
        s = (trace[mask1] + 1.0).sqrt() * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = (1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]).clamp(min=0).sqrt() * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s.clamp(min=1e-8)
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s.clamp(min=1e-8)
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s.clamp(min=1e-8)

    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = (1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]).clamp(min=0).sqrt() * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s.clamp(min=1e-8)
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s.clamp(min=1e-8)
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s.clamp(min=1e-8)

    mask4 = ~mask1 & ~mask2 & ~mask3
    if mask4.any():
        s = (1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]).clamp(min=0).sqrt() * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s.clamp(min=1e-8)
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s.clamp(min=1e-8)
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s.clamp(min=1e-8)
        q[mask4, 3] = 0.25 * s

    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return q
