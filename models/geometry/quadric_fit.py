"""
Local 2nd-order surface fitting on k-NN patches.

The geometric init is intentionally "center-fixed": `c_init` is always the
query point, while the fit only seeds local orientation / scale / curvature.
In addition to the original fit-quality signal, this module now reports
principal-curvature and tangent anisotropy diagnostics used by the refactored
geometry-first head.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def _safe_eig_sym(A: Tensor) -> tuple[Tensor, Tensor]:
    """Return ascending eigenvalues / eigenvectors for symmetric matrices."""
    vals, vecs = torch.linalg.eigh(A)
    return vals, vecs


def _build_quadric_design(xy: Tensor) -> Tensor:
    """Build the LS design matrix for `z = ax^2 + by^2 + cxy + dx + ey + f`."""
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    ones = torch.ones_like(x)
    return torch.cat([x * x, y * y, x * y, x, y, ones], dim=-1)


def _solve_ls_normal_eq(Phi: Tensor, z: Tensor, reg: float = 1e-4) -> Tensor:
    """Solve the regularised LS normal equation in batch form."""
    Phi_t = Phi.transpose(-1, -2)
    A = Phi_t @ Phi
    I = torch.eye(6, dtype=Phi.dtype, device=Phi.device).unsqueeze(0)
    A = A + reg * I
    b = Phi_t @ z.unsqueeze(-1)
    return torch.linalg.solve(A, b).squeeze(-1)


def fit_local_quadrics(
    points: Tensor,
    neighbors: Tensor,
    k_eff: Tensor,
    k_min: int = 8,
    k_target: int = 16,
    r_max: float = 2.0,
    r_max_clip: float = 1.0,
    ls_reg: float = 1e-4,
) -> dict:
    """
    Fit local quadrics and derive geometric initialisation tensors.

    Returns:
        'c_init':            [B, N, 3]    query-point anchor.
        'R_init':            [B, N, 3, 3] local frame with canonical tangent order.
        's_init':            [B, N, 3]    (s1, s2, s3), with `s1 >= s2 >= 0`.
        'fit_quality':       [B, N, 4]    [fit_residual, planarity, support, reach].
        'use_geom_init':     [B, N] bool.
        'kappa1_init':       [B, N]       curvature aligned with `s1`.
        'kappa2_init':       [B, N]       curvature aligned with `s2`.
        'tangent_aniso':     [B, N]       `abs(log((s1+eps)/(s2+eps)))`.
        'curvature_aniso':   [B, N]       normalised curvature difference.
    """
    B, N, K, _ = neighbors.shape
    device = points.device
    dtype = points.dtype
    BN = B * N

    p_flat = points.reshape(BN, 3)
    nbr_flat = neighbors.reshape(BN, K, 3)
    k_eff_flat = k_eff.reshape(BN)

    use_geom_flat = k_eff_flat >= k_min

    q_prime = nbr_flat - p_flat.unsqueeze(1)
    k_range = torch.arange(K, device=device).unsqueeze(0)
    valid_mask = k_range < k_eff_flat.unsqueeze(1)
    valid_f = valid_mask.to(dtype=dtype)
    q_prime = q_prime * valid_f.unsqueeze(-1)

    k_eff_safe = k_eff_flat.to(dtype=dtype).clamp(min=1.0)
    cov = torch.bmm(q_prime.transpose(1, 2), q_prime) / k_eff_safe.view(BN, 1, 1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    eig_vals, eig_vecs = _safe_eig_sym(cov)
    lam3 = eig_vals[:, 0]
    lam2 = eig_vals[:, 1]
    lam1 = eig_vals[:, 2]

    # `eigh` returns ascending eigenpairs; flip so columns become [u1, u2, u3]
    # where u3 is the least-variance normal.
    R_pca = eig_vecs.flip(-1)

    u1 = R_pca[:, :, 0]
    u2 = R_pca[:, :, 1]
    u3 = R_pca[:, :, 2]
    u3_cross = torch.cross(u1, u2, dim=1)
    sign_u3 = (u3_cross * u3).sum(dim=1).sign()
    sign_u3 = sign_u3 + (sign_u3 == 0).to(dtype=dtype)
    R_pca = R_pca.clone()
    R_pca[:, :, 2] = u3 * sign_u3.unsqueeze(1)

    n_valid = valid_f.sum(dim=1).clamp(min=1.0)

    q_pca = torch.bmm(q_prime, R_pca)
    z_mean = (q_pca[:, :, 2] * valid_f).sum(dim=1) / n_valid
    flip_normal = z_mean < 0
    flip_sign = torch.where(
        flip_normal,
        torch.full_like(z_mean, -1.0),
        torch.ones_like(z_mean),
    )
    q_pca = q_pca.clone()
    q_pca[:, :, 2] = q_pca[:, :, 2] * flip_sign.unsqueeze(1)
    R_pca = R_pca.clone()
    R_pca[:, :, 2] = R_pca[:, :, 2] * flip_sign.unsqueeze(1)

    xy = q_pca[:, :, :2]
    z_vals = q_pca[:, :, 2]
    z_vals_valid = z_vals * valid_f
    Phi = _build_quadric_design(xy)
    Phi_valid = Phi * valid_f.unsqueeze(-1)
    theta = _solve_ls_normal_eq(Phi_valid, z_vals_valid, reg=ls_reg)

    a_c = theta[:, 0]
    b_c = theta[:, 1]
    c_c = theta[:, 2]

    z_pred = (Phi_valid * theta.unsqueeze(1)).sum(dim=-1)
    residuals = (z_pred - z_vals_valid) ** 2
    mean_res = residuals.sum(dim=1) / n_valid
    z_mean_valid = z_vals_valid.sum(dim=1) / n_valid
    z_var = ((z_vals_valid - z_mean_valid.unsqueeze(1)) ** 2 * valid_f).sum(dim=1) / n_valid
    fit_residual = mean_res / (z_var + _EPS)

    H = torch.stack(
        [
            torch.stack([2.0 * a_c, c_c], dim=-1),
            torch.stack([c_c, 2.0 * b_c], dim=-1),
        ],
        dim=-2,
    )
    H = 0.5 * (H + H.transpose(-1, -2))

    try:
        h_vals, h_vecs = _safe_eig_sym(H)
    except Exception:
        h_vals = torch.zeros(BN, 2, dtype=dtype, device=device)
        h_vecs = torch.eye(2, dtype=dtype, device=device).unsqueeze(0).expand(BN, -1, -1)

    # Enforce a proper 2D rotation so the final 3D frame stays right-handed.
    V = h_vecs.clone()
    neg_det = torch.linalg.det(V) < 0
    if neg_det.any():
        V[neg_det, :, 1] = -V[neg_det, :, 1]

    V_full = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(BN, -1, -1).clone()
    V_full[:, :2, :2] = V
    R_refined = torch.bmm(R_pca, V_full)

    # Recompute tangent variances after the Hessian-aligned tangent rotation.
    e1_raw = R_refined[:, :, 0]
    e2_raw = R_refined[:, :, 1]
    proj1 = (q_prime * e1_raw.unsqueeze(1)).sum(dim=-1)
    proj2 = (q_prime * e2_raw.unsqueeze(1)).sum(dim=-1)
    proj1_valid = proj1 * valid_f
    proj2_valid = proj2 * valid_f
    mean1 = proj1_valid.sum(dim=1) / n_valid
    mean2 = proj2_valid.sum(dim=1) / n_valid
    var1 = ((proj1_valid - mean1.unsqueeze(1)) ** 2 * valid_f).sum(dim=1) / n_valid
    var2 = ((proj2_valid - mean2.unsqueeze(1)) ** 2 * valid_f).sum(dim=1) / n_valid
    s1_raw = var1.clamp(min=0.0).sqrt()
    s2_raw = var2.clamp(min=0.0).sqrt()
    kappa1_raw = h_vals[:, 0]
    kappa2_raw = h_vals[:, 1]

    # Canonical tangent order: always expose the larger tangent extent as s1.
    swap_axes = s1_raw < s2_raw
    e1 = torch.where(swap_axes.unsqueeze(1), e2_raw, e1_raw)
    e2 = torch.where(swap_axes.unsqueeze(1), -e1_raw, e2_raw)
    s1_init = torch.where(swap_axes, s2_raw, s1_raw)
    s2_init = torch.where(swap_axes, s1_raw, s2_raw)
    kappa1_init = torch.where(swap_axes, kappa2_raw, kappa1_raw)
    kappa2_init = torch.where(swap_axes, kappa1_raw, kappa2_raw)

    R_canonical = R_refined.clone()
    R_canonical[:, :, 0] = e1
    R_canonical[:, :, 1] = e2
    neg_det = torch.linalg.det(R_canonical) < 0
    if neg_det.any():
        R_canonical[neg_det, :, 1] = -R_canonical[neg_det, :, 1]

    # Elliptic-paraboloid-only semantics keep s1/s2 positive and store curvature
    # sign in s3.
    s3_init = 0.5 * (kappa1_init + kappa2_init)

    lam_sum = lam1 + lam2 + lam3 + _EPS
    planarity = lam3 / lam_sum
    support = (k_eff_flat.to(dtype=dtype) / float(k_target)).clamp(0.0, 1.0)
    nbr_dists = q_prime.norm(dim=-1)
    mean_dist = (nbr_dists * valid_f).sum(dim=1) / n_valid
    reach = (mean_dist / (r_max + _EPS)).clamp(0.0, 1.0)
    fit_quality = torch.stack([fit_residual, planarity, support, reach], dim=-1)

    tangent_aniso = torch.log((s1_init + _EPS) / (s2_init + _EPS)).abs()
    curvature_aniso = (kappa1_init - kappa2_init).abs() / (
        kappa1_init.abs() + kappa2_init.abs() + _EPS
    )

    c_init = p_flat.clone()
    R_identity = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(BN, -1, -1)
    R_init = torch.where(
        use_geom_flat.view(BN, 1, 1).expand_as(R_canonical),
        R_canonical,
        R_identity.clone(),
    )

    s_stack = torch.stack([s1_init, s2_init, s3_init], dim=-1)
    s_fallback = torch.zeros(BN, 3, dtype=dtype, device=device)
    s_init_out = torch.where(
        use_geom_flat.view(BN, 1).expand_as(s_stack),
        s_stack,
        s_fallback,
    )

    fq_fallback = torch.tensor(
        [r_max_clip, 1.0 / 3.0, 0.0, 1.0],
        dtype=dtype,
        device=device,
    ).unsqueeze(0).expand(BN, -1)
    fit_quality_out = torch.where(
        use_geom_flat.view(BN, 1).expand_as(fit_quality),
        fit_quality,
        fq_fallback.clone(),
    )

    zeros = torch.zeros(BN, dtype=dtype, device=device)
    kappa1_out = torch.where(use_geom_flat, kappa1_init, zeros)
    kappa2_out = torch.where(use_geom_flat, kappa2_init, zeros)
    tangent_aniso_out = torch.where(use_geom_flat, tangent_aniso, zeros)
    curvature_aniso_out = torch.where(use_geom_flat, curvature_aniso, zeros)

    return {
        "c_init": c_init.view(B, N, 3),
        "R_init": R_init.view(B, N, 3, 3),
        "s_init": s_init_out.view(B, N, 3),
        "fit_quality": fit_quality_out.view(B, N, 4),
        "use_geom_init": use_geom_flat.view(B, N),
        "kappa1_init": kappa1_out.view(B, N),
        "kappa2_init": kappa2_out.view(B, N),
        "tangent_aniso": tangent_aniso_out.view(B, N),
        "curvature_aniso": curvature_aniso_out.view(B, N),
    }
