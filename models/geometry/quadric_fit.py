"""
Local 2nd-order surface fitting on k-NN patches.

`points` are used as neighbourhood anchors only.  The fitted QGS centre is the
surface point under the unweighted neighbour mean in the local PCA chart.  A
general quadratic is first used to estimate the local tangent frame; the final
initial surface is then fit directly in QGS form, `z = a*x^2 + b*y^2`, against
the neighbouring LiDAR samples in that local frame.
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


def _solve_weighted_ls_normal_eq(
    Phi: Tensor,
    z: Tensor,
    weight: Tensor,
    reg: float = 1e-4,
) -> Tensor:
    """Solve a regularised weighted LS normal equation in batch form."""
    n_params = Phi.shape[-1]
    w_sqrt = weight.clamp(min=0.0).sqrt().unsqueeze(-1)
    Phi_w = Phi * w_sqrt
    z_w = z * w_sqrt.squeeze(-1)
    Phi_t = Phi_w.transpose(-1, -2)
    A = Phi_t @ Phi_w
    I = torch.eye(n_params, dtype=Phi.dtype, device=Phi.device).unsqueeze(0)
    A = A + reg * I
    b = Phi_t @ z_w.unsqueeze(-1)
    return torch.linalg.solve(A, b).squeeze(-1)


def _nonzero_sign(x: Tensor) -> Tensor:
    """Return +/-1 with exact zeros mapped to +1."""
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


def fit_local_quadrics(
    points: Tensor,
    neighbors: Tensor,
    k_eff: Tensor,
    k_min: int = 8,
    k_target: int = 16,
    r_max: float = 2.0,
    r_max_clip: float = 1.0,
    ls_reg: float = 1e-4,
    quadric_gamma: float = 1.0,
    quadric_kappa_max: float = 5.0,
    quadric_eps_lambda: float = 0.01,
    quadric_eps_kappa: float = 1e-3,
    quadric_eps_s: float = 1e-3,
    quadric_eps_s3: float = 1e-4,
) -> dict:
    """
    Fit local quadrics and derive geometric initialisation tensors.

    Returns:
        'c_init':            [B, N, 3]    fitted surface centre near pbar.
        'R_init':            [B, N, 3, 3] local frame with canonical tangent order.
        's_init':            [B, N, 3]    signed (s1, s2) plus positive s3.
        'fit_quality':       [B, N, 4]    [fit_residual, planarity, support, reach].
        'use_geom_init':     [B, N] bool.
        'kappa1_init':       [B, N]       curvature aligned with `s1`.
        'kappa2_init':       [B, N]       curvature aligned with `s2`.
        'tangent_aniso':     [B, N]       `abs(log((|s1|+eps)/(|s2|+eps)))`.
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

    k_range = torch.arange(K, device=device).unsqueeze(0)
    valid_mask = k_range < k_eff_flat.unsqueeze(1)
    valid_f = valid_mask.to(dtype=dtype)

    k_eff_safe = k_eff_flat.to(dtype=dtype).clamp(min=1.0)
    n_valid = valid_f.sum(dim=1).clamp(min=1.0)

    # Unweighted neighbour mean and covariance define the local tangent chart.
    pbar = (nbr_flat * valid_f.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(1)
    q_pbar = (nbr_flat - pbar.unsqueeze(1)) * valid_f.unsqueeze(-1)
    cov = torch.bmm(q_pbar.transpose(1, 2), q_pbar) / k_eff_safe.view(BN, 1, 1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    eig_vals, eig_vecs = _safe_eig_sym(cov)
    lam3 = eig_vals[:, 0]
    lam2 = eig_vals[:, 1]
    lam1 = eig_vals[:, 2]

    # `eigh` returns ascending eigenpairs; flip so columns become [u1, u2, u3]
    # where u3 is the least-variance normal.  Orient u3 deterministically by
    # its dominant world component, then rebuild u2 so the frame is proper.
    R_pca = eig_vecs.flip(-1)
    u1 = R_pca[:, :, 0]
    u3 = R_pca[:, :, 2]
    major_idx = u3.abs().argmax(dim=1, keepdim=True)
    major_val = u3.gather(1, major_idx).squeeze(1)
    u3 = torch.where((major_val < 0).unsqueeze(1), -u3, u3)
    u2 = torch.nn.functional.normalize(torch.cross(u3, u1, dim=1), dim=1, eps=_EPS)
    u1 = torch.nn.functional.normalize(torch.cross(u2, u3, dim=1), dim=1, eps=_EPS)
    R_pca = torch.stack([u1, u2, u3], dim=-1)

    q_pca = torch.bmm(nbr_flat - pbar.unsqueeze(1), R_pca)

    xy = q_pca[:, :, :2]
    z_vals = q_pca[:, :, 2]
    Phi = _build_quadric_design(xy)

    # Weighted LS is the only place where the original query point A acts as an
    # anchor.  PCA/support statistics remain centred on the valid neighbour mean.
    dist_to_anchor = (nbr_flat - p_flat.unsqueeze(1)).norm(dim=-1)
    dist_for_median = torch.where(
        valid_mask,
        dist_to_anchor,
        torch.full_like(dist_to_anchor, float("inf")),
    )
    dist_sorted, _ = dist_for_median.sort(dim=1)
    median_idx = ((k_eff_flat.clamp(min=1) - 1) // 2).long().view(BN, 1)
    sigma = dist_sorted.gather(1, median_idx).squeeze(1)
    sigma = torch.where(torch.isfinite(sigma), sigma, torch.ones_like(sigma))
    sigma = sigma.clamp(min=quadric_eps_lambda)
    weights = torch.exp(-(dist_to_anchor ** 2) / (2.0 * sigma.unsqueeze(1) ** 2))
    weights = weights * valid_f
    theta = _solve_weighted_ls_normal_eq(Phi, z_vals, weights, reg=ls_reg)

    a_c = theta[:, 0]
    b_c = theta[:, 1]
    c_c = theta[:, 2]
    d_c = theta[:, 3]
    e_c = theta[:, 4]
    f_c = theta[:, 5]

    weight_sum = weights.sum(dim=1).clamp(min=_EPS)

    mu = pbar + R_pca[:, :, 2] * f_c.unsqueeze(1)
    normal_local = torch.stack([-d_c, -e_c, torch.ones_like(d_c)], dim=-1)
    normal_local = torch.nn.functional.normalize(normal_local, dim=-1, eps=_EPS)
    normal = torch.bmm(R_pca, normal_local.unsqueeze(-1)).squeeze(-1)
    normal = torch.nn.functional.normalize(normal, dim=-1, eps=_EPS)

    H = torch.stack(
        [
            torch.stack([2.0 * a_c, c_c], dim=-1),
            torch.stack([c_c, 2.0 * b_c], dim=-1),
        ],
        dim=-2,
    )
    H = 0.5 * (H + H.transpose(-1, -2))

    g1 = d_c
    g2 = e_c
    denom = (1.0 + g1 * g1 + g2 * g2).sqrt().clamp(min=_EPS)
    I_first = torch.stack(
        [
            torch.stack([1.0 + g1 * g1, g1 * g2], dim=-1),
            torch.stack([g1 * g2, 1.0 + g2 * g2], dim=-1),
        ],
        dim=-2,
    )
    II_second = H / denom.view(BN, 1, 1)
    i_vals, i_vecs = _safe_eig_sym(I_first)
    I_inv_sqrt = i_vecs @ torch.diag_embed(i_vals.clamp(min=_EPS).rsqrt()) @ i_vecs.transpose(-1, -2)
    shape_sym = I_inv_sqrt @ II_second @ I_inv_sqrt
    shape_sym = 0.5 * (shape_sym + shape_sym.transpose(-1, -2))
    h_vals, h_vecs = _safe_eig_sym(shape_sym)
    h_vals = h_vals.clamp(min=-float(quadric_kappa_max), max=float(quadric_kappa_max))

    param_dirs = I_inv_sqrt @ h_vecs
    v1 = param_dirs[:, :, 0]
    v2 = param_dirs[:, :, 1]
    tangent1_local = torch.stack([v1[:, 0], v1[:, 1], g1 * v1[:, 0] + g2 * v1[:, 1]], dim=-1)
    tangent2_local = torch.stack([v2[:, 0], v2[:, 1], g1 * v2[:, 0] + g2 * v2[:, 1]], dim=-1)
    e1_raw = torch.bmm(R_pca, tangent1_local.unsqueeze(-1)).squeeze(-1)
    e2_raw = torch.bmm(R_pca, tangent2_local.unsqueeze(-1)).squeeze(-1)
    e1_raw = torch.nn.functional.normalize(e1_raw, dim=-1, eps=_EPS)
    e2_raw = torch.nn.functional.normalize(e2_raw, dim=-1, eps=_EPS)

    # Tangent extent of the observed local patch.  These RMS projections measure
    # how far the 16-neighbour patch spreads along each tangent axis; they are
    # used as the preferred size of the Gaussian footprint, not as curvature.
    q_mu = (nbr_flat - mu.unsqueeze(1)) * valid_f.unsqueeze(-1)
    proj1 = (q_mu * e1_raw.unsqueeze(1)).sum(dim=-1)
    proj2 = (q_mu * e2_raw.unsqueeze(1)).sum(dim=-1)
    rms1 = ((proj1 ** 2) * valid_f).sum(dim=1) / n_valid
    rms2 = ((proj2 ** 2) * valid_f).sum(dim=1) / n_valid
    s1_raw_abs = float(quadric_gamma) * rms1.clamp(min=0.0).sqrt()
    s2_raw_abs = float(quadric_gamma) * rms2.clamp(min=0.0).sqrt()
    s1_raw_abs = s1_raw_abs.clamp(min=float(quadric_eps_lambda))
    s2_raw_abs = s2_raw_abs.clamp(min=float(quadric_eps_lambda))
    kappa1_raw = h_vals[:, 0]
    kappa2_raw = h_vals[:, 1]

    # Canonical tangent order: expose the wider tangent extent as |s1|.
    swap_axes = s1_raw_abs < s2_raw_abs
    e1 = torch.where(swap_axes.unsqueeze(1), e2_raw, e1_raw)
    e2 = torch.where(swap_axes.unsqueeze(1), -e1_raw, e2_raw)
    s1_abs = torch.where(swap_axes, s2_raw_abs, s1_raw_abs)
    s2_abs = torch.where(swap_axes, s1_raw_abs, s2_raw_abs)
    kappa1_init = torch.where(swap_axes, kappa2_raw, kappa1_raw)
    kappa2_init = torch.where(swap_axes, kappa1_raw, kappa2_raw)

    R_canonical = torch.stack([e1, e2, normal], dim=-1)
    neg_det = torch.linalg.det(R_canonical) < 0
    if neg_det.any():
        R_canonical[neg_det, :, 1] = -R_canonical[neg_det, :, 1]
        e2 = R_canonical[:, :, 1]

    # Final constrained QGS patch fit.  The general quadratic above only
    # estimates centre/frame; the initial QGS coefficients are fitted directly
    # to neighbour samples in the final local frame:
    #
    #   w ~= a*u^2 + b*v^2
    #
    # where (u, v) are tangent-plane coordinates and w is height along the
    # local normal.  This is the actual local patch surface used for QGS init.
    q_qgs = torch.bmm(nbr_flat - mu.unsqueeze(1), R_canonical)
    u = q_qgs[:, :, 0]
    v = q_qgs[:, :, 1]
    w = q_qgs[:, :, 2]
    Phi_qgs = torch.stack([u * u, v * v], dim=-1)
    qgs_theta = _solve_weighted_ls_normal_eq(Phi_qgs, w, weights, reg=ls_reg)
    qgs_a = qgs_theta[:, 0].clamp(
        min=-0.5 * float(quadric_kappa_max),
        max=0.5 * float(quadric_kappa_max),
    )
    qgs_b = qgs_theta[:, 1].clamp(
        min=-0.5 * float(quadric_kappa_max),
        max=0.5 * float(quadric_kappa_max),
    )

    z_qgs_pred = (Phi_qgs * torch.stack([qgs_a, qgs_b], dim=-1).unsqueeze(1)).sum(dim=-1)
    qgs_residuals = (z_qgs_pred - w) ** 2
    mean_res = (qgs_residuals * weights).sum(dim=1) / weight_sum
    z_mean_valid = (w * weights).sum(dim=1) / weight_sum
    z_var = (((w - z_mean_valid.unsqueeze(1)) ** 2) * weights).sum(dim=1) / weight_sum
    fit_residual = mean_res / (z_var + _EPS)

    kappa1_init = 2.0 * qgs_a
    kappa2_init = 2.0 * qgs_b
    coeff_eps = 0.5 * float(quadric_eps_kappa)
    abs_a = qgs_a.abs()
    abs_b = qgs_b.abs()
    axis1_curved = abs_a >= coeff_eps
    axis2_curved = abs_b >= coeff_eps
    near_flat = ~axis1_curved & ~axis2_curved

    # Renderer convention:
    #
    #   w = s3 * (sign(s1)*u^2/|s1|^2 + sign(s2)*v^2/|s2|^2)
    #
    # so a = s3*sign(s1)/|s1|^2 and b = s3*sign(s2)/|s2|^2.
    # The pair (a, b) alone does not uniquely determine (s1, s2, s3), so choose
    # s3 that keeps |s1| and |s2| close to the observed tangent extents, then
    # derive |s1| and |s2| so the curved axes reproduce the fitted coefficients.
    support_s3_1 = abs_a * s1_abs.clamp(min=float(quadric_eps_s)).square()
    support_s3_2 = abs_b * s2_abs.clamp(min=float(quadric_eps_s)).square()
    both_curved_s3 = torch.sqrt(
        support_s3_1.clamp(min=float(quadric_eps_s3))
        * support_s3_2.clamp(min=float(quadric_eps_s3))
    )
    one_curved_s3 = torch.where(axis1_curved, support_s3_1, support_s3_2)
    s3_init = torch.where(
        near_flat,
        torch.full_like(both_curved_s3, float(quadric_eps_s3)),
        torch.where(
            axis1_curved & axis2_curved,
            both_curved_s3,
            one_curved_s3,
        ).abs().clamp(min=float(quadric_eps_s3)),
    )

    s1_abs = torch.where(
        axis1_curved,
        torch.sqrt(s3_init / abs_a.clamp(min=coeff_eps)),
        s1_abs,
    ).clamp(min=float(quadric_eps_s))
    s2_abs = torch.where(
        axis2_curved,
        torch.sqrt(s3_init / abs_b.clamp(min=coeff_eps)),
        s2_abs,
    ).clamp(min=float(quadric_eps_s))

    sign1 = torch.where(axis1_curved, _nonzero_sign(qgs_a), torch.ones_like(qgs_a))
    sign2 = torch.where(axis2_curved, _nonzero_sign(qgs_b), torch.ones_like(qgs_b))

    # Keep the downstream canonical invariant |s1| >= |s2|.  Swapping tangent
    # axes does not change the represented QGS patch; it only renames u/v.
    swap_scale = s1_abs < s2_abs
    R_swapped = torch.stack(
        [R_canonical[:, :, 1], -R_canonical[:, :, 0], R_canonical[:, :, 2]],
        dim=-1,
    )
    R_canonical = torch.where(
        swap_scale.view(BN, 1, 1).expand_as(R_canonical),
        R_swapped,
        R_canonical,
    )
    s1_abs, s2_abs = (
        torch.where(swap_scale, s2_abs, s1_abs),
        torch.where(swap_scale, s1_abs, s2_abs),
    )
    sign1, sign2 = (
        torch.where(swap_scale, sign2, sign1),
        torch.where(swap_scale, sign1, sign2),
    )
    kappa1_init, kappa2_init = (
        torch.where(swap_scale, kappa2_init, kappa1_init),
        torch.where(swap_scale, kappa1_init, kappa2_init),
    )
    s1_init = sign1 * s1_abs
    s2_init = sign2 * s2_abs

    lam_sum = lam1 + lam2 + lam3 + _EPS
    planarity = lam3 / lam_sum
    support = (k_eff_flat.to(dtype=dtype) / float(k_target)).clamp(0.0, 1.0)
    mean_dist = (dist_to_anchor * valid_f).sum(dim=1) / n_valid
    reach = (mean_dist / (r_max + _EPS)).clamp(0.0, 1.0)
    fit_quality = torch.stack([fit_residual, planarity, support, reach], dim=-1)

    tangent_aniso = torch.log((s1_abs + _EPS) / (s2_abs + _EPS)).abs()
    curvature_aniso = (kappa1_init - kappa2_init).abs() / (
        kappa1_init.abs() + kappa2_init.abs() + _EPS
    )

    c_init = torch.where(use_geom_flat.unsqueeze(1), mu, p_flat)
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
