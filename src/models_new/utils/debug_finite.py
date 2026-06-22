"""Lightweight non-finite (NaN/Inf) tracking helpers for training diagnosis.

Used to localize the gstep~30649 collapse (see scripts/nan_collapse_diagnosis.md):
distinguishes a forward-side blowup (e.g. unbounded exp scale -> Inf) from a
rasterizer-backward NaN, and reports the first offending tensor.
"""
from __future__ import annotations

import torch


def tensor_report(t) -> str:
    """One-line absmax/NaN/Inf summary of a tensor (finite-aware)."""
    if not torch.is_tensor(t):
        return "n/a"
    tf = t if t.is_floating_point() else t.float()
    n_nan = int(torch.isnan(tf).sum())
    n_inf = int(torch.isinf(tf).sum())
    fin = torch.isfinite(tf)
    fmax = float(tf[fin].abs().max()) if bool(fin.any()) else float("nan")
    return f"absmax={fmax:.4g} nan={n_nan} inf={n_inf} numel={tf.numel()}"


def is_finite(t) -> bool:
    return torch.is_tensor(t) and bool(torch.isfinite(t).all())


def first_nonfinite(named):
    """named: iterable of (name, tensor). Returns (name, tensor) of first
    non-finite tensor, else (None, None). Non-tensors are skipped."""
    for name, t in named:
        if torch.is_tensor(t) and not bool(torch.isfinite(t).all()):
            return name, t
    return None, None


def gaussian_health_stats(batch_gaussians):
    """Health metrics that plain absmax misses:
      - scaling_logit_med / _max : median & max of |raw scale logit| (exp overflows
        ~88; median≪max => only outlier gaussians are degenerate, not the whole set).
      - rot_quat_norm_min : MIN raw-quaternion norm. F.normalize's gradient blows up
        as ||q||->0 (0/0), NOT when ||q|| is large -> this is the correct rotation
        risk signal (absmax is the wrong one).
    """
    scal_abs = []
    quat_min_norm = float("inf")
    for b_gs in batch_gaussians:
        if not isinstance(b_gs, dict):
            continue
        s = b_gs.get("scaling")
        if torch.is_tensor(s) and s.numel():
            fin = torch.isfinite(s)
            if bool(fin.any()):
                scal_abs.append(s[fin].abs().reshape(-1))
        q = b_gs.get("rotation")
        if torch.is_tensor(q) and q.numel():
            n = q.norm(dim=-1)
            nf = n[torch.isfinite(n)]
            if nf.numel():
                quat_min_norm = min(quat_min_norm, float(nf.min()))
    out = {}
    if scal_abs:
        alls = torch.cat(scal_abs)
        out["scaling_logit_med"] = float(alls.median())
        out["scaling_logit_max"] = float(alls.max())
    if quat_min_norm != float("inf"):
        out["rot_quat_norm_min"] = quat_min_norm
    return out


def gaussian_param_absmax(batch_gaussians, keys=("scaling", "opacity", "rotation", "position", "shs")):
    """Finite absmax per raw gaussian parameter across a batch list of b_gs dicts.

    `scaling` here is the *raw logit* (pre-exp); exp overflows fp32 around 88.
    """
    out = {}
    for k in keys:
        mx = 0.0
        found = False
        for b_gs in batch_gaussians:
            if not isinstance(b_gs, dict):
                continue
            v = b_gs.get(k)
            if torch.is_tensor(v) and v.numel():
                fin = torch.isfinite(v)
                if bool(fin.any()):
                    mx = max(mx, float(v[fin].abs().max()))
                    found = True
        if found:
            out[k] = mx
    return out
