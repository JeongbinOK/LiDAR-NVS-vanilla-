"""Statistical analysis of the predicted 2D-Gaussian sizes.

Each 2D Gaussian's footprint is its 1-sigma disk with semi-axes
(su, sv) = softplus(scaling[:, :2]) in METERS. We summarize the size
distribution and how it relates to geometry -- in particular whether the disks
grow with distance from the ref-frame origin (the typical surfel failure mode:
far / sparse regions get blown-up Gaussians). Produces a JSON report + a 2x2
diagnostic figure and a printable summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except Exception:                                       # pragma: no cover
    _HAVE_SCIPY = False


def collect_window_stats(g2p_model, b_gs, t_target: float) -> dict:
    """Per-Gaussian size/geometry arrays for ONE window at its target time."""
    means = g2p_model.get_means3D(b_gs, t_target).detach().cpu().numpy()   # [N,3]
    sca = b_gs["scaling"].detach()
    su = F.softplus(sca[:, 0]).cpu().numpy()
    sv = F.softplus(sca[:, 1]).cpu().numpy()
    n = su.shape[0]

    is_dyn = b_gs.get("is_dynamic")
    is_dyn = (is_dyn.detach().cpu().numpy().astype(bool)
              if torch.is_tensor(is_dyn) else np.zeros((n,), bool))
    opac = b_gs.get("opacity")
    if torch.is_tensor(opac):
        opac = opac.detach().float().cpu().numpy().reshape(-1)
        if opac.shape[0] != n:
            opac = np.full((n,), np.nan, np.float32)
    else:
        opac = np.full((n,), np.nan, np.float32)

    r_max = np.maximum(su, sv)
    r_min = np.minimum(su, sv)
    return {
        "su": su, "sv": sv,
        "r_geo": np.sqrt(su * sv),               # effective disk radius (m)
        "r_max": r_max,                          # largest semi-axis (m)
        "aspect": r_max / np.maximum(r_min, 1e-9),
        "dist_xy": np.linalg.norm(means[:, :2], axis=1),
        "dist_xyz": np.linalg.norm(means, axis=1),
        "z": means[:, 2],
        "is_dyn": is_dyn,
        "opacity": opac,
    }


def _concat(records: list, key: str) -> np.ndarray:
    return np.concatenate([r[key] for r in records]) if records else np.zeros((0,))


def _pct(a, qs=(50, 90, 95, 99, 99.9)):
    return {f"p{q}": (float(np.percentile(a, q)) if a.size else float("nan")) for q in qs}


def _corr(x, y):
    """Pearson + Spearman; robust to constant/empty input."""
    out = {}
    if x.size < 3 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return {"pearson": float("nan"), "spearman": float("nan")}
    if _HAVE_SCIPY:
        out["pearson"] = float(pearsonr(x, y)[0])
        out["spearman"] = float(spearmanr(x, y)[0])
    else:
        out["pearson"] = float(np.corrcoef(x, y)[0, 1])
        xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
        out["spearman"] = float(np.corrcoef(xr, yr)[0, 1])
    return out


_DIST_BINS = np.array([0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 1e9], np.float64)


def _binned(dist, r):
    rows = []
    for i in range(len(_DIST_BINS) - 1):
        lo, hi = _DIST_BINS[i], _DIST_BINS[i + 1]
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "range_m": [float(lo), (None if hi > 1e8 else float(hi))],
            "count": int(m.sum()),
            "median_r": float(np.median(r[m])),
            "p95_r": float(np.percentile(r[m], 95)),
            "max_r": float(r[m].max()),
        })
    return rows


def _degenerate(stats, thr):
    r = stats["r_max"]
    m = r > thr
    if m.sum() == 0:
        return {"threshold_m": thr, "count": 0, "frac": 0.0}
    return {
        "threshold_m": thr,
        "count": int(m.sum()),
        "frac": float(m.mean()),
        "frac_dynamic": float(stats["is_dyn"][m].mean()),
        "median_dist_xy": float(np.median(stats["dist_xy"][m])),
        "median_abs_z": float(np.median(np.abs(stats["z"][m]))),
        "median_aspect": float(np.median(stats["aspect"][m])),
        "median_opacity": float(np.nanmedian(stats["opacity"][m])),
    }


def analyze_gaussian_sizes(records: list, out_dir, tag: str = "") -> str:
    """Aggregate per-window records -> JSON + figure + printable summary."""
    out_dir = Path(out_dir)
    keys = ["su", "sv", "r_geo", "r_max", "aspect",
            "dist_xy", "dist_xyz", "z", "is_dyn", "opacity"]
    S = {k: _concat(records, k) for k in keys}
    n = S["r_geo"].size
    if n == 0:
        return "[gaussian-stats] no Gaussians collected."

    dyn = S["is_dyn"].astype(bool)
    log_r = np.log10(np.clip(S["r_geo"], 1e-4, None))

    report = {
        "num_windows": len(records),
        "num_gaussians": int(n),
        "frac_dynamic": float(dyn.mean()),
        "size_m": {
            "r_geo": {**_pct(S["r_geo"]), "mean": float(S["r_geo"].mean()),
                      "max": float(S["r_geo"].max())},
            "r_max": {**_pct(S["r_max"]), "mean": float(S["r_max"].mean()),
                      "max": float(S["r_max"].max())},
            "aspect": {**_pct(S["aspect"]), "max": float(S["aspect"].max())},
        },
        "by_kind": {
            "static": {"count": int((~dyn).sum()),
                       "median_r_geo": float(np.median(S["r_geo"][~dyn])) if (~dyn).any() else float("nan"),
                       "p95_r_geo": float(np.percentile(S["r_geo"][~dyn], 95)) if (~dyn).any() else float("nan"),
                       "max_r_geo": float(S["r_geo"][~dyn].max()) if (~dyn).any() else float("nan")},
            "dynamic": {"count": int(dyn.sum()),
                        "median_r_geo": float(np.median(S["r_geo"][dyn])) if dyn.any() else float("nan"),
                        "p95_r_geo": float(np.percentile(S["r_geo"][dyn], 95)) if dyn.any() else float("nan"),
                        "max_r_geo": float(S["r_geo"][dyn].max()) if dyn.any() else float("nan")},
        },
        "correlation_size_vs_geometry": {
            "dist_xy__r_geo": _corr(S["dist_xy"], S["r_geo"]),
            "dist_xy__log10_r_geo": _corr(S["dist_xy"], log_r),
            "dist_xyz__log10_r_geo": _corr(S["dist_xyz"], log_r),
            "abs_z__log10_r_geo": _corr(np.abs(S["z"]), log_r),
        },
        "binned_by_dist_xy": _binned(S["dist_xy"], S["r_geo"]),
        "degenerate": [_degenerate(S, t) for t in (2.0, 5.0, 10.0)],
    }
    name = f"gaussian_size_stats{('_' + tag) if tag else ''}"
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
    _plot(S, log_r, out_dir / f"{name}.png", tag)

    # printable summary
    c1 = report["correlation_size_vs_geometry"]["dist_xy__log10_r_geo"]
    deg5 = report["degenerate"][1]
    bins = report["binned_by_dist_xy"]
    near = next((b for b in bins if b["range_m"][0] == 0), None)
    far = bins[-1] if bins else None
    lines = [
        f"[gaussian-stats{(' ' + tag) if tag else ''}] {n:,} Gaussians over {len(records)} windows "
        f"({report['frac_dynamic']*100:.1f}% dynamic)",
        f"  r_geo (m): median={report['size_m']['r_geo']['p50']:.3f} "
        f"p95={report['size_m']['r_geo']['p95']:.3f} p99={report['size_m']['r_geo']['p99']:.3f} "
        f"max={report['size_m']['r_geo']['max']:.2f}",
        f"  r_max (m): p95={report['size_m']['r_max']['p95']:.3f} "
        f"p99.9={report['size_m']['r_max']['p99.9']:.3f} max={report['size_m']['r_max']['max']:.2f}",
        f"  static median r_geo={report['by_kind']['static']['median_r_geo']:.3f} "
        f"(max {report['by_kind']['static']['max_r_geo']:.1f}) | "
        f"dynamic median={report['by_kind']['dynamic']['median_r_geo']:.3f} "
        f"(max {report['by_kind']['dynamic']['max_r_geo']:.1f})",
        f"  corr(dist_xy, log10 r_geo): pearson={c1['pearson']:.3f} spearman={c1['spearman']:.3f}  "
        f"(>0 => bigger Gaussians farther out)",
    ]
    if near and far:
        lines.append(f"  median r_geo: {near['range_m'][0]:.0f}-{near['range_m'][1]:.0f}m="
                     f"{near['median_r']:.3f}  vs  >{far['range_m'][0]:.0f}m={far['median_r']:.3f}")
    lines.append(
        f"  degenerate r_max>5m: {deg5['count']} ({deg5['frac']*100:.3f}%)" +
        (f", {deg5['frac_dynamic']*100:.0f}% dynamic, median dist_xy={deg5['median_dist_xy']:.1f}m, "
         f"aspect={deg5['median_aspect']:.1f}, opacity={deg5['median_opacity']:.2f}"
         if deg5["count"] else ""))
    lines.append(f"  -> {name}.json / {name}.png")
    return "\n".join(lines)


def _plot(S, log_r, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dyn = S["is_dyn"].astype(bool)
    rng = np.random.default_rng(0)
    n = S["r_geo"].size
    idx = rng.choice(n, min(n, 80000), replace=False)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Gaussian 1σ size analysis{(' — ' + tag) if tag else ''} "
                 f"(N={n:,})", fontsize=13)

    # (a) distance vs effective radius (log y)
    a = ax[0, 0]
    a.scatter(S["dist_xy"][idx], np.clip(S["r_geo"][idx], 1e-3, None),
              s=2, alpha=0.08, c="#2a9d8f", linewidths=0)
    centers = 0.5 * (_DIST_BINS[:-1] + np.minimum(_DIST_BINS[1:], 130))
    med = [np.median(S["r_geo"][(S["dist_xy"] >= _DIST_BINS[i]) & (S["dist_xy"] < _DIST_BINS[i + 1])])
           if ((S["dist_xy"] >= _DIST_BINS[i]) & (S["dist_xy"] < _DIST_BINS[i + 1])).any() else np.nan
           for i in range(len(_DIST_BINS) - 1)]
    a.plot(centers, med, "-o", c="#e63946", lw=2, ms=4, label="median per bin")
    a.set_yscale("log"); a.set_xlim(0, 130)
    a.set_xlabel("dist from ref origin (xy, m)"); a.set_ylabel("r_geo = √(su·sv) [m]")
    a.set_title("size vs distance"); a.legend(); a.grid(alpha=0.2)

    # (b) histogram of r_max (log x)
    b = ax[0, 1]
    b.hist(np.clip(S["r_max"], 1e-3, None), bins=np.logspace(-3, 2, 60),
           color="#457b9d", alpha=0.85)
    for thr, col in ((2, "#f4a261"), (5, "#e63946")):
        b.axvline(thr, c=col, ls="--", label=f"{thr} m")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("r_max = max(su,sv) [m]"); b.set_ylabel("count")
    b.set_title("size distribution"); b.legend(); b.grid(alpha=0.2)

    # (c) median & p95 vs distance
    c = ax[1, 0]
    p95 = [np.percentile(S["r_geo"][(S["dist_xy"] >= _DIST_BINS[i]) & (S["dist_xy"] < _DIST_BINS[i + 1])], 95)
           if ((S["dist_xy"] >= _DIST_BINS[i]) & (S["dist_xy"] < _DIST_BINS[i + 1])).any() else np.nan
           for i in range(len(_DIST_BINS) - 1)]
    c.plot(centers, med, "-o", c="#2a9d8f", label="median")
    c.plot(centers, p95, "-s", c="#e63946", label="p95")
    c.set_xlim(0, 130); c.set_xlabel("dist from ref origin (xy, m)")
    c.set_ylabel("r_geo [m]"); c.set_title("median / p95 vs distance")
    c.legend(); c.grid(alpha=0.2)

    # (d) aspect vs r_max -> are big ones round disks or thin slivers?
    d = ax[1, 1]
    d.scatter(np.clip(S["r_max"][idx], 1e-3, None),
              np.clip(S["aspect"][idx], 1, None),
              s=2, alpha=0.08, c="#8d6e9c", linewidths=0)
    d.set_xscale("log"); d.set_yscale("log")
    d.axvline(5, c="#e63946", ls="--", label="r_max=5 m")
    d.set_xlabel("r_max [m]"); d.set_ylabel("aspect = r_max / r_min")
    d.set_title("anisotropy vs size"); d.legend(); d.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
