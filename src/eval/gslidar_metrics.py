"""GS-LiDAR-style evaluation metrics.

Ported faithfully from the official GS-LiDAR repo
(github.com/fudan-zvg/GS-LiDAR : utils/metrics_utils.py + chamfer/fscore.py),
which itself follows the LiDAR4D / LiDAR-NeRF range-image evaluation protocol.

Intentional differences (documented):
  * `scale` (GS-LiDAR's `scale_factor`) defaults to 1.0. GS-LiDAR scales the
    scene down for GS stability and divides depth by `scale_factor` at eval to
    return to meters. Our renderer already outputs depth in meters
    (g2p.scale_factor == 1.0), so no rescaling is applied -- exactly as the user
    requested. Pass `scale != 1.0` only if you deliberately render scaled depth.
  * `point_metrics` back-projects the range image with the camera's
    `row_to_theta` (nuScenes beams are non-uniform / ring-snapped) instead of
    GS-LiDAR's uniform-elevation `pano_to_lidar`. Pred and GT use the SAME
    mapping, so this is the geometrically-correct analogue of GS-LiDAR's step.
    The Chamfer-distance and F-score formulas are identical to upstream.

All image metrics expect tensors/arrays shaped [1, H, W] (one panorama).
"""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

try:
    import lpips as _lpips
except ImportError:  # pragma: no cover
    _lpips = None

from ..models_new.utils.chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from ..models_new.utils.graphics_utils import lidar4d_range_image_to_points


# ---------------------------------------------------------------------------
# Shared heavy objects (build once, reuse across all windows)
# ---------------------------------------------------------------------------
class MetricBackends:
    """Holds the (single) LPIPS network and Chamfer op so we don't rebuild them
    per sequence/window."""

    def __init__(self, lpips_net: str = "alex"):
        if _lpips is not None:
            self.lpips_fn = _lpips.LPIPS(net=lpips_net).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None
        self.chamfer = chamfer_3DDist()


def _np1hw(x):
    """-> numpy [1, H, W] float."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    return x


def _lpips_score(lpips_fn, pred_np, gt_np):
    """Faithful to GS-LiDAR: only computed when H >= 32, AlexNet, normalize=True.

    GS-LiDAR passes `from_numpy(pred).squeeze(0)` (=[H, W]); the LPIPS scaling
    layer broadcasts the single channel to 3, which is identical to replicating
    the channel. We replicate explicitly for robustness across lpips versions
    (numerically equivalent)."""
    if lpips_fn is None or gt_np.shape[-2] < 32:
        return float("nan")
    p = torch.from_numpy(pred_np).float()        # [1, H, W]
    g = torch.from_numpy(gt_np).float()
    p = p.repeat(3, 1, 1).unsqueeze(0)            # [1, 3, H, W]
    g = g.repeat(3, 1, 1).unsqueeze(0)
    with torch.no_grad():
        return float(lpips_fn(p, g, normalize=True).item())


def _ssim(pred_np, gt_np):
    p = pred_np.squeeze(0)
    g = gt_np.squeeze(0)
    data_range = float(np.max(g) - np.min(g))
    if data_range <= 0:
        data_range = 1e-6
    return float(structural_similarity(p, g, data_range=data_range))


# ---------------------------------------------------------------------------
# Depth (range image)  -- GS-LiDAR DepthMeter.compute_depth_errors
# ---------------------------------------------------------------------------
def depth_errors(pred, gt, backends: MetricBackends, scale: float = 1.0,
                 min_depth: float = 1e-6, max_depth: float = 80.0) -> dict:
    pred = _np1hw(pred) / scale
    gt = _np1hw(gt) / scale
    pred = pred.copy()
    gt = gt.copy()
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth
    gt[gt < min_depth] = min_depth
    gt[gt > max_depth] = max_depth

    rmse = float(np.sqrt(((gt - pred) ** 2).mean()))
    medae = float(np.median(np.abs(gt - pred)))
    lpips_v = _lpips_score(backends.lpips_fn, pred, gt)
    ssim = _ssim(pred, gt)
    psnr = float(10.0 * np.log10(max_depth ** 2 / np.mean((pred - gt) ** 2)))
    return {"rmse": rmse, "medae": medae, "lpips": lpips_v, "ssim": ssim, "psnr": psnr}


# ---------------------------------------------------------------------------
# Intensity  -- GS-LiDAR IntensityMeter.compute_intensity_errors
# ---------------------------------------------------------------------------
def intensity_errors(pred, gt, backends: MetricBackends, scale: float = 1.0,
                     min_intensity: float = 1e-6, max_intensity: float = 1.0) -> dict:
    pred = _np1hw(pred) / scale
    gt = _np1hw(gt) / scale
    pred = pred.copy()
    gt = gt.copy()
    pred[pred < min_intensity] = min_intensity
    pred[pred > max_intensity] = max_intensity
    gt[gt < min_intensity] = min_intensity
    gt[gt > max_intensity] = max_intensity

    rmse = float(np.sqrt(((gt - pred) ** 2).mean()))
    medae = float(np.median(np.abs(gt - pred)))
    lpips_v = _lpips_score(backends.lpips_fn, pred, gt)
    ssim = _ssim(pred, gt)
    psnr = float(10.0 * np.log10(max_intensity ** 2 / np.mean((pred - gt) ** 2)))
    return {"rmse": rmse, "medae": medae, "lpips": lpips_v, "ssim": ssim, "psnr": psnr}


# ---------------------------------------------------------------------------
# Raydrop  -- GS-LiDAR RaydropMeter
# ---------------------------------------------------------------------------
def raydrop_errors(pred, gt, ratio: float = 0.5) -> dict:
    pred = _np1hw(pred)
    gt = _np1hw(gt)
    rmse = float(np.sqrt(((gt - pred) ** 2).mean()))
    preds_mask = np.where(pred > ratio, 1, 0)
    acc = float((preds_mask == gt).mean())
    tp = float(np.sum((gt == 1) & (preds_mask == 1)))
    fp = float(np.sum((gt == 0) & (preds_mask == 1)))
    fn = float(np.sum((gt == 1) & (preds_mask == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"rmse": rmse, "acc": acc, "f1": float(f1)}


# ---------------------------------------------------------------------------
# Point cloud (Chamfer + F-score)  -- GS-LiDAR PointsMeter + fscore
# ---------------------------------------------------------------------------
def _fscore(dist1, dist2, threshold: float = 0.05):
    # dist1/dist2 are SQUARED euclidean distances (chamfer_3DDist convention),
    # so the 0.05 threshold matches GS-LiDAR exactly.
    p1 = torch.mean((dist1 < threshold).float(), dim=1)
    p2 = torch.mean((dist2 < threshold).float(), dim=1)
    f = 2 * p1 * p2 / (p1 + p2)
    f[torch.isnan(f)] = 0
    return f, p1, p2


def _range_to_points(depth_1hw, row_to_theta, vfov, hfov, near, far):
    return lidar4d_range_image_to_points(
        depth_1hw,
        vfov,
        hfov,
        row_to_theta=row_to_theta,
        min_range=near,
        max_range=far,
    ).contiguous()


def point_metrics(pred_depth, gt_depth, row_to_theta, backends: MetricBackends,
                  vfov, hfov=(-180.0, 180.0), scale: float = 1.0,
                  near: float = 0.0, far: float = 80.0,
                  fscore_threshold: float = 0.05) -> dict:
    """LiDAR4D point metrics for ``[1,H,W]`` range images in metres.

    ``pred_depth`` should already be hard-masked by predicted raydrop. We apply
    the LiDAR4D/GS-LiDAR 80 m support symmetrically and do not add a CD-only
    near crop, hence the default ``near=0``. Both point sets use symmetric
    squared Chamfer distance.
    """
    pred_depth = pred_depth / scale
    gt_depth = gt_depth / scale
    if not torch.is_tensor(row_to_theta):
        row_to_theta = torch.as_tensor(row_to_theta)
    row_to_theta = row_to_theta.to(pred_depth.device, dtype=pred_depth.dtype)

    pred_lidar = _range_to_points(pred_depth, row_to_theta, vfov, hfov, near, far)
    gt_lidar = _range_to_points(gt_depth, row_to_theta, vfov, hfov, near, far)
    if pred_lidar.shape[0] == 0 or gt_lidar.shape[0] == 0:
        return {"cd": float("nan"), "fscore": float("nan"),
                "pred_points": int(pred_lidar.shape[0]), "gt_points": int(gt_lidar.shape[0])}

    dist1, dist2, _, _ = backends.chamfer(pred_lidar[None].contiguous(),
                                          gt_lidar[None].contiguous())
    cd = float((dist1.mean() + dist2.mean()).item())
    f, _, _ = _fscore(dist1, dist2, fscore_threshold)
    return {"cd": cd, "fscore": float(f.cpu()[0]),
            "pred_points": int(pred_lidar.shape[0]), "gt_points": int(gt_lidar.shape[0])}
