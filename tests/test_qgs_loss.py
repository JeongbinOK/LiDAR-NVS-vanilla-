from __future__ import annotations

from types import SimpleNamespace

import torch

from nn.render_utils import build_gt_normal_map, make_lidar_ray_grid
from nn.qgs_loss import QGSLoss


def test_qgs_loss_penalizes_alpha_collapse():
    loss_fn = QGSLoss(w_depth=1.0, w_intensity=1.0, w_raydrop=1.0)

    target = {
        "range_image": torch.tensor([[10.0]]),
        "intensity_image": torch.tensor([[0.25]]),
        "valid_mask": torch.tensor([[True]]),
    }

    good = SimpleNamespace(
        range=torch.tensor([[10.0]]),
        middepth=torch.tensor([[10.0]]),
        intensity=torch.tensor([[0.25]]),
        alpha_accum=torch.tensor([[1.0]]),
    )
    bad = SimpleNamespace(
        range=torch.tensor([[0.0]]),
        middepth=torch.tensor([[0.0]]),
        intensity=torch.tensor([[0.0]]),
        alpha_accum=torch.tensor([[0.0]]),
    )

    good_loss = loss_fn(good, target, drop_prob=torch.tensor([[0.0]]))
    bad_loss = loss_fn(bad, target, drop_prob=torch.tensor([[1.0]]))

    assert bad_loss["total"] > good_loss["total"]
    assert bad_loss["raydrop"] > good_loss["raydrop"]


def test_qgs_loss_uses_raw_alpha_blended_intensity():
    loss_fn = QGSLoss(w_depth=0.0, w_intensity=1.0, w_raydrop=0.0)

    target = {
        "range_image": torch.tensor([[10.0]]),
        "intensity_image": torch.tensor([[0.2]]),
        "valid_mask": torch.tensor([[True]]),
    }
    rendered = SimpleNamespace(
        range=torch.tensor([[5.0]]),
        middepth=torch.tensor([[10.0]]),
        intensity=torch.tensor([[0.2]]),
        alpha_accum=torch.tensor([[0.25]]),
    )

    loss = loss_fn(rendered, target, drop_prob=torch.tensor([[0.0]]))

    assert torch.allclose(loss["intensity"], torch.tensor(0.0))


def test_qgs_loss_qgs_normal_term_is_zero_when_normals_align():
    loss_fn = QGSLoss(w_depth=0.0, w_intensity=0.0, w_raydrop=0.0, w_normal=1.0)

    h, w = 3, 3
    ray_grid = make_lidar_ray_grid(h, w, -0.2, 0.2)
    alpha = torch.full((h, w), 0.5)
    middepth = torch.ones(h, w)
    target = {
        "range_image": torch.ones(h, w),
        "intensity_image": torch.zeros(h, w),
        "valid_mask": torch.ones(h, w, dtype=torch.bool),
    }

    ref = build_gt_normal_map(middepth, alpha > loss_fn.alpha_eps, ray_grid)["normal_image"]
    rendered = SimpleNamespace(
        range=middepth.clone(),
        middepth=middepth,
        intensity=torch.zeros(h, w),
        alpha_accum=alpha,
        normal=alpha.unsqueeze(0) * ref,
        curvature=torch.zeros(h, w),
    )

    loss = loss_fn(rendered, target, drop_prob=torch.zeros(h, w), ray_grid=ray_grid)

    assert torch.allclose(loss["normal"], torch.tensor(0.0), atol=1e-5)


def test_qgs_loss_curvature_guidance_downweights_high_curvature_pixels():
    loss_fn = QGSLoss(w_depth=0.0, w_intensity=0.0, w_raydrop=0.0, w_normal=1.0)

    h, w = 3, 3
    ray_grid = make_lidar_ray_grid(h, w, -0.2, 0.2)
    alpha = torch.ones(h, w)
    middepth = torch.ones(h, w)
    target = {
        "range_image": torch.ones(h, w),
        "intensity_image": torch.zeros(h, w),
        "valid_mask": torch.ones(h, w, dtype=torch.bool),
    }

    ref = build_gt_normal_map(middepth, alpha > loss_fn.alpha_eps, ray_grid)["normal_image"]
    rendered_low = SimpleNamespace(
        range=middepth.clone(),
        middepth=middepth,
        intensity=torch.zeros(h, w),
        alpha_accum=alpha,
        normal=-ref.clone(),
        curvature=torch.full((h, w), 1e-4),
    )
    rendered_high = SimpleNamespace(
        range=middepth.clone(),
        middepth=middepth,
        intensity=torch.zeros(h, w),
        alpha_accum=alpha,
        normal=-ref.clone(),
        curvature=torch.full((h, w), 1e4),
    )

    low = loss_fn(rendered_low, target, drop_prob=torch.zeros(h, w), ray_grid=ray_grid)
    high = loss_fn(rendered_high, target, drop_prob=torch.zeros(h, w), ray_grid=ray_grid)

    assert high["normal"] < low["normal"]
