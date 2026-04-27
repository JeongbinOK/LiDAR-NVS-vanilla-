from __future__ import annotations

import torch

from nn.render_utils import build_gt_normal_map, build_target_lidar_image


def test_build_target_lidar_image_uses_first_hit_intensity():
    xyz = torch.tensor([
        [0.0, 10.0, 0.0],  # farther
        [0.0, 5.0, 0.0],   # nearer (first hit)
    ])
    intensity = torch.tensor([0.2, 0.8])

    out = build_target_lidar_image(
        xyz,
        intensity,
        height=1,
        width=1,
        el_min_rad=-0.5,
        el_max_rad=0.5,
        r_near=0.1,
        r_far=20.0,
    )

    assert torch.allclose(out["range_image"], torch.tensor([[5.0]]))
    assert torch.allclose(out["intensity_image"], torch.tensor([[0.8]]))


def test_build_target_lidar_image_first_hit_tie_is_deterministic():
    xyz = torch.tensor([
        [0.0, 5.0, 0.0],  # same range, same pixel
        [0.0, 5.0, 0.0],
    ])
    intensity = torch.tensor([0.3, 0.9])

    out = build_target_lidar_image(
        xyz,
        intensity,
        height=1,
        width=1,
        el_min_rad=-0.5,
        el_max_rad=0.5,
        r_near=0.1,
        r_far=20.0,
    )

    # Tie-break keeps the earliest point for deterministic supervision.
    assert torch.allclose(out["intensity_image"], torch.tensor([[0.3]]))


def test_build_gt_normal_map_requires_vertical_neighbors():
    h, w = 3, 3
    range_image = torch.ones(h, w)
    valid_mask = torch.ones(h, w, dtype=torch.bool)
    valid_mask[0, 1] = False

    ray_grid = torch.zeros(3, h, w)
    ray_grid[1] = 1.0

    out = build_gt_normal_map(range_image, valid_mask, ray_grid)
    normal_valid = out["normal_valid"]

    # Center pixel has valid left/right but invalid upper neighbor -> invalid.
    assert not bool(normal_valid[1, 1])
