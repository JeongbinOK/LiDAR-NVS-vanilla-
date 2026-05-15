from __future__ import annotations

from types import SimpleNamespace

import torch

from nn.lidar_geometry import points_to_lidar_maps
from nn.render_utils import build_gt_normal_map


def _make_cfg_2x1():
    """2-row, 1-column panorama with beams at -0.1, +0.1 rad."""
    import math
    return SimpleNamespace(
        ring_to_elevation_deg=(math.degrees(-0.1), math.degrees(0.1)),
        lidar_height=2,
        lidar_width=1,
        r_near=0.1,
        r_far=20.0,
    )


def test_points_to_lidar_maps_uses_first_hit_intensity():
    cfg = _make_cfg_2x1()
    # Both points at z=0 → el=0; nearest beam tie-breaks to whichever row's
    # elevation is closer (we just verify range/intensity, not which row).
    xyz = torch.tensor([
        [0.0, 10.0, 0.0],  # farther
        [0.0, 5.0, 0.0],   # nearer (first hit)
    ])
    intensity = torch.tensor([0.2, 0.8])
    out = points_to_lidar_maps(xyz, intensity, cfg)
    # Exactly one pixel hit, range=5, intensity=0.8.
    assert int(out["valid_mask"].sum()) == 1
    hit_range = out["range_image"][out["valid_mask"]]
    hit_intensity = out["intensity_image"][out["valid_mask"]]
    assert torch.allclose(hit_range, torch.tensor([5.0]))
    assert torch.allclose(hit_intensity, torch.tensor([0.8]))


def test_points_to_lidar_maps_first_hit_tie_is_deterministic():
    cfg = _make_cfg_2x1()
    xyz = torch.tensor([
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
    ])
    intensity = torch.tensor([0.3, 0.9])
    out = points_to_lidar_maps(xyz, intensity, cfg)
    # Tie-break keeps the earliest point for deterministic supervision.
    hit_intensity = out["intensity_image"][out["valid_mask"]]
    assert torch.allclose(hit_intensity, torch.tensor([0.3]))


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
