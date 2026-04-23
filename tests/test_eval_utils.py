from __future__ import annotations

import torch

from config import QGSConfig
from nn.eval_utils import gaussian_slice_stats


def test_gaussian_slice_stats_counts_visibility_and_drop():
    cfg = QGSConfig()
    primitives = {
        "means3D": torch.tensor([
            [0.0, 10.0, 0.0],   # visible
            [0.0, 80.0, 0.0],   # too far
            [0.0, 1.0, 10.0],   # elevation out of FOV
        ]),
    }
    radii = torch.tensor([1.0, 0.0, 0.0])
    n_touched = torch.tensor([3, 0, 0])
    viewmatrix = torch.eye(4)

    stats = gaussian_slice_stats(primitives, radii, n_touched, viewmatrix, cfg)

    assert stats["n_generated"] == 3
    assert stats["n_frustum_visible"] == 1
    assert stats["n_positive_radius"] == 1
    assert stats["n_touched"] == 1
    assert stats["n_dropped_out_of_fov"] == 2
    assert stats["n_dropped_zero_radius"] == 0
    assert stats["n_dropped_untouched"] == 0
    assert stats["n_dropped_raster"] == 2
