from __future__ import annotations

import json

import torch

from config import QGSConfig
from nn.eval_utils import (
    _context_diagnostics,
    _summarize_context_groups,
    gaussian_slice_stats,
    load_cfg_from_checkpoint,
)


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


def test_context_diagnostics_counts_init_and_subfloor_points():
    primitives = {
        "aux": {
            "g_rot": torch.tensor([[0.2, 0.7, 1.0]]),
            "g_center": torch.tensor([[0.25, 0.75, 1.0]]),
            "g_scale": torch.tensor([[0.3, 0.8, 1.0]]),
            "omega_local": torch.zeros(1, 3, 3),
            "delta_c": torch.zeros(1, 3, 3),
            "delta_mu": torch.zeros(1, 3),
            "delta_gap": torch.zeros(1, 3),
            "delta_log_abs_s3": torch.zeros(1, 3),
        },
        "geom_init": {
            "use_geom_init": torch.tensor([[True, False, True]]),
            "tangent_aniso": torch.zeros(1, 3),
            "curvature_aniso": torch.zeros(1, 3),
            "kappa1_init": torch.zeros(1, 3),
            "kappa2_init": torch.zeros(1, 3),
        },
        "opacities": torch.full((3, 1), 0.5),
        "k_eff": torch.tensor([16.0, 4.0, 12.0]),
        "scales": torch.tensor([
            [0.3, 0.2, 0.1],
            [0.3, 0.2, 0.1],
            [0.3, 0.2, 0.1],
        ]),
        "means3D": torch.zeros(3, 3),
        "diagnostics": {"memory_stages": []},
    }

    diag = _context_diagnostics("dynamic_0", "dynamic", primitives, num_points=3)

    assert diag["n_points_with_init"] == 2
    assert diag["n_points_without_init"] == 1
    assert diag["n_points_subfloor"] == 1
    assert diag["n_points_with_init"] + diag["n_points_without_init"] == diag["n_input_points"]


def test_context_group_summary_accumulates_static_and_dynamic_totals():
    groups = _summarize_context_groups([
        {
            "name": "static",
            "context_type": "static",
            "skipped": False,
            "n_input_points": 10,
            "n_points_with_init": 7,
            "n_points_without_init": 3,
            "n_points_subfloor": 3,
        },
        {
            "name": "dynamic_0",
            "context_type": "dynamic",
            "skipped": False,
            "n_input_points": 5,
            "n_points_with_init": 4,
            "n_points_without_init": 1,
            "n_points_subfloor": 1,
        },
        {
            "name": "dynamic_1",
            "context_type": "dynamic",
            "skipped": True,
            "n_input_points": 2,
            "n_points_with_init": None,
            "n_points_without_init": None,
            "n_points_subfloor": None,
        },
    ])

    assert groups["static"]["realized_contexts"] == 1
    assert groups["static"]["input_points"] == 10
    assert groups["static"]["points_with_init"] == 7
    assert groups["static"]["points_without_init"] == 3
    assert groups["dynamic"]["realized_contexts"] == 1
    assert groups["dynamic"]["skipped_contexts"] == 1
    assert groups["dynamic"]["input_points"] == 7
    assert groups["dynamic"]["points_with_init"] == 4
    assert groups["dynamic"]["points_without_init"] == 1


def test_load_cfg_from_checkpoint_preserves_legacy_ptv3_stem_width(tmp_path):
    run_dir = tmp_path / "outputs" / "train_001"
    ckpt_dir = run_dir / "ckpt"
    cfg_dir = run_dir / "configs"
    ckpt_dir.mkdir(parents=True)
    cfg_dir.mkdir(parents=True)

    config = {
        "input_feature_dim": 8,
    }
    with open(cfg_dir / "config.json", "w") as f:
        json.dump(config, f)
    checkpoint_path = ckpt_dir / "best_model.pt"
    checkpoint_path.write_bytes(b"")

    cfg = load_cfg_from_checkpoint(str(checkpoint_path))

    assert cfg.input_feature_dim == 8
    assert cfg.primitive_mode == "per_point"
    assert cfg.ptv3_model_in_channels == 8
    assert cfg.ptv3_decoupled_stem is False
    assert cfg.ptv3_pdnorm_bn is False
    assert cfg.ptv3_pdnorm_ln is False
