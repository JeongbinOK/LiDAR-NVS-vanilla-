from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from config import OFFICIAL_FULL_PTV3_BACKBONE_PARAMS, QGSConfig, resolve_lidar_latent_dim
from nn.model import LIDAR_LATENT_DIM, QGSModel
from nn.ptv3.model import Point
from nn.ptv3.wrapper import PTv3Backbone


class _RecorderBackbone(nn.Module):
    supports_context_type = True

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.seen_context_type: str | None = None
        self.seen_feature_width: int | None = None
        self.seen_offsets: torch.Tensor | None = None

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: torch.Tensor,
        *,
        context_type: str,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.seen_context_type = context_type
        self.seen_feature_width = int(point_features.shape[1])
        self.seen_offsets = offsets
        return torch.zeros(xyz.shape[0], self.out_channels, device=xyz.device, dtype=xyz.dtype)


class _ContextAgnosticBackbone(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, xyz: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(xyz.shape[0], self.out_channels, device=xyz.device, dtype=xyz.dtype)


class _StubHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        features: torch.Tensor,
        geom_init: dict,
        is_dynamic_flag: torch.Tensor,
        intensity_input: torch.Tensor | None = None,
    ) -> dict:
        batch_size, num_points, _ = features.shape
        device = features.device
        dtype = features.dtype
        return {
            "center": geom_init["c_init"],
            "R": geom_init["R_init"],
            "s": geom_init["s_init"],
            "alpha": torch.full((batch_size, num_points), 0.5, device=device, dtype=dtype),
            "intensity": intensity_input,
            "latent": torch.zeros(batch_size, num_points, self.latent_dim, device=device, dtype=dtype),
            "aux": {
                "g_rot": torch.zeros(batch_size, num_points, device=device, dtype=dtype),
                "g_center": torch.zeros(batch_size, num_points, device=device, dtype=dtype),
                "g_scale": torch.zeros(batch_size, num_points, device=device, dtype=dtype),
                "omega_local": torch.zeros(batch_size, num_points, 3, device=device, dtype=dtype),
                "delta_c": torch.zeros(batch_size, num_points, 3, device=device, dtype=dtype),
                "delta_mu": torch.zeros(batch_size, num_points, device=device, dtype=dtype),
                "delta_gap": torch.zeros(batch_size, num_points, device=device, dtype=dtype),
                "delta_log_abs_s3": torch.zeros(batch_size, num_points, device=device, dtype=dtype),
            },
        }


def _stub_knn(xyz: torch.Tensor, candidate_xyz: torch.Tensor, **kwargs) -> dict:
    num_points = xyz.shape[0]
    idx = torch.zeros(num_points, 1, dtype=torch.long, device=xyz.device)
    mask = torch.ones(num_points, 1, dtype=torch.bool, device=xyz.device)
    k_eff = torch.ones(num_points, dtype=torch.long, device=xyz.device)
    return {"idx": idx, "mask": mask, "k_eff": k_eff}


def _stub_neighbors(candidate_xyz: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return candidate_xyz[idx]


def _stub_geom_init(points: torch.Tensor, neighbors: torch.Tensor, k_eff: torch.Tensor, **kwargs) -> dict:
    _, num_points, _ = points.shape
    device = points.device
    dtype = points.dtype
    return {
        "c_init": points.clone(),
        "R_init": torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(1, num_points, 3, 3).contiguous(),
        "s_init": torch.full((1, num_points, 3), 0.1, device=device, dtype=dtype),
        "fit_quality": torch.zeros(1, num_points, 4, device=device, dtype=dtype),
        "use_geom_init": torch.ones(1, num_points, device=device, dtype=torch.bool),
        "kappa1_init": torch.zeros(1, num_points, device=device, dtype=dtype),
        "kappa2_init": torch.zeros(1, num_points, device=device, dtype=dtype),
        "tangent_aniso": torch.zeros(1, num_points, device=device, dtype=dtype),
        "curvature_aniso": torch.zeros(1, num_points, device=device, dtype=dtype),
    }


def _anchor_output(num_anchors: int, token_width: int = 22):
    from models.geometry.voxel_anchor import VoxelAnchorOutput

    return VoxelAnchorOutput(
        c_init=torch.randn(num_anchors, 3),
        R_init=torch.eye(3).view(1, 3, 3).expand(num_anchors, 3, 3).contiguous(),
        s_init=torch.full((num_anchors, 3), 0.1),
        kappa1=torch.zeros(num_anchors),
        kappa2=torch.zeros(num_anchors),
        fit_quality=torch.zeros(num_anchors, 4),
        tangent_aniso=torch.zeros(num_anchors),
        curvature_aniso=torch.zeros(num_anchors),
        n_points=torch.full((num_anchors,), 8, dtype=torch.long),
        i_mean=torch.full((num_anchors,), 0.5),
        i_std=torch.zeros(num_anchors),
        src_ratio=torch.zeros(num_anchors),
        token=torch.randn(num_anchors, token_width),
        use_geom_init=torch.ones(num_anchors, dtype=torch.bool),
        k_eff=torch.full((num_anchors,), 8, dtype=torch.long),
        diagnostics={"query_voxels": num_anchors, "final_anchors": num_anchors},
    )


def test_qgs_model_forward_context_rejects_invalid_context_type(monkeypatch):
    backbone = _ContextAgnosticBackbone(8)
    monkeypatch.setattr("nn.model.PTv3Backbone", lambda *args, **kwargs: backbone)

    model = QGSModel(QGSConfig(feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM, primitive_mode="per_point"))
    xyz = torch.zeros(4, 3)
    intensity = torch.zeros(4)

    with pytest.raises(ValueError, match="context_type"):
        model.forward_context(xyz, intensity, context_type="unknown")


def test_qgs_model_forward_context_passes_context_type_to_opt_in_backbone(monkeypatch):
    cfg = QGSConfig(feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM, primitive_mode="per_point")
    recorder = _RecorderBackbone(cfg.feature_dim)
    monkeypatch.setattr("nn.model.PTv3Backbone", lambda *args, **kwargs: recorder)

    model = QGSModel(cfg)
    model.qgs_head = _StubHead(cfg.lidar_latent_dim)

    monkeypatch.setattr("nn.model.hybrid_radius_knn", _stub_knn)
    monkeypatch.setattr("nn.model.gather_neighbors", _stub_neighbors)
    monkeypatch.setattr("nn.model.fit_local_quadrics", _stub_geom_init)

    xyz = torch.randn(6, 3)
    intensity = torch.sigmoid(torch.randn(6))
    out = model.forward_context(
        xyz,
        intensity,
        context_type="dynamic",
        is_dynamic_flag=torch.ones(6),
    )

    assert recorder.seen_context_type == "dynamic"
    assert out["means3D"].shape == (6, 3)


def test_qgs_model_keeps_context_api_with_context_agnostic_backbone(monkeypatch):
    backbone = _ContextAgnosticBackbone(8)
    monkeypatch.setattr("nn.model.PTv3Backbone", lambda *args, **kwargs: backbone)

    cfg = QGSConfig(feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM, primitive_mode="per_point")
    model = QGSModel(cfg)
    model.qgs_head = _StubHead(cfg.lidar_latent_dim)

    monkeypatch.setattr("nn.model.hybrid_radius_knn", _stub_knn)
    monkeypatch.setattr("nn.model.gather_neighbors", _stub_neighbors)
    monkeypatch.setattr("nn.model.fit_local_quadrics", _stub_geom_init)

    xyz = torch.randn(6, 3)
    intensity = torch.sigmoid(torch.randn(6))
    out = model.forward_context(
        xyz,
        intensity,
        context_type="static",
        is_dynamic_flag=torch.zeros(6),
    )

    assert out["means3D"].shape == (6, 3)


def test_qgs_config_validates_primitive_mode():
    with pytest.raises(ValueError, match="primitive_mode"):
        QGSConfig(primitive_mode="bad")


def test_qgs_config_auto_resolves_lidar_latent_dim_from_extension():
    cfg = QGSConfig()

    assert cfg.lidar_latent_dim == LIDAR_LATENT_DIM
    assert resolve_lidar_latent_dim(None) == LIDAR_LATENT_DIM


def test_qgs_config_rejects_explicit_lidar_latent_dim_mismatch():
    with pytest.raises(ValueError, match="LIDAR_LATENT_DIM"):
        QGSConfig(lidar_latent_dim=LIDAR_LATENT_DIM + 1)


def test_qgs_model_uses_mode_specific_input_width(monkeypatch):
    seen = []

    def _make_backbone(*args, **kwargs):
        seen.append((kwargs["in_channels"] if "in_channels" in kwargs else args[0], kwargs))
        return _ContextAgnosticBackbone(kwargs["out_channels"])

    monkeypatch.setattr("nn.model.PTv3Backbone", _make_backbone)

    QGSModel(QGSConfig(primitive_mode="per_point", feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM))
    QGSModel(QGSConfig(primitive_mode="voxel_anchor", feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM))

    assert seen[0][0] == 8
    assert seen[1][0] == 22


def test_qgs_model_forward_anchor_context_uses_22d_token(monkeypatch):
    cfg = QGSConfig(primitive_mode="voxel_anchor", feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM)
    recorder = _RecorderBackbone(cfg.feature_dim)
    monkeypatch.setattr("nn.model.PTv3Backbone", lambda *args, **kwargs: recorder)

    model = QGSModel(cfg)
    model.qgs_head = _StubHead(cfg.lidar_latent_dim)
    out = model.forward_anchor_context(_anchor_output(3), context_type="dynamic")

    assert recorder.seen_context_type == "dynamic"
    assert recorder.seen_feature_width == 22
    assert out["means3D"].shape == (3, 3)


def test_qgs_model_forward_anchor_contexts_batched_preserves_empty_slots(monkeypatch):
    cfg = QGSConfig(primitive_mode="voxel_anchor", feature_dim=8, lidar_latent_dim=LIDAR_LATENT_DIM)
    recorder = _RecorderBackbone(cfg.feature_dim)
    monkeypatch.setattr("nn.model.PTv3Backbone", lambda *args, **kwargs: recorder)

    model = QGSModel(cfg)
    model.qgs_head = _StubHead(cfg.lidar_latent_dim)
    outs = model.forward_anchor_contexts_batched(
        [_anchor_output(2), _anchor_output(0), _anchor_output(1)],
        context_type="dynamic",
    )

    assert outs[0] is not None
    assert outs[1] is None
    assert outs[2] is not None
    assert recorder.seen_offsets.tolist() == [2, 3]
    assert outs[0]["aux"]["g_rot"].shape == (1, 2)
    assert outs[2]["aux"]["g_rot"].shape == (1, 1)


def test_ptv3_backbone_forwards_condition_with_native_8ch_input():
    backbone = PTv3Backbone(
        in_channels=8,
        out_channels=4,
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(8, 16, 32),
        enc_num_head=(1, 2, 4),
        enc_patch_size=(8, 8, 8),
        dec_depths=(1, 1),
        dec_channels=(16, 8),
        dec_num_head=(2, 1),
        dec_patch_size=(8, 8),
        enable_flash=False,
    )

    captured: dict[str, object] = {}

    def _fake_forward(data_dict: dict) -> SimpleNamespace:
        captured.update(data_dict)
        return SimpleNamespace(feat=torch.ones(data_dict["coord"].shape[0], 16))

    backbone.ptv3.forward = _fake_forward  # type: ignore[method-assign]

    xyz = torch.randn(5, 3)
    feats = torch.randn(5, 8)
    out = backbone(xyz, feats, context_type="dynamic")

    assert captured["condition"] == "dynamic"
    assert captured["feat"].shape == (5, 8)
    assert out.shape == (5, 4)


def test_ptv3_backbone_rejects_point_count_mismatch():
    backbone = PTv3Backbone(
        in_channels=8,
        out_channels=4,
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(8, 16, 32),
        enc_num_head=(1, 2, 4),
        enc_patch_size=(8, 8, 8),
        dec_depths=(1, 1),
        dec_channels=(16, 8),
        dec_num_head=(2, 1),
        dec_patch_size=(8, 8),
        enable_flash=False,
    )

    def _fake_forward(data_dict: dict) -> SimpleNamespace:
        num_points = data_dict["coord"].shape[0]
        return SimpleNamespace(feat=torch.ones(num_points - 1, 16))

    backbone.ptv3.forward = _fake_forward  # type: ignore[method-assign]

    xyz = torch.randn(5, 3)
    feats = torch.randn(5, 8)

    with pytest.raises(RuntimeError, match="changed point count"):
        backbone(xyz, feats)


def test_ptv3_serialization_accepts_single_grid_cell_cloud():
    point = Point(
        coord=torch.zeros(12, 3),
        feat=torch.randn(12, 22),
        grid_size=0.1,
        offset=torch.tensor([6, 12], dtype=torch.long),
    )

    point.serialization(order=("hilbert", "hilbert-trans"))

    assert point.serialized_depth == 1
    assert point.serialized_code.shape == (2, 12)


def test_qgs_config_defaults_to_official_full_with_context_branches():
    cfg = QGSConfig()
    assert cfg.primitive_mode == "voxel_anchor"
    assert cfg.ptv3_model_in_channels == 22
    assert cfg.resolved_input_channels() == 22
    assert cfg.ptv3_batch_norm_eval is True
    assert cfg.ptv3_decoupled_stem is True
    assert cfg.ptv3_pdnorm_bn is True
    assert cfg.ptv3_pdnorm_ln is True
    assert tuple(cfg.ptv3_condition_names) == ("static", "dynamic")
    assert tuple(cfg.ptv3_stride) == (2, 2)
    assert tuple(cfg.ptv3_enc_depths) == (2, 2, 4)
    assert tuple(cfg.ptv3_enc_channels) == (32, 64, 128)
    assert tuple(cfg.ptv3_dec_depths) == (2, 2)
    assert tuple(cfg.ptv3_dec_channels) == (64, 128)


def test_legacy_unbranched_official_full_parameter_count_matches_target():
    cfg = QGSConfig(
        primitive_mode="per_point",
        ptv3_decoupled_stem=False,
        ptv3_pdnorm_bn=False,
        ptv3_pdnorm_ln=False,
        ptv3_stride=(2, 2, 2, 2),
        ptv3_enc_depths=(2, 2, 2, 6, 2),
        ptv3_enc_channels=(32, 64, 128, 256, 512),
        ptv3_enc_num_head=(2, 4, 8, 16, 32),
        ptv3_enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        ptv3_dec_depths=(2, 2, 2, 2),
        ptv3_dec_channels=(64, 64, 128, 256),
        ptv3_dec_num_head=(4, 4, 8, 16),
        ptv3_dec_patch_size=(1024, 1024, 1024, 1024),
    )
    backbone_kwargs = cfg.ptv3_backbone_kwargs()
    backbone_kwargs["enable_flash"] = False
    backbone = PTv3Backbone(
        in_channels=cfg.input_feature_dim,
        out_channels=cfg.feature_dim,
        **backbone_kwargs,
    )

    assert backbone.parameter_count(include_projection=False) == OFFICIAL_FULL_PTV3_BACKBONE_PARAMS


def test_ptv3_backbone_can_temporarily_freeze_batch_norm_stats():
    backbone = PTv3Backbone(
        in_channels=8,
        out_channels=4,
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(8, 16, 32),
        enc_num_head=(1, 2, 4),
        enc_patch_size=(8, 8, 8),
        dec_depths=(1, 1),
        dec_channels=(16, 8),
        dec_num_head=(2, 1),
        dec_patch_size=(8, 8),
        enable_flash=False,
        batch_norm_eval=True,
    )
    backbone.train()
    bn_modules = [m for m in backbone.ptv3.modules() if isinstance(m, nn.BatchNorm1d)]

    assert bn_modules
    assert all(module.training for module in bn_modules)
    with backbone._temporary_batch_norm_eval():
        assert all(not module.training for module in bn_modules)
    assert all(module.training for module in bn_modules)
