from __future__ import annotations

import torch

from models.geometry.cartesian_voxel import CartesianVoxelizer
from models.geometry.spherical_voxel import SphericalVoxelizer
from models.geometry.spherical_voxel import SphericalVoxelOutput
from models.geometry.voxel_anchor import (
    VoxelAnchorBuilder,
    _masked_neighbor_intensity_stats,
    _quad_fit_and_token,
    build_anchor_token,
)


def test_spherical_query_voxel_min_points_is_separate_from_anchor_k_min():
    xyz = torch.tensor([[1.0, 0.0, 0.0]])
    intensity = torch.tensor([0.5])
    src = torch.tensor([0.0])

    vox = SphericalVoxelizer(query_voxel_min_points=1)(xyz, intensity, src)

    assert vox.query_xyz.shape[0] == 1
    assert vox.n_points.tolist() == [1]


def test_cartesian_query_voxel_min_points_is_separate_from_anchor_k_min():
    xyz = torch.tensor([[0.01, 0.0, 0.0]])
    intensity = torch.tensor([0.5])
    src = torch.tensor([1.0])

    vox = CartesianVoxelizer(voxel_size=1.0, query_voxel_min_points=1)(xyz, intensity, src)

    assert vox.query_xyz.shape[0] == 1
    assert vox.n_points.tolist() == [1]


def test_anchor_builder_drops_sparse_query_after_knn_filter():
    xyz = torch.tensor([[1.0, 0.0, 0.0]])
    intensity = torch.tensor([0.5])
    src = torch.tensor([0.0])
    builder = VoxelAnchorBuilder(query_voxel_min_points=1, k_min=8, k_target=8)

    out = builder(xyz, intensity, src)

    assert out.token.shape[0] == 0
    assert out.diagnostics["query_voxels"] == 1
    assert out.diagnostics["k_eff_pass"] == 0
    assert out.diagnostics["final_anchors"] == 0
    assert out.diagnostics["fallback_reason"] == "k_eff_lt_k_min"


def test_static_dual_frame_voxelizes_frame1_queries_in_frame1_sensor_space():
    builder = VoxelAnchorBuilder(
        dphi_deg=10.0,
        dtheta_deg=10.0,
        dr_m=0.1,
        query_voxel_min_points=1,
        knn_r_min=0.0,
        knn_r_max=100.0,
    )
    pose_1_to_0 = torch.eye(4)
    pose_1_to_0[0, 3] = 10.0

    xyz = torch.tensor([
        [2.0, 0.0, 0.0],   # frame0-native point
        [11.0, 0.0, 0.0],  # frame1 point already lifted to frame0; native range is 1m
    ])
    intensity = torch.tensor([0.2, 0.8])
    src = torch.tensor([0.0, 1.0])

    vox, r_max = builder._dual_frame_vox(xyz, intensity, src, pose_1_to_0)

    assert torch.allclose(vox.query_xyz, xyz)
    assert torch.allclose(vox.src_ratio, torch.tensor([0.0, 1.0]))
    expected_r = torch.cat([
        builder._voxel_r_max(torch.tensor([[2.0, 0.0, 0.0]])),
        builder._voxel_r_max(torch.tensor([[1.0, 0.0, 0.0]])),
    ])
    wrong_frame0_r = builder._voxel_r_max(torch.tensor([[11.0, 0.0, 0.0]]))[0]
    assert torch.allclose(r_max, expected_r)
    assert not torch.allclose(r_max[1], wrong_frame0_r)


def test_anchor_quad_fit_uses_configured_quadric_params(monkeypatch):
    captured = {}

    def fake_knn(**kwargs):
        device = kwargs["points"].device
        idx = torch.arange(8, device=device).view(1, 8)
        mask = torch.ones(1, 8, dtype=torch.bool, device=device)
        return {"idx": idx, "mask": mask, "k_eff": torch.tensor([8], device=device)}

    def fake_fit_local_quadrics(points, neighbors, k_eff, **kwargs):
        captured.update(kwargs)
        device = points.device
        dtype = points.dtype
        return {
            "c_init": points.clone(),
            "R_init": torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3),
            "s_init": torch.full((1, 1, 3), 0.2, device=device, dtype=dtype),
            "fit_quality": torch.tensor([[[0.1, 0.0, 1.0, 0.0]]], device=device, dtype=dtype),
            "use_geom_init": torch.ones(1, 1, dtype=torch.bool, device=device),
            "kappa1_init": torch.zeros(1, 1, device=device, dtype=dtype),
            "kappa2_init": torch.zeros(1, 1, device=device, dtype=dtype),
            "tangent_aniso": torch.zeros(1, 1, device=device, dtype=dtype),
            "curvature_aniso": torch.zeros(1, 1, device=device, dtype=dtype),
        }

    monkeypatch.setattr("models.geometry.voxel_anchor.hybrid_radius_knn", fake_knn)
    monkeypatch.setattr("models.geometry.voxel_anchor.fit_local_quadrics", fake_fit_local_quadrics)

    query = torch.zeros(1, 3)
    vox = SphericalVoxelOutput(
        query_xyz=query,
        n_points=torch.tensor([8]),
        i_mean=torch.tensor([0.5]),
        i_std=torch.tensor([0.0]),
        src_ratio=torch.tensor([0.0]),
        point_voxel=torch.zeros(8, dtype=torch.long),
        voxel_hash=torch.zeros(1, dtype=torch.long),
    )
    candidates = torch.randn(8, 3)

    out = _quad_fit_and_token(
        vox,
        candidates_xyz=candidates,
        candidates_intensity=torch.full((8,), 0.5),
        k_min=8,
        k_target=8,
        residual_threshold=1.0,
        filter_mode="residual",
        planarity_threshold=0.2,
        token_variant="full",
        knn_chunk_size=8,
        knn_method="bruteforce",
        r_max_fn=lambda q: torch.ones(q.shape[0], device=q.device, dtype=q.dtype),
        quadric_gamma=1.73,
        quadric_kappa_max=7.0,
        quadric_eps_lambda=0.02,
        quadric_eps_kappa=0.004,
        quadric_eps_s=0.005,
        quadric_eps_s3=0.006,
    )

    assert out.use_geom_init.tolist() == [True]
    assert captured["quadric_gamma"] == 1.73
    assert captured["quadric_kappa_max"] == 7.0
    assert captured["quadric_eps_lambda"] == 0.02
    assert captured["quadric_eps_kappa"] == 0.004
    assert captured["quadric_eps_s"] == 0.005
    assert captured["quadric_eps_s3"] == 0.006


def test_anchor_token_normalizes_only_ptv3_input_values():
    c_init = torch.tensor([[100.0, -100.0, 25.0]])
    R_init = torch.eye(3).view(1, 3, 3)
    s_init = torch.tensor([[1e-4, 10.0, 0.5]])
    fit_quality = torch.tensor([[5.0, 0.25, 10.0, 1.0]])

    token = build_anchor_token(
        c_init=c_init,
        R_init=R_init,
        s_init=s_init,
        kappa1=torch.tensor([20.0]),
        kappa2=torch.tensor([-20.0]),
        fit_quality=fit_quality,
        tangent_aniso=torch.tensor([10.0]),
        curvature_aniso=torch.tensor([0.5]),
        i_mean=torch.tensor([1.5]),
        i_std=torch.tensor([-0.5]),
        token_variant="full",
    )

    assert token.shape == (1, 22)
    assert torch.isfinite(token).all()
    assert token[:, :3].abs().max() <= 2.0
    assert token[:, 3:9].abs().max() <= 1.0
    assert token[:, 9:12].abs().max() <= 2.0
    assert token[:, 12:14].abs().max() <= 2.0
    assert token[:, 14:18].min() >= 0.0
    assert token[:, 14:18].max() <= 2.0
    assert token[:, 18:20].min() >= 0.0
    assert token[:, 18:20].max() <= 2.0
    assert token[:, 20:22].tolist() == [[1.0, 0.0]]
    assert c_init.tolist() == [[100.0, -100.0, 25.0]]
    assert fit_quality.tolist() == [[5.0, 0.25, 10.0, 1.0]]


def test_anchor_token_with_normal_keeps_25d_width_for_empty_output():
    builder = VoxelAnchorBuilder(token_variant="with_normal")
    out = builder(torch.zeros(0, 3), torch.zeros(0), torch.zeros(0))

    assert out.token.shape == (0, 25)


def test_anchor_intensity_stats_use_masked_knn_support():
    intensity = torch.tensor([0.1, 0.3, 0.9])
    idx = torch.tensor([
        [0, 1, -1],
        [2, 0, 1],
    ])
    mask = idx >= 0

    mean, std = _masked_neighbor_intensity_stats(intensity, idx, mask)

    expected_mean0 = torch.tensor(0.2)
    expected_std0 = torch.tensor(0.1)
    expected_mean1 = torch.tensor((0.9 + 0.1 + 0.3) / 3.0)
    expected_std1 = torch.stack([
        (intensity[[2, 0, 1]] - expected_mean1).pow(2).mean().sqrt()
    ])[0]
    assert torch.allclose(mean, torch.stack([expected_mean0, expected_mean1]))
    assert torch.allclose(std, torch.stack([expected_std0, expected_std1]), atol=1e-6)
