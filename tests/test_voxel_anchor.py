from __future__ import annotations

import torch

from models.geometry.cartesian_voxel import CartesianVoxelizer
from models.geometry.spherical_voxel import SphericalVoxelizer
from models.geometry.voxel_anchor import VoxelAnchorBuilder, build_anchor_token


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
