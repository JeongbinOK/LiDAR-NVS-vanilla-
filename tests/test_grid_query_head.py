import math
from types import SimpleNamespace
import unittest

import torch

from src.models_new.module.builders.common import (
    _aggregate_points_to_cells,
    aggregate_points_to_cells_with_membership,
    aggregate_points_to_cells_with_seeds,
    counts_to_variable_k,
)
from src.models_new.module.builders.grid_intensity import OccupiedGridTokenBuilder
from src.models_new.module.gaussian_assembly import (
    assemble_batch_gaussians,
    gradient_scale_identity,
)
from src.models_new.module.grid_query_head import (
    GridSlotHead,
    GridTemporalAggregator,
    ViewTokenPositionEncoder,
    _radius_pairs,
)
from src.models_new.module.m3_g2p import GausRender
from src.models_new.utils.loss import Loss


ATTN_DIM = 12  # 2 heads -> head_dim 6, matching Utonia 3D RoPE's %6 contract.


def _cfg(**overrides):
    values = {
        "K_max": 3,
        "points_per_gaussian": 4,
        "bg_radius_m": 0.8,
        "bg_max_kv_per_frame": 8,
        "num_heads": 2,
        "bg_layers": 2,
        "fg_layers": 2,
        "grad_balance": "sqrt_k",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _gs_params(**overrides):
    values = {
        "shs": 2,
        "opacity": 1,
        "scaling": 1,
        "rotation": 1,
        "offset": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _UnitMapper:
    coord_scale = 1.0
    feature_grid_size = 1.0

    @staticmethod
    def to_feature_grid(points, metric_origin):
        return points - metric_origin

    @staticmethod
    def metric_origin(input_coord, input_grid_coord):
        return input_coord.new_zeros(3)


def _aggregate_seed_cell(
    points,
    points_per_gaussian,
    k_max=3,
    voxel_coord=None,
    exp=None,
    count_mode="legacy",
    seed_mode=None,
):
    intensity = torch.arange(1, points.shape[0] + 1, dtype=points.dtype)
    grid_coord = torch.tensor([[0, 0, 0]])
    voxel_feature = torch.tensor([[3.0, 4.0]])
    if voxel_coord is None:
        voxel_coord = torch.tensor([[0.5, 0.5, 0.5]])
    return aggregate_points_to_cells_with_seeds(
        points, intensity, grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(), points_per_gaussian, k_max, exp=exp,
        count_mode=count_mode, seed_mode=seed_mode,
    )


def _seed_inputs(anchor, k_max=1):
    seed = anchor[:, None, :].expand(-1, k_max, -1).clone()
    return seed, torch.zeros_like(seed)


def _builder_cfg(anchor_mode, **grid_overrides):
    return SimpleNamespace(
        anchor_mode=anchor_mode,
        grid_query=_cfg(**grid_overrides),
        intensity_encoder=SimpleNamespace(type="mlp"),
        int_proj=SimpleNamespace(in_dim=5, out_dim=4),
        r_far=80.0,
    )


def _learned_cfg(k_max=4, tau=1.0, **overrides):
    return _cfg(
        count_mode="learned_gumbel",
        learned_count=SimpleNamespace(
            K_max=k_max,
            tau=tau,
            seed_mode="range_quantile",
        ),
        **overrides,
    )


def _viewpoint_cfg(k_max=4, tau=1.0, **overrides):
    return _cfg(
        count_mode="learned_gumbel_viewpt",
        learned_count=SimpleNamespace(
            K_max=k_max,
            tau=tau,
            seed_mode="range_quantile",
            viewpoint=SimpleNamespace(
                position_frequencies=8,
                position_scale_m=110.0,
            ),
        ),
        **overrides,
    )


def _empty_scene(num_frames):
    pose = [torch.eye(4).repeat(num_frames, 1, 1)]
    bbox = [[torch.empty(0, 7) for _ in range(num_frames)]]
    bbox_iids = [[torch.empty(0, dtype=torch.long) for _ in range(num_frames)]]
    timestamps = [torch.linspace(0.0, 1.0, num_frames)]
    return pose, bbox, bbox_iids, timestamps


def test_raw_count_to_k_contract():
    raw_count = torch.tensor([1, 4, 5, 8, 9, 100])
    actual = counts_to_variable_k(raw_count, points_per_gaussian=4, k_max=3)
    assert actual.tolist() == [1, 1, 2, 2, 3, 3]


def test_builder_raw_count_is_aligned_with_occupied_token_order():
    class Mapper:
        coord_scale = 1.0
        feature_grid_size = 1.0

        @staticmethod
        def to_feature_grid(points, metric_origin):
            return points - metric_origin

    points = torch.tensor([
        [0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1],
    ])
    intensity = torch.tensor([1.0, 3.0, 7.0])
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    voxel_feature = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    voxel_coord = torch.tensor([
        [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5],
    ])
    pos, feature, _, occupied, raw_count = _aggregate_points_to_cells(
        points, intensity, grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), Mapper(),
    )
    assert occupied.tolist() == [True, True, False]
    assert raw_count.tolist() == [2, 1]
    torch.testing.assert_close(feature, voxel_feature[:2])
    torch.testing.assert_close(pos, voxel_coord[:2])


def test_grid_builder_returns_frame_local_seeds_and_spherical_placeholder():
    lidar_points = torch.tensor([
        [0.10, 0.1, 0.1, 1.0], [0.20, 0.1, 0.1, 2.0],
        [0.70, 0.1, 0.1, 3.0], [0.80, 0.1, 0.1, 4.0],
    ])
    offset = torch.tensor([2, 4])
    features = {
        "feat": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "grid_coord": torch.tensor([[0, 0, 0], [0, 0, 0]]),
        "coord": torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.4, 0.4]]),
        "offset": torch.tensor([1, 2]),
    }
    ptv3_input = {
        "coord": torch.tensor([[0.1, 0.1, 0.1], [0.7, 0.1, 0.1]]),
        "grid_coord": torch.tensor([[0, 0, 0], [0, 0, 0]]),
        "offset": torch.tensor([1, 2]),
    }

    grid_out = OccupiedGridTokenBuilder(_builder_cfg("grid"))(
        lidar_points, offset, None, features, ptv3_input, _UnitMapper()
    )
    seed_data = grid_out.grid_seeds
    assert grid_out.raw_memberships is None
    assert len(seed_data) == 2
    torch.testing.assert_close(seed_data[0].seed_sensor[0, 0], lidar_points[1, :3])
    torch.testing.assert_close(seed_data[1].seed_sensor[0, 0], lidar_points[3, :3])
    assert seed_data[0].anchor_k.tolist() == [1]
    assert seed_data[1].anchor_k.tolist() == [1]

    spherical_out = OccupiedGridTokenBuilder(_builder_cfg("spherical"))(
        lidar_points, offset, None, features, ptv3_input, _UnitMapper()
    )
    assert spherical_out.grid_seeds is None
    assert len(spherical_out.raw_memberships) == 2
    for frame, membership in enumerate(spherical_out.raw_memberships):
        start = 2 * frame
        torch.testing.assert_close(
            membership.points_sensor, lidar_points[start:start + 2, :3]
        )
        assert membership.token_index.tolist() == [0, 0]


def test_raw_membership_preserves_input_points_and_compact_token_rows():
    points = torch.tensor([
        [1.10, 0.1, 0.1], [0.10, 0.1, 0.1],
        [1.20, 0.1, 0.1], [0.20, 0.1, 0.1],
    ])
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    voxel_feature = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    voxel_coord = torch.tensor([
        [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5],
    ])
    result = aggregate_points_to_cells_with_membership(
        points, torch.ones(4), grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(),
    )
    _, _, _, occupied, raw_count, membership = result

    assert occupied.tolist() == [True, True, False]
    assert raw_count.tolist() == [2, 2]
    torch.testing.assert_close(membership.points_sensor, points)
    assert membership.token_index.tolist() == [1, 0, 1, 0]


def test_r_quantile_seeds_for_k1_k2_k3():
    points = torch.tensor([
        [0.05, 0.1, 0.1],
        [0.15, 0.1, 0.1],
        [0.25, 0.1, 0.1],
        [0.35, 0.1, 0.1],
        [0.45, 0.1, 0.1],
        [0.55, 0.1, 0.1],
    ])
    seed_k1 = _aggregate_seed_cell(points, 6)[-1].seed_sensor
    seed_k2 = _aggregate_seed_cell(points, 3)[-1].seed_sensor
    seed_k3 = _aggregate_seed_cell(points, 2)[-1].seed_sensor

    torch.testing.assert_close(seed_k1[0, 0], points[3])
    torch.testing.assert_close(seed_k2[0, :2], points[[1, 4]])
    torch.testing.assert_close(seed_k3[0], points[[1, 3, 5]])
    torch.testing.assert_close(seed_k1[0, 1:], torch.zeros(2, 3))
    torch.testing.assert_close(seed_k2[0, 2], torch.zeros(3))


def test_exp1_uses_one_observed_centroid_medoid_per_token_independent_of_raw_count():
    points = torch.tensor([
        [0.05, 0.1, 0.1], [0.15, 0.1, 0.1], [0.25, 0.1, 0.1],
        [0.35, 0.1, 0.1], [0.45, 0.1, 0.1],
    ])
    seed_data = _aggregate_seed_cell(
        points, points_per_gaussian=1, k_max=3, exp=1
    )[-1]

    # The mean is x=0.25, so the observed medoid is that raw point, not a
    # synthetic cell center or an extra raw-count-driven slot.
    assert seed_data.anchor_k.tolist() == [1]
    torch.testing.assert_close(seed_data.seed_sensor[0, 0], points[2])
    torch.testing.assert_close(seed_data.seed_sensor[0, 1:], torch.zeros(2, 3))


def test_exp2_uses_two_identical_token_coordinate_seeds_with_two_slots():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]])
    token_coord = torch.tensor([[0.2, 0.3, 0.4]])
    seed_data = _aggregate_seed_cell(
        points, points_per_gaussian=99, k_max=3, voxel_coord=token_coord, exp=2
    )[-1]

    assert seed_data.anchor_k.tolist() == [2]
    torch.testing.assert_close(seed_data.seed_sensor[0, :2], token_coord.expand(2, -1))
    torch.testing.assert_close(
        seed_data.delta_sensor[0, :2],
        (token_coord - torch.tensor([[0.5, 0.5, 0.5]])).expand(2, -1),
    )
    torch.testing.assert_close(seed_data.seed_sensor[0, 2], torch.zeros(3))


def test_learned_count_builds_k1_to_k4_range_quantile_seed_bank():
    points = torch.tensor([
        [0.05, 0.1, 0.1], [0.15, 0.1, 0.1],
        [0.25, 0.1, 0.1], [0.35, 0.1, 0.1],
        [0.45, 0.1, 0.1], [0.55, 0.1, 0.1],
        [0.65, 0.1, 0.1], [0.75, 0.1, 0.1],
    ])
    seed_data = _aggregate_seed_cell(
        points,
        points_per_gaussian=None,
        k_max=4,
        count_mode="learned_gumbel",
        seed_mode="range_quantile",
    )[-1]

    assert seed_data.anchor_k is None
    assert seed_data.seed_sensor.shape == (1, 4, 4, 3)
    # N=8: K1->[4], K2->[2,6], K3->[1,4,6], K4->[1,3,5,7].
    expected_rows = ([4], [2, 6], [1, 4, 6], [1, 3, 5, 7])
    for candidate_index, rows in enumerate(expected_rows):
        k = candidate_index + 1
        torch.testing.assert_close(
            seed_data.seed_sensor[0, candidate_index, :k],
            points[torch.tensor(rows)],
        )
        torch.testing.assert_close(
            seed_data.delta_sensor[0, candidate_index, :k],
            points[torch.tensor(rows)] - torch.tensor([0.5, 0.5, 0.5]),
        )
        torch.testing.assert_close(
            seed_data.seed_sensor[0, candidate_index, k:],
            torch.zeros(4 - k, 3),
        )


def test_viewpoint_seed_bank_uses_common_medoid_and_additional_quantiles():
    points = torch.tensor([
        [0.05, 0.1, 0.1], [0.15, 0.1, 0.1], [0.25, 0.1, 0.1],
        [0.35, 0.1, 0.1], [0.45, 0.1, 0.1],
    ])
    seed_data = _aggregate_seed_cell(
        points,
        points_per_gaussian=None,
        k_max=4,
        count_mode="learned_gumbel_viewpt",
        seed_mode="range_quantile",
    )[-1]

    assert seed_data.anchor_k is None
    assert seed_data.seed_sensor.shape == (1, 4, 4, 3)
    # The observed point nearest the raw-point mean is Common for every K.
    torch.testing.assert_close(
        seed_data.seed_sensor[0, :, 0], points[2].expand(4, -1)
    )
    # K_total=4 has three Additional range quantiles at ranks [0, 2, 4].
    torch.testing.assert_close(
        seed_data.seed_sensor[0, 3, 1:4], points[[0, 2, 4]]
    )


def test_viewpoint_additional_quantiles_repeat_observed_points_when_sparse():
    points = torch.tensor([[0.10, 0.1, 0.1], [0.90, 0.1, 0.1]])
    seed_data = _aggregate_seed_cell(
        points,
        points_per_gaussian=None,
        k_max=4,
        count_mode="learned_gumbel_viewpt",
        seed_mode="range_quantile",
    )[-1]
    # Three Additional slots from two observations use ranks [0, 1, 1].
    torch.testing.assert_close(
        seed_data.seed_sensor[0, 3, 1:4], points[[0, 1, 1]]
    )


def test_grid_builder_exp_switches_only_grid_seed_construction():
    grid_builder = OccupiedGridTokenBuilder(_builder_cfg("grid", exp=2))
    assert grid_builder.grid_seed_config == ("legacy", 4, 3, 2, None)
    learned_builder = OccupiedGridTokenBuilder(_builder_cfg(
        "grid",
        count_mode="learned_gumbel",
        # These invalid legacy values prove this branch does not read them.
        K_max=-1,
        points_per_gaussian=-1,
        exp=99,
        learned_count=SimpleNamespace(
            K_max=4, tau=1.0, seed_mode="range_quantile",
        ),
    ))
    assert learned_builder.grid_seed_config == (
        "learned_gumbel", None, 4, None, "range_quantile",
    )
    spherical_builder = OccupiedGridTokenBuilder(_builder_cfg("spherical", exp=2))
    assert spherical_builder.grid_seed_config is None


def test_equal_range_ties_use_xyz_order_and_ignore_input_permutation():
    points = torch.tensor([
        [0.10, 0.20, 0.0],
        [0.20, 0.10, 0.0],
    ])
    permuted = points[[1, 0]]
    seed_a = _aggregate_seed_cell(points, 2, k_max=1)[-1].seed_sensor
    seed_b = _aggregate_seed_cell(permuted, 2, k_max=1)[-1].seed_sensor

    # Equal ranges sort by x then y then z; rank floor(0.5 * 2)=1 selects row 1.
    torch.testing.assert_close(seed_a[0, 0], torch.tensor([0.20, 0.10, 0.0]))
    torch.testing.assert_close(seed_b, seed_a)


def test_seed_selection_is_isolated_per_frame():
    frame0 = torch.tensor([[0.10, 0.1, 0.1], [0.20, 0.1, 0.1]])
    frame1 = torch.tensor([[0.70, 0.1, 0.1], [0.80, 0.1, 0.1]])
    seed0 = _aggregate_seed_cell(frame0, 2, k_max=1)[-1].seed_sensor
    seed1 = _aggregate_seed_cell(frame1, 2, k_max=1)[-1].seed_sensor

    torch.testing.assert_close(seed0[0, 0], frame0[1])
    torch.testing.assert_close(seed1[0, 0], frame1[1])
    assert not torch.equal(seed0, seed1)


def test_seed_rows_follow_occupied_token_order_across_cells():
    points = torch.tensor([
        [1.10, 0.1, 0.1], [0.10, 0.1, 0.1],
        [1.20, 0.1, 0.1], [0.20, 0.1, 0.1],
    ])
    grid_coord = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    voxel_feature = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    voxel_coord = torch.tensor([
        [0.4, 0.5, 0.5], [1.4, 0.5, 0.5], [2.4, 0.5, 0.5],
    ])
    result = aggregate_points_to_cells_with_seeds(
        points, torch.ones(4), grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(), points_per_gaussian=2, k_max=2,
    )
    _, feature, _, occupied, raw_count, seed_data = result
    seeds = seed_data.seed_sensor

    assert occupied.tolist() == [True, True, False]
    assert raw_count.tolist() == [2, 2]
    torch.testing.assert_close(feature, voxel_feature[:2])
    torch.testing.assert_close(seeds[:, 0], torch.stack([points[3], points[2]]))


def test_delta_uses_geometric_center_not_pooled_position_and_is_padded():
    points = torch.tensor([[0.1, 0.1, 0.1], [0.8, 0.6, 0.4]])
    pooled_position = torch.tensor([[0.2, 0.2, 0.2]])
    pos, _, _, _, _, seed_data = _aggregate_seed_cell(
        points, 2, k_max=3, voxel_coord=pooled_position
    )
    seeds = seed_data.seed_sensor
    delta = seed_data.delta_sensor

    torch.testing.assert_close(pos, pooled_position)
    torch.testing.assert_close(seeds[0, 0], points[1])
    torch.testing.assert_close(delta[0, 0], points[1] - torch.tensor([0.5, 0.5, 0.5]))
    assert not torch.equal(delta[0, 0], points[1] - pooled_position[0])
    torch.testing.assert_close(seeds[0, 1:], torch.zeros(2, 3))
    torch.testing.assert_close(delta[0, 1:], torch.zeros(2, 3))


def test_variable_slot_packing_and_frame_offsets():
    torch.manual_seed(4)
    head = GridSlotHead(_cfg(), _gs_params(), dim=8)
    raw_count = torch.tensor([1, 4, 5, 8, 9, 100])
    anchor_k = counts_to_variable_k(raw_count, 4, 3)
    feature = torch.randn(6, 8)
    delta = torch.randn(6, 3, 3)
    raw_params, packing = head(
        feature, anchor_k, delta, torch.tensor([3, 6])
    )

    assert packing["anchor_k"].tolist() == [1, 1, 2, 2, 3, 3]
    assert packing["gaussian_offset"].tolist() == [4, 12]
    assert raw_params.shape == (12, 6)
    assert packing["anchor_index"].tolist() == [
        0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5,
    ]
    assert packing["slot_index"].tolist() == [
        0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2,
    ]

    anchor_frame = torch.tensor([0, 0, 0, 1, 1, 1])
    gaussian_frame = anchor_frame[packing["anchor_index"]]
    assert gaussian_frame.tolist() == [0] * 4 + [1] * 8
    metadata = torch.tensor([10, 11, 12, 20, 21, 22])
    assert metadata[packing["anchor_index"]].tolist() == [
        10, 11, 12, 12, 20, 20, 21, 21, 21, 22, 22, 22,
    ]


def test_k_heads_derive_the_production_parameter_width_from_gs_params():
    production_params = _gs_params(shs=32, scaling=2, rotation=4, offset=3)
    head = GridSlotHead(_cfg(), production_params, dim=8)
    assert head.param_dim == 42
    assert [module.out_features for module in head.k_heads] == [42, 84, 126]


def test_learned_count_adds_k4_head_and_post_fusion_mlp_only_in_new_mode():
    legacy = GridSlotHead(_cfg(), _gs_params(), dim=8)
    learned = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)

    assert not hasattr(legacy, "count_predictor")
    assert len(legacy.k_heads) == 3
    assert learned.count_predictor[-1].out_features == 4
    assert [module.out_features for module in learned.k_heads] == [6, 12, 18, 24]
    assert learned.trunk[0].in_features == 8 + 4 * 3


def test_view_token_position_encoder_inverts_the_target_pose_per_token():
    encoder = ViewTokenPositionEncoder(SimpleNamespace(
        position_frequencies=8,
        position_scale_m=110.0,
    ))
    assert encoder.out_dim == 48

    # T_(0<-t): the target sensor sits at (2, 0, 0) rotated +90 deg about z.
    pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    pose[1, :3, 3] = torch.tensor([2.0, 0.0, 0.0])
    pose[1, :3, :3] = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    position_ref = torch.tensor([[5.0, 1.0, 0.5], [5.0, 1.0, 0.5]])
    view_position = encoder.to_view_frame(position_ref, pose)
    torch.testing.assert_close(view_position[0], position_ref[0])
    # R^T (p - t) = R^T (3, 1, 0.5) = (1, -3, 0.5)
    torch.testing.assert_close(
        view_position[1], torch.tensor([1.0, -3.0, 0.5])
    )
    assert encoder(position_ref, pose).shape == (2, 48)
    # The same viewpoint now yields different embeddings for different tokens.
    assert not torch.equal(
        encoder(position_ref, pose)[0], encoder(position_ref, pose)[1]
    )


def test_view_token_position_bands_do_not_wrap_inside_the_sensing_range():
    encoder = ViewTokenPositionEncoder(SimpleNamespace(
        position_frequencies=8,
        position_scale_m=110.0,
    ))
    x = torch.linspace(-109.0, 109.0, 25)
    positions = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
    identity = torch.eye(4).unsqueeze(0).repeat(x.numel(), 1, 1)
    encoded = encoder(positions, identity)

    # Layout is [ch_x sin(band 0..7), ch_x cos(band 0..7), ch_y ...], so band 0
    # of x is (encoded[:, 0], encoded[:, 8]). Its 220 m period covers the whole
    # +-110 m sensing range exactly once: atan2 recovers the position without a
    # wrap, so no two distances collide in the embedding.
    recovered = torch.atan2(encoded[:, 0], encoded[:, 8]) * 110.0 / torch.pi
    torch.testing.assert_close(recovered, x, atol=1e-3, rtol=0.0)


def test_viewpoint_head_has_agreed_240_to_384_to_384_to_192_mlp():
    head = GridSlotHead(_viewpoint_cfg(), _gs_params(), dim=192)
    assert isinstance(head.view_mlp[0], torch.nn.LayerNorm)
    assert head.view_mlp[0].normalized_shape == (240,)
    assert (head.view_mlp[1].in_features, head.view_mlp[1].out_features) == (240, 384)
    assert (head.view_mlp[3].in_features, head.view_mlp[3].out_features) == (384, 384)
    assert (head.view_mlp[5].in_features, head.view_mlp[5].out_features) == (384, 192)


def test_viewpoint_k1_uses_common_only_and_render_loss_trains_router():
    class FixedK1Head(GridSlotHead):
        def _gumbel_selection(self, logits):
            soft = torch.softmax(logits / self.gumbel_tau, dim=-1)
            hard = torch.zeros_like(soft)
            hard[:, 0] = 1.0
            return hard - soft.detach() + soft

    torch.manual_seed(41)
    head = FixedK1Head(_viewpoint_cfg(tau=0.7), _gs_params(), dim=8)
    feature = torch.randn(1, 8, requires_grad=True)
    target_pose = [torch.eye(4).repeat(2, 1, 1)]
    target_pose[0][1, 0, 3] = 2.0
    raw_params, packing = head(
        feature,
        None,
        torch.randn(1, 4, 4, 3),
        torch.tensor([1]),
        target_pose=target_pose,
        anchor_batch=torch.tensor([0]),
        anchor_position_ref=torch.tensor([[6.0, 1.0, 0.5]]).expand(1, 2, 3),
    )

    assert packing["anchor_k"].tolist() == [1, 1]
    assert packing["view_index"].tolist() == [-1]
    assert raw_params.shape == (1, head.param_dim)
    assert packing["common_view_gate"].shape == (1, 2)
    packing["k_logits"].retain_grad()
    stored_opacity = raw_params[:, head.opacity_start:head.opacity_end]
    b_gs = {
        "view_index": packing["view_index"],
        "opacity": stored_opacity,
        "common_view_gate": packing["common_view_gate"],
    }
    loss = stored_opacity.new_zeros(())
    for view_index in range(2):
        selected = GausRender.select_target_view(b_gs, view_index)
        assert selected["view_index"].tolist() == [-1]
        torch.testing.assert_close(selected["opacity"], stored_opacity)
        loss = loss + torch.sigmoid(selected["opacity"]).sum()
    loss.backward()

    assert head.common_predictor[-1].weight.grad is not None
    assert head.common_predictor[-1].weight.grad.abs().sum() > 0
    assert head.count_predictor[-1].weight.grad is not None
    assert head.count_predictor[-1].weight.grad.abs().sum() > 0
    assert packing["k_logits"].grad is not None
    assert packing["k_logits"].grad.abs().sum() > 0
    assert head.additional_trunk[0].weight.grad is None
    assert all(module.weight.grad is None for module in head.additional_heads)


def test_viewpoint_mixed_k_packs_common_once_and_preserves_per_view_total_k():
    class FixedMixedKHead(GridSlotHead):
        def _gumbel_selection(self, logits):
            selected = torch.tensor([1, 3, 2, 4], device=logits.device) - 1
            soft = torch.softmax(logits / self.gumbel_tau, dim=-1)
            hard = torch.zeros_like(soft).scatter_(1, selected[:, None], 1.0)
            return hard - soft.detach() + soft

    head = FixedMixedKHead(_viewpoint_cfg(), _gs_params(), dim=8)
    raw_params, packing = head(
        torch.randn(2, 8),
        None,
        torch.randn(2, 4, 4, 3),
        torch.tensor([2]),
        target_pose=[torch.eye(4).repeat(2, 1, 1)],
        anchor_batch=torch.tensor([0, 0]),
        anchor_position_ref=torch.tensor(
            [[6.0, 1.0, 0.5], [9.0, -2.0, 0.0]]
        ).unsqueeze(1).expand(2, 2, 3),
    )

    # Old packing had sum(K)=10 rows. Shared Common uses N+sum(K-1)=8.
    assert raw_params.shape == (8, head.param_dim)
    assert packing["anchor_k"].tolist() == [1, 3, 2, 4]
    assert packing["view_index"].tolist() == [-1, 1, 1, -1, 0, 1, 1, 1]
    assert packing["slot_index"].tolist() == [0, 1, 2, 0, 1, 1, 2, 3]
    assert packing["gaussian_offset"].tolist() == [8]

    b_gs = {
        "view_index": packing["view_index"],
        "opacity": raw_params[:, head.opacity_start:head.opacity_end],
        "common_view_gate": packing["common_view_gate"],
    }
    grouped = GausRender.group_target_views(b_gs, 2)
    # View 0 sees K=[1,2], view 1 sees K=[3,4].
    assert grouped[0].numel() == 3
    assert grouped[1].numel() == 7


class _FixedKHead(GridSlotHead):
    """Route every decision to a scripted K while keeping the soft backward."""

    selected_k = None

    def _gumbel_selection(self, logits):
        selected = torch.as_tensor(
            self.selected_k, device=logits.device, dtype=torch.long
        ) - 1
        soft = torch.softmax(logits / self.gumbel_tau, dim=-1)
        hard = torch.zeros_like(soft).scatter_(1, selected[:, None], 1.0)
        return hard - soft.detach() + soft


def _viewpoint_forward(head, feature, position_ref, target_pose, delta_p=None):
    num_anchors = feature.shape[0]
    if delta_p is None:
        torch.manual_seed(7)
        delta_p = torch.randn(num_anchors, head.k_max, head.k_max, 3)
    if position_ref.ndim == 2:
        # Static tokens: one ref center shared by every target view.
        num_views = max(int(p.shape[0]) for p in target_pose)
        position_ref = position_ref.unsqueeze(1).expand(-1, num_views, -1)
    return head(
        feature,
        None,
        delta_p,
        torch.tensor([num_anchors]),
        target_pose=target_pose,
        anchor_batch=torch.zeros(num_anchors, dtype=torch.long),
        anchor_position_ref=position_ref,
    )


def test_viewpoint_router_sees_each_token_from_the_target_viewpoint():
    torch.manual_seed(3)
    head = GridSlotHead(_viewpoint_cfg(), _gs_params(), dim=8)
    # count_predictor's output layer is zero-initialised for a uniform K prior,
    # so give it a weight to make the router input observable in the logits.
    torch.nn.init.normal_(head.count_predictor[-1].weight, std=0.1)
    # Two tokens sharing one feature so only their positions can differ.
    feature = torch.randn(1, 8).repeat(2, 1)
    position_ref = torch.tensor([[4.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
    moved = torch.eye(4).repeat(2, 1, 1)
    moved[1, :3, 3] = torch.tensor([12.0, 0.0, 0.0])

    _, packing = _viewpoint_forward(head, feature, position_ref, [moved])
    logits = packing["k_logits"]
    # decision rows are (token0, view0), (token0, view1), (token1, view0), ...
    assert logits.shape == (4, head.k_max)
    # Same viewpoint, different token position -> different router input.
    assert not torch.allclose(logits[0], logits[2])
    # Same token, different viewpoint -> different router input.
    assert not torch.allclose(logits[0], logits[1])
    # A token 12 m in front of view 0 is the 4 m token seen from view 1, so the
    # router input is identical for those two decisions.
    _, shifted = _viewpoint_forward(
        head,
        feature,
        torch.tensor([[16.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),
        [moved],
    )
    torch.testing.assert_close(shifted["k_logits"][1], logits[0])


def test_viewpoint_additional_gaussians_do_not_depend_on_the_target_view():
    torch.manual_seed(11)
    head = _FixedKHead(_viewpoint_cfg(), _gs_params(), dim=8)
    head.selected_k = [3, 3]
    feature = torch.randn(1, 8)
    far_apart = torch.eye(4).repeat(2, 1, 1)
    far_apart[1, :3, 3] = torch.tensor([25.0, -8.0, 0.0])

    raw_params, packing = _viewpoint_forward(
        head, feature, torch.tensor([[7.0, 2.0, 0.5]]), [far_apart]
    )
    # 1 Common + 2 Additional for each of the two views.
    assert packing["view_index"].tolist() == [-1, 0, 0, 1, 1]
    # Both views routed to K=3, so their Additional parameters must be equal
    # even though the two viewpoints are 26 m apart.
    torch.testing.assert_close(raw_params[1:3], raw_params[3:5])


def test_viewpoint_router_feature_reaches_gaussians_only_through_the_gate():
    torch.manual_seed(19)
    head = _FixedKHead(_viewpoint_cfg(tau=0.7), _gs_params(), dim=8)
    head.selected_k = [1, 3]
    # A zero-initialised router output layer zeroes the chain rule into view_mlp
    # for exactly one step; measure the trained-state path instead.
    torch.nn.init.normal_(head.count_predictor[-1].weight, std=0.1)
    feature = torch.randn(1, 8, requires_grad=True)
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, 3] = torch.tensor([5.0, 0.0, 0.0])

    raw_params, packing = _viewpoint_forward(
        head, feature, torch.tensor([[9.0, 1.0, 0.5]]), [poses]
    )
    # Everything except opacity bypasses the router gate entirely.
    non_opacity = torch.cat([
        raw_params[:, :head.opacity_start], raw_params[:, head.opacity_end:],
    ], dim=-1)
    non_opacity.sum().backward(retain_graph=True)

    assert head.common_predictor[-1].weight.grad.abs().sum() > 0
    assert head.additional_trunk[0].weight.grad.abs().sum() > 0
    assert head.additional_heads[1].weight.grad.abs().sum() > 0
    assert feature.grad.abs().sum() > 0
    # The 192D view feature feeds the count router alone: no Gaussian parameter
    # other than the gated opacity may pull gradient through it. The gate's cat
    # keeps the router in the graph, so this must be zero rather than absent.
    assert head.view_mlp[1].weight.grad.abs().sum() == 0
    assert head.count_predictor[-1].weight.grad.abs().sum() == 0

    head.zero_grad(set_to_none=True)
    feature.grad = None
    packing["k_logits"].retain_grad()
    b_gs = {
        "view_index": packing["view_index"],
        "opacity": raw_params[:, head.opacity_start:head.opacity_end],
        "common_view_gate": packing["common_view_gate"],
    }
    loss = raw_params.new_zeros(())
    for view_index in range(2):
        selected = GausRender.select_target_view(b_gs, view_index)
        loss = loss + torch.sigmoid(selected["opacity"]).sum()
    loss.backward()

    # K=1 (view 0) trains the router through the deferred Common gate, K=3
    # (view 1) through its Additional opacities. Both must be live.
    assert packing["k_logits"].grad[0].abs().sum() > 0
    assert packing["k_logits"].grad[1].abs().sum() > 0
    assert head.view_mlp[1].weight.grad.abs().sum() > 0
    assert head.view_mlp[5].weight.grad.abs().sum() > 0
    assert head.count_predictor[-1].weight.grad.abs().sum() > 0
    assert head.common_predictor[-1].weight.grad.abs().sum() > 0
    assert head.additional_trunk[0].weight.grad.abs().sum() > 0
    assert head.additional_heads[1].weight.grad.abs().sum() > 0
    # The fused token feeds Common, Additional and the router, so it collects
    # gradient from all three paths.
    assert feature.grad.abs().sum() > 0


def test_viewpoint_zero_init_router_output_delays_view_mlp_by_one_step():
    """Document the only gradient that is not live at initialisation.

    ``count_predictor[-1]`` is zeroed so step 0 starts from a uniform K policy.
    Because view_mlp now reaches the loss only through those logits, its own
    gradient is zero until that layer moves off zero -- which its own non-zero
    gradient guarantees after the first optimizer step.
    """
    torch.manual_seed(23)
    head = _FixedKHead(_viewpoint_cfg(tau=0.7), _gs_params(), dim=8)
    head.selected_k = [2, 3]
    feature = torch.randn(1, 8, requires_grad=True)
    raw_params, packing = _viewpoint_forward(
        head, feature, torch.tensor([[9.0, 1.0, 0.5]]),
        [torch.eye(4).repeat(2, 1, 1)],
    )
    torch.testing.assert_close(
        packing["k_logits"], torch.zeros_like(packing["k_logits"])
    )
    raw_params[:, head.opacity_start:head.opacity_end].sum().backward()

    assert head.view_mlp[5].weight.grad.abs().sum() == 0
    assert head.count_predictor[-1].weight.grad.abs().sum() > 0


def test_renderer_selects_only_the_requested_target_union():
    b_gs = {
        "view_index": torch.tensor([-1, 0, 1, 0, 1]),
        "position": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "opacity": torch.arange(5, dtype=torch.float32).unsqueeze(-1),
        "scaling": torch.ones(5, 2),
        "rotation": torch.ones(5, 4),
        "shs": torch.ones(5, 32),
        "instance_id": torch.full((5,), -1, dtype=torch.long),
        "is_dynamic": torch.zeros(5, dtype=torch.bool),
        "common_view_gate": torch.ones(1, 2),
        "fg_masks": {-1: torch.zeros(5, dtype=torch.bool)},
        "object_trajectories": {},
    }
    grouped = GausRender.group_target_views(b_gs, 2)
    selected = GausRender.select_target_view(b_gs, 1, grouped[1])
    assert selected["view_index"].tolist() == [-1, 1, 1]
    torch.testing.assert_close(selected["position"], b_gs["position"][[0, 2, 4]])
    torch.testing.assert_close(selected["opacity"], b_gs["opacity"][[0, 2, 4]])
    assert selected["fg_masks"][-1].shape == (3,)


def test_assembly_keeps_one_common_gate_table_per_batch():
    gs_raw = {
        "opacity": torch.arange(6, dtype=torch.float32).unsqueeze(-1),
    }
    out_coord = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    agg_meta = {
        "box_assign": torch.full((6,), -1, dtype=torch.long),
        "instance_id": torch.full((6,), -1, dtype=torch.long),
        "is_dynamic": torch.zeros(6, dtype=torch.bool),
        "coord_ref": out_coord.clone(),
        "view_index": torch.tensor([-1, 0, 1, -1, 0, 1]),
        "anchor_index": torch.tensor([0, 0, 0, 1, 1, 1]),
        "common_view_gate": torch.tensor([[1.0, 0.5], [0.25, 1.0]]),
        "bbox_ref_by_frame": [torch.empty(0, 7), torch.empty(0, 7)],
    }
    batch = assemble_batch_gaussians(
        gs_raw,
        out_coord,
        agg_meta,
        torch.tensor([3, 6]),
        [0, 1],
        [torch.empty(0, 7), torch.empty(0, 7)],
        [torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
        True,
        out_coord.device,
    )

    assert len(batch) == 2
    torch.testing.assert_close(
        batch[0]["common_view_gate"], torch.tensor([[1.0, 0.5]])
    )
    torch.testing.assert_close(
        batch[1]["common_view_gate"], torch.tensor([[0.25, 1.0]])
    )


def test_shared_common_scale_regularization_matches_physical_view_repetition():
    loss_fn = Loss(SimpleNamespace(
        w_chamfer=0.0,
        w_depth=0.0,
        w_depth_median=0.0,
        w_intensity=0.0,
        w_raydrop=0.0,
        w_scale=1.0,
        enable_lpips=False,
    ))
    reference = torch.zeros(())

    common_expanded = torch.tensor([[3.0, 3.0]], requires_grad=True)
    additional_expanded = torch.tensor(
        [[4.0, 4.0], [5.0, 5.0]], requires_grad=True
    )
    expanded_scaling = torch.cat([
        common_expanded,
        additional_expanded[0:1],
        common_expanded,
        additional_expanded[1:2],
    ])
    expanded_loss = loss_fn._scale_regularization(
        [{"scaling": expanded_scaling}], reference
    )
    expanded_loss.backward()

    common_shared = torch.tensor([[3.0, 3.0]], requires_grad=True)
    additional_shared = torch.tensor(
        [[4.0, 4.0], [5.0, 5.0]], requires_grad=True
    )
    shared_loss = loss_fn._scale_regularization([{
        "scaling": torch.cat([common_shared, additional_shared]),
        "view_index": torch.tensor([-1, 0, 1]),
        "common_view_gate": torch.ones(1, 2),
    }], reference)
    shared_loss.backward()

    torch.testing.assert_close(shared_loss, expanded_loss)
    torch.testing.assert_close(common_shared.grad, common_expanded.grad)
    torch.testing.assert_close(additional_shared.grad, additional_expanded.grad)


def test_scale_loss_is_computed_but_not_added_when_weight_is_zero():
    loss_fn = Loss(SimpleNamespace(
        w_chamfer=0.0,
        w_depth=0.0,
        w_depth_median=0.0,
        w_intensity=0.0,
        w_raydrop=0.0,
        w_scale=0.0,
        scale_max_m=2.5,
        enable_lpips=False,
    ))
    raw_scaling = torch.tensor([[4.0, 4.0]], requires_grad=True)
    raw_loss = loss_fn._scale_regularization(
        [{"scaling": raw_scaling}], torch.zeros(())
    )
    assert float(raw_loss) > 0.0

    total = loss_fn.w_scale * raw_loss
    total.backward()
    torch.testing.assert_close(raw_scaling.grad, torch.zeros_like(raw_scaling))


def test_learned_count_eval_argmax_routes_exactly_to_k2_joint_head():
    head = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)
    head.eval()
    with torch.no_grad():
        head.count_predictor[-1].weight.zero_()
        head.count_predictor[-1].bias.copy_(
            torch.tensor([-3.0, 5.0, -2.0, -1.0])
        )
        for k, k_head in enumerate(head.k_heads, start=1):
            k_head.weight.zero_()
            bias = torch.arange(k, dtype=torch.float32).add_(10 * k)
            k_head.bias.copy_(
                bias[:, None].expand(k, head.param_dim).reshape(-1)
            )

    raw_params, packing = head(
        torch.randn(2, 8),
        None,
        torch.zeros(2, 4, 4, 3),
        torch.tensor([2]),
    )

    assert packing["anchor_k"].tolist() == [2, 2]
    assert packing["gaussian_offset"].tolist() == [4]
    torch.testing.assert_close(
        packing["k_selection"],
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]).expand(2, -1),
    )
    expected = torch.tensor([20.0, 21.0, 20.0, 21.0])
    torch.testing.assert_close(
        raw_params, expected[:, None].expand(-1, head.param_dim)
    )


def test_learned_count_selected_opacity_gate_trains_router_and_only_k3_head():
    class FixedK3Head(GridSlotHead):
        def _gumbel_selection(self, logits):
            soft = torch.softmax(logits / self.gumbel_tau, dim=-1)
            hard = torch.zeros_like(soft)
            hard[:, 2] = 1.0
            return hard - soft.detach() + soft

    torch.manual_seed(17)
    head = FixedK3Head(_learned_cfg(tau=0.7), _gs_params(), dim=8)
    feature = torch.randn(1, 8, requires_grad=True)
    raw_params, packing = head(
        feature,
        None,
        torch.randn(1, 4, 4, 3),
        torch.tensor([1]),
    )

    assert packing["anchor_k"].tolist() == [3]
    assert torch.all((packing["k_selection"] == 0) | (packing["k_selection"] == 1))
    assert torch.allclose(
        packing["k_selection"].sum(dim=-1),
        torch.ones(1),
    )
    torch.testing.assert_close(
        packing["packed_selected_gate"],
        torch.ones(3),
    )

    packing["k_logits"].retain_grad()
    opacity = raw_params[:, head.opacity_start:head.opacity_end]
    torch.sigmoid(opacity).sum().backward()

    predictor_grad = head.count_predictor[-1].weight.grad
    assert predictor_grad is not None and predictor_grad.abs().sum() > 0
    assert packing["k_logits"].grad is not None
    assert torch.all(packing["k_logits"].grad.abs() > 0)
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert head.k_heads[0].weight.grad is None
    assert head.k_heads[1].weight.grad is None
    assert head.k_heads[2].weight.grad is not None
    assert head.k_heads[2].weight.grad.abs().sum() > 0
    assert head.k_heads[3].weight.grad is None


def test_learned_count_handles_empty_frames():
    head = GridSlotHead(_learned_cfg(), _gs_params(), dim=8)
    raw_params, packing = head(
        torch.zeros(0, 8),
        None,
        torch.zeros(0, 4, 4, 3),
        torch.tensor([0, 0]),
    )

    assert raw_params.shape == (0, head.param_dim)
    assert packing["k_selection"].shape == (0, 4)
    assert packing["gaussian_offset"].tolist() == [0, 0]


def test_k_specific_heads_route_mixed_anchors_to_slot_flat_order():
    head = GridSlotHead(_cfg(), _gs_params(), dim=8)
    for k, k_head in enumerate(head.k_heads, start=1):
        torch.nn.init.zeros_(k_head.weight)
        bias = torch.arange(k, dtype=torch.float32).add_(10 * k)
        with torch.no_grad():
            k_head.bias.copy_(bias[:, None].expand(k, head.param_dim).reshape(-1))

    feature = torch.randn(4, 8)
    anchor_k = torch.tensor([2, 1, 3, 2])
    raw_params, packing = head(
        feature, anchor_k, torch.zeros(4, 3, 3), torch.tensor([4])
    )
    expected = torch.tensor([20, 21, 10, 30, 31, 32, 20, 21], dtype=torch.float32)
    torch.testing.assert_close(
        raw_params, expected[:, None].expand(-1, head.param_dim)
    )
    assert packing["anchor_index"].tolist() == [0, 0, 1, 2, 2, 2, 3, 3]
    assert packing["slot_index"].tolist() == [0, 1, 0, 0, 1, 2, 0, 1]


def test_k_head_backward_reaches_trunk_features_and_only_selected_head():
    torch.manual_seed(11)
    head = GridSlotHead(_cfg(), _gs_params(), dim=8)
    feature = torch.randn(2, 8, requires_grad=True)
    raw_params, _ = head(
        feature, torch.tensor([2, 2]), torch.randn(2, 3, 3), torch.tensor([2])
    )
    raw_params.square().sum().backward()

    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert head.trunk[0].weight.grad is not None
    assert head.trunk[0].weight.grad.abs().sum() > 0
    assert head.k_heads[0].weight.grad is None
    assert head.k_heads[1].weight.grad is not None
    assert head.k_heads[1].weight.grad.abs().sum() > 0
    assert head.k_heads[2].weight.grad is None


def test_gradient_balance_is_forward_identity_on_every_final_output():
    slot_k = torch.tensor([1.0, 2.0, 3.0])
    weight = slot_k.rsqrt()
    widths = {"position": 3, "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4}
    tensors = {
        name: torch.randn(3, width, requires_grad=True)
        for name, width in widths.items()
    }
    balanced = {
        name: gradient_scale_identity(value, weight)
        for name, value in tensors.items()
    }
    for name in tensors:
        torch.testing.assert_close(balanced[name], tensors[name])
    sum(value.sum() for value in balanced.values()).backward()
    for value in tensors.values():
        expected = weight[:, None].expand_as(value)
        torch.testing.assert_close(value.grad, expected)


def test_missing_grad_balance_scope_preserves_historical_output_balancing():
    head = GridSlotHead(_cfg(), _gs_params(), dim=8)
    assert head.grad_balance_scope == "output"
    torch.testing.assert_close(
        head.gradient_weight(torch.tensor([1, 2, 4]), torch.float32),
        torch.tensor([1.0, 2.0**-0.5, 0.5]),
    )


def test_token_grad_balance_scales_only_shared_token_path():
    torch.manual_seed(29)
    unbalanced = GridSlotHead(
        _cfg(K_max=4, grad_balance="none", grad_balance_scope="token"),
        _gs_params(),
        dim=8,
    )
    balanced = GridSlotHead(
        _cfg(K_max=4, grad_balance="sqrt_k", grad_balance_scope="token"),
        _gs_params(),
        dim=8,
    )
    balanced.load_state_dict(unbalanced.state_dict())

    base_feature = torch.randn(2, 8)
    feature_unbalanced = base_feature.clone().requires_grad_()
    feature_balanced = base_feature.clone().requires_grad_()
    delta = torch.randn(2, 4, 3)
    anchor_k = torch.tensor([4, 4])
    raw_unbalanced, _ = unbalanced(
        feature_unbalanced, anchor_k, delta, torch.tensor([2])
    )
    raw_balanced, packing = balanced(
        feature_balanced, anchor_k, delta, torch.tensor([2])
    )

    # The balancing operator is a backward-only identity.
    torch.testing.assert_close(raw_balanced, raw_unbalanced, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        balanced.gradient_weight(packing["slot_k"], raw_balanced.dtype),
        torch.ones(8),
    )

    raw_unbalanced.square().sum().backward()
    raw_balanced.square().sum().backward()

    # K-head parameters see the same training signal. Only the gradient
    # returning through the shared trunk/token feature is scaled by 1/sqrt(4).
    torch.testing.assert_close(
        balanced.k_heads[3].weight.grad,
        unbalanced.k_heads[3].weight.grad,
    )
    torch.testing.assert_close(
        balanced.k_heads[3].bias.grad,
        unbalanced.k_heads[3].bias.grad,
    )
    torch.testing.assert_close(
        balanced.trunk[0].weight.grad,
        0.5 * unbalanced.trunk[0].weight.grad,
    )
    torch.testing.assert_close(
        balanced.trunk[0].bias.grad,
        0.5 * unbalanced.trunk[0].bias.grad,
    )
    torch.testing.assert_close(
        feature_balanced.grad,
        0.5 * feature_unbalanced.grad,
    )


def test_token_grad_balance_does_not_scale_learned_router_gradient():
    class FixedK4Head(GridSlotHead):
        def _gumbel_selection(self, logits):
            soft = torch.softmax(logits / self.gumbel_tau, dim=-1)
            hard = torch.zeros_like(soft)
            hard[:, 3] = 1.0
            return hard - soft.detach() + soft

    torch.manual_seed(31)
    unbalanced = FixedK4Head(
        _learned_cfg(
            grad_balance="none",
            grad_balance_scope="token",
        ),
        _gs_params(),
        dim=8,
    )
    balanced = FixedK4Head(
        _learned_cfg(
            grad_balance="sqrt_k",
            grad_balance_scope="token",
        ),
        _gs_params(),
        dim=8,
    )
    balanced.load_state_dict(unbalanced.state_dict())

    base_feature = torch.randn(2, 8)
    delta = torch.randn(2, 4, 4, 3)
    outputs = []
    packings = []
    for head in (unbalanced, balanced):
        feature = base_feature.clone().requires_grad_()
        raw_params, packing = head(
            feature, None, delta, torch.tensor([2])
        )
        packing["k_logits"].retain_grad()
        opacity = raw_params[:, head.opacity_start:head.opacity_end]
        torch.sigmoid(opacity).sum().backward()
        outputs.append(raw_params.detach())
        packings.append(packing)

    torch.testing.assert_close(outputs[1], outputs[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        packings[1]["k_logits"].grad,
        packings[0]["k_logits"].grad,
    )
    torch.testing.assert_close(
        balanced.count_predictor[-1].weight.grad,
        unbalanced.count_predictor[-1].weight.grad,
    )
    torch.testing.assert_close(
        balanced.count_predictor[-1].bias.grad,
        unbalanced.count_predictor[-1].bias.grad,
    )


def test_spatial_hash_uses_exact_radius_and_per_source_frame_cap():
    query = torch.tensor([[0.0, 0.0, 0.0]])
    source = torch.tensor([
        [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0],
        [0.79, 0.0, 0.0], [0.81, 0.0, 0.0],
    ])
    q, s = _radius_pairs(query, source, radius=0.8, max_neighbors=3)
    assert q.tolist() == [0, 0, 0]
    assert s.tolist() == [0, 1, 2]


def test_grid_attention_uses_post_norm_residual_without_layerscale():
    torch.manual_seed(7)
    aggregator = GridTemporalAggregator(_cfg(), dim=ATTN_DIM, r_far=80.0).eval()
    cross = aggregator.bg_attention
    self_attn = aggregator.fg_attention

    for module in (cross, self_attn):
        assert not hasattr(module, "attn_scale")
        assert not hasattr(module, "ffn_scale")

    feature = torch.randn(4, ATTN_DIM)
    position = torch.randn(4, 3)
    pair_query = torch.tensor([0, 0, 1, 2, 3])
    pair_feature = torch.randn(pair_query.numel(), ATTN_DIM)
    pair_position = torch.randn(pair_query.numel(), 3)
    outputs = (
        cross(feature, pair_feature, pair_query, position, pair_position),
        self_attn(feature, position),
    )

    # Both stacks end with LayerNorm(x + FFN(x)), matching the spherical block.
    for output in outputs:
        torch.testing.assert_close(
            output.mean(dim=-1), torch.zeros(output.shape[0]), atol=1e-5, rtol=0.0
        )
        torch.testing.assert_close(
            output.var(dim=-1, unbiased=False),
            torch.ones(output.shape[0]), atol=2e-4, rtol=0.0,
        )


def test_background_excludes_same_frame_and_no_match_keeps_normalized_query():
    torch.manual_seed(5)
    aggregator = GridTemporalAggregator(_cfg(), dim=ATTN_DIM, r_far=80.0).eval()
    # Frame 0 rows 1 and 2 are close to each other, but have no other-frame match.
    anchor = torch.tensor([
        [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.1, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ])
    feat = torch.randn(4, ATTN_DIM)
    offset = torch.tensor([3, 4])
    frame_batch = torch.tensor([0, 0])
    pose, bbox, bbox_iids, timestamps = _empty_scene(2)
    seed, delta = _seed_inputs(anchor)

    metadata = aggregator._token_metadata(
        anchor, seed, delta, offset, frame_batch, pose, bbox, bbox_iids, timestamps
    )
    pair_query, pair_source = aggregator._background_pairs(metadata)
    assert pair_query.numel() == 2
    assert torch.all(metadata["frame"][pair_query] != metadata["frame"][pair_source])
    distances = (metadata["out"][pair_query] - metadata["out"][pair_source]).norm(dim=-1)
    assert torch.all(distances <= 0.8)

    out, out_coord, seed_out, delta_out, _ = aggregator(
        feat, anchor, seed, delta, offset, frame_batch,
        pose, bbox, bbox_iids, timestamps,
    )
    torch.testing.assert_close(out_coord, anchor)
    torch.testing.assert_close(seed_out[:, 0], anchor)
    torch.testing.assert_close(delta_out, torch.zeros_like(delta_out))
    expected = aggregator.bg_attention.query_norm(feat)
    torch.testing.assert_close(out[1:3], expected[1:3], rtol=0.0, atol=0.0)


def test_temporal_aggregator_runs_without_seeds_for_spherical_reuse():
    torch.manual_seed(5)
    aggregator = GridTemporalAggregator(_cfg(), dim=ATTN_DIM, r_far=80.0).eval()
    anchor = torch.tensor([
        [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.1, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ])
    feat = torch.randn(4, ATTN_DIM)
    offset = torch.tensor([3, 4])
    frame_batch = torch.tensor([0, 0])
    pose, bbox, bbox_iids, timestamps = _empty_scene(2)
    seed, delta = _seed_inputs(anchor)

    out_seeded, coord_seeded, _, _, _ = aggregator(
        feat, anchor, seed, delta, offset, frame_batch,
        pose, bbox, bbox_iids, timestamps,
    )
    out_none, coord_none, seed_none, delta_none, meta = aggregator(
        feat, anchor, None, None, offset, frame_batch,
        pose, bbox, bbox_iids, timestamps,
    )

    # Seed geometry is grid-only; the fused features and coordinates that the
    # spherical head consumes must not depend on its presence.
    assert seed_none is None and delta_none is None
    assert meta["seed_ref"] is None
    torch.testing.assert_close(out_none, out_seeded, rtol=0.0, atol=0.0)
    torch.testing.assert_close(coord_none, coord_seeded, rtol=0.0, atol=0.0)

    try:
        aggregator(
            feat, anchor, None, delta, offset, frame_batch,
            pose, bbox, bbox_iids, timestamps,
        )
    except ValueError as error:
        assert "delta_sensor requires seed_sensor" in str(error)
    else:
        raise AssertionError("delta_sensor without seed_sensor was accepted")


def test_background_seeds_and_delta_use_each_tokens_own_frame_pose():
    aggregator = GridTemporalAggregator(_cfg(), dim=ATTN_DIM, r_far=80.0).eval()
    anchor = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    seed = torch.tensor([[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]])
    delta = torch.tensor([[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]])
    feat = torch.randn(2, ATTN_DIM)
    pose_matrix = torch.eye(4).repeat(2, 1, 1)
    pose_matrix[1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    pose_matrix[1, 0, 3] = 10.0
    pose = [pose_matrix]
    bbox = [[torch.empty(0, 7), torch.empty(0, 7)]]
    bbox_iids = [[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)]]

    _, _, seed_out, delta_out, _ = aggregator(
        feat, anchor, seed, delta, torch.tensor([1, 2]), torch.tensor([0, 0]),
        pose, bbox, bbox_iids, [torch.tensor([0.0, 1.0])],
    )
    torch.testing.assert_close(seed_out[0, 0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(seed_out[1, 0], torch.tensor([9.0, 1.0, 0.0]))
    torch.testing.assert_close(delta_out[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(delta_out[1, 0], torch.tensor([-1.0, 0.0, 0.0]))


def test_foreground_attends_full_persistent_instance_without_radius_or_mixing():
    torch.manual_seed(17)
    aggregator = GridTemporalAggregator(_cfg(), dim=ATTN_DIM, r_far=200.0).eval()
    # The same two instances move 50 m.  Their box-local coordinates align, and
    # each instance must form its own length-2 full-attention sequence despite
    # exceeding the 0.8 m background radius.
    anchor = torch.tensor([
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
        [50.0, 0.0, 0.0], [150.0, 0.0, 0.0],
    ])
    boxes0 = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        [100.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    boxes1 = torch.tensor([
        [50.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        [150.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    pose = [torch.eye(4).repeat(2, 1, 1)]
    bbox = [[boxes0, boxes1]]
    bbox_iids = [[torch.tensor([7, 8]), torch.tensor([7, 8])]]
    timestamps = [torch.tensor([0.0, 1.0])]
    offset = torch.tensor([2, 4])
    frame_batch = torch.tensor([0, 0])
    feat = torch.randn(4, ATTN_DIM)
    seed, delta = _seed_inputs(anchor)

    out_a, coord_a, _, _, meta = aggregator(
        feat, anchor, seed, delta, offset, frame_batch,
        pose, bbox, bbox_iids, timestamps,
    )
    assert meta["is_dynamic"].tolist() == [True, True, True, True]
    instance_ids, instance_counts = torch.unique(
        meta["instance_id"], return_counts=True
    )
    assert instance_ids.tolist() == [7, 8]
    assert instance_counts.tolist() == [2, 2]
    torch.testing.assert_close(coord_a, torch.zeros_like(coord_a), atol=1e-6, rtol=0.0)

    feat_changed = feat.clone()
    feat_changed[[1, 3]] += 100.0  # modify only instance 8
    out_b, _, _, _, _ = aggregator(
        feat_changed, anchor, seed, delta, offset, frame_batch,
        pose, bbox, bbox_iids, timestamps,
    )
    torch.testing.assert_close(out_a[[0, 2]], out_b[[0, 2]], rtol=0.0, atol=0.0)


def test_foreground_seed_and_delta_use_the_tokens_box_local_frame():
    aggregator = GridTemporalAggregator(_cfg(), dim=ATTN_DIM, r_far=80.0).eval()
    anchor = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    seed = torch.tensor([[[10.0, 1.0, 0.0]], [[20.0, 1.0, 0.0]]])
    delta = torch.tensor([[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]])
    feat = torch.randn(2, ATTN_DIM)
    boxes = [
        torch.tensor([[10.0, 0.0, 0.0, 4.0, 4.0, 4.0, math.pi / 2]]),
        torch.tensor([[20.0, 0.0, 0.0, 4.0, 4.0, 4.0, math.pi / 2]]),
    ]
    pose = [torch.eye(4).repeat(2, 1, 1)]
    bbox = [boxes]
    bbox_iids = [[torch.tensor([7]), torch.tensor([7])]]
    timestamps = [torch.tensor([0.0, 1.0])]

    _, anchor_out, seed_out, delta_out, meta = aggregator(
        feat, anchor, seed, delta, torch.tensor([1, 2]), torch.tensor([0, 0]),
        pose, bbox, bbox_iids, timestamps,
    )
    assert meta["is_dynamic"].tolist() == [True, True]
    torch.testing.assert_close(anchor_out, torch.zeros_like(anchor_out), atol=1e-6, rtol=0.0)
    expected = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    torch.testing.assert_close(seed_out, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(delta_out, expected, atol=1e-6, rtol=0.0)


def test_invalid_or_empty_raw_counts_still_pack_one_slot():
    # Occupied cells are positive in production; defensive clamping keeps a
    # malformed count from silently dropping an anchor.
    for count in (0, -3):
        actual = counts_to_variable_k(torch.tensor([count]), 4, 3)
        assert actual.item() == 1


class GridQueryHeadTests(unittest.TestCase):
    """Expose the function-style checks to Python's dependency-free test runner."""


for _name, _function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(GridQueryHeadTests, _name, staticmethod(_function))
del _name, _function


if __name__ == "__main__":
    unittest.main(verbosity=2)
