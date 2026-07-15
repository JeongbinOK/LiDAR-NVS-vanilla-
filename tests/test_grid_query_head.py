import math
from types import SimpleNamespace
import unittest

import torch

from src.models_new.module.builders.common import (
    aggregate_points_to_cells,
    aggregate_points_to_cells_with_seeds,
    counts_to_variable_k,
)
from src.models_new.module.builders.grid_intensity import OccupiedGridTokenBuilder
from src.models_new.module.gaussian_assembly import gradient_scale_identity
from src.models_new.module.grid_query_head import (
    GridSlotHead,
    GridTemporalAggregator,
    _radius_pairs,
)


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


def _aggregate_seed_cell(points, points_per_gaussian, k_max=3, voxel_coord=None):
    intensity = torch.arange(1, points.shape[0] + 1, dtype=points.dtype)
    grid_coord = torch.tensor([[0, 0, 0]])
    voxel_feature = torch.tensor([[3.0, 4.0]])
    if voxel_coord is None:
        voxel_coord = torch.tensor([[0.5, 0.5, 0.5]])
    return aggregate_points_to_cells_with_seeds(
        points, intensity, grid_coord, voxel_feature, voxel_coord,
        torch.zeros(3), _UnitMapper(), points_per_gaussian, k_max,
    )


def _seed_inputs(anchor, k_max=1):
    seed = anchor[:, None, :].expand(-1, k_max, -1).clone()
    return seed, torch.zeros_like(seed)


def _builder_cfg(anchor_mode):
    return SimpleNamespace(
        anchor_mode=anchor_mode,
        grid_query=_cfg(),
        intensity_encoder=SimpleNamespace(type="mlp"),
        int_proj=SimpleNamespace(in_dim=5, out_dim=4),
        r_far=80.0,
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
    pos, feature, _, occupied, raw_count = aggregate_points_to_cells(
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
    assert len(seed_data) == 2
    torch.testing.assert_close(seed_data[0].seed_sensor[0, 0], lidar_points[1, :3])
    torch.testing.assert_close(seed_data[1].seed_sensor[0, 0], lidar_points[3, :3])
    assert seed_data[0].anchor_k.tolist() == [1]
    assert seed_data[1].anchor_k.tolist() == [1]

    spherical_out = OccupiedGridTokenBuilder(_builder_cfg("spherical"))(
        lidar_points, offset, None, features, ptv3_input, _UnitMapper()
    )
    assert spherical_out.grid_seeds is None


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
