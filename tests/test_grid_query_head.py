from types import SimpleNamespace
import unittest

import torch

from src.models_new.module.builders.common import aggregate_points_to_cells
from src.models_new.module.gaussian_assembly import gradient_scale_identity
from src.models_new.module.grid_query_head import (
    GridSlotHead,
    GridTemporalAggregator,
    _radius_pairs,
    counts_to_variable_k,
)


def _cfg(**overrides):
    values = {
        "K_max": 3,
        "points_per_gaussian": 4,
        "bg_radius_m": 0.8,
        "bg_max_kv_per_frame": 8,
        "num_heads": 2,
        "bg_layers": 2,
        "fg_layers": 2,
        "layer_scale": 1.0e-5,
        "film_scale": 0.1,
        "count_condition_cap": 32,
        "grad_balance": "sqrt_k",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_variable_slot_packing_and_frame_offsets():
    torch.manual_seed(4)
    head = GridSlotHead(_cfg(), dim=8, r_far=80.0)
    raw_count = torch.tensor([1, 4, 5, 8, 9, 100])
    feature = torch.randn(6, 8)
    sensor_range = torch.arange(1, 7, dtype=torch.float32)
    slot_feature, packing = head(
        feature, raw_count, sensor_range, torch.tensor([3, 6])
    )

    assert packing["anchor_k"].tolist() == [1, 1, 2, 2, 3, 3]
    assert packing["gaussian_offset"].tolist() == [4, 12]
    assert slot_feature.shape == (12, 8)
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


def test_slot_codes_and_film_outputs_differ_at_initialization():
    torch.manual_seed(11)
    head = GridSlotHead(_cfg(), dim=8, r_far=80.0)
    feature = torch.randn(1, 8)
    slot_feature, packing = head(
        feature, torch.tensor([9]), torch.tensor([12.0]), torch.tensor([1])
    )
    expected_u = torch.tensor([1.0 / 6.0, 0.5, 5.0 / 6.0])
    torch.testing.assert_close(packing["slot_u"], expected_u)
    assert torch.unique(packing["conditioner_input"], dim=0).shape[0] == 3
    assert (slot_feature[0] - slot_feature[1]).abs().max() > 0
    assert (slot_feature[1] - slot_feature[2]).abs().max() > 0

    predictor = torch.nn.Linear(8, 42)
    gaussian_output = predictor(slot_feature)
    assert (gaussian_output[0] - gaussian_output[1]).abs().max() > 0
    assert (gaussian_output[1] - gaussian_output[2]).abs().max() > 0


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


def test_background_excludes_same_frame_and_no_match_is_identity():
    torch.manual_seed(5)
    aggregator = GridTemporalAggregator(_cfg(), dim=8, r_far=80.0).eval()
    # Frame 0 rows 1 and 2 are close to each other, but have no other-frame match.
    anchor = torch.tensor([
        [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.1, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ])
    feat = torch.randn(4, 8)
    offset = torch.tensor([3, 4])
    frame_batch = torch.tensor([0, 0])
    pose, bbox, bbox_iids, timestamps = _empty_scene(2)

    metadata = aggregator._token_metadata(
        anchor, offset, frame_batch, pose, bbox, bbox_iids, timestamps
    )
    pair_query, pair_source = aggregator._background_pairs(metadata)
    assert pair_query.numel() == 2
    assert torch.all(metadata["frame"][pair_query] != metadata["frame"][pair_source])
    distances = (metadata["out"][pair_query] - metadata["out"][pair_source]).norm(dim=-1)
    assert torch.all(distances <= 0.8)

    out, out_coord, _ = aggregator(
        feat, anchor, offset, frame_batch, pose, bbox, bbox_iids, timestamps
    )
    torch.testing.assert_close(out_coord, anchor)
    torch.testing.assert_close(out[1:3], feat[1:3], rtol=0.0, atol=0.0)


def test_foreground_attends_full_persistent_instance_without_radius_or_mixing():
    torch.manual_seed(17)
    aggregator = GridTemporalAggregator(_cfg(), dim=8, r_far=200.0).eval()
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
    feat = torch.randn(4, 8)

    out_a, coord_a, meta = aggregator(
        feat, anchor, offset, frame_batch, pose, bbox, bbox_iids, timestamps
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
    out_b, _, _ = aggregator(
        feat_changed, anchor, offset, frame_batch, pose, bbox, bbox_iids, timestamps
    )
    torch.testing.assert_close(out_a[[0, 2]], out_b[[0, 2]], rtol=0.0, atol=0.0)


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
