from types import SimpleNamespace

import torch

from src.models_new.module.anchor_modes import (
    build_grid_gaussian_seeds,
    build_spherical_gaussian_seeds,
)
from src.models_new.module.builders import (
    SUPPORTED_ANCHOR_MODES,
    resolve_anchor_mode,
)


def test_only_spherical_and_grid_modes_are_supported():
    assert SUPPORTED_ANCHOR_MODES == ("spherical", "grid")
    assert resolve_anchor_mode(SimpleNamespace(anchor_mode="SPHERICAL")) == "spherical"
    assert resolve_anchor_mode(SimpleNamespace(anchor_mode="grid")) == "grid"
    try:
        resolve_anchor_mode(SimpleNamespace(anchor_mode="spherical_legacy"))
    except ValueError as error:
        assert "expected one of" in str(error)
    else:
        raise AssertionError("unsupported anchor mode was accepted")


def test_spherical_head_adapts_to_common_seed_contract():
    feature = torch.randn(3, 4)
    position = torch.randn(3, 3)
    frame_offset = torch.tensor([3])
    metadata = {
        "box_assign": torch.full((3,), -1),
        "instance_id": torch.full((3,), -1),
        "is_dynamic": torch.zeros(3, dtype=torch.bool),
        "coord_ref": position,
        "bbox_ref_by_frame": [torch.empty(0, 7)],
    }

    class QueryHead:
        def __call__(self, *args):
            return feature, position, torch.ones(3), frame_offset, metadata

    seeds = build_spherical_gaussian_seeds(
        QueryHead(),
        torch.empty(0, 4),
        torch.empty(0, 3),
        torch.tensor([0]),
        torch.tensor([0]),
        [],
        [],
        None,
    )
    assert seeds.feature is feature
    assert seeds.position is position
    assert seeds.frame_offset is frame_offset
    assert seeds.metadata is metadata
    assert seeds.gradient_weight is None
    assert seeds.raw_params is None


def test_grid_head_expands_positions_metadata_and_gradient_weights():
    token_feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    token_position = torch.tensor([
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
    ])
    anchor_position = token_position + 10.0
    seed_position = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    delta_p = torch.randn(3, 2, 3)
    seed_ref = seed_position + 20.0
    anchor_metadata = {
        "box_assign": torch.tensor([-1, 0, 1]),
        "instance_id": torch.tensor([-1, 7, 8]),
        "is_dynamic": torch.tensor([False, True, True]),
        "coord_ref": token_position + 20.0,
        "seed_ref": seed_ref,
        "bbox_ref_by_frame": [torch.empty(0, 7)],
    }

    class TemporalAggregator:
        def __call__(self, *args):
            return (
                token_feature, anchor_position, seed_position, delta_p,
                anchor_metadata,
            )

    class SlotHead:
        def __init__(self):
            self.delta_p = None

        def __call__(self, feature, anchor_k, delta, token_offset):
            self.delta_p = delta
            anchor_index = torch.tensor([0, 1, 1, 2])
            return torch.arange(24, dtype=torch.float32).reshape(4, 6), {
                "anchor_index": anchor_index,
                "slot_index": torch.tensor([0, 0, 1, 0]),
                "slot_k": torch.tensor([1, 2, 2, 1]),
                "gaussian_offset": torch.tensor([4]),
            }

        @staticmethod
        def gradient_weight(slot_k, dtype):
            return slot_k.to(dtype).rsqrt()

    slot_head = SlotHead()
    seeds = build_grid_gaussian_seeds(
        TemporalAggregator(),
        slot_head,
        token_feature,
        token_position,
        torch.tensor([1, 2, 1]),
        torch.zeros(3, 2, 3),
        torch.zeros(3, 2, 3),
        torch.tensor([3]),
        torch.tensor([0]),
        [],
        [],
        None,
        None,
    )

    anchor_index = torch.tensor([0, 1, 1, 2])
    slot_index = torch.tensor([0, 0, 1, 0])
    assert seeds.feature is None
    torch.testing.assert_close(
        seeds.raw_params, torch.arange(24, dtype=torch.float32).reshape(4, 6)
    )
    torch.testing.assert_close(seeds.position, seed_position[anchor_index, slot_index])
    torch.testing.assert_close(slot_head.delta_p, delta_p)
    assert seeds.frame_offset.tolist() == [4]
    assert seeds.metadata["instance_id"].tolist() == [-1, 7, 7, 8]
    assert seeds.metadata["is_dynamic"].tolist() == [False, True, True, True]
    torch.testing.assert_close(
        seeds.gradient_weight,
        torch.tensor([1.0, 2.0**-0.5, 2.0**-0.5, 1.0]),
    )
    torch.testing.assert_close(
        seeds.metadata["coord_ref"], seed_ref[anchor_index, slot_index]
    )
