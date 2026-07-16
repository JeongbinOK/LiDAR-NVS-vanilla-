import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models_new.module.anchor_modes import (
    build_grid_gaussian_seeds,
    build_spherical_gaussian_seeds,
)
from src.models_new.module.builders import (
    SUPPORTED_ANCHOR_MODES,
    resolve_anchor_mode,
)
from src.models_new.module.spherical_query_head import (
    SphericalQueryHead,
    _select_raw_seed_slots,
)
from src.models_new.utils import boxes as box_utils


def _cell_center_from_point(bins, point):
    """Independent spherical-cell-centre math (bin index via the shared bins)."""
    idx3, valid = bins.bin_coords(point.view(1, 3))
    assert bool(valid.item())
    i_theta, i_phi, i_lr = (int(v) for v in idx3[0])
    theta = (i_theta + 0.5) * bins.dtheta - math.pi / 2
    phi = (i_phi + 0.5) * bins.dphi - math.pi
    r = bins.r_min * math.exp((i_lr + 0.5) * bins.dlogr)
    return torch.tensor([
        r * math.cos(theta) * math.cos(phi),
        r * math.cos(theta) * math.sin(phi),
        r * math.sin(theta),
    ], dtype=torch.float32)


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
    delta = torch.randn(3, 3)
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
            return feature, position, delta, torch.ones(3), frame_offset, metadata

    seeds = build_spherical_gaussian_seeds(
        QueryHead(),
        torch.empty(0, 4),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, dtype=torch.long),
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
    assert seeds.delta is delta
    assert seeds.gradient_weight is None
    assert seeds.raw_params is None


def _spherical_cfg():
    return SimpleNamespace(
        K=3,
        max_kv=64,
        dtheta_deg=2.0,
        dphi_deg=3.0,
        dlogr=0.18,
        r_min=2.0,
        r_max=80.0,
        n_layers=1,
        attn_chunk=32,
        ffn_ratio=4,
    )


class _CaptureSeedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_pos = None

    def forward(
        self, queries, kv_feat, kv_pos, anchor_ids, num_anchors,
        query_pos, position_scale=None,
    ):
        self.query_pos = query_pos.detach().clone()
        slot = torch.arange(queries.shape[1], device=queries.device, dtype=queries.dtype)
        output = slot.view(1, -1, 1).expand_as(queries).clone()
        # Deliberately return positions that must not become Gaussian centres.
        wrong_weighted_position = torch.full_like(query_pos, -999.0)
        return output, wrong_weighted_position


def test_raw_seed_quantiles_use_canonical_middle_and_outer_slots():
    anchor_sensor = torch.tensor([
        [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0],
    ])
    raw_points = torch.tensor([
        [10.1, 0.0, 0.0],
        [20.2, 0.0, 0.0], [20.1, 0.0, 0.0],
        [40.3, 0.0, 0.0], [39.7, 0.0, 0.0],
        [40.2, 0.0, 0.0], [39.8, 0.0, 0.0],
    ])
    raw_anchor = torch.tensor([0, 1, 1, 2, 2, 2, 2])
    seeds, active, counts = _select_raw_seed_slots(
        raw_points, raw_anchor, anchor_sensor
    )

    assert counts.tolist() == [1, 2, 4]
    assert active.nonzero().tolist() == [
        [0, 1], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2],
    ]
    torch.testing.assert_close(seeds[0, 1], raw_points[0])
    torch.testing.assert_close(seeds[1, [0, 2]], raw_points[[2, 1]])
    torch.testing.assert_close(seeds[2], raw_points[[4, 5, 3]])


def test_raw_seed_selection_rejects_an_occupied_anchor_without_raw_membership():
    try:
        _select_raw_seed_slots(
            torch.tensor([[10.0, 0.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        )
    except RuntimeError as error:
        assert "at least one own-frame raw point" in str(error)
    else:
        raise AssertionError("missing raw membership was silently accepted")


def test_spherical_head_packs_raw_centres_and_passes_seed_positions_to_rope():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CaptureSeedAttention()
    head.attn = capture
    token_position = torch.tensor([
        [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0],
    ])
    raw_points = torch.tensor([
        [10.1, 0.0, 0.0],
        [20.2, 0.0, 0.0], [20.1, 0.0, 0.0],
        [40.3, 0.0, 0.0], [39.7, 0.0, 0.0],
        [40.2, 0.0, 0.0], [39.8, 0.0, 0.0],
    ])
    raw_token_index = torch.tensor([0, 1, 1, 2, 2, 2, 2])

    output, centre, delta, _, frame_offset, metadata = head(
        torch.randn(3, 48), token_position, raw_points, raw_token_index,
        torch.tensor([3]), torch.tensor([0]), [torch.eye(4).unsqueeze(0)],
        [[torch.empty(0, 7)]], [[torch.empty(0, dtype=torch.long)]],
    )

    expected = raw_points[[0, 2, 1, 4, 5, 3]]
    torch.testing.assert_close(centre, expected)
    torch.testing.assert_close(metadata["coord_ref"], expected)
    # delta_p is measured from each anchor's spherical cell centre (the cells
    # here are defined by the raw points themselves, one per range cluster).
    anchor_centers = torch.stack([
        _cell_center_from_point(head.bins, raw_points[0]),
        _cell_center_from_point(head.bins, raw_points[1]),
        _cell_center_from_point(head.bins, raw_points[3]),
    ])
    torch.testing.assert_close(
        delta, expected - anchor_centers[[0, 1, 1, 2, 2, 2]],
        atol=1e-4, rtol=1e-5,
    )
    assert metadata["slot_index"].tolist() == [1, 0, 2, 0, 1, 2]
    assert output[:, 0].tolist() == [1.0, 0.0, 2.0, 0.0, 1.0, 2.0]
    assert frame_offset.tolist() == [6]
    assert capture.query_pos.shape == (3, 3, 3)
    assert not torch.any(centre == -999.0)


def test_spherical_attention_and_d_plus_3_predictor_backward():
    torch.manual_seed(13)
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    token_position = torch.tensor([
        [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0],
    ])
    raw_points = torch.tensor([
        [10.1, 0.0, 0.0],
        [20.2, 0.0, 0.0], [20.1, 0.0, 0.0],
        [40.3, 0.0, 0.0], [39.7, 0.0, 0.0],
        [40.2, 0.0, 0.0], [39.8, 0.0, 0.0],
    ])
    feature = torch.randn(3, 48, requires_grad=True)
    output, _, delta, _, _, _ = head(
        feature, token_position, raw_points,
        torch.tensor([0, 1, 1, 2, 2, 2, 2]),
        torch.tensor([3]), torch.tensor([0]), [torch.eye(4).unsqueeze(0)],
        [[torch.empty(0, 7)]], [[torch.empty(0, dtype=torch.long)]],
    )
    predictor = nn.Linear(48 + 3, 4)
    prediction = predictor(torch.cat([output, delta.to(output.dtype)], dim=-1))
    prediction.square().mean().backward()

    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert head.query_embed.grad is not None
    assert torch.all(head.query_embed.grad.abs().sum(dim=-1) > 0)
    assert predictor.weight.grad[:, -3:].abs().sum() > 0


def test_spherical_foreground_seed_and_delta_use_own_bbox_local_frame():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CaptureSeedAttention()
    head.attn = capture
    token_position = torch.tensor([[5.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    raw_points = torch.tensor([[5.0, 1.0, 0.0], [8.0, 1.0, 0.0]])
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[1, 0, 3] = 10.0
    boxes = [[
        torch.tensor([[5.0, 0.0, 0.0, 4.0, 4.0, 4.0, torch.pi / 2]]),
        torch.tensor([[8.0, 0.0, 0.0, 4.0, 4.0, 4.0, torch.pi / 2]]),
    ]]
    instance_ids = [[torch.tensor([7]), torch.tensor([7])]]

    _, centre, delta, _, frame_offset, metadata = head(
        torch.randn(2, 48), token_position, raw_points, torch.tensor([0, 1]),
        torch.tensor([1, 2]), torch.tensor([0, 0]), [pose], boxes, instance_ids,
    )

    expected_local = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(centre, expected_local, atol=1e-6, rtol=0.0)
    # delta_p subtracts the spherical cell centre expressed in the same
    # box-local frame as the seed.
    box0_ref = boxes[0][0][0]
    box1_ref = boxes[0][1][0].clone()
    box1_ref[0] += 10.0                        # translation-only pose -> ref
    center0 = _cell_center_from_point(head.bins, raw_points[0])
    center1_ref = _cell_center_from_point(head.bins, raw_points[1]) \
        + torch.tensor([10.0, 0.0, 0.0])
    expected_delta = expected_local - torch.cat([
        box_utils.points_to_box_local(center0.view(1, 3), box0_ref),
        box_utils.points_to_box_local(center1_ref.view(1, 3), box1_ref),
    ])
    torch.testing.assert_close(delta, expected_delta, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        metadata["coord_ref"], torch.tensor([[5.0, 1.0, 0.0], [18.0, 1.0, 0.0]])
    )
    torch.testing.assert_close(
        capture.query_pos[:, 1], expected_local, atol=1e-6, rtol=0.0
    )
    assert metadata["is_dynamic"].tolist() == [True, True]
    assert metadata["slot_index"].tolist() == [1, 1]
    assert frame_offset.tolist() == [1, 2]


def test_spherical_mixed_cells_split_into_pure_label_group_anchors():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CaptureSeedAttention()
    head.attn = capture
    token_position = torch.tensor([
        [6.0, 0.0, 0.0], [12.0, 0.0, 0.0],    # frame 0
        [6.0, 0.0, 0.0], [12.0, 0.0, 0.0],    # frame 1
    ])
    raw_points = torch.tensor([
        [6.0, 0.0, 0.0], [6.4, 0.0, 0.0],     # cell A: 1 fg + 1 bg -> 2 anchors
        [12.0, 0.0, 0.0], [11.9, 0.0, 0.0],   # cell B: bg only -> 1 anchor
        [6.0, 0.0, 0.0], [6.4, 0.0, 0.0],
        [12.0, 0.0, 0.0], [11.9, 0.0, 0.0],
    ])
    raw_token_index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    pose = torch.eye(4).repeat(2, 1, 1)
    box = torch.tensor([[6.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]])
    boxes = [[box.clone(), box.clone()]]
    instance_ids = [[torch.tensor([9]), torch.tensor([9])]]

    _, centre, _, _, frame_offset, metadata = head(
        torch.randn(4, 48), token_position, raw_points, raw_token_index,
        torch.tensor([2, 4]), torch.tensor([0, 0]), [pose], boxes, instance_ids,
    )

    # Each frame: the mixed cell A splits into a bg anchor (its road point,
    # 1 gaussian) and an fg anchor (its car point, 1 gaussian) -- no vote, no
    # dropped minority point -- followed by cell B's bg anchor (2 gaussians).
    assert frame_offset.tolist() == [4, 8]
    assert metadata["is_dynamic"].tolist() == [False, True, False, False] * 2
    assert metadata["instance_id"].tolist() == [-1, 9, -1, -1] * 2
    assert metadata["slot_index"].tolist() == [1, 1, 0, 2] * 2
    assert metadata["anchor_raw_count"].tolist() == [1, 1, 2, 2] * 2
    # bg gaussian stays at the bg point; the fg gaussian is the fg point in
    # box-local coordinates (identity yaw box centred at the fg point).
    torch.testing.assert_close(
        metadata["coord_ref"][:4],
        torch.tensor([
            [6.4, 0.0, 0.0], [6.0, 0.0, 0.0],
            [11.9, 0.0, 0.0], [12.0, 0.0, 0.0],
        ]),
    )
    torch.testing.assert_close(centre[1], torch.tensor([0.0, 0.0, 0.0]))


def test_spherical_anchor_embedding_is_raw_count_weighted_token_mean():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)

    class _CaptureQueries(nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = None

        def forward(
            self, queries, kv_feat, kv_pos, anchor_ids, num_anchors,
            query_pos, position_scale=None,
        ):
            self.queries = queries.detach().clone()
            return queries, query_pos

    capture = _CaptureQueries()
    head.attn = capture
    # one spherical cell holding 10 raw points: 3 -> token0, 6 -> token1,
    # 1 -> token2, so the anchor embedding must pool 0.3/0.6/0.1 of their
    # fused features (weighted sum, no MLP) before the shared LayerNorm.
    token_position = torch.tensor([
        [30.0, 0.0, 0.0], [30.4, 0.0, 0.0], [30.8, 0.0, 0.0],
    ])
    raw_points = torch.tensor([30.0, 0.0, 0.0]) + \
        torch.linspace(0.0, 0.9, 10).view(-1, 1) * torch.tensor([[1.0, 0.0, 0.0]])
    raw_token_index = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 2])
    feature = torch.randn(3, 48)

    head(
        feature, token_position, raw_points, raw_token_index,
        torch.tensor([3]), torch.tensor([0]), [torch.eye(4).unsqueeze(0)],
        [[torch.empty(0, 7)]], [[torch.empty(0, dtype=torch.long)]],
    )

    expected_pool = 0.3 * feature[0] + 0.6 * feature[1] + 0.1 * feature[2]
    expected_queries = head.query_embed + head.p0_norm(expected_pool).unsqueeze(0)
    torch.testing.assert_close(
        capture.queries[0], expected_queries, atol=1e-5, rtol=1e-5
    )


class _CapturePairsAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_ids = None
        self.kv_pos = None

    def forward(
        self, queries, kv_feat, kv_pos, anchor_ids, num_anchors,
        query_pos, position_scale=None,
    ):
        self.anchor_ids = anchor_ids.detach().clone()
        self.kv_pos = kv_pos.detach().clone()
        return queries, query_pos


def test_spherical_kv_dedups_own_cell_tokens_and_rebins_other_frame_raws():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CapturePairsAttention()
    head.attn = capture
    # frame 0: one token, two raw points in one cell; frame 1: the same cell
    # plus a second cell one azimuth bin away (phi ~ 4 deg at the same range).
    token_position = torch.tensor([
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0], [9.9756, 0.6976, 0.0],
    ])
    raw_points = torch.tensor([
        [10.0, 0.0, 0.0], [10.05, 0.0, 0.0],
        [10.02, 0.0, 0.0], [9.9756, 0.6976, 0.0],
    ])
    raw_token_index = torch.tensor([0, 0, 1, 2])
    pose = torch.eye(4).repeat(2, 1, 1)
    boxes = [[torch.empty(0, 7), torch.empty(0, 7)]]
    instance_ids = [[
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
    ]]

    head(
        torch.randn(3, 48), token_position, raw_points, raw_token_index,
        torch.tensor([1, 3]), torch.tensor([0, 0]), [pose], boxes, instance_ids,
    )

    # anchor 0 (frame 0): own-cell token 0 (two raws dedup to one pair) + the
    # other frame's re-binned same-cell raw -> token 1. anchor 1 (frame 1):
    # mirror of anchor 0. anchor 2 (frame 1, adjacent azimuth cell): own token
    # only -- with the 1x1x1 neighbourhood no cross-cell pair may appear.
    assert capture.anchor_ids.tolist() == [0, 0, 1, 1, 2]


def test_spherical_fg_kv_uses_instance_raw_membership_not_token_labels():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CapturePairsAttention()
    head.attn = capture
    # Each frame: one fg raw point inside a small box, assigned to a token
    # whose centroid sits OUTSIDE the box (10.6 vs half extent 0.25).
    token_position = torch.tensor([[10.6, 0.0, 0.0], [10.6, 0.0, 0.0]])
    raw_points = torch.tensor([[10.1, 0.0, 0.0], [10.1, 0.0, 0.0]])
    raw_token_index = torch.tensor([0, 1])
    pose = torch.eye(4).repeat(2, 1, 1)
    box = torch.tensor([[10.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]])
    boxes = [[box.clone(), box.clone()]]
    instance_ids = [[torch.tensor([5]), torch.tensor([5])]]

    head(
        torch.randn(2, 48), token_position, raw_points, raw_token_index,
        torch.tensor([1, 2]), torch.tensor([0, 0]), [pose], boxes, instance_ids,
    )

    # A token-label rule would leave both fg anchors with an empty K/V set.
    # Raw membership keeps them: every anchor of instance 5 attends to both
    # frames' instance tokens at their box-local positions.
    assert capture.anchor_ids.tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(
        capture.kv_pos, torch.tensor([[0.6, 0.0, 0.0]] * 4)
    )


def test_spherical_head_keeps_fp32_positions_under_bf16_features():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CapturePairsAttention()
    head.attn = capture
    # mixed bg + fg pairs force the fg box-local position buffer to be
    # concatenated with the fp32 ref positions of the bg pairs.
    token_position = torch.tensor([
        [10.6, 0.0, 0.0], [30.0, 0.0, 0.0],
        [10.6, 0.0, 0.0], [30.0, 0.0, 0.0],
    ])
    raw_points = torch.tensor([
        [10.1, 0.0, 0.0], [30.0, 0.0, 0.0],
        [10.1, 0.0, 0.0], [30.0, 0.0, 0.0],
    ])
    raw_token_index = torch.tensor([0, 1, 2, 3])
    pose = torch.eye(4).repeat(2, 1, 1)
    box = torch.tensor([[10.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]])
    boxes = [[box.clone(), box.clone()]]
    instance_ids = [[torch.tensor([5]), torch.tensor([5])]]

    _, seed, delta, _, _, _ = head(
        torch.randn(4, 48).bfloat16(), token_position, raw_points,
        raw_token_index, torch.tensor([2, 4]), torch.tensor([0, 0]),
        [pose], boxes, instance_ids,
    )

    assert capture.kv_pos.dtype == torch.float32
    assert seed.dtype == torch.float32 and delta.dtype == torch.float32


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
