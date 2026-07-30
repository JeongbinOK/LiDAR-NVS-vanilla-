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
    _select_evidence_seeds,
)
from src.models_new.utils.attention import (
    AnchorQueryCrossAttention,
    AnchorQuerySelfAttention,
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
    delta = torch.randn(3, 3)
    frame_offset = torch.tensor([3])
    metadata = {
        "box_assign": torch.full((3,), -1),
        "instance_id": torch.full((3,), -1),
        "is_dynamic": torch.zeros(3, dtype=torch.bool),
        "coord_ref": position,
        "bbox_ref_by_frame": [torch.empty(0, 7)],
    }
    fused = torch.randn(3, 4)
    seen = {}

    class TemporalAggregator:
        def __call__(self, feat, token_position, seed_sensor, delta_sensor, *args):
            seen["input_feature"] = feat
            assert seed_sensor is None and delta_sensor is None
            return fused, token_position, None, None, {}

    class QueryHead:
        def __call__(self, feat, *args, **kwargs):
            seen["head_feature"] = feat
            return feature, position, delta, torch.ones(3), frame_offset, metadata

    seeds = build_spherical_gaussian_seeds(
        TemporalAggregator(),
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
        None,
    )
    # The query head must consume the temporally fused features, not the raw
    # trunk output.
    assert seen["head_feature"] is fused
    assert seeds.feature is feature
    assert seeds.position is position
    assert seeds.frame_offset is frame_offset
    assert seeds.metadata is metadata
    assert seeds.delta is delta
    assert seeds.gradient_weight is None
    assert seeds.raw_params is None
    assert seeds.routing_stats is None
    assert seeds.routing_budget_logits is None


def _spherical_cfg():
    return SimpleNamespace(
        max_kv=64,
        token_stride_m=0.4,
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
        self, queries, kv_feat, kv_pos, kv_anchor_ids, num_anchors,
        query_pos, position_scale=None, query_anchor_ids=None,
    ):
        self.query_pos = query_pos.detach().clone()
        return torch.arange(
            queries.shape[0], device=queries.device, dtype=queries.dtype
        ).unsqueeze(-1).expand_as(queries).clone()


class _PassQuerySelfAttention(nn.Module):
    def forward(
        self,
        queries,
        query_pos,
        query_anchor_ids,
        num_anchors,
        position_scale=None,
    ):
        return queries


def test_evidence_seed_is_observed_point_nearest_each_anchor_token_mean():
    raw_points = torch.tensor([
        [10.1, 0.0, 0.0],
        [20.2, 0.0, 0.0], [20.1, 0.0, 0.0],
        [40.3, 0.0, 0.0], [39.7, 0.0, 0.0],
        [40.2, 0.0, 0.0], [39.8, 0.0, 0.0],
    ])
    raw_anchor = torch.tensor([0, 1, 1, 2, 2, 2, 2])
    raw_token = torch.tensor([0, 1, 2, 3, 3, 4, 4])
    seeds, query_anchor, query_token, seed_row, anchor_count, evidence_count = \
        _select_evidence_seeds(
            raw_points, raw_anchor, raw_token, num_anchors=3, num_tokens=5,
        )

    # Unique evidence pairs are (0,0), (1,1), (1,2), (2,3), (2,4).
    assert query_anchor.tolist() == [0, 1, 1, 2, 2]
    assert query_token.tolist() == [0, 1, 2, 3, 4]
    assert anchor_count.tolist() == [1, 2, 4]
    assert evidence_count.tolist() == [1, 1, 1, 2, 2]
    # Two-point ties are broken by ascending x, and every seed is a raw row.
    assert seed_row.tolist() == [0, 1, 2, 4, 6]
    torch.testing.assert_close(seeds, raw_points[seed_row])


def test_evidence_seed_selection_rejects_an_anchor_without_raw_membership():
    try:
        _select_evidence_seeds(
            torch.tensor([[10.0, 0.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([0]),
            num_anchors=2,
            num_tokens=1,
        )
    except RuntimeError as error:
        assert "at least one own-frame raw point" in str(error)
    else:
        raise AssertionError("missing raw membership was silently accepted")


def test_variable_query_attention_matches_per_anchor_calls_and_backpropagates():
    torch.manual_seed(21)
    attention = AnchorQueryCrossAttention(
        48, num_heads=8, n_layers=1, chunk=2
    )
    query_anchor = torch.tensor([0, 1, 1, 1, 2, 2])
    kv_anchor = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    queries = torch.randn(query_anchor.numel(), 48, requires_grad=True)
    query_pos = torch.randn(query_anchor.numel(), 3)
    memory = torch.randn(kv_anchor.numel(), 48, requires_grad=True)
    kv_pos = torch.randn(kv_anchor.numel(), 3)

    output = attention.forward_variable(
        queries, memory, kv_pos, query_anchor, kv_anchor, 3, query_pos
    )
    starts = [0, 1, 4]
    counts = [1, 3, 2]
    for anchor, (start, count) in enumerate(zip(starts, counts)):
        q_rows = torch.arange(start, start + count)
        k_rows = (kv_anchor == anchor).nonzero(as_tuple=True)[0]
        expected = attention(
            queries[q_rows].unsqueeze(0),
            memory[k_rows],
            kv_pos[k_rows],
            torch.zeros(k_rows.numel(), dtype=torch.long),
            1,
            query_pos[q_rows].unsqueeze(0),
        )
        torch.testing.assert_close(output[q_rows], expected[0])

    output.square().mean().backward()
    assert queries.grad is not None and queries.grad.abs().sum() > 0
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert attention.q_proj[0].weight.grad is not None
    assert attention.q_proj[0].weight.grad.abs().sum() > 0
    assert attention.kv_proj[0].weight.grad is not None
    assert attention.kv_proj[0].weight.grad.abs().sum() > 0


def test_query_self_attention_is_anchor_local_and_uses_plain_residual():
    torch.manual_seed(23)
    attention = AnchorQuerySelfAttention(
        48,
        num_heads=8,
        n_layers=1,
        varlen_backend="fp32_bucket",
    )
    query_anchor = torch.tensor([0, 0, 1, 1, 1])
    query_position = torch.randn(5, 3)
    query = torch.randn(5, 48, requires_grad=True)

    output = attention(query, query_position, query_anchor, 2)
    changed_query = query.detach().clone()
    changed_query[0, 0] += 2.0
    changed_output = attention(
        changed_query, query_position, query_anchor, 2
    )

    assert not torch.allclose(output[1], changed_output[1])
    torch.testing.assert_close(
        output[query_anchor == 1],
        changed_output[query_anchor == 1],
    )
    assert not any(
        "gate" in name or "layer_scale" in name
        for name, _ in attention.named_parameters()
    )

    output.square().mean().backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert attention.qkv[0].weight.grad is not None
    assert attention.qkv[0].weight.grad.abs().sum() > 0


def test_spherical_head_packs_raw_centres_and_passes_seed_positions_to_rope():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CaptureSeedAttention()
    head.attn = capture
    head.query_self_attn = _PassQuerySelfAttention()
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

    # One query per unique (anchor, token). All raw points of each range group
    # map to one token here, so the three anchors emit one query each.
    expected = raw_points[[0, 2, 6]]
    torch.testing.assert_close(centre, expected)
    torch.testing.assert_close(metadata["coord_ref"], expected)
    torch.testing.assert_close(
        delta,
        (expected - token_position) / 0.4,
        atol=1e-5, rtol=1e-5,
    )
    assert metadata["slot_index"].tolist() == [0, 0, 0]
    assert metadata["source_token_index"].tolist() == [0, 1, 2]
    assert output[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert frame_offset.tolist() == [3]
    assert capture.query_pos.shape == (3, 3)


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
    assert head.p0_norm.weight.grad is not None
    assert head.p0_norm.weight.grad.abs().sum() > 0
    assert head.attn.kv_proj[0].weight.grad is not None
    assert head.attn.kv_proj[0].weight.grad.abs().sum() > 0
    assert head.query_self_attn.qkv[0].weight.grad is not None
    assert head.query_self_attn.qkv[0].weight.grad.abs().sum() > 0
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
    # Each source token is at its own box centre, so the raw seed is +1m on
    # local x and the head receives +1/0.4 = +2.5.
    expected_delta = torch.tensor([[2.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    torch.testing.assert_close(delta, expected_delta, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        metadata["coord_ref"], torch.tensor([[5.0, 1.0, 0.0], [18.0, 1.0, 0.0]])
    )
    torch.testing.assert_close(
        capture.query_pos, expected_local, atol=1e-6, rtol=0.0
    )
    assert metadata["is_dynamic"].tolist() == [True, True]
    assert metadata["slot_index"].tolist() == [0, 0]
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

    # Each frame: the mixed cell A splits into one bg and one fg anchor. Cell B
    # has two raw points but only one supporting token, hence one evidence query.
    assert frame_offset.tolist() == [3, 6]
    assert metadata["is_dynamic"].tolist() == [False, True, False] * 2
    assert metadata["instance_id"].tolist() == [-1, 9, -1] * 2
    assert metadata["slot_index"].tolist() == [0, 0, 0] * 2
    assert metadata["anchor_raw_count"].tolist() == [1, 1, 2] * 2
    assert metadata["evidence_raw_count"].tolist() == [1, 1, 2] * 2
    # bg gaussian stays at the bg point; the fg gaussian is the fg point in
    # box-local coordinates (identity yaw box centred at the fg point).
    torch.testing.assert_close(
        metadata["coord_ref"][:3],
        torch.tensor([
            [6.4, 0.0, 0.0], [6.0, 0.0, 0.0],
            [11.9, 0.0, 0.0],
        ]),
    )
    torch.testing.assert_close(centre[1], torch.tensor([0.0, 0.0, 0.0]))


def test_spherical_queries_keep_each_source_token_feature_before_attention():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)

    class _CaptureQueries(nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = None

        def forward(
            self, queries, kv_feat, kv_pos, kv_anchor_ids, num_anchors,
            query_pos, position_scale=None, query_anchor_ids=None,
        ):
            self.queries = queries.detach().clone()
            return queries

    capture = _CaptureQueries()
    head.attn = capture
    # One spherical cell holds 10 raw points: 3 -> token0, 6 -> token1,
    # 1 -> token2. Raw multiplicity must not collapse the three query identities.
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

    expected_queries = head.p0_norm(feature)
    torch.testing.assert_close(capture.queries, expected_queries, atol=1e-5, rtol=1e-5)


class _CapturePairsAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_ids = None
        self.kv_pos = None

    def forward(
        self, queries, kv_feat, kv_pos, kv_anchor_ids, num_anchors,
        query_pos, position_scale=None, query_anchor_ids=None,
    ):
        self.anchor_ids = kv_anchor_ids.detach().clone()
        self.kv_pos = kv_pos.detach().clone()
        return queries


def test_spherical_kv_dedups_own_cell_tokens_and_stays_own_frame():
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

    # Cross-frame context is fused upstream by the shared temporal aggregator,
    # so each anchor keeps exactly its own frame's member tokens: anchor 0
    # (frame 0) dedups its two raws to the single token-0 pair even though
    # frame 1 populates the same spherical cell, and the frame-1 anchors hold
    # one own token each.
    assert capture.anchor_ids.tolist() == [0, 1, 2]


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
    # Raw membership keeps them: each anchor of instance 5 attends to its OWN
    # frame's instance token at the box-local position (the other frame's
    # token is fused upstream, not routed here).
    assert capture.anchor_ids.tolist() == [0, 1]
    torch.testing.assert_close(
        capture.kv_pos, torch.tensor([[0.6, 0.0, 0.0]] * 2)
    )


def test_spherical_head_keeps_fp32_positions_under_bf16_features():
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)
    capture = _CapturePairsAttention()
    head.attn = capture
    head.query_self_attn = _PassQuerySelfAttention()
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


def test_spherical_pipeline_runs_shared_fusion_then_own_frame_queries():
    """End-to-end seam test: real aggregator (no seeds) -> real spherical head."""
    from src.models_new.module.grid_query_head import GridTemporalAggregator

    torch.manual_seed(3)
    aggregator = GridTemporalAggregator(
        SimpleNamespace(
            K_max=3, points_per_gaussian=4, bg_radius_m=0.8,
            bg_max_kv_per_frame=8, num_heads=2, bg_layers=1, fg_layers=1,
            grad_balance="sqrt_k",
        ),
        dim=48, r_far=80.0,
    )
    head = SphericalQueryHead(_spherical_cfg(), dim=48, r_far=80.0)

    # Two frames x (bg token pair within the 0.8 m radius + one boxed token).
    token_position = torch.tensor([
        [10.0, 0.0, 0.0], [30.0, 0.0, 0.0],
        [10.3, 0.0, 0.0], [30.0, 0.0, 0.0],
    ])
    raw_points = torch.tensor([
        [10.1, 0.0, 0.0], [30.0, 0.1, 0.0],
        [10.2, 0.0, 0.0], [29.9, 0.0, 0.0],
    ])
    raw_token_index = torch.tensor([0, 1, 2, 3])
    pose = torch.eye(4).repeat(2, 1, 1)
    box = torch.tensor([[10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    boxes = [[box.clone(), box.clone()]]
    instance_ids = [[torch.tensor([4]), torch.tensor([4])]]
    feature = torch.randn(4, 48, requires_grad=True)

    seeds = build_spherical_gaussian_seeds(
        aggregator, head, feature, token_position, raw_points,
        raw_token_index, torch.tensor([2, 4]), torch.tensor([0, 0]),
        [pose], boxes, instance_ids, [torch.tensor([0.0, 1.0])],
    )

    assert seeds.raw_params is None and seeds.gradient_weight is None
    assert seeds.feature.shape == (4, 48)
    assert seeds.metadata["is_dynamic"].tolist() == [True, False] * 2
    # fg seeds live box-local, bg seeds in the ref frame.
    torch.testing.assert_close(
        seeds.position[0], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(seeds.position[1], raw_points[1])
    assert seeds.frame_offset.tolist() == [2, 4]

    (seeds.feature.square().mean() + seeds.delta.square().mean()).backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert aggregator.bg_attention.kv_proj[0].weight.grad is not None
    assert aggregator.bg_attention.kv_proj[0].weight.grad.abs().sum() > 0
    assert head.attn.kv_proj[0].weight.grad is not None
    assert head.attn.kv_proj[0].weight.grad.abs().sum() > 0


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


def test_learned_grid_head_st_selects_matching_k_seed_geometry():
    token_feature = torch.randn(1, 4)
    token_position = torch.zeros(1, 3)
    candidate_seed = torch.zeros(1, 4, 4, 3)
    for k in range(1, 5):
        candidate_seed[0, k - 1, :k, 0] = (
            10 * k + torch.arange(k, dtype=torch.float32)
        )
    candidate_seed.requires_grad_()
    candidate_delta = candidate_seed.clone()
    candidate_ref = candidate_seed + 100.0
    selection = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]], requires_grad=True
    )
    budget_logits = torch.zeros(1, 4, requires_grad=True)
    anchor_metadata = {
        "box_assign": torch.tensor([-1]),
        "instance_id": torch.tensor([-1]),
        "is_dynamic": torch.tensor([False]),
        "coord_ref": token_position,
        "seed_ref": candidate_ref,
        "bbox_ref_by_frame": [torch.empty(0, 7)],
    }

    class TemporalAggregator:
        def __call__(self, *args):
            return (
                token_feature,
                token_position,
                candidate_seed,
                candidate_delta,
                anchor_metadata,
            )

    class SlotHead:
        k_max = 4

        def __call__(self, feature, anchor_k, delta, token_offset):
            assert anchor_k is None
            assert delta is candidate_delta
            return torch.zeros(2, 6), {
                "anchor_index": torch.tensor([0, 0]),
                "slot_index": torch.tensor([0, 1]),
                "slot_k": torch.tensor([2, 2]),
                "anchor_k": torch.tensor([2]),
                "gaussian_offset": torch.tensor([2]),
                "k_logits": budget_logits,
                "k_selection": selection,
                "selected_only": True,
            }

        @staticmethod
        def gradient_weight(slot_k, dtype):
            return torch.ones_like(slot_k, dtype=dtype)

    seeds = build_grid_gaussian_seeds(
        TemporalAggregator(),
        SlotHead(),
        token_feature,
        token_position,
        None,
        candidate_seed,
        candidate_delta,
        torch.tensor([1]),
        torch.tensor([0]),
        [],
        [],
        None,
        None,
    )

    torch.testing.assert_close(
        seeds.position[:, 0], torch.tensor([20.0, 21.0])
    )
    torch.testing.assert_close(
        seeds.metadata["coord_ref"][:, 0], torch.tensor([120.0, 121.0])
    )
    assert seeds.routing_stats is not None
    assert seeds.routing_stats["selected_k"].tolist() == [2]
    torch.testing.assert_close(
        seeds.routing_stats["k_logits"], torch.zeros(1, 4)
    )
    torch.testing.assert_close(
        seeds.routing_stats["token_position_sensor"], token_position
    )
    assert seeds.routing_stats["is_dynamic"].tolist() == [False]
    assert all(
        not value.requires_grad for value in seeds.routing_stats.values()
    )
    assert seeds.routing_budget_logits is budget_logits
    assert seeds.routing_budget_logits.requires_grad
    (seeds.position.sum() + seeds.metadata["coord_ref"].sum()).backward()
    assert candidate_seed.grad is not None
    assert candidate_seed.grad.abs().sum() > 0
    assert selection.grad is None
