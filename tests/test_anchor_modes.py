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
    _range_quantile_seed_rows,
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
            return feature, position, delta, frame_offset, metadata

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


def _learned_spherical_cfg():
    return SimpleNamespace(
        count_mode="learned_gumbel",
        max_kv=64,
        token_stride_m=0.4,
        dtheta_deg=4.5,
        dphi_deg=3.0,
        dlogr=0.18,
        dr_m=0.8,
        r_min=2.0,
        r_max=110.0,
        grad_balance="sqrt_k",
        learned_count=SimpleNamespace(
            K_max=4,
            tau=1.0,
            stat_embedding_dim=64,
            stat_min_raw=6,
            stat_min_valid_loo=3,
            stat_loo_eps=1.0e-3,
            stat_range_min_m=2.5,
            stat_range_max_m=110.0,
            stat_loo_knee_m=0.01,
            stat_loo_cap_m=0.8,
            ray_azimuth_count=1085,
            local_attn_heads=8,
            local_attn_layers=2,
            local_ffn_ratio=4,
            grad_balance_scope="token",
        ),
    )


def _ring_elevations():
    return [
        -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33,
        -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
         -9.33,  -8.00,  -6.66,  -5.33,  -4.00,  -2.67,  -1.33,   0.00,
          1.33,   2.67,   4.00,   5.33,   6.67,   8.00,   9.33,  10.67,
    ]


def test_learned_spherical_statistics_use_expected_ray_capacity_and_log_loo():
    torch.manual_seed(21)
    head = SphericalQueryHead(
        _learned_spherical_cfg(),
        dim=48,
        r_far=110.0,
        ring_to_elevation_deg=_ring_elevations(),
    )
    # Six rays intersect the exact plane x=10 inside one angular/radial cell.
    raw_points = torch.tensor([
        [10.0, 0.15, 0.15], [10.0, 0.25, 0.15], [10.0, 0.35, 0.15],
        [10.0, 0.15, 0.35], [10.0, 0.25, 0.35], [10.0, 0.35, 0.35],
    ])
    token_position = torch.tensor([
        [10.0, 0.2, 0.2], [10.0, 0.3, 0.3],
    ])
    feature = torch.randn(2, 48, requires_grad=True)
    pose = torch.eye(4).unsqueeze(0)
    empty_box = torch.empty(0, 7)
    empty_iid = torch.empty(0, dtype=torch.long)

    anchor_feature, seed_bank, delta_bank, anchor_offset, metadata = head(
        feature,
        token_position,
        raw_points,
        torch.tensor([0, 0, 0, 1, 1, 1]),
        torch.tensor([2]),
        torch.tensor([0]),
        [pose],
        [[empty_box]],
        [[empty_iid]],
    )
    assert anchor_feature.shape == (1, 48)
    assert seed_bank.shape == delta_bank.shape == (1, 4, 4, 3)
    assert anchor_offset.tolist() == [1]
    assert metadata["router_feature"].shape == (1, 112)
    assert metadata["neighbor_count"].tolist() == [1]
    assert metadata["stat_valid"].tolist() == [True]
    statistic = metadata["router_statistics"][0]
    capacity = metadata["anchor_ray_capacity"][0]
    # The occupied theta bin contains four physical rings. Its capacity uses
    # the phase-independent expectation 4 * 1085/120, not an arbitrary 9/10
    # integer azimuth-column allocation.
    torch.testing.assert_close(
        capacity, torch.tensor(4.0 * 1085.0 / 120.0)
    )
    torch.testing.assert_close(statistic[0], 6.0 / capacity)
    expected_range = (
        raw_points.norm(dim=-1).mean() - 2.5
    ) / (110.0 - 2.5)
    torch.testing.assert_close(statistic[1], expected_range)
    assert statistic[2:5].abs().max() < 1.0e-8
    assert statistic[5].item() == 1.0

    # Every active K-row seed is a real observed point; padded slots are zero.
    for k in range(1, 5):
        for slot in range(k):
            assert bool((raw_points == seed_bank[0, k - 1, slot]).all(dim=1).any())
        torch.testing.assert_close(
            seed_bank[0, k - 1, k:],
            torch.zeros_like(seed_bank[0, k - 1, k:]),
        )


def test_learned_spherical_loo_log_normalization_has_physical_knee_and_cap():
    head = SphericalQueryHead(
        _learned_spherical_cfg(),
        dim=48,
        r_far=110.0,
        ring_to_elevation_deg=_ring_elevations(),
    )
    residual_m = torch.tensor([0.0, 0.01, 0.10, 0.80, 2.00], dtype=torch.float64)
    normalized = head._normalize_loo_statistic(residual_m)
    expected = torch.log1p(
        residual_m.clamp(max=0.8) / 0.01
    ) / torch.log1p(torch.tensor(0.8 / 0.01, dtype=torch.float64))
    torch.testing.assert_close(normalized, expected)
    assert normalized[0].item() == 0.0
    assert normalized[-1].item() == 1.0


def test_learned_spherical_seed_rows_match_grid_range_quantiles():
    # Input/anchor order is deliberately interleaved and not range-sorted.
    # Every selected row is observed, and K-specific ranks match Grid mode.
    points = torch.tensor([
        [4.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    rows = _range_quantile_seed_rows(
        points,
        torch.tensor([0, 1, 0, 1, 0, 0]),
        num_anchors=2,
        k_max=4,
    )
    assert rows[0, 0, :1].tolist() == [4]          # r = 3
    assert rows[0, 1, :2].tolist() == [5, 0]       # r = 2, 4
    assert rows[0, 2, :3].tolist() == [2, 4, 0]    # r = 1, 3, 4
    assert rows[0, 3, :4].tolist() == [2, 5, 4, 0]
    # N_raw=2 < K repeats real points, exactly like Grid mode.
    assert rows[1, 3, :4].tolist() == [3, 3, 1, 1]


def test_learned_spherical_router_uses_statistics_but_k_head_uses_anchor_feature():
    from src.models_new.module.grid_query_head import GridSlotHead

    cfg = _learned_spherical_cfg()
    head = SphericalQueryHead(
        cfg, dim=48, r_far=110.0,
        ring_to_elevation_deg=_ring_elevations(),
    )
    slot_head = GridSlotHead(
        cfg,
        SimpleNamespace(shs=2, opacity=1, scaling=3, rotation=4, offset=3),
        dim=48,
        router_dim=head.router_dim,
    ).eval()
    # Deterministically select K=3 in eval mode.
    with torch.no_grad():
        slot_head.count_predictor[-1].bias[2] = 5.0

    class IdentityTemporalAggregator:
        def __call__(self, feat, position, seed, delta, *args):
            return feat, position, None, None, {}

    raw_points = torch.tensor([
        [10.0, 0.15, 0.15], [10.0, 0.25, 0.15], [10.0, 0.35, 0.15],
        [10.0, 0.15, 0.35], [10.0, 0.25, 0.35], [10.0, 0.35, 0.35],
    ])
    token_position = torch.tensor([
        [10.0, 0.2, 0.2], [10.0, 0.3, 0.3],
    ])
    feature = torch.randn(2, 48, requires_grad=True)
    pose = torch.eye(4).unsqueeze(0)
    seeds = build_spherical_gaussian_seeds(
        IdentityTemporalAggregator(),
        head,
        feature,
        token_position,
        raw_points,
        torch.tensor([0, 0, 0, 1, 1, 1]),
        torch.tensor([2]),
        torch.tensor([0]),
        [pose],
        [[torch.empty(0, 7)]],
        [[torch.empty(0, dtype=torch.long)]],
        None,
        slot_head=slot_head,
    )
    assert seeds.feature is None and seeds.raw_params.shape == (3, 13)
    assert seeds.position.shape == (3, 3)
    assert seeds.frame_offset.tolist() == [3]
    assert seeds.routing_stats["selected_k"].tolist() == [3]
    # The Gaussian trunk contract remains D + 3*Kmax, not router_dim + 3*Kmax.
    assert slot_head.trunk[0].in_features == 48 + 3 * 4
    assert slot_head.count_predictor[0].in_features == 48 + 64

    # In training, the selected activated-opacity ST gate must connect render
    # loss back through the statistic MLP and count router.
    slot_head.train()
    with torch.no_grad():
        slot_head.count_predictor[-1].weight.copy_(
            torch.linspace(
                -0.02, 0.02,
                slot_head.count_predictor[-1].weight.numel(),
            ).reshape_as(slot_head.count_predictor[-1].weight)
        )
    torch.manual_seed(22)
    train_seeds = build_spherical_gaussian_seeds(
        IdentityTemporalAggregator(),
        head,
        feature,
        token_position,
        raw_points,
        torch.tensor([0, 0, 0, 1, 1, 1]),
        torch.tensor([2]),
        torch.tensor([0]),
        [pose],
        [[torch.empty(0, 7)]],
        [[torch.empty(0, dtype=torch.long)]],
        None,
        slot_head=slot_head,
    )
    torch.sigmoid(train_seeds.raw_params[:, 2:3]).sum().backward()
    assert slot_head.count_predictor[-1].weight.grad is not None
    assert slot_head.count_predictor[-1].weight.grad.abs().sum() > 0
    assert head.stat_encoder[0].weight.grad is not None
    assert head.stat_encoder[0].weight.grad.abs().sum() > 0
    assert feature.grad is not None and feature.grad.abs().sum() > 0


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

    output, centre, delta, frame_offset, metadata = head(
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
    output, _, delta, _, _ = head(
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

    _, centre, delta, frame_offset, metadata = head(
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

    _, centre, _, frame_offset, metadata = head(
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

    _, seed, delta, _, _ = head(
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


def test_viewpoint_grid_head_stores_common_once_and_additional_per_view():
    token_feature = torch.randn(1, 4)
    token_position = torch.zeros(1, 3)
    candidate_seed = torch.zeros(1, 4, 4, 3)
    candidate_ref = torch.zeros_like(candidate_seed)
    for total_k in range(1, 5):
        values = 10 * total_k + torch.arange(total_k, dtype=torch.float32)
        candidate_seed[0, total_k - 1, :total_k, 0] = values
        candidate_ref[0, total_k - 1, :total_k, 0] = values + 100.0
    candidate_delta = candidate_seed.clone()
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
        count_mode = "learned_gumbel_viewpt"
        k_max = 4

        def __call__(
            self, feature, anchor_k, delta, token_offset,
            target_pose=None, anchor_batch=None, anchor_position_ref=None,
        ):
            assert target_pose is poses
            assert anchor_batch.tolist() == [0]
            # The router needs one ref-frame token center per target view, not
            # sensor-frame coordinates. This token is static, so both views
            # repeat the aggregator's observed center.
            assert anchor_position_ref.shape == (1, 2, 3)
            torch.testing.assert_close(
                anchor_position_ref,
                anchor_metadata["coord_ref"].unsqueeze(1).expand(1, 2, 3),
            )
            return torch.zeros(3, 6), {
                "anchor_index": torch.tensor([0, 0, 0]),
                "decision_index": torch.tensor([-1, 1, 1]),
                "decision_anchor_index": torch.tensor([0, 0]),
                "decision_view_index": torch.tensor([0, 1]),
                "view_index": torch.tensor([-1, 1, 1]),
                "is_common": torch.tensor([True, False, False]),
                "common_view_gate": torch.ones(1, 2),
                "slot_index": torch.tensor([0, 1, 2]),
                "slot_k": torch.tensor([1, 3, 3]),
                "anchor_k": torch.tensor([1, 3]),
                "gaussian_offset": torch.tensor([3]),
                "k_logits": torch.zeros(2, 4, requires_grad=True),
                "k_selection": torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]),
                "view_dependent": True,
            }

        @staticmethod
        def gradient_weight(slot_k, dtype):
            return torch.ones_like(slot_k, dtype=dtype)

    poses = [torch.eye(4).repeat(2, 1, 1)]
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
        target_pose=poses,
    )

    torch.testing.assert_close(
        seeds.position[:, 0], torch.tensor([10.0, 31.0, 32.0])
    )
    torch.testing.assert_close(
        seeds.metadata["coord_ref"][:, 0],
        torch.tensor([110.0, 131.0, 132.0]),
    )
    assert seeds.metadata["view_index"].tolist() == [-1, 1, 1]
    assert seeds.metadata["common_view_gate"].shape == (1, 2)
    assert seeds.routing_stats["selected_k"].tolist() == [1, 3]
    assert seeds.routing_stats["view_index"].tolist() == [0, 1]


def test_viewpoint_router_receives_each_token_in_the_target_sensor_frame():
    """End-to-end seam test: real aggregator -> real viewpoint slot head.

    The router input must be the token center expressed in the *target* sensor
    frame, which composes the source-frame pose with the inverse target pose.
    Feeding raw sensor coordinates would silently work for frame 0 only.
    """
    from src.models_new.module.grid_query_head import GridSlotHead, GridTemporalAggregator

    torch.manual_seed(5)
    k_max, dim = 4, 48
    aggregator = GridTemporalAggregator(
        SimpleNamespace(
            K_max=3, points_per_gaussian=4, bg_radius_m=0.8,
            bg_max_kv_per_frame=8, num_heads=2, bg_layers=1, fg_layers=1,
            grad_balance="sqrt_k",
        ),
        dim=dim, r_far=80.0,
    )
    head = GridSlotHead(
        SimpleNamespace(
            count_mode="learned_gumbel_viewpt",
            grad_balance="sqrt_k",
            learned_count=SimpleNamespace(
                K_max=k_max, tau=1.0, grad_balance_scope="token",
                viewpoint=SimpleNamespace(
                    position_frequencies=8, position_scale_m=110.0,
                ),
            ),
        ),
        SimpleNamespace(shs=2, opacity=1, scaling=1, rotation=1, offset=1),
        dim=dim,
    )

    captured = {}

    def capture_router_position(module, args):
        captured["position"] = args[0]

    head.view_position_encoder.register_forward_pre_hook(capture_router_position)

    # Frame 1's sensor sits 2 m ahead of frame 0, and target view 1 shares that
    # pose, so the second view must see frame-1 tokens at their sensor coords.
    shifted = torch.eye(4)
    shifted[0, 3] = 2.0
    pose = torch.stack([torch.eye(4), shifted])
    # Frame 1's sensor coords are chosen so both frames land within the 0.8 m
    # background radius in the ref frame, exercising the real fusion attention.
    token_position = torch.tensor([
        [10.0, 0.0, 0.0], [30.0, 0.0, 0.0],
        [8.3, 0.0, 0.0], [28.0, 0.0, 0.0],
    ])
    seed_sensor = token_position[:, None, None, :].repeat(1, k_max, k_max, 1)
    feature = torch.randn(4, dim, requires_grad=True)

    seeds = build_grid_gaussian_seeds(
        aggregator, head, feature, token_position, None,
        seed_sensor, torch.zeros_like(seed_sensor),
        torch.tensor([2, 4]), torch.tensor([0, 0]),
        [pose], [[torch.empty(0, 7), torch.empty(0, 7)]], None,
        [torch.tensor([0.0, 1.0])],
        target_pose=[pose],
    )

    # Rows are (token, view) in anchor-major order for 4 tokens x 2 views.
    expected = torch.tensor([
        [10.0, 0.0, 0.0], [8.0, 0.0, 0.0],     # frame 0, ref == sensor
        [30.0, 0.0, 0.0], [28.0, 0.0, 0.0],
        [10.3, 0.0, 0.0], [8.3, 0.0, 0.0],     # frame 1, ref == sensor + 2 m
        [30.0, 0.0, 0.0], [28.0, 0.0, 0.0],
    ])
    view_position = head.view_position_encoder.to_view_frame(
        captured["position"], pose.repeat(4, 1, 1)
    )
    torch.testing.assert_close(view_position, expected)

    assert seeds.metadata["common_view_gate"].shape == (4, 2)
    seeds.raw_params.sum().backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert aggregator.bg_attention.kv_proj[0].weight.grad.abs().sum() > 0


def test_dynamic_router_position_matches_the_renderer_box_trajectory():
    """A dynamic token must reach the router where the renderer will draw it."""
    from src.models_new.module.grid_query_head import GridTemporalAggregator
    from src.models_new.module.m3_g2p import GausRender
    from src.models_new.module.anchor_modes import target_view_token_positions

    torch.manual_seed(9)
    aggregator = GridTemporalAggregator(
        SimpleNamespace(
            K_max=3, points_per_gaussian=4, bg_radius_m=0.8,
            bg_max_kv_per_frame=8, num_heads=2, bg_layers=1, fg_layers=1,
            grad_balance="sqrt_k",
        ),
        dim=48, r_far=80.0,
    )

    # One car driving +x and yawing, observed at t=0.0 and t=1.0. Boxes are
    # [x, y, z, w, l, h, yaw]; the token sits 0.5 m ahead of the box center.
    box_t0 = torch.tensor([[20.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0]])
    box_t1 = torch.tensor([[32.0, 3.0, 0.0, 2.0, 4.0, 2.0, 0.6]])
    token_position = torch.tensor([
        [20.5, 0.0, 0.0], [5.0, 0.0, 0.0],       # frame 0: on the car, background
        [32.5, 3.0, 0.0], [5.0, 0.0, 0.0],       # frame 1
    ])
    pose = torch.eye(4).repeat(2, 1, 1)
    source_times = torch.tensor([0.0, 1.0])
    instance_ids = [[torch.tensor([7]), torch.tensor([7])]]

    _, anchor_position_out, _, _, anchor_metadata = aggregator(
        torch.randn(4, 48), token_position, None, None,
        torch.tensor([2, 4]), torch.tensor([0, 0]),
        [pose], [[box_t0, box_t1]], instance_ids, [source_times],
    )
    assert anchor_metadata["is_dynamic"].tolist() == [True, False, True, False]

    target_times = torch.tensor([0.0, 0.25, 1.0])
    positions = target_view_token_positions(
        anchor_metadata,
        anchor_position_out,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 0]),
        instance_ids,
        [source_times],
        [target_times],
        num_views=3,
    )
    assert positions.shape == (4, 3, 3)

    # Background rows never move.
    for row in (1, 3):
        torch.testing.assert_close(
            positions[row],
            anchor_metadata["coord_ref"][row].expand(3, 3),
        )

    # Dynamic rows must match the renderer's own interpolate + replay, term for
    # term, so the router and the rasterizer never disagree about where the car
    # is at a given render time.
    trajectory = {
        "timestamps": source_times,
        "bbox_ref": torch.cat([box_t0, box_t1]),
    }
    renderer = GausRender.__new__(GausRender)
    for view, t in enumerate(target_times.tolist()):
        box_t = renderer.interpolate_box_ref(
            trajectory, t, positions.device, positions.dtype
        )
        for row in (0, 2):
            expected = GausRender.box_local_to_ref(
                anchor_position_out[row].unsqueeze(0), box_t
            )
            torch.testing.assert_close(positions[row, view], expected[0])

    # At the observed times the replay reproduces the observed ref center.
    torch.testing.assert_close(
        positions[0, 0], anchor_metadata["coord_ref"][0]
    )
    torch.testing.assert_close(
        positions[2, 2], anchor_metadata["coord_ref"][2]
    )
    # A mid-window view genuinely moves the car; the old static input would have
    # handed the router a position several metres off.
    assert (positions[0, 1] - positions[0, 0]).norm() > 2.0
