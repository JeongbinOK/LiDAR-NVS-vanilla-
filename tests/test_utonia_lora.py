from types import SimpleNamespace

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from src.models_new.utonia.lora import (
    LoRALinear,
    LoRASubMConv3d,
    adapt_utonia_embedding_to_xyzi,
    build_xyzi_utonia_input,
    freeze_utonia_and_enable_active_lora,
    inject_utonia_lora,
    resolve_utonia_lora_settings,
)


def _settings(**overrides):
    values = {
        "enable": True,
        "rank": 16,
        "alpha": 16.0,
        "dropout": 0.0,
        "input_mode": "xyzi",
    }
    values.update(overrides)
    return resolve_utonia_lora_settings(
        SimpleNamespace(utonia_lora=SimpleNamespace(**values))
    )


def test_xyzi_stem_migration_preserves_pretrained_xyz_function():
    torch.manual_seed(3)
    backbone = nn.Module()
    backbone.embedding = nn.Module()
    backbone.embedding.in_channels = 9
    backbone.embedding.stem = nn.Module()
    backbone.embedding.stem.linear = nn.Linear(9, 7)
    original = backbone.embedding.stem.linear

    xyz = torch.randn(5, 3)
    intensity = torch.rand(5, 1)
    expected = original(torch.cat([xyz, torch.zeros(5, 6)], dim=-1))

    old_width = adapt_utonia_embedding_to_xyzi(backbone)
    migrated = backbone.embedding.stem.linear
    actual = migrated(torch.cat([xyz, intensity], dim=-1))

    assert old_width == 9
    assert migrated.in_features == 4
    assert backbone.embedding.in_channels == 4
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(migrated.weight[:, 3]) == 0


def test_xyzi_input_is_aligned_and_does_not_mutate_collated_feat():
    coord = torch.randn(4, 3)
    strength = torch.rand(4)
    legacy_feat = torch.randn(4, 9)
    source = {"coord": coord, "strength": strength, "feat": legacy_feat}

    result = build_xyzi_utonia_input(source)

    assert result is not source
    assert source["feat"] is legacy_feat
    torch.testing.assert_close(result["feat"][:, :3], coord)
    torch.testing.assert_close(result["feat"][:, 3], strength)


def test_all_linear_adapters_are_zero_init_noops_and_only_lora_is_trainable():
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.GELU(),
        nn.Sequential(nn.Linear(12, 6), nn.Linear(6, 4)),
    )
    feature = torch.randn(3, 8)
    expected = model(feature).detach()

    report = inject_utonia_lora(model, _settings())
    trainable = freeze_utonia_and_enable_active_lora(model, [model])
    actual = model(feature)

    assert report.linear_modules == 3
    assert report.conv_modules == 0
    assert trainable > 0
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert module.base_layer.weight.grad is None
            assert module.lora_up.weight.grad is not None
            assert module.lora_down.weight.requires_grad


def test_inactive_stage_adapters_are_installed_but_not_trainable():
    model = nn.Module()
    model.active = nn.Sequential(nn.Linear(8, 8))
    model.inactive = nn.Sequential(nn.Linear(8, 8))

    report = inject_utonia_lora(model, _settings())
    freeze_utonia_and_enable_active_lora(model, [model.active])

    assert report.linear_modules == 2
    assert isinstance(model.active[0], LoRALinear)
    assert isinstance(model.inactive[0], LoRALinear)
    assert model.active[0].lora_up.weight.requires_grad
    assert not model.inactive[0].lora_up.weight.requires_grad


def test_sparse_conv_lora_uses_spatial_down_then_pointwise_up():
    model = nn.Module()
    model.conv = spconv.SubMConv3d(
        8, 12, kernel_size=3, bias=True, indice_key="unit-stage"
    )

    report = inject_utonia_lora(model, _settings())

    assert report.linear_modules == 0
    assert report.conv_modules == 1
    assert isinstance(model.conv, LoRASubMConv3d)
    assert model.conv.lora_down.in_channels == 8
    assert model.conv.lora_down.out_channels == 16
    assert tuple(model.conv.lora_down.kernel_size) == (3, 3, 3)
    assert model.conv.lora_up.in_channels == 16
    assert model.conv.lora_up.out_channels == 12
    assert tuple(model.conv.lora_up.kernel_size) == (1, 1, 1)
    assert torch.count_nonzero(model.conv.lora_up.weight) == 0
