"""Compare the working Dynamic 2DGS split against the pre-refactor Git HEAD.

This is an explicit review tool rather than an ordinary unit test because it
loads the historical monolith directly from reviewed baseline commit
``a44d941a236948b1d9248936deaf2616d12de453``. It composes the current variant
overlays for both implementations, then checks initialization, checkpoint
structure, strict restore, forward values, and backward gradients.
"""
from __future__ import annotations

import gc
import importlib
import json
import pathlib
import subprocess
import sys
import types

import torch
from omegaconf import OmegaConf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import DYNAMIC_VARIANTS, compose_fresh_config
from tests.test_checkpoint_compatibility import (
    _BACKEND_CLASS,
    _PROPOSAL_VARIANTS,
)


DIM = 144
BASELINE_COMMIT = "a44d941a236948b1d9248936deaf2616d12de453"
OUTPUT_FIELDS = ("position", "shs", "opacity", "scaling", "rotation", "velocity")


def _load_modules():
    package = types.ModuleType("src.models_new.module")
    package.__path__ = [
        str(REPO_ROOT / "src/models_new/module")
    ]
    package.__package__ = package.__name__
    sys.modules[package.__name__] = package

    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASELINE_COMMIT}:src/models_new/module/dynamic_gaussian.py",
        ],
        text=True,
    )
    head = types.ModuleType("src.models_new.module._head_dynamic_gaussian")
    head.__file__ = (
        f"<{BASELINE_COMMIT}:src/models_new/module/dynamic_gaussian.py>"
    )
    head.__package__ = "src.models_new.module"
    sys.modules[head.__name__] = head
    exec(compile(source, head.__file__, "exec"), head.__dict__)
    current = importlib.import_module("src.models_new.module.dynamic_gaussian")
    return head, current


def _constructor_inputs(variant):
    config, _source = compose_fresh_config(
        OmegaConf.from_dotlist([f"model.variant={variant}"])
    )
    head_kwargs = {"proposal_dim": DIM} if variant in _PROPOSAL_VARIANTS else {}
    current_kwargs = dict(head_kwargs)
    count = getattr(config.p2g, "grid_query", None)
    adaptive = (
        count is not None
        and str(getattr(count, "count_mode", "legacy")).lower()
        == "learned_gumbel"
    )
    if adaptive:
        head_kwargs["gaussian_count_cfg"] = count
        current_kwargs["gaussian_count_cfg"] = count
    else:
        current_kwargs["gaussians_per_token"] = int(getattr(count, "K_max", 1))
    positional = (
        config.dynamic_2dgs,
        config.p2g.gs_params,
        DIM,
        float(config.p2g.head_offset_bound),
    )
    return positional, head_kwargs, current_kwargs, adaptive


def _adaptive_seed_bank(position):
    bank = position.new_zeros(position.shape[0], 3, 3, 3)
    delta = torch.zeros_like(bank)
    choices = (
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        torch.tensor([
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]),
    )
    for index, offsets in enumerate(choices):
        k = index + 1
        bank[:, index, :k] = position[:, None, :] + offsets
        delta[:, index, :k] = offsets
    return bank, delta


def _run(model, variant, adaptive, feature, common, seed_delta):
    input_feature = feature.clone().requires_grad_()
    kwargs = {}
    proposal_feature = None
    if variant in _PROPOSAL_VARIANTS:
        proposal_feature = (feature * 0.7 + 0.1).clone().requires_grad_()
        kwargs["motion_proposal_feature"] = proposal_feature
    if adaptive:
        kwargs["seed_delta_sensor"] = seed_delta
    torch.manual_seed(456)
    output = model(input_feature, *common, **kwargs)["batch_gaussians"][0]
    sum(output[name].float().sum() for name in OUTPUT_FIELDS).backward()
    parameter_grad = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    proposal_grad = (
        None if proposal_feature is None else proposal_feature.grad.detach().clone()
    )
    return output, input_feature.grad.detach(), proposal_grad, parameter_grad


def _close(left, right):
    return bool(
        torch.isfinite(left).all()
        and torch.isfinite(right).all()
        and torch.allclose(left, right, atol=1.0e-6, rtol=1.0e-6)
    )


def _optional_gradients_match(left, right):
    if (left is None) != (right is None):
        return False
    return left is None or _close(left, right)


def main():
    head_module, current_module = _load_modules()
    report = {}
    for variant in DYNAMIC_VARIANTS:
        positional, head_kwargs, current_kwargs, adaptive = _constructor_inputs(
            variant
        )
        class_name = _BACKEND_CLASS[variant]
        torch.manual_seed(777)
        head = getattr(head_module, class_name)(*positional, **head_kwargs).eval()
        torch.manual_seed(777)
        current = getattr(current_module, class_name)(
            *positional, **current_kwargs
        ).eval()
        head_state = head.state_dict()
        current_state = current.state_dict()
        state_schema = [
            (name, tuple(value.shape), value.dtype)
            for name, value in head_state.items()
        ] == [
            (name, tuple(value.shape), value.dtype)
            for name, value in current_state.items()
        ]
        initialized_values = state_schema and all(
            torch.equal(value, current_state[name])
            for name, value in head_state.items()
        )
        parameter_order = [name for name, _ in head.named_parameters()] == [
            name for name, _ in current.named_parameters()
        ]
        incompatible = current.load_state_dict(head_state, strict=True)
        strict_restore = not (
            incompatible.missing_keys or incompatible.unexpected_keys
        )

        torch.manual_seed(123)
        feature = torch.randn(4, DIM)
        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.2, 0.0],
        ])
        if adaptive:
            seed, seed_delta = _adaptive_seed_bank(position)
        else:
            seed = position + torch.tensor([0.02, 0.01, 0.0])
            seed_delta = None
        common = (
            position,
            seed,
            torch.tensor([2, 4]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
        )
        head_run = _run(head, variant, adaptive, feature, common, seed_delta)
        current_run = _run(
            current, variant, adaptive, feature, common, seed_delta
        )
        output_values = all(
            _close(head_run[0][name], current_run[0][name])
            for name in OUTPUT_FIELDS
        )
        feature_gradient = _close(head_run[1], current_run[1])
        proposal_gradient = _optional_gradients_match(head_run[2], current_run[2])
        parameter_gradients = all(
            _optional_gradients_match(gradient, current_run[3][name])
            for name, gradient in head_run[3].items()
        )
        report[variant] = {
            "state_schema": state_schema,
            "seeded_initialization": initialized_values,
            "parameter_order": parameter_order,
            "strict_restore": strict_restore,
            "forward": output_values,
            "feature_gradient": feature_gradient,
            "proposal_gradient": proposal_gradient,
            "parameter_gradients": parameter_gradients,
        }
        del head, current, head_run, current_run
        gc.collect()

    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(all(checks.values()) for checks in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
