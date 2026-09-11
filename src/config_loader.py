"""Compose experiment configs and restore checkpoint-local configs safely.

The project intentionally keeps OmegaConf dot-list CLI overrides (``key=value``)
instead of introducing a second configuration framework. Fresh runs compose a
shared base with one allow-listed model-variant overlay. Checkpoint-backed runs
prefer embedded configs and retain the historical W&B fallback.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .experiment_config.definitions import (
    ADAPTIVE_COUNT_DYNAMIC_VARIANTS,
    ATTENTION_VELOCITY_VARIANTS,
    BARRIER_MATCH_DYNAMIC_VARIANTS,
    DEFAULT_CONFIG_PATH,
    DURATION_OFFSET_VELOCITY_VARIANTS,
    DYNAMIC_VARIANT,
    DYNAMIC_VARIANT_V1,
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V6,
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
    DYNAMIC_VARIANT_V8,
    DYNAMIC_VARIANT_V9,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    DYNAMIC_VARIANT_V11_2,
    DYNAMIC_VARIANT_V11_3,
    DYNAMIC_VARIANT_V11_4,
    DYNAMIC_VARIANTS,
    LEARNED_COUNT_MODES,
    LEGACY_COUNT_MODE,
    LEGACY_VARIANT,
    PHYSICAL_VELOCITY_VARIANTS,
    PROPOSAL_VELOCITY_VARIANTS,
    REFINED_PROPOSAL_VELOCITY_VARIANTS,
    REPO_ROOT,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    STRUCTURAL_DEFAULTS,
    VARIANT_CONFIG_PATHS,
    WARPED_PROPOSAL_VELOCITY_VARIANTS,
    _CHECKPOINT_PROTECTED_PATHS,
    _CHECKPOINT_STRUCTURAL_ROOTS,
    _IMPLEMENTED_VARIANTS,
    _REQUIRED_ROOTS,
    _SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS,
    _UTONIA_ENCODER_STAGE_DIMS,
)
from .experiment_config.helpers import (
    _as_config,
    _as_experiment_config,
    _has_path,
    _is_experiment_config,
    _reject_unknown_keys,
    _require_choice,
    _require_in_open_range,
    _require_positive,
    _variant,
)
from .experiment_config.validation import (
    _configured_count_mode,
    _configured_p2g_trunk_dim,
    _validate_3d_rope_width,
    _validate_adaptive_gaussian_count,
    _validate_dynamic_gaussian_count,
    _validate_fixed_gaussian_count,
    _validate_utonia_lora_config,
    _variant_label,
    assert_model_variant_implemented,
    validate_experiment_config,
)


def _load_variant_overlay(variant: str) -> tuple[DictConfig, Path]:
    try:
        path = VARIANT_CONFIG_PATHS[variant]
    except KeyError as exc:
        allowed = ", ".join(sorted(VARIANT_CONFIG_PATHS))
        raise ValueError(
            f"Unknown model.variant={variant!r}; expected one of: {allowed}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Variant config does not exist: {path}")
    overlay = OmegaConf.load(path)
    declared = _variant(overlay, default="")
    if declared != variant:
        raise ValueError(
            f"Variant overlay {path} declares model.variant={declared!r}, "
            f"expected {variant!r}"
        )
    return overlay, path


def compose_fresh_config(cli=None) -> tuple[DictConfig, str]:
    """Compose defaults -> shared base -> selected variant -> CLI overrides."""

    cli = _as_config(cli)
    base = OmegaConf.load(DEFAULT_CONFIG_PATH)
    selected = _variant(cli, default=_variant(base))
    overlay, overlay_path = _load_variant_overlay(selected)
    config = OmegaConf.merge(
        OmegaConf.create(STRUCTURAL_DEFAULTS),
        base,
        overlay,
        cli,
    )
    if _variant(config) != selected:
        raise ValueError(
            "The composed model.variant changed after selecting its overlay: "
            f"selected={selected!r}, composed={_variant(config)!r}"
        )
    validate_experiment_config(config)
    source = f"base={DEFAULT_CONFIG_PATH.resolve()} overlay={overlay_path.resolve()}"
    return config, source


def _load_embedded_checkpoint_config(checkpoint: Path):
    """Read config metadata embedded by ``ModelWrapper.on_save_checkpoint``."""

    try:
        import torch

        try:
            blob = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            blob = torch.load(checkpoint, map_location="cpu")
    except Exception:
        # Historical checkpoints and lightweight test fixtures may not contain
        # readable checkpoint metadata. Their W&B artifact is the fallback.
        return None

    if not isinstance(blob, Mapping) or "experiment_config" not in blob:
        return None
    config = _as_experiment_config(blob["experiment_config"])
    if config is None:
        raise RuntimeError(
            f"Checkpoint contains invalid experiment_config metadata: {checkpoint}"
        )
    return config


def load_checkpoint_config(
    checkpoint,
    *,
    allow_legacy_wandb_fallback: bool = False,
) -> tuple[DictConfig, str]:
    """Load checkpoint-bound config, with a historical W&B fallback."""

    checkpoint = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    embedded = _load_embedded_checkpoint_config(checkpoint)
    if embedded is not None:
        config = OmegaConf.merge(OmegaConf.create(STRUCTURAL_DEFAULTS), embedded)
        validate_experiment_config(config)
        return config, f"embedded checkpoint config: {checkpoint}"

    if not allow_legacy_wandb_fallback:
        raise RuntimeError(
            "Historical checkpoint has no embedded experiment config. Its only "
            "available config source is the mutable wandb/latest-run link, which "
            "may belong to a later resumed run. Inspect that artifact and pass "
            "allow_legacy_wandb_fallback=true to opt in explicitly."
        )

    config_file = checkpoint.parent / "wandb/latest-run/files/config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            "Checkpoint execution needs its run-local W&B config, but it was "
            f"not found: {config_file}"
        )

    raw = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"Unexpected W&B config format: {config_file}")
    unwrapped = {
        key: value["value"]
        if isinstance(value, Mapping) and "value" in value
        else value
        for key, value in raw.items()
        if key != "_wandb"
    }
    stored = _as_experiment_config(unwrapped)
    if stored is None:
        raise RuntimeError(
            f"W&B config does not contain a valid experiment config: {config_file}"
        )

    config = OmegaConf.merge(OmegaConf.create(STRUCTURAL_DEFAULTS), stored)
    validate_experiment_config(config)
    return config, (
        "legacy W&B fallback (mutable latest-run): "
        f"{config_file.resolve()}"
    )


def _merge_checkpoint_cli(stored, cli) -> DictConfig:
    cli = _as_config(cli)
    config = OmegaConf.merge(stored, cli)
    changed_roots = []
    for root in _CHECKPOINT_STRUCTURAL_ROOTS:
        if root not in cli:
            continue
        before = OmegaConf.select(stored, root, default=None)
        after = OmegaConf.select(config, root, default=None)
        before_value = (
            OmegaConf.to_container(before, resolve=True)
            if OmegaConf.is_config(before)
            else before
        )
        after_value = (
            OmegaConf.to_container(after, resolve=True)
            if OmegaConf.is_config(after)
            else after
        )
        if before_value != after_value:
            changed_roots.append(root)
    changed_paths = []
    for path in _CHECKPOINT_PROTECTED_PATHS:
        if not _has_path(cli, path):
            continue
        before = OmegaConf.select(stored, path, default=None)
        after = OmegaConf.select(config, path, default=None)
        if before != after:
            changed_paths.append(path)
    if changed_roots or changed_paths:
        details = []
        if changed_roots:
            details.append("roots: " + ", ".join(changed_roots))
        if changed_paths:
            details.append("paths: " + ", ".join(changed_paths))
        raise ValueError(
            "Checkpoint structural config cannot be changed by CLI; changed "
            + "; ".join(details)
        )
    validate_experiment_config(config)
    return config


def _resolve_with_checkpoint(cli, checkpoint_key: str) -> tuple[DictConfig, str]:
    cli = _as_config(cli)
    checkpoint = OmegaConf.select(cli, checkpoint_key, default=None)
    if not checkpoint:
        return compose_fresh_config(cli)
    allow_legacy = bool(
        OmegaConf.select(cli, "allow_legacy_wandb_fallback", default=False)
    )
    stored, source = load_checkpoint_config(
        checkpoint,
        allow_legacy_wandb_fallback=allow_legacy,
    )
    return _merge_checkpoint_cli(stored, cli), source


def resolve_main_config(cli=None) -> tuple[DictConfig, str]:
    """Resolve train/test entrypoint config, including exact resume configs."""

    cli = _as_config(cli)
    mode = str(OmegaConf.select(cli, "mode", default="train")).lower()
    checkpoint_key = "train.ckpt_path" if mode == "train" else "test.ckpt_path"
    return _resolve_with_checkpoint(cli, checkpoint_key)


def resolve_eval_config(cli=None) -> tuple[DictConfig, str]:
    return _resolve_with_checkpoint(cli, "test.ckpt_path")

__all__ = [
    "ADAPTIVE_COUNT_DYNAMIC_VARIANTS",
    "BARRIER_MATCH_DYNAMIC_VARIANTS",
    "DEFAULT_CONFIG_PATH",
    "ATTENTION_VELOCITY_VARIANTS",
    "DYNAMIC_VARIANT",
    "DYNAMIC_VARIANT_V1",
    "DYNAMIC_VARIANT_V3",
    "DYNAMIC_VARIANT_V3_1",
    "DYNAMIC_VARIANT_V4",
    "DYNAMIC_VARIANT_V5",
    "DYNAMIC_VARIANT_V6",
    "DYNAMIC_VARIANT_V7",
    "DYNAMIC_VARIANT_V7_1",
    "DYNAMIC_VARIANT_V7_2",
    "DYNAMIC_VARIANT_V8",
    "DYNAMIC_VARIANT_V9",
    "DYNAMIC_VARIANT_V10",
    "DYNAMIC_VARIANT_V11",
    "DYNAMIC_VARIANT_V11_1",
    "DYNAMIC_VARIANT_V11_2",
    "DYNAMIC_VARIANT_V11_3",
    "DYNAMIC_VARIANT_V11_4",
    "DYNAMIC_VARIANTS",
    "DURATION_OFFSET_VELOCITY_VARIANTS",
    "LEGACY_VARIANT",
    "PHYSICAL_VELOCITY_VARIANTS",
    "PROPOSAL_VELOCITY_VARIANTS",
    "REFINED_PROPOSAL_VELOCITY_VARIANTS",
    "ROUTER_CAPABLE_DYNAMIC_VARIANTS",
    "WARPED_PROPOSAL_VELOCITY_VARIANTS",
    "STRUCTURAL_DEFAULTS",
    "VARIANT_CONFIG_PATHS",
    "assert_model_variant_implemented",
    "compose_fresh_config",
    "load_checkpoint_config",
    "resolve_eval_config",
    "resolve_main_config",
    "validate_experiment_config",
]
