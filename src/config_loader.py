"""Compose experiment configs and restore checkpoint-local configs safely.

The project intentionally keeps OmegaConf dot-list CLI overrides (``key=value``)
instead of introducing a second configuration framework.  Fresh runs compose a
shared base with one allow-listed model-variant overlay.  Checkpoint-backed runs
prefer the immutable config embedded in new checkpoints and never mix in the
checkout's current overlay.  Historical checkpoints fall back to W&B metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "nuscene_train.yaml"

LEGACY_VARIANT = "bbox_rigid_v1"
DYNAMIC_VARIANT_V1 = "dynamic_2dgs_direct_velocity_v1"
DYNAMIC_VARIANT_V3 = "dynamic_2dgs_physical_velocity_v3"
DYNAMIC_VARIANT_V3_1 = "dynamic_2dgs_physical_velocity_v3_1"
DYNAMIC_VARIANT_V4 = "dynamic_2dgs_attention_velocity_v4"
DYNAMIC_VARIANT = "dynamic_2dgs_attention_velocity_v5"
# V4 and V5 share the attention-initialized velocity backend; V5 only adds
# QK-Norm and a finer motion-head RoPE band on top of the same parameters.
ATTENTION_VELOCITY_VARIANTS = (DYNAMIC_VARIANT_V4, DYNAMIC_VARIANT)
DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V1,
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT,
)

# V3 and V3.1 share one architecture. V4 keeps their physical-time contract and
# adds attention-derived velocity initialization.
PHYSICAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT,
)

VARIANT_CONFIG_PATHS = {
    LEGACY_VARIANT: REPO_ROOT / "config" / "variants" / f"{LEGACY_VARIANT}.yaml",
    DYNAMIC_VARIANT_V1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V1}.yaml"
    ),
    DYNAMIC_VARIANT_V3: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V3}.yaml"
    ),
    DYNAMIC_VARIANT_V3_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V3_1}.yaml"
    ),
    DYNAMIC_VARIANT_V4: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V4}.yaml"
    ),
    DYNAMIC_VARIANT: REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT}.yaml",
}

# W&B omits empty mappings, and configs saved before model variants existed have
# no model block.  Left-merging these defaults restores the historical contract.
STRUCTURAL_DEFAULTS = {
    "model": {"variant": LEGACY_VARIANT},
    "g2g": {},
}

_REQUIRED_ROOTS = (
    "model",
    "p2g",
    "g2g",
    "g2p",
    "data",
    "loss",
    "train",
    "test",
    "logger",
)

# These roots control model construction or shape-neutral rendering semantics.
# Runtime knobs can change during resume/evaluation, but saved weights must not
# be reinterpreted by modifying one of these subtrees from the CLI.
_CHECKPOINT_STRUCTURAL_ROOTS = (
    "model",
    "p2g",
    "g2g",
    "g2p",
    "dynamic_2dgs",
)

_CHECKPOINT_PROTECTED_PATHS = (
    "data.window_us",
    "data.sample_gap_us",
    "data.gt_middle_count",
    "data.pair_mode",
    "data.pair_kf_stride",
    "data.mode",
    "data.bbox_json_path",
    "data.bbox_json_paths",
    "data.vfov",
    "data.hfov",
    "data.image_height",
    "data.image_width",
    "data.ring_to_elevation_deg",
    "data.ego_radius",
)

_IMPLEMENTED_VARIANTS = {LEGACY_VARIANT, *DYNAMIC_VARIANTS}


def _as_config(value=None) -> DictConfig:
    if value is None:
        return OmegaConf.create()
    if OmegaConf.is_config(value):
        return value
    return OmegaConf.create(value)


def _has_path(config, dotted_path: str) -> bool:
    current = config
    for part in dotted_path.split("."):
        if not OmegaConf.is_dict(current) or part not in current:
            return False
        current = current[part]
    return True


def _variant(config, *, default=LEGACY_VARIANT) -> str:
    value = OmegaConf.select(config, "model.variant", default=default)
    return str(value)


def _reject_unknown_keys(config, dotted_path: str, allowed) -> None:
    block = OmegaConf.select(config, dotted_path, default=None)
    if block is None or not OmegaConf.is_dict(block):
        raise ValueError(f"{dotted_path} must be a config mapping")
    unknown = sorted(set(block.keys()) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unsupported keys under {dotted_path}: {', '.join(unknown)}"
        )


def _validate_3d_rope_width(dim: int, num_heads: int, label: str) -> None:
    if dim % num_heads != 0:
        raise ValueError(f"{label}: dim={dim} must be divisible by heads={num_heads}")
    head_dim = dim // num_heads
    if head_dim % 6 != 0:
        raise ValueError(
            f"{label}: head_dim={head_dim} must be divisible by six for "
            "the xyz/sin-cos 3D RoPE split"
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


def validate_experiment_config(config) -> None:
    missing = [key for key in _REQUIRED_ROOTS if key not in config]
    if missing:
        raise ValueError("Experiment config is missing root keys: " + ", ".join(missing))

    variant = _variant(config)
    if variant not in VARIANT_CONFIG_PATHS:
        allowed = ", ".join(sorted(VARIANT_CONFIG_PATHS))
        raise ValueError(
            f"Unknown model.variant={variant!r}; expected one of: {allowed}"
        )

    if variant in DYNAMIC_VARIANTS:
        if str(OmegaConf.select(config, "p2g.anchor_mode", default="")) != "grid":
            raise ValueError(f"{variant} requires p2g.anchor_mode=grid")
        if OmegaConf.select(config, "dynamic_2dgs", default=None) is None:
            raise ValueError(f"{variant} requires a dynamic_2dgs config block")
        trunk_dim = int(OmegaConf.select(config, "p2g.agg_mlp.out_dim"))
        temporal_heads = int(OmegaConf.select(
            config, "dynamic_2dgs.temporal.num_heads"
        ))
        _validate_3d_rope_width(
            trunk_dim, temporal_heads, "dynamic temporal attention"
        )
        if bool(OmegaConf.select(
            config, "p2g.joint_refiner.enable", default=False
        )):
            joint_heads = int(OmegaConf.select(
                config, "p2g.joint_refiner.num_heads"
            ))
            _validate_3d_rope_width(
                trunk_dim, joint_heads, "post-fusion joint attention"
            )
        if variant in PHYSICAL_VELOCITY_VARIANTS:
            _reject_unknown_keys(
                config,
                "dynamic_2dgs",
                {"temporal", "gaussian_head", "motion", "regularization"},
            )
            temporal_keys = {
                "implementation", "layers", "num_heads", "mlp_ratio",
                "time_embedding_dim", "time_frequencies", "rope_base",
                "rope_position_scale", "time_reference_sec",
                "layer_scale_init",
            }
            if variant in ATTENTION_VELOCITY_VARIANTS:
                temporal_keys.add("motion_head_count")
            if variant == DYNAMIC_VARIANT:
                temporal_keys.update({
                    "qk_norm", "motion_rope_base", "motion_rope_position_scale",
                })
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.temporal",
                temporal_keys,
            )
            if variant in ATTENTION_VELOCITY_VARIANTS:
                motion_heads = int(OmegaConf.select(
                    config, "dynamic_2dgs.temporal.motion_head_count"
                ))
                if not 0 < motion_heads <= temporal_heads:
                    raise ValueError(
                        "dynamic_2dgs.temporal.motion_head_count must be in "
                        f"[1, {temporal_heads}]"
                    )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.gaussian_head",
                {"initial_opacity", "initial_scale_m"},
            )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.motion",
                {"zero_init"},
            )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.regularization",
                {
                    "small_motion",
                    "velocity_l2",
                },
            )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.regularization.small_motion",
                {"enabled", "weight", "epsilon_m", "warmup_steps", "ramp_steps"},
            )
            if _has_path(config, "dynamic_2dgs.regularization.velocity_l2"):
                _reject_unknown_keys(
                    config,
                    "dynamic_2dgs.regularization.velocity_l2",
                    {"enabled", "weight", "warmup_steps", "ramp_steps"},
                )
def assert_model_variant_implemented(config) -> None:
    """Fail before logger/output side effects for design-only variants."""

    variant = _variant(config)
    if variant not in _IMPLEMENTED_VARIANTS:
        raise NotImplementedError(
            f"model.variant={variant!r} is a valid design config, but its model "
            "backend is not implemented yet"
        )


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


def _is_experiment_config(value) -> bool:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)
    return isinstance(value, Mapping) and "p2g" in value and "data" in value


def _as_experiment_config(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)
    if not isinstance(value, Mapping):
        return None

    for key in ("cfg", "config", "hparams"):
        nested = value.get(key)
        if _is_experiment_config(nested):
            value = nested
            break
    if not _is_experiment_config(value):
        return None

    config = dict(value)
    config.pop("_wandb", None)
    return OmegaConf.create(config)


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
    "DEFAULT_CONFIG_PATH",
    "ATTENTION_VELOCITY_VARIANTS",
    "DYNAMIC_VARIANT",
    "DYNAMIC_VARIANT_V1",
    "DYNAMIC_VARIANT_V3",
    "DYNAMIC_VARIANT_V3_1",
    "DYNAMIC_VARIANT_V4",
    "DYNAMIC_VARIANTS",
    "LEGACY_VARIANT",
    "PHYSICAL_VELOCITY_VARIANTS",
    "STRUCTURAL_DEFAULTS",
    "VARIANT_CONFIG_PATHS",
    "assert_model_variant_implemented",
    "compose_fresh_config",
    "load_checkpoint_config",
    "resolve_eval_config",
    "resolve_main_config",
    "validate_experiment_config",
]
