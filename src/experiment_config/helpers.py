"""Shared OmegaConf normalization and lookup helpers."""
from __future__ import annotations

from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf

from .definitions import LEGACY_VARIANT


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
def _require_positive(config, *paths, kind=float) -> None:
    """Every named path must hold a positive number."""
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is None or kind(value) <= kind(0):
            raise ValueError(f"{path} must be positive")


def _require_choice(config, path: str, allowed) -> None:
    """The named path must hold one of ``allowed``, compared lower-case."""
    value = OmegaConf.select(config, path)
    text = "" if value is None else str(value).lower()
    if text not in allowed:
        options = ", ".join(repr(option) for option in allowed)
        raise ValueError(f"{path} must be one of: {options}")


def _require_in_open_range(config, path: str, low: float, high: float) -> None:
    """The named path must hold a number strictly inside ``(low, high)``."""
    value = OmegaConf.select(config, path)
    if value is None or not low < float(value) < high:
        raise ValueError(f"{path} must be in ({low}, {high})")
