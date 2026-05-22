"""Compatibility package that exposes the implementation under src/models."""

from pathlib import Path

_src_models = Path(__file__).resolve().parent.parent / "src" / "models"
if _src_models.is_dir():
    __path__.append(str(_src_models))
