"""Shared grid-token builder selected by the downstream anchor mode."""


SUPPORTED_ANCHOR_MODES = ("spherical", "grid")


def resolve_anchor_mode(cfg):
    """Validate and normalize the two supported anchor modes."""
    mode = str(getattr(cfg, "anchor_mode", "spherical")).lower()
    if mode not in SUPPORTED_ANCHOR_MODES:
        raise ValueError(
            f"Unknown p2g.anchor_mode={mode!r}; expected one of "
            f"{SUPPORTED_ANCHOR_MODES}"
        )
    return mode


def build_token_builder(cfg):
    """Build the tokenization path shared by spherical and grid modes."""
    mode = resolve_anchor_mode(cfg)
    from .grid_intensity import OccupiedGridTokenBuilder

    return mode, OccupiedGridTokenBuilder(cfg)


__all__ = ["SUPPORTED_ANCHOR_MODES", "build_token_builder", "resolve_anchor_mode"]
