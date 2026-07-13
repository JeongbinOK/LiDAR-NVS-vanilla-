"""Config-driven anchor/token builder factory."""


SUPPORTED_ANCHOR_MODES = ("grid", "spherical", "spherical_legacy")


def build_anchor_builder(cfg):
    """Build only the selected implementation and report its token contract."""
    mode = str(getattr(cfg, "anchor_mode", "spherical")).lower()
    if mode not in SUPPORTED_ANCHOR_MODES:
        raise ValueError(
            f"Unknown p2g.anchor_mode={mode!r}; expected one of "
            f"{SUPPORTED_ANCHOR_MODES}"
        )
    if mode == "spherical_legacy":
        from .spherical_anchor import SphericalAnchorBuilder

        return mode, SphericalAnchorBuilder(cfg), False

    # Both grid primitives and the spherical-query head consume Utonia-grid
    # tokens with a selected intensity encoder.  The query head changes only the
    # downstream aggregation strategy.
    from .grid_intensity import GridIntensityBuilder

    return mode, GridIntensityBuilder(cfg), True


__all__ = ["SUPPORTED_ANCHOR_MODES", "build_anchor_builder"]
