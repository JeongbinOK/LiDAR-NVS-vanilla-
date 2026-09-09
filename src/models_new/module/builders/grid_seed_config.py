"""Configuration contract for occupied-grid seed generation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSeedConfig:
    """How many seeds each occupied token gets, and where they are placed.

    ``count_mode="legacy"`` decides the count up front and stores one padded
    ``(N, K_max, 3)`` seed set; the learned modes defer it to a router that runs
    after temporal fusion and store a ``(N, K_max, K_max, 3)`` candidate bank.

    In legacy mode ``fixed_count`` is what separates the two ways a count can be
    decided up front. False derives a per-token ``K`` from that token's own-frame
    raw point count (``ceil(count / points_per_gaussian)``), which is what the
    grid anchor head consumes. True gives *every* token exactly ``k_max`` seeds,
    which is what a fixed-count Gaussian head needs: it owns one parameter block
    per slot, so the slot count cannot vary from token to token.

    ``exp`` selects the placement rule within legacy mode: ``None`` spreads the
    seeds over the token's raw points by range quantile, ``1`` puts them on the
    token's observed medoid, and ``2`` puts them on the Utonia cell coordinate.
    Repeated slots start at the same coordinate on purpose; independent
    predictor outputs and learned offsets separate them.
    """

    count_mode: str
    k_max: int
    points_per_gaussian: int | None = None
    exp: int | None = None
    seed_mode: str | None = None
    fixed_count: bool = False

    LEGACY_COUNT_MODE = "legacy"
    LEARNED_COUNT_MODES = ("learned_gumbel", "learned_gumbel_viewpt")

    def __post_init__(self):
        if self.count_mode not in (
            self.LEGACY_COUNT_MODE, *self.LEARNED_COUNT_MODES
        ):
            raise ValueError(
                "grid count_mode must be 'legacy', 'learned_gumbel', or "
                "'learned_gumbel_viewpt'"
            )
        if int(self.k_max) <= 0:
            raise ValueError("grid K_max must be positive")
        if self.exp is not None and int(self.exp) not in (1, 2):
            raise ValueError("grid seed exp must be null, 1, or 2")
        if self.learned and self.fixed_count:
            raise ValueError(
                "a learned count router cannot also fix the Gaussian count"
            )

    @property
    def learned(self) -> bool:
        return self.count_mode in self.LEARNED_COUNT_MODES

    @property
    def gaussians_per_token(self) -> int:
        """Slots every token fills, for the modes where that is a constant."""
        if not self.fixed_count:
            raise ValueError(
                "only fixed_count seeds have one Gaussian count per token"
            )
        return int(self.k_max)


def resolve_grid_seed_config(cfg, *, fixed_count=False):
    """Read ``p2g.grid_query`` into the seed contract the caller needs.

    ``fixed_count`` is the Dynamic 2DGS path: every occupied token emits the
    same number of Gaussians, so the count is ``K_max`` rather than something
    a router or a raw point total decides. Omitting the ``grid_query`` block
    there means one Gaussian per token seeded at its observed medoid, which is
    what every fixed-count variant has always done.
    """
    grid_query = getattr(cfg, "grid_query", None)
    if grid_query is None:
        if not fixed_count:
            raise ValueError(
                "p2g.grid_query config block is required for anchor_mode='grid'"
            )
        return GridSeedConfig(
            count_mode="legacy", k_max=1, points_per_gaussian=1, exp=1,
            fixed_count=True,
        )

    count_mode = str(getattr(grid_query, "count_mode", "legacy")).lower()
    if count_mode not in (
        GridSeedConfig.LEGACY_COUNT_MODE, *GridSeedConfig.LEARNED_COUNT_MODES
    ):
        raise ValueError(
            "p2g.grid_query.count_mode must be 'legacy', 'learned_gumbel', "
            "or 'learned_gumbel_viewpt'"
        )

    if count_mode != GridSeedConfig.LEGACY_COUNT_MODE:
        if fixed_count:
            raise ValueError(
                f"count_mode={count_mode!r} predicts the Gaussian count and "
                "cannot serve a fixed-count head"
            )
        learned_count = getattr(grid_query, "learned_count", None)
        if learned_count is None:
            raise ValueError(
                "p2g.grid_query.learned_count is required for "
                f"count_mode={count_mode!r}"
            )
        k_max = getattr(learned_count, "K_max", None)
        if k_max is None or int(k_max) <= 0:
            raise ValueError(
                "p2g.grid_query.learned_count.K_max must be positive"
            )
        seed_mode = str(
            getattr(learned_count, "seed_mode", "range_quantile")
        ).lower()
        if seed_mode != "range_quantile":
            raise ValueError(
                "p2g.grid_query.learned_count.seed_mode currently supports "
                "only 'range_quantile'"
            )
        # Legacy points_per_gaussian/exp are deliberately not read here: K is
        # predicted after temporal feature fusion.
        return GridSeedConfig(
            count_mode=count_mode, k_max=int(k_max), seed_mode=seed_mode,
        )

    k_max = getattr(grid_query, "K_max", None)
    points_per_gaussian = getattr(grid_query, "points_per_gaussian", None)
    if k_max is None:
        raise ValueError("p2g.grid_query.K_max is required for legacy seeds")
    k_max = int(k_max)
    if k_max <= 0:
        raise ValueError("p2g.grid_query.K_max must be positive")
    exp = getattr(grid_query, "exp", None)
    exp = None if exp is None else int(exp)
    if exp is not None and exp not in (1, 2):
        raise ValueError("p2g.grid_query.exp must be null, 1, or 2")
    if exp is not None and exp > k_max:
        raise ValueError(f"p2g.grid_query.exp={exp} requires K_max >= {exp}")
    # points_per_gaussian only turns a raw point total into a per-token K, so a
    # fixed count never consults it and does not have to supply it.
    if points_per_gaussian is None:
        if not fixed_count:
            raise ValueError(
                "p2g.grid_query.points_per_gaussian is required for legacy "
                "grid seeds"
            )
    elif int(points_per_gaussian) <= 0:
        raise ValueError("p2g.grid_query.points_per_gaussian must be positive")
    return GridSeedConfig(
        count_mode="legacy",
        k_max=k_max,
        points_per_gaussian=(
            None if points_per_gaussian is None else int(points_per_gaussian)
        ),
        exp=exp,
        fixed_count=bool(fixed_count),
    )


__all__ = ["GridSeedConfig", "resolve_grid_seed_config"]
