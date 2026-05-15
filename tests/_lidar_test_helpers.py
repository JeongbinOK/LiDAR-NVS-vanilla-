"""Shared test helpers for the nonuniform LiDAR geometry refactor.

Tests that previously used a linear `(el_min, el_max)` FOV can use
`uniform_lidar_cfg(...)` to get a SimpleNamespace cfg with a uniformly-spaced
`ring_to_elevation_deg` table — that table's effective bounds happen to equal
the input `el_min/el_max`, so per-row centers behave identically to the old
linear convention.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch


def uniform_row_elevation_rad(el_min: float, el_max: float, H: int) -> torch.Tensor:
    """Uniform row centers (rad) such that the effective bounds == el_min/el_max."""
    step = (el_max - el_min) / H
    return torch.tensor(
        [el_min + (k + 0.5) * step for k in range(H)],
        dtype=torch.float32,
    )


def uniform_ring_to_elevation_deg(el_min_rad: float, el_max_rad: float, H: int) -> tuple:
    """Same as uniform_row_elevation_rad but in degrees, as a hashable tuple."""
    return tuple(math.degrees(e) for e in uniform_row_elevation_rad(el_min_rad, el_max_rad, H).tolist())


def uniform_lidar_cfg(
    H: int,
    W: int,
    el_min_rad: float,
    el_max_rad: float,
    *,
    r_near: float = 0.2,
    r_far: float = 100.0,
    lidar_sigma: float = 3.0,
):
    """SimpleNamespace cfg compatible with nn.lidar_geometry consumers."""
    return SimpleNamespace(
        ring_to_elevation_deg=uniform_ring_to_elevation_deg(el_min_rad, el_max_rad, H),
        lidar_height=int(H),
        lidar_width=int(W),
        lidar_sigma=float(lidar_sigma),
        r_near=float(r_near),
        r_far=float(r_far),
    )
