"""Shared configuration and scalar-encoding helpers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default

class SinusoidalScalarEncoder(nn.Module):
    """Sinusoidal scalar encoding followed by a learned MLP projection."""

    def __init__(self, out_dim: int, num_frequencies: int = 8,
                 hidden_dim: int | None = None):
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_frequencies = int(num_frequencies)
        self.hidden_dim = (
            max(self.out_dim, 32) if hidden_dim is None else int(hidden_dim)
        )
        if (
            self.out_dim <= 0
            or self.num_frequencies <= 0
            or self.hidden_dim <= 0
        ):
            raise ValueError("time embedding dimensions must be positive")
        frequencies = math.pi * (2.0 ** torch.arange(
            self.num_frequencies, dtype=torch.float32
        ))
        self.register_buffer("frequencies", frequencies, persistent=False)
        in_dim = 1 + 2 * self.num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(-1, 1)
        phase = value.float() * self.frequencies.reshape(1, -1)
        encoded = torch.cat([value.float(), phase.sin(), phase.cos()], dim=-1)
        return self.mlp(encoded.to(dtype=value.dtype))

__all__ = ["SinusoidalScalarEncoder"]

