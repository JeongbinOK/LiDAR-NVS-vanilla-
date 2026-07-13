"""Config-driven intensity encoder construction and execution.

This is the single selection boundary for the three supported intensity paths:
``ptv3`` (full-batch hierarchical attention), ``sparse`` (per-frame sparse
convolution), and ``mlp`` (per-cell statistics).  Concrete hierarchical encoder
implementations stay in their focused modules; callers do not branch on type.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import split_by_offset


SUPPORTED_INTENSITY_ENCODERS = ("ptv3", "sparse", "mlp")


class IntensityMLPEncoder(nn.Module):
    """Encode per-cell ``[mean, variance, theta, phi, range]`` statistics."""

    def __init__(self, in_dim: int = 5, out_dim: int = 64,
                 hidden: int | None = None, r_far: float = 70.0):
        super().__init__()
        if in_dim != 5:
            raise ValueError(
                "IntensityMLPEncoder expects 5D "
                "[mean_i, var_i, theta, phi, r] input"
            )
        width = int(hidden) if hidden else int(out_dim)
        self._log_r_far = math.log1p(float(r_far))
        self.net = nn.Sequential(
            nn.Linear(6, width),
            nn.SiLU(),
            nn.Linear(width, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def _encode(self, x5):
        mean_i, var_i, theta, phi, radius = x5.unbind(dim=-1)
        theta_n = theta / (math.pi / 2.0)
        log_r_n = torch.log1p(radius.clamp_min(0.0)) / self._log_r_far
        return torch.stack(
            [mean_i, var_i, theta_n, torch.sin(phi), torch.cos(phi), log_r_n],
            dim=-1,
        )

    def forward(self, x5):
        if x5.shape[0] == 0:
            return x5.new_zeros((0, self.norm.normalized_shape[0]))
        return self.norm(self.net(self._encode(x5)))


def resolve_intensity_encoder_type(cfg) -> str:
    """Resolve the explicit type, retaining the old sparse-enable fallback."""
    encoder_cfg = getattr(cfg, "intensity_encoder", None)
    if encoder_cfg is not None:
        encoder_type = str(getattr(encoder_cfg, "type", "ptv3")).lower()
    else:
        sparse_cfg = getattr(cfg, "intensity_sparse", None)
        sparse_enabled = (
            bool(getattr(sparse_cfg, "enable", True))
            if sparse_cfg is not None else True
        )
        encoder_type = "sparse" if sparse_enabled else "mlp"
    if encoder_type not in SUPPORTED_INTENSITY_ENCODERS:
        raise ValueError(
            f"Unknown intensity_encoder.type={encoder_type!r}; expected one of "
            f"{SUPPORTED_INTENSITY_ENCODERS}"
        )
    return encoder_type


def build_intensity_encoder(cfg):
    """Return ``(type, module, output_dim)`` for the configured intensity path."""
    encoder_type = resolve_intensity_encoder_type(cfg)
    encoder_cfg = getattr(cfg, "intensity_encoder", None)
    r_far = float(getattr(cfg, "r_far", 80.0))

    if encoder_type == "ptv3":
        from .intensity_ptv3 import IntensityPTv3Encoder

        ptv3_cfg = getattr(encoder_cfg, "ptv3", None)
        if ptv3_cfg is None:
            raise ValueError(
                "intensity_encoder.type='ptv3' requires "
                "intensity_encoder.ptv3 config"
            )
        module = IntensityPTv3Encoder(
            n_stages=int(cfg.utonia_feature_stage) + 1,
            cfg_ptv3=ptv3_cfg,
            r_far=r_far,
            coord_scale=0.2,
        )
        return encoder_type, module, int(module.out_dim)

    if encoder_type == "sparse":
        from .intensity_sparse import IntensitySparseEncoder

        sparse_cfg = getattr(cfg, "intensity_sparse", None)
        channels = (
            tuple(getattr(sparse_cfg, "channels", (16, 32, 48, 64, 96)))
            if sparse_cfg is not None else (16, 32, 48, 64, 96)
        )
        depth = int(getattr(sparse_cfg, "depth", 1)) if sparse_cfg is not None else 1
        out_dim = int(cfg.int_proj.out_dim)
        module = IntensitySparseEncoder(
            n_down=int(cfg.utonia_feature_stage),
            out_dim=out_dim,
            in_ch=5,
            channels=channels,
            depth=depth,
            r_far=r_far,
            coord_scale=0.2,
        )
        return encoder_type, module, out_dim

    out_dim = int(cfg.int_proj.out_dim)
    module = IntensityMLPEncoder(
        int(cfg.int_proj.in_dim), out_dim, r_far=r_far
    )
    return encoder_type, module, out_dim


def encode_intensity_features(
    encoder_type,
    encoder,
    *,
    ptv3_input,
    input_coord_list,
    input_grid_coord_list,
    token_grid_coord_list,
    occupied_masks,
    occupied_grid_coord_list,
    cell_statistics_list,
):
    """Run any configured encoder and return one feature tensor per frame.

    The builder supplies alignment metadata once.  This function owns all
    type-specific execution and alignment rules.
    """
    if encoder_type == "ptv3":
        enc_gc, enc_feat, enc_offset = encoder(ptv3_input)
        enc_gc_list = split_by_offset(enc_gc, enc_offset)
        enc_feat_list = split_by_offset(enc_feat, enc_offset)
        output = []
        for frame_idx, (enc_gc_i, enc_feat_i, token_gc_i, occupied) in enumerate(
            zip(enc_gc_list, enc_feat_list, token_grid_coord_list, occupied_masks)
        ):
            token_gc_i = token_gc_i.to(enc_gc_i.device)
            if enc_gc_i.shape != token_gc_i.shape or not torch.equal(
                enc_gc_i.long(), token_gc_i.long()
            ):
                raise RuntimeError(
                    f"PTv3 intensity grid mismatch at frame {frame_idx}: expected "
                    "row-alignment with the Utonia stage grid, got "
                    f"{tuple(enc_gc_i.shape)} vs {tuple(token_gc_i.shape)}"
                )
            output.append(enc_feat_i[occupied])
        return output

    if encoder_type == "sparse":
        from .intensity_sparse import gather_by_grid_coord

        if "strength" not in ptv3_input:
            raise KeyError(
                "sparse intensity encoding requires ptv3_input['strength']; "
                "enable keep_strength in the dataloader transform"
            )
        strength_list = split_by_offset(
            ptv3_input["strength"].to(input_coord_list[0].device),
            ptv3_input["offset"],
        )
        output = []
        for coord, grid_coord, strength, target_gc in zip(
            input_coord_list,
            input_grid_coord_list,
            strength_list,
            occupied_grid_coord_list,
        ):
            stage_gc, stage_feat = encoder(coord, grid_coord, strength.reshape(-1))
            output.append(gather_by_grid_coord(target_gc, stage_gc, stage_feat))
        return output

    return [encoder(cell_stats) for cell_stats in cell_statistics_list]


__all__ = [
    "IntensityMLPEncoder",
    "SUPPORTED_INTENSITY_ENCODERS",
    "build_intensity_encoder",
    "encode_intensity_features",
    "resolve_intensity_encoder_type",
]
