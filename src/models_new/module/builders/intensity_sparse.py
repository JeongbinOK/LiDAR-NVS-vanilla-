"""Sparse (SubMConv3d) intensity encoder that mirrors the Utonia grid hierarchy.

Motivation: the old ``IntensityEncoder`` is a per-cell MLP with **zero receptive
field** -- one 0.4/0.8 m cell's [mean_i, var_i, theta, phi, r] through an MLP. A
convolution, by contrast, gives reflectance a real multi-scale spatial context
(retro-reflective plates, lane paint, vegetation texture span several cells).

Alignment: we start from the SAME input voxel grid Utonia sees
(``ptv3_input['grid_coord']``, 0.05 m) and apply ``n_down = utonia_feature_stage``
non-overlapping kernel-2/stride-2 down convs (each output coord = floor(in/2)),
matching Utonia's ``GridPooling`` (``grid_coord // 2`` iterated). So the stage-k
occupied cells coincide with the Utonia tokens, and the per-cell feature is
gathered onto the token grid by grid_coord (``gather_by_grid_coord``).
NB: this coincidence needs the SparseConvTensor spatial_shape rounded up to a
multiple of ``2**n_down`` (see ``forward``); a plain ``max+1`` box makes spconv
truncate the outermost coarse cell on even-extent axes at every stage, which
would make the occupied set a strict subset of the Utonia tokens.

Why conv still needs [theta, sin phi, cos phi, log r] as *input features*:
SubMConv is translation-equivariant, so it captures relative local structure but
is blind to absolute range/direction from the sensor -- yet intensity depends on
absolute range (1/r^2) and incidence (ray direction). Those must be fed, not
derived from the grid. r is log1p-compressed and normalized by log1p(r_far), the
same convention as ``builders/common.encode_ray_meta``.

Norm: LayerNorm on features (batch-independent -> stable with batch_size 2 and
frame-varying point counts; consistent with the PTv3 LN style; safer given the
NaN-collapse history) rather than the BatchNorm usual for sparse convs. Each conv
is post-normed (conv -> LN -> SiLU); the 5D input is already ~unit scale.

Downsampling uses kernel-2/stride-2 SparseConv (non-overlapping -> floor(in/2),
matching Utonia's GridPooling exactly). The context/mixing blocks use kernel-3
SubMConv (fixed resolution), mirroring Utonia's kernel-3 xCPE.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from .common import xyz_to_theta_phi_r


def ray_input_feat(coord, intensity, coord_scale, log_r_far):
    """Scaled sensor coord (N,3) + intensity (N,) -> (N,5) ray input feature
    [intensity, theta_n, sin phi, cos phi, log_r_n].

    Shared by the sparse and ptv3 intensity encoders' 5D ray mode: ``coord`` is
    the sensor-frame position scaled by ``coord_scale`` (origin at the sensor), so
    range/direction are recovered by undoing that scale. r is log1p-compressed and
    normalized by ``log_r_far`` (== log1p(r_far)), matching ``encode_ray_meta``.
    """
    metric = coord / coord_scale
    tpr = xyz_to_theta_phi_r(metric)
    theta_n = tpr[..., 0] / (math.pi / 2.0)
    phi = tpr[..., 1]
    log_r_n = torch.log1p(tpr[..., 2].clamp_min(0.0)) / log_r_far
    return torch.stack(
        [intensity, theta_n, torch.sin(phi), torch.cos(phi), log_r_n], dim=-1)


def gather_by_grid_coord(target_gc, src_gc, src_feat):
    """Gather ``src_feat`` rows onto ``target_gc`` by exact integer grid_coord
    match (both frame-local, same origin). Misses -> zero rows. All (*,3) long."""
    C = src_feat.shape[1]
    if target_gc.shape[0] == 0:
        return src_feat.new_zeros((0, C))
    if src_gc.shape[0] == 0:
        return src_feat.new_zeros((target_gc.shape[0], C))
    gmax = torch.maximum(target_gc.max(0).values, src_gc.max(0).values) + 1
    stride_x = (gmax[1] * gmax[2]).to(torch.long)
    stride_y = gmax[2].to(torch.long)

    def _key(gc):
        gc = gc.to(torch.long)
        return gc[:, 0] * stride_x + gc[:, 1] * stride_y + gc[:, 2]

    src_key = _key(src_gc)
    order = torch.argsort(src_key)
    src_key_s = src_key[order]
    tgt_key = _key(target_gc)
    pos = torch.searchsorted(src_key_s, tgt_key)
    pos_c = pos.clamp(max=src_key_s.numel() - 1)
    found = (pos < src_key_s.numel()) & (src_key_s[pos_c] == tgt_key)
    out = src_feat[order[pos_c]]
    return torch.where(found.unsqueeze(-1), out, torch.zeros_like(out))


class IntensitySparseEncoder(nn.Module):
    """[strength, theta_n, sin phi, cos phi, log_r_n] per input voxel -> per
    stage-k cell feature (out_dim), aligned to the Utonia token grid.

    ``n_down`` = ``utonia_feature_stage`` so switching stage 3<->4 auto-tracks
    (0.4 m / 0.8 m). ``channels`` needs at least ``n_down + 1`` entries.
    """

    def __init__(self, n_down: int, out_dim: int = 64, in_ch: int = 5,
                 channels=(16, 32, 48, 64, 96), depth: int = 1,
                 r_far: float = 80.0, coord_scale: float = 0.2):
        super().__init__()
        self.n_down = int(n_down)
        self.coord_scale = float(coord_scale)
        self._log_r_far = math.log1p(float(r_far))
        C = list(channels)[: self.n_down + 1]
        if len(C) < self.n_down + 1:
            raise ValueError(
                f"channels needs >= n_down+1={self.n_down + 1} entries, got {channels}"
            )
        self.act = nn.SiLU()

        self.stem = spconv.SubMConv3d(in_ch, C[0], 3, padding=1, bias=False,
                                      indice_key="isp_stem")
        self.stem_ln = nn.LayerNorm(C[0])

        self.downs = nn.ModuleList()
        self.down_lns = nn.ModuleList()
        self.ctx = nn.ModuleList()
        self.ctx_lns = nn.ModuleList()
        for s in range(self.n_down):
            self.downs.append(
                spconv.SparseConv3d(C[s], C[s + 1], 2, stride=2, bias=False,
                                    indice_key=f"isp_down{s}"))
            self.down_lns.append(nn.LayerNorm(C[s + 1]))
            self.ctx.append(nn.ModuleList([
                spconv.SubMConv3d(C[s + 1], C[s + 1], 3, padding=1, bias=False,
                                  indice_key=f"isp_ctx{s}")
                for _ in range(depth)]))
            self.ctx_lns.append(nn.ModuleList([
                nn.LayerNorm(C[s + 1]) for _ in range(depth)]))

        self.head = nn.Linear(C[self.n_down], out_dim)
        self.out_ln = nn.LayerNorm(out_dim)

    def _input_feat(self, coord, intensity):
        """coord (N,3) scaled sensor coord (x coord_scale, origin at sensor) +
        intensity (N,) -> (N,5) [intensity, theta_n, sin phi, cos phi, log_r_n]."""
        return ray_input_feat(coord, intensity, self.coord_scale, self._log_r_far)

    def forward(self, coord, grid_coord, intensity):
        """One frame. coord/grid_coord (N,3), intensity (N,). Returns
        (stage_grid_coord (M,3) long, stage_feat (M,out_dim))."""
        if coord.shape[0] == 0:
            return (grid_coord.new_zeros((0, 3)),
                    self.head.weight.new_zeros((0, self.head.out_features)))
        feat_in = self._input_feat(coord, intensity)
        gc = grid_coord.to(torch.int32)
        # spconv's SparseConv3d output shape = (in - kernel)//stride + 1, which drops
        # the outermost coarse cell whenever an axis extent is even -- and this recurs
        # at every down stage. Utonia's GridPooling (unique(grid_coord // 2), iterated)
        # never truncates, so a plain (max + 1) bounding box makes the stage-k occupied
        # set a strict subset of the Utonia token grid, and those boundary tokens get
        # zero-filled by gather_by_grid_coord. Rounding each axis up to a multiple of
        # 2**n_down keeps every stage-k cell (== unique(input_gc // 2**k)), so the
        # encoder's occupied set matches features['grid_coord'] exactly at every stage.
        # The extra empty space adds no active sites, so interior cells are unaffected.
        pool = 1 << self.n_down
        spatial = ((gc.max(0).values // pool + 1) * pool).tolist()
        indices = torch.cat(
            [torch.zeros((gc.shape[0], 1), dtype=torch.int32, device=gc.device), gc],
            dim=1)
        x = spconv.SparseConvTensor(feat_in, indices, spatial, 1)

        x = x.replace_feature(self.act(self.stem_ln(self.stem(x).features)))
        for s in range(self.n_down):
            x = self.downs[s](x)
            x = x.replace_feature(self.act(self.down_lns[s](x.features)))
            for blk, ln in zip(self.ctx[s], self.ctx_lns[s]):
                # submanifold: output sites/order match input -> safe residual
                y = blk(x)
                x = x.replace_feature(x.features + self.act(ln(y.features)))

        feat = self.out_ln(self.head(x.features))
        stage_gc = x.indices[:, 1:].to(torch.long)
        return stage_gc, feat
