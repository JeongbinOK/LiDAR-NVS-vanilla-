"""Mini PointTransformerV3 intensity encoder that mirrors the frozen Utonia backbone.

Motivation: the sparse (SubMConv3d) encoder gives intensity a k=3 convolution
receptive field only; the Utonia geometry stream it fuses with is attention-based.
This encoder instead reuses ``PointTransformerV3`` itself (serialization -> RoPE
patch attention + xCPE -> GridPooling), so intensity gets the same multi-scale
attentive context as the geometry stream, and the two streams are fused only at
the end (no intermediate Utonia feature injection).

Alignment: it starts from the SAME input voxel grid Utonia sees
(``ptv3_input['grid_coord']``, 0.05 m) with the SAME per-frame ``offset``, and
uses the SAME stride-2 ``GridPooling`` (``unique(grid_coord // 2 | batch << 48,
sorted=True)``, serialization-independent). So at stage ``utonia_feature_stage``
its occupied cells coincide with the Utonia tokens *row for row* -- the builder
asserts ``torch.equal`` and then indexes with the same ``occ`` mask as the Utonia
feature, never gathering. The mini model runs ONCE on the full batch (like Utonia),
not per frame.

Input feature (config ``input_feats``):
- ``[intensity]`` (default): 1 channel, ``ptv3_input['strength']`` (already /255).
  intensity-only has no absolute range/direction; the fusion's Utonia stream
  (feat=coord) and the SphericalQueryHead's ray re-injection compensate downstream.
- ``[intensity, ray]``: 5 channels via ``ray_input_feat`` -- kept as a one-line A/B
  switch, since intensity physically depends on absolute range/incidence.

Constraints: ``GridPooling.shuffle_orders`` is hardcoded True by
``PointTransformerV3.__init__`` and not threaded through, so it is turned off on
every ``GridPooling`` after construction (shuffle only reorders the order list, not
feature rows -- pure noise + reproducibility loss for a small encoder). RoPE needs
``head_dim % 6 == 0`` (channels [36,72,108,144]/heads [2,4,6,8] -> 18), validated on
the truncated (used) stages only. No projection head: output is LayerNorm(C_last).
"""
from __future__ import annotations

import math

import torch.nn as nn

from ...utonia.model import GridPooling, PointTransformerV3
from ...utonia.structure import Point
from .intensity_sparse import ray_input_feat


class IntensityPTv3Encoder(nn.Module):
    """Per input voxel -> per stage-k cell feature (C_last), aligned to the Utonia
    token grid. ``n_stages`` = ``utonia_feature_stage + 1`` so switching the stage
    auto-tracks the resolution; the config channel/head/depth lists are truncated to
    the first ``n_stages`` entries.
    """

    def __init__(self, n_stages: int, cfg_ptv3, r_far: float = 80.0,
                 coord_scale: float = 0.2):
        super().__init__()
        n = int(n_stages)
        self.coord_scale = float(coord_scale)
        self._log_r_far = math.log1p(float(r_far))

        input_feats = list(getattr(cfg_ptv3, "input_feats", ["intensity"]))
        self.input_feats = input_feats
        self.use_ray = input_feats != ["intensity"]
        in_channels = 5 if self.use_ray else 1

        channels = list(getattr(cfg_ptv3, "channels", [36, 72, 108, 144, 192]))
        num_heads = list(getattr(cfg_ptv3, "num_heads", [2, 4, 6, 8, 12]))
        depths = list(getattr(cfg_ptv3, "depths", [1, 1, 2, 2, 2]))
        patch_size = int(getattr(cfg_ptv3, "patch_size", 1024))
        mlp_ratio = float(getattr(cfg_ptv3, "mlp_ratio", 2.0))
        order = tuple(getattr(cfg_ptv3, "order", ["z", "hilbert"]))

        if min(len(channels), len(num_heads), len(depths)) < n:
            raise ValueError(
                f"intensity_encoder.ptv3 channels/num_heads/depths each need >= "
                f"n_stages={n} entries, got lengths "
                f"{len(channels)}/{len(num_heads)}/{len(depths)}")
        enc_channels = tuple(channels[:n])
        enc_num_head = tuple(num_heads[:n])
        enc_depths = tuple(depths[:n])
        enc_patch_size = (patch_size,) * n

        # RoPE (model.py Point3DRoPE) needs head_dim % 6 == 0 on every USED stage.
        for c, h in zip(enc_channels, enc_num_head):
            if c % h != 0 or (c // h) % 6 != 0:
                raise ValueError(
                    f"intensity_encoder.ptv3 stage channels={c}/heads={h} give "
                    f"head_dim={c / h}; RoPE needs head_dim divisible by 6")

        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            order=order,
            stride=(2,) * (n - 1),
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            mlp_ratio=mlp_ratio,
            drop_path=0.0,
            shuffle_orders=False,
            enable_flash=True,
            enc_mode=True,
            mask_token=False,
        )
        # GridPooling.shuffle_orders is hardcoded True and not exposed via the ctor.
        for m in self.ptv3.modules():
            if isinstance(m, GridPooling):
                m.shuffle_orders = False

        c_last = enc_channels[-1]
        self.out_ln = nn.LayerNorm(c_last)
        self.out_dim = c_last

    def forward(self, ptv3_input):
        """One full-batch pass. Returns (stage grid_coord (M,3), stage feat
        (M,C_last), offset (n_frames,)) at the ``utonia_feature_stage`` resolution."""
        if "strength" not in ptv3_input:
            raise KeyError(
                "intensity_encoder.type='ptv3' needs ptv3_input['strength'] (per "
                "input-voxel sampled intensity); set keep_strength=True on the "
                "dataloader transform.")
        device = self.out_ln.weight.device
        coord = ptv3_input["coord"].to(device)
        grid_coord = ptv3_input["grid_coord"].to(device)
        offset = ptv3_input["offset"].to(device)
        strength = ptv3_input["strength"].to(device).reshape(-1)
        if self.use_ray:
            feat = ray_input_feat(coord, strength, self.coord_scale, self._log_r_far)
        else:
            feat = strength.reshape(-1, 1)

        point = Point(dict(coord=coord, grid_coord=grid_coord, feat=feat, offset=offset))
        point = self.ptv3.embedding(point)
        point.serialization(order=self.ptv3.order, shuffle_orders=False)
        point.sparsify()
        for s in range(len(self.ptv3.enc)):
            point = self.ptv3.enc[s](point)
        return point.grid_coord, self.out_ln(point.feat), point.offset
