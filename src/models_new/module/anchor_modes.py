"""Mode-specific conversion from fused grid tokens to Gaussian seeds.

Both anchor modes consume the same refined Utonia/intensity tokens AND the
same temporal-fusion stage (``GridTemporalAggregator``: background 0.8 m
radius cross-frame attention + per-instance box-local self-attention), so
every token-feature operation is identical across modes.  This module is the
only boundary where the gaussian-generation semantics differ:

* ``spherical`` runs the shared aggregator without grid seeds, then bins each
  frame's bbox-labelled raw points into spherical cells about its sensor
  origin; every occupied (cell, label) group is a context anchor, while every
  unique (anchor, supporting token) pair emits one raw-surface-seeded query
  that attends only to own-frame tokens. Mixed boundary cells split into pure
  bg/instance anchors.
* ``grid`` keeps each occupied Cartesian token as an anchor, runs the shared
  aggregator with its padded seed geometry, and expands each token through
  either the legacy metadata-selected K head or a learned hard
  Gumbel-Softmax ST K router.

The caller owns all trainable modules so their existing state-dict prefixes
(``squery_head.*``, ``grid_temporal_agg.*``, and ``grid_slot_head.*``) stay
unchanged; ``grid_temporal_agg.*`` is now instantiated for both modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GaussianSeedBatch:
    """Flat mode-specific positions and either head features or raw GS parameters."""

    feature: Optional[torch.Tensor]
    position: torch.Tensor
    frame_offset: torch.Tensor
    metadata: dict
    delta: Optional[torch.Tensor] = None
    gradient_weight: Optional[torch.Tensor] = None
    raw_params: Optional[torch.Tensor] = None


def build_spherical_gaussian_seeds(
    temporal_aggregator,
    query_head,
    fused_feature,
    token_position,
    raw_point_sensor,
    raw_token_index,
    token_offset,
    frame_batch_idx,
    pose,
    bbox,
    bbox_instance_ids,
    timestamps,
):
    """Fuse tokens across frames, then run the own-frame spherical query head."""
    fused_feature, _coord_out, _seed_out, _seed_delta, agg_meta = temporal_aggregator(
        fused_feature,
        token_position,
        None,
        None,
        token_offset,
        frame_batch_idx,
        pose,
        bbox,
        bbox_instance_ids,
        timestamps,
    )
    # The aggregator already produced per-token ref coords and per-frame
    # ref-frame boxes; the spherical head reuses them instead of recomputing
    # the bit-identical apply_pose / transform_boxes_to_ref (its frame loop
    # still runs for the head-specific raw-point labelling).
    feature, position, delta, _anchor_range, frame_offset, metadata = query_head(
        fused_feature,
        token_position,
        raw_point_sensor,
        raw_token_index,
        token_offset,
        frame_batch_idx,
        pose,
        bbox,
        bbox_instance_ids,
        token_ref=agg_meta.get("coord_ref"),
        bbox_ref_by_frame=agg_meta.get("bbox_ref_by_frame"),
    )
    return GaussianSeedBatch(
        feature=feature,
        position=position,
        frame_offset=frame_offset,
        metadata=metadata,
        delta=delta,
    )


def build_grid_gaussian_seeds(
    temporal_aggregator,
    slot_head,
    fused_feature,
    token_position,
    anchor_k,
    seed_sensor,
    delta_sensor,
    token_offset,
    frame_batch_idx,
    pose,
    bbox,
    bbox_instance_ids,
    timestamps,
):
    """Aggregate grid tokens, pack variable slots, and expand token metadata."""
    (
        anchor_feature, _anchor_position, seed_position, delta_p, anchor_metadata,
    ) = temporal_aggregator(
        fused_feature,
        token_position,
        seed_sensor,
        delta_sensor,
        token_offset,
        frame_batch_idx,
        pose,
        bbox,
        bbox_instance_ids,
        timestamps,
    )
    raw_params, packing = slot_head(
        anchor_feature,
        anchor_k,
        delta_p,
        token_offset,
    )
    k_selection = packing.get("k_selection")
    if k_selection is not None:
        expected_shape = (
            anchor_feature.shape[0],
            slot_head.k_max,
            slot_head.k_max,
            3,
        )
        if tuple(seed_position.shape) != expected_shape:
            raise ValueError(
                "learned grid seed_position must have shape "
                f"{expected_shape}, got {tuple(seed_position.shape)}"
            )
        seed_ref = anchor_metadata["seed_ref"]
        if tuple(seed_ref.shape) != expected_shape:
            raise ValueError(
                "learned grid seed_ref must have shape "
                f"{expected_shape}, got {tuple(seed_ref.shape)}"
            )
        geometry_weight = k_selection.to(dtype=seed_position.dtype)
        # Forward is exactly the selected K candidate because k_selection is
        # hard one-hot. Its backward derivative is the soft Gumbel relaxation,
        # so both Gaussian parameters and seed geometry train the K predictor.
        seed_position = (
            geometry_weight[:, :, None, None] * seed_position
        ).sum(dim=1)
        seed_ref = (
            geometry_weight[:, :, None, None] * seed_ref
        ).sum(dim=1)
    else:
        seed_ref = anchor_metadata["seed_ref"]

    anchor_index = packing["anchor_index"]
    slot_index = packing["slot_index"]
    metadata = {
        "box_assign": anchor_metadata["box_assign"][anchor_index],
        "instance_id": anchor_metadata["instance_id"][anchor_index],
        "is_dynamic": anchor_metadata["is_dynamic"][anchor_index],
        "coord_ref": seed_ref[anchor_index, slot_index],
        "bbox_ref_by_frame": anchor_metadata["bbox_ref_by_frame"],
    }
    gradient_weight = slot_head.gradient_weight(
        packing["slot_k"], seed_position.dtype
    )
    return GaussianSeedBatch(
        feature=None,
        position=seed_position[anchor_index, slot_index],
        frame_offset=packing["gaussian_offset"],
        metadata=metadata,
        gradient_weight=gradient_weight,
        raw_params=raw_params,
    )


__all__ = [
    "GaussianSeedBatch",
    "build_grid_gaussian_seeds",
    "build_spherical_gaussian_seeds",
]
