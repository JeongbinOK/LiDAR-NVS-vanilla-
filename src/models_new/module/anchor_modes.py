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
    routing_stats: Optional[dict] = None
    routing_budget_logits: Optional[torch.Tensor] = None


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
    feature, position, delta, frame_offset, metadata = query_head(
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
    target_pose=None,
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
    if getattr(slot_head, "count_mode", None) == "learned_gumbel_viewpt":
        token_counts = torch.diff(
            token_offset,
            prepend=token_offset.new_zeros(1),
        )
        anchor_batch = torch.repeat_interleave(
            frame_batch_idx.to(device=anchor_feature.device), token_counts
        )
        raw_params, packing = slot_head(
            anchor_feature,
            anchor_k,
            delta_p,
            token_offset,
            target_pose=target_pose,
            anchor_batch=anchor_batch,
        )
    else:
        # Preserve the legacy/mock head call contract exactly.
        raw_params, packing = slot_head(
            anchor_feature,
            anchor_k,
            delta_p,
            token_offset,
        )
    k_selection = packing.get("k_selection")
    routing_stats = None
    routing_budget_logits = None
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
        selected_index = packing["anchor_k"] - 1
        if packing.get("view_dependent", False):
            decision_index = packing["decision_index"]
            anchor_index = packing["anchor_index"]
            slot_index = packing["slot_index"]
            is_common = packing.get("is_common", packing["view_index"] == -1)
            common_rows = is_common.nonzero(as_tuple=True)[0]
            additional_rows = (~is_common).nonzero(as_tuple=True)[0]
            packed_seed_position = seed_position.new_empty(
                (anchor_index.numel(), 3)
            )
            packed_seed_ref = seed_ref.new_empty((anchor_index.numel(), 3))

            # Common geometry is unconditionally taken from the fixed K=1 row.
            # It cannot silently become view/K-dependent even if another
            # candidate row's slot zero changes in the seed builder.
            packed_seed_position = packed_seed_position.index_copy(
                0,
                common_rows,
                seed_position[anchor_index[common_rows], 0, 0],
            )
            packed_seed_ref = packed_seed_ref.index_copy(
                0,
                common_rows,
                seed_ref[anchor_index[common_rows], 0, 0],
            )
            if additional_rows.numel() > 0:
                additional_decision = decision_index[additional_rows]
                packed_seed_position = packed_seed_position.index_copy(
                    0,
                    additional_rows,
                    seed_position[
                        anchor_index[additional_rows],
                        selected_index[additional_decision],
                        slot_index[additional_rows],
                    ],
                )
                packed_seed_ref = packed_seed_ref.index_copy(
                    0,
                    additional_rows,
                    seed_ref[
                        anchor_index[additional_rows],
                        selected_index[additional_decision],
                        slot_index[additional_rows],
                    ],
                )
            routing_anchor = packing["decision_anchor_index"]
        else:
            anchor_rows = torch.arange(
                anchor_feature.shape[0], device=anchor_feature.device
            )
            # Geometry follows the same hard expert route as its K-specific head.
            # Do not mix unrelated candidate seed positions in the ST backward:
            # learned-count gradients are carried solely by the selected expert's
            # activated-opacity gate inside GridSlotHead.
            selected_seed_position = seed_position[anchor_rows, selected_index]
            selected_seed_ref = seed_ref[anchor_rows, selected_index]
            anchor_index = packing["anchor_index"]
            slot_index = packing["slot_index"]
            packed_seed_position = selected_seed_position[
                anchor_index, slot_index
            ]
            packed_seed_ref = selected_seed_ref[anchor_index, slot_index]
            routing_anchor = anchor_rows
        # This detached side-output is one row per routing decision (one per
        # token in learned_gumbel, one per token/target-view pair in viewpt). It
        # never enters Gaussian assembly, temporal rendering, or the loss graph.
        routing_stats = {
            "k_logits": packing["k_logits"].detach(),
            "selected_k": packing["anchor_k"].detach(),
            "token_position_sensor": token_position[routing_anchor].detach(),
            "is_dynamic": anchor_metadata["is_dynamic"][routing_anchor].detach(),
        }
        if packing.get("view_dependent", False):
            routing_stats["view_index"] = packing[
                "decision_view_index"
            ].detach()
        # Loss-only side output. Unlike routing_stats, this tensor intentionally
        # retains autograd and is removed before the temporal model/renderer.
        routing_budget_logits = packing["k_logits"]
    else:
        seed_ref = anchor_metadata["seed_ref"]
        anchor_index = packing["anchor_index"]
        slot_index = packing["slot_index"]
        packed_seed_position = seed_position[anchor_index, slot_index]
        packed_seed_ref = seed_ref[anchor_index, slot_index]

    metadata = {
        "box_assign": anchor_metadata["box_assign"][anchor_index],
        "instance_id": anchor_metadata["instance_id"][anchor_index],
        "is_dynamic": anchor_metadata["is_dynamic"][anchor_index],
        "coord_ref": packed_seed_ref,
        "bbox_ref_by_frame": anchor_metadata["bbox_ref_by_frame"],
    }
    if "view_index" in packing:
        metadata["view_index"] = packing["view_index"]
    if "common_view_gate" in packing:
        metadata["anchor_index"] = anchor_index
        metadata["common_view_gate"] = packing["common_view_gate"]
    gradient_weight = slot_head.gradient_weight(
        packing["slot_k"], seed_position.dtype
    )
    return GaussianSeedBatch(
        feature=None,
        position=packed_seed_position,
        frame_offset=packing["gaussian_offset"],
        metadata=metadata,
        gradient_weight=gradient_weight,
        raw_params=raw_params,
        routing_stats=routing_stats,
        routing_budget_logits=routing_budget_logits,
    )


__all__ = [
    "GaussianSeedBatch",
    "build_grid_gaussian_seeds",
    "build_spherical_gaussian_seeds",
]
