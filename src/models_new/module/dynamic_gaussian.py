"""Compatibility facade for the Dynamic 2DGS implementation.

The implementation lives in :mod:`.dynamic`; this module keeps the historical
import and pickle lookup path available to existing configs and checkpoints.
"""
from __future__ import annotations

from .dynamic.common import SinusoidalScalarEncoder, _cfg_get
from .dynamic.attention_matching import ConsensusAttentionMotionMatcher
from .dynamic.temporal import (
    GroupedGainBarrierCrossAttention,
    GroupedHeadRMSNorm,
    LayerWeightedDistanceBiasCrossAttention,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    ParallelBidirectionalCrossAttention,
    TimeConditionedParallelCrossAttention,
)
from .dynamic.heads import (
    EmbeddedInitDurationVelocityHead,
    EmbeddedInitVelocityHead,
    FeatureOnlyVelocityOffsetHead,
    GaussianAttributeHead,
    InitConditionedVelocityHead,
    MotionConditionedVelocityHead,
    PhysicalVelocityHead,
    SeedConditionedGaussianAttributeHead,
)
from .dynamic.motion_proposals import (
    ChunkedDenseMotionProposal,
    PairedFrameMotionProposal,
    ProjectedDenseMotionProposal,
    SparseMotionProposal,
    StraightThroughTop4MotionProposal,
)
from .dynamic.backends import (
    AttentionInitializedVelocityGaussianBackend,
    ConsensusAttentionVelocityGaussianBackend,
    DynamicGaussianBackend,
    LayerMixtureMotionMixin,
    LayerWeightedAttentionVelocityGaussianBackend,
    MaxSpeedBarrierVelocityGaussianBackend,
    PhysicalVelocityGaussianBackend,
    PostAttentionProposalVelocityGaussianBackend,
    ProposalInitializedVelocityGaussianBackend,
    SingleGaussianBarrierVelocityGaussianBackend,
    StraightThroughProposalVelocityGaussianBackend,
    WarpedProposalVelocityGaussianBackend,
)
from .dynamic.trajectory import DynamicGausTemp

__all__ = [
    "AttentionInitializedVelocityGaussianBackend",
    "ChunkedDenseMotionProposal",
    "ConsensusAttentionMotionMatcher",
    "ConsensusAttentionVelocityGaussianBackend",
    "DynamicGaussianBackend",
    "DynamicGausTemp",
    "EmbeddedInitDurationVelocityHead",
    "EmbeddedInitVelocityHead",
    "FeatureOnlyVelocityOffsetHead",
    "GaussianAttributeHead",
    "GroupedGainBarrierCrossAttention",
    "GroupedHeadRMSNorm",
    "InitConditionedVelocityHead",
    "LayerMixtureMotionMixin",
    "LayerWeightedAttentionVelocityGaussianBackend",
    "LayerWeightedDistanceBiasCrossAttention",
    "MaxSpeedBarrierLayerWeightedCrossAttention",
    "MaxSpeedBarrierVelocityGaussianBackend",
    "MotionConditionedVelocityHead",
    "PairedFrameMotionProposal",
    "ParallelBidirectionalCrossAttention",
    "PhysicalVelocityGaussianBackend",
    "PhysicalVelocityHead",
    "PostAttentionProposalVelocityGaussianBackend",
    "ProjectedDenseMotionProposal",
    "ProposalInitializedVelocityGaussianBackend",
    "SeedConditionedGaussianAttributeHead",
    "SingleGaussianBarrierVelocityGaussianBackend",
    "SinusoidalScalarEncoder",
    "SparseMotionProposal",
    "StraightThroughProposalVelocityGaussianBackend",
    "StraightThroughTop4MotionProposal",
    "TimeConditionedParallelCrossAttention",
    "WarpedProposalVelocityGaussianBackend",
]
