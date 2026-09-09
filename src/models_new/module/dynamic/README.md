# Dynamic 2DGS module map

`dynamic_gaussian.py` is the compatibility facade. New code should import the
semantic module that owns it:

- `common.py`: config lookup and scalar encoding
- `temporal.py`: bidirectional temporal attention families
- `attention_matching.py`: final-attention consensus correspondence
- `heads.py`: Gaussian attributes and velocity decoders
- `motion_proposals.py`: independent paired-frame proposal families
- `backends.py`: backend composition and shared motion mixin
- `trajectory.py`: output trajectory contract validation

Config variant names in the table omit their shared `dynamic_2dgs_` prefix.

| Config variant suffix | Backend | Temporal / matching | Motion head | Gaussian count |
| --- | --- | --- | --- | --- |
| `direct_velocity_v1` | `DynamicGaussianBackend` | `ParallelBidirectionalCrossAttention` | `MotionConditionedVelocityHead` | exactly 1 |
| `physical_velocity_v3`, `v3_1` | `PhysicalVelocityGaussianBackend` | `TimeConditionedParallelCrossAttention` | `PhysicalVelocityHead` | fixed by config |
| `attention_velocity_v4`, `v5` | `AttentionInitializedVelocityGaussianBackend` | time-conditioned attention readout | `PhysicalVelocityHead` | fixed by config |
| `attention_velocity_v6` | `ProposalInitializedVelocityGaussianBackend` | time-conditioned attention + sparse proposal | physical or init-conditioned head from config | fixed by config |
| `attention_velocity_v7`, `v7_1` | `WarpedProposalVelocityGaussianBackend` | sparse proposal then warped temporal attention | `EmbeddedInitDurationVelocityHead` | fixed by config |
| `attention_velocity_v7_2` | `StraightThroughProposalVelocityGaussianBackend` | Top-4 STE proposal then warped temporal attention | `FeatureOnlyVelocityOffsetHead` | fixed by config |
| `attention_velocity_v8` | `ConsensusAttentionVelocityGaussianBackend` | final-layer attention + `ConsensusAttentionMotionMatcher` | `EmbeddedInitVelocityHead` | fixed by config |
| `attention_velocity_v9` | `PostAttentionProposalVelocityGaussianBackend` | temporal attention then projected dense proposal | `FeatureOnlyVelocityOffsetHead` | fixed by config |
| `attention_velocity_v10` | `LayerWeightedAttentionVelocityGaussianBackend` | `LayerWeightedDistanceBiasCrossAttention` | `FeatureOnlyVelocityOffsetHead` | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v11` | `MaxSpeedBarrierVelocityGaussianBackend` | `MaxSpeedBarrierLayerWeightedCrossAttention` | `EmbeddedInitVelocityHead` | learned K=1..3 by default; fixed available by config |
| `attention_velocity_v11_1` | `SingleGaussianBarrierVelocityGaussianBackend` | `GroupedGainBarrierCrossAttention` | `EmbeddedInitVelocityHead` | fixed by config |

Checkpoint compatibility depends on the registered module tree, parameter
assignment order, and tensor shapes. The split adds no `nn.Module` wrappers and
keeps constructor bodies and attribute assignment order intact. The historical
`dynamic_gaussian` module explicitly reexports every implementation symbol, so
existing imports and pickle lookups through that module remain resolvable. The
V10-family constructor also retains `gaussian_count_cfg` as its fifth positional
argument; new fixed-count calls should pass `gaussians_per_token` by keyword.
