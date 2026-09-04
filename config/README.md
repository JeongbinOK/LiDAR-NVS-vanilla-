# Experiment configuration

Fresh runs are composed in this order:

1. `nuscene_train.yaml` shared base
2. one allow-listed file under `variants/`
3. OmegaConf CLI overrides

The selector uses the existing dot-list syntax:

```bash
python main.py model.variant=bbox_rigid_v1
python main.py model.variant=dynamic_2dgs_direct_velocity_v1
python main.py model.variant=dynamic_2dgs_physical_velocity_v3
python main.py model.variant=dynamic_2dgs_physical_velocity_v3_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v4
python main.py model.variant=dynamic_2dgs_attention_velocity_v5
python main.py model.variant=dynamic_2dgs_attention_velocity_v6
python main.py model.variant=dynamic_2dgs_attention_velocity_v7
python main.py model.variant=dynamic_2dgs_attention_velocity_v7_1
python main.py model.variant=dynamic_2dgs_attention_velocity_v7_2
python main.py model.variant=dynamic_2dgs_attention_velocity_v8
python main.py model.variant=dynamic_2dgs_attention_velocity_v9
python main.py model.variant=dynamic_2dgs_attention_velocity_v10
python main.py model.variant=dynamic_2dgs_attention_velocity_v11
```

This `/4d` worktree defaults to `dynamic_2dgs_attention_velocity_v7_2` for a
fresh run. `bbox_rigid_v1` is the historical box-routed model. V1 through V7.1,
V8, V9, V10, and V11 remain available for exact checkpoint reconstruction and
controlled A/B runs.
V4/V5 reuse temporal cross-attention heads for velocity initialization. V6 does
not: it adds an independent, time-free Siamese motion proposal before temporal
cross-attention and uses the latter only for feature refinement. Its overlay
defaults to `device=[0]` and `data.bbox_json_path=null`; both remain ordinary
CLI-overridable runtime settings.

Dynamic V1-V9 emit one observed-medoid-seeded Gaussian per occupied endpoint
token; V10 and V11 select one to three range-quantile-seeded Gaussians. V3
through V6, V7.2, V8, V9, V10, and V11 embed
`time_coordinate = relative_seconds / time_reference_sec`, where
`time_reference_sec=1.0` is fixed across samples. An irregular 0.8-second pair
therefore supplies endpoint coordinates `[0, 0.8]`, rather than `[0, 1]`. Its
scalar time becomes 17 raw Fourier values and passes through one
`17 -> 64 -> 576` MLP before being added to the 576D fused feature. Its
zero-initialized, unbounded final Linear directly predicts velocity in m/s, and
the renderer applies physical seconds:

`position(t) = position_source + velocity_mps * delta_time_sec`.

The current fusion projects the normalized `432D` Utonia and `144D`
intensity features as `576 -> 1024 -> 1024 -> 576` with a SiLU after each
hidden Linear. The
spatial refiner and temporal cross-attention both keep this `576D` width and use
12 heads (`48D` per head). V6 motion instead reads the frozen `432D` Utonia
stage-3 token directly. It applies non-affine LayerNorm, one endpoint-shared
zero-initialized residual bottleneck adapter, an endpoint-shared projection,
and L2 normalization. The adapter is exactly an identity path at initialization.
The symmetric content/distance score is globally evaluated in chunks and
truncated to a K=16 recall pool. Reverse
probability softly re-ranks that pool, then only M=4 tokens plus one dustbin
enter the final row softmax. Its threshold is a learned global intercept plus a
zero-initialized linear residual predicted from each time-free Utonia token, so
it can learn token-specific abstention without seeing temporal attention. A
density-mismatched pair is never hard-rejected solely by its reverse rank.
There is no separate concentration/cycle confidence. Dustbin-aware displacement directly forms
`v_init`. The V6 velocity head consumes only the refined feature and `v_init`,
then predicts `v_offset`; final velocity is the additive residual
`v_init + v_offset`. The dustbin confidence-weights the proposal but does not
gate the residual head.

`motion.residual_hidden_dim` sets that head's refiner width. The variant ships
`96`, i.e. `Linear(435, 96) -> SiLU -> Linear(96, 3)` and 42k parameters.
Omitting the key rebuilds the historical two `dim`-wide layers
(`435 -> 432 -> 432 -> 3`, 376k parameters), which is what every V6 checkpoint
before this key contains; those checkpoints restore their own embedded config
and are unaffected. Both forms keep the zero-initialized output Linear, so a
fresh model still starts at `v_final = v_init`.

V6's `regularization.velocity_l2` block keeps its historical name for config
compatibility. `mode=offset_l2` computes `mean(v_offset^2)` over all tokens and
xyz components, without pulling correspondence-derived `v_init` toward zero.
The default weight is `0.05` with a 500-step ramp. Historical variants can keep
using `final_l2` or `final_group_l1` on rendered final velocity.

V7 keeps V6's sparse Top-K/Top-M proposal structure but changes all three
conditioning paths:

- Proposal descriptors are exactly `L2Norm(f_utonia)`. No LayerNorm, adapter,
  or learned projection is constructed on the descriptor path.
- The candidate softmax contains only real candidates. An independent
  `3 -> 16 -> SiLU -> 1` MLP predicts `p_unmatched` from normalized Top-16
  entropy, the magnitude of the best reciprocal-re-ranked displacement, and
  normalized reciprocal probability mass (missing reverse entries contribute
  zero). Displacement is represented as
  `tanh(||delta_xyz|| / (search_speed_max_mps * raw_duration_sec))`, so its
  scale remains physical and invariant to matching direction without treating
  the 30 m/s speed limit as a fixed 30 m distance clamp. Its zero output weight
  and logit bias initialize every token at `p_unmatched=0.05`.
- Temporal attention uses the hard-gated initializer:
  `q_pos = x_query + stopgrad(v_init) * signed_delta_t`. A confidently
  unmatched query therefore stays at its observed position. Keys stay at
  observed positions, no confidence gate multiplies cross-attention, and no
  temporal time embedding module is constructed. With `base=100` and
  `position_scale=2*pi/5`, the six bands span 5.00, 10.77, 23.21, 50.00,
  107.72, and 232.08 m.

V7 applies a conservative STE dustbin decision to the final initializer. In
the forward pass `p_unmatched >= 0.9` sets `v_init=0`; all other tokens retain
the complete `v_match` (there is no initial 0.95 shrink). In the backward pass
the gate uses the derivative of `1-p_unmatched`. The logged hard-reject fraction
is therefore important: a rising value means proposal supervision is being
routed only through the dustbin predictor for those rejected tokens.

V7.1 retains V7's binary-forward STE, direct descriptor, proposal Q-warp, and
offset decoder, but replaces all three Dustbin evidence channels. After the
K=16 reciprocal re-ranking, exactly M=4 selected hypotheses define
`d_bar = sum_i w_i d_i` and
`spread = sqrt(sum_i w_i ||d_i-d_bar||^2)`. The MLP input is
`[||d_bar||/(||d_bar||+4m), spread/(spread+0.5m), sum_i w_i reciprocal_i]`.
Thus entropy, the single best displacement, the all-K reciprocal sum, and
`delta_t` are absent from Dustbin evidence. The two scale values map their
named metric distance to 0.5 without clipping. A fresh model still starts at
`p_unmatched=0.05`; the hard threshold is 0.5, so forward matching remains
strictly 0/1 while the `1-p_unmatched` backward surrogate has its midpoint at
the same boundary. V7.1 raises the global search-speed floor from 2 m/s to
4 m/s. This widens the soft spatial prior used to score correspondence
candidates; it does not impose a minimum predicted object velocity.

V7.2 uses V7's pre-cross initializer and detached Q-coordinate warp, but
replaces the matcher with

`w = p_dense + stopgrad(w_top4_normalized - p_dense)`.

Here `p_dense` is the all-key softmax of direct L2-normalized Utonia cosine
scores. Consequently the forward coordinate is exactly the renormalized Top-4
expectation, while backward sends coordinate-loss gradient through every key.
Top-4 membership itself is still discrete. V7.2 has no K=16 preselection,
learned/token-dependent search radius, reciprocal score, dustbin, or unmatched
gate. It uses only the fixed physical prior
`-(distance / (30 m/s * raw_duration_sec))^2`; at a one-second interval its
penalties at 10, 20, and 30 m are 0.111, 0.444, and 1.0 logits. Query chunks of
128 cap score memory and are recomputed during backward.

Unlike V7/V7.1, V7.2 restores the exact V5 endpoint-time path before its four
cross-attention layers: raw relative seconds `[0, duration]` pass through
`Fourier -> 64D -> bias-free Linear -> token_dim`, and that result is added once
as `f = f + E(time)`. The matcher still sees the original time-free LoRA Utonia
feature. Q RoPE continues to use
`x + stopgrad(v_init * signed_duration)`, while K RoPE uses observed positions.
The V7.2 offset is feature-only:
`LN(refined_feature) -> Linear(432,96) -> SiLU -> Linear(96,3)`. It has exactly
one activation and no explicit `v_init` or duration branch. The final Linear is
zero initialized, so a fresh run begins at `v_final=v_init`; `offset_l2` remains
the velocity regularizer.

The V7/V7.1 offset decoder concatenates `LN(refined_feature)`,
`MLP(v_init): 3 -> 32 -> SiLU -> 32`, and
`MLP([sin,cos]_4(raw_duration_sec)): 8 -> 16 -> SiLU -> 16`. At the default
432D Utonia stage this is `480 -> 96 -> SiLU -> 3`; the final Linear is zero
initialized, so a fresh run starts exactly at `v_final=v_init`.

V8 removes V7.1's pre-attention LoRA-feature proposal and moves matching to the
last of 12 temporal layers. Heads 0-3 still take part in the ordinary dense
all-head feature update. For coordinate readout only, their individual dense
row-softmax probabilities are equal-averaged into one consensus distribution;
one K=16 pool is reciprocal-re-ranked and exactly M=4 candidates form the
weighted coordinate expectation. Thus M is total per token, not per head.
There is no distance bias, dustbin, separate motion RoPE, or Q-coordinate warp.
The matcher consumes the same final Q/K after the shared cross-attention RoPE
(`base=100`, `position_scale=2*pi/5`). The full-key probability mass captured
by Top-K is preserved as a concentration diagnostic, while the selected
probabilities are renormalized within K before reciprocal re-ranking. Thus the
reverse `p(i|j)` has the same conditional 1/K scale as V7. Forward and reverse
probabilities remain differentiable after the hard support indices are chosen;
the Top-K membership itself is not differentiable.

V8 restores endpoint-time conditioning. Raw relative seconds `[0, duration]`
pass through the historical `Fourier -> 64D -> bias-free Linear -> token_dim`
path, so stage-3 LoRA adds a 432D vector to each 432D token; setting
`time_embedding_dim=432` is neither required nor used. Final motion remains
`v_init + v_offset`. The V8 offset decoder concatenates
`LN(refined_feature)` with `MLP(stopgrad(v_init)): 3 -> 32 -> SiLU -> 32`, then
applies `464 -> 96 -> SiLU -> 3`. It has no separate duration branch; temporal
endpoint conditioning already exposes cadence. Its final Linear is zero
initialized. V8 uses `velocity_l2.mode=final_l2`, so the prior acts on the
rendered total velocity `v_init + v_offset` and sends a restoring gradient into
both correspondence and the learned correction.

V9 keeps V6/V7's separation between the proposal and the attention heads but
moves the proposal *behind* temporal refinement. Twelve time-conditioned
cross-attention layers produce `f'`, and that one tensor then feeds two
independent consumers: the descriptor branch that yields `v_init`, and the
feature-only offset head that yields `v_offset`. Because the trunk feeds the
matcher directly, correspondence gradient reaches the same twelve layers that
render the Gaussians, and V7/V7.2's detached Q-coordinate warp is neither
present nor needed. Q and K RoPE both use observed reference positions.

The V9 descriptor is `L2Norm(Linear(432,128)(LN(f')))`. This replaces V7.2's
parameter-free `direct_l2` on raw pre-attention Utonia tokens with one learned
endpoint-shared projection whose width is decoupled from the trunk, so the
matcher sees endpoint time and cross-frame context that V6/V7/V7.2 matchers
never do. Scoring keeps V7.2's exact form, `cosine / 0.07 -
(distance / (v_max * raw_duration_sec))^2`, and changes only `v_max`: 30 m/s
becomes 5 m/s. At a one-second interval, 5, 10, and 15 m therefore cost 1.0,
4.0, and 9.0 logits where V7.2 charged 0.028, 0.111, and 0.25.

The readout is the plain dense expectation `x_m = softmax(score) @ x_key` over
every opposite-frame key. V9 has no Top-K support, straight-through surrogate,
reciprocal re-ranking, dustbin, or unmatched gate, so both the rendered
coordinate and the gradient use all keys. `motion_effective_support` (the
full-support `exp(entropy)`) is the diagnostic that matters here: a dense
readout fails by spreading mass over the scene, and V7.2's Top-4 forward
cannot. `motion_candidate_probability_mass` and
`motion_hard_soft_displacement_cosine` keep V7.2's definitions and so read
directly across the two matchers: they report the mass a Top-4 truncation would
have captured and how far its coordinate would have pointed from the rendered
dense one.

V9's concentration diagnostics need a Top-K over the full dense score matrix,
which the dense readout itself never computes -- in V7.2 the same Top-K
produced the rendered coordinate, so it was free. Measured at 432D/128D over
~7.3k tokens per frame, that Top-K plus the full-support entropy is ~4.3% of a
training step, and `torch.utils.checkpoint` runs it a second time during the
backward recompute. `ModelWrapper` therefore switches the matcher's
`collect_diagnostics` flag on only for batches that `metrics.interval` will
actually log, which amortizes the cost to ~0.1%. The gate follows `global_step`
/ `batch_idx`, so every DDP rank agrees and no rank can reach a `sync_dist`
log another rank skipped; the flag lives outside `state_dict`, and matchers
that do not expose it -- every pre-V9 proposal -- keep computing their
statistics on every step exactly as their checkpoints did.

V9's offset head is V7.2's unchanged:
`LN(f') -> Linear(432,96) -> SiLU -> Linear(96,3)`, zero-initialized, so a fresh
run starts at `v_final = v_init`. Its temporal stack is V7.2's with `layers`
raised from 4 to 12; `qk_norm`, `motion_head_count`, `tie_motion_qk_init`, and
the V5 motion RoPE are all absent. V9 uses
`velocity_l2.mode=final_group_l2` on the rendered total velocity.

V10 derives correspondence directly from every temporal layer and all 12 heads.
It requires the frozen Utonia XYZI base with active stage-3 LoRA adapters; the
legacy separate intensity/fusion path is rejected for this variant.
For layer `l`, `W_l` is the equal mean of the head probabilities. A single
learned token is broadcast per frame/query direction and appended only to Q, so
it pools the opposite endpoint without becoming a coordinate-less matching key.
After each layer's FFN, one shared `Linear(432,96) -> SiLU -> Linear(96,1)` MLP
predicts `a_l`; `softmax_l(a_l)` gives `k_l`. The coordinate readout computes
`sum_l k_l (W_l @ xyz)` directly, which is algebraically identical to first
forming `W=sum_l k_l W_l` but does not retain twelve dense score matrices.

V10 replaces temporal 3D RoPE with
`-(distance / (30 m/s * raw_duration_sec))^2` in the cross-attention logits.
The bias is encoded exactly with four augmented Q/K channels, so varlen
FlashAttention remains available and no pairwise bias tensor or per-pair MLP is
materialized. At a 0.5-second interval the radius is 15 m, making the prior
deliberately conservative. After cross-attention, `LN(refined_feature)` enters
the existing grid `learned_gumbel` router with exactly K={1,2,3}. Training uses
hard Gumbel-Softmax STE and evaluation uses logit argmax; only the selected
K-specific joint head executes and its K range-quantile seeds become K
Gaussians. The selected activated-opacity gate supplies rendering-loss gradient
to all router logits, while `budget.enable=false` removes the separate count
budget objective. `v_offset` keeps V9's feature-only normalized MLP. Motion is
predicted once per token and index-expanded unchanged to all K child Gaussians.
V10 regularizes `v_total=v_init+v_offset` with
`0.01 * mean(||v_total||_2^2)`, keeps Chamfer at `0.02`, and sets the scale-loss
weight to zero.

V11 keeps V10's whole post-attention stack -- the same `learned_gumbel`
K={1,2,3} router over `LN(f')`, the same feature-only `v_offset`, one token
velocity index-expanded to every child Gaussian, and the same LoRA requirement --
and changes only how correspondence is produced inside temporal attention.

Its position encoding is split across heads. Head 0 is the sole correspondence
head and is the only head whose probabilities touch coordinates; it carries no
RoPE. Its logit instead takes one additive max-speed soft barrier,

`bias_ij = -a * relu(||x_i - x_j|| / (v_max * |dt|) - 1)^2`, with
`v_max = 30 m/s` and `a = 4`.

The bias is exactly zero inside the radius `R = v_max * dt` a top-speed object
could cover over the physical endpoint interval, and grows quadratically outside
it: 0 at `R`, 1 logit at `1.5R`, 4 at `2R`, and 16 at `3R`. Because `v_init` is
the readout displacement divided by that same interval, the barrier is a soft
cap on `v_init` at `v_max`. QK-Norm holds the content logit spread near +-1 at
initialization, so `a=4` is decisive at `2R` -- the same penalty V10's `-(d/R)^2`
charges there -- while leaving everything inside `R` free, which V10's prior does
not. Heads 1-11 keep ordinary metric 3D RoPE at `base=100` and
`1.2566 rad/m` (`2*pi/scale = 5 m` for the highest-frequency band).

The hinge is not a bilinear form, so unlike V10's quadratic it cannot be encoded
in extra Q/K channels. Head 0 therefore runs an explicit softmax, chunked over
`match_chunk_size` query rows and recomputed in backward the way V9's dense
proposal is; heads 1-11 stay on varlen FlashAttention. Head 0's V carries the
ordinary feature channels plus opposite-frame xyz, so one pass returns both its
feature update and `W_l @ xyz`. `match_chunk_size` trades attention recompute
against peak `N_q x N_k` score memory and never changes the result.

`v_init` mixes the twelve head-0 coordinate expectations under `softmax(a_l)`
over twelve plain learned scalars. Unlike V10 there is no query-only global token
and no per-sample scorer MLP: the mixture is shared by every token and every
sample, so it can only learn which *depth* resolves correspondence, starting from
an exact uniform 1/12.

V11's offset head is V8's rather than V10's feature-only one. Head 0's attended
*feature* channels never carry the metric coordinate expectation itself, so the
correction cannot see what it is correcting unless the initializer is embedded:
`e_v = Linear(3,32) -> SiLU -> Linear(32,32)` on a detached `v_init`, then
`Linear(96,3)(SiLU(Linear(432+32,96)([LN(f'), e_v])))`. The output Linear is still
zero-initialized, so a fresh run starts at `v_total = v_init`, and detaching keeps
`v_init + v_offset` the only route from the offset loss back into correspondence.

V11 keeps Chamfer at `0.02` and the scale-loss weight at zero, and regularizes
with `velocity_l2.mode=split_group_l2` at `init_weight = offset_weight = 0.01`.

`velocity_l2.mode` selects the magnitude prior for every variant:

- `final_l2` -- `mean(v_x^2, v_y^2, v_z^2)`, the historical STORM-calibrated
  normalization. This is `mean(||v||^2) / 3`, three times weaker than a
  squared-norm prior at the same weight.
- `final_group_l2` -- `mean(||v||_2^2)`. Same numerator as `final_l2` over one
  count per Gaussian rather than per component.
- `final_group_l1` -- `mean(||v||_2)`, which keeps a non-vanishing restoring
  force near zero.
- `offset_l2` -- `final_l2` applied to the learned residual only, leaving the
  correspondence initializer unregularized.
- `offset_group_l2` -- `final_group_l2` applied to that same residual:
  `mean(||v_offset||_2^2)`, three times `offset_l2` at an equal weight.
- `split_group_l2` -- `init_weight * mean(||v_init||_2^2) + offset_weight *
  mean(||v_offset||_2^2)`, charged independently. It takes `init_weight` and
  `offset_weight` instead of `weight`, and both are rejected by every other mode.
  Only the optimized total sees those weights: `loss_velocity_l2` stays the raw
  unweighted magnitude (here `mean(||v_init||^2) + mean(||v_offset||^2)`) so the
  one wandb-visible number remains comparable across variants, and
  `loss_velocity_l2_init` / `loss_velocity_l2_offset` log the two means beside it.

In every mode `loss_velocity_l2` is the raw term; the weighted contribution
(`wc_velocity_l2`) is computed for the total but is not sent to wandb.

The three families fail in opposite directions, which is why V11 splits them.
Charging the *sum* makes the cross term `2<v_init, v_offset>` payable by
cancellation rather than by being right: V5's residual became a pure shrink
operator at median `cos(init, offset) = -0.997`, and the V8 `finall2` run ended at
`|v_init| = 13.4`, `|v_offset| = 13.0`, `|v_total| = 0.50` m/s -- an offset that
existed only to erase its own initializer. Charging *only the offset* removes that
incentive but leaves the matcher unbounded, and then `v_init` is worth exactly
what the matcher is worth: 0.9-1.8 m/s under V6/V7/V7.1's descriptor matchers,
4.5 under V7.2's, and **64.3 m/s** under V8's broken Top-K readout -- the same
architecture as the `finall2` run above, differing only in this key. Under
offset-only reg `|v_offset|` also sits at 0.33-0.55 m/s in every such run, i.e.
the prior pins the correction to near-nothing. Splitting bounds `v_init` directly
while leaving the correction free to be worth what it explains.

`barrier_weight` stays at 4. Re-scoring V11's trained head-0 features at other
weights shows a=0 to a=4 is decisive (mass beyond the radius 11.2% -> 0.85%,
`|dp|` 8.5 m -> 2.26 m) and a=4 to a=64 changes nothing past the fourth decimal:
the residual mass sits just outside the radius, where the hinge is near zero by
construction and no weight reaches it.

For V3-V6, the fixed time reference only conditions network features; it is not the
sample-dependent endpoint duration and is not used to reinterpret renderer
units. V3's metric 3D RoPE uses `base=10` and `position_scale=0.2` radians/metre,
which exactly matches the active pretrained Utonia path (`coord=xyz_m*0.2`,
`rope_base=10`) while allowing distinct query and key position arrays. V6's
sparse local refiner has an independent RoPE scale: `base=10`,
`position_scale=4.0` radians/metre. At the stage-3 0.4 m cell spacing, one local
step therefore advances the highest-frequency phase by about 1.6 radians.
V7/V7.1 remove that temporal feature conditioning; their one-second duration
reference only fixes the units of the offset head's raw-duration Fourier input.

`src/config_loader.py` owns composition for both training and evaluation. New
checkpoints embed the resolved config, so every checkpoint carries immutable
configuration provenance. Historical checkpoints fall back to
`<checkpoint-dir>/wandb/latest-run/files/config.yaml`; that path is a mutable
legacy fallback and is reported explicitly. Neither path merges current
checkout overlays into an older run. Checkpoint-backed commands may change
runtime/training knobs, but CLI changes under `model`, `p2g`, `g2g`, `g2p`, or
`dynamic_2dgs` are rejected if they alter the stored value. Configs saved before
variants existed are interpreted as `bbox_rigid_v1`.

Here, "resolved config" means the final effective mapping after base, variant,
and CLI merging, with OmegaConf interpolations such as `${model.variant}` and
`${train.batch_size}` replaced by their concrete values. It records what the
program was given, rather than only which YAML files were selected. Shared
conditional blocks for inactive encoder alternatives can still be present; the
    the selected variant's `dynamic_2dgs` block rejects unsupported keys.

The resolved config does not currently record Git `HEAD` or uncommitted edits.
A Git SHA would identify the exact checked-out commit at launch (not necessarily
the newest commit on the remote), while a dirty flag/diff hash would be needed
to distinguish local changes made after that commit.

Because `wandb/latest-run` can point at a later resume, historical checkpoints
do not use that fallback automatically. After inspecting the artifact, opt in
with `allow_legacy_wandb_fallback=true`. Ray geometry and temporal-window fields
under `data` are protected as structural semantics; dataset paths, splits, and
worker counts remain runtime-overridable.

## Compact validation (`data.compact_val`)

Validation after every epoch runs the whole `val` split by default: 5,719
windows in the shipped `pair_mode=keyframe` contract. Consecutive flat-index
entries step one keyframe at a time, so they overlap heavily, and GT-bbox
labelling shows the split is 85% dynamic / 6% static — the aggregate metric is
therefore both slow and dominated by geometry the velocity head cannot change.

`data.compact_val` replaces the val dataloader's index with a fixed subset:

```yaml
data:
  compact_val:
    enable: true
    path: config/compact_val_windows.json
```

The shipped manifest holds 200 windows weighted toward motion: **180 dynamic + 20
static**. The static 20 are a no-motion control, not a representative sample —
they exist to catch a regression in plain geometry, while the metric that moves
when the velocity head changes comes from the dynamic 180.
`tools/build_compact_val_windows.py` builds it from **nuScenes GT annotations**
(never the tracking JSON): for each window it measures every annotated
instance's global-frame motion between the keyframes the window spans, keeping
only instances with `>= 10` LiDAR returns within 80 m of the ego. A window is
`dynamic` at a peak instance speed `>= 1.0 m/s`, `static` below `0.2 m/s`
(including windows with no qualifying box), and excluded in between. The 180
dynamic windows are stratified over four difficulty quartiles — scored by peak
displacement, moving-point count, mover count, and proximity of the nearest
mover — so the pool spans easy single-distant-car windows through dense
close-range traffic rather than only the easy tail. The 20 static windows are
stratified by box count so they are not all empty streets. Both pools cap
per-scene picks and forbid overlapping windows within a scene; the dynamic 180
lands on 119 of the 148 scenes that contain any motion at all.

Note the population this is drawn from: GT-bbox labelling puts the val split at
85% dynamic, 6% static and 9% borderline, and all 348 static windows live in
just 40 scenes. Nothing-moving-within-80m is the rare case on nuScenes, which is
why the static side is a small control rather than half the subset.

A window is addressed by its two endpoint LIDAR_TOP `sample_data` tokens, which
is a function of the sampling contract. The manifest records that contract
(`version`, `split`, `window_us`, `sample_gap_us`, `pair_mode`,
`pair_kf_stride`) and loading re-checks it, so a manifest cannot be silently
applied to windows it does not address; change any of those keys and rebuild:

```bash
python tools/build_compact_val_windows.py --n-static 20 --n-dynamic 180
```

`enable: false` restores full-split validation. `train`/`test` dataloaders are
untouched.

The Dynamic overlays deliberately keep `data.mode=bbox`; boxes are still loaded
and collated for dataset compatibility, but the dynamic backend never consumes
them. The dataloader exposes both normalized `timestamps` for attention and
relative `timestamps_sec` plus `window_duration_sec` for physical transport.
V3 and later use `timestamps_sec` for both time conditioning and physical
transport; normalized timestamps remain for historical model contracts.

Only keys consumed by the implementation remain in each dynamic overlay.
Attention geometry uses the shared Utonia-compatible `Rotary3D`; V3 and later
do not concatenate a second explicit position encoding into either prediction
head.

Add new variants to the loader allow-list instead of accepting arbitrary config
paths.  This keeps a misspelled or path-traversing selector from silently loading
an unintended architecture.
