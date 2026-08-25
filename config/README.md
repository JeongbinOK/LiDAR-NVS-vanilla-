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
```

This `/4d` worktree defaults to `dynamic_2dgs_attention_velocity_v6` for a
fresh run. `bbox_rigid_v1` is the historical box-routed model. V1 through V5
remain available for exact checkpoint reconstruction and controlled A/B runs.
V4/V5 reuse temporal cross-attention heads for velocity initialization. V6 does
not: it adds an independent, time-free Siamese motion proposal before temporal
cross-attention and uses the latter only for feature refinement. Its overlay
defaults to `device=[0]` and `data.bbox_json_path=null`; both remain ordinary
CLI-overridable runtime settings.

All Dynamic variants emit one observed-medoid-seeded Gaussian per occupied
endpoint token. V3 and later embed
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
probability softly re-ranks that pool, then only M=4 tokens plus one learned
global dustbin enter the final row softmax. A density-mismatched pair is never
hard-rejected solely by its reverse rank. There is no separate
concentration/cycle confidence. Dustbin-aware displacement directly forms
`v_init`. The V6 velocity head consumes only the refined feature and `v_init`,
then predicts `v_offset`; final velocity is the additive residual
`v_init + v_offset`. The dustbin confidence-weights the proposal but does not
gate the residual head.

V6's `regularization.velocity_l2` block keeps its historical name for config
compatibility, but `mode=final_group_l1` computes `mean(||v_final||_2)`. The
default weight is `0.05` with a 500-step ramp.

The fixed time reference only conditions network features; it is not the
sample-dependent endpoint duration and is not used to reinterpret renderer
units. V3's metric 3D RoPE uses `base=10` and `position_scale=0.2` radians/metre,
which exactly matches the active pretrained Utonia path (`coord=xyz_m*0.2`,
`rope_base=10`) while allowing distinct query and key position arrays. V6's
sparse local refiner has an independent RoPE scale: `base=10`,
`position_scale=4.0` radians/metre. At the stage-3 0.4 m cell spacing, one local
step therefore advances the highest-frequency phase by about 1.6 radians.

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
