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
```

This `/4d` worktree defaults to `dynamic_2dgs_attention_velocity_v4` for a
fresh run. `bbox_rigid_v1` is the historical box-routed model. V3_1 is
architecturally identical to V3 and adds only the STORM-style velocity_l2
prior that keeps the motion head from diverging. V4 preserves that objective,
adds a four-head soft-correspondence velocity initializer, and applies a
source-frame mean/median depth-consistency loss during training. V1, V3, and
V3.1 remain available for checkpoint reconstruction and controlled A/B
comparisons.

All Dynamic variants emit one observed-medoid-seeded Gaussian per occupied
endpoint token and use synchronous full cross-attention between endpoint sets.
V3 and later embed
`time_coordinate = relative_seconds / time_reference_sec`, where
`time_reference_sec=1.0` is fixed across samples. An irregular 0.8-second pair
therefore supplies endpoint coordinates `[0, 0.8]`, rather than `[0, 1]`. Its
zero-initialized, unbounded final Linear directly predicts velocity in m/s, and
the renderer applies physical seconds:

`position(t) = position_source + velocity_mps * delta_time_sec`.

The current V3/V3_1/V4 fusion projects the normalized `432D` Utonia and `144D`
intensity features as `576 -> 1024 -> 1024 -> 576` with a SiLU after each
hidden Linear. The
spatial refiner and temporal cross-attention both keep this `576D` width and use
12 heads (`48D` per head). The Gaussian and physical-velocity heads consume the
same all-head refined token feature. In V4, the first four heads of the final
temporal layer also apply their existing Q/K attention probabilities to
opposite-frame token xyz. The resulting source-to-opposite displacement is
divided by signed physical delta-t, and the velocity head predicts a
zero-initialized additive residual. At each supervised target time, V4 also
renders frame-0-only and frame-1-only transported Gaussians and applies L1 to
their mean and median depths on `gt_depth > 0` rays. This auxiliary branch
detaches source centres and non-motion Gaussian attributes, so its gradients
reach the shared temporal feature through velocity rather than changing opacity
or shape. The Gaussian starts at the observed medoid plus a zero-initialized
learned offset; the medoid-to-token delta is not fed into the head. The velocity
head has no separate absolute position/Fourier input.

The fixed time reference only conditions network features; it is not the
sample-dependent endpoint duration and is not used to reinterpret renderer
units. V3's metric 3D RoPE uses `base=10` and `position_scale=0.2` radians/metre,
which exactly matches the active pretrained Utonia path (`coord=xyz_m*0.2`,
`rope_base=10`) while allowing distinct query and key position arrays.

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
V3/V3.1/V4 use `timestamps_sec` for both time conditioning and physical
transport; normalized timestamps remain for historical model contracts.

Only keys consumed by the implementation remain in each dynamic overlay.
Attention geometry uses the shared Utonia-compatible `Rotary3D`; V3/V3.1/V4
do not concatenate a second explicit position encoding into either prediction
head.

Add new variants to the loader allow-list instead of accepting arbitrary config
paths.  This keeps a misspelled or path-traversing selector from silently loading
an unintended architecture.
