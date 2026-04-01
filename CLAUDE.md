# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Conda env: `lnvs` (activated when running experiments on this server)
- Data: `/data1/nuScenes` (NuScenes v1.0-trainval, 700 scenes, 27430 pairs)
- The dataset loader lives **outside** this repo at `/data1/nuScenes/loader/dataset.py`, injected via `sys.path.insert` using `config.data_root`

```bash
conda activate lnvs
# Run training
CUDA_VISIBLE_DEVICES=<N> python train.py

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=<N> python train.py --resume outputs/train_XXX/ckpt/best_model.pt

# Evaluate
python eval.py --ckpt outputs/train_XXX/ckpt/best_model.pt
```


## Pipeline Architecture

The model is a single-forward-pass pipeline: **raw LiDAR → Gaussian parameters**.

```
xyz [N,3] + intensity [N,1]
    │
    ▼ ego-mask (radius > 2.5m)
    │
[Stage 1] PTv3Backbone (nn/ptv3/)
    │  → per-point features [N, D=64]
    │
[Stage 2] VoxelCenterPredictor (nn/vote.py)
    │  offset MLP → vote_xyz [N,3]
    │  voxelize vote_xyz (1m voxels) → seed centers [V,3], seed feats [V,D]
    │
[Stage 3] DiffSoftClustering (nn/diff_cluster.py)
    │  iterative soft k-means (4 iters, tau-annealed)
    │  → centers [K,3], center_feats [K,D], assign [N,K]
    │
[Stage 3.5] CrossAttentionRefiner (nn/refine.py)
    │  cross-attn (centers ← top-64 assigned points) + self-attn + FFN
    │  → refined center_feats [K,D]
    │
[Stage 4] GaussianParameterHead (nn/gaussian_head.py)
    │  PCA warmstart (no grad) → q_pca, s_pca
    │  residual MLP correction → mu, q, s, alpha
    │  s = (s_pca * exp(mlp_s)).clamp(min=0.1)
    │  q = F.normalize(q_pca + mlp_q)
    ▼
gaussians: {mu [K,3], q [K,4], s [K,2 or K,3], alpha [K,1]}
assign: [N,K]
```

K = number of occupied voxels (varies per frame, ~hundreds to low thousands).

## Loss

`nn/losses.py` — assignment-weighted Gaussian NLL, fully self-supervised:

- **2D surfel**: `NLL = gamma_nll + maha_2d + log_det_2d`
  - `gamma_nll = (d·n)² / 0.25` — clamped to `1e4` (σ_perp=0.5m)
  - `maha_2d = (d·u / s₀)² + (d·v / s₁)²`
  - `log_det_2d = 2*(log s₀ + log s₁)`
- **3D**: `NLL = maha_3d + log_det_3d`
- `assign_topk_w` is renormalized to sum-to-1 before weighting

The loss is self-regularizing: `log_det` prevents scale collapse, `maha` prevents scale explosion. Loss values are typically **negative** (expected, not a bug).

## Training Details

- `bfloat16` autocast (`torch.autocast("cuda", dtype=torch.bfloat16)`)
- `AdamW` + `CosineAnnealingLR` (eta_min = lr × 1e-2)
- `clip_grad_norm_(max_norm=1.0)`
- Temperature `tau` anneals linearly: `1.0 → 0.2` over epochs
- Per-frame NaN/inf detection: skips individual frames, not entire batches
- Gradient accumulation: raw `loss.backward()` per valid frame, then `grad /= valid_frames` after the loop — correctly normalises even when frames are skipped
- Gradient NaN/inf check before `optimizer.step()` — prevents parameter corruption
- Checkpoint saved only when model parameters are NaN-free

## Key Config Choices

| Parameter | gaustering branch | 3d-gaussian branch |
|-----------|-------------------|---------------------|
| `primitive_type` | `"2d"` | `"3d"` |
| `num_epochs` | 50 | 50 |
| `cluster_feat_weight` | 0.1 | 0.1 |
| `data_root` | `/data1/nuScenes` | `/data1/nuScenes` |

`cluster_feat_weight=0.0` disables feature distance in clustering (spatial-only k-means) — was required to fix OOM.

## Worktrees

| Path | Branch | Purpose |
|------|--------|---------|
| `/data/jeongbin/gaustering` | `gaustering` | 2D surfel (main/latest) |
| `/data/jeongbin/3d-gaussian` | `3d-gaussian` | 3D Gaussian experiments |
| `/data/jeongbin/vanilla` | `main` | baseline |

## Outputs

Each run creates `outputs/train_NNN/` with:
- `configs/config.json` — saved hyperparameters
- `ckpt/best_model.pt` — best loss checkpoint (includes epoch, loss, optimizer, scheduler state)
- `ckpt/epoch_NNN.pt` — periodic checkpoint every 10 epochs

## Known Numerical Issues

**`gamma_nll` overflow (2D mode only)**: The loss can produce `inf` via `(d·n)²/0.25` in bfloat16 — especially early in training. `torch.isnan` alone is **insufficient**; always use `torch.isfinite`. The `clip_grad_norm_` on inf gradients produces NaN via `inf × 0` (IEEE 754), which permanently corrupts parameters. Not applicable to `primitive_type="3d"` (this branch), which has no `gamma_nll` term.

**PCA SVD in bfloat16**: bfloat16 has only ~3 significant decimal digits, so the `1e-6` covariance regularisation is effectively zero in bfloat16, making SVD numerically unstable for sparse clusters. Fixed: `_pca_warmstart` explicitly upcasts inputs to float32 before computing covariance and SVD, then downcasts results back to the original dtype.
