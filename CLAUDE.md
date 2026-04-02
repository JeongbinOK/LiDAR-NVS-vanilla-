# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Conda env: `lnvs` (activated when running experiments on this server)
- Data 경로 (서버마다 다름):
  - **RTX 3090 (local)**: `~/data/nuScenes` → `config.data_root = "~/data/nuScenes"`
  - **RTX 4090 (서버)**: `/data1/nuScenes` → `config.data_root = "/data1/nuScenes"`
- The dataset loader lives **outside** this repo at `{data_root}/loader/dataset.py`, injected via `sys.path.insert` using `config.data_root`

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
    │  vote_xyz는 seed 생성에만 사용; 이후 단계는 원본 xyz 기준
    │
[Stage 3] DiffSoftClustering (nn/diff_cluster.py)
    │  iterative soft k-means (4 iters, tau-annealed)
    │  거리 계산 및 center 업데이트 모두 원본 xyz 기준 (vote_xyz 아님)
    │  → centers [K,3], center_feats [K,D], assign [N,K]
    │
[Stage 3.5] CrossAttentionRefiner (nn/refine.py)
    │  cross-attn (centers ← top-64 assigned points) + self-attn + FFN
    │  center_feats만 업데이트, centers 위치는 변경 없음
    │  → refined center_feats [K,D]
    │
[Stage 4] GaussianParameterHead (nn/gaussian_head.py)
    │  PCA warmstart (no grad, xyz 기준 top-32 points per Gaussian)
    │  → q_pca, s_pca
    │  residual MLP correction → mu, q, s, alpha
    │  mu = centers + mlp_mu(center_feats)  ← centers에서 이동 가능
    │  s = (s_pca * exp(mlp_s)).clamp(min=0.1)
    │  q = F.normalize(q_pca + mlp_q)
    ▼
gaussians: {mu [K,3], q [K,4], s [K,2 or K,3], alpha [K,1]}
assign: [N,K]
```

K = number of occupied voxels (varies per frame, ~hundreds to low thousands).

## Loss

`nn/losses.py` — **per-Gaussian** assignment-weighted NLL, fully self-supervised.

### 축 방향: per-Gaussian (Point 기준 아님)

```
assign [N, K] → .T.topk(top_m, dim=-1) → [K, M]
```

각 Gaussian k가 상위 M개 점을 수집하여 NLL을 최소화한다. 모든 K개 Gaussian이 gradient를 받으므로 dead cluster 문제가 없다. `top_m = pca_topk = 32`로 PCA와 동일한 축/집합을 사용한다.

### NLL 수식

- **2D surfel**: `NLL = gamma_nll + maha_2d + log_det_2d`
  - `gamma_nll = (d·n)² / 0.25` — clamped to `1e4` (σ_perp=0.5m)
  - `maha_2d = (d·u / s₀)² + (d·v / s₁)²`
  - `log_det_2d = 2*(log s₀ + log s₁)`
- **3D**: `NLL = maha_3d + log_det_3d`
- 가중치는 per-Gaussian으로 정규화 (sum-to-1 per cluster)

Loss는 self-regularizing: `log_det`가 scale collapse를 막고, `maha`가 scale explosion을 막는다. Loss 값은 보통 **음수** (정상, 버그 아님).

## Training Details

- `bfloat16` autocast (`torch.autocast("cuda", dtype=torch.bfloat16)`)
- `AdamW` + `CosineAnnealingLR` (eta_min = lr × 1e-2)
- `clip_grad_norm_(max_norm=1.0)`
- Temperature `tau` anneals linearly: `1.0 → 0.2` over epochs
- Per-frame NaN/inf detection: skips individual frames, not entire batches
- Gradient accumulation: raw `loss.backward()` per valid frame, then `grad /= valid_frames` after the loop — correctly normalises even when frames are skipped
- Gradient NaN/inf check before `optimizer.step()` — prevents parameter corruption
- Checkpoint saved only when model parameters are NaN-free

## Monitoring Metrics (train log)

| 지표 | 의미 |
|------|------|
| `loss` | 전체 NLL loss |
| `tau` | 현재 clustering temperature |
| `s` | 평균 Gaussian scale |
| `K` | 프레임당 Gaussian 수 |
| `gamma_rms` | 포인트의 법선 방향 오차 RMS (2D only, 3D에서는 dist_rms와 동일) |
| `cov` | assign > 0.1 포인트 비율 (coverage) |
| `vote_off` | 포인트 투표 이동 거리 (VoxelCenterPredictor offset) |
| `mu_off` | Gaussian center가 clustering center에서 이동한 거리 (mlp_mu residual) — 크면 assign-mu mismatch 위험 |

## Key Config Choices

| Parameter | 값 | 비고 |
|-----------|-----|------|
| `primitive_type` | `"3d"` | 3D Gaussian |
| `pca_topk` | `32` | PCA 및 loss 모두 per-Gaussian top-M으로 사용 |
| `cluster_feat_weight` | `0.0` | 0.0 이상이면 backward에서 OOM 발생 가능 (RTX 3090 24GB 기준) |
| `top_k_assign` | `8` | 현재 loss에서 미사용 (per-point 개념 → per-Gaussian 전환으로 불필요) |
| `seed_voxel_size` | `1.0m` | K 결정 (장면 크기에 비례) |

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
