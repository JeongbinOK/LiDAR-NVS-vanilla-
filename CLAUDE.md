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

## Worktrees

| Path | Branch | Purpose |
|------|--------|---------|
| `/data/jeongbin/gaustering` | `gaustering` | 2D surfel (main/latest) |
| `/data/jeongbin/3d-gaussian` | `3d-gaussian` | 3D Gaussian experiments |
| `/data/jeongbin/vanilla` | `main` | baseline |
| `/data/jeongbin/qgs` | `qgs` | Quadratic Gaussian Surfel |

## Task & Design

### Goal
**LiDAR Novel View Synthesis (Feed-forward)**
- Input: 2 consecutive LiDAR keyframes (input_0 @ t=0, input_1 @ t=1)
- GT: ~9 intermediate sweeps between the two frames (timestamps normalized to [0,1])
- Output: synthesized LiDAR point cloud at arbitrary t ∈ [0,1]

### Strategy
**Static background**: remove dynamic bbox points from both frames → merge in LiDAR_0 frame → PTv3 → static Gaussians (fixed)

**Dynamic objects** (per tracked instance):
1. Extract points inside bbox from each frame
2. Transform both to LiDAR_0 frame (using `rel_input_1_pose`)
3. Subtract each bbox's center + undo yaw → canonical local space (origin-centered, forward-facing)
4. Merge both frames' points → denser local point cloud
5. PTv3 → dynamic Gaussians (in canonical local space)
6. Trajectory: store `(center_0, yaw_0, center_1, yaw_1)` from GT BBoxes
7. Render @ t: translate/rotate canonical Gaussians by `lerp(center, t)` + `slerp(yaw, t)`

**Rendering**: composite static + dynamic Gaussians → compare vs GT sweep → loss

### Dataloader (`{data_root}/loader/dataset.py`)
- `mode='nvs'`: point clouds + poses only
- `mode='bbox'`: additionally returns `boxes_0/1` (Tensor B×7, each frame's own LiDAR sensor frame) and `instance_ids_0/1` (same object = same int ID across frames)
- Box format: `[x, y, z, w, l, h, yaw]` (nuScenes wlh convention)
- `rel_input_1_pose`: LiDAR_1 → LiDAR_0 transform

## Outputs

Each run creates `outputs/train_NNN/` with:
- `configs/config.json` — saved hyperparameters
- `ckpt/best_model.pt` — best loss checkpoint (includes epoch, loss, optimizer, scheduler state)
- `ckpt/epoch_NNN.pt` — periodic checkpoint every 10 epochs
- `configs/git_info.json' — Latest git info for reproducibility
