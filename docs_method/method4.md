# Method 4: Neural 2D Gaussian Clustering v2

LiDAR 포인트 클라우드를 2D Gaussian surfel 집합으로 클러스터링하는 **완전 신경망 기반** 파이프라인.
Method 3 대비 주요 변경: Gumbel top-K/Gumbel-Softmax 제거, Voxel 기반 seeding + differentiable soft clustering 도입, Cross-Attention Refinement 추가.

---

## Method 3 → 4 주요 변경점

| 항목 | Method 3 | Method 4 |
|------|----------|----------|
| **Seed 생성** | Gumbel top-K + Ball-query NMS | Offset voting + Voxel pooling (scatter_mean) |
| **할당** | Learned affinity + Gumbel-Softmax | Distance-based soft k-means (softmax(-d²/τ)) |
| **Temperature** | Gumbel τ (seed + assign 공유) | Clustering τ만 (1.0→0.2) |
| **Backbone** | Custom 3-block transformer | PTv3 (primary) / Custom (fallback) |
| **Refinement** | 없음 | Cross-Attention Refiner (2 layers) |
| **Alpha** | sigmoid(MLP(feat)) | 고정값 1.0 |
| **Scale** | softplus(log(s_pca) + MLP) | s_pca * exp(MLP).clamp, min=0.1 |
| **Loss** | L_surface + L_spread + L_size | 단일 Gaussian NLL (self-regularizing) |

---

## 전체 파이프라인

```
LiDAR Frame (N × 4: xyz + intensity)
          │
          ▼ ego-vehicle mask (r < 2.5m 제거)
          │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 1: PTv3 Backbone                     │
│                                             │
│  [xyz, intensity] → PTv3 encoder-decoder    │
│  Grid voxelize (0.1m) → serialize (Z/Hilbert)│
│  Encoder: (2,2,2) blocks, ch (32,64,128)    │
│  Decoder: (2,2) blocks, ch (64,64)          │
│  → Linear proj → features [N, 64]           │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Voxel Center Seeding (vote.py)    │
│                                             │
│  features → MLP(64→64→3, zero-init) → offset│
│  vote_xyz = xyz + offset                    │
│  floor(vote_xyz / 1.0m) → voxel 좌표 (detach)│
│  scatter_mean(vote_xyz) → voxel_centers [V,3]│
│  scatter_mean(features) → voxel_feats [V,64]│
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Diff Soft Clustering (4 iters)    │
│                                             │
│  centers = voxel_centers  (init)            │
│  반복 ×4:                                    │
│    d = spatial_dist + 0.1 * feat_dist       │
│    assign = softmax(-d² / τ)    [N, K]      │
│    centers = (assign.T @ vote_xyz) / Σw     │
│    center_feats = (assign.T @ feats) / Σw   │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 3.5: Cross-Attention Refinement      │
│                                             │
│  ×2 layers:                                 │
│    1) Local cross-attn: center → top-64 pts │
│    2) Self-attn: centers ↔ centers          │
│    3) FFN: Linear(64→128) → GELU → Linear  │
│    각 sub-layer: residual + LayerNorm       │
│          ↓                                  │
│  center_feats [K, 64] (refined)             │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 4: Gaussian Parameter Head           │
│                                             │
│  PCA warm-start (no_grad):                  │
│    top-128 pts → weighted covariance → SVD  │
│    → q_pca [K,4], s_pca [K,2]              │
│  Residual MLP (zero-init):                  │
│    mu = centers + MLP_mu(feat)              │
│    q  = normalize(q_pca + MLP_q(feat))      │
│    s  = s_pca * exp(MLP_s(feat)), min=0.1   │
│    α  = 1.0 (fixed)                         │
│  n = u × v  (from quaternion)               │
│          ↓                                  │
│  gaussians: mu[K,3], q[K,4], s[K,2],       │
│             u[K,3], v[K,3], n[K,3], α[K,1]  │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Loss: Gaussian NLL (self-supervised)       │
│                                             │
│  top-8 sparsify per point                   │
│  2D: L = Σ w_ij · (γ²/σ⊥² + maha + log_det)│
│  단일 term, auxiliary loss 없음              │
└─────────────────────────────────────────────┘
```

---

## Stage 1: PTv3 Backbone

**파일**: `nn/ptv3/wrapper.py`, `nn/ptv3/model.py`

PTv3 (Point Transformer V3)를 feature backbone으로 사용.
입력 `[xyz, intensity]` → PTv3 dict format 변환 → encoder-decoder → `[N, 64]` feature 추출.

### 입력 변환

```python
feat = cat([xyz, intensity], dim=1)  # [N, 4]
data_dict = {
    coord: xyz,           # [N, 3]
    feat: feat,           # [N, 4]
    grid_size: 0.1,       # voxel 해상도
    offset: tensor([N]),  # batch offset (single frame)
}
```

### PTv3 구성

| 구분 | Depths | Channels | Heads | Patch Size |
|------|--------|----------|-------|------------|
| **Encoder** | (2, 2, 2) | (32, 64, 128) | (2, 4, 8) | (1024, 1024, 1024) |
| **Decoder** | (2, 2) | (64, 64) | (4, 4) | (1024, 1024) |
| **Stride** | (2, 2) | — | — | — |

- Grid voxelization (0.1m) → sparse convolution (spconv)
- Serialization: Z-order, Trans-Z, Hilbert, Trans-Hilbert 4개 공간 충전 곡선 (기본값)
- Flash Attention 활성화
- Decoder 출력 `[N, 64]` → 필요시 Linear projection

### Fallback: Custom Backbone (`nn/backbone.py`)

`backbone_type="custom"` 설정 시 사용.
- `Linear(4→64) → LayerNorm → ReLU`
- 3 Transformer blocks:
  - Even (0, 2): Z-order 정렬 → windowed attention (W=48)
  - Odd (1): Hilbert 정렬 → shifted window attention (W/2)
- Multi-head self-attention (4 heads, head_dim=16)

---

## Stage 2: Voxel Center Seeding

**파일**: `nn/vote.py` — `VoxelCenterPredictor`

각 포인트가 클러스터 중심 방향으로의 offset을 예측하고, offset 적용 후 voxel pooling으로 seed center를 생성.

### 알고리즘

```
1. offset = MLP(features)                    # [N, 3], zero-init
2. vote_xyz = xyz + offset                   # [N, 3]
3. voxel_coords = floor(vote_xyz / 1.0m)     # detach (이산 할당)
4. linear_hash = x·D_y·D_z + y·D_z + z      # collision-free
5. unique → voxel_ids [N]
6. voxel_centers = scatter_mean(vote_xyz)     # [V, 3], differentiable
7. voxel_feats   = scatter_mean(features)     # [V, 64]
```

### 설계 포인트

| 결정 | 이유 |
|------|------|
| **Zero-init** | 초기: vote_xyz = xyz, 학습하면서 점진적으로 offset 발생 |
| **Voxel coords detach** | 이산 할당에는 gradient 불필요, scatter_mean 값만 미분 가능 |
| **Voxel size = 1.0m** | 차량 스케일에 적합한 해상도, K ≈ N/voxel 자동 결정 |

### Gradient 경로

```
Loss → mu → centers (soft clustering) → voxel_centers (scatter_mean)
     → vote_xyz → offset_mlp → backbone features
```

---

## Stage 3: Differentiable Soft Clustering

**파일**: `nn/diff_cluster.py` — `DiffSoftClustering`

Voxel seed에서 시작하여 반복적 soft k-means로 클러스터 중심을 refine.

### 알고리즘 (T=4 iterations)

```python
centers = voxel_centers      # [K, 3] (init)
center_feats = voxel_feats   # [K, 64] (init)

for t in range(4):
    # 1. 거리 계산
    d_spatial = cdist(vote_xyz, centers)                   # [N, K]
    d_feat    = cdist(features, center_feats)              # [N, K]
    d = d_spatial + 0.1 * d_feat                           # combined

    # 2. Soft assignment
    assign = softmax(-d² / τ, dim=-1)                      # [N, K]

    # 3. Weighted center update
    w = assign.sum(dim=0).clamp(min=1e-4)                  # [K]
    centers      = (assign.T @ vote_xyz) / w.unsqueeze(-1) # [K, 3]
    center_feats = (assign.T @ features) / w.unsqueeze(-1) # [K, 64]
```

### Temperature 스케줄

```
τ(epoch) = τ_start + (epoch / (num_epochs - 1)) × (τ_end - τ_start)
         = 1.0 → 0.2 (linear annealing over 50 epochs)
```

- τ = 1.0: soft assignment (탐색, 넓은 gradient 전파)
- τ → 0.2: near-hard assignment (수렴, 결정론적)

### Method 3과의 차이

| Method 3 (Gumbel-Softmax) | Method 4 (Distance soft k-means) |
|---------------------------|----------------------------------|
| Learned affinity MLP | 직접적 거리 계산 (spatial + feature) |
| Gumbel noise로 stochastic | Temperature만으로 soft/hard 제어 |
| KNN 기반 top-8 후보 | 전체 center와 dense 거리 (cdist) |
| 별도 파라미터 (affinity MLP, score proj) | 파라미터 없음 (pure geometric) |

---

## Stage 3.5: Cross-Attention Refinement

**파일**: `nn/refine.py` — `CrossAttentionRefiner`

Clustering 후 center feature를 local context와 inter-center 관계로 보강.

### RefineLayer 구조 (×2 layers)

```
Input: center_feats [K, 64], point_feats [N, 64], assign [N, K]

1. Local Cross-Attention
   ┌──────────────────────────────────────────┐
   │  assign.T.topk(64) → local_idx [K, 64]  │
   │  local_feats = point_feats[local_idx]    │
   │                                          │
   │  Q = proj(center_feats)  [K, 1, H, d]   │
   │  K,V = proj(local_feats) [K, 64, H, d]  │
   │  → scaled_dot_product_attention          │
   │  → cross_out [K, 64]                     │
   └──────────────────────────────────────────┘
   center_feats = LayerNorm(center_feats + cross_out)

2. Self-Attention
   ┌──────────────────────────────────────────┐
   │  nn.MultiheadAttention(64, 4 heads)      │
   │  centers ↔ centers (global interaction)  │
   └──────────────────────────────────────────┘
   center_feats = LayerNorm(center_feats + self_attn_out)

3. FFN
   ┌──────────────────────────────────────────┐
   │  Linear(64→128) → GELU → Linear(128→64) │
   └──────────────────────────────────────────┘
   center_feats = center_feats + FFN(LayerNorm(center_feats))
```

### 역할

| Sub-layer | 기능 |
|-----------|------|
| **Local cross-attn** | 각 center가 자기에게 할당된 상위 64개 point를 참조하여 local geometry 반영 |
| **Self-attn** | center 간 정보 교환 (인접 Gaussian 간 일관성) |
| **FFN** | 비선형 변환으로 표현력 확보 |

---

## Stage 4: Gaussian Parameter Head

**파일**: `nn/gaussian_head.py` — `GaussianParameterHead`

PCA warm-start로 기하학적으로 합리적인 초기값 제공, residual MLP로 학습.

### Step 1: PCA Warm-Start (no_grad)

```python
# 각 cluster의 top-128 할당 점 수집
_, top_idx = assign.T.topk(128)           # [K, 128]
top_xyz = vote_xyz[top_idx]               # [K, 128, 3]
top_w = assign.T.gather(1, top_idx)       # [K, 128]

# Weighted covariance
d = top_xyz - centers.unsqueeze(1)        # [K, 128, 3]
w_norm = top_w / top_w.sum(1, keepdim=True)
cov = einsum('kmi,kmj,km->kij', d, d, w_norm) + 1e-6·I  # [K, 3, 3]

# SVD → principal axes
U, S, Vh = svd(cov)
u_pca = Vh[:, 0]                          # 최대 분산 방향
v_pca = Vh[:, 1]                          # 두 번째 분산 방향
n_pca = cross(u_pca, v_pca)              # surface normal

# Normal 방향 보정: sensor(원점) 방향으로 flip
flip = sign(dot(n_pca, -centers))
n_pca, v_pca = n_pca * flip, v_pca * flip

# Rotation → Quaternion
rot_pca = stack([u_pca, v_pca, n_pca])   # [K, 3, 3]
q_pca = rotation_matrix_to_quaternion(rot_pca)  # [K, 4] (w,x,y,z)

# Scale: singular value의 sqrt
s_pca = sqrt(S[:, :2])                    # [K, 2] (2D surfel)
```

### Step 2: Residual MLP (with gradient)

```python
mu    = centers + MLP_mu(center_feats)                    # [K, 3]
q     = normalize(q_pca + MLP_q(center_feats))            # [K, 4]
s     = (s_pca * exp(MLP_s(center_feats).clamp(-3, 3)))   # [K, 2]
        .clamp(min=0.1)
alpha = ones(K, 1)                                        # 고정

# Derived vectors
R = quaternion_to_rotation(q)
u = R[:, 0, :]                # tangent vector 1
v = R[:, 1, :]                # tangent vector 2
n = cross(u, v)               # surface normal
```

### MLP 구조 (3개 동일)

```
Linear(64→32) → ReLU → Linear(32→out_dim)
                        └─ zero-init (초기값 = PCA 그대로)
```

### 출력 Gaussian Parameters

| 파라미터 | Shape | 의미 |
|----------|-------|------|
| `mu` | [K, 3] | 클러스터 중심 위치 |
| `q` | [K, 4] | 회전 quaternion (w, x, y, z) |
| `s` | [K, 2] | tangent plane 상의 scale (σ_u, σ_v) |
| `alpha` | [K, 1] | opacity (1.0 고정) |
| `u` | [K, 3] | 1st tangent vector |
| `v` | [K, 3] | 2nd tangent vector |
| `n` | [K, 3] | surface normal (u × v) |

---

## Loss: Gaussian NLL

**파일**: `nn/losses.py` — `ClusteringLoss`

단일 self-supervised 목적함수. Auxiliary loss 없이 Gaussian NLL만으로 학습.

### Sparsification

전체 `assign [N, K]`에서 각 점마다 **top-8** assignment만 사용:
```python
assign_topk_w, assign_topk_idx = assign.topk(8, dim=-1)  # [N, 8]
```
→ 메모리 효율 (N×K → N×8)

### 2D Surfel NLL

각 점 p와 후보 Gaussian j에 대해:

```
d = p - μ_j                                    # [N, 8, 3]

# Quaternion → frame vectors
R = quat_to_rot(q_j)
u_j, v_j = R[:, 0], R[:, 1]
n_j = u_j × v_j

# 분해
γ  = (d · n_j)                                  # off-plane distance
d_u = (d · u_j)                                 # in-plane (u 방향)
d_v = (d · v_j)                                 # in-plane (v 방향)

# NLL 3개 항
γ_nll    = γ² / σ⊥²                             # σ⊥ = 0.05m (고정)
maha     = (d_u / s_u)² + (d_v / s_v)²         # Mahalanobis distance
log_det  = 2 · (ln(s_u) + ln(s_v))             # log-determinant

nll = γ_nll + maha + log_det
```

### Self-Regularization

| NLL 항 | s → 0 일 때 | s → ∞ 일 때 | 역할 |
|--------|------------|------------|------|
| `maha` | → ∞ (분모 감소) | → 0 | scale 폭발 방지 |
| `log_det` | → -∞ | → +∞ | scale 축소 방지 |
| **합계** | log_det 지배 → ↑ | maha 지배 → ↑ | **자동 균형** |

→ MLE 목적함수 자체가 regularizer 역할, auxiliary loss 불필요.

### 최종 Loss

```
L = mean_i [ Σ_j  w_ij · nll_ij ]

w_ij: top-8 soft assignment weight
nll_ij: 2D surfel NLL
```

### 3D Gaussian NLL (primitive="3d" 모드)

```
d_local = R^T · d                               # local frame 변환
maha    = Σ_dim (d_local / s)²                   # 3D Mahalanobis
log_det = 2 · Σ ln(s)
nll     = maha + log_det
```

---

## Training Loop

**파일**: `train.py`

### 데이터

- **NuScenesNVSDataset**: stereo pair (input_0, input_1) 로딩
- 각 프레임: `[N, 4]` (x, y, z, intensity)
- Ego-vehicle mask: `||xyz|| ≤ 2.5m` 제거

### 학습 흐름

```python
for epoch in range(50):
    tau = linear_anneal(1.0 → 0.2)

    for batch in dataloader:
        optimizer.zero_grad()
        for frame in [input_0, input_1]:           # stereo pair
            with autocast(bfloat16):
                output = model(xyz, intensity, tau)
                loss_dict = loss_fn(xyz, output)
            (loss / num_frames).backward()         # gradient 누적
        clip_grad_norm_(1.0)
        optimizer.step()
```

### 주요 설정

| 항목 | 값 |
|------|----|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | bfloat16 autocast |
| Batch size | 2 (× 2 frames = 4 forward/batch) |
| Epochs | 50 |
| Checkpoint | best_model.pt + 매 10 epoch |

### 평가 메트릭 (GT 없이)

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| `gamma_rms` | √(mean(γ²)) | surface 법선 방향 오차 |
| `dist_rms` | √(mean(‖p-μ‖²)) | center까지 거리 |
| `coverage` | mean(max_k(w_ik) > 0.1) | 유효 할당 비율 |

---

## Gradient 흐름 전체도

```
┌─ ClusteringLoss ─────────────────────────────────────┐
│  nll_2d(mu, s, q, assign)                            │
│     ↓ ∂L/∂mu, ∂L/∂s, ∂L/∂q, ∂L/∂w                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Stage 4: GaussianHead                               │
│     mu ← centers + MLP_mu(feats)    ✓ grad           │
│     q  ← q_pca + MLP_q(feats)      ✓ grad (MLP only)│
│     s  ← s_pca * exp(MLP_s)        ✓ grad (MLP only)│
│     q_pca, s_pca ← SVD(cov)        ✗ no_grad        │
│                            ↑                         │
│  Stage 3.5: Refiner                                  │
│     center_feats ← cross_attn + self_attn + FFN      │
│                            ↑                         │
│  Stage 3: Soft Clustering                            │
│     assign = softmax(-d²/τ)         ✓ grad           │
│     centers = assign.T @ vote_xyz   ✓ grad           │
│                            ↑                         │
│  Stage 2: Voter                                      │
│     vote_xyz = xyz + offset_mlp     ✓ grad           │
│     voxel_centers = scatter_mean    ✓ grad (values)  │
│     voxel_coords = floor(detach)    ✗ detach         │
│                            ↑                         │
│  Stage 1: PTv3 Backbone                              │
│     [xyz, intensity] → features     ✓ grad           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 하이퍼파라미터 전체 (`config.py`)

| 카테고리 | 파라미터 | 기본값 | 설명 |
|----------|----------|--------|------|
| **Backbone** | `backbone_type` | `"ptv3"` | PTv3 or custom |
| | `feature_dim` | 64 | 전체 feature 차원 |
| **PTv3** | `ptv3_grid_size` | 0.1 | voxel 해상도 (m) |
| | `ptv3_stride` | (2, 2) | downsampling stride |
| | `ptv3_enc_depths` | (2, 2, 2) | encoder block 수 |
| | `ptv3_enc_channels` | (32, 64, 128) | encoder 채널 수 |
| | `ptv3_enc_num_head` | (2, 4, 8) | encoder attention heads |
| | `ptv3_enc_patch_size` | (1024, 1024, 1024) | serialization patch |
| | `ptv3_dec_depths` | (2, 2) | decoder block 수 |
| | `ptv3_dec_channels` | (64, 64) | decoder 채널 수 |
| | `ptv3_dec_num_head` | (4, 4) | decoder attention heads |
| | `ptv3_dec_patch_size` | (1024, 1024) | decoder patch |
| | `ptv3_enable_flash` | True | Flash Attention |
| **Seeding** | `seed_voxel_size` | 1.0 | vote voxel 크기 (m) |
| **Clustering** | `cluster_iters` | 4 | soft k-means 반복 |
| | `cluster_feat_weight` | 0.1 | feature 거리 가중치 |
| | `cluster_tau_start` | 1.0 | 초기 temperature |
| | `cluster_tau_end` | 0.2 | 최종 temperature |
| **Refinement** | `refine_layers` | 2 | RefineLayer 수 |
| | `refine_heads` | 4 | attention heads |
| | `refine_local_topk` | 64 | cross-attn 참조 점 수 |
| **Gaussian** | `primitive_type` | `"2d"` | 2D surfel / 3D |
| | `pca_topk` | 128 | PCA 참조 점 수 |
| **Loss** | `top_k_assign` | 8 | sparsification K |
| **Training** | `lr` | 1e-3 | learning rate |
| | `weight_decay` | 1e-4 | AdamW weight decay |
| | `num_epochs` | 50 | 전체 epoch |
| | `batch_size` | 2 | batch size |
| **Data** | `ego_radius` | 2.5 | ego mask 반경 (m) |

---

## 파일 구조

```
gaustering/
├── train.py                    # 학습 루프
├── config.py                   # NeuralClusteringConfig
├── eval.py                     # 평가 스크립트
├── diagnose.py                 # gradient/수렴 테스트
│
├── nn/
│   ├── model.py                # NeuralClusteringModel (5 stage orchestrator)
│   ├── backbone.py             # Custom transformer backbone (fallback)
│   ├── vote.py                 # Stage 2: VoxelCenterPredictor
│   ├── diff_cluster.py         # Stage 3: DiffSoftClustering
│   ├── refine.py               # Stage 3.5: CrossAttentionRefiner
│   ├── gaussian_head.py        # Stage 4: GaussianParameterHead
│   ├── losses.py               # ClusteringLoss (NLL)
│   │
│   ├── seed_gen.py             # (미사용) Gumbel top-K seed generation
│   ├── soft_assign.py          # (미사용) Gumbel-Softmax assignment
│   │
│   └── ptv3/
│       ├── wrapper.py          # PTv3Backbone wrapper
│       ├── model.py            # PointTransformerV3 본체
│       └── serialization/
│           ├── default.py      # encode/decode 진입점
│           ├── z_order.py      # Z-order (Morton) curve
│           └── hilbert.py      # Hilbert curve
│
└── docs_method/
    ├── method2.md
    ├── method3.md
    └── method4.md              # ← 이 문서
```

---

## 설계 철학 요약

| 원칙 | 구현 |
|------|------|
| **End-to-end differentiable** | 전체 파이프라인에 gradient가 흐름 (PCA, voxel coords만 detach) |
| **Stochastic → deterministic** | τ annealing으로 soft→hard 수렴, Gumbel noise 불필요 |
| **Geometry-first** | 거리 기반 clustering + PCA warm-start, learned affinity 제거 |
| **Self-regularizing loss** | 단일 NLL = MLE, log-det가 자동 scale 균형 |
| **Zero-init residual** | 초기값 = PCA 기하학, 학습은 보정만 담당 |
| **Scalable seeding** | Gumbel top-K (O(N log K)) 대신 voxel pooling (O(N)), K 자동 결정 |
