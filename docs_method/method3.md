# Method 3: Neural 2D Gaussian Clustering

LiDAR 포인트 클라우드를 2D Gaussian surfel(면 소편) 집합으로 클러스터링하는 **완전 신경망 기반** 방법.
외부 GT 없이 포인트 클라우드 자체의 기하학적 구조만으로 self-supervised 학습.

---

## 전체 파이프라인

```
LiDAR Frame (N × 4: xyz + intensity)
          │
          ▼ ego-vehicle mask (r < 2.5m 제거)
          │
          ▼
┌─────────────────────────────────────────────┐
│  Module A: Point Feature Backbone           │
│                                             │
│  [xyz, intensity] → Linear(4→64) → LN+ReLU │
│          ↓                                  │
│  Block 0: Z-order sort → window attn (W=48)│
│  Block 1: Hilbert sort → shift+window attn │
│  Block 2: Z-order sort → window attn       │
│          ↓                                  │
│  features [N, 64]                           │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Module B: Seed Generation                  │
│                                             │
│  features → MLP(64→32→1) → raw_logits [N]  │
│           → sigmoid → all_scores [N]        │
│           → Gumbel top-k(1500)              │
│           → Ball-query NMS                  │
│           → K cap (N / 30)                  │
│          ↓                                  │
│  seed_indices [K], raw_logits [K]           │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Module C: Soft Assignment                  │
│                                             │
│  KNN: 각 점 → 가장 가까운 seed 8개 탐색     │
│  seed_logit → score_proj(1→64) → score_embed│
│  f_j_enriched = f_j + score_embed           │
│  affinity = MLP([f_i; f_j_enriched; f_i-f_j])│
│  geo_bias = -dist² / σ²                     │
│  logits = affinity + geo_bias               │
│  weights = Gumbel-Softmax(logits, τ)        │
│          ↓                                  │
│  assign_indices [N, 8], assign_weights [N, 8]│
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Module D: Gaussian Parameter Head          │
│                                             │
│  Weighted aggregation:                      │
│    mu_pca = Σ(w·xyz) / Σw                  │
│    cluster_feat = Σ(w·f) / Σw              │
│    cov = Σ(w·(xyz-mu)(xyz-mu)ᵀ) / Σw      │
│  PCA (no_grad SVD): q_pca, s_pca           │
│  Residual MLP:                              │
│    mu = mu_pca + MLP_mu(feat)              │
│    q  = normalize(q_pca + MLP_q(feat))     │
│    s  = softplus(log(s_pca) + MLP_s(feat)) │
│    α  = sigmoid(MLP_α(feat))               │
│  n = u × v  (quaternion으로부터 결정론적)   │
│          ↓                                  │
│  gaussians: mu[K,3], n[K,3], s[K,2],       │
│             q[K,4], u[K,3], v[K,3], α[K,1] │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  Loss (Self-Supervised)                     │
│                                             │
│  L_surface = Σ_j w_j · ((p-μ_j)·n_j)²    │
│  L_compact = Σ_j w_j · ||p-μ_j||²         │
│  L_scale   = mean(s₀ · s₁)                 │
│                                             │
│  L = L_surface + 0.01·L_compact + 0.01·L_scale │
└─────────────────────────────────────────────┘
```

---

## Module A: Point Feature Backbone

### 설계 아이디어
Point Transformer v3(PTv3) 논문에서 영감을 받은 from-scratch 구현.
핵심 아이디어: **공간 채움 곡선으로 3D 포인트를 1D 순서로 직렬화 → windowed self-attention** 으로 local context를 학습.

### 이중 직렬화 (Dual Serialization)
- **Z-order(Morton) curve**: 좌표 비트를 인터리빙(x₀y₀z₀x₁y₁z₁...)해 1D key 생성
- **Hilbert curve**: Skilling transpose 알고리즘으로 key 생성. Z-order보다 locality 보존 우수 — 공간적으로 인접한 점이 Hilbert 순서에서도 인접할 확률이 더 높음
- **블록 교대**: 짝수 블록 Z-order, 홀수 블록 Hilbert → 두 곡선의 locality 실패 케이스를 상호 보완
- **홀수 블록 shifted window (W/2)**: Swin Transformer 방식으로 window 경계를 넘어서는 정보 흐름 확보

### Windowed Self-Attention
- 직렬화 순서로 정렬 후 W=48개씩 window 분할
- 각 window 내에서 multi-head self-attention (H=4)
- `F.scaled_dot_product_attention` → Flash SDP 자동 활성 (RTX 3090, SM 8.6)
- positional encoding 없음 (PTv3와 동일)

### 하이퍼파라미터
| 파라미터 | 값 | 의미 |
|---------|-----|------|
| feature_dim | 64 | 잠재 특징 차원 |
| num_blocks | 3 | transformer 블록 수 |
| window_size | 48 | 로컬 attention 범위 |
| num_heads | 4 | multi-head attention |

---

## Module B: Seed Generation

### 설계 아이디어
포인트 클라우드에서 클러스터 중심 후보(seed)를 선택.
학습 가능한 MLP가 각 점의 "seed 적합도"를 예측.

### 흐름
1. `raw_logits = MLP(features)` [N] — 각 점의 seediness score (sigmoid 전)
2. `all_scores = sigmoid(raw_logits)` — topk 선택용 확률
3. **Gumbel top-k** (훈련 시): `perturbed = all_scores + Gumbel(τ)` → topk(1500)
   - Gumbel noise가 탐색(exploration)을 유도, τ 어닐링으로 점차 결정론적으로 수렴
4. **Ball-query NMS**: NMS 반경 내에서 score 낮은 seed 억제 → 공간적 중복 제거
5. **Adaptive K cap**: NMS 후 `K ≤ N / target_cluster_size` 로 상한 → train/eval K 일관성

### Gradient 연결 (핵심)
`topk()`는 정수 인덱스를 반환하므로 gradient가 끊김.
**해결**: `raw_logits`를 Module C로 직접 전달 → `score_proj`를 통해 f_j에 주입 → 기존 loss의 gradient가 MLP까지 흐름.

```
loss → affinity → f_j_enriched → score_embed → raw_logits[seed_indices] → MLP ✓
```

`raw_logits[seed_indices]`는 gather 연산이므로 differentiable.

---

## Module C: Soft Assignment

### 설계 아이디어
각 점을 K개의 seed 중 가장 적합한 cluster에 배정.
**Learned affinity + distance bias** 의 조합으로 배정 확률 계산.

### Score-Enriched f_j
```python
score_embed = score_proj(raw_logits[seed_indices][knn_idx])  # [N, k, 64]
f_j_enriched = f_j + score_embed
```
- seed의 raw logit을 `Linear(1→64)`로 feature 공간에 투영 (zero-init → 학습 시작 시 영향 없음)
- f_j에 더해 "이 seed가 얼마나 좋은 중심인가" 정보를 affinity 계산에 반영
- `f_i - f_j`는 원본 유지 → 기하학적 유사도 신호 보존

### Affinity 계산
```python
interaction = [f_i; f_j_enriched; f_i - f_j]  # [N, k, 192]
affinity = MLP(interaction)                     # [N, k]
geo_bias = -dist² / σ²                          # 거리 기반 prior
logits = affinity + geo_bias
```
- `f_i - f_j`: 점과 seed의 feature 차이 (같은 표면인지 여부)
- `f_j_enriched`: seed 정체성 + seed 품질
- `geo_bias`: 가까운 seed를 선호하는 soft prior

### Gumbel-Softmax
- **훈련**: `weights = Gumbel-Softmax(logits, τ)` — soft, differentiable
- **추론**: argmax → one-hot (hard, deterministic)
- τ: 1.0 → 0.1 어닐링 → 훈련 후반 hard assignment에 수렴

---

## Module D: Gaussian Parameter Head

### 설계 아이디어
K개 클러스터 각각에 대해 2D Gaussian surfel 파라미터 예측.
**PCA warm-start + residual MLP correction** 구조.

### Weighted Aggregation
assign_weights를 기반으로 클러스터별 포인트 집합을 soft하게 정의:
```
mu_pca  = Σᵢ wᵢⱼ · xyzᵢ / Σᵢ wᵢⱼ        (가중 중심)
feat_j  = Σᵢ wᵢⱼ · featᵢ / Σᵢ wᵢⱼ        (가중 feature)
cov_j   = Σᵢ wᵢⱼ · (xyzᵢ-μⱼ)(xyzᵢ-μⱼ)ᵀ  (가중 공분산)
```

### PCA Warm-start (no_grad)
- 공분산 행렬의 SVD → 주축 방향(u, v, n)과 축 크기(s) 획득
- **SVD 전체를 `torch.no_grad()` 래핑**: SVD backward는 수치적으로 불안정, 이미 detach하므로 gradient 계산 불필요 → 메모리·속도 절감
- 초기 파라미터를 기하학적으로 올바른 값으로 시작 → 학습 안정성

### Residual MLP
- PCA 결과를 기준으로 잔차만 예측
- 마지막 레이어 zero-init → 초기 출력 ≈ PCA 결과
- `n = u × v`: 법선은 quaternion으로부터 결정론적으로 도출 (독립적 파라미터 없음)

---

## Loss 설계

### L_surface (w=1.0) — 핵심 손실
```
γ_j = (p - μ_j) · n_j        (점-to-plane 거리)
L_surface = mean_p[ Σ_j w_j · γ_j² ]
```
각 점이 배정된 Gaussian의 평면 위에 놓이도록 강제.
soft weight를 써서 여러 후보 Gaussian에 부드럽게 기여 → Gumbel τ 감소로 점차 hard 수렴.

### L_compact (w=0.01)
```
L_compact = mean_p[ Σ_j w_j · ||p - μ_j||² ]
```
클러스터가 너무 크게 퍼지지 않도록 억제. w가 작은 이유: 지나치게 세분화되면 K가 폭발하는 부작용 방지.

### L_scale (w=0.01)
```
L_scale = mean_k[ s_k,0 · s_k,1 ]
```
Gaussian 면적(두 반축 곱)에 패널티. 모든 점을 하나의 거대한 Gaussian이 커버하는 trivial solution 방지.

---

## 핵심 설계 결정 및 근거

| 결정 | 근거 |
|------|------|
| PTv3 from-scratch (라이브러리 미사용) | pointcept 의존성 없음, 커스터마이징 용이, 핵심 아이디어 이미 80% 자체 구현 |
| Z-order + Hilbert 교대 | Z-order 경계 locality 실패를 Hilbert가 보완, 비용 증가 없음 |
| Normal 추정 KNN 제거 | backbone feature가 표면 기하를 인코딩 → learned affinity가 대체 |
| SVD no_grad | backward 수치 불안정, 결과 detach하므로 gradient 불필요 |
| score_proj (zero-init) | 학습 시작 시 기존 동작 보존, 점진적으로 seed 품질 정보 반영 |
| Adaptive K cap | Gumbel noise로 인한 train/eval K 불일치 방지 |
| Soft loss (τ 어닐링) | Gaussian이 soft weight로 만들어지므로 loss도 soft가 일관적, 훈련 후반 hard 수렴 |
| Pseudo-GT 없음 | 클러스터링 품질이 자체 기하학으로 측정 가능 (γ_rms, coverage) |

---

## 하이퍼파라미터 요약

| 파라미터 | 값 |
|---------|-----|
| feature_dim | 64 |
| num_blocks | 3 |
| window_size | 48 |
| num_heads | 4 |
| k_max | 1500 |
| target_cluster_size | 30 |
| top_k_seeds | 8 |
| ego_radius | 2.5m |
| τ_start → τ_end | 1.0 → 0.1 |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| epochs | 100 |
| w_surface : w_compact : w_scale | 1.0 : 0.01 : 0.01 |
| total params | 124,172 |
