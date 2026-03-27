# Neural Clustering with 2D Gaussians

## 0. Overview

### 문제 정의

method1.md에서 설명한 hand-crafted 파이프라인(~12초/프레임)은 PCA, 복셀 시딩, split/merge 등의 규칙 기반 단계를 순차적으로 수행한다. 이 접근의 한계:

1. **속도**: 단계별 순차 실행, 특히 refinement의 반복 루프가 병목
2. **고정된 규칙**: split 기준(gamma_rms > 0.05), merge 기준(각도 < 5도) 등이 하이퍼파라미터로 고정
3. **End-to-end 학습 불가**: 각 단계가 독립적이므로 전체 최적화 불가

**Neural pipeline은 이 전체를 단일 forward pass 네트워크로 대체한다.**

### 전체 파이프라인 위치

```
전체 시스템: 2 LiDAR frames → [Clustering with 2D Gaussian] → Temporal → Rendering
                                ├── Baseline: ~12초, 규칙 기반 (method1.md)
                                └── Neural:  ~0.5초, 학습 기반 (이 문서)
```

**현재 설계 범위**: `[Clustering with 2D Gaussian]` 모듈. 각 프레임을 독립적으로 처리한다.

Gaussian의 역할은 **Structure** — 점들을 의미 있는 그룹으로 묶는 도구이다. 렌더링은 temporal module 이후에 수행되므로, 이 모듈에서는 rendering loss를 사용하지 않는다.

### 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **No Pseudo-GT** | Hand-crafted 파이프라인의 출력을 GT로 사용하지 않음 |
| **No Rendering Loss** | Gaussian = structure, 렌더링은 temporal 이후 |
| **Self-supervised** | 입력 point cloud의 기하학이 유일한 supervision |
| **기존 dataloader 사용** | `~/data/nuScenes/loader/dataset.py`의 `NuScenesNVSDataset`을 그대로 사용 |

### 아키텍처 요약

```
Single Frame (N×4)
    ↓
[A] Point Feature Backbone        Z-order 직렬화 + 윈도우 어텐션
    ↓  N×D per-point features
[B] Seed Generation                Gumbel top-K + Ball-query NMS
    ↓  K개 seed indices
[C] Soft Assignment                Learned affinity + Geometric bias + Gumbel-Softmax
    ↓  N×K sparse soft matrix
[D] Gaussian Parameter Head        Weighted PCA 초기화 + Residual MLP
    ↓
Output: K개 2D Gaussian surfels (μ, q, s, n, α) + point-to-Gaussian assignments
```

### 입출력

**입력**: 단일 LiDAR 프레임 (N ≈ 22k점 after ego-masking, xyz + intensity)

**출력**:
```python
{
    "gaussians": {
        "mu":    [K, 3],    # Gaussian 중심 위치
        "q":     [K, 4],    # 쿼터니언 (w, x, y, z)
        "s":     [K, 2],    # 2D 스케일 (장축, 단축)
        "n":     [K, 3],    # 표면 법선 (q에서 도출: n = u × v)
        "alpha": [K, 1],    # 불투명도
        "u":     [K, 3],    # 탄젠트 벡터 u
        "v":     [K, 3],    # 탄젠트 벡터 v
    },
    "assign_indices": [N, 8],  # 각 점의 top-8 후보 seed 인덱스
    "assign_weights": [N, 8],  # soft assignment 가중치 (합 ≈ 1)
    "seed_indices":   [K],     # seed 점 인덱스
    "seed_scores":    [N],     # 전체 점의 seediness 점수
    "normals":        [N, 3],  # PCA 법선 (geometric bias용)
}
```

---

## 1. Module A: Point Feature Backbone

> 파일: `nn/backbone.py`

### 1-1. 설계 근거

클러스터링에 두 가지 정보가 필요하다:
1. **Local surface geometry**: 같은 표면의 점은 비슷한 법선/곡률
2. **Surface boundary 인식**: 공간적으로 가까워도 다른 표면이면 분리

PCA/KNN만으로는 (1)은 가능하지만 (2)는 불가하다 — local neighborhood만 보기 때문에 경계 부근에서 양쪽 면의 점이 섞인다. **이웃 정보를 전파하여 boundary를 인식할 수 있는 feature extractor**가 필요하다.

Attention 블록을 쌓으면 multi-hop 정보 전파로 "내 이웃은 이렇지만 그 너머는 다르다"를 인코딩할 수 있다.

### 1-2. Space-Filling Curve (Z-Order) 직렬화

3D 점을 1D로 직렬화하여 windowed attention의 이웃을 정의한다.

**Z-order (Morton) 코드 계산:**

```
1. 좌표 정규화:  xyz → [0, 1023] 정수 (10비트)
2. 비트 인터리빙:
   x=5 (0101₂), y=3 (0011₂), z=2 (0010₂)
   → x₀ y₀ z₀ x₁ y₁ z₁ x₂ y₂ z₂ ... = 30비트 정수
3. 정렬: Z-order 키로 오름차순 정렬
```

**이웃 정의로 타당한 이유**: 정렬된 순서에서 window(W=48)를 잡으면 그 안의 점들은 대부분 3D에서도 가까이 위치한다. KNN 그래프를 명시적으로 구축하는 대신(O(N²)), O(N log N) 정렬로 이웃을 근사한다.

### 1-3. Receptive Field 확장

단일 블록의 window=48은 약 1-2m 범위(local)이다. 블록을 쌓으면 receptive field가 확장된다:

| 블록 수 | 범위 | 의미 |
|---------|------|------|
| 1 | ~48 이웃 직접 접촉 (~2m) | 즉각적 이웃 |
| 2 | 이웃의 이웃 정보 (~5m) | Medium range |
| 3 | 3-hop (~10m+) | 표면 경계 감지에 충분 |

**Shifted window**: 홀수 블록에서 W/2만큼 시프트하여 윈도우 경계의 정보 단절을 방지한다 (Swin Transformer와 동일 기법).

```
Block 0: [----window 0----][----window 1----][----window 2----]...
Block 1:     [----window 0----][----window 1----][----window 2--]...  (W/2 시프트)
Block 2: [----window 0----][----window 1----][----window 2----]...
```

### 1-4. 구조 상세

```
Input: xyz (N×3) + intensity (N×1) → cat → N×4
  ↓
Linear(4 → 64) + LayerNorm + ReLU         # 초기 임베딩
  ↓
Z-order 직렬화 (정렬)
  ↓
┌─ SerializedTransformerBlock × 3 ──────────────────────────┐
│  LayerNorm → WindowedAttention(4 heads, W=48) → residual  │
│  LayerNorm → FFN(64 → 128 → 64) → residual                │
│  (홀수 블록: W/2 시프트)                                    │
└────────────────────────────────────────────────────────────┘
  ↓
역정렬 (원래 순서 복원)
  ↓
Output: F ∈ R^(N×64) per-point features
```

**LayerNorm** (not BatchNorm): Point cloud는 프레임마다 점 개수/분포가 다르다. BatchNorm의 batch 통계가 불안정하므로, 각 점의 feature를 독립적으로 정규화하는 LayerNorm을 사용한다.

**WindowedAttention 내부:**
```
QKV = Linear(D → 3D)
Q, K, V를 num_heads개로 분할 → head_dim = 64/4 = 16
Attention = softmax(QK^T / √d_k) × V     (PyTorch scaled_dot_product_attention)
Output projection = Linear(D → D)
패딩 토큰은 attention mask로 제외
```

| 파라미터 | 값 | 근거 |
|---------|---|------|
| D (feature_dim) | 64 | LiDAR sparse geometry에 충분. 메모리 효율적 |
| L (num_blocks) | 3 | 3-hop으로 medium-range context 커버 |
| W (window_size) | 48 | nuScenes 점 밀도에서 ~1-2m 반경 |
| H (num_heads) | 4 | head_dim=16, standard practice |

---

## 2. Module B: Seed Generation

> 파일: `nn/seed_gen.py`

### 2-1. 설계 근거

클러스터링 = N→K 매핑. K개 seed를 먼저 정하면 나머지는 assignment 문제로 환원된다. Seed 품질이 전체 클러스터링을 결정한다.

좋은 seed: **표면 내부에 위치, 서로 적절히 이격, 해당 패치의 대표점**

| 방법 | 문제 |
|------|------|
| Voxel grid | 빈 공간에도 seed, 표면 무시 |
| FPS | 공간 균일성만, 표면 무시 |
| **학습된 seediness score** | "표면 패치의 좋은 대표점인가?"를 직접 예측 |

### 2-2. 구조

```
Input: F (N×64)
  ↓
MLP: 64 → 32 → 1, Sigmoid → s_i ∈ [0, 1]    # per-point seediness score
  ↓
Gumbel top-K (training) / deterministic top-K (inference)
  → K_max = 1500개 후보
  ↓
Ball-query NMS (adaptive radius)
  → K_final개 seed (중복 제거)
  ↓
Output: seed_indices (K_final,), seed_scores (K_final,), all_scores (N,)
```

### 2-3. Differentiable Top-K: Gumbel Noise

**문제**: `topk(scores, K)`의 backward에서 선택되지 않은 N-K개 점은 gradient=0 → score 개선 기회 없음 (rich-get-richer).

**해결**: score에 Gumbel noise를 추가하여 매 forward마다 다른 점이 선택될 수 있게 한다:

```python
gumbel_noise = -log(-log(Uniform(0, 1)))
perturbed = scores + gumbel_noise × τ
_, indices = perturbed.topk(K_max)
```

**적절한 점이 탈락해도 학습이 진행되는 이유:**
1. score=0.9인 점이 탈락하려면 noise가 매우 큰 음수여야 → 확률 낮음
2. 같은 frame이 여러 epoch에 반복됨 → 탈락해도 다른 epoch에서 선택 → gradient 평균화
3. **τ annealing** (1.0 → 0.1): 학습 후기에는 거의 deterministic → 탈락 확률 ≈ 0

**Inference**: noise 없이 순수 topk. 결정론적.

### 2-4. Ball-Query NMS

Top-K로 선택된 K_max개 후보에서 공간적으로 중복되는 seed를 제거한다.

```
1. Score 내림차순 정렬
2. 가장 높은 score의 seed를 선택 (keep)
3. 반경 r 내의 모든 seed를 억제 (suppress)
4. 다음으로 높은 미억제 seed를 선택, 반복
```

**적응형 반경 계산** (이상치에 강건):

```
p02, p98 = xyz의 2%, 98% 백분위수
effective_range = p98 - p02                     (이상치 제거된 유효 범위)
effective_vol = Π(effective_range)              (유효 부피)
voxel_vol = effective_vol / (N / target_cluster_size)
voxel_size = voxel_vol^(1/3)
nms_radius = voxel_size × 0.3                  (nms_radius_factor)
```

전체 bbox 대신 백분위수 범위를 사용하는 이유: nuScenes LiDAR는 원거리에 sparse한 이상치가 있어 bbox가 과도하게 팽창한다. 예: 전체 bbox 765k m³ vs 유효 범위 기반 ~50k m³.

---

## 3. Module C: Soft Assignment

> 파일: `nn/soft_assign.py`

### 3-1. 설계 근거

Hard assignment(argmin)는 미분 불가 → end-to-end 학습 불가. Soft assignment로 gradient를 전달하고, inference 시에는 argmax로 hard assignment를 사용한다.

### 3-2. Spatial KNN과 Gumbel Softmax의 역할 분리

이 둘은 서로 다른 단계이다:

| 단계 | 역할 | 타입 |
|------|------|------|
| **Spatial KNN (top-8)** | **후보 범위** 결정 | Hard filter |
| **Gumbel Softmax** | 후보 **내에서** 할당 | Soft decision |

KNN만 쓰면 → 가장 가까운 seed에 hard assign (미분 불가).
KNN + Gumbel Softmax → top-8 내에서 확률 분배 → gradient flow.

### 3-3. Learned Affinity

```
f_i: 점 i의 feature          [D]
f_j: seed j의 feature        [D]

interaction = [f_i; f_j; f_i - f_j]    # 3D 차원 = 192
a_ij = MLP(192 → 64 → 1) → scalar
```

`f_i - f_j`를 명시적으로 제공하는 이유: 방향성 있는 차이 — "어떤 측면에서 다른가"를 MLP가 implicit으로 학습하지 않아도 된다.

### 3-4. Geometric Bias

```
g_ij = -||xyz_i - seed_j||² / σ²  -  λ_n × (1 - |n_i · n_seed_j|)
         \___ spatial ___/            \_____ normal consistency _____/
```

"가까운 점 + 비슷한 법선 → 같은 클러스터"라는 기하학적 prior를 명시적으로 주입한다.

- σ²: 전체 점-seed 거리의 평균 (자동 스케일 조정, detach)
- λ_n: 법선 일관성 가중치 (기본 1.0)
- 법선은 Module A 이전에 PCA로 추정 (비학습, detach)

Learned affinity는 이 prior 위에 추가되어, 기하학만으로 결정 불가능한 부분(경계, 재질 변화)에 집중한다.

### 3-5. 전체 구조

```
Input: F (N×D), xyz (N×3), normals (N×3), seed_indices (K,)
  ↓
Spatial KNN: top-8 nearest seeds per point → knn_idx [N, 8]
  ↓
Learned Affinity: MLP([f_i; f_j; f_i - f_j]) → a_ij [N, 8]
  ↓
Geometric Bias: g_ij = -dist²/σ² - λ(1 - |n·n'|) → [N, 8]
  ↓
logit_ij = a_ij + g_ij
  ↓
Training: Gumbel-Softmax(logits, τ) → soft weights [N, 8]
Inference: argmax → one-hot weights [N, 8]
  ↓
Output: assign_indices [N, 8], assign_weights [N, 8]
```

---

## 4. Module D: Gaussian Parameter Prediction

> 파일: `nn/gaussian_head.py`

### 4-1. 설계 근거

2D Gaussian surfel = local plane approximation. PCA가 최적 평면을 수학적으로 정확히 계산하므로, **PCA를 초기값으로 사용하고, 네트워크는 residual만 학습**한다.

### 4-2. Weighted Aggregation

Soft assignment matrix P (N×8 sparse)를 이용하여 각 클러스터의 대표 feature와 중심점을 계산한다.

```
F_cluster_j = Σ_i P_ij × F_i / Σ_i P_ij       # 가중 평균 feature (K×D)
μ_weighted_j = Σ_i P_ij × xyz_i / Σ_i P_ij    # 가중 중심 (K×3)
```

구현: `scatter_add_`로 GPU 벡터화. 각 점이 최대 8개 클러스터에 기여하므로, 입력을 N×8으로 flatten하여 한 번의 scatter로 처리한다.

### 4-3. Differentiable PCA

각 클러스터의 가중 공분산 행렬을 계산하고 SVD로 분해한다:

```
C_j = Σ_i P_ij × (xyz_i - μ_j)(xyz_i - μ_j)^T / Σ_i P_ij   (가중 3×3 공분산)
C_j += 1e-6 × I                                                (수치 안정성)
SVD(C_j) → Vh = [u, v, n_pca], S = [λ₀, λ₁, λ₂]

s_pca = [√λ₀, √λ₁]                                           (PCA 기반 scaling)
q_pca = rotation_matrix_to_quaternion([u, v, n_pca])           (PCA 기반 rotation)
```

**법선 방향 통일**: `n_pca`가 센서 원점을 향하도록 뒤집음. `v_pca`도 함께 뒤집어 오른손 좌표계를 유지.

**PCA는 detach**: SVD backward는 특이값이 가까울 때 수치적으로 불안정하다 (gradient에 1/(s_i - s_j) 항). PCA 결과를 detach하여 gradient를 차단하고, 네트워크는 residual MLP를 통해 학습한다. PCA는 좋은 warm start를 제공하는 역할만 한다.

### 4-4. Residual Parameter Prediction

PCA 초기값 위에 MLP가 보정값을 예측한다:

```
μ = μ_weighted + MLP_μ(F_cluster)                   # 위치: 가중 중심 + 보정
q = normalize(q_pca + MLP_q(F_cluster))              # 회전: PCA + 보정
s = softplus(log(s_pca) + MLP_s(F_cluster))          # 스케일: PCA + 보정
α = sigmoid(MLP_α(F_cluster))                        # 불투명도: 순수 학습

각 MLP: D → D/2 → output_dim
```

**초기화**: MLP_μ, MLP_q, MLP_s의 마지막 층은 가중치/편향을 0으로 초기화한다. 이렇게 하면 학습 초기에 네트워크 출력 ≈ PCA 결과이므로, 합리적인 시작점에서 학습이 시작된다.

### 4-5. Normal은 q에서 Deterministic하게 도출

n을 별도 MLP로 예측하면 q(rotation)의 u, v와 불일치할 수 있다. 대신:

```
q → R(q) → u = R[0, :], v = R[1, :]
n = u × v   (외적, 자동으로 u, v에 직교)
```

외적은 완전히 differentiable하다:
```
n = [u₁v₂ - u₂v₁, u₂v₀ - u₀v₂, u₀v₁ - u₁v₀]
```
`torch.cross(u, v, dim=-1)`로 gradient가 u, v → q → MLP_q까지 흘러간다.

### 4-6. 전체 구조 요약

```
Input: F (N×D), xyz (N×3), P (N×8 sparse), seed_indices (K,)
  ↓
Weighted Aggregation (scatter_add)
  → F_cluster (K×D), μ_weighted (K×3)
  ↓
Weighted PCA (공분산 + SVD, detach)
  → q_pca (K×4), s_pca (K×2)
  ↓
Residual Heads
  ┌─ μ = μ_weighted + MLP_μ(F_cluster)           [K, 3]
  ├─ q = normalize(q_pca + MLP_q(F_cluster))      [K, 4]
  ├─   → R(q) → u, v → n = u × v                 [K, 3]
  ├─ s = softplus(log(s_pca) + MLP_s(F_cluster)) [K, 2]
  └─ α = sigmoid(MLP_α(F_cluster))                [K, 1]
  ↓
Output: {μ, q, s, n, α, u, v}
```

---

## 5. Loss Design

> 파일: `nn/losses.py`

### 5-0. 설계 원칙

1. Pseudo-GT 미사용
2. Rendering loss 미사용 — Gaussian = Structure
3. 입력 point cloud의 기하학이 유일한 supervision

### 5-1. L_surface: Surface Reconstruction Loss (Primary)

**"각 점이 자신의 Gaussian 표면에 가까운가?"**

각 점 i에 대해, 할당된 후보 Gaussian j들과의 off-plane 거리를 soft assignment 가중치로 합산한다:

```
γ_ij = (xyz_i - μ_j) · n_j                         # off-plane 거리
L_surface = (1/N) Σ_i Σ_j P_ij × γ_ij²             # 가중 평균
```

Soft assignment를 사용하여 gradient가 μ, n, 그리고 P(assignment weights)까지 모두 흐른다.

Gradient path: L → γ² → n → u×v → q → MLP_q → F_cluster → backbone

### 5-2. L_CD: Chamfer Distance (Coverage + Faithfulness)

**"Gaussian이 입력 점들을 빈틈없이 커버하는가?"**

Gaussian surfel 표면에서 미분 가능하게 점을 샘플링한다:

```
sampled_j = μ_j + α × u_j + β × v_j,   α,β ~ N(0, diag(s_j²))
```

양방향 Chamfer distance:
```
L_in→gauss = (1/N) Σ_i min_j ||xyz_i - sampled_j||²    # 모든 점이 Gaussian 근처?
L_gauss→in = (1/M) Σ_j min_i ||sampled_j - xyz_i||²    # Gaussian이 실제 점 근처에만?
L_CD = L_in→gauss + L_gauss→in
```

L_surface만으로는 개별 점-Gaussian 품질만 측정한다. CD로 전역 커버리지를 확인한다.

**구현 세부**: 클러스터당 8개 점을 샘플링. 메모리 절약을 위해 4096개 점씩 chunked cdist를 수행한다.

### 5-3. L_assign: Assignment Regularization

```
L_entropy = -(1/N) Σ_i Σ_j P_ij × log(P_ij)    # sharp assignment 유도
L_compact = (1/N) Σ_i Σ_j P_ij × ||xyz_i - μ_j||²   # 공간적 compactness
L_assign = L_entropy + L_compact
```

- **Entropy**: soft assignment가 하나의 seed에 집중되도록 유도 (uniform → peaked)
- **Compact**: 할당된 점이 Gaussian 중심에 가깝도록 유도

### 5-4. L_scale: Scale Regularization

```
L_scale = (1/K) Σ_j s_j[0] × s_j[1]             # 거대 Gaussian 방지
```

제약 없으면 Gaussian이 과도하게 커져서 여러 표면을 걸칠 수 있다.

### 5-5. Total Loss

```
L = 1.0 × L_surface + 0.5 × L_CD + 0.01 × L_assign + 0.01 × L_scale
```

| 항 | 가중치 | 역할 |
|----|--------|------|
| L_surface | 1.0 | 평면 적합도 (핵심) |
| L_CD | 0.5 | 전역 커버리지 |
| L_assign | 0.01 | Assignment 정규화 |
| L_scale | 0.01 | 크기 정규화 |

---

## 6. Training Procedure

> 파일: `train.py`

### 6-1. Temperature Annealing

Gumbel noise(Module B)와 Gumbel-Softmax(Module C)의 temperature τ를 선형으로 감소시킨다:

```
τ(epoch) = τ_start + (τ_end - τ_start) × epoch / (num_epochs - 1)
         = 1.0 → 0.1 (선형)
```

| τ | Module B (seed selection) | Module C (assignment) |
|---|---------------------------|----------------------|
| 1.0 (초기) | 탐색적: 다양한 seed 시도 | 부드러운 할당: 여러 seed에 분산 |
| 0.1 (후기) | 결정적: 최적 seed 유지 | 날카로운 할당: 하나의 seed에 집중 |

### 6-2. 데이터 로딩

`~/data/nuScenes/loader/dataset.py`의 `NuScenesNVSDataset`을 사용한다. 이 데이터로더는:
- **2 프레임 쌍** (input_0, input_1)을 반환 — 전체 파이프라인의 입력 형태와 일치
- 각 프레임은 `(N, 4)` 텐서 (xyz + intensity)
- pose 정보, intermediate sweep GT 등도 함께 반환 (temporal module에서 사용)
- `nvs_collate_fn`으로 가변 길이 point cloud를 리스트로 처리하여 **batch_size > 1** 지원

Training loop에서 수행:
- **Ego-vehicle 마스크**: 원점에서 2.5m 이내 점 제거
- 배치 내 각 프레임(input_0, input_1)을 독립적으로 모델에 통과
- loss는 배치 내 전체 프레임(B × 2)에 대해 평균

Model forward pass 내에서 수행:
- **PCA 법선 추정**: KNN(k=30) + SVD로 per-point normal 계산 (Module C의 geometric bias용)
- 비학습, detach 처리

### 6-3. Optimizer

- **AdamW** (lr=1e-3, weight_decay=1e-4)
- **Gradient clipping**: max_norm=1.0
- **NaN 방지**: loss가 NaN이면 해당 batch skip

### 6-4. Checkpointing

Best loss 기준으로 `outputs/best_model.pt`에 저장:
```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "loss": float,
}
```

---

## 7. Evaluation Metrics

### 7-1. Geometric Self-Evaluation (GT 불필요)

| 지표 | 수식 | 목표 |
|------|------|------|
| **gamma_rms** | √(mean(γ²)), γ = (p - μ)·n | < 0.05m (baseline 수준) |
| **Coverage** | max(P, dim=1) > 0.1인 점 비율 | > 95% |
| **num_gaussians (K)** | seed 수 | ~100-400 (프레임에 따라) |

### 7-2. Chamfer Distance

Gaussian surfel에서 점 샘플링 → 입력과 CD 측정. 양방향이므로 커버리지 + 정확도 동시 평가.

### 7-3. Overfit Protocol

| 단계 | 설명 | 성공 기준 |
|------|------|----------|
| 1-pair overfit | 단일 쌍(2프레임), 50-1000 epoch | loss → ~0, gamma_rms < 0.01 |
| 10-pair overfit | 10쌍 학습, 별도 10쌍 평가 | gap < 20% |
| 전체 학습 | nuScenes 전체 train split | eval gamma_rms ≤ 1.2× baseline |

---

## 8. 현재 검증 결과

### 모델 규모

```
학습 가능 파라미터: 124,044
```

| 모듈 | 주요 파라미터 |
|------|-------------|
| Backbone (embed + 3 blocks) | Linear(4→64), 3×(QKV + proj + FFN) |
| Seed MLP | 64 → 32 → 1 |
| Affinity MLP | 192 → 64 → 1 |
| Gaussian heads | 4 × (64 → 32 → output) |

### 1-frame Overfit (50 epochs)

```
N = 22,472점, K ≈ 115 seeds

Epoch  1: loss=69.45, surface=65.43, gamma_rms=8.08, K=118, coverage=100%
Epoch 15: loss=11.16, surface= 7.60, gamma_rms=2.74, K=115, coverage=100%
Epoch 35: loss= 5.78, surface= 2.56, gamma_rms=1.59, K=116, coverage=100%
Epoch 50: loss= 6.03, surface= 2.45, gamma_rms=1.57, K=117, coverage=100%
```

Surface loss 27x 감소, gamma_rms 5.2x 감소. 수렴 추세 확인.

### Performance

| 측정 | 훈련 | 추론 |
|------|------|------|
| Forward pass | ~0.6s | ~0.5s |
| GPU memory | ~2.3 GB | ~0.7 GB |

---

## 9. 하이퍼파라미터 전체 목록

> 파일: `config.py:NeuralClusteringConfig`

| 파라미터 | 값 | 모듈 | 의미 |
|----------|-----|------|------|
| `feature_dim` | 64 | A | Backbone feature 차원 |
| `num_blocks` | 3 | A | Transformer block 수 |
| `window_size` | 48 | A | Attention window 크기 |
| `num_heads` | 4 | A | Multi-head attention head 수 |
| `k_max` | 1500 | B | Top-K 후보 seed 수 |
| `nms_radius_factor` | 0.3 | B | NMS 반경 = voxel_size × factor |
| `target_cluster_size` | 30 | B | 클러스터당 목표 점 수 (반경 계산용) |
| `top_k_seeds` | 8 | C | 점당 후보 seed 수 |
| `geo_lambda_normal` | 1.0 | C | 법선 geometric bias 가중치 |
| `ego_radius` | 2.5m | pre | 자차 마스크 반경 |
| `knn_k` | 30 | pre | PCA 법선 추정용 이웃 수 |
| `gumbel_tau_start` | 1.0 | train | 초기 Gumbel temperature |
| `gumbel_tau_end` | 0.1 | train | 최종 Gumbel temperature |
| `lr` | 1e-3 | train | 학습률 |
| `weight_decay` | 1e-4 | train | AdamW weight decay |
| `num_epochs` | 100 | train | Epoch 수 |
| `batch_size` | 2 | train | 배치 크기 (프레임 쌍 수) |
| `w_surface` | 1.0 | loss | Surface loss 가중치 |
| `w_cd` | 0.5 | loss | Chamfer Distance 가중치 |
| `w_assign` | 0.01 | loss | Assignment 정규화 가중치 |
| `w_scale` | 0.01 | loss | Scale 정규화 가중치 |
| `num_cd_samples` | 8 | loss | CD용 Gaussian당 샘플 수 |

---

## 10. 파일 구조

```
gaustering/
├── nn/                         # Neural clustering pipeline
│   ├── __init__.py             # exports NeuralClusteringModel
│   ├── backbone.py             # Module A: Z-order serialized point transformer
│   ├── seed_gen.py             # Module B: Gumbel top-K + ball-query NMS
│   ├── soft_assign.py          # Module C: learned affinity + geometric bias
│   ├── gaussian_head.py        # Module D: weighted PCA + residual heads
│   ├── losses.py               # L_surface + L_CD + L_assign + L_scale
│   └── model.py                # Full model: A → B → C → D
│
├── baseline/                   # Hand-crafted pipeline (method1.md)
│   ├── __init__.py
│   ├── pipeline.py             # cluster_frame() 전체 파이프라인
│   ├── assignment.py           # Stage 3
│   ├── refinement.py           # Stage 4
│   ├── ground.py               # 지면 검출
│   └── seeding.py              # Stage 2
│
├── config.py                   # ClusteringConfig + NeuralClusteringConfig
├── data_loader.py              # nuScenes .pcd.bin 로딩 (변경 없음)
├── geometry.py                 # KNN, PCA (backbone/assignment에서 재사용)
├── gaussian_fit.py             # quaternion 변환 (gaussian_head에서 재사용)
├── visualization.py            # 통계 플롯
├── train.py                    # 학습 루프
├── test_nuscenes.py            # Baseline 단일 프레임 테스트
├── eval_batch.py               # Baseline 배치 평가
└── viz_server.py               # 3D 인터랙티브 뷰어
```

---

## 11. 사용법

### 11-1. Training

**전체 학습** (nuScenes 전체 train split):

```bash
python train.py --data-root ~/data/nuScenes --epochs 100 --batch-size 4
```

**1-pair overfit** (디버깅/검증용):

```bash
python train.py --overfit 1 --epochs 1000
```

**10-pair overfit** (일반화 테스트):

```bash
python train.py --overfit 10 --epochs 200
```

**주요 인자:**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data-root` | `~/data/nuScenes` | nuScenes 데이터 경로 |
| `--epochs` | 100 | 학습 epoch 수 |
| `--lr` | 1e-3 | 학습률 |
| `--batch-size` | 2 | 배치 크기 (프레임 쌍 수, 실제 프레임 수 = B×2) |
| `--overfit` | 0 | N > 0이면 N개 쌍만 사용 |
| `--device` | `cuda` | 디바이스 |

학습 중 best loss 기준으로 `outputs/best_model.pt`가 자동 저장된다.

### 11-2. Inference (Python API)

```python
import torch
from config import NeuralClusteringConfig
from nn.model import NeuralClusteringModel
from data_loader import load_pcd_bin

# 모델 로드
cfg = NeuralClusteringConfig()
model = NeuralClusteringModel(cfg).to("cuda")
ckpt = torch.load("outputs/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 단일 프레임 추론
data = load_pcd_bin("path/to/frame.pcd.bin", device="cuda")
xyz = data["xyz"]
intensity = data["intensity"]

# Ego masking
mask = torch.norm(xyz, dim=1) > cfg.ego_radius
xyz, intensity = xyz[mask], intensity[mask]

with torch.no_grad():
    output = model(xyz, intensity, tau=0.1)

# 결과 접근
gaussians = output["gaussians"]
print(f"Gaussians: {gaussians['mu'].shape[0]}")
print(f"mu:    {gaussians['mu'].shape}")      # [K, 3]
print(f"q:     {gaussians['q'].shape}")       # [K, 4]
print(f"s:     {gaussians['s'].shape}")       # [K, 2]
print(f"n:     {gaussians['n'].shape}")       # [K, 3]
print(f"alpha: {gaussians['alpha'].shape}")   # [K, 1]

# Hard assignment (inference에서는 one-hot)
assign_idx = output["assign_indices"]    # [N, 8]
assign_w = output["assign_weights"]      # [N, 8] — one-hot in eval mode
hard = assign_idx.gather(1, assign_w.argmax(dim=1, keepdim=True)).squeeze(1)  # [N]
```

### 11-3. Evaluation

**Neural 모델 vs Baseline 비교:**

```python
import sys, os, torch
sys.path.insert(0, os.path.expanduser("~/data/nuScenes/loader"))
from dataset import NuScenesNVSDataset
from config import NeuralClusteringConfig, ClusteringConfig
from nn.model import NeuralClusteringModel
from baseline.pipeline import cluster_frame
from data_loader import list_lidar_files

device = "cuda"

# --- Neural ---
cfg = NeuralClusteringConfig(device=device)
model = NeuralClusteringModel(cfg).to(device)
ckpt = torch.load("outputs/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dataset = NuScenesNVSDataset(dataroot=os.path.expanduser("~/data/nuScenes"), split='train')
sample = dataset[0]
pts = sample["input_0"]  # (N, 4): xyz + intensity
xyz = pts[:, :3].to(device)
intensity = pts[:, 3:4].to(device)
mask = torch.norm(xyz, dim=1) > cfg.ego_radius
xyz, intensity = xyz[mask], intensity[mask]

with torch.no_grad():
    output = model(xyz, intensity, tau=0.1)

g = output["gaussians"]
hard = output["assign_indices"].gather(
    1, output["assign_weights"].argmax(dim=1, keepdim=True)
).squeeze(1)
gamma = ((xyz - g["mu"][hard]) * g["n"][hard]).sum(dim=1)
neural_gamma_rms = gamma.pow(2).mean().sqrt().item()

print(f"Neural: K={g['mu'].shape[0]}, gamma_rms={neural_gamma_rms:.5f}")

# --- Baseline ---
files = list_lidar_files(os.path.expanduser("~/data/nuScenes"))
baseline_result = cluster_frame(files[0], ClusteringConfig())
baseline_gamma_rms = baseline_result["_gamma_rms"].median().item()
print(f"Baseline: K={baseline_result['num_gaussians']}, gamma_rms={baseline_gamma_rms:.5f}")
```

### 11-4. Visualization

**Baseline 3D viewer** (기존 viz_server.py 활용):

```bash
python -m gaustering.viz_server --frame-idx 0 --port 8890
# 브라우저에서 http://localhost:8890 접속
```

**Neural 결과를 baseline viewer 포맷으로 변환:**

```python
import sys, os, torch, json
sys.path.insert(0, os.path.expanduser("~/data/nuScenes/loader"))
from dataset import NuScenesNVSDataset
from config import NeuralClusteringConfig
from nn.model import NeuralClusteringModel
import numpy as np

device = "cuda"
cfg = NeuralClusteringConfig(device=device)
model = NeuralClusteringModel(cfg).to(device)
ckpt = torch.load("outputs/best_model.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dataset = NuScenesNVSDataset(dataroot=os.path.expanduser("~/data/nuScenes"), split='train')
sample = dataset[0]
pts = sample["input_0"]
xyz = pts[:, :3].to(device)
intensity = pts[:, 3:4].to(device)
mask = torch.norm(xyz, dim=1) > cfg.ego_radius
xyz, intensity = xyz[mask], intensity[mask]

with torch.no_grad():
    output = model(xyz, intensity, tau=0.1)

g = output["gaussians"]
K = g["mu"].shape[0]
hard = output["assign_indices"].gather(
    1, output["assign_weights"].argmax(dim=1, keepdim=True)
).squeeze(1)

# Viewer용 데이터 구성
rng = np.random.RandomState(42)
palette = rng.rand(K + 1, 3).tolist()
palette[-1] = [0.3, 0.3, 0.3]

viewer_data = {
    "points": xyz.cpu().numpy().tolist(),
    "color_idx": hard.cpu().numpy().tolist(),
    "palette": palette,
    "gaussians": {
        "centers": g["mu"].cpu().numpy().tolist(),
        "normals": g["n"].cpu().numpy().tolist(),
        "tangent_u": g["u"].cpu().numpy().tolist(),
        "tangent_v": g["v"].cpu().numpy().tolist(),
        "scaling": g["s"].cpu().numpy().tolist(),
        "gamma_rms": [0.0] * K,  # neural 모델에서는 별도 계산 필요
    },
    "stats": {
        "num_points": int(xyz.shape[0]),
        "num_gaussians": int(K),
        "coverage": 1.0,
        "gamma_rms_median": 0.0,
    },
}

with open("neural_result.json", "w") as f:
    json.dump(viewer_data, f)
print("Saved neural_result.json — use viz_server.py's HTML to view")
```

---

## 12. Baseline과의 비교

| 항목 | Baseline (method1) | Neural (method2) |
|------|-------------------|-----------------|
| **처리 방식** | 6단계 순차 규칙 | 단일 forward pass |
| **시드 결정** | 복셀 그리드 + FPS | 학습된 seediness score |
| **할당** | 비용 함수 argmin (hard) | Learned affinity + Gumbel-Softmax (soft) |
| **법선 추정** | KNN + PCA (비학습) | Backbone feature에 내재 |
| **Gaussian 피팅** | 닫힌 형태 PCA | PCA 초기화 + Residual 학습 |
| **학습 여부** | 하이퍼파라미터 수동 조정 | End-to-end 자기지도 학습 |
| **속도** | ~12초/프레임 | ~0.5초/프레임 (추론) |
| **메모리** | ~55 MB | ~700 MB (추론) |
| **Refinement** | Split/Merge 반복 | 불필요 (네트워크가 직접 학습) |
| **Ground 처리** | RANSAC 분리 후 별도 시딩 | 네트워크가 implicit 학습 |

---

## 부록 A: Gradient Flow 요약

```
L_surface
  → γ² = ((xyz - μ) · n)²
  → μ: MLP_μ ← cluster_feats ← (backbone features × assign_weights)
  → n: cross(u, v) ← quaternion_to_matrix(q) ← normalize(q_pca + MLP_q)
     → MLP_q ← cluster_feats ← backbone features × assign_weights
  → assign_weights: Gumbel-Softmax(logits)
     → logits: affinity_MLP(features) + geo_bias
        → affinity_MLP ← backbone features
        → geo_bias: detached (no grad)

L_CD
  → sampled = μ + α×u + β×v, α,β ~ N(0, s²)
  → μ, u, v, s ← Gaussian heads ← cluster_feats ← backbone

L_assign
  → assign_weights ← Gumbel-Softmax(affinity + geo_bias)
  → μ ← Gaussian head

L_scale
  → s ← Gaussian head ← cluster_feats ← backbone
```

모든 학습 가능 파라미터(backbone, seed MLP, affinity MLP, 4개 Gaussian heads)에 gradient가 도달한다.
단, PCA(SVD) 결과(q_pca, s_pca)와 geometric bias(σ², normals)는 detach되어 gradient가 차단된다.

## 부록 B: 구현 중 발견된 이슈와 해결

| 이슈 | 원인 | 해결 |
|------|------|------|
| Epoch 2부터 NaN loss | SVD backward의 수치 불안정성 (특이값 근접 시) | q_pca, s_pca를 detach — 네트워크는 residual만 학습 |
| In-place op autograd 에러 | `n_pca[flip] *= -1` 이 계산 그래프를 파괴 | `torch.where`로 out-of-place 연산으로 변경 |
| NMS 후 seed 수 과소 (K=17) | 전체 bbox가 이상치로 팽창 (765k m³) | 2-98 백분위수 기반 유효 부피 계산으로 변경 |
