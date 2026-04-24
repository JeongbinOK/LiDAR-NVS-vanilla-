# QGS-Flow: Feed-forward Quadratic Gaussian Surfels for LiDAR Novel View Synthesis

> **Core thesis**: LiDAR는 표면을 직접 관측하는 modality이므로, local 2차 곡면 fitting으로 QGS의 강력한 geometric prior를 closed-form으로 얻을 수 있다. 이 prior 위에 PTv3 feature로 residual만 학습하면, 2-frame sparse 감독으로도 feed-forward QGS for LiDAR NVS가 수렴한다.

---

## 1. Context

**Task.** 2프레임 LiDAR 키프레임 (t=0, t=1)과 precomputed bbox/tracking을 입력으로, 임의 시간 t ∈ [0,1]과 임의 LiDAR sensor pose에서의 point cloud를 합성. 학습 감독은 두 keyframe 사이 ~9 intermediate sweeps. 학습 시 sensor pose = GT sweep의 pose (control 불가), eval 시 임의 pose OOD 일반화까지 목표.

**Gap.**
- LiDAR NVS 최신 연구 (SplatAD'25, LiDAR-RT'25, GS-LiDAR'25, LiDAR4D'24, DyNFL) **전부 per-scene 최적화**
- Feed-forward Gaussian Splatting 연구 (pixelSplat, MVSplat, HiSplat) **모두 image 기반, LiDAR 없음**
- QGS (ICCV'25) **RGB/mesh 전용, LiDAR 적용 전무**
- 교집합 (feed-forward × LiDAR × QGS × 2-frame) = 빈 공간

---

## 2. Related Work Snapshot

| Method | Rep | Feed-fwd? | LiDAR? | Dynamic |
|---|---|---|---|---|
| LiDAR4D CVPR'24 | 4D hash+plane | ✗ | ✓ | 4D field |
| DyNFL | Per-obj NeRF | ✗ | ✓ | Rigid SE(3) |
| GS-LiDAR ICLR'25 | 2DGS panoramic | ✗ | ✓ | Vibration |
| SimULi | Sim LiDAR | ✗ | ✓ | (참조용 — LiDAR rendering convention) |
| SplatAD CVPR'25 | 3DGS+MLP | ✗ | ✓ (cam+LiDAR) | Rigid + per-instance learnable offsets |
| LiDAR-RT CVPR'25 | 2DGS+OptiX | ✗ | ✓ | Rigid |
| QGS ICCV'25 | Paraboloid | ✗ | ✗ | Static |
| pixelSplat/MVSplat | 3DGS (image) | ✓ | ✗ | Static |

**GS-LiDAR / SimULi 참고**: Panoramic spherical splatting + alpha-blended range + raydrop modeling. 우리 rasterizer 설계는 이들의 LiDAR rendering convention (spherical projection, intensity scaling, drop) 위에 QGS primitive를 끼워 넣는 방식. 차별점: QGS (곡면 primitive) + feed-forward + 명시적 canonical dynamic (vs vibration).

---

## 3. Representation Choice — QGS (근거 포함)

### 3.1 2DGS vs QGS

| Axis | 2DGS | QGS |
|---|---|---|
| Primitive | flat disk | paraboloid (곡면) |
| Params | c, R, (s₁,s₂), α | c, R, (s₁,s₂,s₃), α |
| Ray-primitive | ray-plane | ray-paraboloid (2차식) |
| Density | Euclidean on disk | **Geodesic** on quadric |

### 3.2 s₃ = 곡률 크기의 수식 근거

QGS implicit `sign(s₁)/s₁²·x̂² + sign(s₂)/s₂²·ŷ² − ẑ/s₃ = 0`을 ẑ로 풀면:
```
ẑ = s₃·(sign(s₁)/s₁²·x̂² + sign(s₂)/s₂²·ŷ²)
```
주곡률 `κ₁ = 2·s₃·sign(s₁)/s₁²`, `κ₂ = 2·s₃·sign(s₂)/s₂²`.
- **s₃ = 곡률 크기 scale**
- **sign(s₁)·sign(s₂) = signature** (++ bowl, +− saddle, +0 cylinder)
- **s₃ → 0 → flat 2DGS 특수해**

**Signed-scale semantics**: `sign(s₁)·sign(s₂)`가 surface signature를 담는다. 현재 구현은 초기 fitting에서 이 부호를 정하고 head는 `s1/s2` 부호를 보존한 채 magnitude만 보정한다.

### 3.3 왜 QGS인가 (LiDAR + 2-frame 특수성)

1. **Supervision density per primitive**: LiDAR는 surface-sparse sample. 곡면을 flat disk로 덮으면 N개 primitive에 감독 분산. QGS는 1 paraboloid로 커버 → 감독 밀도 ×N.
2. **Vehicle fit**: 차체는 전형적 2차 곡면. QGS는 natural fit.
3. **Marginal cost**: s₃ 한 파라미터 추가. Residual MLP에 output 2~3 neuron 추가일 뿐.

**결정**: **QGS primary**, 2DGS는 ablation baseline (s₃ = 0 고정). 근거: 위 3가지 + 사용자 요구 (novelty).

---

## 4. Architecture: QGS-Flow

### 4.1 Input Preparation

```
Raw inputs (from dataset + detector/tracker):
  P0 ∈ ℝ^{N0×4}, P1 ∈ ℝ^{N1×4}       (xyz + intensity, each sensor frame)
  T_{1→0} ∈ SE(3)                    (Frame1 sensor → Frame0 sensor)
  Bboxes_0, Bboxes_1                 (per-frame sensor coord, see CLAUDE.md)
  instance_ids (vehicle class only)

Pre-merge to Frame-0 sensor coord:
  P1' ← T_{1→0}(P1)
  Merged = P0 ∪ P1'                  (origin = Frame-0 LiDAR sensor)

Ego-motion vector (Q1):
  e_dir = T_{1→0}.translation ∈ ℝ³   # Frame1 sensor position in Frame0 coords
                                       = "vehicle has been here → there"
                                       broadcasts to all points (single 3-vector per scene)

Per-point feature channels (근거 명시):
  [ xyz                      // 3   — Frame-0 sensor coord
    intensity                // 1   — LiDAR reflectance
    time_scalar ∈ {0, 1}     // 1   — frame identity
    e_dir (broadcast)        // 3   — ego forward direction (Q1)
  ]                          // Total: 8D input features
```

**선택 근거**:
- **Scalar time (Q2)**: time_id가 {0,1} 2값뿐. Linear layer는 input concat 시 `Σ W_ij·x_i + W_t·time + b_j` 형태. time=0이면 bias=`b_j`, time=1이면 bias=`W_t+b_j` → frame별로 다른 affine. 후속 nonlinearity stack을 거치면 임의의 frame-conditional 변환 표현 가능. 즉 별도 embedding layer 없이 첫 linear가 자연스럽게 frame-bias로 분리한다는 뜻 (Occam's razor).
- **Ego-motion vector e_dir (Q1)**: 기존 per-point sensor_origin (3·N values) 대체. 이유:
  (a) 모든 점이 Frame-0 sensor coord에 있으므로 Frame-0 sensor origin = (0,0,0)는 trivially 알려짐 — 저장 불필요
  (b) 누락된 정보는 "Frame-1 sensor가 어디 있었는가" 즉 ego가 어느 방향으로 움직였는가
  (c) `e_dir = T_{1→0}.translation` 단일 3-vector로 충분 — broadcast해서 모든 점에 concat
  (d) ego motion 자체가 vehicle 진행방향에 대한 strong cue (도로 방향, surface visibility 비대칭 등)

### 4.2 Scene Decomposition (parameter-free)

```
Tracked-dynamic = vehicle class AND instance_id present in BOTH frames
Untracked / single-frame instances → 정적 객체로 편입 (Q3, see below)

dynamic_mask[i] = point_i ∈ any tracked-dynamic bbox (frame-matched)
static_mask    = ¬dynamic_mask  (포함: untracked instance points)

Static branch:
  P_static ← Merged[static_mask]   (Frame-0 sensor coord)

Dynamic branch, per tracked vehicle instance i:
  P0^i = P0  ∩ bbox_0^i
  P1^i = P1' ∩ bbox_1^i
  Canonical transform (per frame, per instance):
    Q0^i = R(−yaw_0^i)·(P0^i − c_0^i)
    Q1^i = R(−yaw_1^i)·(P1^i − c_1^i)
  P_canon^i = Q0^i ∪ Q1^i          (denser canonical supervision)
```

**Untracked instances → static (Q3, 결정 변경)**:
- 이전 plan: drop. 문제: GT sweep에 해당 점이 존재 → render 결과는 절대 GT와 일치 불가.
- 변경: untracked instance의 점은 **정적 객체로 간주**. 짧은 keyframe 간격 (~0.5s nuScenes)에서 단일 프레임만 보이는 객체는 거의 stationary이거나 motion이 작아 정적 근사가 reasonable.
- Phase A (시간 모델링 X)에서는 모든 untracked 점이 자연스럽게 정적 처리됨.
- Phase B (시간 모델링 O)에서는 untracked 점이 frame0/frame1 시점에서 자기 자리에 그대로 있는 것으로 가정 → temporal artifact 가능성은 있으나 drop보다 reconstruction error 작음.
- 통계 (untracked 비율, 평균 motion magnitude)는 logger로 기록 → reviewer report용.

### 4.3 Backbone — Time-aware PTv3

```
Static: F_static ← PTv3_bg(P_static, features)
Dynamic: F_canon^i ← PTv3_obj(P_canon^i, features)   # shared weights, per instance
```

**Backbone 선택 근거**:
- **PTv3 (default)**: Per-point feature, serialized attention, nuScenes SOTA (Point Transformer V3, CVPR'24). Dense per-point output이 우리 head에 필수.
- **대안 ablation**: spconv (빠르지만 voxel-level interpolation 필요), OctFormer (scale-aware지만 구현 복잡도↑). Ablation에서 비교.
- **Single vs two-branch**: single backbone on merged cloud (time scalar로 frame 구분). Two-branch cross-attention은 (a) params 2x, (b) merge 효과가 time scalar + xyz로 이미 달성, (c) feed-forward initial overfit 위험 감소. Ablation 후보지만 default 아님.

### 4.4 Geometric-init Residual Head (Main Anchor)

**Step-by-step**:

```
For each point p with backbone feature f:

  # Hybrid radius-k neighbor (Q4 재설명, see §7.3)
  # ⚠ 중요 (Q11): k-NN은 **동일 context** 안에서만 검색.
  #   - dynamic point p (instance i) → candidates = canonical_cloud[instance_id == i]
  #   - static point p               → candidates = static_cloud
  #   Cross-instance k-NN은 fit을 오염시키므로 금지.
  r_max(p) = clip(r_floor + α·||p||, r_floor, r_ceiling)
  N_radius = {q ∈ same-context : ‖q − p‖ ≤ r_max(p)}
  if |N_radius| < k_min:
      use_geom_init = False             # absolute fallback (k_min=8, Q12)
  else:
      nbrs = top-k_target nearest within N_radius
      use_geom_init = True

  if use_geom_init:
      # Local-tangent signed-QGS init
      # A = current/query point은 NBH 선택과 LS weight anchor로만 사용.
      1. p̄ = mean(nbrs)                         # valid k-NN의 unweighted mean
      2. C = cov(nbrs − p̄);  U, Λ = eig(C)
         R_pca = [u1, u2, u3] with u3 = least-variance normal
      3. PCA chart: q_j = R_pca^T · (nbrs_j − p̄)
      4. Weighted least squares in PCA chart:
         z = a·x² + b·y² + c·x·y + d·x + e·y + f
         w_j = exp(−‖nbrs_j − A‖² / (2σ²)),  σ = median_j ‖nbrs_j − A‖
      5. Initial center is not the paraboloid vertex:
         μ_init = p̄ + u3·f                     # surface point near p̄
      6. Fitted-gradient normal correction:
         n = normalize(−d·u1 − e·u2 + u3)
      7. Shape operator eigendecomposition at μ_init:
         H_pca = [[2a, c], [c, 2b]]          # PCA chart second derivative
         g = (d,e)
         I  = [[1+d², d·e], [d·e, 1+e²]]
         II = H_pca / sqrt(1+d²+e²)
         shape operator from (I, II) → principal curvatures/directions
         final principal directions e1,e2 are tangent to n
         R_init = [e1, e2, n]
         final local QGS Hessian is diag(κ1, κ2), not H_pca:
         z_local ≈ 0.5·(κ1·x_local² + κ2·y_local²)
      8. Tangent support sizes around μ_init:
         |s1_init| = γ·sqrt(mean(dot(nbrs_j−μ_init, e1)²))
         |s2_init| = γ·sqrt(mean(dot(nbrs_j−μ_init, e2)²))
         enforce |s1_init| ≥ |s2_init|
         sign(s1_init) = sign(κ1), sign(s2_init) = sign(κ2)
         near-zero curvature follows the dominant curvature sign; both near-zero → +.
      9. s3_init is positive curvature magnitude from LS:
         κ1 = 2·s3·sign(s1)/s1²,  κ2 = 2·s3·sign(s2)/s2²
         near-flat → s3_init = quadric_eps_s3
      10. fit_quality = [fit_residual, λ3/(λ1+λ2+λ3),
                         k_eff/k_target, avg_nbr_dist/r_max]   # 4D, see Q5

  # ── Geometry-first residual head ──────────────
  gate_input = [fit_quality, tangent_aniso, curvature_aniso]
  raw_geom = MLP_geom(concat([f, is_dynamic_flag]))
      → outputs: omega(3), delta_c(3), log_mean, log_gap, log_abs_s3
  g_rot, g_center, g_scale = sigmoid(MLP_gate(gate_input))
  if use_geom_init is False:
      g_rot = g_center = g_scale = 1       # fallback = absolute correction mode

  center = μ_init + g_center·tanh(delta_c)·head_center_bound
  R      = R_init · exp_so3(g_rot·tanh(omega)·rot_bound)
  |s1|, |s2| = ordered log_mean/log_gap residual update
  sign(s1), sign(s2) are preserved from signed init
  s3     = positive magnitude update via log_abs_s3
  alpha, intensity, latent are predicted by separate heads after geometry is fixed
```

**fit_quality 4D 정확한 정의 (Q1)**:

| 차원 | 정의 | 의미 (낮음 ↔ 높음) | 정규화 |
|---|---|---|---|
| `fit_residual` | PCA chart에서 weighted 2차 fit `z = ax² + by² + cxy + dx + ey + f` 후 `mean_w((z_pred − z_actual)²) / σ_z²` | low = 2차 곡면이 잘 맞음 / high = noise 또는 비-2차 | `σ_z² = weighted var(z_actual)` (scene scale-invariant) |
| `λ3/(λ1+λ2+λ3)` | PCA 공분산의 가장 작은 고유치 비율 (surface variation) | low = 매우 평평 / high = 두꺼운/노이즈 | 자동 [0, 1/3] |
| `k_eff/k_target` | radius 안에서 실제 선택된 이웃 수 / 목표 이웃 수 (예: 16) | low = sparse / high = 충분한 support | [0, 1] (cap) |
| `avg_nbr_dist/r_max` | 선택된 이웃들의 평균 거리 / radius 상한 | low = 조밀 / high = radius 가장자리 (외삽 위험) | [0, 1] (radius bound) |

4개를 곱하지 않고 raw로 넘겨야 MLP가 비선형 결합을 학습한다.

**왜 fit_quality만 (12D handcrafted descriptor 폐기, Q2)**:
이전 plan은 `local_stats = [λ1,λ2,λ3, planarity, linearity, κ1,κ2, H_mean, K_gauss, …]` 등 12~15D geometric descriptor를 head input에 concat했음. **재검토 결과 redundant + 위험 → fit_quality 4D만 남김.**

**두 종류의 uncertainty**:

| 구분 | 무엇 | 줄일 수 있는가 |
|---|---|---|
| Aleatoric | 데이터 자체 노이즈 (LiDAR sensor noise, beam divergence, intensity 변동) | 데이터 더 많아도 못 줄임 |
| Epistemic | 모델/추정의 지식 부족 (이 위치에서 fit이 신뢰 가능한가?) | 관측 늘어나면 줄어듦 |

fit_quality 4차원은 전부 **epistemic** 신호. "현재 이 위치에서의 geometric init이 얼마나 신뢰 가능한가"에 대한 측정.

**(b) Init = strong prior → handcrafted feature는 redundant**:
- **Prior**: parameter 초기값을 직접 제공 → optimizer는 그 근처에서 최적해를 찾음. 우리의 `c_init, R_init, s_init`이 이 역할.
- **Feature**: input으로 들어가 비선형 변환 후 출력에 영향. Prior가 없을 때 의미가 큼.

PointPillars처럼 prior 없는 모델은 handcrafted feature (local mean offset 등)가 prior 역할을 일부 대신. 우리는 prior가 이미 명시적으로 들어가니, 동일 정보(λ, κ)를 다시 input으로 넣으면:
1. MLP가 "init을 따를지" vs "raw stat으로 새로 예측할지" 두 경로 사이 ambiguous → 학습 신호 분산
2. Init이 0인 fallback case에서만 raw stat이 의미 있어야 함 → 그 외 경우엔 noise 추가

**(c) fit_quality = explicit gating signal (epistemic gate, N-1a flagship)**:

게이팅이란 한 입력 신호가 다른 신호의 영향력 크기를 조절하는 메커니즘 (GRU update gate, MoE router 등).

핵심 직관:
- Init 신뢰 가능 (fit_residual ↓, k_eff ↑, planarity ↓) → init 근처에서 미세조정만
- Init 신뢰 불가 (fit_residual ↑, k_eff ↓, reach ↑) → MLP가 전권으로 덮어써야 함
- Sub-floor (k_eff < k_min): init 아예 없음 → 처음부터 전권 MLP

기존 feed-forward GS들은 이 "init을 얼마나 믿을지"를 단일 MLP가 **암묵적으로** 학습해야 함 (input에 f만 들어감). 우리는 그 답을 analytic으로 이미 알고 있음 (`fit_quality`가 그 답). 그래서 현재 구현은 `fit_quality + tangent/curvature anisotropy`로 residual 크기를 group-wise gate한다:

```python
# Geometry head
raw_geom = MLP_geom(concat([f, is_dynamic_flag]))
gate_input = concat([fit_quality, tangent_aniso, curvature_aniso])
g_rot, g_center, g_scale = σ(MLP_gate(gate_input))

# Bounded residual modulation
center = μ_init + g_center * tanh(raw_delta_c) * head_center_bound
R      = R_init · exp_so3(g_rot * tanh(raw_omega) * rot_bound)
scale  = signed/ordered log_mean-log_gap + positive log_abs_s3 update
```

구조적 특성:
- **Group gate**: `g_rot`, `g_center`, `g_scale` 세 scalar가 각각 rotation / center / scale residual freedom을 조절한다.
- **Gate bias init = −2.0**: 학습 시작 시 `gate ≈ σ(−2) ≈ 0.12`. Default는 "init 근처에서 작게 보정".
- **Sub-floor 통합 (§7.6)**: `k_eff < k_min`이면 세 gate를 1로 강제 + `fit_quality = [r_max_clip, 1/3, 0, 1]` (최악값) 주입 → absolute correction mode.
- **Bounded residual**: center, rotation, scale 모두 explicit bound 안에서만 움직여 좋은 local init이 뒤집히지 않게 한다.

효과 (왜 novelty):
- Feed-forward GS는 sparse supervision에서 수렴이 느림 → 우리는 "init 신뢰도 판단"을 analytic으로 넘겨서 MLP 학습 부담 1/2. 2-frame에서 수렴 가속이 paper main result.
- Gate 값은 학습 후 interpretability 제공 (E-6 ablation: gate ↑ with synthetic noise ↑).

**유사 사례**:
- Vision aleatoric/epistemic loss (Kendall & Gal NeurIPS'17): 모델이 자기 prediction의 σ를 함께 출력 → loss에서 weight 역할
- PCPNet (CGF'18, normal estimation): PCA-derived feature를 MLP에 직접 input, sparse supervision에서 효과적
- Calibration networks: 다른 head의 logit을 받아 confidence 보정
- 우리의 fit_quality는 explicit한 epistemic signal이라 head가 implicit calibration을 학습할 필요 없음 → 학습이 빠름
- Hybrid analytic-learned 일반 원칙: 분석적으로 풀 수 있는 부분은 분석적으로, 학습은 residual만. 입력에 분석 결과를 그대로 다시 넣지 않는다.

**결정**: handcrafted geometry 12D → fit_quality 4D로 축소. Ablation으로 (a) fit_quality 0개, (b) 4개, (c) 12개 확장 비교.

**Shared head across static/dynamic**: 하나의 MLP가 모든 점 처리. 근거: (1) 곡률/크기는 object-class-agnostic, (2) dynamic data 희소성, (3) backbone 입력에서 이미 static/dynamic 분리됨, (4) 1-bit `is_dynamic_flag` concat하여 self-modulation. Separate head는 ablation 후보.

### 4.5 Rendering @ time t, sensor pose T_sensor

**Phase A (시간 모델링 X — 우선 검증)**:
```
For each tracked dynamic instance i, render at frame ∈ {0, 1}:
  if frame == 0:
      T_obj = (R(yaw_0^i), c_0^i)
  else:
      T_obj = (R(yaw_1^i), c_1^i)
  QGS_world^i ← T_obj · QGS_canon^i

QGS_all = QGS_static ∪ ⋃_i QGS_world^i
Range, Intensity, Drop ← LiDARQGSRasterizer(QGS_all, T_sensor=frame_pose, H, W)
```
→ Loss against P0 / P1 (자기 자신). 시간축 없음. Feed-forward scene representation 검증.

**Phase B (시간 모델링 — Phase A 검증 후)**:
```
For each tracked dynamic instance i with (c_0, yaw_0, c_1, yaw_1):
  # Velocity (position-only Hermite, Q6)
  (v̂_0, v̂_1) = VelocityEstimateHead(bbox_0, bbox_1, ...)   # §6.4, ω̂ 제거

  # Pitch (road slope, Q4)
  Δp_world = c_1 − c_0
  d_xy     = sqrt(Δp_world.x² + Δp_world.y²)
  pitch_est = 0 if d_xy < ε_pitch else atan2(Δp_world.z, d_xy)   # ε_pitch = 1.0m

  # Position: Cubic Hermite (see §6.3)
  c_t = h00·c_0 + h10·v̂_0 + h01·c_1 + h11·v̂_1

  # Yaw: shortest-path slerp (Q6)
  Δyaw  = atan2(sin(yaw_1 − yaw_0), cos(yaw_1 − yaw_0))   # ∈ [−π, π]
  yaw_t = atan2(sin(yaw_0 + t·Δyaw), cos(yaw_0 + t·Δyaw))

  # Canonical → world transform: apply yaw first (Z), then pitch (Y)
  QGS_world^i ← R_z(yaw_t) · R_y(pitch_est) · QGS_canon^i + c_t

QGS_all = QGS_static ∪ ⋃_i QGS_world^i
Range, Intensity, Drop ← LiDARQGSRasterizer(QGS_all, T_sensor, H, W)
```

---

## 5. LiDAR-QGS Rasterizer

### 5.1 QGS 원 수식

- Local implicit: `α_q·x̂² + β_q·ŷ² − γ_q·ẑ = 0` with α_q = sign(s₁)/s₁², β_q = sign(s₂)/s₂², γ_q = 1/s₃
- Normal: `n̂_raw = (2α_q·x̂, 2β_q·ŷ, −γ_q)`, normalize
- Geodesic density: `G(p̂₀) = exp(−l(a(θ),ρ)² / (2σ(θ)²))` — closed-form arc length

### 5.2 Ray-paraboloid Intersection (exact)

Ray `p(t) = o + t·d` to QGS local frame `(R_q, c_q)`:
```
ô = R_q^T·(o − c_q)
d̂ = R_q^T·d
```

Substitute into implicit → 2차식 `A·t² + B·t + C = 0`:
```
A = α_q·d̂_x² + β_q·d̂_y²
B = 2·(α_q·ô_x·d̂_x + β_q·ô_y·d̂_y) − γ_q·d̂_z
C = α_q·ô_x² + β_q·ô_y² − γ_q·ô_z

t_± = (−B ± √(B² − 4·A·C)) / (2·A)
```

**해 의미**:
- 두 실근 → ray가 곡면을 2회 crossing
- 판별식 < 0 → 빗나감 → contribution = 0
- Valid hit 조건: 판별식 > 0 AND `G(p̂(t_±)) > thr_density` (geodesic 3σ)

**Saddle case corner** (Q17): 판별식 > 0인 두 해가 각각 다른 saddle arm에 있을 수 있음 → 각각 독립적으로 density 평가 후 alpha blending에 둘 다 contribute.

### 5.3 Alpha-blended Range (alpha blending 유지)

**원칙 (Q5, Q6)**: LiDAR GT는 단일 거리지만 표면은 여러 QGS 중첩 → expected range는 alpha blending으로. GS-LiDAR/SplatAD/LiDAR-RT와 일치. QGS의 geodesic density G는 이 blending의 smooth weight 제공 (Q6: alpha blending 없이는 QGS 사용 의미 퇴색).

```
교차점 t 순 정렬 (front → back)
contribution_i = α_i · G_i(p̂_i^{hit})
T_i = Π_{j<i}(1 − contribution_j)
w_i = T_i · contribution_i                    # i번째 QGS 기여 weight

r_rendered  = Σ_i w_i · t_i
i_rendered  = Σ_i w_i · intensity_i
n_rendered  = Σ_i w_i · n̂_i
κ_rendered  = Σ_i w_i · |mean_curvature_i|
feat_agg    = Σ_i w_i · latent_i              # for raydrop MLP
α_accum     = Σ_i w_i                         # total opacity mass
```

**α_accum 용도 (Q6)**:
1. `p_hit = α_accum` → raydrop geometric factor
2. `r_normalized = r_rendered / (α_accum + ε)` — low-mass ray의 range bias 보정
3. Valid mask: `α_accum > τ_valid`인 ray에만 L_range 적용

### 5.4 Tile Culling via Spherical AABB (SplatAD Jacobian 미사용)

**왜 SplatAD Eq.10 미사용**: Σ_s = J·Σ·J^T는 3D Gaussian 공분산의 1차 Jacobian 근사. QGS는 (a) 2D manifold + geodesic density → Euclidean Σ 없음, (b) saddle/bowl 변화에서 linearization 오차, (c) 곡률↑ → 오차↑.

**대안**: per-primitive spherical AABB (Q7):
```
1. QGS의 representative extreme points sampling:
   - Base ellipse 경계 8 points (θ ∈ {0, π/4, π/2, ...})
   - Paraboloid 정점 along z-axis (3 points: z_min, 0, z_max at 3σ geodesic)
   → 11 world-space points per QGS

2. World → sensor frame → spherical:
   (φ, ω, r) = [atan2(y,x), arcsin(z/√(x²+y²+z²)), √(...)]

3. AABB: (φ_min, φ_max, ω_min, ω_max)

4. Corner cases (Q7):
   - Azimuth wraparound (crosses ±π): split into 2 AABBs
   - Pole singularity (|ω| > 1.4 rad): assert/log warning
     (nuScenes automotive LiDAR: typically |ω| < 30° → 문제 없음)
   - Behind sensor (r < 0): skip
```

**근거**: AABB는 culling-only, 렌더링 수식과 무관 → 과보수성(overestimate) 허용. GS-LiDAR 및 SplatAD도 유사한 tile-based culling.

**Engineering ablation note (Q5)**: SplatAD는 32×8 (azim×elev) tile + sorted-elev binning으로 GPU thread block 효율을 극대화함. 우리는 우선 spherical AABB 먼저 구현 (correctness 확보), 성능 부족 시 32×8 tile 도입을 Phase A2 ablation으로. Rolling-shutter time correction (SplatAD `r_exp = Σ(rᵢ + v_r^S·t_l)·αᵢ·Tᵢ`)은 Phase B에서 dynamic motion이 큰 경우에만 고려.

### 5.5 Raydrop with Curvature Awareness (별도 Drop Map, Q3)

**3개의 분리된 output map**:
1. `range_map (H, W)` — alpha-blended depth
2. `intensity_map (H, W)` — alpha-blended intensity
3. `drop_map (H, W)` — ray-drop probability (**별도 map**)

조사 결과 **모든 LiDAR NVS 논문 (SplatAD, SimULi, GS-LiDAR, LiDAR-RT, LiDAR4D)이 drop을 별도 map으로 만든다**. 우리는 SplatAD 패턴 (per-Gauss feat → MLP) + curvature 추가 (S-1 novelty).

**분해 (Q3, Q8)**:
```
α_accum(u,v) = Σ_i T_i · α_i · G_i              # geometric coverage [0,1]
p_hit        = α_accum                          # 기하학적 hit 확률 (analytic)
p_drop_phys  = σ(MLP_drop(feat_agg, κ_render, cosθ_inc, log(1+r), ray_dir))
p_drop(u,v)  = (1 − p_hit) + p_hit · p_drop_phys
```
- `1 − p_hit = 1 − α_accum`: geometric miss — analytical
- `p_drop_phys`: hit 조건부 물리 drop (material, angle, curvature) — MLP 학습

```
feat_input = [ feat_agg,              # (d-dim latent, alpha-blended)
               cos(θ_inc_rendered),   # incidence angle (see §5.6)
               log(1 + κ_rendered),   # curvature-aware (novel)
               log(1 + r_rendered),   # range factor
               ray_direction ]         # sensor-local
p_drop_physical = σ(MLP_drop(feat_input))
```

**왜 p_hit을 MLP input에서 제외 (Q8)**:
- 분해 설계: geometric drop은 이미 explicit — MLP는 물리 factor만
- Gradient 분리: p_hit → α, G 학습, MLP → intensity/curvature/drop 학습
- Interpretability: "왜 drop인가" 분해 가능 (geom vs physical)
- Ablation으로 p_hit concat 버전 비교 계획

**`feat_agg` 정의 (Q8)**: 각 QGS primitive가 head에서 latent vector `latent_i ∈ ℝ^d_latent`를 출력 → rendering 시 alpha-blended 집계 `feat_agg = Σ_i w_i · latent_i`. Range/intensity/normal과 동일 가중치 → 일관성.

**곡률 반영 근거**: 기존 SplatAD/LiDAR-GS는 curvature를 drop logit에 넣지 않음. 물리적으로 고곡률 영역은 beam footprint 내 normal 급변 → incoherent return → drop 확률↑. **Novel supporting contribution**.

### 5.6 Incidence Angle per Ray

```
d_local = R_q^T · d                      # ray in QGS local frame
n̂ = normalize((2α_q·x̂_hit, 2β_q·ŷ_hit, −γ_q))
cos(θ_inc)_i = max(0, −<d_local, n̂>)     # [0,1] range

# alpha-blended across QGS along ray:
cos(θ_inc)_rendered = Σ_i w_i · cos(θ_inc)_i
```

---

## 6. Dynamic Object Pipeline (vehicle-only, Position-Hermite + Yaw-slerp — Phase B)

**Frame gap = 2** (nuScenes 0.5s × 2 = **Δt_real = 1.0s**). Intermediate sweeps도 이 1초 간격 안에 포함된 sweep들 사용. Loader `mode='bbox'`가 실제 몇 개의 intermediate sweep을 반환하는지는 구현 후 첫 batch 로그로 확인.

### 6.1 v1 Scope

**Vehicle class only** (car, truck, bus, trailer, construction_vehicle).
- 보행자/자전거/기타 (non-rigid): static branch에 편입.
- Untracked vehicle (한 프레임에만 존재): static branch에 편입 (Q3).
- "rigid SE(2)" 정정 (Q9): 차량은 도로면을 따라 거의 평면 운동 (roll/pitch ≈ 0) → translation 2D + yaw 1D = SE(2) motion model로 충분히 근사 가능. "ground truth"는 부정확한 표현이었고 "appropriate motion model"이 정확. 실제 trajectory는 SE(2) 위의 경로.

### 6.2 Shape-Motion Decomposition

- **Shape**: canonical bbox-local QGS (time-independent)
- **Motion**: position via **Cubic Hermite Spline**, yaw via **shortest-path slerp** (Q6)

Two-frame 조건에서 4D deformation field는 under-determined → shape-motion 분리로 network는 "canonical static shape"만 학습 (쉬운 subtask).

### 6.3 Position-Hermite + Yaw-Slerp 구체식 (Q6 확정)

**사용자 결정 (Q6)**: Position은 Hermite 유지 (v̂_0, v̂_1 추정 필요), Yaw만 옵션 B (shortest-path slerp). **ω (angular velocity)는 제거**.

**Notation**: hat 기호는 "Hermite parameter t ∈ [0,1] domain으로 scale된 양".
- `v` (m/s, real velocity) → `v̂ = Δt_real · v` (m per unit-t)

```
Δt_real = 1.0s   # frame gap = 2 × 0.5s nuScenes keyframe
v̂_0 = Δt_real · v_0 ;  v̂_1 = Δt_real · v_1

# ── Position: Cubic Hermite ────────────────────────────
h00(t) = 2t³−3t²+1;  h10(t) = t³−2t²+t
h01(t) = −2t³+3t²;   h11(t) = t³−t²
c(t) = h00·c_0 + h10·v̂_0 + h01·c_1 + h11·v̂_1

# ── Yaw: Shortest-path slerp (옵션 B, ω 없음) ──────────
Δyaw  = atan2(sin(yaw_1 − yaw_0), cos(yaw_1 − yaw_0))   # ∈ [−π, π]
yaw(t) = atan2(sin(yaw_0 + t·Δyaw), cos(yaw_0 + t·Δyaw))

# ── Pitch (road slope, Q4) ─────────────────────────────
Δp_world = c_1 − c_0;  d_xy = sqrt(Δp_world.x² + Δp_world.y²)
pitch_est = 0 if d_xy < ε_pitch else atan2(Δp_world.z, d_xy)   # ε_pitch = 1.0m

# ── Canonical → world transform @ time t ───────────────
QGS_world^i(t) = R_z(yaw(t)) · R_y(pitch_est) · QGS_canon^i + c(t)
```

**왜 position만 Hermite, yaw는 slerp**:
- Position: 가속/감속 profile을 잘 표현하려면 v_0, v_1이 필요 → 하지만 tracker가 주지 않으므로 **Velocity Head로 추정** (§6.4). Hermite는 이 추정된 v를 반영해 매끄러운 trajectory 생성.
- Yaw: 자동차 회전은 가속/감속이 position보다 작고 (도로 곡률 느림), ω_0, ω_1 추정은 신호가 약해 noise-dominated. Shortest-path slerp으로 충분.

**Pitch 추정 (Q4)**: `atan2(Δz, d_xy)` 단순 공식. d_xy < 1m (정지/극저속 차량)은 LiDAR z-noise (~10cm) 대비 `atan(0.1/2) ≈ 3°`로 이미 noise-dominated → pitch = 0으로 처리. **Full SE(3) Rodrigues는 overkill** (roll/yaw는 이미 bbox에서 주어짐, pitch 하나만 필요).

### 6.4 Velocity Estimation Head (Q6, Q7, Q8 — v̂만 출력, ω̂ 제거)

**문제 제기**: tracker가 keyframe별 (c, yaw, w, l, h)는 주지만 velocity는 (a) 제공 안 되거나 (b) 노이즈가 큼. Position Hermite에는 v_0, v_1이 필요.

**Yaw는 slerp이라 ω 불필요 (Q6, Q8)**: ω는 Hermite 기반 yaw 보간 때문에 필요했던 값. Yaw가 slerp으로 바뀌었으므로 ω̂_0, ω̂_1은 **제거**.

**해결 — 평균속력 baseline + MLP residual (v̂ only)**:
```
구현 전 확인: tracker bbox 좌표가 어느 frame인지 (global / ego-pose / Frame-0 sensor)
→ 모두 Frame-0 sensor coord로 통일 후 아래 계산. c_0, c_1, yaw_0, yaw_1은 Frame-0 sensor 기준.

# Pitch (Q4)
Δp_world = c_1 − c_0
d_xy     = sqrt(Δp_world.x² + Δp_world.y²)
pitch_0  = 0 if d_xy < 1.0 else atan2(Δp_world.z, d_xy)

# 평균 속력 (방향 무관, magnitude만)
S_avg = ‖Δp_world‖₂ / Δt_real          # [m/s], Δt_real = 1.0s

# 자차 기준 displacement (Q7: yaw뿐 아니라 pitch도 적용)
R_world_to_local = R_y(−pitch_0) · R_z(−yaw_0)
Δp_local = R_world_to_local · Δp_world  # z 성분은 거의 0 (차량 진행면 위)

# Yaw delta embedding (cyclic-safe)
Δyaw       = atan2(sin(yaw_1 − yaw_0), cos(yaw_1 − yaw_0))
Δyaw_embed = (cos(Δyaw), sin(Δyaw))

# Bbox dims + pitch
size      = (w, l, h)
f_ctx     = [mean(F_canon^i frame=0), mean(F_canon^i frame=1)]   # optional

x_in = concat([S_avg, Δp_local, Δyaw_embed, size, pitch_0, f_ctx?])

# Output: v̂_0, v̂_1 only (ω̂ 제거)
(v̂_0_pred, v̂_1_pred) = VelocityHead(x_in)   # 6D output
```

**Pitch-aware Δp_local 효과 (Q7)**: `Δp_local`의 z 성분이 noise scale로 줄어들고, 진행 방향(x)과 lateral(y)만 의미 있는 신호로 남음 → MLP 학습이 쉬워짐.

**Magnitude grounding**: head는 `Δp_local`을 baseline으로 받음 → 출력은 "시작 속도와 끝 속도로 분배할지"의 weighting. 등속이면 v̂_0 = v̂_1 = Δp_world. MLP가 가속/감속 case에 residual 추가.

**Loss-driven training**: GT intermediate sweep과 비교하는 L_range가 v̂_0, v̂_1을 통해 trajectory shape에 gradient를 흘려보냄 → end-to-end 학습.

**Fallback**: tracker가 velocity를 직접 제공하면 → MLP 우회 + linear fallback (`v̂ = Δp_world`)을 두 모드로 ablation.

### 6.5 Tracker Noise — Pose-refine Head (feed-forward compliant) (Q9, Q10)

**Per-instance learnable residual 폐기** (feed-forward 원칙 위배). **Shared MLP로 대체, keyframe(t=0, t=1)에만 적용**.

**Shared Pose-refine Head (velocity 제거, pitch 반영, keyframe only)**:
```
# SplatAD 참조: per-scene trained per-instance (v_act, ω_act + Δpose) → 우리는 feed-forward이므로 shared MLP context-driven

input = [
  mean(F_canon^i in bbox_0),         # backbone feature 평균 (frame 0)
  mean(F_canon^i in bbox_1),         # backbone feature 평균 (frame 1)
  bbox_0 (7D: c, w, l, h, yaw),
  bbox_1 (7D),
  Δp_local (with pitch, Q4+Q7),      # 두 frame 사이 vehicle-local displacement
  Δyaw_embed (cos(Δyaw), sin(Δyaw)),
  pitch_est,                         # 도로 경사각
]
# velocity는 입력에서 제거 — 우리가 갖고 있지 않음 (Q9)

(Δc_0, Δyaw_0, Δc_1, Δyaw_1) = tanh(MLP_pose_refine(input)) · clamp_radius

# 보정된 keyframe pose (t=0, t=1에만 적용)
c_0'   = c_0 + Δc_0
yaw_0' = yaw_0 + Δyaw_0
c_1'   = c_1 + Δc_1
yaw_1' = yaw_1 + Δyaw_1

# 중간 시점 t의 pose는 보정된 두 끝점으로 Hermite + slerp 계산
c(t)   = Hermite(c_0', v̂_0, c_1', v̂_1; t)
yaw(t) = slerp(yaw_0', yaw_1'; t)
```

**왜 중간 t에 보정 안 하나 (사용자 결정)**:
- p_0, p_1, v_0, v_1이 정확히 추정되면 Hermite 궤적이 충분히 정확.
- 중간 시점마다 별도 보정하면 rigid motion 가정 깨짐 (trajectory discontinuity).
- 끝점 2개 보정이 full trajectory를 올바른 방향으로 shift하는 것으로 충분.

**clamp_radius**: Δc ±0.3m, Δyaw ±0.05rad (≈2.9°). TransFusion-L + MCTrack bbox는 대체로 정확하므로 큰 보정은 false positive.

**학습 신호 — rendering loss only (Q10)**:
- **Shape Consistency Loss 삭제** (Q10 사용자 지적 수용).
- **이유**: 두 프레임에서 같은 객체를 봐도 **관측된 부분(visible surface)이 전혀 다름** (예: frame 0은 뒷범퍼 30점, frame 1은 왼쪽 문짝 40점). 같은 canonical frame에서 Chamfer로 비교하면:
  - pose가 완벽해도 Chamfer ≠ 0 (부위가 다름)
  - Chamfer 값은 "pose 오차"와 "관측 부위 차이"가 섞여 있어 **pose 오차 신호로 쓰기 힘듦**
  - pose-refine head가 "두 cloud가 겹치게" 잘못된 방향으로 학습할 위험 (false signal)
- Pose refinement은 **L_range rendering loss만으로 학습** (SplatAD가 실제 채택한 메커니즘):
  - 잘못된 pose → 잘못된 위치에 렌더링 → L_range↑ → backprop으로 (Δc, Δyaw) 보정.

**안정화 보조**: 초기 학습 중 shape/pose 신호 분리를 위해 L2 regularization:
```
L_pose_reg = ‖Δc_0‖² + ‖Δc_1‖² + ‖Δyaw_0‖² + ‖Δyaw_1‖²     # weight ~1e-4
```
Pose가 tracker 값 근처를 유지하도록 soft prior.

### 6.6 Untracked Instances (Q3 — 정정)

**결정: 정적 객체로 편입** (이전 plan의 "drop"은 폐기).
- 근거: drop은 GT sweep에 존재하는 점을 우리는 절대 못 만드는 mismatch → 학습 loss와 eval metric 둘 다 해 없는 영역.
- 짧은 keyframe 간격 (~0.5s)에서 한 프레임에만 보이는 객체는 occlusion / 진입·이탈 경계 / tracker miss → 대부분 motion 작음. 정적 근사가 reconstruction에 더 유리.
- 통계 (untracked instance 비율, untracked 점이 차지하는 ratio, motion mean) → ablation table용.
- v2: existence mask + temporal interpolation 고민 가능 (그러나 v1 scope 밖).

---

## 7. Main Anchor: Geometric-Init Residual Head (상세)

이 섹션이 **"왜 우리만 feed-forward QGS가 작동하는가"** 원인 규명.

### 7.1 핵심 아이디어

- QGS는 파라미터 많고 (c, R, s₁, s₂, s₃, α, intensity, latent) 곡률은 local에 민감
- Random init → 희박 감독으로 수렴 어려움
- 하지만 **LiDAR 포인트는 표면 직접 관측** → 각 point의 k-NN에서 local 2차 곡면이 **closed-form으로 추출 가능**
- 이미지 feed-forward (pixelSplat/MVSplat)는 epipolar/cost-volume depth라는 **간접** 경로만 가능 — 우리는 **직접 geometric prior**

### 7.2 PCA가 곡면에서도 유효한 이유 (Q2 정당화)

곡면이어도 **local k-NN patch**에선 tangent plane 근사가 유효 (patch size << curvature radius). Moving Least Squares (MLS) literature의 표준 전제. PCA는 tangent plane을 주고, 그 위에 2차 회귀가 곡률을 준다 — 2-step process.

### 7.3 Hybrid radius-k Neighbor (Q4, Q11, Q12 재설명)

"Hybrid"는 **세 가지 안전장치를 합친 neighbor selection 전략**:

**⚠ Context separation (Q11)**: k-NN은 **동일 instance 내 / static cloud 내**에서만 검색 (cross-instance 금지).
- dynamic point (instance i) → candidates = canonical_cloud[instance_id == i]
- static point → candidates = static_cloud
- 이유: 두 차량이 가까이 있으면 다른 차의 점이 k-NN에 포함 → fit 오염. Static-dynamic mix도 동일 문제.
- 구현: instance별 batch 호출로 `torch_cluster.knn` 또는 `pytorch3d.ops.knn_points` 독립 실행.

1. **Radius bound (upper)**: 거리 `r_max(p)` 이내 점만 후보.
   - `r_max(p) = clip(r_floor + α·‖p‖, r_floor, r_ceiling)`
   - 예: `r_floor = 0.3m`, `α = 0.01`, `r_ceiling = 2.0m`.
   - 원거리 (sparse) 점일수록 radius 약간 늘리되 상한으로 cap → scene 끝 garbage 점 배제.

2. **k cap (upper)**: dense 영역에서 `k_target = 16`개만 nearest로.
   - 2차 fit 파라미터 6 + noise margin → k ≥ 12 권장
   - 그 이상은 over-smoothed → local 곡면 정보 손실
   - Ablation: k_target ∈ {12, 16, 24, 32}

3. **k floor (fallback, Q12a)**: `k_eff < k_min = 8` 이면 geometric init **비활성** (fallback = absolute prediction).
   - head가 `[f, zeros, is_dynamic_flag]` 받고 absolute QGS params 예측.
   - **k_min=8 근거**: 2차 fit unknown 6개 → exact fit(k=6)은 noise 그대로 외움, `k ≥ 1.5·d = 9` 권장. **8은 약간 보수적 (메모리/지연 절약)**. Ablation: k_min ∈ {6, 7, 8, 10, 12}.

**Pseudocode**:
```
candidates = canonical_cloud[instance_id == i] if dynamic else static_cloud
N_radius = {q ∈ candidates : ‖q − p‖ ≤ r_max(p)}
if |N_radius| < k_min:        # case (3)
    use_geom_init = False      # absolute fallback
else:
    nbrs = top-k_target nearest within N_radius   # case (1) ∩ case (2)
    use_geom_init = True
```

**Sub-floor 점 처리 (Q12b)**: "Fallback 대신 가우시안 생성 자체를 skip"은 거부. 이유:
- Hole in rendering: sparse but valid 영역(원거리 차량 등) 전체 누락 → L_range 무한 증가
- Loss masking 일관성 붕괴: GT에 있는 ray를 우리는 못 만들 때 drop 신호와 충돌
- Backbone gradient 손실: PTv3가 그 점의 feature를 학습 불가
- **Default = fallback (absolute prediction)**. Ablation으로 (a) drop, (b) fallback 비교 수치화.

### 7.4 2차 fit & Hessian 고유치 분해 (Q3 확인)

현재 구현은 **A를 이차곡면 중심으로 쓰지 않고**, A 주변 k-NN을 선택한 뒤 이웃 평균 `p̄` 근처의 surface point를 QGS center로 잡는다. 즉 처음 LS에서 맞추는 일반식은 선형항을 포함하지만, 최종 QGS local frame은 `μ_init`에서 다시 잡은 tangent frame이라 선형항 없는 canonical form이 된다.

**Local-tangent signed-QGS init**:
```
A = query point / neighbor selection anchor
p̄ = mean(nbrs)
C = cov(nbrs − p̄)
R_pca = [u1, u2, u3],  u3 = least-variance normal

q_j = R_pca^T · (nbrs_j − p̄)
z = a·x² + b·y² + c·x·y + d·x + e·y + f
w_j = exp(−‖nbrs_j − A‖² / (2σ²)),  σ = median_j ‖nbrs_j − A‖

μ_init = p̄ + u3·f
n = normalize(−d·u1 − e·u2 + u3)

H_pca = [[2a, c], [c, 2b]]              # 초기 PCA chart의 second derivative
g = (d,e)
I  = [[1+d², d·e], [d·e, 1+e²]]         # first fundamental form
II = H_pca / sqrt(1+d²+e²)              # second fundamental form
shape operator from (I, II) → (κ1, κ2), (e1, e2)
R_init = [e1, e2, n]
```

최종 QGS local chart:
```
z_local ≈ 0.5·(κ1·x_local² + κ2·y_local²)
```
따라서 최종 초기 QGS에는 `dx, dy` 선형항과 `cxy` 교차항이 없다. `H_pca=[[2a,c],[c,2b]]`는 버리지 않고 `d,e`와 함께 `(I,II)`를 만들어 최종 tangent plane의 shape operator로 변환된다. 최종 frame의 Hessian은 `H_pca`가 아니라 `diag(κ1,κ2)`이다.

각 기준점 A마다 `fit_local_quadrics(...)`가 내는 geometric-init 출력:
- `c_init = μ_init`: `p̄` 근처 fitted surface point. fallback이면 A.
- `R_init = [e1,e2,n]`: corrected normal과 principal tangent axes.
- `s_init = [s1,s2,s3]`: `s1/s2`는 signed tangent support, `s3`는 positive curvature magnitude.
- `fit_quality = [fit_residual, planarity, support, reach]`.
- `use_geom_init`: `k_eff >= k_min` 여부.
- diagnostics: `kappa1_init`, `kappa2_init`, `tangent_aniso`, `curvature_aniso`.

`s1/s2/s3` 계산:
```
|s1| = γ·sqrt(mean(dot(nbrs_j−μ_init, e1)²))
|s2| = γ·sqrt(mean(dot(nbrs_j−μ_init, e2)²))
|s1| ≥ |s2|
sign(s1), sign(s2) follow κ1,κ2
s3 = LS positive solution of:
  κ1 = 2·s3·sign(s1)/|s1|²
  κ2 = 2·s3·sign(s2)/|s2|²
near-flat → s3 = quadric_eps_s3
```

### 7.5 Ordering Ambiguity Non-issue (재확인)

- s₁, s₂: tangent extent (length unit) vs s₃: curvature coefficient (different semantic) → 순서 비교 무의미
- s₁ ↔ s₂ ambiguity: `|s1| ≥ |s2|` ordering과 `R_init=[e1,e2,n]` principal frame으로 고정
- 현재 head는 `s1/s2` sign을 보존하고 magnitude만 log_mean/log_gap residual로 보정. 부호 flip은 초기 fitting의 curvature signature가 담당한다.

### 7.6 Sub-floor fallback as gate=1 injection (Plan v3 §3-2 통합)

Plan v2의 fallback (`if use_geom_init / else absolute prediction`)을 §4.4의 gated-residual 안으로 흡수. 별도 분기가 아닌 **same equation의 limit case**로 표현:

```
if k_eff < k_min:
    fit_quality = [r_max_clip, 1/3, 0, 1]   # 최악값 주입 (planarity·support·reach 모두 worst)
    g_rot = g_center = g_scale = 1           # force full residual freedom
    c_init      = p                          # geometric init은 raw point 위치
    R_init      = I, s_init = (0, 0, 0)       # neutral init
# 이 상태로 §4.4의 bounded residual 식을 그대로 사용:
center = p + tanh(delta_c)·head_center_bound
R      = exp_so3(tanh(omega)·rot_bound)
|s1|, |s2|, s3 = fallback base + residual update
```

**왜 이게 깔끔한가**:
- Codebase에 `if-else` 분기 두 갈래 없음 — gate가 자동으로 분기 역할 수행
- `r_max_clip`은 학습 중 `fit_residual`의 running-max로 동적 갱신 (Plan v3 §8 R2)
- Sub-floor 비율이 데이터셋에 따라 달라져도 architecture 자체는 변경 없음
- E-5 ablation (no fallback injection, sub-floor도 geom init 시도) 와 직접 비교 가능 → fallback 프레임워크의 효용 정량화

**Sub-floor 처리 정책 (Q12b 재확인)**: drop은 hole/loss-mask mismatch/backbone gradient loss 야기 → default는 위 fallback (gate=1 injection). Drop은 ablation으로만 비교.

---

## 8. Loss Design (staged, QGS-paper-consistent)

### Stage 1 — Core
| Term | Formula |
|---|---|
| `L_range` | `L1(r_rendered, r_gt)` on rays with `α_accum > τ_valid` |
| `L_drop` | `BCE(p_drop, drop_gt)` on all rays |
| `L_intensity` | `L1(i_rendered, i_gt)` on valid rays (clamped [0,1]) |

### Stage 2 — Geometry (curvature-aware, Q10 수정)
| Term | Formula |
|---|---|
| `L_normal` (curvature-aware) | `Σ_ray w(κ_ray) · (1 − <n_pred, n_gt>)` with `w(κ) = exp(−β·κ_rendered)` |
| `L_pose_reg` (dynamic only) | `Σ_i (‖Δc_0^i‖² + ‖Δc_1^i‖² + ‖Δyaw_0^i‖² + ‖Δyaw_1^i‖²)` weight ~1e-4 |

**`L_shape_consistency` 삭제 (Q10)**: partial observability로 Chamfer 신호가 오염됨 (두 프레임에서 visible surface가 다름 → pose 오차와 부위 차이가 섞여 pose 신호로 사용 불가). Pose refinement는 L_range만으로 학습.

**`L_pose_reg` 추가 근거**: rendering loss만으로 학습할 때 초기 shape-pose 신호 분리를 돕는 soft prior. Pose가 tracker 값 근처 유지.

**곡률 weight 수식 근거**: QGS 원 논문의 normal loss 설계 따름. 고곡률 영역에서 normal loss weight를 감소시켜 **over-smoothing 방지** — 평평한 영역에서는 강하게, 곡면에서는 약하게 normal 감독. β는 QGS 논문 값 참고 (β ≈ 1~5 range).

**GT normal 소스**: GT sweep의 depth map 인접 픽셀 finite difference (spherical 좌표에서) 또는 GT 3D points의 k-NN PCA normal.

### Stage 3 — Optional
| Term | Formula | When |
|---|---|---|
| `L_los` | `Σ_ray Σ_{t_i < t_gt−ε} α_i·G_i` | floaters 잔존 |
| `L_cycle` | `render@t=0 vs P0` + `render@t=1 vs P1` | sparse 감독 보강 |
| `L_reg` | `‖Δpose_res‖² + ‖Δτ_s‖²` | 학습 초기 stability |

### 제거: L_chamfer3D
- L_range와 정보 중복 → 학습 loss에서 제외
- **Eval metric으로만** 사용

---

## 9. Novelty Summary (Plan v3 재정렬)

### Main Anchor (N series)

**N-1a. Analytic Epistemic Gating for Feed-forward GS Residual Head** ⭐ **flagship** (§4.4, §7.6) — feed-forward GS residual head 안에 analytic geometric fit quality `(fit_residual, planarity, support, reach)`와 tangent/curvature anisotropy를 **residual freedom gate**로 명시 주입. `g_rot`, `g_center`, `g_scale`이 bounded residual 크기를 조절한다. Sub-floor fallback도 같은 식 안에서 `gate=1` injection으로 통일. Per-scene posterior (PhysGS) / parameter-Hessian (FisherRF) / NBV epistemic (HERE, AREA3D) 어디에도 없는 빈 슬롯 — feed-forward 내부 explicit epistemic conditioning은 prior work 부재.

**N-1b. Local-Tangent Signed-Quadric Prior + Residual Head** (§7) — N-1a의 prerequisite. LiDAR의 surface-direct observation을 `p̄`-centered PCA + A-weighted 2nd-order LS + local shape-operator eigendecomp으로 closed-form QGS init `(c_init=μ_init, R_init, signed s1/s2, positive s3)`으로 변환. 이 init이 있어야 N-1a의 "얼마나 보정할지" 의미가 정의됨.

**N-2. Time/Ego-aware 2-Frame Conditioning** (§4.1–4.3) — time scalar + ego-motion direction vector (e_dir = T_{1→0}.translation)로 single backbone에서 2-frame feature 합성. 차량 진행 방향을 explicit signal로 주입.

**N-3. Canonical Shape + Position-Hermite + Velocity Head** (§6) —
  - **N-3a**: Shape-motion decomposition (canonical bbox-local shape, time-independent).
  - **N-3b**: **Position-only Cubic Hermite** (yaw는 shortest-path slerp, pitch는 atan2(Δz, d_xy)). Linear lerp baseline 대비 가속/감속 profile 표현.
  - **N-3c**: Velocity Estimation Head — Δp_local (pitch-aware) + S_avg + Δyaw_embed → **v̂_0, v̂_1 only** (shared MLP, ω̂는 slerp 채택으로 제거). `S_avg·2·σ(MLP)` finite-diff init + correction 형식. Tracker가 velocity를 주지 않거나 noisy해도 feed-forward 호환.

### Supporting (S series)

**S-1. LiDAR-QGS Rasterizer** (§5) — exact ray-paraboloid intersection + spherical AABB culling (no Jacobian approximation) + curvature-aware raydrop.

**~~S-2. Curvature-aware L_normal~~** → **implementation detail로 강등** (novelty claim 제외). §8 Stage 2 곡률 weighting은 QGS 원논문 식 차용이라 paper claim 불필요. 코드는 유지.

**~~S-3. First LiDAR application of QGS~~** → **삭제** ("first X+Y" 프레이밍은 약함, reviewer가 novelty로 인정 안 함). 2DGS 대비 primitive efficiency 비교는 ablation table 안에 흡수.

**~~S-4. Shape Consistency Loss~~** — **삭제 (Q10)**. Partial observability로 Chamfer 신호 오염 → pose refinement는 rendering loss만으로 학습.

### Positioning (v3)

기둥은 **N-1a (epistemic gating)**. 나머지는 supporting structure.

- Prior work 공란 (2026-04 WebSearch 검증):
  - PhysGS, Bayesian GS: per-scene posterior. ✗ feed-forward.
  - FisherRF, Active GS: parameter-space Hessian post-hoc. ✗ residual head 내부 gating 아님.
  - HERE (ICLR'26), AREA3D: epistemic → next-best-view. ✗ capture loop용.
  - G3Splat: degeneracy 분석. ✗ uncertainty 신호 아님.
  - pixelSplat / MVSplat / YoNoSplat: feed-forward image GS. ✗ uncertainty 신호 없음.
  - SplatAD / GS-LiDAR / LiDAR-RT: per-scene LiDAR NVS. ✗ feed-forward 아니고 gating 없음.
- 우리 슬롯 = _explicit analytic geometric fit quality → feed-forward GS residual prediction의 multiplicative gate_.

- **(feed-forward) × (LiDAR) × (QGS) × (2-frame)** 4축 교집합도 여전히 선행 없음 — N-1a가 작동 가능한 specific 환경.

---

## 10. Corner Cases & Critical Review (Q17)

| # | Corner case | 증상 | 처리 |
|---|---|---|---|
| 1 | bbox 내 points가 너무 적음 (< k_min) | Canonical branch fit 불가 | Instance drop + statistics log |
| 2 | 도로 ground plane (거대 평면) | QGS 곡률 over-kill, 비효율 | near-flat이면 `s3=quadric_eps_s3`로 제한되어 거의 2DGS처럼 동작. 별도 ground head 불필요. |
| 3 | Occluded surfaces | LiDAR 못 보는 영역 | 예측 안 함 (no extrapolation); eval metric에서 no-GT 영역 제외 |
| 4 | Instance ID switch (교차 차량) | Canonical cloud misalign | L_range 발산으로 detect 가능, outlier 기반 instance drop (shape cons 삭제되어 간접 detect) |
| 5 | Large motion + no velocity | Linear fallback 오차 큼 | Velocity Head 출력이 linear baseline에서 residual 흡수, 마지막 수단으로 drop |
| 6 | Extreme sensor pose (train OOD) | Feed-forward extrapolation failure | Limitation 명시, OOD stage 실험으로 quantify |
| 7 | t boundary (t=0, t=1) consistency | 렌더 = input frames와 일치해야 | L_cycle로 enforce |
| 8 | Intensity range | 정규화 범위 불일치 | [0,1] 정규화 고정, head output sigmoid clamp |
| 9 | Saddle ray-hit ambiguity | 두 교차점이 다른 arm | 각각 geodesic threshold 독립 평가 후 alpha blending에 모두 참여 |
| 10 | Negative s₃ (concave) | 수식 부호 뒤집힘 | α_q, β_q, γ_q에 sign 보존한 채 전 수식 유효 |
| 11 | k-NN이 먼 점 끌어옴 (scene 끝) | Garbage prior | Radius-bounded k-NN + fallback (k_min=8) |
| 12 | Tracker 완전 실패한 instance | Frame 0 only / Frame 1 only | **정적 객체로 편입** (Q3 정정), 통계 logger로 비율 리포트 |
| 13 | Cross-frame xyz jitter (ego-pose noise) | Merged cloud가 떨림 | Ego-pose는 nuScenes가 high accuracy → noise-free 가정. 실패 시 ICP refinement (v2) |
| 14 | Pitch degeneracy (주차/서행, d_xy 작음) | atan2가 LiDAR z-noise에 민감 | **d_xy < 1.0m → pitch = 0** (Q4). 통계 logger로 비율 모니터링 |
| 15 | k_eff < k_min (sparse far-field 점) | 2차 fit numerically unstable | **Fallback = absolute prediction** (Q12 사용자 확정). Ablation으로 drop 옵션 비교 |
| 16 | Cross-instance k-NN | 인접 차량 점이 fit 오염 | **instance별로 k-NN 분리** (Q11), static/dynamic pool 분리 |
| 17 | Azimuth wraparound | AABB 분할 필요 | §5.4에서 명시 처리 |

### Critical Weaknesses (정직하게)
- **Tracker dependency**: 우리는 offline detector/tracker에 의존 → upstream noise sensitivity. Mitigation: pose-refine head (keyframe only) + L_pose_reg + noise injection robustness study.
- **Sparse supervision for dynamic**: bbox 내 points가 종종 < 50 → canonical QGS를 충분히 define 못 할 수 있음. Mitigation: Point augmentation (symmetry prior for vehicle bilateral symmetry — v2 후보).
- **Feed-forward generalization on dynamic**: 동적 instance 다양성 (차량 종류) 일반화. Mitigation: nuScenes train split 전체 + Waymo/KITTI transfer.
- **QGS CUDA kernel 확장 cost**: will-zzy/QGS **snapshot 복사** 후 spherical projection + LiDAR-specific accumulation 구현. 1~2주 engineering.

---

## 11. Verification Plan

§13의 Phase A → Phase B → Phase C 순서를 따름. 각 phase의 verification은 해당 phase 마지막 step에 명시.

**핵심 metrics** (모든 phase 공통):
- Range MAE / RMSE
- Chamfer distance (reconstructed point cloud)
- Raydrop precision / recall / F1
- Intensity MAE
- Static-only vs dynamic-only breakdown

**Baselines**:
- Feed-forward 2DGS (same arch, s₃=0 fixed)
- (Optional) Per-scene QGS for upper-bound 비교

### Primary ablations — Epistemic Gating (N-1a flagship 검증)

이 표가 paper의 main result. A2.1 / A2.2 구현 순서에서 자동 부산물로 산출됨.

| # | Variant | 의도 | 기대 방향 |
|---|---|---|---|
| **E-1** | **fit_quality 4D + gated-residual (full)** | flagship | baseline |
| E-2 | fit_quality 0D (no gating, no concat) | head가 backbone feature `f`만으로 학습 | MAE ↑, 수렴 속도 ↓ |
| E-3 | fit_quality 1D (residual만 사용, planarity·support·reach 제거) | 4-tuple 각 차원 기여도 분해 | 부분 회복 |
| **E-4** | gated-residual 식 OFF (concat만, multiplicative gate 제거) | 4D 정보는 유지하되 `(1−g)·base + g·corr` 구조 자체 제거 | MAE 약간 ↑ → multiplicative 구조 효용 증명 |
| E-5 | sub-floor도 geom init 시도 (no gate=1 injection) | §7.6 fallback 프레임워크의 효용 | 학습 발산/불안정 예상 |
| E-6 | synthetic GT noise 주입 (controlled aleatoric) → gate 값 측정 | gate가 실제로 epistemic role 하는지 정성 검증 | gate ↑ with noise ↑ (monotonic) |

**E-1 vs E-2 MAE 격차 = paper의 main number.** E-4가 multiplicative gate 자체의 가치 분리, E-6이 gate semantics qualitative validation.

### Secondary ablations (Phase A 이후 누적)

- w/o canonical dynamic (world-frame dynamic)
- k_target ∈ {12, 16, 24, 32}, k_min ∈ {6, 7, 8, 10, 12}
- k_eff < k_min: (a) gate=1 injection (default) vs (b) drop point (Q12)
- Shared vs separate head (static / dynamic)
- Backbone: PTv3 vs spconv (via BACKBONE_REGISTRY swap, Q14)
- 2DGS (s₃=0) vs QGS — primitive efficiency (구 S-3 흡수)
- Phase B:
  - Position: Hermite vs linear lerp
  - Yaw: slerp (default) vs Hermite (historical comparison)
  - Velocity head ON (v̂ only) / OFF (linear v̂=Δp)
  - Pose-refine ON / OFF
  - Pitch ON / OFF (fixed 0)
  - Frame gap ∈ {1, 2, 4}
  - Tracker noise injection robustness
- w/o curvature-aware drop (S-1)
- **제거**: shape consistency ablation (loss 자체 삭제, Q10), S-2 curvature-aware L_normal ablation (implementation detail로 강등)

---

## 12. File Blueprint

```
models/
  qgs_nvs.py              # pipeline orchestration
  backbone/                               # Q14: registry pattern
    __init__.py           # BACKBONE_REGISTRY + build_backbone(config)
    base.py               # BackboneBase (abstract interface)
                          #   forward(xyz[N,3], features[N,C_in], batch_idx?) → [N, C_out]
    ptv3_backbone.py      # PTv3 wrapper + time scalar + e_dir broadcast
    custom_backbone.py    # 기존 nn/backbone.py 이전
    spconv_backbone.py    # ablation 후보
  head/
    qgs_head.py           # geometric init + residual MLP (shared static/dynamic, fit_quality 4D + is_dynamic_flag)
    pose_refine_head.py   # shared MLP, t=0/t=1 keyframe only (Q9)
    velocity_head.py      # v̂_0, v̂_1 only — ω̂ 제거 (Q6, Q8)
  geometry/
    quadric_fit.py        # PyTorch p̄-centered PCA + A-weighted LS + local-tangent signed-QGS init
    knn_radius.py         # chunked GPU k-NN (Q13)
                          #   - pytorch3d.ops.knn_points wrapper with chunking (pairwise distance OOM 방지)
                          #   - instance-별 분리 호출 (Q11: cross-instance 금지)
                          #   - training 시작 시 memory profile logger 1회
renderer/
  lidar_qgs_rasterizer/   # Q15: will-zzy/QGS snapshot 복사 (fork 아님)
    SOURCE.md             # 출처 기록 (original commit hash, license, modifications)
    ray_paraboloid.cu     # exact A·t²+B·t+C intersection
    spherical_aabb.cu     # tile culling with wraparound handling
    alpha_blend_range.cu  # range/intensity/normal/feat composition
    raydrop_mlp.cu        # feat_agg → drop logit (별도 drop map, Q3)
  dynamic_transform.py    # canonical ↔ world transform (Hermite position + slerp yaw + pitch)
losses/
  lidar_nvs.py            # range/drop/intensity/normal(curv)/los/cycle/pose_reg
                          # L_shape_consistency 제거 (Q10)
configs/
  qgs_nvs.py              # all hyperparams, backbone_type + backbone_kwargs dict
utils/
  tracking_stats.py       # untracked instance + pitch degeneracy 통계 logger
```

---

## 13. 구현 순서 (Q10 — 시간 모델링 분리)

**핵심 원칙 (Q10)**: 시간 모델링까지 한꺼번에 빌드하면 무엇이 실패 원인인지 분리 어려움.
먼저 "시간을 빼고 Feed-forward로 두 frame scene을 표현 가능한가?"부터 검증.
검증 후에 시간축 도입.

---

### Phase A — Feed-forward QGS Scene Representation (NO time modeling)

**목표**: 두 frame을 입력받아 그 두 frame 자기 자신을 잘 reconstruct하는지 검증.
- 입력: P0, P1, T_{1→0}, bboxes_{0,1}, instance_ids
- 모든 점을 Frame-0 sensor coord로 merge
- Static + canonical-dynamic decomposition
- Feed-forward QGS 생성 (geometric-init residual head)
- 렌더링 2회: (a) frame 0 sensor pose, (b) frame 1 sensor pose
  - frame 1 렌더 시 dynamic instance는 `R(yaw_1)·QGS_canon + c_1`로 변환
- Loss = L_range(render@frame0, P0) + L_range(render@frame1, P1) (+ L_drop, L_intensity)
- **시간 모델링, Hermite, velocity head 모두 제외**

**구현 단계 A1–A5**:

1. **A1. Geometry & data utils**
   - `quadric_fit.py` (p̄-centered PCA + A-weighted 2nd-order LS + local-tangent signed-QGS init) — unit test
   - `knn_radius.py` (hybrid radius-k with fallback)
   - `decomposition.py` (static/dynamic split, canonical transform, untracked → static)

2. **A2. Backbone & head** — local-tangent QGS init 위에 bounded geometry residual head를 올림.

   - **A2.0 (공통)**: Time-aware PTv3 input pipeline (8D features: xyz, intensity, time_scalar, e_dir broadcast). Backbone registry로 `ptv3` 등록.

   - **A2.1 — Geometry-first bounded residual head**:
     - Geometry output: `omega(3), delta_c(3), log_mean, log_gap, log_abs_s3`
     - `center = μ_init + g_center·tanh(delta_c)·head_center_bound`
     - `R = R_init·exp_so3(g_rot·tanh(omega)·rot_bound)`
     - `s1/s2` sign 보존, magnitude만 ordered log_mean/log_gap residual로 보정
     - `s3`는 positive curvature magnitude로 log_abs residual 보정

   - **A2.2 — Analytic gate add-on (N-1a flagship)**:
     - `MLP_gate(gate_input)` 추가, `gate_input=[fit_quality,tangent_aniso,curvature_aniso]`
     - Group gates: `g_rot`, `g_center`, `g_scale`
     - Gate bias init = −2.0 (학습 시작 시 작은 residual 보정)
     - Sub-floor fallback도 §7.6의 gate=1 injection으로 통합

3. **A3. LiDAR-QGS rasterizer (CUDA)** — 가장 큰 engineering cost
   - **전략 (Q15 사용자 확정)**: `will-zzy/QGS`를 **snapshot 복사**해서 우리 repo 안에서 독립적으로 빌드. Fork 아님, upstream과 상호작용 없음.
   - **왜 copy-in**: upstream 변경 추적 / rebase 관리 비용 없음, QGS의 camera-centric 구조를 LiDAR-centric으로 자유 refactor 가능. License 조건(attribution)만 준수.
   - **왜 gsplat(SplatAD base) 대신 QGS base**: gsplat은 Gaussian density primitive 중심 → paraboloid intersection을 새로 작성해야 함. QGS는 `A·t²+B·t+C=0`이 이미 구현됨 → primitive 수학 재사용 가치가 infrastructure 재사용보다 큼.

   **복사 절차**:
   ```bash
   # 1. QGS snapshot clone + commit pin
   git clone https://github.com/will-zzy/QGS /tmp/QGS-snapshot
   cd /tmp/QGS-snapshot && git rev-parse HEAD > /tmp/qgs_commit.txt

   # 2. rasterizer 디렉토리만 우리 repo로 복사 (원본 .git 제거)
   cp -r /tmp/QGS-snapshot/submodules/diff-quadratic-rasterization \
         /data/jeongbin/qgs/renderer/lidar_qgs_rasterizer/
   rm -rf /data/jeongbin/qgs/renderer/lidar_qgs_rasterizer/.git

   # 3. 출처/라이선스 명시
   #    renderer/lidar_qgs_rasterizer/SOURCE.md에 기록:
   #      - Origin: will-zzy/QGS @ <commit-hash>
   #      - Copied on: <date>
   #      - Upstream license: <라이선스 파일 복사 포함>
   #      - Modifications: spherical projection, LiDAR channel layout, curvature-aware drop
   ```

   **A3 세부 단계 (5-step port plan)**:
   - **A3.1** — Snapshot 그대로 빌드 (camera projection) 후 small RGB scene에서 forward/backward가 돌아가는지 sanity check. Upstream 코드가 정상 복사되었는지 검증.
   - **A3.2** — Camera projection을 **spherical projection**으로 교체 (ref: `GS-LiDAR/diff-gaussian-rasterization-2d/`의 panoramic kernel). Azimuth wraparound 처리. 이 단계에서 range map만 alpha-blend.
   - **A3.3** — **Channel layout 확장**: QGS의 hardcoded `rendered_image[3:6]=normal, [11:12]=curvature` (ref: QGS `gaussian_renderer/__init__.py:141–161`) 을 다음으로 재설계 — `range, intensity, drop_logit, α_accum, normal(3), curvature, feat_agg(D_f)`.
   - **A3.4** — **Curvature-aware drop MLP 통합** (S-1 novelty): feat_agg + κ_render + cosθ_inc + log(1+r) + ray_dir → p_drop_phys. Drop map 분해: `p_drop = (1−α_accum) + α_accum · p_drop_phys` (§5.5).
   - **A3.5** — **Single static frame self-rendering test** → range MAE가 2DGS baseline보다 나은지 확인 후 A4 진입.

   **Reference 코드 위치 (실제 구현 참조용)**:
   - SplatAD: `gsplat/rendering.py::lidar_rasterization`, `gsplat/cuda/csrc/{projection,rasterization}.cu` (LiDAR mode flag)
   - GS-LiDAR: `diff-gaussian-rasterization-2d/`, `gaussian_renderer/panoramic_renderer`
   - LiDAR-RT: `submodules/` OptiX kernel — **우리는 쓰지 않음** (OptiX 빌드 비용 + lnvs env 복잡)
   - QGS base: `submodules/diff-quadratic-rasterization` (우리가 복사해 오는 대상)

   **Engineering ablation note**: SplatAD의 `32×8 (azim×elev) tile + sorted-elev binning`은 spherical AABB 대비 GPU thread block 효율 높음. A3 완료 후 성능 부족하면 도입. Rolling-shutter time correction도 Phase B option.

4. **A4. Two-frame self-reconstruction loss**
   - `L_range_self = L1(render@frame0, P0) + L1(render@frame1, P1)`
   - `L_drop_self`, `L_intensity_self` 마찬가지
   - 시간축 무관, dynamic은 bbox 제공된 pose로만 변환

5. **A5. Phase A 검증 (Pass criteria for moving to Phase B)**
   - Single batch overfit: P0, P1 둘 다 거의 완벽 reconstruction
   - Full nuScenes mini-train: range MAE < 목표값 (구체 값은 baseline 대비)
   - Static-only / Dynamic-only metric breakdown
   - Ablation: w/o geometric-init, w/o canonical dynamic, fit_quality 0/4/12, k ∈ {12,16,24,32}, 2DGS (s₃=0)
   - **이 시점에 결과가 나쁘면 시간축 추가는 무의미** → Phase B 진입 보류

---

### Phase B — Time Modeling (Phase A 검증 후 진입)

**목표**: 두 keyframe 사이 임의의 t ∈ [0,1]에 대한 합성 + intermediate sweep supervision.

6. **B1. Pose-refine head (shared)** — tracker noise 대응. 입력에서 velocity 제거, Δp_local(pitch 반영) + pitch_est 포함. 보정은 **t=0, t=1 keyframe에만** 적용 (중간 t는 Hermite 결과 그대로). `L_pose_reg = ‖Δc‖² + ‖Δyaw‖²` (weight ~1e-4).
7. **B2. Velocity Estimation Head** (§6.4) — 입력 `[S_avg, Δp_local(pitch), Δyaw_embed, size, pitch_est, f_ctx?]` → **(v̂_0, v̂_1) 6D 출력만**. ω̂ 제거 (yaw는 slerp).
8. **B3. Hermite + slerp transform util** — Position은 Cubic Hermite `c(t) = h00·c_0 + h10·v̂_0 + h01·c_1 + h11·v̂_1`, Yaw는 shortest-path slerp `Δyaw = atan2(sin(yaw_1−yaw_0), cos(...))`. Canonical→world: `R_z(yaw(t))·R_y(pitch_est)·QGS_canon + c(t)`. Δt_real = 1.0s (frame_gap=2).
9. **B4. Intermediate sweep dataloader 추가** — loader `mode='bbox'`가 반환하는 1s interval 내 sweep 수를 첫 batch에서 로깅 확인.
10. **B5. Loss Stage 2** — curvature-aware L_normal (S-2). **Shape consistency loss는 제거됨** (partial observability 문제, Q10). Pose-refine head는 rendering loss + L_pose_reg로만 학습.
11. **B6. Loss Stage 3** — L_los, L_cycle, L_reg (필요 시)
12. **B7. Phase B 검증** — intermediate sweep reconstruction + ablation:
    - Position Hermite vs linear lerp (velocity head 필요성 검증)
    - Velocity head ON/OFF (v̂만 출력, ω̂ 없음)
    - Pose-refine head ON/OFF (SplatAD parity)
    - Pitch ON/OFF (`atan2` vs 0 고정)
    - frame_gap ∈ {1, 2, 4} 영향
    - Tracker noise injection robustness
    - k_min ∈ {6, 7, 8, 10, 12}; sub-floor policy {fallback, drop}

---

### Phase C — Generalization
13. KITTI-360, Waymo zero-shot OOD evaluation

각 phase는 선행 phase 검증 후 진입. Phase A 실패 시 phase B는 의미 없음.
