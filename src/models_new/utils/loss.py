import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity

#from pytorch3d import chamfer_loss
from ..utils.chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
try:
    import lpips
except ImportError:
    lpips = None

DEFAULT_SCALE_MAX_M = 2.5


def budget_weight_at_step(
    max_weight: float,
    step: int,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    """Return a 0 -> max_weight linear schedule in optimizer-step units."""
    max_weight = float(max_weight)
    step = int(step)
    warmup_steps = int(warmup_steps)
    ramp_steps = int(ramp_steps)
    if max_weight < 0.0:
        raise ValueError("budget weight must be non-negative")
    if warmup_steps < 0 or ramp_steps < 0:
        raise ValueError("budget warmup_steps and ramp_steps must be non-negative")
    if step < warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return max_weight
    progress = min(max((step - warmup_steps) / ramp_steps, 0.0), 1.0)
    return max_weight * progress


def expected_k_sum_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the noise-free categorical expected-K sum for local tokens."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("budget k_logits must have shape (N, K) with K >= 2")
    work_logits = logits.float()
    k_values = torch.arange(
        1,
        work_logits.shape[1] + 1,
        device=work_logits.device,
        dtype=work_logits.dtype,
    )
    policy = torch.softmax(work_logits, dim=-1)
    return (policy * k_values).sum(dim=-1).sum()


def distributed_token_mean(
    local_value_sum: torch.Tensor,
    local_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an exact global token mean with DDP-correct local gradients.

    The collective operates only on detached sufficient statistics. Each rank
    then adds a zero-valued local surrogate whose derivative is multiplied by
    ``world_size``. PyTorch DDP averages parameter gradients across ranks, so
    the resulting averaged gradient equals that of one global token-weighted
    mean, even when ranks contain different numbers of tokens.
    """
    if local_value_sum.ndim != 0:
        raise ValueError("local_value_sum must be a scalar tensor")
    local_token_count = int(local_token_count)
    if local_token_count < 0:
        raise ValueError("local_token_count must be non-negative")

    count = local_value_sum.detach().new_tensor(float(local_token_count))
    reduced = torch.stack((local_value_sum.detach(), count))
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()

    global_sum, global_count = reduced.unbind()
    if not bool((global_count > 0).item()):
        return local_value_sum * 0.0, global_count

    local_surrogate = (
        float(world_size)
        * (local_value_sum - local_value_sum.detach())
        / global_count
    )
    global_mean = global_sum / global_count + local_surrogate
    return global_mean, global_count


def target_budget_loss(
    logits: torch.Tensor,
    *,
    target_mean_k: float,
    max_weight: float,
    step: int,
    warmup_steps: int,
    ramp_steps: int,
    one_sided: bool = False,
) -> dict[str, torch.Tensor]:
    """Build the normalized global expected-mean-K target loss.

    ``one_sided=True`` turns ``target_mean_k`` from an alignment target into a
    ceiling: only the positive violation is penalized, so the router carries no
    budget gradient at all while the global expected mean K stays at or below
    it, and rendering alone decides the allocation inside that headroom. The
    default two-sided form pins the mean to the target from both directions and
    is what an alignment phase (e.g. the pseudo-GT guide) wants.

    ``violation`` is always reported signed so the logged diagnostic still
    shows how much headroom is left under a ceiling.
    """
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("budget k_logits must have shape (N, K) with K >= 2")
    k_max = int(logits.shape[1])
    target_mean_k = float(target_mean_k)
    if not 1.0 <= target_mean_k <= float(k_max):
        raise ValueError(
            f"budget target_mean_k must be in [1, {k_max}], "
            f"got {target_mean_k}"
        )

    local_expected_k_sum = expected_k_sum_from_logits(logits)
    expected_mean_k, global_token_count = distributed_token_mean(
        local_expected_k_sum,
        logits.shape[0],
    )
    effective_weight = budget_weight_at_step(
        max_weight,
        step,
        warmup_steps,
        ramp_steps,
    )
    weight = expected_mean_k.new_tensor(effective_weight)
    if not bool((global_token_count > 0).item()):
        zero = local_expected_k_sum * 0.0
        return {
            "expected_mean_k": zero,
            "global_token_count": global_token_count,
            "violation": zero,
            "loss": zero,
            "weight": weight,
            "weighted_loss": zero,
        }

    violation = expected_mean_k - target_mean_k
    # clamp_min is exactly zero-gradient below the ceiling, so a router sitting
    # under budget receives no count pressure whatsoever.
    penalized = violation.clamp_min(0.0) if bool(one_sided) else violation
    normalized_violation = penalized / float(k_max - 1)
    loss = normalized_violation.square()
    return {
        "expected_mean_k": expected_mean_k,
        "global_token_count": global_token_count,
        "violation": violation,
        "loss": loss,
        "weight": weight,
        "weighted_loss": weight * loss,
    }


def router_guide_loss(
    logits: torch.Tensor,
    target_k: torch.Tensor,
    *,
    geometry_valid: torch.Tensor,
    raw_count: torch.Tensor,
    weight: float,
    min_raw_points: int,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Exact global-token CE for the Grid pseudo-GT router guide.

    Targets are one-based physical K values.  Invalid geometry is intentionally
    retained in CE as its conservative K=1 policy; ``geometry_valid`` and
    ``raw_count`` are used only for globally reduced diagnostics.

    ``label_smoothing`` is what keeps the Gumbel router exploring.  Under the
    Gumbel-max trick a hard sample lands on the argmax class with probability
    ``softmax(logits)_argmax`` regardless of tau, so plain CE -- whose optimum
    drives that probability to one -- anneals exploration to zero.  Smoothing
    by ``eps`` over ``E`` classes caps the optimum at ``1 - eps*(E-1)/E`` and
    therefore pins the exploration rate at ``eps*(E-1)/E`` for the whole guide
    phase (eps=0.1, E=3 -> 6.7%).  SplatWeaver uses eps=0.1 for the same reason.

    The value/gradient construction mirrors ``distributed_token_mean`` so DDP's
    rank-averaged parameter gradients equal one global token-weighted CE even
    when ranks contain different token counts.
    """
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("router guide logits must have shape (N, K), K >= 2")
    token_count = int(logits.shape[0])
    aligned = (target_k, geometry_valid, raw_count)
    if any(item.ndim != 1 or item.shape[0] != token_count for item in aligned):
        raise ValueError("router guide fields must be one-dimensional and logit-aligned")
    target_k = target_k.to(device=logits.device, dtype=torch.long)
    geometry_valid = geometry_valid.to(device=logits.device, dtype=torch.bool)
    raw_count = raw_count.to(device=logits.device, dtype=torch.long)
    k_max = int(logits.shape[1])
    if target_k.numel() > 0 and bool(
        ((target_k < 1) | (target_k > k_max)).any()
    ):
        raise ValueError("router guide target_k must lie in [1, K_max]")
    weight = float(weight)
    min_raw_points = int(min_raw_points)
    label_smoothing = float(label_smoothing)
    if weight < 0.0:
        raise ValueError("router guide weight must be non-negative")
    if min_raw_points < 1:
        raise ValueError("router guide min_raw_points must be positive")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("router guide label_smoothing must be in [0, 1)")

    if token_count > 0:
        local_loss_sum = F.cross_entropy(
            logits.float(),
            target_k - 1,
            reduction="sum",
            label_smoothing=label_smoothing,
        )
        predicted_k = logits.detach().argmax(dim=-1) + 1
        correct = (predicted_k == target_k).sum()
        class_counts = torch.stack([
            (target_k == k).sum() for k in range(1, k_max + 1)
        ])
    else:
        local_loss_sum = logits.sum() * 0.0
        correct = target_k.new_zeros(())
        class_counts = target_k.new_zeros((k_max,))

    detached_statistics = torch.cat([
        local_loss_sum.detach().reshape(1),
        logits.detach().new_tensor([
            float(correct.item()),
            float(token_count),
            float(target_k.sum().item()),
            float(geometry_valid.sum().item()),
            float((raw_count < min_raw_points).sum().item()),
        ]),
        class_counts.to(device=logits.device, dtype=logits.dtype),
    ])
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(detached_statistics, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()

    (
        global_loss_sum,
        global_correct,
        global_count,
        global_target_sum,
        global_valid_count,
        global_low_support_count,
    ) = detached_statistics[:6]
    global_class_counts = detached_statistics[6:]
    if not bool((global_count > 0).item()):
        zero = local_loss_sum * 0.0
        return {
            "loss": zero,
            "weighted_loss": zero,
            "weight": zero.detach(),
            "accuracy": zero.detach(),
            "mean_target_k": zero.detach(),
            "geometry_valid_fraction": zero.detach(),
            "low_support_fraction": zero.detach(),
            "target_fractions": zero.detach().expand(k_max),
            "global_token_count": global_count,
        }

    local_surrogate = (
        float(world_size)
        * (local_loss_sum - local_loss_sum.detach())
        / global_count
    )
    loss = global_loss_sum / global_count + local_surrogate
    weight_tensor = loss.new_tensor(weight)
    return {
        "loss": loss,
        "weighted_loss": weight_tensor * loss,
        "weight": weight_tensor.detach(),
        "accuracy": (global_correct / global_count).detach(),
        "mean_target_k": (global_target_sum / global_count).detach(),
        "geometry_valid_fraction": (global_valid_count / global_count).detach(),
        "low_support_fraction": (
            global_low_support_count / global_count
        ).detach(),
        "target_fractions": (global_class_counts / global_count).detach(),
        "global_token_count": global_count,
    }


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 역전파(Loss)에 사용될 가중치들
        self.w_chamfer      = cfg.w_chamfer
        self.w_depth        = cfg.w_depth
        self.w_depth_median = getattr(cfg, "w_depth_median", cfg.w_depth)
        self.w_intensity    = cfg.w_intensity
        self.w_raydrop      = cfg.w_raydrop
        self.w_scale        = float(getattr(cfg, "w_scale", 0.0))
        self.scale_max_m    = float(
            getattr(cfg, "scale_max_m", DEFAULT_SCALE_MAX_M)
        )
        if not self.scale_max_m > 0.0:
            raise ValueError("loss.scale_max_m must be positive")
        self.chamfer = chamfer_3DDist()
        self.enable_lpips = bool(getattr(cfg, "enable_lpips", False))
        # LPIPS is an expensive logging metric, not part of the training loss.
        if self.enable_lpips and lpips is not None:
            self.lpips_fn = lpips.LPIPS(net='vgg').eval()
            # 훈련 과정에서 LPIPS 가중치가 업데이트되지 않도록 고정
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        else:
            self.lpips_fn = None

    def _calculate_psnr(self, pred, gt, mask=None, peak=1.0):
        """Per-frame LiDAR4D / GS-LiDAR PSNR, averaged over frames.

        GS-LiDAR clamps depth/intensity to [1e-6, peak] before computing
        10*log10(peak^2 / MSE). Official evaluation updates its meter once per
        frame and then averages the resulting PSNR values, so a flattened
        multi-camera batch must not form one global MSE. ``mask`` preserves this
        repo's valid-only diagnostic while retaining the same per-frame order.

        This implementation stays on the input device. The previous NumPy path
        synchronized and copied every training batch even though PSNR itself is
        a simple reduction.
        """
        with torch.no_grad():
            out_device = pred.device
            out_dtype = pred.dtype
            pred = pred.clamp(1e-6, peak).float()
            gt = gt.clamp(1e-6, peak).float()
            squared_error = (pred - gt).square()
            if squared_error.ndim == 0:
                squared_error = squared_error.reshape(1, 1)
            elif squared_error.ndim == 1:
                squared_error = squared_error.unsqueeze(0)
            else:
                squared_error = squared_error.reshape(squared_error.shape[0], -1)

            if mask is not None:
                mask = mask.bool()
                if mask.ndim == 0:
                    mask = mask.reshape(1, 1)
                elif mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                else:
                    mask = mask.reshape(mask.shape[0], -1)
                counts = mask.sum(dim=1)
                keep = counts > 0
                if not bool(keep.any()):
                    return torch.tensor(0.0, device=out_device, dtype=out_dtype)
                mse = (
                    (squared_error * mask.to(squared_error.dtype)).sum(dim=1)
                    / counts.clamp_min(1).to(squared_error.dtype)
                )[keep]
            else:
                mse = squared_error.mean(dim=1)
            peak_sq = squared_error.new_tensor(float(peak) ** 2)
            psnr = 10.0 * torch.log10(peak_sq / mse)
            return psnr.mean().to(device=out_device, dtype=out_dtype)

    def _calculate_ssim(self, pred, gt, peak=1.0):
        """GS-LiDAR eval-style SSIM.

        Uses the same skimage.metrics.structural_similarity call as GS-LiDAR
        eval, after the same [1e-6, max] clamp.
        """
        with torch.no_grad():
            out_device = pred.device
            out_dtype = pred.dtype
            pred = pred.clamp(1e-6, peak)
            gt = gt.clamp(1e-6, peak)
            pred_np = pred.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            scores = []
            for pred_item, gt_item in zip(pred_np, gt_np):
                pred_img = np.squeeze(pred_item, axis=0)
                gt_img = np.squeeze(gt_item, axis=0)
                scores.append(
                    structural_similarity(
                        pred_img,
                        gt_img,
                        data_range=np.max(gt_img) - np.min(gt_img),
                    )
                )
            return torch.tensor(float(np.mean(scores)), device=out_device, dtype=out_dtype)

    @staticmethod
    def _flatten_maps(*maps):
        """[B, Cam, H, W] -> [B*Cam, 1, H, W] for image metrics."""
        B, Cam, H, W = maps[0].shape
        return [m.view(-1, 1, H, W) for m in maps]

    def _calculate_lpips(self, pred, gt):
        """[B, C, H, W] 텐서의 LPIPS 스코어 계산"""
        if self.lpips_fn is None:
            return None
        
        # LPIPS 네트워크의 디바이스를 현재 데이터의 GPU 위치와 강제 동기화
        self.lpips_fn = self.lpips_fn.to(pred.device)

        # 1채널 맵일 경우 3채널 복사
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)
            
        # [0, 1] 범위를 LPIPS 목적 범위인 [-1, 1]로 스케일링
        p_img = torch.clamp((pred * 2.0) - 1.0, -1.0, 1.0)
        g_img = torch.clamp((gt * 2.0) - 1.0, -1.0, 1.0)
        
        with torch.no_grad():
            score = self.lpips_fn(p_img, g_img).mean()
        return score

    def _scale_regularization(self, gaussians, reference):
        """Penalize only Gaussians exceeding ``self.scale_max_m``.

        The squared log-ratio is averaged over violating Gaussians in each
        violating sample, then over only those samples. Non-violating Gaussians
        and samples therefore contribute neither gradients nor reduction
        denominator. Viewpoint mode stores Common once even though it appears in
        every render union, so a violating Common row is weighted by its number
        of valid target views; this is exactly equivalent to the old physical
        repetition without retaining V copies. ``w_scale=0`` is a true off
        switch and does not require Gaussian tensors to be passed.
        """
        if self.w_scale == 0.0:
            return reference.new_zeros(())
        if gaussians is None:
            raise ValueError("loss.w_scale > 0 requires Gaussian outputs")
        if isinstance(gaussians, dict):
            gaussians = gaussians.get("batch_gaussians", gaussians.get("gaussians"))
        if gaussians is None:
            raise ValueError("Could not find batch Gaussian outputs for scale regularization")

        sample_losses = []
        connected_zero = reference.new_zeros(())
        for batch_item in gaussians:
            if batch_item is None:
                continue
            raw_scale = batch_item.get("scaling")
            if raw_scale is None:
                raise KeyError("Gaussian output is missing 'scaling'")
            if raw_scale.shape[0] == 0:
                continue
            connected_zero = connected_zero + raw_scale.sum() * 0.0
            scales = F.softplus(raw_scale[:, :2])
            max_scale = scales.amax(dim=-1)
            excess = F.relu(
                torch.log(max_scale.clamp_min(1e-6))
                - max_scale.new_tensor(self.scale_max_m).log()
            )
            multiplicity = torch.ones_like(excess)
            gaussian_view = batch_item.get("view_index")
            common_view_gate = batch_item.get("common_view_gate")
            if gaussian_view is not None and common_view_gate is not None:
                common_mask = gaussian_view.to(device=excess.device) == -1
                if common_view_gate.ndim != 2:
                    raise ValueError(
                        "common_view_gate must have shape (N_common, V)"
                    )
                common_multiplicity = common_view_gate.detach().ne(0).sum(
                    dim=-1
                ).clamp_min(1).to(device=excess.device, dtype=excess.dtype)
                multiplicity[common_mask] = common_multiplicity
            violated = excess > 0
            if violated.any():
                violated_weight = multiplicity[violated]
                sample_losses.append(
                    (excess[violated].square() * violated_weight).sum()
                    / violated_weight.sum()
                )

        if not sample_losses:
            return connected_zero
        return torch.stack(sample_losses).mean()

    def forward(
        self,
        all_renders,
        *,
        gaussians=None,
        routing_budget=None,
        metric_mode="train",
        compute_valid_metrics=True,
        compute_raydrop_metrics=True,
    ):
        losses = {}

        # 데이터 세팅
        pred_depth = all_renders["depth"]
        pred_depth_median = all_renders["depth_median"]
        gt_depth = all_renders["gt_depth"]
        pred_intensity = all_renders["intensity_sh"]
        gt_intensity = all_renders["gt_intensity_sh"]
        pred_raydrop = all_renders["raydrop"]
        gt_raydrop = all_renders["gt_raydrop"]

        if pred_depth.dim() == 5 and pred_depth.shape[2] == 1:
            pred_depth = pred_depth.squeeze(2)
            pred_depth_median = pred_depth_median.squeeze(2)
            gt_depth = gt_depth.squeeze(2)
            pred_intensity = pred_intensity.squeeze(2)
            gt_intensity = gt_intensity.squeeze(2)
            pred_raydrop = pred_raydrop.squeeze(2)
            gt_raydrop = gt_raydrop.squeeze(2)
        
        valid = gt_depth > 0  # valid 마스크 (B, C, H, W)

        # ----------------------------------------------------------
        # 1. 실제 학습에 쓰일 핵심 4가지 Loss 계산 (Backpropagation 타겟)
        # ----------------------------------------------------------
        if valid.any():
            losses["loss_depth"] = F.l1_loss(pred_depth[valid], gt_depth[valid])
            # median: w_depth_median>0 이면 학습 loss, 0이면 detached 모니터링 metric (집중 gradient 회피).
            median_l1 = F.l1_loss(pred_depth_median[valid], gt_depth[valid])
            losses["loss_depth_median"] = median_l1 if self.w_depth_median > 0 else median_l1.detach()
            losses["loss_intensity"] = F.l1_loss(pred_intensity[valid], gt_intensity[valid])
        else:
            losses["loss_depth"] = torch.tensor(0.0, device=gt_depth.device)
            losses["loss_depth_median"] = torch.tensor(0.0, device=gt_depth.device)
            losses["loss_intensity"] = torch.tensor(0.0, device=gt_depth.device)

        # Raydrop Loss는 요청하신 대로 이진 분류를 처리하는 BCE Loss로 반영합니다.
        # 안전한 로그 연산을 위해 수치적 안정성이 잡힌 예측 맵을 클리핑 하거나 binary 형태로 계산합니다.
        losses["loss_raydrop"] = F.binary_cross_entropy(
            torch.clamp(pred_raydrop, 1e-7, 1.0 - 1e-7),
            gt_raydrop
        )

        chamfer_terms = []
        pred_point_counts = []
        gt_point_counts = []
        for pred_list, gt_list in zip(all_renders["render_points"], all_renders["gt_points"]):
            for pred_pts, gt_pts in zip(pred_list, gt_list):
                pred_pts = pred_pts.reshape(-1, 3)
                gt_pts = gt_pts.reshape(-1, 3)
                pred_point_counts.append(pred_pts.shape[0])
                gt_point_counts.append(gt_pts.shape[0])
                if pred_pts.numel() == 0 or gt_pts.numel() == 0:
                    continue
                dist1, dist2, _, _ = self.chamfer(
                    pred_pts.unsqueeze(0).contiguous(),
                    gt_pts.unsqueeze(0).contiguous(),
                )
                chamfer_terms.append(dist1.mean() + dist2.mean())
        if chamfer_terms:
            losses["loss_chamfer"] = torch.stack(chamfer_terms).mean()
        else:
            losses["loss_chamfer"] = torch.tensor(0.0, device=gt_depth.device)
        losses["chamfer_valid_pairs"] = torch.tensor(
            len(chamfer_terms), device=gt_depth.device, dtype=gt_depth.dtype
        )
        losses["render_points_mean"] = torch.tensor(
            sum(pred_point_counts) / max(len(pred_point_counts), 1),
            device=gt_depth.device,
            dtype=gt_depth.dtype,
        )
        losses["gt_points_mean"] = torch.tensor(
            sum(gt_point_counts) / max(len(gt_point_counts), 1),
            device=gt_depth.device,
            dtype=gt_depth.dtype,
        )
        losses["loss_scale"] = self._scale_regularization(gaussians, gt_depth)

        # ----------------------------------------------------------
        # 2. 로그 및 평가 전용 Metrics 계산 (역전파 제외 / 단순 로깅용)
        # ----------------------------------------------------------
        primary_intensity_pred = None
        primary_depth_pred = None
        if compute_valid_metrics or compute_raydrop_metrics:
            p_depth_2d, g_depth_2d, p_int_2d, g_int_2d, valid_2d = self._flatten_maps(
                pred_depth,
                gt_depth,
                pred_intensity,
                gt_intensity,
                valid,
            )

        if compute_valid_metrics:
            # Diagnostic only: measure value quality on GT-hit support while
            # separating it from predicted raydrop quality.
            p_depth_valid_2d = p_depth_2d * valid_2d
            g_depth_valid_2d = g_depth_2d * valid_2d
            p_int_valid_2d = p_int_2d * valid_2d
            g_int_valid_2d = g_int_2d * valid_2d
            losses["intensity_psnr_valid"] = self._calculate_psnr(
                p_int_2d, g_int_2d, mask=valid_2d
            )
            losses["intensity_ssim_valid"] = self._calculate_ssim(
                p_int_valid_2d, g_int_valid_2d, peak=1.0
            )
            losses["depth_psnr_valid"] = self._calculate_psnr(
                p_depth_2d, g_depth_2d, mask=valid_2d, peak=80.0
            )
            losses["depth_ssim_valid"] = self._calculate_ssim(
                p_depth_valid_2d, g_depth_valid_2d, peak=80.0
            )

        if compute_raydrop_metrics:
            # Official LiDAR4D / GS-LiDAR image protocol: hard-mask predicted
            # no-return rays, then evaluate the full depth/intensity maps.
            pred_keep = (pred_raydrop.detach() <= 0.5).to(dtype=pred_depth.dtype)
            pred_depth_raydrop = pred_depth * pred_keep
            pred_intensity_raydrop = pred_intensity * pred_keep
            p_depth_raydrop_2d, p_int_raydrop_2d = self._flatten_maps(
                pred_depth_raydrop,
                pred_intensity_raydrop,
            )
            losses["intensity_psnr_raydrop"] = self._calculate_psnr(
                p_int_raydrop_2d, g_int_2d
            )
            losses["intensity_ssim_raydrop"] = self._calculate_ssim(
                p_int_raydrop_2d, g_int_2d, peak=1.0
            )
            losses["depth_psnr_raydrop"] = self._calculate_psnr(
                p_depth_raydrop_2d, g_depth_2d, peak=80.0
            )
            losses["depth_ssim_raydrop"] = self._calculate_ssim(
                p_depth_raydrop_2d, g_depth_2d, peak=80.0
            )

        use_raydrop_metrics = metric_mode in {"val", "test", "eval"}
        if use_raydrop_metrics and compute_raydrop_metrics:
            primary_intensity_pred = p_int_raydrop_2d
            primary_depth_pred = p_depth_raydrop_2d
            losses["intensity_psnr"] = losses["intensity_psnr_raydrop"]
            losses["intensity_ssim"] = losses["intensity_ssim_raydrop"]
            losses["depth_psnr"] = losses["depth_psnr_raydrop"]
            losses["depth_ssim"] = losses["depth_ssim_raydrop"]
        elif compute_valid_metrics:
            primary_intensity_pred = p_int_valid_2d
            primary_depth_pred = p_depth_valid_2d
            losses["intensity_psnr"] = losses["intensity_psnr_valid"]
            losses["intensity_ssim"] = losses["intensity_ssim_valid"]
            losses["depth_psnr"] = losses["depth_psnr_valid"]
            losses["depth_ssim"] = losses["depth_ssim_valid"]
        elif compute_raydrop_metrics:
            primary_intensity_pred = p_int_raydrop_2d
            primary_depth_pred = p_depth_raydrop_2d
            losses["intensity_psnr"] = losses["intensity_psnr_raydrop"]
            losses["intensity_ssim"] = losses["intensity_ssim_raydrop"]
            losses["depth_psnr"] = losses["depth_psnr_raydrop"]
            losses["depth_ssim"] = losses["depth_ssim_raydrop"]

        # Intensity Metrics
        if self.enable_lpips and primary_intensity_pred is not None:
            intensity_lpips = self._calculate_lpips(primary_intensity_pred, g_int_2d)
            if intensity_lpips is not None:
                losses["intensity_lpips"] = intensity_lpips

        # Depth Metrics (depth는 raw 미터값이므로 peak/data_range를 실제 범위 80m로 지정)
        if self.enable_lpips and primary_depth_pred is not None:
            depth_lpips = self._calculate_lpips(primary_depth_pred, g_depth_2d)
            if depth_lpips is not None:
                losses["depth_lpips"] = depth_lpips

        # ----------------------------------------------------------
        # 3. 가중치 결합을 통한 최종 Total Loss 정의
        # ----------------------------------------------------------
        # median은 w_depth_median으로 토글: 0이면 학습 제외(detached metric), >0이면 학습 loss로 합산.
        losses["total"] = (
            self.w_depth        * losses["loss_depth"]        +
            self.w_depth_median * losses["loss_depth_median"] +
            self.w_intensity    * losses["loss_intensity"]    +
            self.w_raydrop      * losses["loss_raydrop"]      +
            self.w_chamfer      * losses["loss_chamfer"]      +
            self.w_scale        * losses["loss_scale"]
        )

        # 항별 가중 기여(w·loss) 로깅 — 절대 loss가 아니라 이 값들을 보고 weight를 등화한다.
        losses["wc_depth"]        = (self.w_depth        * losses["loss_depth"]).detach()
        losses["wc_depth_median"] = (self.w_depth_median * losses["loss_depth_median"]).detach()
        losses["wc_intensity"]    = (self.w_intensity    * losses["loss_intensity"]).detach()
        losses["wc_raydrop"]      = (self.w_raydrop      * losses["loss_raydrop"]).detach()
        losses["wc_chamfer"]      = (self.w_chamfer      * losses["loss_chamfer"]).detach()
        losses["wc_scale"]        = (self.w_scale        * losses["loss_scale"]).detach()

        # Optional learned-count regularization belongs to the same loss
        # assembly as the rendering terms. Existing callers omit this argument
        # and therefore preserve their historical total exactly.
        if routing_budget is not None:
            budget_logits = routing_budget.get("logits")
            if budget_logits is None:
                raise ValueError("routing_budget requires attached router logits")
            terms = target_budget_loss(
                budget_logits,
                target_mean_k=routing_budget["target_mean_k"],
                max_weight=routing_budget["weight"],
                step=routing_budget["step"],
                warmup_steps=routing_budget["warmup_steps"],
                ramp_steps=routing_budget["ramp_steps"],
                one_sided=routing_budget.get("one_sided", False),
            )
            losses["loss_budget"] = terms["loss"]
            losses["wc_budget"] = terms["weighted_loss"].detach()
            losses["budget_expected_mean_k"] = terms["expected_mean_k"].detach()
            losses["budget_violation"] = terms["violation"].detach()
            losses["budget_effective_weight"] = terms["weight"].detach()
            losses["total"] = losses["total"] + terms["weighted_loss"]

        return losses
