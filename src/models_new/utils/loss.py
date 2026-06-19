import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#from pytorch3d import chamfer_loss
from ..utils.chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
try:
    import lpips
except ImportError:
    lpips = None

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
        """MSE 기반의 PSNR 계산. peak = 데이터의 최대값(intensity=1.0, depth=80m 등).
        PSNR = 20*log10(peak) - 10*log10(mse) 이므로 peak가 맞아야 dB 스케일이 의미를 가진다."""
        if mask is not None:
            if not mask.any(): return torch.tensor(0.0, device=pred.device)
            mse = F.mse_loss(pred[mask], gt[mask])
        else:
            mse = F.mse_loss(pred, gt)

        mse = torch.clamp(mse, min=1e-10)
        return 20 * math.log10(peak) - 10 * torch.log10(mse)

    def _calculate_ssim(self, pred, gt, data_range=1.0):
        """[B, C, H, W] 차원 대응 경량화 2D SSIM.
        data_range = 데이터의 dynamic range(L). 안정화 상수 C1=(0.01L)^2, C2=(0.03L)^2 이
        값의 크기에 비례해야 의미를 갖는다(아래 설명 참고)."""
        mu_x = pred.mean(dim=[-2, -1], keepdim=True)
        mu_y = gt.mean(dim=[-2, -1], keepdim=True)
        sigma_x = pred.var(dim=[-2, -1], keepdim=True)
        sigma_y = gt.var(dim=[-2, -1], keepdim=True)
        sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean(dim=[-2, -1], keepdim=True)

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

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

    def forward(self, all_renders):
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

        # ----------------------------------------------------------
        # 2. 로그 및 평가 전용 Metrics 계산 (역전파 제외 / 단순 로깅용)
        # ----------------------------------------------------------
        # 2D 이미지화 시키기 위해 고차원 구조 [B, Cam, H, W] -> [B*Cam, 1, H, W] 평탄화
        B, Cam, H, W = pred_depth.shape
        p_depth_2d = pred_depth.view(-1, 1, H, W)
        g_depth_2d = gt_depth.view(-1, 1, H, W)
        p_int_2d = pred_intensity.view(-1, 1, H, W)
        g_int_2d = gt_intensity.view(-1, 1, H, W)
        valid_2d = valid.view(-1, 1, H, W)

        # Intensity Metrics
        losses["intensity_psnr"] = self._calculate_psnr(p_int_2d, g_int_2d, mask=valid_2d)
        losses["intensity_ssim"] = self._calculate_ssim(p_int_2d, g_int_2d)
        if self.enable_lpips:
            intensity_lpips = self._calculate_lpips(p_int_2d, g_int_2d)
            if intensity_lpips is not None:
                losses["intensity_lpips"] = intensity_lpips

        # Depth Metrics (depth는 raw 미터값이므로 peak/data_range를 실제 범위 80m로 지정)
        losses["depth_psnr"] = self._calculate_psnr(p_depth_2d, g_depth_2d, mask=valid_2d, peak=80.0)
        losses["depth_ssim"] = self._calculate_ssim(p_depth_2d, g_depth_2d, data_range=80.0)
        if self.enable_lpips:
            depth_lpips = self._calculate_lpips(p_depth_2d, g_depth_2d)
            if depth_lpips is not None:
                losses["depth_lpips"] = depth_lpips

        # ----------------------------------------------------------
        # 3. 가중치 결합을 통한 최종 Total Loss 정의 (4개만 반영)
        # ----------------------------------------------------------
        # median은 w_depth_median으로 토글: 0이면 학습 제외(detached metric), >0이면 학습 loss로 합산.
        losses["total"] = (
            self.w_depth        * losses["loss_depth"]        +
            self.w_depth_median * losses["loss_depth_median"] +
            self.w_intensity    * losses["loss_intensity"]    +
            self.w_raydrop      * losses["loss_raydrop"]      +
            self.w_chamfer      * losses["loss_chamfer"]
        )

        # 항별 가중 기여(w·loss) 로깅 — 절대 loss가 아니라 이 값들을 보고 weight를 등화한다.
        losses["wc_depth"]        = (self.w_depth        * losses["loss_depth"]).detach()
        losses["wc_depth_median"] = (self.w_depth_median * losses["loss_depth_median"]).detach()
        losses["wc_intensity"]    = (self.w_intensity    * losses["loss_intensity"]).detach()
        losses["wc_raydrop"]      = (self.w_raydrop      * losses["loss_raydrop"]).detach()
        losses["wc_chamfer"]      = (self.w_chamfer      * losses["loss_chamfer"]).detach()

        return losses
