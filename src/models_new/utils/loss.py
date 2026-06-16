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
        self.w_chamfer   = cfg.w_chamfer    
        self.w_depth     = cfg.w_depth      
        self.w_intensity = cfg.w_intensity  
        self.w_raydrop   = cfg.w_raydrop    
        self.chamfer = chamfer_3DDist()
        # LPIPS VGG 모델 초기화
        if lpips is not None:
            self.lpips_fn = lpips.LPIPS(net='vgg').eval()
            # 훈련 과정에서 LPIPS 가중치가 업데이트되지 않도록 고정
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        else:
            self.lpips_fn = None
            print("[Warning] LPIPS 패키지가 없습니다. 로그 획득 시 제외됩니다.")

    def _calculate_psnr(self, pred, gt, mask=None):
        """MSE 기반의 PSNR 계산"""
        if mask is not None:
            if not mask.any(): return torch.tensor(0.0, device=pred.device)
            mse = F.mse_loss(pred[mask], gt[mask])
        else:
            mse = F.mse_loss(pred, gt)
        
        if mse <= 1e-8:
            return torch.tensor(50.0, device=pred.device)
        
        return 20 * math.log10(1.0) - 10 * torch.log10(mse)

    def _calculate_ssim(self, pred, gt):
        """[B, C, H, W] 차원 대응 경량화 2D SSIM"""
        mu_x = pred.mean(dim=[-2, -1], keepdim=True)
        mu_y = gt.mean(dim=[-2, -1], keepdim=True)
        sigma_x = pred.var(dim=[-2, -1], keepdim=True)
        sigma_y = gt.var(dim=[-2, -1], keepdim=True)
        sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean(dim=[-2, -1], keepdim=True)
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

    def _calculate_lpips(self, pred, gt):
        """[B, C, H, W] 텐서의 LPIPS 스코어 계산"""
        if self.lpips_fn is None:
            return torch.tensor(0.0, device=pred.device)
        
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
        gt_depth = all_renders["gt_depth"]
        pred_intensity = all_renders["intensity_sh"]
        gt_intensity = all_renders["gt_intensity_sh"]
        
        valid = gt_depth > 0  # valid 마스크 (B, C, H, W)

        # ----------------------------------------------------------
        # 1. 실제 학습에 쓰일 핵심 4가지 Loss 계산 (Backpropagation 타겟)
        # ----------------------------------------------------------
        if valid.any():
            losses["loss_depth"] = F.mae_loss(pred_depth[valid], gt_depth[valid])
            losses["loss_intensity"] = F.mae_loss(pred_intensity[valid], gt_intensity[valid])
        else:
            losses["loss_depth"] = torch.tensor(0.0, device=gt_depth.device)
            losses["loss_intensity"] = torch.tensor(0.0, device=gt_depth.device)

        # Raydrop Loss는 요청하신 대로 이진 분류를 처리하는 BCE Loss로 반영합니다.
        # 안전한 로그 연산을 위해 수치적 안정성이 잡힌 예측 맵을 클리핑 하거나 binary 형태로 계산합니다.
        losses["loss_raydrop"] = F.binary_cross_entropy(
            torch.clamp(all_renders["raydrop"], 1e-7, 1.0 - 1e-7), 
            all_renders["gt_raydrop"]
        )

        # Chamfer Distance
        pred_pts_batch = torch.stack([
                    p.reshape(-1, 3) for b_list in all_renders["render_points"] for p in b_list
                ], dim=0) # 형상: (B*C, N_pred, 3)
                
        gt_pts_batch = torch.stack([
            g.reshape(-1, 3) for b_list in all_renders["gt_points"] for g in b_list
        ], dim=0) # 형상: (B*C, N_gt, 3)

        # 2. 단 한 번의 CUDA 커널 호출로 전체 거리 계산
        dist1, dist2, _, _ = self.chamfer(pred_pts_batch, gt_pts_batch)
        
        # 3. 차원 전체에 대한 평균을 한 번에 구함 (수식 결과 동일)
        losses["loss_chamfer"] = dist1.mean() + dist2.mean()

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
        losses["intensity_lpips"] = self._calculate_lpips(p_int_2d, g_int_2d)

        # Depth Metrics
        losses["depth_psnr"] = self._calculate_psnr(p_depth_2d, g_depth_2d, mask=valid_2d)
        losses["depth_ssim"] = self._calculate_ssim(p_depth_2d, g_depth_2d)
        losses["depth_lpips"] = self._calculate_lpips(p_depth_2d, g_depth_2d)

        # ----------------------------------------------------------
        # 3. 가중치 결합을 통한 최종 Total Loss 정의 (4개만 반영)
        # ----------------------------------------------------------
        losses["total"] = (
            self.w_depth     * losses["loss_depth"]     +
            self.w_intensity * losses["loss_intensity"] +
            self.w_raydrop   * losses["raydrop"]   +  # config의 이름에 맞게 매핑 유지
            self.w_chamfer   * losses["loss_chamfer"]
        )

        return losses