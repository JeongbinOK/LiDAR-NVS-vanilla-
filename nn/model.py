"""Quadratic Gaussian Splatting (QGS) feed-forward model.

The model is intentionally context-agnostic: callers provide a point set plus
per-point features, and the module returns primitive parameters for that
context. Training can then build a pair-wise scene from static and dynamic
subsets without changing the backbone/head internals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diff_quadratic_rasterization import LIDAR_LATENT_DIM
from models.geometry.knn_radius import gather_neighbors, hybrid_radius_knn
from models.geometry.quadric_fit import fit_local_quadrics
from models.head.qgs_head import QGSHead
from nn.backbone import PointFeatureBackbone
from nn.ptv3.wrapper import PTv3Backbone
from nn.render_utils import rotmat_to_quat


def _cuda_memory_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "alloc_mb": float(torch.cuda.memory_allocated(device) / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / 1024**2),
        "max_alloc_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2),
    }


class QGSModel(nn.Module):
    """Static-only feed-forward QGS predictor.

    The model is intentionally minimal for Phase A: every input point becomes
    one Gaussian primitive. Dynamic-instance branching (Phase B) lives outside
    this module and will be added later.
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.feature_dim
        self.input_feature_dim = int(getattr(cfg, "input_feature_dim", 8))

        if getattr(cfg, "backbone_type", "ptv3") == "ptv3":
            self.backbone = PTv3Backbone(
                in_channels=self.input_feature_dim,
                out_channels=D,
                grid_size=getattr(cfg, "ptv3_grid_size", 0.1),
                stride=getattr(cfg, "ptv3_stride", (2, 2)),
                enc_depths=getattr(cfg, "ptv3_enc_depths", (2, 2, 2)),
                enc_channels=getattr(cfg, "ptv3_enc_channels", (32, 64, 128)),
                enc_num_head=getattr(cfg, "ptv3_enc_num_head", (2, 4, 8)),
                enc_patch_size=getattr(cfg, "ptv3_enc_patch_size", (1024, 1024, 1024)),
                dec_depths=getattr(cfg, "ptv3_dec_depths", (2, 2)),
                dec_channels=getattr(cfg, "ptv3_dec_channels", (64, 64)),
                dec_num_head=getattr(cfg, "ptv3_dec_num_head", (4, 4)),
                dec_patch_size=getattr(cfg, "ptv3_dec_patch_size", (1024, 1024)),
                enable_flash=getattr(cfg, "ptv3_enable_flash", True),
                conv_algo=getattr(cfg, "ptv3_conv_algo", "native"),
            )
        else:
            self.backbone = PointFeatureBackbone(
                dim=D, num_blocks=cfg.num_blocks,
                window_size=cfg.window_size, num_heads=cfg.num_heads,
                in_channels=self.input_feature_dim,
            )

        latent_dim = getattr(cfg, "lidar_latent_dim", LIDAR_LATENT_DIM)
        if latent_dim != LIDAR_LATENT_DIM:
            raise ValueError(
                f"cfg.lidar_latent_dim={latent_dim} must equal "
                f"diff_quadratic_rasterization.LIDAR_LATENT_DIM={LIDAR_LATENT_DIM}"
            )
        self.qgs_head = QGSHead(
            feature_dim=D,
            latent_dim=latent_dim,
            hidden_dim=getattr(cfg, "head_hidden_dim", 128),
            gated=getattr(cfg, "use_gated_head", True),
            alpha_bias_init=getattr(cfg, "head_alpha_bias_init", 2.2),
            intensity_residual=getattr(cfg, "head_intensity_residual", True),
            center_mode=getattr(cfg, "head_center_mode", "fixed"),
            rot_tilt_deg=getattr(cfg, "rot_tilt_deg", 10.0),
            rot_spin_deg=getattr(cfg, "rot_spin_deg", 30.0),
            scale_log_mean_bound=getattr(cfg, "scale_log_mean_bound", None),
            scale_log_gap_bound=getattr(cfg, "scale_log_gap_bound", None),
            s3_log_bound=getattr(cfg, "s3_log_bound", None),
            s3_fallback_abs=getattr(cfg, "s3_fallback_abs", 0.05),
        )

        # geometry hyperparameters
        self.knn_k_target = int(getattr(cfg, "knn_k_target", 16))
        self.knn_k_min    = int(getattr(cfg, "knn_k_min", 8))
        self.knn_chunk    = int(getattr(cfg, "knn_chunk_size", 1024))

    # ------------------------------------------------------------------
    def _build_point_features(
        self,
        xyz: Tensor,
        intensity_norm: Tensor,
        time_scalar: Tensor | None = None,
        ego_motion: Tensor | None = None,
    ) -> Tensor:
        if intensity_norm.dim() == 2 and intensity_norm.shape[-1] == 1:
            intensity_norm = intensity_norm.squeeze(-1)
        if intensity_norm.dim() != 1:
            raise ValueError(
                f"intensity_norm must be [N] or [N,1]; got {tuple(intensity_norm.shape)}"
            )

        N = xyz.shape[0]
        device = xyz.device
        dtype = xyz.dtype

        if time_scalar is None:
            time_scalar = torch.zeros(N, device=device, dtype=dtype)
        elif time_scalar.dim() == 2 and time_scalar.shape[-1] == 1:
            time_scalar = time_scalar.squeeze(-1)
        if time_scalar.shape != (N,):
            raise ValueError(
                f"time_scalar must be [N]; got {tuple(time_scalar.shape)}"
            )

        if ego_motion is None:
            ego_motion = torch.zeros(3, device=device, dtype=dtype)
        if ego_motion.dim() == 1:
            ego_motion = ego_motion.unsqueeze(0).expand(N, 3)
        elif ego_motion.shape != (N, 3):
            raise ValueError(
                f"ego_motion must be [3] or [N,3]; got {tuple(ego_motion.shape)}"
            )

        return torch.cat(
            [
                xyz,
                intensity_norm.unsqueeze(-1),
                time_scalar.unsqueeze(-1),
                ego_motion,
            ],
            dim=-1,
        )

    def forward_context(
        self,
        xyz: Tensor,
        intensity_norm: Tensor | None = None,
        *,
        point_features: Tensor | None = None,
        time_scalar: Tensor | None = None,
        ego_motion: Tensor | None = None,
        is_dynamic_flag: Tensor | None = None,
        neighbor_xyz: Tensor | None = None,
        return_diagnostics: bool = False,
        **kwargs,
    ) -> dict:
        """
        Args:
            xyz:            [N, 3] point positions in the context frame.
            intensity_norm: [N] in [0, 1]. Used as the residual anchor in QGSHead
                            when intensity_residual=True.
            point_features:  optional [N, C] raw feature matrix. If omitted, the
                            default 8D contract [xyz, intensity, time, e_dir] is
                            constructed here.
            time_scalar:    [N] frame identity in {0, 1}.
            ego_motion:     [3] or [N, 3] broadcast motion vector.
            is_dynamic_flag:[N] float in {0, 1}; passed into the head.
            neighbor_xyz:   optional [M, 3] candidate pool for k-NN. Defaults to
                            `xyz` (context-local neighbourhood).

        Returns dict (per-Gaussian, all on `xyz.device`):
            'means3D'   [N, 3]   primitive centres in sensor frame.
            'scales'    [N, 3]   (s1, s2, s3); s3 may be signed.
            'rotations' [N, 4]   unit quaternion (w, x, y, z).
            'opacities' [N, 1]   in (0, 1).
            'intensity' [N]      in (0, 1).
            'latent'    [N, L]   per-Gaussian latent for the drop head.
            'features'  [N, D]   raw backbone features (diagnostic).
            'aux'       dict     QGSHead aux dict (gate, residuals).
            'geom_init' dict     output of `fit_local_quadrics` (diagnostic).
            'k_eff'     [N]      effective neighbour count.
        """
        N = xyz.shape[0]
        device = xyz.device
        if intensity_norm is None and point_features is None:
            raise ValueError("intensity_norm or point_features must be provided")

        if intensity_norm is None:
            if point_features is None:
                raise ValueError("point_features must be provided when intensity_norm is None")
            if point_features.dim() == 2 and point_features.shape[1] >= 4:
                intensity_norm = point_features[:, 3]
            else:
                raise ValueError(
                    "cannot infer intensity_norm from point_features with "
                    f"shape {tuple(point_features.shape)}"
                )

        if point_features is None:
            point_features = self._build_point_features(
                xyz, intensity_norm, time_scalar=time_scalar, ego_motion=ego_motion
            )
        elif point_features.shape[0] != N:
            raise ValueError(
                f"point_features must match xyz N={N}; got {point_features.shape[0]}"
            )

        diagnostics = {"memory_stages": []}

        def _record_stage(stage: str) -> None:
            if not return_diagnostics:
                return
            diagnostics["memory_stages"].append({
                "stage": stage,
                **_cuda_memory_snapshot(device),
            })

        if return_diagnostics and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            _record_stage("start")

        # 1. Backbone — preserve N (PTv3 segmentation mode upsamples back).
        # PTv3 + flash-attn requires bf16/fp16; the rest of the pipeline
        # (quadric fit, head) runs in fp32 to avoid linalg dtype mismatches.
        if return_diagnostics and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if xyz.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.backbone(xyz, point_features)
            features = features.float()
        else:
            features = self.backbone(xyz, point_features)
        _record_stage("backbone")
        if features.shape[0] != N:
            raise RuntimeError(
                f"Backbone changed point count: {N} → {features.shape[0]} — "
                "QGSHead expects per-input-point features."
            )

        # 2. k-NN within the same sweep (static-only Phase A).
        candidate_xyz = xyz if neighbor_xyz is None else neighbor_xyz
        if return_diagnostics and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        knn = hybrid_radius_knn(
            xyz,
            candidate_xyz,
            k_target=self.knn_k_target,
            k_min=self.knn_k_min,
            chunk_size=self.knn_chunk,
        )
        _record_stage("knn")
        if return_diagnostics and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        nbr_xyz = gather_neighbors(candidate_xyz, knn["idx"], knn["mask"])   # [N, K, 3]
        _record_stage("gather")

        # 3. Quadric fit → geometric init (gradient stops here — fit is in xyz only).
        if return_diagnostics and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        geom_init = fit_local_quadrics(
            points=xyz.unsqueeze(0),                               # [1, N, 3]
            neighbors=nbr_xyz.unsqueeze(0),                        # [1, N, K, 3]
            k_eff=knn["k_eff"].unsqueeze(0),                       # [1, N]
            k_min=self.knn_k_min,
            k_target=self.knn_k_target,
        )
        _record_stage("quadric_fit")

        # 4. Residual head.
        if is_dynamic_flag is None:
            is_dynamic = torch.zeros(1, N, device=device, dtype=features.dtype)
        else:
            if is_dynamic_flag.shape != (N,):
                raise ValueError(
                    f"is_dynamic_flag must be [N]; got {tuple(is_dynamic_flag.shape)}"
                )
            is_dynamic = is_dynamic_flag.to(device=device, dtype=features.dtype).unsqueeze(0)
        if return_diagnostics and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        head_out = self.qgs_head(
            features.unsqueeze(0),                                  # [1, N, D]
            geom_init,
            is_dynamic,
            intensity_input=intensity_norm.unsqueeze(0),            # [1, N]
        )
        _record_stage("head")

        # 5. Pack primitives for the rasterizer (drop the batch dim).
        R = head_out["R"][0]                                        # [N, 3, 3]
        rotations = rotmat_to_quat(R)                               # [N, 4]

        return {
            "means3D":   head_out["center"][0],                     # [N, 3]
            "scales":    head_out["s"][0],                          # [N, 3]
            "rotations": rotations,                                 # [N, 4]
            "opacities": head_out["alpha"][0].unsqueeze(-1),        # [N, 1]
            "intensity": head_out["intensity"][0],                  # [N]
            "latent":    head_out["latent"][0],                     # [N, L]
            "features":  features,                                  # [N, D]
            "aux":       head_out["aux"],
            "geom_init": geom_init,
            "k_eff":     knn["k_eff"],
            "diagnostics": diagnostics,
        }

    def forward(self, xyz: Tensor, intensity: Tensor, intensity_norm: Tensor | None = None, **kwargs) -> dict:
        """Backward-compatible wrapper for older single-context callers."""
        return self.forward_context(
            xyz,
            intensity_norm if intensity_norm is not None else intensity,
            **kwargs,
        )
