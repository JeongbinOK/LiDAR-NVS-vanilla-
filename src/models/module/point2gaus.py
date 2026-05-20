from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diff_quadratic_rasterization import LIDAR_LATENT_DIM
from ..geometry.voxel_anchor import VoxelAnchorOutput
from ..head.qgs_head import QGSHead
from ..ptv3.wrapper import PTv3Backbone, validate_context_type
from ..utils.render_utils import rotmat_to_quat


class QGSModel(nn.Module):
    """Feed-forward QGS predictor for voxel-anchor primitive tokens."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.feature_dim
        self.token_feature_dim = int(
            cfg.resolved_input_channels()
            if hasattr(cfg, "resolved_input_channels")
            else getattr(cfg, "anchor_token_dim", 22)
        )
        backbone_kwargs = cfg.ptv3_backbone_kwargs()
        backbone_kwargs["model_in_channels"] = max(
            int(backbone_kwargs.get("model_in_channels", self.token_feature_dim)),
            self.token_feature_dim,
        )

        self.backbone = PTv3Backbone(
            in_channels=self.token_feature_dim,
            out_channels=D,
            **backbone_kwargs,
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
            alpha_bias_init=getattr(cfg, "head_alpha_bias_init", 2.2),
            intensity_residual=getattr(cfg, "head_intensity_residual", True),
            center_bound=getattr(cfg, "head_center_bound", 0.3),
            rot_tilt_deg=getattr(cfg, "rot_tilt_deg", 10.0),
            rot_spin_deg=getattr(cfg, "rot_spin_deg", 30.0),
            scale_log_mean_bound=getattr(cfg, "scale_log_mean_bound", None),
            scale_log_gap_bound=getattr(cfg, "scale_log_gap_bound", None),
            s3_log_bound=getattr(cfg, "s3_log_bound", None),
            s3_fallback_abs=getattr(cfg, "s3_fallback_abs", 0.05),
        )

        # Init-summary normalization.
        self.knn_k_target = int(getattr(cfg, "knn_k_target", 16))
    def aaa(self, batch):
        return out
    def _run_backbone(
        self,
        xyz: Tensor,
        point_features: Tensor,
        *,
        context_type: str,
        offsets: Tensor | None = None,
    ) -> Tensor:
        backbone_kwargs = {}
        if offsets is not None:
            backbone_kwargs["offsets"] = offsets
        if getattr(self.backbone, "supports_context_type", False):
            backbone_kwargs["context_type"] = context_type
        if xyz.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.backbone(xyz, point_features, **backbone_kwargs)
            return features.float()
        return self.backbone(xyz, point_features, **backbone_kwargs)

    @staticmethod
    def _geom_init_from_anchor(anchor_out: VoxelAnchorOutput) -> dict:
        return {
            "c_init": anchor_out.c_init.unsqueeze(0),
            "R_init": anchor_out.R_init.unsqueeze(0),
            "s_init": anchor_out.s_init.unsqueeze(0),
            "fit_quality": anchor_out.fit_quality.unsqueeze(0),
            "use_geom_init": anchor_out.use_geom_init.unsqueeze(0),
            "kappa1_init": anchor_out.kappa1.unsqueeze(0),
            "kappa2_init": anchor_out.kappa2.unsqueeze(0),
            "tangent_aniso": anchor_out.tangent_aniso.unsqueeze(0),
            "curvature_aniso": anchor_out.curvature_aniso.unsqueeze(0),
        }

    @staticmethod
    def _pack_primitives(
        head_out: dict,
        features: Tensor,
        geom_init: dict,
        k_eff: Tensor,
        diagnostics: dict | None = None,
    ) -> dict:
        R = head_out["R"][0]
        rotations = rotmat_to_quat(R)
        return {
            "means3D": head_out["center"][0],
            "scales": head_out["s"][0],
            "rotations": rotations,
            "opacities": head_out["alpha"][0].unsqueeze(-1),
            "raydrop": head_out["raydrop"][0],
            "intensity": head_out["intensity"][0],
            "latent": head_out["latent"][0],
            "features": features,
            "aux": head_out["aux"],
            "geom_init": geom_init,
            "k_eff": k_eff,
            "diagnostics": diagnostics or {"memory_stages": []},
        }

    @staticmethod
    def _slice_aux(aux: dict, start: int, end: int) -> dict:
        sliced = {}
        for key, value in aux.items():
            if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == 1:
                sliced[key] = value[:, start:end]
            else:
                sliced[key] = value
        return sliced

    def _build_init_summary(self, geom_init: dict, k_eff: Tensor) -> Tensor:
        """Build the 13D per-anchor init-quality summary consumed by QGSHead."""
        s_init = geom_init["s_init"]
        fit_quality = (geom_init["fit_quality"] / 2.0).clamp(0.0, 2.0)
        aniso = torch.stack(
            [
                geom_init["tangent_aniso"],
                geom_init["curvature_aniso"],
            ],
            dim=-1,
        )
        aniso = (aniso / 2.0).clamp(0.0, 2.0)
        log_abs_s = (torch.log(s_init.abs().clamp(min=1e-6)) / 4.0).clamp(-2.0, 2.0)
        kappa = torch.stack(
            [
                geom_init["kappa1_init"],
                geom_init["kappa2_init"],
            ],
            dim=-1,
        )
        kappa = (kappa / 5.0).clamp(-2.0, 2.0)
        use_geom = geom_init["use_geom_init"].to(dtype=s_init.dtype).unsqueeze(-1)
        if k_eff.dim() == 1:
            k_eff = k_eff.unsqueeze(0)
        k_eff_norm = (k_eff.to(device=s_init.device, dtype=s_init.dtype) / max(self.knn_k_target, 1)).clamp(0.0, 1.0)
        return torch.cat(
            [
                fit_quality,
                aniso,
                log_abs_s,
                kappa,
                use_geom,
                k_eff_norm.unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward_anchor_context(
        self,
        anchor_out: VoxelAnchorOutput,
        *,
        context_type: str,
        return_diagnostics: bool = False,
    ) -> dict | None:
        """Predict primitives from a post-filter voxel-anchor output."""
        context_type = validate_context_type(context_type)
        N = int(anchor_out.c_init.shape[0])
        if N == 0:
            return None
        if anchor_out.token.shape != (N, self.token_feature_dim):
            raise ValueError(
                f"anchor token must be [{N}, {self.token_feature_dim}], "
                f"got {tuple(anchor_out.token.shape)}"
            )

        diagnostics = {"memory_stages": []}
        if return_diagnostics and anchor_out.c_init.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(anchor_out.c_init.device)

        features = self._run_backbone(
            anchor_out.c_init,
            anchor_out.token,
            context_type=context_type,
        )
        if features.shape[0] != N:
            raise RuntimeError(
                f"Backbone changed anchor count: {N} → {features.shape[0]}"
            )
        geom_init = self._geom_init_from_anchor(anchor_out)
        is_dynamic = torch.full(
            (1, N),
            1.0 if context_type == "dynamic" else 0.0,
            device=features.device,
            dtype=features.dtype,
        )
        head_out = self.qgs_head(
            features.unsqueeze(0),
            geom_init,
            is_dynamic,
            intensity_input=anchor_out.i_mean.unsqueeze(0),
            init_summary=self._build_init_summary(geom_init, anchor_out.k_eff),
        )
        return self._pack_primitives(
            head_out,
            features,
            geom_init,
            anchor_out.k_eff,
            diagnostics,
        )

    def forward_anchor_contexts_batched(
        self,
        anchor_outputs: list[VoxelAnchorOutput | None],
        *,
        context_type: str = "dynamic",
    ) -> list[dict | None]:
        """Run PTv3 once over non-empty voxel-anchor outputs and split back."""
        if not anchor_outputs:
            return []
        context_type = validate_context_type(context_type)
        valid_indices = [
            idx for idx, out in enumerate(anchor_outputs)
            if out is not None and out.c_init.shape[0] > 0
        ]
        results: list[dict | None] = [None] * len(anchor_outputs)
        if not valid_indices:
            return results

        valid = [anchor_outputs[idx] for idx in valid_indices]
        assert all(out is not None for out in valid)
        sizes = [int(out.c_init.shape[0]) for out in valid if out is not None]
        cat_xyz = torch.cat([out.c_init for out in valid if out is not None], dim=0)
        cat_token = torch.cat([out.token for out in valid if out is not None], dim=0)
        if cat_token.shape[1] != self.token_feature_dim:
            raise ValueError(
                f"anchor token width {cat_token.shape[1]} does not match "
                f"model input width {self.token_feature_dim}"
            )
        offsets = torch.cumsum(
            torch.tensor(sizes, dtype=torch.long, device=cat_xyz.device), dim=0
        )
        features = self._run_backbone(
            cat_xyz,
            cat_token,
            context_type=context_type,
            offsets=offsets,
        )

        geom_init = {
            "c_init": torch.cat([out.c_init for out in valid if out is not None], dim=0).unsqueeze(0),
            "R_init": torch.cat([out.R_init for out in valid if out is not None], dim=0).unsqueeze(0),
            "s_init": torch.cat([out.s_init for out in valid if out is not None], dim=0).unsqueeze(0),
            "fit_quality": torch.cat([out.fit_quality for out in valid if out is not None], dim=0).unsqueeze(0),
            "use_geom_init": torch.cat([out.use_geom_init for out in valid if out is not None], dim=0).unsqueeze(0),
            "kappa1_init": torch.cat([out.kappa1 for out in valid if out is not None], dim=0).unsqueeze(0),
            "kappa2_init": torch.cat([out.kappa2 for out in valid if out is not None], dim=0).unsqueeze(0),
            "tangent_aniso": torch.cat([out.tangent_aniso for out in valid if out is not None], dim=0).unsqueeze(0),
            "curvature_aniso": torch.cat([out.curvature_aniso for out in valid if out is not None], dim=0).unsqueeze(0),
        }
        k_eff = torch.cat([out.k_eff for out in valid if out is not None], dim=0)
        i_mean = torch.cat([out.i_mean for out in valid if out is not None], dim=0)
        is_dynamic = torch.full(
            (1, cat_xyz.shape[0]),
            1.0 if context_type == "dynamic" else 0.0,
            device=features.device,
            dtype=features.dtype,
        )
        head_out = self.qgs_head(
            features.unsqueeze(0),
            geom_init,
            is_dynamic,
            intensity_input=i_mean.unsqueeze(0),
            init_summary=self._build_init_summary(geom_init, k_eff),
        )

        center_splits = torch.split(head_out["center"][0], sizes, dim=0)
        scale_splits = torch.split(head_out["s"][0], sizes, dim=0)
        rot_splits = torch.split(rotmat_to_quat(head_out["R"][0]), sizes, dim=0)
        alpha_splits = torch.split(head_out["alpha"][0].unsqueeze(-1), sizes, dim=0)
        raydrop_splits = torch.split(head_out["raydrop"][0], sizes, dim=0)
        int_splits = torch.split(head_out["intensity"][0], sizes, dim=0)
        latent_splits = torch.split(head_out["latent"][0], sizes, dim=0)
        feat_splits = torch.split(features, sizes, dim=0)
        k_eff_splits = torch.split(k_eff, sizes, dim=0)

        start = 0
        for local_idx, orig_idx in enumerate(valid_indices):
            out = valid[local_idx]
            assert out is not None
            end = start + sizes[local_idx]
            sub_geom = self._geom_init_from_anchor(out)
            results[orig_idx] = {
                "means3D": center_splits[local_idx],
                "scales": scale_splits[local_idx],
                "rotations": rot_splits[local_idx],
                "opacities": alpha_splits[local_idx],
                "raydrop": raydrop_splits[local_idx],
                "intensity": int_splits[local_idx],
                "latent": latent_splits[local_idx],
                "features": feat_splits[local_idx],
                "aux": self._slice_aux(head_out["aux"], start, end),
                "geom_init": sub_geom,
                "k_eff": k_eff_splits[local_idx],
                "diagnostics": {"memory_stages": []},
            }
            start = end
        return results
