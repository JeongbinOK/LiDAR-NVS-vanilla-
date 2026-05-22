from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diff_quadratic_rasterization import LIDAR_LATENT_DIM
from models.geometry import decompose_scene
from models.geometry.voxel_anchor import (
    DynamicVoxelAnchorBuilder,
    VoxelAnchorBuilder,
    VoxelAnchorOutput,
)
from models.head.qgs_head import QGSHead
from models.ptv3.wrapper import PTv3Backbone, validate_context_type
from utils.lidar_geometry import make_lidar_ray_grid, points_to_lidar_maps
from utils.qgs_loss import QGSLoss
from utils.render_utils import (
    build_gt_normal_map,
    quat_to_rotmat,
    render_primitives,
    rotmat_to_quat,
)


class QGSPrimitiveModel(nn.Module):
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


def _batch_item(batch, key, idx):
    value = batch[key]
    if isinstance(value, (list, tuple)):
        return value[idx]
    if torch.is_tensor(value):
        return value[idx]
    return value[idx]


def _ensure_2d_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.dim() == 3 and pose.shape[0] == 1:
        return pose[0]
    return pose


def _filter_frame_points(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    cfg,
    ring: torch.Tensor | None = None,
):
    r = xyz.norm(dim=1)
    keep = (r > cfg.ego_radius) & (r < cfg.r_far)
    if ring is None:
        return xyz[keep], intensity[keep]
    return xyz[keep], intensity[keep], ring[keep]


def _make_static_anchor_builder(cfg) -> VoxelAnchorBuilder:
    return VoxelAnchorBuilder(
        k_min=cfg.knn_k_min,
        k_target=cfg.knn_k_target,
        residual_threshold=cfg.anchor_residual_threshold,
        filter_mode=cfg.anchor_filter_mode,
        planarity_threshold=cfg.anchor_planarity_threshold,
        token_variant=cfg.anchor_token_variant,
        knn_chunk_size=min(cfg.knn_chunk_size, 256),
        quadric_gamma=cfg.quadric_gamma,
        quadric_kappa_max=cfg.quadric_kappa_max,
        quadric_eps_lambda=cfg.quadric_eps_lambda,
        quadric_eps_kappa=cfg.quadric_eps_kappa,
        quadric_eps_s=cfg.quadric_eps_s,
        quadric_eps_s3=cfg.quadric_eps_s3,
    )


def _make_dynamic_anchor_builder(cfg) -> DynamicVoxelAnchorBuilder:
    return DynamicVoxelAnchorBuilder(
        k_min=cfg.knn_k_min,
        k_target=cfg.knn_k_target,
        residual_threshold=cfg.anchor_residual_threshold,
        filter_mode=cfg.anchor_filter_mode,
        planarity_threshold=cfg.anchor_planarity_threshold,
        token_variant=cfg.anchor_token_variant,
        knn_chunk_size=min(cfg.knn_chunk_size, 256),
        quadric_gamma=cfg.quadric_gamma,
        quadric_kappa_max=cfg.quadric_kappa_max,
        quadric_eps_lambda=cfg.quadric_eps_lambda,
        quadric_eps_kappa=cfg.quadric_eps_kappa,
        quadric_eps_s=cfg.quadric_eps_s,
        quadric_eps_s3=cfg.quadric_eps_s3,
    )


def _transform_primitives(primitives: dict, T: torch.Tensor) -> dict:
    R = quat_to_rotmat(primitives["rotations"])
    R_out = T[:3, :3] @ R
    t = T[:3, 3]
    means = (T[:3, :3] @ primitives["means3D"].T).T + t
    out = dict(primitives)
    out["means3D"] = means
    out["rotations"] = rotmat_to_quat(R_out)
    return out


def _box_to_pose(box: torch.Tensor) -> torch.Tensor:
    c = torch.cos(box[6])
    s = torch.sin(box[6])
    pose = torch.eye(4, device=box.device, dtype=box.dtype)
    pose[0, 0] = c
    pose[0, 1] = -s
    pose[1, 0] = s
    pose[1, 1] = c
    pose[0, 3] = box[0]
    pose[1, 3] = box[1]
    pose[2, 3] = box[2]
    return pose


def _box1_to_frame0_pose(box: torch.Tensor, rel_input_1_pose: torch.Tensor) -> torch.Tensor:
    rel = rel_input_1_pose.to(device=box.device, dtype=box.dtype)
    return rel @ _box_to_pose(box)


def _concat_primitives(primitives: list[dict]) -> dict:
    if not primitives:
        return {}
    keys = primitives[0].keys()
    out = {}
    for key in keys:
        values = [p[key] for p in primitives]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = values
    return out


@torch.no_grad()
def _primitive_residual_stats(primitives: list[dict]) -> dict[str, torch.Tensor]:
    rows = []
    weights = []
    for prim in primitives:
        aux = prim.get("aux", {})
        if not aux:
            continue
        n = int(prim["means3D"].shape[0])
        if n == 0:
            continue
        omega_local_deg = aux["omega_local"][0] * (180.0 / torch.pi)
        row = {
            "diag_omega_deg": omega_local_deg.norm(dim=-1).mean(),
            "diag_tilt_deg": omega_local_deg[..., :2].norm(dim=-1).mean(),
            "diag_spin_deg": omega_local_deg[..., 2].abs().mean(),
            "diag_delta_c": aux["delta_c"][0].norm(dim=-1).mean(),
            "diag_delta_mu": aux["delta_mu"][0].abs().mean(),
            "diag_delta_mu_sign": aux["delta_mu"][0].mean(),
            "diag_delta_gap": aux["delta_gap"][0].abs().mean(),
            "diag_delta_gap_sign": aux["delta_gap"][0].mean(),
            "diag_delta_s3": aux["delta_log_abs_s3"][0].abs().mean(),
            "diag_delta_s3_sign": aux["delta_log_abs_s3"][0].mean(),
            "diag_delta_int": aux["delta_logit_intensity"][0].abs().mean(),
        }
        rows.append(row)
        weights.append(n)

    if not rows:
        return {}

    total = float(sum(weights))
    return {
        key: torch.stack([
            row[key] * (weight / total)
            for row, weight in zip(rows, weights)
        ]).sum()
        for key in rows[0]
    }


class Point2GaussianModel(nn.Module):
    """Raw pair batch -> differentiable QGS loss dictionary."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.primitive_model = QGSPrimitiveModel(cfg)
        self.loss_fn = QGSLoss(
            w_depth=getattr(cfg, "loss_w_range", 1.0),
            w_intensity=getattr(cfg, "loss_w_intensity", 0.1),
            w_raydrop=getattr(cfg, "loss_w_raydrop", 0.1),
            w_distortion=getattr(cfg, "loss_w_distortion", 0.05),
            w_normal=getattr(cfg, "loss_w_normal", 0.05),
            alpha_eps=getattr(cfg, "loss_alpha_eps", 0.5),
        )

    def forward(self, batch, batch_idx: int | None = None, mode: str = "train") -> dict:
        device = next(self.primitive_model.parameters()).device
        ray_dir = make_lidar_ray_grid(self.cfg, device=device)
        batch_size = len(batch["input_0"])

        total_loss = None
        loss_sums: dict[str, torch.Tensor] = {}
        valid_pairs = 0

        for idx in range(batch_size):
            out = self._process_pair(batch, idx=idx, device=device, ray_dir=ray_dir)
            if out is None or not torch.isfinite(out["total"]):
                continue
            valid_pairs += 1
            total_loss = out["total"] if total_loss is None else total_loss + out["total"]
            for key, value in out.items():
                if torch.is_tensor(value):
                    detached = value.detach()
                    loss_sums[key] = detached if key not in loss_sums else loss_sums[key] + detached

        if valid_pairs == 0:
            zero = next(self.primitive_model.parameters()).sum() * 0.0
            return {
                "total": zero,
                "losses": {"total": zero.detach()},
                "num_valid_pairs": 0,
            }

        assert total_loss is not None
        scale = 1.0 / float(valid_pairs)
        losses = {key: value * scale for key, value in loss_sums.items()}
        return {
            "total": total_loss * scale,
            "losses": losses,
            "num_valid_pairs": valid_pairs,
        }

    def _process_pair(
        self,
        batch,
        *,
        idx: int,
        device: torch.device,
        ray_dir: torch.Tensor,
    ) -> dict | None:
        cfg = self.cfg
        model = self.primitive_model
        loss_fn = self.loss_fn

        p0 = _batch_item(batch, "input_0", idx).to(device)
        p1 = _batch_item(batch, "input_1", idx).to(device)
        rel_input_1_pose = _ensure_2d_pose(_batch_item(batch, "input_1_pose", idx)).to(device)

        xyz0 = p0[:, :3].float()
        i0_raw = p0[:, 3].float()
        i0 = (i0_raw / 255.0).clamp(0.0, 1.0)
        ring0 = p0[:, 4].to(torch.int64) if p0.shape[1] >= 5 else None
        xyz1 = p1[:, :3].float()
        i1_raw = p1[:, 3].float()
        i1 = (i1_raw / 255.0).clamp(0.0, 1.0)
        ring1 = p1[:, 4].to(torch.int64) if p1.shape[1] >= 5 else None

        boxes_0 = (
            _batch_item(batch, "boxes_0", idx).to(device)
            if "boxes_0" in batch
            else torch.empty(0, 7, device=device)
        )
        boxes_1 = (
            _batch_item(batch, "boxes_1", idx).to(device)
            if "boxes_1" in batch
            else torch.empty(0, 7, device=device)
        )
        instance_ids_0 = (
            _batch_item(batch, "instance_ids_0", idx).to(device)
            if "instance_ids_0" in batch
            else torch.empty(0, dtype=torch.long, device=device)
        )
        instance_ids_1 = (
            _batch_item(batch, "instance_ids_1", idx).to(device)
            if "instance_ids_1" in batch
            else torch.empty(0, dtype=torch.long, device=device)
        )

        if ring0 is not None:
            xyz0, i0, ring0 = _filter_frame_points(xyz0, i0, cfg, ring=ring0)
            xyz1, i1, ring1 = _filter_frame_points(xyz1, i1, cfg, ring=ring1)
        else:
            xyz0, i0 = _filter_frame_points(xyz0, i0, cfg)
            xyz1, i1 = _filter_frame_points(xyz1, i1, cfg)
        if xyz0.shape[0] < cfg.knn_k_min or xyz1.shape[0] < cfg.knn_k_min:
            return None

        if boxes_0.numel() and boxes_1.numel():
            scene = decompose_scene(
                xyz0,
                xyz1,
                i0,
                i1,
                boxes_0,
                boxes_1,
                instance_ids_0,
                instance_ids_1,
                rel_input_1_pose,
            )
        else:
            p1_in_frame0 = (rel_input_1_pose[:3, :3] @ xyz1.T).T + rel_input_1_pose[:3, 3]
            scene = {
                "static_xyz": torch.cat([xyz0, p1_in_frame0], dim=0),
                "static_intensity": torch.cat([i0, i1], dim=0),
                "static_time": torch.cat([
                    torch.zeros(xyz0.shape[0], device=device, dtype=xyz0.dtype),
                    torch.ones(xyz1.shape[0], device=device, dtype=xyz1.dtype),
                ], dim=0),
                "dynamic": [],
                "untracked_stats": {},
            }

        inv_pose = torch.linalg.inv(rel_input_1_pose)
        frame0_primitives = []
        frame1_primitives = []
        diagnostic_primitives = []

        dyn_list = scene["dynamic"]
        dyn_anchor_outputs = []
        if dyn_list:
            dynamic_builder = _make_dynamic_anchor_builder(cfg)
            dyn_anchor_pairs = dynamic_builder(dyn_list)
            dyn_anchor_outputs = [out for _, out in dyn_anchor_pairs]
            dyn_prims_list = model.forward_anchor_contexts_batched(
                dyn_anchor_outputs,
                context_type="dynamic",
            )
        else:
            dyn_prims_list = []

        static_xyz_parts = [scene["static_xyz"].to(device)]
        static_i_parts = [scene["static_intensity"].to(device)]
        static_t_parts = [scene["static_time"].to(device)]
        for dyn, dyn_anchor_out, dyn_prims in zip(dyn_list, dyn_anchor_outputs, dyn_prims_list):
            if dyn_prims is None or dyn_anchor_out.c_init.shape[0] == 0:
                static_xyz_parts.append(dyn["fallback_xyz"].to(device))
                static_i_parts.append(dyn["fallback_intensity"].to(device))
                static_t_parts.append(dyn["fallback_time"].to(device))
                continue
            diagnostic_primitives.append(dyn_prims)
            if dyn.get("box_0") is not None:
                frame0_primitives.append(
                    _transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device)))
                )
            if dyn.get("box_1") is not None:
                frame1_primitives.append(
                    _transform_primitives(
                        dyn_prims,
                        _box1_to_frame0_pose(dyn["box_1"].to(device), rel_input_1_pose),
                    )
                )

        static_xyz = torch.cat(static_xyz_parts, dim=0)
        static_i = torch.cat(static_i_parts, dim=0)
        static_t = torch.cat(static_t_parts, dim=0)
        static_anchor_out = _make_static_anchor_builder(cfg)(
            static_xyz,
            static_i,
            static_t,
            pose_frame1_in_frame0=rel_input_1_pose,
        )
        static_prims = model.forward_anchor_context(static_anchor_out, context_type="static")
        if static_prims is not None:
            diagnostic_primitives.append(static_prims)
            frame0_primitives.append(static_prims)
            frame1_primitives.append(static_prims)

        frame0 = _concat_primitives(frame0_primitives)
        frame1 = _concat_primitives(frame1_primitives)
        if not frame0 or not frame1:
            return None

        target0 = points_to_lidar_maps(xyz0, i0, cfg, ring=ring0)
        target0.update(build_gt_normal_map(target0["range_image"], target0["valid_mask"], ray_dir))
        target1 = points_to_lidar_maps(xyz1, i1, cfg, ring=ring1)
        target1.update(build_gt_normal_map(target1["range_image"], target1["valid_mask"], ray_dir))

        rendered0 = render_primitives(
            frame0,
            cfg,
            viewmatrix=torch.eye(4, device=device, dtype=static_xyz.dtype),
            campos=torch.zeros(3, device=device, dtype=static_xyz.dtype),
        )
        rendered1 = render_primitives(
            frame1,
            cfg,
            viewmatrix=inv_pose,
            campos=rel_input_1_pose[:3, 3],
        )

        loss0 = loss_fn(rendered0, target0, rendered0.raydrop, ray_dir)
        loss1 = loss_fn(rendered1, target1, rendered1.raydrop, ray_dir)
        total = loss0["total"] + loss1["total"]
        loss_dict = {
            "total": total,
            "depth": 0.5 * (loss0["depth"] + loss1["depth"]),
            "depth_range": 0.5 * (loss0["depth_range"] + loss1["depth_range"]),
            "depth_median": 0.5 * (loss0["depth_median"] + loss1["depth_median"]),
            "intensity": 0.5 * (loss0["intensity"] + loss1["intensity"]),
            "raydrop": 0.5 * (loss0["raydrop"] + loss1["raydrop"]),
            "distortion": 0.5 * (loss0["distortion"] + loss1["distortion"]),
            "normal": 0.5 * (loss0["normal"] + loss1["normal"]),
            "n_valid": 0.5 * (loss0["n_valid"] + loss1["n_valid"]),
            "coverage": 0.5 * (loss0["coverage"] + loss1["coverage"]),
            "raydrop_hit": 0.5 * (loss0["raydrop_hit"] + loss1["raydrop_hit"]),
            "raydrop_miss": 0.5 * (loss0["raydrop_miss"] + loss1["raydrop_miss"]),
            "drop_prob_hit_mean": 0.5 * (loss0["drop_prob_hit_mean"] + loss1["drop_prob_hit_mean"]),
            "drop_prob_miss_mean": 0.5 * (loss0["drop_prob_miss_mean"] + loss1["drop_prob_miss_mean"]),
            "alpha_hit_mean": 0.5 * (loss0["alpha_hit_mean"] + loss1["alpha_hit_mean"]),
            "alpha_miss_mean": 0.5 * (loss0["alpha_miss_mean"] + loss1["alpha_miss_mean"]),
            "intensity_gt_mean": 0.5 * (loss0["intensity_gt_mean"] + loss1["intensity_gt_mean"]),
            "intensity_pred_mean": 0.5 * (loss0["intensity_pred_mean"] + loss1["intensity_pred_mean"]),
        }
        loss_dict.update(_primitive_residual_stats(diagnostic_primitives))
        return loss_dict
