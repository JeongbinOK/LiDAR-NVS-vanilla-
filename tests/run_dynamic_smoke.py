"""Meaningful one-GPU Dynamic 2DGS integration/overfit smoke.

This is intentionally stronger than a shape-only forward test: it repeats one
fixed real nuScenes batch, verifies renderer gradients reach both the temporal
and velocity branches, checks physical-time motion becomes non-zero, and fails
unless the actual render loss decreases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.config_loader import DYNAMIC_VARIANT, compose_fresh_config
from src.dataloader.nuscene import NuScenesNVSDataset, multiframe_collate_fn
from src.model_wrapper import ModelWrapper


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def forward_loss(model, batch):
    source, target = batch["input"], batch["gt"]
    p2g = model.p2g_model(
        source,
        target_pose=target.get("pose"),
        target_timestamps=target.get("timestamps"),
    )
    p2g.pop("routing_stats", None)
    p2g.pop("routing_budget_logits", None)
    gaussians = model.g2g_model(
        p2g,
        source["timestamps_sec"],
        source["window_duration_sec"],
    )
    renders = model.g2p_model(
        gaussians,
        target,
        compute_points=float(model.cfg.loss.w_chamfer) > 0.0,
    )
    losses = model.loss(
        renders,
        gaussians=gaussians,
        metric_mode="train",
        compute_valid_metrics=False,
        compute_raydrop_metrics=False,
    )
    return losses, gaussians


def forward_memory_probe(model, batch):
    """Keep the full P2G graph while avoiding rasterizer-build dependencies."""
    source, target = batch["input"], batch["gt"]
    p2g = model.p2g_model(
        source,
        target_pose=target.get("pose"),
        target_timestamps=target.get("timestamps"),
    )
    p2g.pop("routing_stats", None)
    p2g.pop("routing_budget_logits", None)
    gaussians = model.g2g_model(
        p2g,
        source["timestamps_sec"],
        source["window_duration_sec"],
    )
    terms = []
    for item in gaussians:
        terms.extend([
            item["shs"].square().mean(),
            item["opacity"].square().mean(),
            item["scaling"].square().mean(),
            (item["velocity"] - 0.1).square().mean(),
        ])
    total = torch.stack(terms).sum()
    zero = total.detach().new_zeros(())
    return {
        "total": total,
        "loss_depth": zero,
        "loss_depth_median": zero,
        "loss_intensity": zero,
        "loss_raydrop": zero,
        "loss_scale": zero,
    }, gaussians


def grad_norm(parameter):
    grad = parameter.grad
    if grad is None:
        return 0.0
    return float(grad.float().norm().item())


def render_objective_by_time(model, renders):
    """Renderer terms split into endpoints versus temporal novel views."""
    pred_depth = renders["depth"].squeeze(2)
    pred_median = renders["depth_median"].squeeze(2)
    pred_intensity = renders["intensity_sh"].squeeze(2)
    pred_raydrop = renders["raydrop"].squeeze(2)
    gt_depth = renders["gt_depth"].squeeze(2)
    gt_intensity = renders["gt_intensity_sh"].squeeze(2)
    gt_raydrop = renders["gt_raydrop"].squeeze(2)
    cfg = model.cfg.loss
    objectives = []
    for view in range(pred_depth.shape[1]):
        valid = gt_depth[:, view] > 0
        depth = F.l1_loss(pred_depth[:, view][valid], gt_depth[:, view][valid])
        median = F.l1_loss(pred_median[:, view][valid], gt_depth[:, view][valid])
        intensity = F.l1_loss(
            pred_intensity[:, view][valid], gt_intensity[:, view][valid]
        )
        raydrop = F.binary_cross_entropy(
            pred_raydrop[:, view].clamp(1.0e-7, 1.0 - 1.0e-7),
            gt_raydrop[:, view],
        )
        objectives.append(
            float(
                cfg.w_depth * depth
                + cfg.w_depth_median * median
                + cfg.w_intensity * intensity
                + cfg.w_raydrop * raydrop
            )
        )
    endpoints = 0.5 * (objectives[0] + objectives[-1])
    middle = sum(objectives[1:-1]) / max(len(objectives) - 2, 1)
    return {"views": objectives, "endpoint_mean": endpoints, "middle_mean": middle}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default=DYNAMIC_VARIANT,
        help="model.variant to smoke; defaults to the fresh-run baseline",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--output", default="logs/dynamic_smoke.json")
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="run the requested steps and report CUDA peaks without overfit assertions",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "extra OmegaConf dot-list overrides, e.g. "
            "p2g.grid_query.K_max=2 p2g.grid_query.exp=1"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Dynamic 2DGS smoke")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    cli = OmegaConf.from_dotlist([
        f"model.variant={args.variant}",
        f"device=[{args.device}]",
        f"train.batch_size={args.batch_size}",
        "data.num_workers=0",
        "data.eval_num_workers=0",
        # Dynamic V7 never consumes boxes. Null also makes the smoke portable to
        # machines without the optional predicted-tracking JSON; the loader may
        # still expose native GT boxes for diagnostics.
        "data.bbox_json_path=null",
        "logger.enable=false",
        f"dynamic_2dgs.temporal.layers={args.attention_layers}",
        *args.override,
    ])
    cfg, _ = compose_fresh_config(cli)
    dataset = NuScenesNVSDataset(cfg.data, split=args.split)
    indices = args.indices or list(range(args.batch_size))
    if len(indices) != args.batch_size:
        raise ValueError("--indices must provide exactly --batch-size entries")
    items = [dataset[index] for index in indices]
    batch = to_device(multiframe_collate_fn(items), device)

    model = ModelWrapper(cfg).to(device).train()
    p2g = model.p2g_model
    if not p2g.utonia_lora_enabled:
        raise RuntimeError("dynamic smoke requires Utonia XYZI LoRA")
    if any(module is not None for module in (
        p2g.anchor_builder.intensity_encoder,
        p2g.utonia_adapter,
        p2g.intensity_agg_mlp,
        p2g.joint_refiner,
    )):
        raise RuntimeError(
            "Utonia LoRA path constructed a legacy intensity/fusion/refiner module"
        )
    utonia_named_parameters = list(
        p2g.feature_extractor.named_parameters()
    )
    trainable_utonia_parameters = [
        (name, parameter)
        for name, parameter in utonia_named_parameters
        if parameter.requires_grad
    ]
    trainable_utonia_count = sum(
        parameter.numel() for _, parameter in trainable_utonia_parameters
    )
    if trainable_utonia_count != p2g.utonia_lora_trainable_parameters:
        raise RuntimeError(
            "active Utonia LoRA parameter count does not match construction report"
        )
    unexpected_utonia_trainable = [
        name for name, _ in trainable_utonia_parameters
        if ".lora_down." not in name and ".lora_up." not in name
    ]
    if unexpected_utonia_trainable:
        raise RuntimeError(
            "non-LoRA Utonia parameters are trainable: "
            + ", ".join(unexpected_utonia_trainable[:5])
        )
    lora_up_name, lora_up_parameter = next(
        (name, parameter)
        for name, parameter in utonia_named_parameters
        if ".lora_up." in name and parameter.requires_grad
    )
    frozen_base_name, frozen_base_parameter = next(
        (name, parameter)
        for name, parameter in utonia_named_parameters
        if ".base_layer." in name
    )
    if frozen_base_parameter.requires_grad:
        raise RuntimeError(f"Utonia base parameter is trainable: {frozen_base_name}")
    parameters = [
        parameter for parameter in p2g.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=float(cfg.train.weight_decay),
    )
    temporal_weight = model.p2g_model.dynamic_backend.temporal.q_proj[0].weight
    dynamic_backend = model.p2g_model.dynamic_backend
    proposal_module = getattr(dynamic_backend, "motion_proposal", None)
    search_speed = (
        getattr(proposal_module, "search_speed", None)
        if proposal_module is not None else None
    )
    proposal_weight = search_speed.weight if search_speed is not None else None
    velocity_weight = (
        model.p2g_model.dynamic_backend.velocity_head.velocity.weight
    )
    dustbin_predictor = (
        getattr(proposal_module, "dustbin_predictor", None)
        if proposal_module is not None else None
    )
    dustbin_predictor_weight = (
        dustbin_predictor[-1].weight
        if dustbin_predictor is not None else None
    )
    count_predictor = getattr(
        getattr(dynamic_backend, "gaussian_head", None),
        "count_predictor",
        None,
    )
    count_router_weight = (
        count_predictor[-1].weight if count_predictor is not None else None
    )

    torch.cuda.reset_peak_memory_stats(device)
    records = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if args.memory_only:
            losses, gaussians = forward_memory_probe(model, batch)
        else:
            losses, gaussians = forward_loss(model, batch)
        total = losses["total"]
        total.backward()
        unused_lora = [
            name for name, parameter in trainable_utonia_parameters
            if parameter.grad is None
        ]
        if unused_lora:
            raise RuntimeError(
                "active Utonia LoRA parameters were unused: "
                + ", ".join(unused_lora[:5])
            )
        temporal_grad = grad_norm(temporal_weight)
        proposal_grad = (
            grad_norm(proposal_weight) if proposal_weight is not None else None
        )
        velocity_grad = grad_norm(velocity_weight)
        dustbin_predictor_grad = (
            grad_norm(dustbin_predictor_weight)
            if dustbin_predictor_weight is not None else None
        )
        count_router_grad = (
            grad_norm(count_router_weight)
            if count_router_weight is not None else None
        )
        lora_up_grad = grad_norm(lora_up_parameter)
        frozen_base_grad = grad_norm(frozen_base_parameter)
        torch.nn.utils.clip_grad_norm_(parameters, float(cfg.train.grad_clip))
        optimizer.step()

        with torch.no_grad():
            speeds = torch.cat([item["velocity"].norm(dim=-1) for item in gaussians])
            middle_time = float(batch["gt"]["timestamps_sec"][0][
                len(batch["gt"]["timestamps_sec"][0]) // 2
            ])
            moved = model.g2p_model.get_means3D(gaussians[0], middle_time)
            displacement = (
                moved - gaussians[0]["position"]
            ).norm(dim=-1).mean()
            proposal_stats = {}
            for field in (
                "p_unmatched",
                "match_probability",
                "motion_top1_probability",
                "motion_effective_support",
                "motion_reciprocal_probability",
                "motion_selected_forward_probability",
                "motion_consensus_entropy",
                "motion_head_js_divergence",
                "motion_candidate_probability_mass",
                "motion_selection_log_margin",
                "motion_hard_soft_displacement_cosine",
                "motion_hard_soft_displacement_norm_ratio",
                "motion_selected_distance_prior_penalty",
                "motion_match_support",
                "motion_candidate_entropy",
                "motion_best_candidate_displacement_m",
                "motion_reciprocal_probability_sum",
                "motion_mean_displacement_m",
                "motion_candidate_spread_m",
                "motion_normalized_mean_displacement",
                "motion_normalized_candidate_spread",
                "motion_soft_match_probability",
                "motion_hard_reject",
                "motion_search_speed_mps",
            ):
                if all(field in item for item in gaussians):
                    value = torch.cat([item[field] for item in gaussians]).float()
                    if field in (
                        "motion_best_candidate_displacement_m",
                        "motion_mean_displacement_m",
                    ):
                        value = value.norm(dim=-1)
                        field = field.removesuffix("_m") + "_magnitude"
                    proposal_stats[f"{field}_mean"] = float(value.mean())
            record = {
                "step": step,
                "total": float(total.detach()),
                "depth": float(losses["loss_depth"].detach()),
                "depth_median": float(losses["loss_depth_median"].detach()),
                "intensity": float(losses["loss_intensity"].detach()),
                "raydrop": float(losses["loss_raydrop"].detach()),
                "scale": float(losses["loss_scale"].detach()),
                "temporal_grad_norm": temporal_grad,
                "proposal_grad_norm": proposal_grad,
                "velocity_grad_norm": velocity_grad,
                "dustbin_predictor_grad_norm": dustbin_predictor_grad,
                "count_router_grad_norm": count_router_grad,
                "utonia_lora_up_grad_norm": lora_up_grad,
                "utonia_frozen_base_grad_norm": frozen_base_grad,
                "velocity_speed_mean_mps": float(speeds.mean()),
                "velocity_speed_max_mps": float(speeds.max()),
                "middle_displacement_mean_m": float(displacement),
                "gaussians": int(sum(item["position"].shape[0] for item in gaussians)),
                **proposal_stats,
            }
            records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    torch.cuda.synchronize(device)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    if args.memory_only:
        if not all(item["utonia_lora_up_grad_norm"] > 0.0 for item in records):
            raise RuntimeError("memory probe did not reach Utonia LoRA")
        if any(
            item["utonia_frozen_base_grad_norm"] != 0.0
            for item in records
        ):
            raise RuntimeError(
                "memory probe unexpectedly reached frozen Utonia base"
            )
        if dustbin_predictor_weight is not None and not all(
            item["dustbin_predictor_grad_norm"] > 0.0 for item in records
        ):
            raise RuntimeError(
                "memory probe did not reach evidence dustbin predictor"
            )
        if count_router_weight is not None and not all(
            item["count_router_grad_norm"] > 0.0 for item in records
        ):
            raise RuntimeError("memory probe did not reach adaptive-K router")
        print(json.dumps({
            "attention_layers": args.attention_layers,
            "batch_size": args.batch_size,
            "gaussians": records[-1]["gaussians"],
            "peak_memory_allocated_gib": peak_gib,
            "peak_memory_reserved_gib": peak_reserved_gib,
            "temporal_grad_norm": records[-1]["temporal_grad_norm"],
            "proposal_grad_norm": records[-1]["proposal_grad_norm"],
            "velocity_grad_norm": records[-1]["velocity_grad_norm"],
            "dustbin_predictor_grad_norm": (
                records[-1]["dustbin_predictor_grad_norm"]
            ),
            "count_router_grad_norm": records[-1]["count_router_grad_norm"],
            "utonia_lora_parameter": lora_up_name,
            "utonia_lora_up_grad_norm": (
                records[-1]["utonia_lora_up_grad_norm"]
            ),
            "utonia_frozen_base_parameter": frozen_base_name,
            "utonia_frozen_base_grad_norm": (
                records[-1]["utonia_frozen_base_grad_norm"]
            ),
        }, sort_keys=True), flush=True)
        return
    initial = records[0]["total"]
    final = records[-1]["total"]
    best = min(item["total"] for item in records)
    with torch.no_grad():
        final_losses, final_gaussians = forward_loss(model, batch)
        final_renders = model.g2p_model(
            final_gaussians, batch["gt"], compute_points=False
        )
        final_by_time = render_objective_by_time(model, final_renders)
        zero_velocity_gaussians = [
            {
                **item,
                "velocity": torch.zeros_like(item["velocity"]),
            }
            for item in final_gaussians
        ]
        zero_renders = model.g2p_model(
            zero_velocity_gaussians, batch["gt"], compute_points=False
        )
        zero_losses = model.loss(
            zero_renders,
            gaussians=zero_velocity_gaussians,
            metric_mode="train",
            compute_valid_metrics=False,
            compute_raydrop_metrics=False,
        )
        zero_by_time = render_objective_by_time(model, zero_renders)
        final_post_update = float(final_losses["total"])
        zero_velocity_total = float(zero_losses["total"])
        temporal = model.p2g_model.dynamic_backend.temporal
        saved_scales = [scale.detach().clone() for scale in temporal.cross_layer_scale]
        for scale in temporal.cross_layer_scale:
            scale.zero_()
        no_cross_losses, _no_cross_gaussians = forward_loss(model, batch)
        no_cross_total = float(no_cross_losses["total"])
        for scale, saved in zip(temporal.cross_layer_scale, saved_scales):
            scale.copy_(saved)

    summary = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "attention_layers": args.attention_layers,
        "dataset_indices": indices,
        "initial_total": initial,
        "final_total": final,
        "best_total": best,
        "relative_final": final / initial,
        "post_update_total": final_post_update,
        "zero_velocity_total": zero_velocity_total,
        "velocity_ablation_delta": zero_velocity_total - final_post_update,
        "full_objective_by_time": final_by_time,
        "zero_velocity_objective_by_time": zero_by_time,
        "velocity_middle_ablation_delta": (
            zero_by_time["middle_mean"] - final_by_time["middle_mean"]
        ),
        "velocity_endpoint_ablation_delta": (
            zero_by_time["endpoint_mean"] - final_by_time["endpoint_mean"]
        ),
        "no_cross_total": no_cross_total,
        "cross_ablation_delta": no_cross_total - final_post_update,
        "peak_memory_gib": peak_gib,
        "records": records,
    }
    if not all(item["temporal_grad_norm"] > 0.0 for item in records):
        raise RuntimeError("render loss did not reach temporal cross-attention")
    if proposal_weight is not None and not all(
        item["proposal_grad_norm"] > 0.0 for item in records
    ):
        raise RuntimeError("render loss did not reach the motion proposal")
    if not all(item["velocity_grad_norm"] > 0.0 for item in records):
        raise RuntimeError("render loss did not reach the velocity output head")
    if dustbin_predictor_weight is not None and not all(
        item["dustbin_predictor_grad_norm"] > 0.0 for item in records
    ):
        raise RuntimeError("render loss did not reach evidence dustbin predictor")
    if count_router_weight is not None and not all(
        item["count_router_grad_norm"] > 0.0 for item in records
    ):
        raise RuntimeError("render loss did not reach adaptive-K router")
    if not all(item["utonia_lora_up_grad_norm"] > 0.0 for item in records):
        raise RuntimeError("render loss did not reach Utonia LoRA")
    if any(item["utonia_frozen_base_grad_norm"] != 0.0 for item in records):
        raise RuntimeError("render loss unexpectedly reached frozen Utonia base")
    if records[-1]["velocity_speed_mean_mps"] <= 1.0e-5:
        raise RuntimeError("velocity stayed identically zero after optimization")
    if records[-1]["middle_displacement_mean_m"] <= 1.0e-6:
        raise RuntimeError("physical-time transport stayed inactive")
    if not final < initial:
        raise RuntimeError(
            f"fixed-batch render loss did not decrease: {initial} -> {final}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
