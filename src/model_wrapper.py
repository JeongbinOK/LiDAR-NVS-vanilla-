import torch
import json
import math
from pathlib import Path
from omegaconf import OmegaConf

from src.models_new.module import (
    Point2Gaus,
    GausTemp,
    GausRender,
    DynamicGausTemp,
    DynamicGausRender,
)
from lightning.pytorch import LightningModule
from src.models_new.utils.loss import (
    Loss,
    budget_weight_at_step,
    distributed_token_mean,
)
from src.models_new.utils.routing_logging import (
    distributed_sum_statistics,
    range_bin_labels,
    routing_sufficient_statistics,
)
from src.config_loader import DYNAMIC_VARIANTS, LEGACY_VARIANT

WANDB_BASE_LOSS_KEYS = {
    "loss_depth",
    "loss_depth_median",
    "loss_intensity",
    "loss_raydrop",
    "loss_scale",
    "total",
    # Reconstruction quality. Keys absent from a given step's loss dict are
    # skipped by _log_losses, so these follow whatever schedule produced them:
    # chamfer is computed for val/test (and for train only when w_chamfer>0),
    # while the PSNR/SSIM pairs follow _metric_schedule.
    "loss_chamfer",
    "intensity_psnr_valid",
    "intensity_ssim_valid",
    "depth_psnr_valid",
    "depth_ssim_valid",
    "intensity_psnr_raydrop",
    "intensity_ssim_raydrop",
    "depth_psnr_raydrop",
    "depth_ssim_raydrop",
}

GLOBAL_REDUCED_LOSS_KEYS = {
    "loss_budget",
    "budget_expected_mean_k",
    "budget_violation",
    "loss_velocity_l2",
}

ADAPTIVE_BUDGET_LOG_KEYS = {
    "loss_budget",
    "budget_expected_mean_k",
    "budget_violation",
}


class ModelWrapper(LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        model_cfg = getattr(cfg, "model", None)
        self.model_variant = str(
            getattr(model_cfg, "variant", LEGACY_VARIANT)
            if model_cfg is not None else LEGACY_VARIANT
        )
        if self.model_variant not in (LEGACY_VARIANT, *DYNAMIC_VARIANTS):
            raise ValueError(f"Unsupported model.variant={self.model_variant!r}")
        #self.optimizer_cfg = cfg.optimizer
        self.p2g_cfg = cfg.p2g
        self.g2g_cfg = cfg.g2g
        self.g2p_cfg = cfg.g2p

        # Set up the model.
        if self.model_variant in DYNAMIC_VARIANTS:
            self.dynamic_cfg = cfg.dynamic_2dgs
            self.p2g_model = Point2Gaus(
                self.p2g_cfg,
                dynamic_cfg=self.dynamic_cfg,
                dynamic_variant=self.model_variant,
            )
            self.g2g_model = DynamicGausTemp(self.g2g_cfg)
            self.g2p_model = DynamicGausRender(self.g2p_cfg)
        else:
            self.dynamic_cfg = None
            self.p2g_model = Point2Gaus(self.p2g_cfg)
            self.g2g_model = GausTemp(self.g2g_cfg)
            self.g2p_model = GausRender(self.g2p_cfg)
        self.loss = Loss(self.cfg.loss)
        self._eval_pair_summaries: dict[str, list[dict]] = {}
        self._metric_interval = int(self._cfg_get("metrics.interval", 50))
        if self._metric_interval <= 0:
            raise ValueError("metrics.interval must be positive")
        # Gradient accumulation repeats one global_step across several batches;
        # this makes the sparse train metric fire once per optimizer step.
        self._last_train_metric_step = -1
        anchor_mode = str(
            self._cfg_get("p2g.anchor_mode", "spherical")
        ).lower()
        routing_block = (
            "p2g.squery" if anchor_mode == "spherical" else "p2g.grid_query"
        )
        self._routing_unit_name = (
            "anchor" if anchor_mode == "spherical" else "token"
        )
        learned_count_block = f"{routing_block}.learned_count"
        routing_count_mode = str(
            self._cfg_get(f"{routing_block}.count_mode", "legacy")
        ).lower()
        self._routing_logging_enable = bool(self._cfg_get(
            f"{learned_count_block}.logging.enable", False
        ))
        self._routing_logging_interval = int(self._cfg_get(
            f"{learned_count_block}.logging.interval", 20
        ))
        self._routing_range_edges = tuple(float(value) for value in self._cfg_get(
            f"{learned_count_block}.logging.range_edges_m",
            (0, 10, 20, 30, 40, 60, 80, 110),
        ))
        if self._routing_logging_interval <= 0:
            raise ValueError(
                "learned_count.logging.interval must be positive"
            )
        # Validates lower-bound ordering once, before the first train batch.
        range_bin_labels(self._routing_range_edges)
        self._last_train_routing_step = -1
        budget_requested = bool(self._cfg_get(
            f"{learned_count_block}.budget.enable", False
        ))
        self._budget_enable = (
            budget_requested
            and routing_count_mode in ("learned_gumbel", "learned_gumbel_viewpt")
        )
        self._budget_target_mean_k = float(self._cfg_get(
            f"{learned_count_block}.budget.target_mean_k", 2.0
        ))
        self._budget_weight = float(self._cfg_get(
            f"{learned_count_block}.budget.weight", 1.0
        ))
        self._budget_warmup_steps = int(self._cfg_get(
            f"{learned_count_block}.budget.warmup_steps", 500
        ))
        self._budget_ramp_steps = int(self._cfg_get(
            f"{learned_count_block}.budget.ramp_steps", 1000
        ))
        if self._budget_enable:
            budget_k_max = int(self._cfg_get(
                f"{learned_count_block}.K_max", 0
            ))
            if not 1.0 <= self._budget_target_mean_k <= float(budget_k_max):
                raise ValueError(
                    "learned_count.budget.target_mean_k must be in "
                    f"[1, {budget_k_max}]"
                )
            if self._budget_weight < 0.0:
                raise ValueError(
                    "learned_count.budget.weight must be non-negative"
                )
            if self._budget_warmup_steps < 0 or self._budget_ramp_steps < 0:
                raise ValueError(
                    "learned_count.budget warmup_steps/ramp_steps must be "
                    "non-negative"
                )
        self._wandb_loss_keys = set(WANDB_BASE_LOSS_KEYS)
        if self._budget_enable:
            self._wandb_loss_keys.add("loss_budget")
        regularization = getattr(self.dynamic_cfg, "regularization", None)
        small_motion = getattr(regularization, "small_motion", None)
        if small_motion is not None and bool(getattr(small_motion, "enabled", False)):
            self._wandb_loss_keys.add("loss_velocity_prior")
        velocity_l2 = getattr(regularization, "velocity_l2", None)
        self._velocity_l2_cfg = (
            velocity_l2
            if velocity_l2 is not None and bool(getattr(velocity_l2, "enabled", False))
            else None
        )
        if self._velocity_l2_cfg is not None:
            if float(getattr(self._velocity_l2_cfg, "weight", 0.0)) < 0.0:
                raise ValueError("velocity_l2.weight must be non-negative")
            mode = str(getattr(self._velocity_l2_cfg, "mode", "final_l2")).lower()
            if mode not in ("final_l2", "final_group_l1"):
                raise ValueError(
                    "velocity_l2.mode must be 'final_l2' or 'final_group_l1'; "
                    f"got {mode!r}"
                )
            self._velocity_l2_mode = mode
            self._wandb_loss_keys.add("loss_velocity_l2")

    def on_save_checkpoint(self, checkpoint) -> None:
        """Bind the resolved experiment config immutably to each checkpoint."""

        checkpoint["experiment_config_schema_version"] = 1
        checkpoint["experiment_config"] = OmegaConf.to_container(
            self.cfg,
            resolve=True,
        )

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")


    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")


    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")


    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        p2g_out = self.p2g_model(
            _input,
            target_pose=gt.get("pose"),
            target_timestamps=gt.get("timestamps"),
        )
        # Keep detached diagnostics outside the renderer/temporal model input.
        routing_stats = p2g_out.pop("routing_stats", None)
        routing_budget_logits = p2g_out.pop("routing_budget_logits", None)
        if self.model_variant in DYNAMIC_VARIANTS:
            out = self.g2g_model(
                p2g_out,
                _input["timestamps_sec"],
                _input["window_duration_sec"],
            )
        else:
            out = self.g2g_model(p2g_out, _input["timestamps"])
        compute_train_points = float(self.cfg.loss.w_chamfer) > 0.0
        all_renders = self.g2p_model(
            out,
            gt,
            compute_points=(prefix != "train" or compute_train_points),
        )

        compute_valid_metrics, compute_official_metrics = self._metric_schedule(
            prefix, batch_idx
        )
        loss_dict = self.loss(
            all_renders,
            gaussians=out,
            routing_budget=self._routing_budget_spec(
                routing_budget_logits,
                prefix=prefix,
            ),
            metric_mode=prefix,
            compute_valid_metrics=compute_valid_metrics,
            compute_raydrop_metrics=compute_official_metrics,
        )
        if self.model_variant in DYNAMIC_VARIANTS:
            self._add_dynamic_motion_regularization(loss_dict, out, prefix=prefix)
        self._log_losses(loss_dict, prefix=prefix, batch_size=self._batch_size(_input))
        self._log_routing_stats(routing_stats, prefix=prefix)

        if prefix != "train":
            self._record_eval_summary(loss_dict, batch_idx=batch_idx, prefix=prefix)

        return loss_dict["total"]

    def _add_dynamic_motion_regularization(self, losses, gaussians, *, prefix):
        """Apply the configured motion priors and log runaway diagnostics."""
        self._add_small_motion_prior(losses, gaussians, prefix=prefix)
        self._add_velocity_l2_prior(losses, gaussians, prefix=prefix)
        self._log_velocity_statistics(gaussians, prefix=prefix)
        self._log_attention_temperature(prefix=prefix)

    def _log_attention_temperature(self, *, prefix: str) -> None:
        """Softmax-temperature alarm for the motion attention (V5 QK-Norm).

        Under QK-Norm ``|q| = gamma * sqrt(head_dim)`` exactly, so the norm gains
        alone bound the logit and no attention probabilities have to be built to
        watch for saturation. V4 collapsed to a hard argmax by epoch 2-9 with
        nothing logged; this is that missing alarm.
        """
        temporal = getattr(
            getattr(self.p2g_model, "dynamic_backend", None), "temporal", None
        )
        stats = temporal.attention_temperature_stats() if temporal is not None else None
        if not stats:
            return
        for name, value in stats.items():
            self.log(
                f"{prefix}/dynamic/{name}",
                float(value),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )

    @staticmethod
    def _active_motion_items(gaussians):
        return [
            item for item in gaussians
            if item is not None and item["velocity"].numel() > 0
        ]

    def _add_velocity_l2_prior(self, losses, gaussians, *, prefix):
        """Final-velocity magnitude prior, selected by ``velocity_l2.mode``.

        ``final_l2`` preserves the historical mean over squared components.
        ``final_group_l1`` is ``mean(||v||_2)``: an L1 penalty over velocity
        vectors that keeps a non-vanishing restoring force near zero. Both act
        on the rendered final velocity, never on proposal/fallback components.

        The raw term is logged in every mode, but only train adds it to the
        optimized total, so validation totals stay a pure reconstruction score.
        """
        cfg = self._velocity_l2_cfg
        if cfg is None:
            return

        mode = getattr(self, "_velocity_l2_mode", "final_l2")
        active = self._active_motion_items(gaussians)
        if active:
            velocity = torch.cat([item["velocity"] for item in active], dim=0)
            if mode == "final_l2":
                local_sum = velocity.square().sum()
                local_count = velocity.numel()
            else:
                local_sum = velocity.norm(dim=-1).sum()
                local_count = velocity.shape[0]
        else:
            # Keep a local graph edge while still entering the same all-reduce
            # as ranks that happened to receive occupied tokens.
            local_sum = losses["total"] * 0.0
            local_count = 0
        raw_loss, _global_count = distributed_token_mean(
            local_sum, local_count
        )
        losses["loss_velocity_l2"] = raw_loss

        if prefix != "train":
            return
        effective_weight = budget_weight_at_step(
            float(getattr(cfg, "weight", 0.0)),
            int(self.global_step),
            int(getattr(cfg, "warmup_steps", 0)),
            int(getattr(cfg, "ramp_steps", 0)),
        )
        weighted = raw_loss * effective_weight
        losses["wc_velocity_l2"] = weighted.detach()
        losses["total"] = losses["total"] + weighted

    @torch.no_grad()
    def _log_velocity_statistics(self, gaussians, *, prefix: str) -> None:
        """Motion-magnitude alarms; V3 diverged without these being visible."""
        active = self._active_motion_items(gaussians)
        if not active:
            # Unreachable during training: the renderer rejects a None batch and
            # every occupied token emits a Gaussian, so all ranks reach the
            # sync_dist logging below together. This guard only covers callers
            # that hand in an empty list.
            return
        velocity = torch.cat([item["velocity"] for item in active], dim=0)
        speed = velocity.norm(dim=-1)
        statistics = {
            "velocity_speed_mean": speed.mean(),
            # Under DDP this reduces to the mean of per-rank maxima rather than
            # a global max. It is still a usable runaway alarm.
            "velocity_speed_max": speed.max(),
            "velocity_abs_mean": velocity.abs().mean(),
            # nuScenes traffic tops out near 30 m/s, so a rising fraction here
            # is motion the scene cannot physically contain.
            "velocity_frac_over_30mps": (speed > 30.0).to(speed.dtype).mean(),
        }
        statistics.update(self._motion_proposal_statistics(active))
        for name, value in statistics.items():
            self.log(
                f"{prefix}/dynamic/{name}",
                value,
                on_step=(prefix == "train"),
                on_epoch=True,
                sync_dist=True,
                batch_size=int(speed.numel()),
            )

    @staticmethod
    def _motion_proposal_statistics(active):
        """Compact diagnostics for V6; older variants return an empty dict."""
        stats = {}

        def _cat(name):
            if any(name not in item for item in active):
                return None
            return torch.cat([item[name] for item in active], dim=0).float()

        for field, label in (
            ("p_unmatched", "p_unmatched_mean"),
            ("match_probability", "match_probability_mean"),
            ("motion_top1_probability", "motion_top1_probability_mean"),
            ("motion_effective_support", "motion_effective_support_mean"),
            (
                "motion_reciprocal_probability",
                "motion_reciprocal_probability_mean",
            ),
            (
                "motion_dustbin_similarity",
                "motion_dustbin_similarity_mean",
            ),
            ("motion_search_speed_mps", "motion_search_speed_mps_mean"),
        ):
            value = _cat(field)
            if value is not None:
                stats[label] = value.mean()

        for field, label in (
            ("velocity_match", "velocity_match_norm_mean"),
            ("velocity_init", "velocity_init_norm_mean"),
            ("velocity_offset", "velocity_offset_norm_mean"),
        ):
            value = _cat(field)
            if value is not None:
                stats[label] = value.norm(dim=-1).mean()
        return stats

    def _add_small_motion_prior(self, losses, gaussians, *, prefix):
        """Optional robust prior on physical displacement over each window."""
        cfg = getattr(getattr(self.dynamic_cfg, "regularization", None),
                      "small_motion", None)
        enabled = bool(getattr(cfg, "enabled", False)) if cfg else False
        if not enabled or prefix != "train":
            return

        active = self._active_motion_items(gaussians)
        if not active:
            reference = losses["total"]
            raw_loss = reference * 0.0
        else:
            displacement = torch.cat([
                item["velocity"] * item["window_duration_sec"].unsqueeze(-1)
                for item in active
            ], dim=0)
            epsilon = float(getattr(cfg, "epsilon_m", 1.0e-3)) if cfg else 1.0e-3
            speed = displacement.norm(dim=-1)
            raw_loss = (
                torch.sqrt(speed.square() + epsilon * epsilon) - epsilon
            ).mean()

        max_weight = float(getattr(cfg, "weight", 0.0)) if cfg else 0.0
        warmup = int(getattr(cfg, "warmup_steps", 0)) if cfg else 0
        ramp = int(getattr(cfg, "ramp_steps", 0)) if cfg else 0
        step = int(self.global_step)
        if step < warmup:
            effective_weight = 0.0
        elif ramp <= 0:
            effective_weight = max_weight
        else:
            effective_weight = max_weight * min(
                1.0, float(step - warmup + 1) / float(ramp)
            )
        weighted = raw_loss * effective_weight
        losses["loss_velocity_prior"] = raw_loss
        losses["wc_velocity_prior"] = weighted.detach()
        losses["total"] = losses["total"] + weighted

    def _metric_schedule(self, prefix: str, batch_idx: int) -> tuple[bool, bool]:
        """Return ``(valid, official_raydrop)`` metric switches for this batch.

        Train computes both diagnostic and official metrics once per configured
        optimizer-step interval. Validation/test computes official LiDAR4D /
        GS-LiDAR metrics for every frame, while the valid-only metrics stay
        sparse. Train keys off ``global_step`` and validation off ``batch_idx``,
        both of which advance identically on every rank, so conditional
        ``sync_dist=True`` logging cannot deadlock under DDP.
        """
        if prefix == "train":
            step = int(self.global_step)
            due = (
                step % self._metric_interval == 0
                and step != self._last_train_metric_step
            )
            if due:
                self._last_train_metric_step = step
            return due, due
        if prefix in {"val", "test", "eval"}:
            return int(batch_idx) % self._metric_interval == 0, True
        return True, True

    def on_validation_epoch_start(self) -> None:
        self._eval_pair_summaries["val"] = []

    def on_validation_epoch_end(self) -> None:
        self._write_eval_aggregate(prefix="val")

    def on_test_epoch_start(self) -> None:
        self._eval_pair_summaries["test"] = []

    def on_test_epoch_end(self) -> None:
        self._write_eval_aggregate(prefix="test")

    @staticmethod
    def _batch_size(_input: dict) -> int:
        pose = _input.get("pose")
        if isinstance(pose, list):
            return len(pose)
        if torch.is_tensor(pose) and pose.dim() > 0:
            return int(pose.shape[0])
        return 1

    def _log_losses(self, losses: dict, *, prefix: str, batch_size: int) -> None:
        for key, value in losses.items():
            if key not in self._wandb_loss_keys:
                continue
            if not torch.is_tensor(value):
                continue
            epoch_only = key in ADAPTIVE_BUDGET_LOG_KEYS
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=not epoch_only,
                on_epoch=True,
                prog_bar=(key == "total" or key.startswith("loss_")),
                # Budget terms already contain an autograd-safe global token
                # reduction and therefore have identical forward values on all
                # ranks. Avoid redundant logging collectives for those keys.
                sync_dist=(key not in GLOBAL_REDUCED_LOSS_KEYS),
                batch_size=batch_size,
            )

    def _routing_budget_spec(
        self,
        routing_budget_logits: torch.Tensor | None,
        *,
        prefix: str,
    ) -> dict | None:
        """Build the optional router input consumed by the shared Loss module."""
        if prefix != "train" or not self._budget_enable:
            return None
        if routing_budget_logits is None:
            raise RuntimeError(
                "learned-count budget is enabled but routing logits are missing"
            )
        return {
            "logits": routing_budget_logits,
            "target_mean_k": self._budget_target_mean_k,
            "weight": self._budget_weight,
            "step": int(self.global_step),
            "warmup_steps": self._budget_warmup_steps,
            "ramp_steps": self._budget_ramp_steps,
        }

    def _log_routing_stats(self, routing: dict | None, *, prefix: str) -> None:
        """Accumulate detached learned-count diagnostics for epoch-only logs."""
        if not self._routing_logging_enable or routing is None:
            return
        if prefix == "train":
            step = int(self.global_step)
            if (
                step % self._routing_logging_interval != 0
                or step == self._last_train_routing_step
            ):
                return
            self._last_train_routing_step = step

        statistics = routing_sufficient_statistics(
            routing,
            self._routing_range_edges,
            include_breakdowns=self._routing_unit_name == "token",
        )
        statistics = distributed_sum_statistics(statistics)
        unit_count = statistics["token_count"]
        if not bool((unit_count > 0).item()):
            return

        base_batch_size = max(1, int(unit_count.item()))

        def log_value(name, value, *, batch_size=base_batch_size):
            self.log(
                f"{prefix}/routing/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,  # statistics were explicitly summed above.
                batch_size=max(1, int(batch_size)),
            )

        log_value("gaussians", statistics["sampled_k_sum"])
        log_value(
            "sampled/mean_k",
            statistics["sampled_k_sum"] / unit_count,
        )
        log_value(
            "argmax/mean_k",
            statistics["argmax_k_sum"] / unit_count,
        )
        k_max = int(statistics["selected_counts"].numel())
        for index in range(k_max):
            k = index + 1
            log_value(
                f"sampled/frac_k{k}",
                statistics["selected_counts"][index] / unit_count,
            )
            log_value(
                f"argmax/frac_k{k}",
                statistics["argmax_counts"][index] / unit_count,
            )

        # Spherical routing only needs a coarse collapse/learning check in W&B:
        # global mean K and per-K sampled/argmax fractions. Keep the historical
        # range and bg/fg diagnostics for Grid without computing them for anchors.
        if self._routing_unit_name == "anchor":
            return

        range_labels = range_bin_labels(self._routing_range_edges)
        for index, label in enumerate(range_labels):
            count = statistics["range_token_counts"][index]
            log_value(
                f"range/{label}/{self._routing_unit_name}_frac",
                count / unit_count,
            )
            if not bool((count > 0).item()):
                continue
            bin_batch_size = int(count.item())
            log_value(
                f"range/{label}/sampled_mean_k",
                statistics["range_sampled_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )
            log_value(
                f"range/{label}/argmax_mean_k",
                statistics["range_argmax_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )

        for group_index, group_name in enumerate(("bg", "fg")):
            count = statistics["group_token_counts"][group_index]
            log_value(
                f"{group_name}/{self._routing_unit_name}_frac",
                count / unit_count,
            )
            if not bool((count > 0).item()):
                continue
            group_batch_size = int(count.item())
            log_value(
                f"{group_name}/sampled_mean_k",
                statistics["group_sampled_k_sums"][group_index] / count,
                batch_size=group_batch_size,
            )
            log_value(
                f"{group_name}/argmax_mean_k",
                statistics["group_argmax_k_sums"][group_index] / count,
                batch_size=group_batch_size,
            )
            for index in range(k_max):
                k = index + 1
                log_value(
                    f"{group_name}/sampled_frac_k{k}",
                    statistics["group_selected_counts"][
                        group_index, index
                    ] / count,
                    batch_size=group_batch_size,
                )
                log_value(
                    f"{group_name}/argmax_frac_k{k}",
                    statistics["group_argmax_counts"][
                        group_index, index
                    ] / count,
                    batch_size=group_batch_size,
                )

    def _record_eval_summary(self, losses: dict, *, batch_idx: int, prefix: str) -> None:
        summary = {
            "epoch": int(getattr(self, "current_epoch", 0)),
            "global_step": int(getattr(self, "global_step", 0)),
            "batch_idx": int(batch_idx),
        }
        for key, value in losses.items():
            if torch.is_tensor(value):
                summary[key] = float(value.detach().mean().cpu())
        self._eval_pair_summaries.setdefault(prefix, []).append(summary)

    def _write_eval_aggregate(self, *, prefix: str) -> None:
        if not self._is_rank_zero():
            return
        summaries = self._eval_pair_summaries.get(prefix, [])
        if not summaries:
            return

        metric_keys = sorted(
            key
            for key in summaries[0].keys()
            if key not in {"epoch", "global_step", "batch_idx"}
        )
        aggregate = {}
        for key in metric_keys:
            values = [item[key] for item in summaries if key in item]
            if values:
                aggregate[key] = sum(values) / len(values)

        out_dir = self._eval_output_dir(prefix)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(getattr(self, "current_epoch", 0)),
            "global_step": int(getattr(self, "global_step", 0)),
            "split": prefix,
            "num_batches": len(summaries),
            "aggregate": aggregate,
            "batches": summaries,
        }
        with open(out_dir / "aggregate_summary.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _eval_output_dir(self, prefix: str) -> Path:
        logger_dir = self._cfg_get("logger.dir", "")
        base = Path(str(logger_dir)) if logger_dir else Path("outputs")
        return base / f"eval_{prefix}"

    def _cfg_get(self, dotted_key: str, default=None):
        cur = self.cfg
        for part in dotted_key.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                try:
                    cur = getattr(cur, part)
                except (AttributeError, KeyError):
                    return default
        return cur

    def _is_rank_zero(self) -> bool:
        return int(getattr(self, "global_rank", 0)) == 0

    def configure_optimizers(self):
        opt_cfg = self.cfg.train
        #lr = float(getattr(opt_cfg, "lr", getattr(self.p2g_cfg, "lr", 5e-4)))
        lr = opt_cfg.lr
        weight_decay = opt_cfg.weight_decay
        # weight_decay = float(
        #     getattr(opt_cfg, "weight_decay", getattr(self.p2g_cfg, "weight_decay", 0.0))
        # )
        trainable_params = [p for p in self.p2g_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Linear warmup (over warmup_iters optimizer steps) -> cosine decay to 0
        # over the full run. Stepped per-optimizer-step (interval="step"), so the
        # warmup count is in optimizer steps (matches config warmup_iters).
        warmup = int(getattr(opt_cfg, "warmup_iters", 0) or 0)
        total_steps = int(self.trainer.estimated_stepping_batches)
        total_steps = max(total_steps, warmup + 1)
        if self._is_rank_zero():
            print(f"[sched] AdamW lr={lr} warmup_iters={warmup} "
                  f"total_optimizer_steps={total_steps} (cosine decay)", flush=True)

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # def _write_eval_artifacts(self, batch, batch_idx: int, *, prefix: str) -> None:
    #     if not self._is_rank_zero():
    #         return
    #     if not self._cfg_get(f"{prefix}.save_artifacts", self._cfg_get("eval.save_artifacts", True)):
    #         return

    #     out_dir = self._eval_output_dir(prefix)
    #     hit_threshold = float(self._cfg_get("eval.hit_threshold", 0.5))
    #     save_pointclouds = bool(self._cfg_get("eval.save_pointclouds", True))
    #     batch_size = len(batch["input_0"])
    #     device = str(next(self.p2g_model.parameters()).device)

    #     self.p2g_model.primitive_model.eval()
    #     summaries = self._eval_pair_summaries.setdefault(prefix, [])
    #     with torch.no_grad():
    #         for local_idx in range(batch_size):
    #             sample = self._sample_from_batch(batch, local_idx)
    #             result = evaluate_pair_sample(
    #                 self.p2g_model.primitive_model,
    #                 self.p2g_model.loss_fn,
    #                 sample,
    #                 device=device,
    #                 cfg=self.p2g_cfg,
    #                 hit_threshold=hit_threshold,
    #             )

    #             pair_idx = self._artifact_pair_idx(batch, batch_idx, local_idx)
    #             summary = {
    #                 "pair_idx": pair_idx,
    #                 "epoch": int(getattr(self, "current_epoch", 0)),
    #                 "global_step": int(getattr(self, "global_step", 0)),
    #                 "batch_idx": int(batch_idx),
    #                 "local_idx": int(local_idx),
    #                 **result["summary"],
    #             }
    #             summaries.append(summary)

    #             pair_dir = out_dir / f"pair_{pair_idx:06d}"
    #             pair_dir.mkdir(parents=True, exist_ok=True)
    #             with open(pair_dir / "summary.json", "w") as f:
    #                 json.dump(summary, f, indent=2)
    #             with open(pair_dir / "diagnostics.json", "w") as f:
    #                 json.dump(summary["scene"]["contexts"], f, indent=2)

    #             if save_pointclouds:
    #                 artifacts = result["artifacts"]
    #                 write_ply(pair_dir / "pred_frame0.ply", artifacts["pred_frame0_pts"], artifacts["pred_frame0_intensity"])
    #                 write_ply(pair_dir / "gt_frame0.ply", artifacts["gt_frame0_pts"], artifacts["gt_frame0_intensity"])
    #                 write_ply(pair_dir / "pred_frame1.ply", artifacts["pred_frame1_pts"], artifacts["pred_frame1_intensity"])
    #                 write_ply(pair_dir / "gt_frame1.ply", artifacts["gt_frame1_pts"], artifacts["gt_frame1_intensity"])

    # def _write_eval_aggregate(self, *, prefix: str) -> None:
    #     if not self._is_rank_zero():
    #         return
    #     summaries = self._eval_pair_summaries.get(prefix, [])
    #     if not summaries:
    #         return
    #     aggregate = {
    #         "epoch": int(getattr(self, "current_epoch", 0)),
    #         "global_step": int(getattr(self, "global_step", 0)),
    #         "split": prefix,
    #         "pairs": [int(s["pair_idx"]) for s in summaries],
    #         "aggregate": self._aggregate_pair_summaries(summaries),
    #     }
    #     out_dir = self._eval_output_dir(prefix)
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     with open(out_dir / "aggregate_summary.json", "w") as f:
    #         json.dump(aggregate, f, indent=2)

    # def _eval_output_dir(self, prefix: str) -> Path:
    #     configured = self._cfg_get(f"{prefix}.out_dir", self._cfg_get("eval.out_dir", ""))
    #     if configured:
    #         return Path(str(configured)) / prefix

    #     logger_dir = self._cfg_get("logger.dir", "")
    #     base = Path(str(logger_dir)) if logger_dir else Path("outputs")
    #     return base / f"eval_{prefix}"

    # def _sample_from_batch(self, batch, idx: int) -> dict:
    #     return {key: self._batch_item(batch, key, idx) for key in batch.keys()}

    # @staticmethod
    # def _batch_item(batch, key, idx: int):
    #     value = batch[key]
    #     if isinstance(value, (list, tuple)):
    #         return value[idx]
    #     if torch.is_tensor(value):
    #         if value.dim() > 0 and value.shape[0] > idx:
    #             return value[idx]
    #         return value
    #     return value

    # @staticmethod
    # def _artifact_pair_idx(batch, batch_idx: int, local_idx: int) -> int:
    #     if "dataset_index" in batch:
    #         item = ModelWrapper._batch_item(batch, "dataset_index", local_idx)
    #         if torch.is_tensor(item):
    #             return int(item.item())
    #         return int(item)
    #     return int(batch_idx) * max(1, len(batch["input_0"])) + int(local_idx)

    # def _cfg_get(self, dotted_key: str, default=None):
    #     value = self.cfg if hasattr(self, "cfg") else None
    #     if value is None:
    #         value = {
    #             "optimizer": self.cfg,
    #             "p2g": self.p2g_cfg,
    #         }
    #     cur = value
    #     for part in dotted_key.split("."):
    #         if cur is None:
    #             return default
    #         if isinstance(cur, dict):
    #             cur = cur.get(part, default)
    #         else:
    #             try:
    #                 cur = getattr(cur, part)
    #             except (AttributeError, KeyError):
    #                 return default
    #     return cur

    # def _is_rank_zero(self) -> bool:
    #     return int(getattr(self, "global_rank", 0)) == 0


    # @staticmethod
    # def _aggregate_pair_summaries(pair_summaries: list[dict]) -> dict:
    #     metric_paths = [
    #         ("frame0", "loss_total"),
    #         ("frame0", "loss_depth"),
    #         ("frame0", "loss_intensity"),
    #         ("frame0", "loss_raydrop"),
    #         ("frame0", "hit_precision"),
    #         ("frame0", "hit_recall"),
    #         ("frame0", "hit_f1"),
    #         ("frame0", "depth_mae_on_gt"),
    #         ("frame0", "approx_chamfer_m"),
    #         ("frame1", "loss_total"),
    #         ("frame1", "loss_depth"),
    #         ("frame1", "loss_intensity"),
    #         ("frame1", "loss_raydrop"),
    #         ("frame1", "hit_precision"),
    #         ("frame1", "hit_recall"),
    #         ("frame1", "hit_f1"),
    #         ("frame1", "depth_mae_on_gt"),
    #         ("frame1", "approx_chamfer_m"),
    #     ]
    #     aggregate = {"num_pairs": len(pair_summaries)}
    #     for frame_key, metric_key in metric_paths:
    #         values = [pair[frame_key][metric_key] for pair in pair_summaries]
    #         name = f"{frame_key}_{metric_key}"
    #         aggregate[name] = {
    #             "mean": float(sum(values) / len(values)),
    #             "min": float(min(values)),
    #             "max": float(max(values)),
    #         }
    #     return aggregate
