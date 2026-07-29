import torch
import json
import math
from pathlib import Path

from src.models_new.module import Point2Gaus, GausTemp, GausRender
#from src.models_new.utils.eval_utils import evaluate_pair_sample, write_ply
from lightning.pytorch import LightningModule
from src.models_new.utils.loss import Loss
from src.models_new.utils.debug_finite import (
    first_nonfinite,
    gaussian_param_absmax,
    tensor_report,
)
from src.models_new.utils.routing_logging import (
    distributed_sum_statistics,
    range_bin_labels,
    routing_sufficient_statistics,
)

WANDB_COMMON_LOSS_KEYS = {
    "loss_depth",
    "loss_depth_median",
    "loss_intensity",
    "loss_raydrop",
    "loss_chamfer",
    "loss_scale",
    "loss_budget",
    "wc_budget",
    "budget_expected_mean_k",
    "budget_violation",
    "budget_effective_weight",
    "render_points_mean",
    "intensity_psnr_valid",
    "intensity_ssim_valid",
    "depth_psnr_valid",
    "depth_ssim_valid",
    "intensity_psnr_raydrop",
    "intensity_ssim_raydrop",
    "depth_psnr_raydrop",
    "depth_ssim_raydrop",
    "total",
}

GLOBAL_REDUCED_LOSS_KEYS = {
    "loss_budget",
    "wc_budget",
    "budget_expected_mean_k",
    "budget_violation",
    "budget_effective_weight",
}

class ModelWrapper(LightningModule):
    def __init__(
        self,
        cfg,
        step_tracker
    ):
        super().__init__()
        self.cfg = cfg
        #self.optimizer_cfg = cfg.optimizer
        self.p2g_cfg = cfg.p2g
        self.g2g_cfg = cfg.g2g
        self.g2p_cfg = cfg.g2p
        self.step_tracker = step_tracker

        # Set up the model.
        self.p2g_model = Point2Gaus(self.p2g_cfg)
        self.g2g_model = GausTemp(self.g2g_cfg)
        self.g2p_model = GausRender(self.g2p_cfg)
        self.loss = Loss(self.cfg.loss)
        self._eval_pair_summaries: dict[str, list[dict]] = {}
        self._metric_interval = int(self._cfg_get("metrics.interval", 50))
        if self._metric_interval <= 0:
            raise ValueError("metrics.interval must be positive")
        self._routing_logging_enable = bool(self._cfg_get(
            "p2g.grid_query.learned_count.logging.enable", False
        ))
        self._routing_logging_interval = int(self._cfg_get(
            "p2g.grid_query.learned_count.logging.interval", 20
        ))
        self._routing_range_edges = tuple(float(value) for value in self._cfg_get(
            "p2g.grid_query.learned_count.logging.range_edges_m",
            (0, 10, 20, 30, 40, 60, 80, 110),
        ))
        if self._routing_logging_interval <= 0:
            raise ValueError(
                "p2g.grid_query.learned_count.logging.interval must be positive"
            )
        # Validates lower-bound ordering once, before the first train batch.
        range_bin_labels(self._routing_range_edges)
        self._last_train_routing_step = -1
        budget_requested = bool(self._cfg_get(
            "p2g.grid_query.learned_count.budget.enable", False
        ))
        self._budget_enable = (
            budget_requested
            and str(self._cfg_get("p2g.anchor_mode", "spherical")).lower() == "grid"
            and str(
                self._cfg_get("p2g.grid_query.count_mode", "legacy")
            ).lower() == "learned_gumbel"
        )
        self._budget_target_mean_k = float(self._cfg_get(
            "p2g.grid_query.learned_count.budget.target_mean_k", 2.0
        ))
        self._budget_weight = float(self._cfg_get(
            "p2g.grid_query.learned_count.budget.weight", 1.0
        ))
        self._budget_warmup_steps = int(self._cfg_get(
            "p2g.grid_query.learned_count.budget.warmup_steps", 500
        ))
        self._budget_ramp_steps = int(self._cfg_get(
            "p2g.grid_query.learned_count.budget.ramp_steps", 1000
        ))
        if self._budget_enable:
            budget_k_max = int(self._cfg_get(
                "p2g.grid_query.learned_count.K_max", 0
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
        # With gradient accumulation, multiple training batches can share one
        # Lightning global_step. Compute sparse metrics only once for that
        # optimizer step.
        self._last_train_metric_step = -1

        # --- NaN-collapse diagnostics (see scripts/nan_collapse_diagnosis.md) ---
        self._dbg_finite = bool(getattr(self.p2g_cfg, "debug_finite_check", False))
        self._dbg_grad_trace = bool(getattr(self.p2g_cfg, "debug_bad_grad_trace", False))
        self._dbg_grad_trace_max = int(getattr(self.p2g_cfg, "debug_bad_grad_trace_max", 16))
        self._dbg_anomaly_batch = int(getattr(self.p2g_cfg, "debug_anomaly_batch", -1))
        self._nonfinite_skips = 0
        self._last_dbg = {}
        self._last_batch_idx = -1
        # per-attribute non-finite *gradient* counts since last optimizer step
        self._attr_grad_nf = {}


    def on_fit_start(self):
        # Precise backward-op localization (slow; raises on first NaN). Opt-in.
        if self._dbg_anomaly_batch >= 0:
            torch.autograd.set_detect_anomaly(True)
            if self._is_rank_zero():
                print("[debug] torch.autograd.set_detect_anomaly(True) enabled")


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")


    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")


    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")


    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        p2g_out = self.p2g_model(_input, batch_idx=batch_idx, mode=prefix)
        # Keep detached diagnostics outside the renderer/temporal model input.
        routing_stats = p2g_out.pop("routing_stats", None)
        routing_budget_logits = p2g_out.pop("routing_budget_logits", None)
        out = self.g2g_model(p2g_out, _input["timestamps"])
        all_renders = self.g2p_model(out, gt)

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
        self._log_losses(loss_dict, prefix=prefix, batch_size=self._batch_size(_input))
        self._log_routing_stats(routing_stats, prefix=prefix)

        if self._dbg_finite and prefix == "train":
            self._debug_forward(out, all_renders, loss_dict, batch_idx)

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if prefix != "train":
            self._record_eval_summary(loss_dict, batch_idx=batch_idx, prefix=prefix)

        return loss_dict["total"]

    def _metric_schedule(self, prefix: str, batch_idx: int) -> tuple[bool, bool]:
        """Return ``(valid, official_raydrop)`` metric switches for this batch.

        Train computes both diagnostic and official metrics once per configured
        optimizer-step interval. Validation/test computes official LiDAR4D /
        GS-LiDAR metrics for every frame, while the valid-only diagnostics stay
        sparse. Every rank follows the same schedule, so conditional
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

    def _debug_forward(self, out, all_renders, loss_dict, batch_idx):
        """Log raw gaussian/render magnitudes and report the first non-finite
        forward tensor. `out` is the GausTemp output (list of per-batch b_gs)."""
        self._last_batch_idx = int(batch_idx)
        batch_gaussians = out if isinstance(out, list) else out.get("batch_gaussians", [])

        stats = gaussian_param_absmax(batch_gaussians)
        self._last_dbg = dict(stats)

        # Per-attribute gradient hooks: isolate WHICH gaussian attribute's grad
        # goes non-finite (scaling vs rotation vs opacity vs ...). The hook fires
        # during Lightning's backward, before on_before_optimizer_step reads it.
        for b_gs in batch_gaussians:
            if not isinstance(b_gs, dict):
                continue
            for attr in ("scaling", "rotation", "opacity", "shs", "position"):
                t = b_gs.get(attr)
                if torch.is_tensor(t) and t.requires_grad:
                    t.register_hook(self._make_grad_check_hook(attr))

        # First non-finite tensor, scanned forward -> backward order.
        named = []
        for b, b_gs in enumerate(batch_gaussians):
            if isinstance(b_gs, dict):
                for k in ("scaling", "opacity", "rotation", "position", "shs"):
                    named.append((f"gauss[{b}].{k}", b_gs.get(k)))
        for k in ("depth", "depth_median", "intensity_sh", "raydrop"):
            named.append((f"render.{k}", all_renders.get(k)))
        for k, v in loss_dict.items():
            named.append((f"loss.{k}", v))

        name, t = first_nonfinite(named)
        if name is not None and self._is_rank_zero():
            print(f"[NONFINITE-FWD] step={self.global_step} epoch={self.current_epoch} "
                  f"batch={batch_idx} first={name} :: {tensor_report(t)} | "
                  f"raw_gauss_absmax={ {k: round(v, 3) for k, v in stats.items()} }", flush=True)

    def _make_grad_check_hook(self, attr):
        def hook(grad):
            if grad is not None and not torch.isfinite(grad).all():
                self._attr_grad_nf[attr] = self._attr_grad_nf.get(attr, 0) + 1
            return grad
        return hook

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
            if key not in WANDB_COMMON_LOSS_KEYS:
                continue
            if not torch.is_tensor(value):
                continue
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=True,
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
        """Log detached learned-count diagnostics without touching render/loss."""
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
            routing, self._routing_range_edges
        )
        statistics = distributed_sum_statistics(statistics)
        token_count = statistics["token_count"]
        if not bool((token_count > 0).item()):
            return

        on_step = prefix == "train"
        base_batch_size = max(1, int(token_count.item()))

        def log_value(name, value, *, batch_size=base_batch_size):
            self.log(
                f"{prefix}/routing/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
                sync_dist=False,  # statistics were explicitly summed above.
                batch_size=max(1, int(batch_size)),
            )

        log_value("tokens", token_count)
        log_value("gaussians", statistics["sampled_k_sum"])
        log_value(
            "sampled/mean_k",
            statistics["sampled_k_sum"] / token_count,
        )
        log_value(
            "policy/expected_mean_k",
            statistics["expected_k_sum"] / token_count,
        )
        log_value(
            "argmax/mean_k",
            statistics["argmax_k_sum"] / token_count,
        )
        log_value(
            "policy/normalized_entropy",
            statistics["entropy_sum"] / token_count,
        )
        log_value(
            "policy/mean_top1_prob",
            statistics["top1_prob_sum"] / token_count,
        )
        log_value(
            "policy/mean_logit_margin",
            statistics["logit_margin_sum"] / token_count,
        )

        k_max = int(statistics["selected_counts"].numel())
        for index in range(k_max):
            k = index + 1
            log_value(
                f"sampled/frac_k{k}",
                statistics["selected_counts"][index] / token_count,
            )
            log_value(
                f"argmax/frac_k{k}",
                statistics["argmax_counts"][index] / token_count,
            )
            log_value(
                f"policy/prob_k{k}",
                statistics["policy_prob_sums"][index] / token_count,
            )

        range_labels = range_bin_labels(self._routing_range_edges)
        for index, label in enumerate(range_labels):
            count = statistics["range_token_counts"][index]
            log_value(f"range/{label}/token_frac", count / token_count)
            if not bool((count > 0).item()):
                continue
            bin_batch_size = int(count.item())
            log_value(
                f"range/{label}/sampled_mean_k",
                statistics["range_sampled_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )
            log_value(
                f"range/{label}/expected_mean_k",
                statistics["range_expected_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )
            log_value(
                f"range/{label}/argmax_mean_k",
                statistics["range_argmax_k_sums"][index] / count,
                batch_size=bin_batch_size,
            )

        for group_index, group_name in enumerate(("bg", "fg")):
            count = statistics["group_token_counts"][group_index]
            log_value(f"{group_name}/token_frac", count / token_count)
            if not bool((count > 0).item()):
                continue
            group_batch_size = int(count.item())
            log_value(
                f"{group_name}/sampled_mean_k",
                statistics["group_sampled_k_sums"][group_index] / count,
                batch_size=group_batch_size,
            )
            log_value(
                f"{group_name}/expected_mean_k",
                statistics["group_expected_k_sums"][group_index] / count,
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
                log_value(
                    f"{group_name}/policy_prob_k{k}",
                    statistics["group_policy_prob_sums"][
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

    def on_before_optimizer_step(self, optimizer, *args):
        """Skip the step if any gradient is non-finite, instead of letting a
        single bad batch poison every parameter + AdamW moment buffer forever.

        Under DDP, gradients are already all-reduced here, so every rank sees
        the same (NaN-propagated) grads and makes the same skip decision -> the
        ranks stay in sync. This is a SAFETY NET, not the cure: a chronic
        trigger degrades into silently skipping many steps (watch the counter).
        """
        nonfinite = False
        bad = []
        for name, p in self.named_parameters():
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all():
                nonfinite = True
                if self._dbg_grad_trace and len(bad) < self._dbg_grad_trace_max:
                    bad.append((name, int(torch.isnan(g).sum()),
                                int(torch.isinf(g).sum()), tuple(g.shape)))

        attr_nf = dict(self._attr_grad_nf)
        self._attr_grad_nf = {}  # reset for next optimizer step

        if nonfinite:
            self._nonfinite_skips += 1
            if self._is_rank_zero():
                print(f"[NONFINITE-GRAD] step={self.global_step} epoch={self.current_epoch} "
                      f"batch={self._last_batch_idx} total_skips={self._nonfinite_skips} "
                      f"| culprit_attr={attr_nf} "
                      f"| last_raw_gauss_absmax={ {k: round(v, 3) for k, v in self._last_dbg.items()} }",
                      flush=True)
                for nm, nan, inf, sh in bad:
                    print(f"    grad {nm}: nan={nan} inf={inf} shape={sh}", flush=True)
        self.log("train/nonfinite_grad_skips", float(self._nonfinite_skips),
                 on_step=True, on_epoch=False, batch_size=1, rank_zero_only=True)
        if nonfinite:
            optimizer.zero_grad(set_to_none=True)

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
