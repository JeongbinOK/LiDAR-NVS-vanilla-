import torch
import json
import math
from pathlib import Path

from src.models_new.module import Point2Gaus, GausTemp, GausRender
from lightning.pytorch import LightningModule
from src.models_new.utils.loss import Loss, router_guide_loss
from src.models_new.utils.routing_logging import (
    distributed_sum_statistics,
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
    "budget_expected_mean_k",
    "budget_violation",
    "loss_router_guide",
    "router_guide_accuracy",
    "router_guide_mean_target_k",
    "router_guide_geometry_valid_fraction",
    "router_guide_low_support_fraction",
    "router_guide_frac_k1",
    "router_guide_frac_k2",
    "router_guide_frac_k3",
    "router_guide_frac_k4",
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

# Historical W&B key names. Lightning appends ``_epoch`` to a metric only when
# it is logged with ``on_step`` and ``on_epoch`` both true, so every run before
# the epoch-only switch wrote ``train/loss_scale_epoch``. Logging the same value
# with ``on_epoch`` alone writes ``train/loss_scale``, which W&B treats as a
# different metric and puts on its own panel, breaking comparison with those
# runs. The suffix is therefore spelled out instead of being inherited from the
# step/epoch fork. The routing budget and guide terms were epoch-only from the
# day they were added and never carried a suffix, so they keep their bare names.
# That list is written out rather than aliased to GLOBAL_REDUCED_LOSS_KEYS,
# which encodes a collective-reduction property and only happens to hold the
# same keys today.
UNSUFFIXED_WANDB_LOSS_KEYS = frozenset({
    "loss_budget",
    "budget_expected_mean_k",
    "budget_violation",
    "loss_router_guide",
    "router_guide_accuracy",
    "router_guide_mean_target_k",
    "router_guide_geometry_valid_fraction",
    "router_guide_low_support_fraction",
    "router_guide_frac_k1",
    "router_guide_frac_k2",
    "router_guide_frac_k3",
    "router_guide_frac_k4",
})


def wandb_loss_key(prefix: str, key: str) -> str:
    """Name an epoch-logged loss the way earlier runs named it."""
    suffix = "" if key in UNSUFFIXED_WANDB_LOSS_KEYS else "_epoch"
    return f"{prefix}/{key}{suffix}"


# Progress-bar entries, in display order. These never reach W&B: they exist so
# the terminal stays readable while the logged surface remains epoch-level.
# ``loss_chamfer`` is included unconditionally because Loss computes the chamfer
# distance on every batch regardless of ``w_chamfer`` -- a zero weight removes
# it from the gradient, not from the forward pass, so the value stays real.
PROGRESS_BAR_LOSS_KEYS = (
    ("total", "total"),
    ("loss_depth", "depth"),
    ("loss_depth_median", "depth_med"),
    ("loss_intensity", "inten"),
    ("loss_raydrop", "raydrop"),
    ("loss_chamfer", "chamfer"),
    ("loss_scale", "scale"),
)

GLOBAL_REDUCED_LOSS_KEYS = {
    "loss_budget",
    "budget_expected_mean_k",
    "budget_violation",
    "loss_router_guide",
    "router_guide_accuracy",
    "router_guide_mean_target_k",
    "router_guide_geometry_valid_fraction",
    "router_guide_low_support_fraction",
    "router_guide_frac_k1",
    "router_guide_frac_k2",
    "router_guide_frac_k3",
    "router_guide_frac_k4",
}


class ModelWrapper(LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        #self.optimizer_cfg = cfg.optimizer
        self.p2g_cfg = cfg.p2g
        self.g2g_cfg = cfg.g2g
        self.g2p_cfg = cfg.g2p

        # Set up the model.
        self.p2g_model = Point2Gaus(self.p2g_cfg)
        self.g2g_model = GausTemp(self.g2g_cfg)
        self.g2p_model = GausRender(self.g2p_cfg)
        self.loss = Loss(self.cfg.loss)
        self._eval_pair_summaries: dict[str, list[dict]] = {}
        self._metric_interval = int(self._cfg_get("metrics.interval", 50))
        if self._metric_interval <= 0:
            raise ValueError("metrics.interval must be positive")
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
        if self._routing_logging_interval <= 0:
            raise ValueError(
                "learned_count.logging.interval must be positive"
            )
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
        # A ceiling rather than an alignment target: rendering decides the
        # allocation freely below target_mean_k. Applies to the post-guide
        # budget only -- the guide phase deliberately pins the mean to the
        # pseudo-GT distribution from both sides.
        self._budget_one_sided = bool(self._cfg_get(
            f"{learned_count_block}.budget.one_sided", False
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
        pseudo_gt_block = "p2g.grid_query.learned_count.pseudo_gt"
        guide_requested = bool(self._cfg_get(
            f"{pseudo_gt_block}.enable", False
        ))
        if guide_requested and not (
            anchor_mode == "grid" and routing_count_mode == "learned_gumbel"
        ):
            raise ValueError(
                "router pseudo-GT requires grid count_mode='learned_gumbel'"
            )
        self._router_guide_enable = guide_requested
        self._router_guide_weight = float(self._cfg_get(
            f"{pseudo_gt_block}.weight", 1.0
        ))
        self._router_guide_train_fraction = float(self._cfg_get(
            f"{pseudo_gt_block}.train_fraction", 0.5
        ))
        self._router_guide_min_raw_points = int(self._cfg_get(
            f"{pseudo_gt_block}.min_raw_points", 3
        ))
        self._router_guide_label_smoothing = float(self._cfg_get(
            f"{pseudo_gt_block}.label_smoothing", 0.1
        ))
        max_epochs = int(self._cfg_get("train.max_epochs", 0))
        self._router_guide_stop_epoch = int(math.ceil(
            self._router_guide_train_fraction * max_epochs
        ))
        if self._router_guide_enable:
            if self._router_guide_weight < 0.0:
                raise ValueError("router pseudo-GT weight must be non-negative")
            if not 0.0 < self._router_guide_train_fraction <= 1.0:
                raise ValueError(
                    "router pseudo-GT train_fraction must be in (0, 1]"
                )
            if max_epochs <= 0 or self._router_guide_stop_epoch <= 0:
                raise ValueError(
                    "router pseudo-GT requires a positive train.max_epochs"
                )
            if self._router_guide_min_raw_points < 3:
                raise ValueError(
                    "router pseudo-GT min_raw_points must be at least 3"
                )
            if not 0.0 <= self._router_guide_label_smoothing < 1.0:
                raise ValueError(
                    "router pseudo-GT label_smoothing must be in [0, 1)"
                )
            # A two-sided budget is an alignment target, and its target is the
            # post-guide budget -- far above the mean K the CE guide teaches.
            # The two would then pull against each other for the whole guide
            # phase. A one-sided ceiling has exactly zero gradient underneath,
            # so the guide runs unopposed without needing the ceiling retuned
            # to whatever mean the current thresholds happen to imply.
            if self._budget_enable and not self._budget_one_sided:
                raise ValueError(
                    "router pseudo-GT requires learned_count.budget.one_sided "
                    "when learned_count.budget is also enabled: a two-sided "
                    "budget target fights the guide's cross entropy"
                )
        # With gradient accumulation, multiple training batches can share one
        # Lightning global_step. Compute sparse metrics only once for that
        # optimizer step.
        self._last_train_metric_step = -1

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")


    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")


    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")


    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        router_guide_active = self._router_guide_active(prefix)
        p2g_out = self.p2g_model(
            _input,
            target_pose=gt.get("pose"),
            target_timestamps=gt.get("timestamps"),
            compute_router_guide=router_guide_active,
        )
        # Keep detached diagnostics outside the renderer/temporal model input.
        routing_stats = p2g_out.pop("routing_stats", None)
        routing_budget_logits = p2g_out.pop("routing_budget_logits", None)
        routing_guide = p2g_out.pop("routing_guide", None)
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
        self._add_router_guide_loss(
            loss_dict,
            routing_guide,
            active=router_guide_active,
        )
        self._log_losses(loss_dict, prefix=prefix, batch_size=self._batch_size(_input))
        self._log_routing_stats(routing_stats, prefix=prefix)


        if prefix != "train":
            self._record_eval_summary(loss_dict, batch_idx=batch_idx, prefix=prefix)

        return loss_dict["total"]

    def _router_guide_active(self, prefix: str) -> bool:
        """Enable online pseudo-GT only for the configured leading epochs."""
        return (
            prefix == "train"
            and self._router_guide_enable
            and int(self.current_epoch) < self._router_guide_stop_epoch
        )

    def _add_router_guide_loss(
        self,
        losses: dict,
        routing_guide: dict | None,
        *,
        active: bool,
    ) -> None:
        if not active:
            if routing_guide is not None:
                raise RuntimeError(
                    "router pseudo-GT was produced outside its active schedule"
                )
            return
        if routing_guide is None:
            raise RuntimeError(
                "router pseudo-GT is active but the model returned no guide fields"
            )
        terms = router_guide_loss(
            routing_guide["logits"],
            routing_guide["target_k"],
            geometry_valid=routing_guide["geometry_valid"],
            raw_count=routing_guide["raw_count"],
            weight=self._router_guide_weight,
            min_raw_points=self._router_guide_min_raw_points,
            label_smoothing=self._router_guide_label_smoothing,
        )
        losses["loss_router_guide"] = terms["loss"]
        losses["wc_router_guide"] = terms["weighted_loss"].detach()
        losses["router_guide_accuracy"] = terms["accuracy"]
        losses["router_guide_mean_target_k"] = terms["mean_target_k"]
        losses["router_guide_geometry_valid_fraction"] = terms[
            "geometry_valid_fraction"
        ]
        losses["router_guide_low_support_fraction"] = terms[
            "low_support_fraction"
        ]
        for index, fraction in enumerate(terms["target_fractions"], start=1):
            losses[f"router_guide_frac_k{index}"] = fraction
        losses["total"] = losses["total"] + terms["weighted_loss"]

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
                wandb_loss_key(prefix, key),
                value,
                on_step=False,
                on_epoch=True,
                # Progress-bar display is handled separately below: a metric
                # logged on_epoch only has nothing to show until the epoch ends,
                # which would leave the bar blank for a whole epoch.
                prog_bar=False,
                # Budget terms already contain an autograd-safe global token
                # reduction and therefore have identical forward values on all
                # ranks. Avoid redundant logging collectives for those keys.
                sync_dist=(key not in GLOBAL_REDUCED_LOSS_KEYS),
                batch_size=batch_size,
            )

        # Terminal-only values. ``logger=False`` keeps these out of W&B, so the
        # logged surface stays epoch-level while the bar still carries numbers.
        # Train reports the running step value so the bar moves within an epoch;
        # validation reports its epoch aggregate, which appears once the val
        # loop finishes and then persists next to the training numbers.
        is_train = prefix == "train"
        for key, label in PROGRESS_BAR_LOSS_KEYS:
            value = losses.get(key)
            if not torch.is_tensor(value):
                continue
            self.log(
                label if is_train else f"{prefix}_{label}",
                value,
                on_step=is_train,
                on_epoch=not is_train,
                prog_bar=True,
                logger=False,
                # Epoch-level entries would otherwise draw a per-key Lightning
                # warning under DDP; a step-level one must not add a collective.
                sync_dist=not is_train,
                batch_size=batch_size,
            )

    def _routing_budget_spec(
        self,
        routing_budget_logits: torch.Tensor | None,
        *,
        prefix: str,
    ) -> dict | None:
        """Build the optional router input consumed by the shared Loss module.

        With ``budget.one_sided`` the target is a ceiling rather than an
        alignment point, so one setting covers both phases: during the guide it
        is inert (the CE drives the mean far below it) and afterwards it is the
        only thing bounding the router, with rendering free underneath.
        """
        if prefix != "train" or not self._budget_enable:
            return None
        if routing_budget_logits is None:
            raise RuntimeError(
                "learned-count budget is enabled but routing logits are missing"
            )
        return {
            "logits": routing_budget_logits,
            "target_mean_k": self._budget_target_mean_k,
            "one_sided": self._budget_one_sided,
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

        statistics = routing_sufficient_statistics(routing)
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
