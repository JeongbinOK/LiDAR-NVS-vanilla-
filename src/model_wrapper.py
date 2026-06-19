import torch
import json
from pathlib import Path

from src.models_new.module import Point2Gaus, GausTemp, GausRender
#from src.models_new.utils.eval_utils import evaluate_pair_sample, write_ply
from lightning.pytorch import LightningModule
from src.models_new.utils.loss import Loss

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
 

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")


    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="test")


    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="val")


    def _shared_step(self, batch, batch_idx, *, prefix: str):
        _input, gt = batch["input"], batch["gt"]
        out = self.p2g_model(_input, batch_idx=batch_idx, mode=prefix)
        out = self.g2g_model(out, _input["timestamps"])
        all_renders = self.g2p_model(out, gt)

        loss_dict = self.loss(all_renders)
        self._log_losses(loss_dict, prefix=prefix, batch_size=self._batch_size(_input))

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if prefix != "train":
            self._record_eval_summary(loss_dict, batch_idx=batch_idx, prefix=prefix)

        return loss_dict["total"]



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
            if not torch.is_tensor(value):
                continue
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key == "total" or key.startswith("loss_")),
                sync_dist=True,
                batch_size=batch_size,
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
        return optimizer

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
