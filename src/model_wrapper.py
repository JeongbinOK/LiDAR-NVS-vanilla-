from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.nn import functional as F
from models import QGSModel
from utils.qgs_loss import QGSLoss

class ModelWrapper(LightningModule):
    def __init__(
        self,
        cfg,
        step_tracker
    )
        super().__init__()
        self.optimizer_cfg = cfg.optimizer
        self.p2g_cfg = cfg.p2g
        self.g2g_cfg = cfg.g2g
        self.g2p_cfg = cfg.g2p
        self.step_tracker = step_tracker

        # Set up the model.
        self.p2g_model = QGSModel(self.p2g_cfg)
        self.g2g_model = None
        self.g2p_model = None
        self.losses = QGSLoss(loss_config)
 

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()        

        B = len(batch["input_0"])
        batch_loss = 0.0
        batch_loss_dicts = []

        #optimizer.zero_grad()
        valid_pairs = 0
        for b in range(B):
            out = self.p2g_model(cfg, batch) # 이렇게 되게 만들어 주세요. 

            # out = self.g2g_model(cfg, batch)
            # out = self.g2p_model(cfg, batch) 
            # out = process_pair(self.p2g_cfg, self.losses, ray_grid, batch, cfg, bad_grad_trace=None) 
            loss = out["total"]
            batch_loss += loss   


        self.log("loss/total", batch_loss)
        self.manual_backward(batch_loss)


        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if self.global_rank == 0:
            print(
                f"train step {self.global_step}; "
                f"loss = {batch_loss:.6f}"
            )

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)


    def test_step(self, batch, batch_idx):
        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
            )

        B = len(batch["input_0"])
        batch_loss = 0.0
        batch_loss_dicts = []

        #optimizer.zero_grad()
        valid_pairs = 0
        for b in range(B):
            out = self.p2g_model(cfg, batch) # 이렇게 되게 만들어 주세요. 

            # out = self.g2g_model(cfg, batch)
            # out = self.g2p_model(cfg, batch) 
            # out = process_pair(self.p2g_cfg, self.losses, ray_grid, batch, cfg, bad_grad_trace=None) 
            loss = out["total"]
            batch_loss += loss   
            
        pair_summaries = []

        # 여기 아래도 마찬가지, batch와 cfg만으로 처리할 수 있께 해주세여
        with torch.no_grad():
            for pair_idx in pair_indices:
                sample = dataset[pair_idx]
                result = evaluate_pair_sample(
                    model,
                    loss_fn,
                    sample,
                    device=args.device,
                    cfg=cfg,
                    hit_threshold=args.hit_threshold,
                )

                summary = {
                    "pair_idx": pair_idx,
                    "checkpoint_epoch": ckpt.get("epoch"),
                    "checkpoint_loss": ckpt.get("loss"),
                    **result["summary"],
                }
                pair_summaries.append(summary)

                pair_dir = out_dir / f"pair_{pair_idx:04d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                with open(pair_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                with open(pair_dir / "diagnostics.json", "w") as f:
                    json.dump(summary["scene"]["contexts"], f, indent=2)

                if args.save_pointclouds:
                    artifacts = result["artifacts"]
                    write_ply(pair_dir / "pred_frame0.ply", artifacts["pred_frame0_pts"], artifacts["pred_frame0_intensity"])
                    write_ply(pair_dir / "gt_frame0.ply", artifacts["gt_frame0_pts"], artifacts["gt_frame0_intensity"])
                    write_ply(pair_dir / "pred_frame1.ply", artifacts["pred_frame1_pts"], artifacts["pred_frame1_intensity"])
                    write_ply(pair_dir / "gt_frame1.ply", artifacts["gt_frame1_pts"], artifacts["gt_frame1_intensity"])

                print(
                    f"[pair {pair_idx:04d}] "
                    f"f0 depth={summary['frame0']['depth_mae_on_gt']:.3f} "
                    f"f1 depth={summary['frame1']['depth_mae_on_gt']:.3f} "
                    f"f0 touched={summary['frame0']['gaussians']['overall']['n_touched']} "
                    f"f1 touched={summary['frame1']['gaussians']['overall']['n_touched']}"
                )

        aggregate = {
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_loss": ckpt.get("loss"),
            "split": args.split,
            "pairs": pair_indices,
            "aggregate": _aggregate_pair_summaries(pair_summaries),
        }
        with open(out_dir / "aggregate_summary.json", "w") as f:
            json.dump(aggregate, f, indent=2)

        print(json.dumps(aggregate, indent=2))
        print(f"saved evaluation outputs to {out_dir}")


    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
            )

        B = len(batch["input_0"])
        batch_loss = 0.0
        batch_loss_dicts = []

        #optimizer.zero_grad()
        valid_pairs = 0
        for b in range(B):
            out = self.p2g_model(cfg, batch) # 이렇게 되게 만들어 주세요. 

            # out = self.g2g_model(cfg, batch)
            # out = self.g2p_model(cfg, batch) 
            # out = process_pair(self.p2g_cfg, self.losses, ray_grid, batch, cfg, bad_grad_trace=None) 
            loss = out["total"]
            batch_loss += loss   
            
        pair_summaries = []
        # 여기 아래도 마찬가지, batch와 cfg만으로 처리할 수 있께 해주세여
        with torch.no_grad():
            for pair_idx in pair_indices:
                sample = dataset[pair_idx]
                result = evaluate_pair_sample(
                    model,
                    loss_fn,
                    sample,
                    device=args.device,
                    cfg=cfg,
                    hit_threshold=args.hit_threshold,
                )

                summary = {
                    "pair_idx": pair_idx,
                    "checkpoint_epoch": ckpt.get("epoch"),
                    "checkpoint_loss": ckpt.get("loss"),
                    **result["summary"],
                }
                pair_summaries.append(summary)

                pair_dir = out_dir / f"pair_{pair_idx:04d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                with open(pair_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                with open(pair_dir / "diagnostics.json", "w") as f:
                    json.dump(summary["scene"]["contexts"], f, indent=2)

                if args.save_pointclouds:
                    artifacts = result["artifacts"]
                    write_ply(pair_dir / "pred_frame0.ply", artifacts["pred_frame0_pts"], artifacts["pred_frame0_intensity"])
                    write_ply(pair_dir / "gt_frame0.ply", artifacts["gt_frame0_pts"], artifacts["gt_frame0_intensity"])
                    write_ply(pair_dir / "pred_frame1.ply", artifacts["pred_frame1_pts"], artifacts["pred_frame1_intensity"])
                    write_ply(pair_dir / "gt_frame1.ply", artifacts["gt_frame1_pts"], artifacts["gt_frame1_intensity"])

                print(
                    f"[pair {pair_idx:04d}] "
                    f"f0 depth={summary['frame0']['depth_mae_on_gt']:.3f} "
                    f"f1 depth={summary['frame1']['depth_mae_on_gt']:.3f} "
                    f"f0 touched={summary['frame0']['gaussians']['overall']['n_touched']} "
                    f"f1 touched={summary['frame1']['gaussians']['overall']['n_touched']}"
                )

        aggregate = {
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_loss": ckpt.get("loss"),
            "split": args.split,
            "pairs": pair_indices,
            "aggregate": _aggregate_pair_summaries(pair_summaries),
        }
        with open(out_dir / "aggregate_summary.json", "w") as f:
            json.dump(aggregate, f, indent=2)

        print(json.dumps(aggregate, indent=2))
        print(f"saved evaluation outputs to {out_dir}")