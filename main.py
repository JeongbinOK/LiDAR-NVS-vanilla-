import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import time
import warnings

import torch
import torch.multiprocessing as mp
# DataLoader workers pass shared-tensor FDs over sockets with the default
# 'file_descriptor' strategy, which fails under load with "received 0 items of
# ancdata" (worker dies -> pin-memory thread exits -> crash). 'file_system' uses
# named shm files instead and is robust. (Confirmed: num_workers=0 runs fine, so
# the fault is purely in the worker FD-passing path.) Runs at module import so it
# also applies in every DDP subprocess Lightning relaunches.
mp.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from src.model_wrapper import ModelWrapper
from src.models_new.utils.model_utils import DataModule
from src.dataloader import dataset_dict

import os
# Enable only when debugging CUDA stack traces; it slows normal training.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings(
    "ignore",
    message=r"Found .* module\(s\) in eval mode at the start of training.*",
)
torch.set_float32_matmul_precision("high")


def print_run_summary(cfg):
    print(
        "Run: "
        f"mode={cfg.mode}, "
        f"seed={cfg.seed}, "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}, "
        f"devices={list(cfg.device)}, "
        f"train_batch_size={cfg.train.batch_size}, "
        f"grad_accum_steps={cfg.train.grad_accum_steps}, "
        f"val_check_interval={cfg.train.val_check_interval}, "
        f"max_epochs={cfg.train.max_epochs}, "
        f"max_steps={cfg.train.max_steps}"
    )


def main(cfg):
    # Seed before logger, model, and DataModule construction so initialization,
    # Gumbel sampling, data shuffling, and DataLoader workers are reproducible.
    seed_everything(int(cfg.seed), workers=True)
    os.makedirs(cfg.logger.dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    callbacks = []
    if cfg.logger.enable:
        import wandb

        os.environ["WANDB__SERVICE_WAIT"] = "300"
        # Keep experiment metrics while suppressing W&B's CPU/GPU/RAM System tab.
        # This is evaluated by the W&B service when the run is initialized.
        wandb_settings = wandb.Settings(
            x_disable_stats=not bool(getattr(cfg.logger, "system_metrics", False)),
        )
        logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.exp_name,
            save_dir=cfg.logger.dir,
            config=OmegaConf.to_container(cfg),
            settings=wandb_settings,
        )

        # On rank != 0, wandb.run is None.
        # if wandb.run is not None:
        #     wandb.run.log_code("src")
    else:
        logger = None
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logger.dir,        # Path where checkpoints will be saved
        filename='{epoch}',        # Filename for the checkpoints
        save_top_k=-1,             # Set to -1 to save all checkpoints
        every_n_epochs=1,          # Save a checkpoint every K epochs
        save_on_train_epoch_end=True,  # Ensure it saves at the end of an epoch, not the beginning
    )

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu",
        logger=logger,
        devices=cfg.device,
        strategy=(
            "ddp_find_unused_parameters_true"
            if len(cfg.device) > 1
            else "auto"
        ),
        callbacks=callbacks + [checkpoint_callback],
        val_check_interval=cfg.train.val_check_interval,
        enable_progress_bar=True,
        gradient_clip_val=cfg.train.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        max_steps=cfg.train.max_steps,
        precision = "32"
    )
    model_wrapper = ModelWrapper(cfg)
    dataset = dataset_dict[cfg.data.dataset_name]
    datamodule = DataModule(dataset, cfg)

    if  cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=datamodule, ckpt_path=cfg.train.ckpt_path)
    else:
        trainer.test(model_wrapper, datamodule=datamodule, ckpt_path=cfg.test.ckpt_path)


if __name__ == '__main__':
    base_conf = OmegaConf.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'config', 'nuscene_train.yaml')
    )
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)
    if 'mode' not in cfg:
            cfg.mode = "eval"
    print_run_summary(cfg)
    main(cfg)
