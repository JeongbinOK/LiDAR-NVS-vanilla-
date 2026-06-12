import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import time
import warnings

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from src.model_wrapper import ModelWrapper
from src.models_new.utils.model_utils import StepTracker, DataModule
from src.dataloader import dataset_dict

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main(cfg):
    os.makedirs(cfg.logger.dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    callbacks = []
    if cfg.logger.enable:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.exp_name,
            save_dir=cfg.logger.dir,
            config=OmegaConf.to_container(cfg),
        )
        callbacks.append(LearningRateMonitor("step", True))

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

    step_tracker = StepTracker()
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
        callbacks=[checkpoint_callback],
        val_check_interval=cfg.train.val_check_interval,
        enable_progress_bar=True,
        # gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.train.max_steps,
        precision = "32"
    )
    #torch.manual_seed(cfg_dict.seed + trainer.global_rank)


    model_wrapper = ModelWrapper(cfg,step_tracker)
    dataset = dataset_dict[cfg.data.dataset_name]
    datamodule = DataModule(dataset, cfg)

    if  cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=datamodule, ckpt_path=cfg.train.ckpt_path)
    else:
        trainer.test(model_wrapper, datamodule=datamodule, ckpt_path=cfg.test.ckpt_path)


if __name__ == '__main__':
    base_conf = OmegaConf.load('/data1/hyuk/LiDAR-NVS-vanilla-/config/nuscene_train.yaml')
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)
    if 'mode' not in cfg:
            cfg.mode = "eval"
    print(cfg)
    main(cfg)