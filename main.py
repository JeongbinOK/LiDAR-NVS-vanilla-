import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import time
import warnings
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from src.model_wrapper import ModelWrapper
from dataloader import dataset_dict

def main(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="eval",
                        help="train/eval, defaul=eval")



    # Wandb logging => 필요한 거 올리기 (추후 얘기)
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logger.dir,        # Path where checkpoints will be saved
        filename='{epoch}',        # Filename for the checkpoints
        save_top_k=-1,             # Set to -1 to save all checkpoints
        every_n_epochs=1,          # Save a checkpoint every K epochs
        save_on_train_epoch_end=True,  # Ensure it saves at the end of an epoch, not the beginning
    )

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=False if cfg.mode == "train" else True,
        # gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )
    #torch.manual_seed(cfg_dict.seed + trainer.global_rank)


    model_wrapper = ModelWrapper(
        cfg,
        step_tracker,
    )

    if  parser.mode == "train":
        # data loader
        dataset = dataset_dict[cfg.dataset_name]
        dataloader = DataLoader(dataset(cfg = cfg.dataset, split="train"), 
                                batch_size=cfg.train.batch_size,
                                num_workers=8, 
                                shuffle=True,
                                pin_memory=True,)
        trainer.fit(model_wrapper, datamodule=dataloader, ckpt_path=checkpoint_path)
    else:
        # data loader
        dataset = dataset_dict[cfg.dataset_name]
        dataloader = DataLoader(dataset(cfg=cfg.dataset, split="test"), 
                                batch_size=cfg.train.batch_size,
                                num_workers=8, 
                                shuffle=True,
                                pin_memory=True,)
        trainer.test(
            model_wrapper,
            datamodule=dataloader,
            ckpt_path=checkpoint_path,
        )


if __name__ == '__main__':

    base_conf = OmegaConf.load('')
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)
    main(cfg)