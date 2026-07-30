import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ...dataloader.nuscene import multiframe_collate_fn


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, cfg):
        super().__init__()  
        self.dataset = dataset
        self.cfg = cfg  

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "train_split", "train"),
            ),
            batch_size=self.cfg.train.batch_size,  
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn = multiframe_collate_fn
        )

    def val_dataloader(self):  
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "eval_split", "val"),
            ),
            batch_size=self.cfg.test.batch_size,  
            num_workers=int(getattr(self.cfg.data, "eval_num_workers", 0)),
            shuffle=False,
            pin_memory=True,
            collate_fn = multiframe_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "test_split", "test"),
            ),
            batch_size=self.cfg.test.batch_size,  
            num_workers=int(getattr(self.cfg.data, "eval_num_workers", 0)),
            shuffle=False,
            pin_memory=True,
            collate_fn = multiframe_collate_fn
        )
