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

    def _compact_val_subset(self):
        """Path of the compact-val manifest, or None for full-split validation.

        Per-epoch validation over the whole val split is ~5.7k heavily
        overlapping windows dominated by geometry the velocity head cannot
        change. `data.compact_val` swaps in a fixed motion-weighted subset
        (180 dynamic + 20 static) built by tools/build_compact_val_windows.py.
        """
        compact = getattr(self.cfg.data, "compact_val", None)
        if compact is None or not bool(getattr(compact, "enable", False)):
            return None
        path = getattr(compact, "path", None)
        if not path:
            raise ValueError(
                "data.compact_val.enable=true requires data.compact_val.path"
            )
        return str(path)

    def val_dataloader(self):  
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "eval_split", "val"),
                window_subset=self._compact_val_subset(),
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
