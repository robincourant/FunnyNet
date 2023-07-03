from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from funnynet.src.data.datasets.hybrid_dataset import HybridDataset


class HydbridDatamodule(LightningDataModule):
    """A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = HybridDataset(split="train", config=self.config)
        weights = torch.tensor([1, 1], dtype=torch.float)
        train_targets = self.train_set.get_classes_for_all_imgs()
        sample_weights = weights[train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(
            self.train_set,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Load val set loader."""
        self.val_set = HybridDataset(split="validation", config=self.config)
        return DataLoader(
            self.val_set,
            drop_last=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = HybridDataset(split="test", config=self.config)
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )


if __name__ == "__main__":
    _ = HydbridDatamodule()
