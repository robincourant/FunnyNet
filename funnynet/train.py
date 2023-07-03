import os.path as osp
from typing import Optional

import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from funnynet.src.data.hybrid_datamodule import HydbridDatamodule
from funnynet.src.callbacks.metrics import MetricLogger
from funnynet.src.models.hybrid_model import HybridModel


def train(config: DictConfig):
    seed_everything(42, workers=True)

    # Split the batch size across gpus
    if config.compnode.strategy == "ddp":
        effective_batch_size = config.model.batch_size
        device_batch_size = effective_batch_size // config.compnode.num_gpus
        config.model.batch_size = device_batch_size

    # Initialize dataset
    with open_dict(config):
        config.data.batch_size = config.model.batch_size
        config.data.num_workers = config.compnode.num_workers
    data_module = HydbridDatamodule(config=config.data)

    # Initialize callbacks
    xp_name = config.xp_name if config.xp_name is not None else config.model.name
    wandb_logger = WandbLogger(
        name=xp_name,
        project=config.project_name,
        offline=config.log_offline,
        save_dir=osp.join(config.output_dir, xp_name),
    )
    checkpoint = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=osp.join(xp_name, config.model.name + "-{epoch}"),
        save_top_k=-1,
    )
    metric_logger = MetricLogger()

    # Initialize model
    with open_dict(config):
        config.model.num_training_samples = len(data_module.train_dataloader())
    model = HybridModel(config=config.model)

    trainer = Trainer(
        devices=config.compnode.num_gpus,
        strategy=config.compnode.strategy,
        accelerator=config.compnode.accelerator,
        max_epochs=config.num_epochs,
        callbacks=[metric_logger, checkpoint],
        logger=wandb_logger,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        # precision=16,
        ############################
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    # Launch model training
    trainer.fit(model, data_module, ckpt_path=config.model.checkpoint)


@hydra.main(version_base="1.3", config_path="./configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    train(config)


if __name__ == "__main__":
    main()
