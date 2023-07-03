import json
import os
import os.path as osp
from typing import Optional

import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
import torchmetrics.functional as M

from funnynet.src.data.hybrid_datamodule import HydbridDatamodule
from funnynet.src.models.hybrid_model import HybridModel


def evaluate(config: DictConfig):
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

    # Initialize model
    with open_dict(config):
        config.model.num_training_samples = len(data_module.train_dataloader())
    if config.model.checkpoint is None:
        model = HybridModel(config=config.model)
    else:
        model = HybridModel.load_from_checkpoint(
            config.model.checkpoint, config=config.model
        )

    trainer = Trainer(
        devices=1,
        # accelerator=config.compnode.accelerator,
        logger=False,
        # precision=16,
        num_sanity_val_steps=0,
    )

    # Launch model evaluation
    trainer.test(model, data_module)

    # Compute evaluation metrics
    preds = trainer.model.test_outputs["preds"]
    labels = trainer.model.test_outputs["labels"]

    accuracy = M.accuracy(preds, labels, task="binary").item()
    precision = M.precision(preds, labels, task="binary").item()
    recall = M.recall(preds, labels, task="binary").item()
    f1_score = M.f1_score(preds, labels, task="binary").item()
    outputs = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    print(
        f"accuracy: {accuracy:.2%} | precision: {precision:.2%}  |",
        f" recall: {recall:.2%} | f1_score: {f1_score:.2%}",
    )
    # Save the test ouptuts
    xp_name = config.xp_name if config.xp_name is not None else config.model.name
    result_dir = osp.join(config.result_dir, xp_name)
    os.makedirs(result_dir, exist_ok=True)
    if config.model.checkpoint is not None:
        save_path = osp.join(result_dir, f"{config.model.checkpoint[-13:-5]}.json")
    else:
        save_path = osp.join(result_dir, f"res-{len(os.listdir(result_dir))}.json")
    with open(save_path, "w") as f:
        json.dump(outputs, f)


@hydra.main(version_base="1.3", config_path="./configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    evaluate(config)


if __name__ == "__main__":
    main()
