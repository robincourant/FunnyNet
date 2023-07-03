from typing import Any, Dict

from pytorch_lightning import Callback, LightningModule, Trainer
import torch
import torchmetrics.functional as M


class MetricLogger(Callback):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _log_metrics(
        metric_dict: Dict[str, torch.Tensor],
        on_step: bool,
        pl_module: LightningModule,
        mode: str,
    ):
        for metric_name, metric_value in metric_dict.items():
            pl_module.log(
                f"{mode}/{metric_name}",
                metric_value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=pl_module.batch_size,
            )

    def _get_metric_dict(
        self, preds: torch.Tensor, labels: torch.Tensor, prefix: str = ""
    ) -> Dict[str, torch.Tensor]:
        accuracy = M.accuracy(preds, labels, task="binary")
        precision = M.precision(preds, labels, task="binary")
        recall = M.recall(preds, labels, task="binary")
        f1_score = M.f1_score(preds, labels, task="binary")

        metric_dict = {
            f"{prefix}accuracy": accuracy.data,
            f"{prefix}precision": precision.data,
            f"{prefix}recall": recall.data,
            f"{prefix}f1_score": f1_score.data,
        }

        return metric_dict

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
    ):
        if batch_idx == 0:
            self.train_outputs = []

        preds = outputs["preds"].argmax(dim=1).detach()
        labels = outputs["labels"].detach()
        bce_loss = outputs["bce_loss"].data
        contrastive_loss = outputs["contrastive_loss"].data
        total_loss = outputs["loss"].data
        eli5_loss = outputs["eli5_loss"].data

        # Log batch-wise metrics
        metric_dict = self._get_metric_dict(preds, labels, "step/")
        metric_dict["step/bce_loss"] = bce_loss
        metric_dict["step/contrastive_loss"] = contrastive_loss
        metric_dict["step/total_loss"] = total_loss
        metric_dict["step/eli5_loss"] = eli5_loss
        self._log_metrics(metric_dict, True, pl_module, "train")

        # Store batch data
        out_dict = {
            "labels": labels,
            "preds": preds,
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "bce_loss": bce_loss,
        }
        out_dict["eli5_loss"] = eli5_loss
        self.train_outputs.append(out_dict)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx == 0:
            self.val_outputs = []

        # Skip sanity check steps
        if trainer.sanity_checking:
            return

        preds = outputs["preds"].argmax(dim=1).detach()
        labels = outputs["labels"].detach()
        bce_loss = outputs["bce_loss"].data
        contrastive_loss = outputs["contrastive_loss"].data
        total_loss = outputs["loss"].data

        # Log batch-wise metrics
        metric_dict = self._get_metric_dict(preds, labels, "step/")
        metric_dict["step/bce_loss"] = bce_loss
        metric_dict["step/contrastive_loss"] = contrastive_loss
        metric_dict["step/total_loss"] = total_loss
        self._log_metrics(metric_dict, False, pl_module, "val")

        # Store batch data
        self.val_outputs.append(
            {
                "labels": labels,
                "preds": preds,
                "total_loss": total_loss,
                "bce_loss": bce_loss,
                "contrastive_loss": contrastive_loss,
            }
        )

    def on_train_epoch_end(self, trainer, pl_module):
        preds = torch.cat([x["preds"] for x in self.train_outputs])
        labels = torch.cat([x["labels"] for x in self.train_outputs])
        bce_loss = torch.stack([x["bce_loss"] for x in self.train_outputs])
        contrastive_loss = torch.stack(
            [x["contrastive_loss"] for x in self.train_outputs]
        )
        total_loss = torch.stack([x["total_loss"] for x in self.train_outputs])

        metric_dict = self._get_metric_dict(preds, labels, "global/")
        metric_dict["global/bce_loss"] = bce_loss.mean().data
        metric_dict["global/contrastive_loss"] = contrastive_loss.mean().data
        metric_dict["global/total_loss"] = total_loss.mean().data
        self._log_metrics(metric_dict, False, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip sanity check steps
        if trainer.sanity_checking:
            return

        preds = torch.cat([x["preds"] for x in self.val_outputs])
        labels = torch.cat([x["labels"] for x in self.val_outputs])
        bce_loss = torch.stack([x["bce_loss"] for x in self.val_outputs])
        contrastive_loss = torch.stack([x["contrastive_loss"] for x in self.val_outputs])
        total_loss = torch.stack([x["total_loss"] for x in self.val_outputs])

        metric_dict = self._get_metric_dict(preds, labels, "global/")
        metric_dict["global/bce_loss"] = bce_loss.mean().data
        metric_dict["global/contrastive_loss"] = contrastive_loss.mean().data
        metric_dict["global/total_loss"] = total_loss.mean().data
        self._log_metrics(metric_dict, False, pl_module, "val")
