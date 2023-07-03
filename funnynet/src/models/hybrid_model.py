from itertools import combinations
from typing import Any, Dict

from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningModule
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel

from ext.byol_a.byol_a.models import AudioNTT2020
from funnynet.src.models.modules.network import avt_projection, avf_projection
from funnynet.src.models.modules.loss import ContrastiveLossELI5
from funnynet.src.models.modules.vit import TimeSformer

MODALITY_TO_PROJ = {("a", "v", "t"): avt_projection, ("a", "v", "f"): avf_projection}


class HybridModel(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.modalities = config.modalities
        self.learning_rate = config.learning_rate
        self.milestones = config.milestones
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.num_training_samples = config.num_training_samples

        proj_args = dict(n_projection_dims=config.proj_embedding_dim)

        # ################ audio network #################
        if "a" in self.modalities:
            self.audio_encoder = config.audio.encoder
            self.audio_net = AudioNTT2020(d=config.audio.n_dims)
            self.audio_net.load_weight(config.audio.weight_path, device=self.device)

            for param in self.audio_net.parameters():
                param.requires_grad = False
            proj_args["n_audio_dims"] = config.audio.n_dims

        # ################ vision network #################
        if "v" in self.modalities:
            self.vision_encoder = config.vision.encoder
            self.vision_net = TimeSformer(
                img_size=config.vision.img_size,
                num_classes=config.vision.num_classes,
                num_frames=config.vision.num_frames,
                attention_type=config.vision.attention_type,
                pretrained_model=config.vision.weight_path,
            )

            for param in self.vision_net.parameters():
                param.requires_grad = False
            proj_args["n_vision_dims"] = config.vision.n_dims

        # ################ language network #################
        if "t" in self.modalities:
            self.text_encoder = config.text.encoder
            self.text_net = BertModel.from_pretrained(config.text.weight_path)

            for param in self.text_net.parameters():
                param.requires_grad = False
            proj_args["n_text_dims"] = config.text.n_dims

        # ################ projection network #############
        self.clf_head = MODALITY_TO_PROJ[tuple(self.modalities)](**proj_args)

        # ################ criteria #############
        class_weights = torch.FloatTensor([1, 1])
        self.bce_loss = CrossEntropyLoss(size_average=True, weight=class_weights)
        self.eli5_loss = ContrastiveLossELI5(self.batch_size)

        self.test_step_outputs = []

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        labels = batch["labels"]
        if "a" in self.modalities:
            audio = batch["audio"]
            self.audio_net.eval()
        if "v" in self.modalities:
            frames = batch["frames"]
            self.vision_net.eval()
        if "t" in self.modalities:
            text_ids = batch["text_ids"]
            text_masks = batch["text_masks"]
            self.text_net.eval()
        if "f" in self.modalities:
            faces = batch["faces"]

        in_feats = []
        with torch.no_grad():
            # ################ audio network #################
            if "a" in self.modalities:
                audio_feat = self.audio_net(audio)
                in_feats.append(audio_feat)

            # ################ vision network #################
            if "v" in self.modalities:
                vision_feat = self.vision_net.model.forward_features(2 * frames - 1)
                in_feats.append(vision_feat)

            # ################ language network #################
            if "t" in self.modalities:
                output_t = self.text_net(text_ids, attention_mask=text_masks)
                text_feat = output_t.last_hidden_state
                in_feats.append(text_feat)

            # ################ face network #################
            if "f" in self.modalities:
                in_feats.append(faces)

        # ################ projection network #############
        out_feats = self.clf_head(*in_feats)
        preds = out_feats[-1]

        # ################ criteria #############
        loss_dict = dict(preds=preds, labels=labels)
        bce_loss = self.bce_loss(preds, labels)
        loss_dict["bce_loss"] = bce_loss
        if len(self.modalities) > 1:
            feats = out_feats[:-1]
            contrastive_loss = torch.tensor(0.0, device=feats[0].device)
            eli5_loss = torch.tensor(0.0, device=feats[0].device)
            for fx, fy in combinations(feats, r=2):
                eli5_loss += self.eli5_loss(fx, fy)
            contrastive_loss += eli5_loss
            loss_dict["eli5_loss"] = eli5_loss
            loss_dict["contrastive_loss"] = contrastive_loss

        loss = bce_loss + 0.1 * contrastive_loss
        loss_dict["loss"] = loss

        return loss_dict

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        labels = batch["labels"]
        if "a" in self.modalities:
            audio = batch["audio"]
        if "v" in self.modalities:
            frames = batch["frames"]
        if "t" in self.modalities:
            text_ids = batch["text_ids"]
            text_masks = batch["text_masks"]
        if "f" in self.modalities:
            faces = batch["faces"]

        in_feats = []
        # ################ audio network #################
        if "a" in self.modalities:
            audio_feat = self.audio_net(audio)
            in_feats.append(audio_feat)

        # ################ vision network #################
        if "v" in self.modalities:
            vision_feat = self.vision_net.model.forward_features(2 * frames - 1)
            in_feats.append(vision_feat)

        # ################ language network #################
        if "t" in self.modalities:
            output_t = self.text_net(text_ids, attention_mask=text_masks)
            text_feat = output_t.last_hidden_state
            in_feats.append(text_feat)

        # ################ face network #################
        if "f" in self.modalities:
            in_feats.append(faces)

        # ################ projection network #############
        out_feats = self.clf_head(*in_feats)
        preds = out_feats[-1]

        # ################ criteria #############
        loss_dict = dict(preds=preds, labels=labels)
        bce_loss = self.bce_loss(preds, labels)
        loss_dict["bce_loss"] = bce_loss
        if len(self.modalities) > 1:
            feats = out_feats[:-1]
            contrastive_loss = torch.tensor(0.0, device=feats[0].device)
            eli5_loss = torch.tensor(0.0, device=feats[0].device)
            for fx, fy in combinations(feats, r=2):
                eli5_loss += self.eli5_loss(fx, fy)
            contrastive_loss += eli5_loss
            loss_dict["eli5_loss"] = eli5_loss
            loss_dict["contrastive_loss"] = contrastive_loss

        loss = bce_loss + 0.1 * contrastive_loss
        loss_dict["loss"] = loss

        return loss_dict

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        labels = batch["labels"]
        if "a" in self.modalities:
            audio = batch["audio"]
        if "v" in self.modalities:
            frames = batch["frames"]
        if "t" in self.modalities:
            text_ids = batch["text_ids"]
            text_masks = batch["text_masks"]

        in_feats = []
        # ################ audio network #################
        if "a" in self.modalities:
            audio_feat = self.audio_net(audio)
            in_feats.append(audio_feat)

        # ################ vision network #################
        if "v" in self.modalities:
            vision_feat = self.vision_net.model.forward_features(2 * frames - 1)
            in_feats.append(vision_feat)

        # ################ language network #################
        if "t" in self.modalities:
            output_t = self.text_net(text_ids, attention_mask=text_masks)
            text_feat = output_t.last_hidden_state
            in_feats.append(text_feat)

        # ################ projection network #############
        out_feats = self.clf_head(*in_feats)
        preds = out_feats[-1]

        self.test_step_outputs.append({"preds": preds, "labels": labels})

    def on_test_epoch_end(self):
        """Compute metrics over all testing samples"""
        preds, labels = [], []
        for out in self.test_step_outputs:
            preds.append(out["preds"])
            labels.append(out["labels"])
        self.test_step_outputs.clear()

        self.test_outputs = {"preds": torch.cat(preds), "labels": torch.cat(labels)}
        if len(self.test_outputs["labels"].shape) > 1:
            self.test_outputs["preds"] = self.test_outputs["preds"].argmax(dim=1)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Define optimizers and LR schedulers."""
        params = list(self.clf_head.parameters())

        # Initialize the optimizer
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # Initialize the learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
