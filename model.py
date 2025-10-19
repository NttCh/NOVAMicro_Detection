#!/usr/bin/env python
"""Model definitions (including the LightningModule)."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
from utils import load_obj

def build_classifier(cfg: Any, num_classes: int) -> nn.Module:
    """
    Build a classification model given a config.
    Automatically replaces the final FC/Classifier layer to match num_classes.
    """

    backbone_cls = load_obj(cfg.model.backbone.class_name)
    model = backbone_cls(**cfg.model.backbone.params)

    if hasattr(model, "fc"):
        head_name = "fc"
        old_head = model.fc
    elif hasattr(model, "classifier"):
        head_name = "classifier"
        old_head = model.classifier
    else:
        raise ValueError(
            f"Don't know how to replace the head for {cfg.model.backbone.class_name}"
        )

    if isinstance(old_head, nn.Sequential):
        in_features = next(
            m for m in reversed(old_head) if isinstance(m, nn.Linear)
        ).in_features
    else:
        in_features = old_head.in_features

    # build and set the new head
    new_head = nn.Linear(in_features, num_classes)
    setattr(model, head_name, new_head)

    return model
    
class LitClassifier(pl.LightningModule):
    """
    LightningModule wrapping a classifier model with training/validation steps.
    Freezing behavior is driven by cfg.training.freeze_strategy + freeze_backbone_epochs.

    training.freeze_strategy: "none" | "feature_extractor" | "warmup_unfreeze"
    training.freeze_backbone_epochs: int (used by "warmup_unfreeze")
    """

    def __init__(self, cfg: Any, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # ---- freezing config ----
        tr_cfg = getattr(self.cfg, "training", {})
        self._freeze_epochs: int = int(getattr(tr_cfg, "freeze_backbone_epochs", 0))
        self._freeze_strategy: str = str(getattr(tr_cfg, "freeze_strategy", "none")).lower()
        self._froze = False

        # ===== VALIDATION accumulators (epoch-level) =====
        self.register_buffer("val_tp",       torch.tensor(0.0), persistent=False)
        self.register_buffer("val_p",        torch.tensor(0.0), persistent=False)
        self.register_buffer("val_pred_pos", torch.tensor(0.0), persistent=False)
        self.register_buffer("val_correct",  torch.tensor(0.0), persistent=False)
        self.register_buffer("val_total",    torch.tensor(0.0), persistent=False)

        # ===== TRAIN accumulators (epoch-level) =====
        self.register_buffer("train_tp",       torch.tensor(0.0), persistent=False)
        self.register_buffer("train_p",        torch.tensor(0.0), persistent=False)
        self.register_buffer("train_pred_pos", torch.tensor(0.0), persistent=False)
        self.register_buffer("train_correct",  torch.tensor(0.0), persistent=False)
        self.register_buffer("train_total",    torch.tensor(0.0), persistent=False)

    # ---- helpers ----
    def _get_head(self):
        return getattr(self.model, "fc", None) or getattr(self.model, "classifier", None)

    def _set_trainable(self, head_only: bool):
        head = self._get_head()
        head_ids = {id(p) for p in head.parameters()} if head is not None else set()
        for p in self.model.parameters():
            p.requires_grad = (id(p) in head_ids) if head_only else True

    def _freeze_bn_running_stats(self):
        import torch.nn as nn
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    # ------------- Lightning hooks -------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        """
        training.freeze_strategy:
          - "none":            full-model train always
          - "feature_extractor": backbone always frozen (head-only)
          - "warmup_unfreeze": head-only for first N epochs, then unfreeze all
        """
        strategy = self._freeze_strategy
        freeze_epochs = self._freeze_epochs

        if strategy == "none" or (freeze_epochs <= 0 and strategy != "feature_extractor"):
            for p in self.model.parameters():
                p.requires_grad = True
            self._froze = False
            return

        if strategy == "feature_extractor":
            self._set_trainable(head_only=True)
            self._freeze_bn_running_stats()
            self._froze = True
            return

        # warmup_unfreeze
        if self.current_epoch < freeze_epochs:
            self._set_trainable(head_only=True)
            self._freeze_bn_running_stats()
            self._froze = True
        elif self._froze:
            for p in self.model.parameters():
                p.requires_grad = True
            self._froze = False

    # reset accumulators
    def on_training_epoch_start(self) -> None:
        self.train_tp.zero_(); self.train_p.zero_(); self.train_pred_pos.zero_()
        self.train_correct.zero_(); self.train_total.zero_()

    def on_validation_epoch_start(self) -> None:
        self.val_tp.zero_(); self.val_p.zero_(); self.val_pred_pos.zero_()
        self.val_correct.zero_(); self.val_total.zero_()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        images = images.to(self.device); labels = labels.to(self.device)

        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # accumulate TRAIN counts
        tp       = ((preds == 1) & (labels == 1)).sum().to(torch.float32)
        p        = (labels == 1).sum().to(torch.float32)
        pred_pos = (preds == 1).sum().to(torch.float32)
        correct  = (preds == labels).sum().to(torch.float32)
        total    = torch.tensor(labels.numel(), device=loss.device, dtype=torch.float32)
        self.train_tp += tp; self.train_p += p; self.train_pred_pos += pred_pos
        self.train_correct += correct; self.train_total += total

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc",  acc,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # We no longer log train_precision/recall/F1/F2 here because they were
        # computed on augmented, train-mode batches. A fair, full-epoch, eval-mode
        # computation is logged by LocalTrainEvalCallback instead.
        pass


    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        images = images.to(self.device); labels = labels.to(self.device)

        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        tp       = ((preds == 1) & (labels == 1)).sum().to(torch.float32)
        p        = (labels == 1).sum().to(torch.float32)
        pred_pos = (preds == 1).sum().to(torch.float32)
        correct  = (preds == labels).sum().to(torch.float32)
        total    = torch.tensor(labels.numel(), device=loss.device, dtype=torch.float32)

        self.val_tp += tp; self.val_p += p; self.val_pred_pos += pred_pos
        self.val_correct += correct; self.val_total += total

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        device = self.device
        recall    = self.val_tp / self.val_p         if self.val_p > 0         else torch.tensor(0.0, device=device)
        precision = self.val_tp / self.val_pred_pos  if self.val_pred_pos > 0  else torch.tensor(0.0, device=device)
        acc       = self.val_correct / self.val_total if self.val_total > 0    else torch.tensor(0.0, device=device)

        denom_f1 = precision + recall
        f1 = (2 * precision * recall / denom_f1) if denom_f1 > 0 else torch.tensor(0.0, device=device)

        beta2 = 2.0; beta2_sq = beta2 * beta2
        denom_f2 = (beta2_sq * precision + recall)
        f2 = ((1 + beta2_sq) * precision * recall / denom_f2) if denom_f2 > 0 else torch.tensor(0.0, device=device)

        self.log("val_acc",       acc,       on_step=False, on_epoch=True, prog_bar=True,  logger=True, sync_dist=True)
        self.log("val_recall",    recall,    on_step=False, on_epoch=True, prog_bar=False,  logger=True, sync_dist=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_f1",        f1,        on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_f2",        f2,        on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_cls = load_obj(self.cfg.optimizer.class_name)
        optimizer_params = dict(getattr(self.cfg, "optimizer").params)
        optimizer_params.pop("gradient_clip_val", None)
        optimizer_params.pop("dropout", None)

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)

        scheduler_cls = load_obj(self.cfg.scheduler.class_name)
        scheduler_params = getattr(self.cfg, "scheduler").params
        scheduler = scheduler_cls(optimizer, **scheduler_params)

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor
        }]