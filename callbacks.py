#!/usr/bin/env python
"""Custom Callbacks: progress bars, fair evaluation, Excel writer, Optuna."""

import os
import math
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from typing import Optional, Iterable, Dict
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

import config

try:
    from sklearn.metrics import roc_auc_score
    _HAS_AUC = True
except Exception:
    _HAS_AUC = False


# ---------- nice progress ----------
class CleanTQDMProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm(); bar.leave = False; return bar


class TrialFoldProgressCallback(pl.Callback):
    def __init__(self, trial_number: Optional[int] = None, total_trials: Optional[int] = None,
                 fold_number: Optional[int] = None, total_folds: Optional[int] = None):
        super().__init__()
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.fold_number = fold_number
        self.total_folds = total_folds

    def on_train_start(self, trainer, pl_module):
        msgs = []
        if self.trial_number is not None and self.total_trials is not None:
            msgs.append(f"Trial {self.trial_number}/{self.total_trials}")
        if self.fold_number is not None and self.total_folds is not None:
            msgs.append(f"Fold {self.fold_number}/{self.total_folds}")
        if msgs: print(" | ".join(msgs))


class OverallProgressCallback(pl.Callback):
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.total_epochs = trainer.max_epochs
    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        e = trainer.current_epoch + 1
        rem = self.total_epochs - trainer.current_epoch
        print(f"[OverallProgress] Epoch {e}/{self.total_epochs} - Remaining: {rem}")


# ---------- fair evaluator ----------
@torch.no_grad()
def _eval_loader(model: torch.nn.Module, loader: Iterable, device: torch.device) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    n = 0
    total_loss = 0.0

    y_true, y_pred = [], []
    probs_acc = []   # collect softmax probs for AUC

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        total_loss += float(ce(logits, labels).item())
        n += int(labels.numel())

        preds = torch.argmax(logits, dim=1)
        y_pred.extend(preds.detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())

        if _HAS_AUC:
            # collect full softmax probabilities for AUC (binary or multiclass)
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs_acc.append(p)

    if n == 0:
        return dict(loss=0.0, acc=0.0, precision=0.0, recall=0.0, f1=0.0, f2=0.0, auc=float("nan"))

    avg_loss = total_loss / n
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f2   = fbeta_score(y_true, y_pred, beta=2, average="binary", zero_division=0)

    auc = float("nan")
    if _HAS_AUC and probs_acc:
        try:
            probs_full = np.concatenate(probs_acc, axis=0)  # (N, C)
            y_true_np = np.asarray(y_true)
            if probs_full.shape[1] == 2:
                # Binary: positive-class column
                auc = float(roc_auc_score(y_true_np, probs_full[:, 1]))
            else:
                # Multiclass: macro AUC via one-vs-rest
                auc = float(roc_auc_score(y_true_np, probs_full, multi_class="ovr", average="macro"))
        except Exception:
            pass

    return dict(loss=avg_loss, acc=acc, precision=prec, recall=rec, f1=f1, f2=f2, auc=auc)

class LocalFairEvalCallback(pl.Callback):
    """
    Evaluate at epoch end with identical code on:
      - train_eval_loader (train set, val transforms)
      - val_loader
    Show train_f2 & val_f2 on progress bar; keep `train_loss` from training_step.
    """
    def __init__(self, train_eval_loader, val_loader):
        super().__init__()
        self.train_eval_loader = train_eval_loader
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        tr = _eval_loader(pl_module, self.train_eval_loader, device)
        va = _eval_loader(pl_module, self.val_loader, device)

        # publish into callback_metrics
        for k, v in tr.items():
            v = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
            trainer.callback_metrics[f"train_{k}"] = torch.tensor(v)
        for k, v in va.items():
            v = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
            trainer.callback_metrics[f"val_{k}"] = torch.tensor(v)

        # show on progress bar
        pl_module.log("train_f2", trainer.callback_metrics["train_f2"], prog_bar=True, on_epoch=True, logger=True)
        pl_module.log("val_f2",   trainer.callback_metrics["val_f2"],   prog_bar=True, on_epoch=True, logger=True)

        # forward to logger at current epoch
        if trainer.logger:
            step = int(trainer.current_epoch)
            trainer.logger.log_metrics({f"train_{k}": float(v) for k, v in tr.items()}, step=step)
            trainer.logger.log_metrics({f"val_{k}": float(v) for k, v in va.items()}, step=step)


# ---------- Excel writer ----------
class MasterValidationMetricsCallback(pl.Callback):
    """
    Writes:
      - logs: per-epoch metrics taken from callback_metrics (train_* and val_*)
      - per_fold_best: best epoch by val_f2
      - summary: mean±std and 95% CI across folds
    """
    def __init__(self, fold_number: int = 0):
        super().__init__()
        eval_root = os.path.join(config.BASE_SAVE_DIR, "eval")
        os.makedirs(eval_root, exist_ok=True)
        self.excel_path = os.path.join(eval_root, "all_eval_metrics.xlsx")
        self.fold_number = fold_number
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics

        def _tf(name):
            v = cm.get(name)
            try:
                if hasattr(v, "item"): return float(v.item())
                return float(v)
            except Exception:
                return None

        epoch = int(trainer.current_epoch + 1)
        row = {"fold": self.fold_number, "epoch": epoch}

        for prefix in ["train", "val"]:
            for m in ["loss", "acc", "precision", "recall", "f1", "f2", "auc"]:
                key = f"{prefix}_{m}"
                val = _tf(key)
                if val is not None:
                    row[key] = val

        self.rows.append(row)

    def on_train_end(self, trainer, pl_module):
        import numpy as _np

        run_df = pd.DataFrame(self.rows)

        # Merge with existing
        if os.path.exists(self.excel_path):
            try:
                old = pd.read_excel(self.excel_path, sheet_name=None)
                logs_df = pd.concat([old.get("logs", pd.DataFrame()), run_df], ignore_index=True)
            except Exception:
                logs_df = run_df.copy()
        else:
            logs_df = run_df.copy()

        # Ensure columns exist
        for c in ["val_f2", "val_recall", "val_precision", "val_loss", "val_acc", "val_auc",
                  "train_f2", "train_loss"]:
            if c not in logs_df.columns:
                logs_df[c] = _np.nan

        # Best per fold by val_f2
        tmp = logs_df.dropna(subset=["val_f2"]).copy()
        if "fold" not in tmp.columns:
            tmp["fold"] = 0
        tmp = tmp.sort_values(["fold", "val_f2", "epoch"], ascending=[True, False, False])
        per_fold_best = tmp.groupby("fold", as_index=False).first().sort_values("fold")

        # Summary (t-based 95% CI, fallback to z=1.96)
        def _tcrit(df_deg):
            try:
                from scipy import stats as _st
                return float(_st.t.ppf(0.975, df=df_deg)) if df_deg > 0 else _np.nan
            except Exception:
                return 1.96 if df_deg > 0 else _np.nan

        k = int(per_fold_best["fold"].nunique()) if len(per_fold_best) else 0
        tcrit = _tcrit(k - 1)

        metrics = ["val_f2", "val_recall", "val_precision", "val_loss", "val_acc", "val_auc"]
        summary_rows = []
        for m in metrics:
            vals = per_fold_best[m].astype(float).to_numpy() if m in per_fold_best.columns else _np.array([])
            if len(vals) == 0:
                mean = std = ci_l = ci_u = _np.nan
            else:
                mean = float(_np.mean(vals))
                std  = float(_np.std(vals, ddof=1)) if k > 1 else 0.0
                se   = (std / _np.sqrt(k)) if k > 1 else _np.nan
                ci_l = (mean - tcrit * se) if k > 1 else _np.nan
                ci_u = (mean + tcrit * se) if k > 1 else _np.nan
            summary_rows.append({"metric": m, "k_folds": k, "mean": mean, "std": std,
                                 "95%_CI_lower": ci_l, "95%_CI_upper": ci_u})
        summary_df = pd.DataFrame(summary_rows)

        # Atomic write
        tmp_path = self.excel_path + ".writing.xlsx"
        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="w") as w:
                logs_df.to_excel(w, sheet_name="logs", index=False)
                per_fold_best.to_excel(w, sheet_name="per_fold_best", index=False)
                summary_df.to_excel(w, sheet_name="summary", index=False)
            os.replace(tmp_path, self.excel_path)
            print(f"[MasterValidationMetricsCallback] wrote → {self.excel_path}")
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass


# ---------- Optuna composite report/prune ----------
class OptunaCompositeReportingCallback(pl.Callback):
    """
    Report a chosen validation metric to Optuna once per epoch and allow pruning.
    Default metric: 'val_f2' (which is logged by LocalFairEvalCallback).
    """
    def __init__(self, trial, cfg, metric_name: str = "val_f2"):
        super().__init__()
        self.trial = trial
        self.cfg = cfg
        self.metric_name = metric_name
        self._last_reported_step: Optional[int] = None

    def _as_float(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item())
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    def on_validation_epoch_end(self, trainer, pl_module):
        step = int(trainer.current_epoch)
        if self._last_reported_step == step:
            return
        m = trainer.callback_metrics.get(self.metric_name)
        val = self._as_float(m)
        if val is None:
            return
        self.trial.report(val, step=step)
        self._last_reported_step = step
        from optuna.exceptions import TrialPruned
        if self.trial.should_prune():
            raise TrialPruned()

class LocalTrainEvalCallback(pl.Callback):
    """
    After each epoch, run a CLEAN evaluation on the TRAIN set using
    the validation transforms (train_eval_loader) and log fair train metrics.
    """
    def __init__(self, train_eval_loader):
        super().__init__()
        self.train_eval_loader = train_eval_loader
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # eval pass (no dropout; BN frozen)
        pl_module.eval()

        all_preds, all_labels = [], []
        total_loss, batches = 0.0, 0
        all_probs_pos = []

        for images, labels in self.train_eval_loader:
            images = images.to(pl_module.device)
            labels = labels.to(pl_module.device)

            logits = pl_module(images)
            loss = self.criterion(logits, labels)
            total_loss += float(loss.item()); batches += 1

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs_pos.extend(probs.detach().cpu().numpy())

        if batches == 0:
            return

        avg_loss = total_loss / batches
        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        rec  = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1   = fbeta_score(all_labels, all_preds, beta=1, average='binary', zero_division=0)
        f2   = fbeta_score(all_labels, all_preds, beta=2, average='binary', zero_division=0)

        # Log as epoch-level metrics; make train_f2 visible on the progress bar
        pl_module.log("train_loss",      avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_acc",       acc,      on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_precision", prec,     on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_recall",    rec,      on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_f1",        f1,       on_step=False, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("train_f2",        f2,       on_step=False, on_epoch=True, prog_bar=True,  logger=True)

        # back to training mode for next epoch
        pl_module.train()
