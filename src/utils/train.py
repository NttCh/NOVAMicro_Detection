#!/usr/bin/env python
"""Training utilities and main training functions (train metrics = full-epoch eval)."""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger as _PLLoggerBase
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

import config as config
from config import cfg
from .callbacks import (
    MasterValidationMetricsCallback,
    OptunaCompositeReportingCallback,
    CleanTQDMProgressBar,
    TrialFoldProgressCallback,
    OverallProgressCallback,
    LocalTrainEvalCallback,
)
from .data import PatchClassificationDataset
from src.model.model import LitClassifier, build_classifier
from .utils import load_obj, set_seed, thai_time, cleanup_cuda


class NullLogger(_PLLoggerBase):
    """No-op logger to satisfy self.log(..., logger=True) without writing files."""

    def __init__(self) -> None:
        super().__init__()
        self._name = "noop"
        self._version = "0"

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def experiment(self):
        return None

    def log_hyperparams(self, params) -> None:
        pass

    def log_metrics(self, metrics, step: int) -> None:
        pass

    def save(self) -> None:
        pass

    def finalize(self, status: str) -> None:
        pass


# ---------------- logger helper ----------------
def build_logger(cfg, run_id: str) -> TensorBoardLogger | NullLogger:
    """
    In HPO (run_mode == 'tune'), attach a NullLogger to avoid Lightning warnings
    from self.log(..., logger=True). Otherwise, use TensorBoardLogger.
    """
    is_tune = str(getattr(cfg, "run_mode", "")).lower() == "tune"
    if is_tune:
        return NullLogger()

    save_dir = os.path.join(config.BASE_SAVE_DIR, run_id)
    os.makedirs(save_dir, exist_ok=True)
    return TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}")


# ---------------- train_stage ----------------
def train_stage(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    trial: Optional["optuna.trial.Trial"] = None,
    suppress_metrics: bool = False,
    trial_number: Optional[int] = None,
    total_trials: Optional[int] = None,
    fold_number: Optional[int] = None,
    total_folds: Optional[int] = None,
    train_idx: Optional[np.ndarray] = None,
    valid_idx: Optional[np.ndarray] = None,
) -> Tuple[LitClassifier, float]:
    """
    Train one stage and return (model, score).

    The returned score equals the metric picked by:
        cfg.training.early_stop_metric  (default: "val_f2")
    """
    if bool(getattr(getattr(cfg, "training", {}), "clear_cuda_before_stage", False)) and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # switches
    is_tune = str(getattr(cfg, "run_mode", "")).lower() == "tune"
    suppress_artifacts = bool(getattr(cfg.training, "suppress_artifacts", False)) or is_tune
    disable_ckpt = bool(getattr(getattr(cfg, "training", {}), "disable_checkpointing", False))
    enable_ckpt = (not suppress_artifacts) and (not disable_ckpt)

    # split
    full_df = pd.read_csv(csv_path)
    if train_idx is not None and valid_idx is not None:
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        valid_df = full_df.iloc[valid_idx].reset_index(drop=True)
        print(f"[INFO] Using provided CV split → Train size: {len(train_df)} | Valid size: {len(valid_df)}")
    else:
        train_df, valid_df = train_test_split(
            full_df, test_size=cfg.data.valid_split, random_state=cfg.training.seed, stratify=full_df[cfg.data.label_col]
        )
        print(f"[INFO] Random split → Train size: {len(train_df)} | Valid size: {len(valid_df)}")

    # transforms
    train_tf = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_tf = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])

    # datasets
    common_cols = dict(image_col=getattr(cfg.data, "image_col", "filename"), label_col=cfg.data.label_col)
    train_ds = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_tf, **common_cols)
    valid_ds = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_tf, **common_cols)
    train_eval_ds = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=valid_tf, **common_cols)

    # loaders
    base_loader_args = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, persistent_workers=cfg.data.num_workers > 0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **base_loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, **base_loader_args)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **base_loader_args)

    # model (+ optional pretrained outside HPO)
    def _normalize_ckpts(x):
        try:
            from omegaconf import ListConfig
            seq = (list, tuple, ListConfig)
        except Exception:
            seq = (list, tuple)

        def ok(p): return isinstance(p, str) and p.strip() and p.strip().lower() != "none"
        if x is None:
            return []
        if isinstance(x, seq):
            return [p for p in x if ok(p)]
        if isinstance(x, str):
            return [x] if ok(x) else []
        return []

    model = build_classifier(cfg, num_classes=num_classes)
    if not is_tune:
        ckpts = _normalize_ckpts(getattr(cfg, "pretrained_ckpt", None))
        if ckpts:
            try:
                print(f"Loading pretrained checkpoint from {ckpts[0]}")
                raw_sd = torch.load(ckpts[0], map_location="cpu")
                sd = {k.replace("model.", ""): v for k, v in raw_sd.items()}
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[WARN] Could not load pretrained_ckpt: {e}")
    lit_model = LitClassifier(cfg=cfg, model=model, num_classes=num_classes)

    # run dir & logger
    run_id = f"{stage_name}_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"
    logger = build_logger(cfg, run_id)
    save_dir = None
    if not suppress_artifacts:
        save_dir = os.path.join(config.BASE_SAVE_DIR, run_id)
        os.makedirs(save_dir, exist_ok=True)

    # epochs
    max_epochs = getattr(getattr(cfg, "trainer", {}), "max_epochs", None)
    if not max_epochs:
        max_epochs = cfg.training.tuning_epochs_detection

    # callbacks
    callbacks: List[pl.Callback] = []
    monitor_metric = getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_f2")
    _maximize = {"val_f2", "val_f1", "val_recall", "val_precision", "val_acc", "val_auc"}
    monitor_mode = "max" if monitor_metric in _maximize else "min"

    if enable_ckpt:
        mc = ModelCheckpoint(
            dirpath=save_dir,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=False,
            filename=f"{stage_name}-" + "{epoch:02d}-{" + monitor_metric + ":.4f}",
        )

    patience = (cfg.training.early_stopping.patience_hpo if trial is not None else cfg.training.early_stopping.patience_final)
    core_callbacks = [
        OverallProgressCallback(),
        TrialFoldProgressCallback(trial_number=trial_number, total_trials=total_trials, fold_number=fold_number, total_folds=total_folds),
        LocalTrainEvalCallback(train_eval_loader=train_eval_loader),
        CleanTQDMProgressBar(),
    ]
    if not suppress_metrics:
        callbacks += [EarlyStopping(monitor=monitor_metric, patience=int(patience), mode=monitor_mode), *([mc] if enable_ckpt else []), *core_callbacks]
    else:
        if enable_ckpt:
            callbacks.append(mc)
        callbacks += core_callbacks

    if trial is not None:
        callbacks.append(OptunaCompositeReportingCallback(trial, cfg, metric_name=monitor_metric))

    if not suppress_artifacts:
        callbacks.append(MasterValidationMetricsCallback(fold_number=fold_number))

    # trainer
    n_train_batches = len(train_loader)
    log_every_n_steps = max(1, min(50, n_train_batches))
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=getattr(cfg.trainer, "gradient_clip_val", None),
        logger=(logger if logger is not None else False),
        callbacks=callbacks,
        enable_model_summary=False,
        enable_checkpointing=enable_ckpt,
        log_every_n_steps=log_every_n_steps,
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # best checkpoint reload
    best_path = ""
    if enable_ckpt:
        best_path = locals().get("mc", None)
        best_path = getattr(best_path, "best_model_path", "") if best_path else ""

    if enable_ckpt and best_path and os.path.exists(best_path):
        print(f"[train_stage] Reloading BEST weights from: {best_path}")
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location="cpu")

        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[train_stage] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

        try:
            import shutil
            best_dir = os.path.join(config.BASE_SAVE_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            dst_ckpt = os.path.join(best_dir, f"best_{stage_name}.ckpt")
            shutil.copy2(best_path, dst_ckpt)
            lit_model.best_ckpt_path = dst_ckpt
            print(f"[train_stage] Copied best checkpoint → {dst_ckpt}")
            # NOTE: removed .pt saving entirely
        except Exception as e:
            print(f"[train_stage] Could not archive best artifacts: {e}")
    else:
        if enable_ckpt:
            print("[train_stage] WARNING: No best_model_path found; returning last-epoch weights.")

    # score (monitored metric)
    def _to_float(x):
        try:
            import torch as _t
            if isinstance(x, _t.Tensor):
                return float(x.detach().cpu().item())
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return None

    metric_value = trainer.callback_metrics.get(monitor_metric)
    score = _to_float(metric_value)

    cleanup_cuda(logger=logger if logger else None, trainer=trainer, models=[lit_model], dataloaders=[train_loader, valid_loader, train_eval_loader])

    return lit_model, float(score)


# ---------------- train_with_cross_validation ----------------
def train_with_cross_validation(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    cv_run: int,
    total_cv: int,
    verbose: bool = True,
    trial: Optional["optuna.trial.Trial"] = None,
) -> Tuple["LitClassifier", float]:
    """
    Single run of K-fold stratified CV.
    Returns the best-fold model and the mean monitored score across folds.
    """
    monitor_metric = getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_f2")
    full_df: pd.DataFrame = pd.read_csv(csv_path)

    skf = StratifiedKFold(n_splits=int(cfg.training.num_folds), shuffle=True, random_state=int(cfg.training.seed))
    val_scores: List[float] = []
    fold_models: List["LitClassifier"] = []
    splits = list(skf.split(full_df, full_df[cfg.data.label_col]))

    for fold, (t_idx, v_idx) in enumerate(splits):
        fold_num = fold + 1
        fold_seed = int(cfg.training.seed) + fold
        set_seed(fold_seed)

        if verbose:
            print(f"CV {cv_run}/{total_cv} | Fold {fold_num}/{cfg.training.num_folds}: ", end="")

        lit_model, val_metric = train_stage(
            cfg=cfg,
            csv_path=csv_path,
            num_classes=num_classes,
            stage_name=f"{stage_name}_fold{fold_num}",
            trial=trial,
            fold_number=fold_num,
            total_folds=int(cfg.training.num_folds),
            train_idx=t_idx,
            valid_idx=v_idx,
        )

        if verbose:
            print(f"| {monitor_metric}: {val_metric:.4f}")

        val_scores.append(float(val_metric))
        fold_models.append(lit_model)

    best_idx = int(np.argmax(val_scores))
    mean_score = float(np.mean(val_scores))

    if verbose:
        std_score = float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0
        summary = " | ".join([f"F{idx+1}:{s:.4f}" for idx, s in enumerate(val_scores)])
        print(f"[CV Summary] {summary}  →  mean±std({monitor_metric}) = {mean_score:.4f} ± {std_score:.4f}")

    return fold_models[best_idx], mean_score


# ---------------- repeated_cross_validation ----------------
def repeated_cross_validation(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    repeats: int,
    trial: Optional["optuna.trial.Trial"] = None,
) -> Tuple["LitClassifier", float]:
    """
    Perform repeated K-fold CV `repeats` times with different seeds.
    Returns the best model among repeats and the mean monitored score.
    """
    monitor_metric = getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_f2")
    base_seed = int(cfg.training.seed)

    all_scores: List[float] = []
    best_models: List["LitClassifier"] = []

    for r in range(int(repeats)):
        local_seed = base_seed + r
        set_seed(local_seed)

        prev_cfg_seed = int(cfg.training.seed)
        cfg.training.seed = local_seed
        try:
            print(f"\n=== Repeated CV run {r+1}/{repeats} (seed={local_seed}) — optimizing {monitor_metric} ===")
            model_cv, avg_score = train_with_cross_validation(
                cfg=cfg,
                csv_path=csv_path,
                num_classes=num_classes,
                stage_name=stage_name,
                cv_run=r + 1,
                total_cv=int(repeats),
                verbose=True,
                trial=trial,
            )
        finally:
            cfg.training.seed = prev_cfg_seed

        all_scores.append(float(avg_score))
        best_models.append(model_cv)

    mean_over_repeats = float(np.mean(all_scores))
    std_over_repeats = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
    print(f"\n[Repeated CV] mean±std({monitor_metric}) over {repeats} runs = {mean_over_repeats:.4f} ± {std_over_repeats:.4f}")

    best_idx = int(np.argmax(all_scores))
    return best_models[best_idx], mean_over_repeats


# ---------------- optuna progress print ----------------
def print_trial_thai_callback(study, trial) -> None:
    """Optuna callback: print best value & Thai timestamp on completion."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        metric_name = getattr(getattr(cfg, "training", {}), "early_stop_metric", "val_f2")
        print(f"[Optuna] Trial complete with {metric_name}={trial.value:.4f} at {thai_time()}")
