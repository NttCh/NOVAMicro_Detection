# Microorganism Binary Classification with PyTorch Lightning and Optuna

This repository trains, tunes, and evaluates a **binary classifier** for stained microscopy images using **PyTorch Lightning**.  
It includes modular components for configuration, data loading, model building, callbacks, training, inference, explainability (**Grad-CAM++**), and hyperparameter optimization (**Optuna**).

---

## Key Features
- **Three modes:** `train`, `tune` (Optuna), `test`
- **Cross-Validation:** K-fold with optional repeated runs and plots
- **Robust ckpt handling:** load/merge `.ckpt` weights even with prefix mismatches
- **Reusable HPO:** retrain from a chosen/best Optuna trial
- **Grad-CAM++ XAI:** heatmaps, overlays, original+overlay panels, evidence boxes
- **Excel outputs:** per-epoch metrics, per-folder predictions, multi-ckpt comparisons

---

## Quick Start (3 steps)

1. Edit **`config.py`**
- Set paths under `data.*` and `test.folder_path`
- Choose `run_mode`:
  - `"train"` → **Train a model**
    - If `reuse.enable=False`: trains from scratch using images in  
      `dataset/train_bac` *(positive)* and `dataset/train_nonbac` *(negative)*
    - If `reuse.enable=True`: reuses the best or selected trial from an existing Optuna study (e.g., `optuna.db`) and retrains with those hyperparameters
  - `"tune"` → run Optuna hyperparameter search  
  - `"test"` → evaluate pretrained model(s) on folders in `test/test_data`
- (Optional) toggle:
  - `use_cv`: enable stratified k-fold cross-validation  
  - `xai.enabled`: generate Grad-CAM⁺⁺ heatmaps for explainability  
  - `optuna.*`: adjust number of trials, samplers, and pruning logic

2. If mode = `"train"` or `"tune"`, Prepare the dataset under `data/train_bac` *(positive)* and `data/train_nonbac` *(negative)*

3. Run **`main.py`**

---

## Configuration (how to set it)
All behavior is driven by `config.py` (OmegaConf DictConfig created from `CFG_DICT`).  
You mostly edit these fields:

- **Mode & CV**
  - `run_mode`: `"train"` | `"tune"` | `"test"`
  - `use_cv`: `True/False`
  - `training.num_folds`, `training.repeated_cv`

- **Reuse (optional)**
  - `reuse.enable`: `True` to retrain with params from a previous Optuna study
  - `reuse.retrain_trial_id`: `-1` = best trial; `>=0` = specific trial id
  - `reuse.study_name`: name of your Optuna study

- **Paths**
  - `general.save_dir`, `general.project_name`
  - `data.negative_dir`, `data.positive_dir`, `data.detection_csv`, `data.folder_path`
  - `test.folder_path`: list of external test folders
  - `pretrained_ckpt`: list of ckpt paths (used in `test`, ignored in `tune`, optional in `train`)

- **Training**
  - `training.seed`
  - `training.tuning_epochs_detection` (also used as `trainer.max_epochs` when not set)
  - Early stopping & patience: `training.early_stopping.*`
  - Freezing: `training.freeze_strategy` (`none` | `feature_extractor` | `warmup_unfreeze`), `freeze_backbone_epochs`

- **Optimization**
  - `optimizer.class_name` (e.g., `"torch.optim.AdamW"`) + `optimizer.params` (`lr`, `weight_decay`)
  - `scheduler.class_name`, `scheduler.monitor`, `scheduler.params` (`mode`, `factor`, `patience`)

- **Transforms**
  - `augmentation.train.augs` / `augmentation.valid.augs` (Albumentations dicts)
  - `augmentation.hpo_resize` / `final_resize`

- **Optuna**
  - `optuna.n_trials`
  - `optuna.sampler` / `optuna.pruner`
  - `optuna.params` (search space: lr, weight_decay, batch_size, gradient_clip_val, …)

- **XAI (Grad-CAM++)**
  - `xai.enabled`
  - `xai.target_layer`: `"auto"` or list like `["layer2","layer3","layer4"]`
  - Output options: `save_heatmap`, `save_overlay`, `save_original_overlay`, `panel`
  - Stabilizers: `tta`, `smooth`, `smooth_n`, `smooth_sigma`
  - `class_names`, `mask_scale_bar`, `mask_rect_wh`


---
