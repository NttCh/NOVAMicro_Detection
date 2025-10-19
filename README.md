# Microorganism Binary Classification with PyTorch Lightning and Optuna

This repository contains code for training a microorganism binary classification model using **PyTorch Lightning**. The code is modular, supporting configuration management, data handling, model building, callbacks, training, inference, explainability, and hyperparameter tuning.

---

## Project Structure

- **Configuration & Mode Settings**  
  The code supports multiple operational modes:
  - **Training Mode**
    - Runs the complete training pipeline.
    - **Tuning Mode:** (`tuning_mode=True`) Hyperparameters are tuned using [Optuna](https://optuna.org). Any provided pretrained checkpoint is ignored.
    - **Cross-Validation (CV):** When enabled (`use_cv=True`), training runs on multiple folds with metrics aggregated.
  - **Test Only Mode**
    - Loads a pretrained checkpoint and runs inference on one or more specified test folders.
    - Saves predictions to Excel (with filenames, predicted labels, probabilities, and thumbnails).
    - Generates evaluation plots (ROC curve, confusion matrix).
    - Optionally produces explainability (Grad-CAM) outputs.

---

## Configuration Settings

All pipeline behavior is controlled by a configuration file (OmegaConf). Key settings:

- **run_mode:** `"train"` or `"test"`  
  Selects between full training or test-only inference.

- **tuning_mode:** `True` or `False`  
  Enables Optuna hyperparameter tuning. If `True`, pretrained checkpoints are ignored.

- **use_cv:** `True` or `False`  
  Enables cross-validation during training.

- **use_optuna:** `True` or `False`  
  If `True`, Optuna runs multiple trials to find the best parameters.

- **general**
  - `save_dir`: Base directory for all outputs (logs, models, plots).
  - `project_name`: Identifier used for logging and output naming.

- **trainer**
  - Devices, accelerator type, mixed precision, gradient clipping.

- **training**
  - `seed`: Random seed for reproducibility.
  - `tuning_epochs_detection`: Epochs for tuning/training.
  - `additional_epochs_detection`: Optional fine-tuning epochs.
  - `cross_validation`: Enable/disable CV.
  - `num_folds`: Number of folds in CV.
  - `repeated_cv`: Number of repeated CV runs.
  - `composite_metric`: `{alpha, beta}` for composite score:  
    `Composite = alpha * Recall - beta * Loss`
  - `freeze_backbone_epochs`: Number of epochs to freeze the backbone during tuning (speeds up training).

- **optimizer & scheduler**  
  - Defines optimizer class (e.g., `torch.optim.AdamW`) and parameters (lr, weight decay).  
  - Scheduler settings (e.g., `ReduceLROnPlateau`).

- **model**  
  - Backbone model (e.g., ResNet101 pretrained on ImageNet).  
  - Final layer is replaced to match binary classification.

- **data**  
  - Paths for positive/negative image folders.  
  - Path to training CSV.  
  - Batch size, workers, label column, validation split.  
  - Optionally auto-generate CSV from folders.

- **augmentation**  
  - Albumentations transforms for training/validation.

- **test**  
  - Test folder(s).  
  - Optional CSV with ground-truth labels for ROC.

- **pretrained_ckpt**  
  - Path(s) to pretrained checkpoint(s). Used only when `tuning_mode=False`.

- **optuna**  
  - `n_trials`: Number of Optuna trials.  
  - `params`: Search space with types (loguniform, float, int, categorical).

- **xai**  
  - `enabled`: Whether to generate Grad-CAM outputs.  
  - `target_layer`: Layer(s) to explain (supports multi-layer fusion).  
  - Options for heatmaps, overlays, panels, top-K patches, scale-bar masking.

---

## Evaluation Metrics

Metrics are computed during validation and logged both in console and Excel:

- **Validation Loss**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **F2 Score (β=2, recall-focused)**
- **Composite Metric:** `alpha * Recall - beta * Loss`

Saved plots and reports include:

- **Confusion Matrix** (PNG)  
- **Classification Report** (Precision/Recall/F1/Support)  
- **ROC Curve with AUC**  

All metrics are saved to `<save_dir>/<date>/<time>/eval/`.

---

## Explainability (XAI)

If enabled (`cfg.xai.enabled=True`), the pipeline generates **Grad-CAM** visualizations:

- **Heatmaps** of attention regions.  
- **Overlay images** (heatmap on original).  
- **Panel images** with top-K evidence patches highlighted.  
- **Evidence crops** saved separately.  
- **CSV summary** of predictions and evidence scores.

Outputs are saved under `<save_dir>/<date>/<time>/eval/xai_*`.

---

## Tuning Mode (Optuna)

When `tuning_mode=True` and `use_optuna=True`:

- **Objective Function:**  
  Optimize composite metric = `alpha * Recall - beta * Loss`.

- **Trial Logging:**  
  Each trial’s hyperparameters and scores are appended to `optuna_trials.xlsx`.  
  The best trial’s parameters are saved in `optuna_best_params.xlsx`.

- **Early Stopping & Pruning:**  
  Poor trials are stopped early to save time.

- **Visualizations:**  
  Optimization history and parameter importance plots are saved in `eval/`.

---

## Final Outputs

After a run, you will obtain:

- **Best Model Checkpoint** (`best_detection.ckpt`) in `best_model/`.  
- **Evaluation Metrics & Plots** (confusion matrix, ROC, classification report).  
- **Optuna Logs** (`optuna_trials.xlsx`, `optuna_best_params.xlsx`) if tuning is used.  
- **XAI Outputs** (heatmaps, overlays, evidence crops, CSV summaries).  
- **Test Predictions** (Excel with predicted labels, probabilities, and images).  

---

## Expected Results

- **Training Mode**
  - Logs training/validation metrics each epoch.
  - Produces TensorBoard logs (`tb_logs`) for curves.
  - Outputs best model checkpoint and evaluation plots.

- **Tuning Mode**
  - Runs multiple Optuna trials.  
  - Stops poor trials early.  
  - Logs all results to Excel and picks best hyperparameters.

- **Test Mode**
  - Loads checkpoints.  
  - Runs inference on test folder(s).  
  - Produces predictions Excel, confusion matrix, ROC, and XAI visualizations.

---

## Installation

Install dependencies with:

```bash
pip install torch torchvision pytorch-lightning optuna albumentations omegaconf scikit-learn matplotlib seaborn tqdm openpyxl pillow
