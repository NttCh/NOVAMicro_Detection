#!/usr/bin/env python
"""
Main script to run training, testing, or tuning (Optuna) based on config:
  cfg.run_mode in {"train", "test", "tune"}

All outputs live under:
  <save_dir>/<YYYYMMDD>/<HHMMSS>_<run_mode>/{best_model,eval,multi_predictions}/
"""

import os
import sys
from pathlib import Path
import warnings
import torch
import optuna
import albumentations as A
from pytorch_lightning import seed_everything

# ---------------- GPU and Environment Settings ----------------
torch.set_float32_matmul_precision("high")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# --------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r".*reported value is ignored because this `step` is already reported.*",
    category=UserWarning,
    module="optuna.trial",
)

# -------- Project imports --------
import config
from config import cfg
from utils import (
    set_seed,
    thai_time,
    load_obj,
    generate_detection_csv,
    make_run_dirs,
    record_trial_history,
)
from model import build_classifier
from train import (
    train_stage,
    repeated_cross_validation,
    print_trial_thai_callback,
)
from inference import (
    evaluate_model,
    predict_folders_to_combined_workbook,
    save_multi_ckpt_comparison,
)
from xai import _maybe_run_xai
from viz import (
    export_optuna_plots,
    plot_cv_mean_std_curves,
    plot_per_fold_curves,
    plot_cv_f2_and_loss_combined,
    plot_cv_f2_and_loss_mean_only,
    plot_cv_mean_loss,
    plot_cv_mean_f2,
)
from optuna_tuner import (
    run_optuna_search,
    apply_best_params_to_cfg,
    _choose_reuse_trial,
    _apply_trial_params_to_cfg,
)

# ==============================================================
# Small helpers (you can move these to utils.py later as-is)
# ==============================================================

def _reuse_enabled(cfg) -> bool:
    reuse_cfg = getattr(cfg.optuna, "reuse", None)
    return bool(reuse_cfg and getattr(reuse_cfg, "enable", False))

def _resolve_pretrained_ckpt_for_training(cfg) -> str | None:
    """
    Return a checkpoint path only when we're in TRAIN mode AND NOT reusing Optuna.
    Otherwise None (train from scratch).
    """
    if str(getattr(cfg, "run_mode", "")).lower() != "train":
        return None
    if _reuse_enabled(cfg):
        return None
    ckpt = getattr(cfg, "pretrained_ckpt", None)
    if ckpt and isinstance(ckpt, str) and os.path.exists(ckpt):
        return ckpt
    return None

def _print_training_plan(cfg, ckpt_to_use: str | None):
    if _reuse_enabled(cfg):
        print("[Plan] TRAIN + REUSE=TRUE → ignore checkpoints; retrain using reused Optuna trial params.")
    else:
        if ckpt_to_use:
            print(f"[Plan] TRAIN + REUSE=FALSE → initialize from checkpoint: {ckpt_to_use}")
        else:
            print("[Plan] TRAIN + REUSE=FALSE → no checkpoint found/provided; training from scratch.")

def _build_valid_transform(cfg):
    """Albumentations valid/test transform from config."""
    return A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])

def _parse_test_folds(cfg):
    """
    Normalize and validate a list of test folders.
    Accepts cfg.test.folder_path (str/list) or cfg.test.folder_paths; supports globbing; de-dups.
    """
    import glob as _glob, ast
    from collections.abc import Sequence
    try:
        from omegaconf import ListConfig
        _seq_types = (list, tuple, set, ListConfig)
    except Exception:
        _seq_types = (list, tuple, set)

    def _norm(p: str) -> str:
        p = os.path.expandvars(os.path.expanduser(str(p).strip()))
        p = os.path.normpath(p)
        return os.path.abspath(p)

    raw = getattr(cfg.test, "folder_paths", None) or getattr(cfg.test, "folder_path", None)

    if isinstance(raw, _seq_types) and not isinstance(raw, (str, bytes)):
        candidates = [str(p) for p in list(raw) if p is not None]
    elif isinstance(raw, (str, bytes)):
        s = raw.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                    candidates = [str(p) for p in parsed if p is not None]
                else:
                    candidates = [s]
            except Exception:
                candidates = [s]
        else:
            candidates = [s]
    else:
        candidates = []

    valid = []
    for item in candidates:
        s = str(item).strip()
        expanded = _glob.glob(_norm(s)) or []
        for path in expanded:
            npath = _norm(path)
            if os.path.isdir(npath):
                valid.append(npath)

    if not valid:
        print("[Main] No valid test folders.")
        return []
    return list(dict.fromkeys(valid))

def _ckpt_output_tag(ckpt_path: str) -> str:
    """
    Build a short, readable tag from the checkpoint filename.
    Examples:
      best_detection.ckpt                    -> "best"
      best_from_best_trial.ckpt              -> "best"
      best_retrain_best_trial_fold2.ckpt     -> "fold2"
      best_retrain_best_trial_fold10.ckpt    -> "fold10"
      anything_else.ckpt                     -> basename w/o .ckpt (clamped)
    """
    stem = Path(ckpt_path).stem.lower()

    if stem in {"best_detection", "best_from_best_trial"} or stem.startswith("best_from_trial"):
        return "best"

    import re
    m = re.search(r"fold\s*([0-9]+)", stem) or re.search(r"_fold\s*([0-9]+)", stem)
    if m:
        return f"fold{int(m.group(1))}"

    import re as _re
    s = _re.sub(r"[^a-z0-9_-]+", "_", stem)
    return s[:24] if len(s) > 24 else s

def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load a Lightning-or-raw checkpoint into a plain nn.Module.
    Handles PyTorch 2.6 default (weights_only=True) by forcing full load.
    Strips 'model.' prefix when present.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    sd = { (k[6:] if k.startswith("model.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Load] {Path(ckpt_path).name}  missing={len(missing)} unexpected={len(unexpected)}")
    # Debug checksum (uncomment if needed)
    # with torch.no_grad():
    #     print("[Debug] first param mean:", float(next(model.parameters()).mean()))

def _run_cv_plots(all_eval_path: str, out_dir: str):
    """Centralized plotting (skips gracefully if Excel not present)."""
    if not os.path.exists(all_eval_path):
        print(f"[Viz] Skipped CV curves: not found → {all_eval_path}")
        return
    plot_cv_mean_std_curves(all_eval_path, out_dir)
    plot_per_fold_curves(all_eval_path, out_dir, metric_col="val_f2")
    plot_per_fold_curves(all_eval_path, out_dir, metric_col="val_loss")
    plot_cv_f2_and_loss_combined(all_eval_path, out_dir)
    plot_cv_f2_and_loss_mean_only(all_eval_path, out_dir)
    plot_cv_mean_loss(all_eval_path, out_dir)
    plot_cv_mean_f2(all_eval_path, out_dir)

def _run_multi_ckpt_tests(base_model, cfg, dirs, best_ckpt_path: str):
    """Predict across multiple ckpts × folders, then run XAI and optionally compare outputs."""
    valid_tf   = _build_valid_transform(cfg)
    raw_ckpts  = getattr(cfg.test, "ckpt_paths", None) or best_ckpt_path
    ckpt_paths = [raw_ckpts] if isinstance(raw_ckpts, str) else list(raw_ckpts)
    ckpt_paths = list(dict.fromkeys(ckpt_paths))
    test_folds = _parse_test_folds(cfg)

    if not test_folds:
        print("[Test] No valid test folders. Skipping test predictions/XAI.")
        return

    ALL_RESULTS_FOR_COMPARE = {}
    for ckpt in ckpt_paths:
        print(f"\n[Test] Loading checkpoint: {ckpt}")
        _load_ckpt_into_model(base_model, ckpt)
        base_model.eval()

        run_tag   = Path(ckpt).parent.parent.name          # e.g. 204752_train_RN50_model3
        ckpt_tag  = _ckpt_output_tag(ckpt)                 # e.g. "best" or "fold2"
        ckpt_name = f"{run_tag}__{ckpt_tag}"               # unique folder name

        out_dir   = os.path.join(dirs["multi_predictions"], ckpt_name)
        os.makedirs(out_dir, exist_ok=True)

        combined_xlsx = os.path.join(out_dir, f"predictions_{ckpt_name}_ALL_FOLDERS.xlsx")
        results_dict = predict_folders_to_combined_workbook(
            model=base_model,
            folders=test_folds,
            transform=valid_tf,
            combined_xlsx_path=combined_xlsx,
            ckpt_print_prefix=f"[{ckpt_name}] "
        )
        ALL_RESULTS_FOR_COMPARE[ckpt_name] = results_dict

        # --- XAI per folder, namespaced by ckpt_name (prevents overwrites) ---
        for folder in test_folds:
            _maybe_run_xai(base_model, valid_tf, folder, dirs, getattr(cfg, "xai", None) or {}, ckpt_tag=ckpt_name)

    if len(ALL_RESULTS_FOR_COMPARE) > 1:
        compare_xlsx = os.path.join(dirs["multi_predictions"], "predictions_COMPARE_ALL_CKPTS.xlsx")
        save_multi_ckpt_comparison(ALL_RESULTS_FOR_COMPARE, compare_xlsx)
        print(f"[Test] Wrote cross-ckpt comparison workbook → {compare_xlsx}")

# ==============================================================
# Mode runners
# ==============================================================

def run_test_only(cfg, dirs):
    print("[Main] TEST ONLY MODE")
    if not cfg.pretrained_ckpt:
        print("Please provide cfg.pretrained_ckpt for testing.")
        sys.exit(1)

    model = build_classifier(cfg, num_classes=2)

    raw_ckpts  = getattr(cfg.test, "ckpt_paths", None) or cfg.pretrained_ckpt
    ckpt_paths = [raw_ckpts] if isinstance(raw_ckpts, str) else list(raw_ckpts)
    ckpt_paths = list(dict.fromkeys(ckpt_paths))

    test_folds = _parse_test_folds(cfg)
    if test_folds:
        print(f"[Main] Test folders: {test_folds}")

    valid_tf = _build_valid_transform(cfg)
    ALL_RESULTS_FOR_COMPARE = {}

    if test_folds:
        for ckpt in ckpt_paths:
            print(f"\n=== Loading checkpoint: {ckpt} ===")
            _load_ckpt_into_model(model, ckpt)
            model.eval()

            run_tag   = Path(ckpt).parent.parent.name
            ckpt_tag  = _ckpt_output_tag(ckpt)
            ckpt_name = f"{run_tag}__{ckpt_tag}"
            out_dir   = os.path.join(dirs["multi_predictions"], ckpt_name)
            os.makedirs(out_dir, exist_ok=True)

            combined_xlsx = os.path.join(out_dir, f"predictions_{ckpt_name}_ALL_FOLDERS.xlsx")
            results_dict = predict_folders_to_combined_workbook(
                model=model,
                folders=test_folds,
                transform=valid_tf,
                combined_xlsx_path=combined_xlsx,
                ckpt_print_prefix=f"[{ckpt_name}] "
            )
            ALL_RESULTS_FOR_COMPARE[ckpt_name] = results_dict

            # --- XAI per folder, namespaced by ckpt_name ---
            for folder in test_folds:
                _maybe_run_xai(model, valid_tf, folder, dirs, getattr(cfg, "xai", None) or {}, ckpt_tag=ckpt_name)

    if len(ALL_RESULTS_FOR_COMPARE) > 1:
        compare_xlsx = os.path.join(dirs["multi_predictions"], "predictions_COMPARE_ALL_CKPTS.xlsx")
        save_multi_ckpt_comparison(ALL_RESULTS_FOR_COMPARE, compare_xlsx)
        print(f"[Test] Wrote cross-ckpt comparison workbook → {compare_xlsx}")

    print("[Main] TEST ONLY complete.")

def run_tune(cfg, dirs):
    print("[Main] TUNING MODE (Optuna)")
    eval_folder = dirs["eval"]

    db_path = os.path.join(cfg.general.save_dir, "optuna.db")
    storage = f"sqlite:///{db_path}"
    study_name = getattr(getattr(cfg, "optuna", {}), "study_name", None)

    best_params, study = run_optuna_search(
        cfg=cfg,
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        show_progress_bar=True,
        callbacks=[print_trial_thai_callback, record_trial_history(eval_folder)],
    )

    # Export artifacts
    try:
        study.trials_dataframe().to_excel(os.path.join(eval_folder, "optuna_trials.xlsx"), index=False)
    except Exception as e:
        print("[Optuna] trials_dataframe export failed:", e)
    export_optuna_plots(study, eval_folder)

    # Map best params back into cfg and optionally finalize
    mapping = {
        "lr": "optimizer.params.lr",
        "weight_decay": "optimizer.params.weight_decay",
        "batch_size": "data.batch_size",
        "gradient_clip_val": "optimizer.params.gradient_clip_val",
    }
    apply_best_params_to_cfg(cfg, best_params, target_nodes=mapping)

    if getattr(cfg.training, "finalize_after_hpo", False):
        print("[Finalize] Retraining with best hyperparameters on CV...")
        cfg.use_cv = True
        detection_model, detection_metric = repeated_cross_validation(
            cfg, cfg.data.detection_csv, num_classes=2, stage_name="final_best", repeats=1
        )
        best_ckpt = os.path.join(dirs["best_model"], "best_detection.ckpt")
        torch.save(detection_model.state_dict(), best_ckpt)
        print(f"[Main] Saved checkpoint → {best_ckpt}")
    else:
        print("[Main] Tuning finished (no finalize).")

def run_train(cfg, dirs):
    print("[Main] TRAINING MODE")
    detection_csv = cfg.data.detection_csv

    ckpt_to_use = _resolve_pretrained_ckpt_for_training(cfg)
    _print_training_plan(cfg, ckpt_to_use)

    # ----- REUSE path -----
    if _reuse_enabled(cfg):
        reuse_cfg = getattr(cfg.optuna, "reuse", None)

        # --- temporarily disable any pretrained ckpt so reuse truly ignores it ---
        _pretrained_backup = getattr(cfg, "pretrained_ckpt", None)
        cfg.pretrained_ckpt = None
        # ------------------------------------------------------------------------

        try:
            db_path  = os.path.join(cfg.general.save_dir, "optuna.db")
            storage  = f"sqlite:///{db_path}"
            study_name = str(getattr(reuse_cfg, "study_name", "") or "").strip()
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
            except KeyError:
                names = optuna.get_all_study_names(storage=storage)
                print(f"[Reuse] Study '{study_name}' not found. Available: {names}")
                candidates = [n for n in names if "bacteria" in n or "optuna" in n]
                if not candidates:
                    raise
                study = optuna.load_study(study_name=candidates[-1], storage=storage)
                print(f"[Reuse] Falling back to '{study.study_name}'")

            trial_id = getattr(reuse_cfg, "retrain_trial_id", None)
            try:
                trial_id = int(trial_id) if trial_id is not None else None
            except Exception:
                trial_id = None

            t = _choose_reuse_trial(study, trial_id)
            print(f"[Reuse] Using Trial #{t.number} | state: {t.state} | value: {t.value} | params: {t.params}")

            _apply_trial_params_to_cfg(cfg, t.params)

            cfg.use_cv = bool(getattr(reuse_cfg, "use_cv", True))
            repeats    = int(getattr(reuse_cfg, "repeats", getattr(cfg.training, "repeated_cv", 1)))
            stage_tag  = f"retrain_trial{t.number}" if (trial_id is not None and trial_id >= 0) else "retrain_best_trial"

            if cfg.use_cv:
                detection_model, detection_metric = repeated_cross_validation(
                    cfg, detection_csv, num_classes=2, stage_name=stage_tag, repeats=repeats
                    # If your function supports it, you can also force: init_ckpt=None
                )
            else:
                detection_model, detection_metric = train_stage(
                    cfg, detection_csv, num_classes=2, stage_name=stage_tag
                    # If your function supports it, you can also force: init_ckpt=None
                )

            # Save ckpt from REUSE run
            best_ckpt_name = (
                f"best_from_trial{t.number}.ckpt" if (trial_id is not None and trial_id >= 0)
                else "best_from_best_trial.ckpt"
            )
            best_ckpt = os.path.join(dirs["best_model"], best_ckpt_name)
            torch.save(detection_model.state_dict(), best_ckpt)
            print(f"[Reuse] Saved checkpoint → {best_ckpt}")

            # Evaluation + Plots
            evaluate_model(detection_model, detection_csv, cfg, stage=stage_tag)
            _run_cv_plots(os.path.join(dirs["eval"], "all_eval_metrics.xlsx"), dirs["eval"])

            # Tests/XAI
            _run_multi_ckpt_tests(detection_model, cfg, dirs, best_ckpt)

            print("[Reuse] DONE.")
            return

        finally:
            # --- always restore the original pretrained ckpt after reuse path finishes ---
            cfg.pretrained_ckpt = _pretrained_backup

    # ----- NORMAL train path -----
    if bool(getattr(cfg, "use_cv", True)):
        detection_model, detection_metric = repeated_cross_validation(
            cfg, detection_csv, num_classes=2, stage_name="detection",
            repeats=getattr(cfg.training, "repeated_cv", 1),
            init_ckpt=ckpt_to_use  # optional: your function should load this if not None
        )
    else:
        detection_model, detection_metric = train_stage(
            cfg, detection_csv, num_classes=2, stage_name="detection",
            init_ckpt=ckpt_to_use  # optional: your function should load this if not None
        )

    best_ckpt = os.path.join(dirs["best_model"], "best_detection.ckpt")
    torch.save(detection_model.state_dict(), best_ckpt)
    print(f"[Main] Saved checkpoint → {best_ckpt}")

    # Optional final evaluation
    if getattr(cfg.training, "finalize_after_hpo", False):
        evaluate_model(detection_model, detection_csv, cfg, stage="detection")
    else:
        print("[Main] Skipping final evaluation (finalize_after_hpo=False).")

    # Plots + Tests/XAI
    _run_cv_plots(os.path.join(dirs["eval"], "all_eval_metrics.xlsx"), dirs["eval"])
    _run_multi_ckpt_tests(detection_model, cfg, dirs, best_ckpt)

# ==============================================================
# main()
# ==============================================================

def main():
    # 1) Seed
    set_seed(cfg.training.seed)
    seed_everything(cfg.training.seed, workers=True)

    # 2) Normalize run mode
    run_mode = str(getattr(cfg, "run_mode", "train")).lower()
    if run_mode not in {"train", "test", "tune"}:
        print(f"[Main] Unknown run_mode='{run_mode}', defaulting to 'train'")
        run_mode = "train"
    cfg.run_mode = run_mode

    # 3) Build run root: <save_dir>/<YYYYMMDD>/<HHMMSS>_<run_mode>/
    now       = thai_time()
    date_str  = now.strftime("%Y%m%d")
    time_mode = f"{now.strftime('%H%M%S')}_{run_mode}"
    config.BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, date_str, time_mode)

    # 4) Create standard subdirs
    dirs = make_run_dirs(config.BASE_SAVE_DIR)  # {"best_model","eval","multi_predictions"}

    # 5) Generate detection CSV if needed
    if getattr(cfg.data, "generate_csv", False) or not os.path.exists(cfg.data.detection_csv):
        generate_detection_csv(cfg.data.negative_dir, cfg.data.positive_dir, cfg.data.detection_csv)

    # 6) Dispatch
    if run_mode == "test":
        run_test_only(cfg, dirs)
    elif run_mode == "tune":
        run_tune(cfg, dirs)
    else:  # "train"
        run_train(cfg, dirs)

    print("[Main] ALL DONE.")

if __name__ == "__main__":
    main()
