# main.py
#!/usr/bin/env python
"""
Run training, testing, or tuning (Optuna) based on the configuration.

`cfg.run_mode` ∈ {"train", "test", "tune"}.
Outputs are stored under:
<save_dir>/<YYYYMMDD>/<HHMMSS>_<run_mode>/{best_model, eval, multi_predictions}/
"""

from pathlib import Path
import os
import sys
import warnings
import optuna
import torch
from pytorch_lightning import seed_everything

# ---------------- GPU and Environment Settings ----------------
torch.set_float32_matmul_precision("high")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings(
    "ignore",
    message=(r".*reported value is ignored because this `step` is already reported.*"),
    category=UserWarning,
    module="optuna.trial",
)

# -------- Project imports (match src/ layout) --------
import config as config
from config import cfg

# data / transforms
from src.utils.data import _build_valid_transform

# inference & evaluation
from src.utils.inference import (
    evaluate_model,
    predict_folders_to_combined_workbook,
    save_multi_ckpt_comparison,
    _parse_test_folds,
    _run_multi_ckpt_tests,
)

# model
from src.model.model import build_classifier, _build_model_for_weights

# optuna
from src.utils.optuna_tuner import (
    run_optuna_search,
    apply_best_params_to_cfg,
    _choose_reuse_trial,
    _apply_trial_params_to_cfg,
    record_trial_history,
)

# training
from src.utils.train import (
    train_stage,
    repeated_cross_validation,
    print_trial_thai_callback,
)

# misc utilities
from src.utils.utils import (
    set_seed,
    thai_time,
    generate_detection_csv,
    make_run_dirs,
    _resolve_pretrained_ckpt_for_training,
    _print_training_plan,
    _reuse_enabled,
    load_weights_into_model,
    assert_train_data_available
)

# visualization & XAI
from src.utils.viz import export_optuna_plots, _run_cv_plots
from src.utils.xai import _maybe_run_xai

# ---------------- Mode Runners ----------------
def run_test_only(cfg, dirs):
    print("[Main] TEST ONLY MODE")
    if not cfg.pretrained_ckpt:
        print("Please provide cfg.pretrained_ckpt for testing.")
        sys.exit(1)

    raw_ckpts = getattr(cfg.test, "ckpt_paths", None) or cfg.pretrained_ckpt
    ckpt_paths = [raw_ckpts] if isinstance(raw_ckpts, str) else list(raw_ckpts)
    ckpt_paths = list(dict.fromkeys(ckpt_paths))

    test_folds = _parse_test_folds(cfg)
    if test_folds:
        print(f"[Main] Test folders: {test_folds}")

    valid_tf = _build_valid_transform(cfg)
    ALL_RESULTS_FOR_COMPARE = {}

    if test_folds:
        for ckpt in ckpt_paths:
            use_path = ckpt  
            print(f"\n=== Loading weights: {use_path} ===")

            # Rebuild model from snapshot if present, else from current cfg
            model = _build_model_for_weights(cfg, use_path, num_classes=2)

            # Robust load (handles Lightning dicts and prefix stripping)
            miss, unexp = load_weights_into_model(model, use_path)
            if miss or unexp:
                print(f"[Warn] loaded with gaps: missing={miss} unexpected={unexp}")
            model.eval()

            run_tag = Path(ckpt).parent.parent.name
            ckpt_tag = Path(use_path).with_suffix("").name
            ckpt_name = f"{run_tag}__{ckpt_tag}"
            out_dir = os.path.join(dirs["multi_predictions"], ckpt_name)
            os.makedirs(out_dir, exist_ok=True)

            combined_xlsx = os.path.join(out_dir, f"predictions_{ckpt_name}_ALL_FOLDERS.xlsx")
            results_dict = predict_folders_to_combined_workbook(
                model=model, folders=test_folds, transform=valid_tf,
                combined_xlsx_path=combined_xlsx, ckpt_print_prefix=f"[{ckpt_name}] "
            )
            ALL_RESULTS_FOR_COMPARE[ckpt_name] = results_dict

            # XAI per folder, namespaced by ckpt_name
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

    try:
        study.trials_dataframe().to_excel(os.path.join(eval_folder, "optuna_trials.xlsx"), index=False)
    except Exception as e:
        print("[Optuna] trials_dataframe export failed:", e)
    export_optuna_plots(study, eval_folder)

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
        detection_model, _ = repeated_cross_validation(cfg, cfg.data.detection_csv, num_classes=2, stage_name="final_best", repeats=1)

        # Save Lightning-style ckpt only
        best_ckpt = os.path.join(dirs["best_model"], "best_detection.ckpt")
        torch.save({"state_dict": detection_model.state_dict()}, best_ckpt)
        print(f"[Main] Saved ckpt → {best_ckpt}")
    else:
        print("[Main] Tuning finished (no finalize).")


def run_train(cfg, dirs):
    print("[Main] TRAINING MODE")
    detection_csv = cfg.data.detection_csv

    ckpt_to_use = _resolve_pretrained_ckpt_for_training(cfg)
    _print_training_plan(cfg, ckpt_to_use)

    # ---------------- REUSE path ----------------
    if _reuse_enabled(cfg):
        reuse_cfg = getattr(cfg, "reuse", None)

        # Temporarily disable any pretrained ckpt so reuse truly ignores it
        _pretrained_backup = getattr(cfg, "pretrained_ckpt", None)
        cfg.pretrained_ckpt = None

        try:
            db_path = os.path.join(cfg.general.save_dir, "optuna.db")
            storage = f"sqlite:///{db_path}"
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

            # Use global CV settings (top-level use_cv + training.repeated_cv)
            cfg.use_cv = bool(getattr(cfg, "use_cv", True))
            repeats = int(getattr(cfg.training, "repeated_cv", 1))
            stage_tag = f"retrain_trial{t.number}" if (trial_id is not None and trial_id >= 0) else "retrain_best_trial"

            if cfg.use_cv:
                detection_model, _ = repeated_cross_validation(
                    cfg=cfg, csv_path=detection_csv, num_classes=2, stage_name=stage_tag, repeats=repeats
                )
            else:
                detection_model, _ = train_stage(
                    cfg=cfg, csv_path=detection_csv, num_classes=2, stage_name=stage_tag
                )

            # Save ckpt ONLY
            best_ckpt_name = (
                f"best_from_trial{t.number}.ckpt" if (trial_id is not None and trial_id >= 0)
                else "best_from_best_trial.ckpt"
            )
            best_ckpt = os.path.join(dirs["best_model"], best_ckpt_name)
            torch.save({"state_dict": detection_model.state_dict()}, best_ckpt)
            print(f"[Reuse] Saved ckpt → {best_ckpt}")

            # Evaluate + Plots + Multi-ckpt tests/XAI
            evaluate_model(detection_model, detection_csv, cfg, stage=stage_tag)
            _run_cv_plots(os.path.join(dirs["eval"], "all_eval_metrics.xlsx"), dirs["eval"])
            _run_multi_ckpt_tests(detection_model, cfg, dirs, best_ckpt)

            print("[Reuse] DONE.")
            return
        finally:
            # Restore original pretrained ckpt after reuse path finishes
            cfg.pretrained_ckpt = _pretrained_backup

    # ---------------- Normal training path ----------------
    if bool(getattr(cfg, "use_cv", True)):
        detection_model, _ = repeated_cross_validation(
            cfg=cfg, csv_path=detection_csv, num_classes=2, stage_name="detection",
            repeats=getattr(cfg.training, "repeated_cv", 1)
        )
    else:
        detection_model, _ = train_stage(
            cfg=cfg, csv_path=detection_csv, num_classes=2, stage_name="detection"
        )

    # Save ckpt ONLY after normal training
    best_ckpt = os.path.join(dirs["best_model"], "best_detection.ckpt")
    torch.save({"state_dict": detection_model.state_dict()}, best_ckpt)
    print(f"[Main] Saved ckpt → {best_ckpt}")

    # Evaluate + Plots
    evaluate_model(detection_model, cfg.data.detection_csv, cfg, stage="detection_cv")
    _run_cv_plots(os.path.join(dirs["eval"], "all_eval_metrics.xlsx"), dirs["eval"])

    # Multi-ckpt tests/XAI — pass the ckpt we just saved
    _run_multi_ckpt_tests(detection_model, cfg, dirs, best_ckpt)


# ---------------- main() ----------------
def main():
    set_seed(cfg.training.seed)
    seed_everything(cfg.training.seed, workers=True)

    run_mode = str(getattr(cfg, "run_mode", "train")).lower()
    if run_mode not in {"train", "test", "tune"}:
        print(f"[Main] Unknown run_mode='{run_mode}', defaulting to 'train'")
        run_mode = "train"
    cfg.run_mode = run_mode

    now = thai_time()
    date_str = now.strftime("%Y%m%d")
    time_mode = f"{now.strftime('%H%M%S')}_{run_mode}"
    config.BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, date_str, time_mode)

    dirs = make_run_dirs(config.BASE_SAVE_DIR)

    if run_mode in {"train", "tune"}:
        # 1) Make sure folders exist and have images
        assert_train_data_available(cfg)

        # 2) CSV present or generate
        csv_path = cfg.data.detection_csv
        want_autogen = bool(getattr(cfg.data, "generate_csv", False))

        if want_autogen:
            # Ensure the parent folder for the CSV exists before writing
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            generate_detection_csv(
                cfg.data.negative_dir,
                cfg.data.positive_dir,
                csv_path,
            )
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"[Main] detection_csv not found: {csv_path}\n"
                    "Set cfg.data.generate_csv=True to build it automatically, "
                    "or provide a valid CSV path."
                )

    if run_mode == "test":
        run_test_only(cfg, dirs)
    elif run_mode == "tune":
        run_tune(cfg, dirs)
    else:
        run_train(cfg, dirs)

    print("[Main] ALL DONE.")


if __name__ == "__main__":
    main()
